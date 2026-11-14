"""端到端链路走查：把一个需求从「自然语言」推到「带预算的行程」，
并在每个步骤打印输入 / 输出，便于学习和排查。

两个用例：
1. 信息完整 → 需求提取 → 景点 → 矩阵 → 天气 → 餐厅 → 排程 → 时间轴 → 预算
2. 信息缺失 → 需求提取后直接返回追问，不进入后续步骤

为了让输出稳定可复现，排程固定用 RuleBasedItineraryPlanner，
需求抽取固定用 RuleBasedRequirementExtractor（不消耗 LLM 调用）。

运行（在 travel_agent/ 目录下，-s 才能看到 print）：
    .venv\\Scripts\\python.exe -m pytest backend/tests/test_pipeline_walkthrough.py -s -v
"""

import asyncio

from backend.agents.planner_agent import DAILY_LIMITS, RuleBasedItineraryPlanner
from backend.agents.requirement_agent import RuleBasedRequirementExtractor
from backend.providers.mock_provider import (
    MockAttractionProvider,
    MockRestaurantProvider,
    MockRouteProvider,
    MockWeatherProvider,
)
from backend.schemas.common import TransportMode
from backend.schemas.itinerary import ItineraryItemType
from backend.services.attraction_service import AttractionService
from backend.services.budget_service import BudgetService
from backend.services.itinerary_builder import build_itinerary
from backend.services.itinerary_service import ItineraryService
from backend.services.requirement_service import RequirementService
from backend.services.restaurant_service import RestaurantService
from backend.services.route_service import RouteService
from backend.services.weather_service import WeatherService


COMPLETE_INPUT = (
    "帮我规划一次日本东京 5 日游，12月10日出发，预算 1000 美元，"
    "喜欢美食、购物、文化景点和自然风光，希望行程不要太紧凑，1 人"
)
INCOMPLETE_INPUT = "我想去东京玩几天，帮我看看"


def _section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def _service() -> RequirementService:
    return RequirementService(extractor=AgentRequirementExtractor())



async def _run_pipeline(requirements) -> None:
    """按 ItineraryService.plan 的顺序逐步执行，并打印每一步的输出。"""
    daily_limit = DAILY_LIMITS[requirements.travel_style]
    candidate_limit = min(
        daily_limit * requirements.duration_days * ItineraryService.CANDIDATE_MULTIPLIER,
        RouteService.MAX_ATTRACTIONS,
    )

    _section("步骤 1：计算候选数")
    print(f"节奏={requirements.travel_style.value}，每日上限={daily_limit}，"
          f"天数={requirements.duration_days}，候选上限={candidate_limit}")

    # 步骤 2：景点
    attraction_service = AttractionService(provider=MockAttractionProvider())
    attraction_response = await attraction_service.search(
        city=requirements.destination,
        interests=requirements.interests or None,
        limit=candidate_limit,
    )
    _section("步骤 2：景点检索")
    print(f"候选数：{attraction_response.total}")
    for item in attraction_response.items:
        print(f"  - {item.id} | {item.name} | {item.category.value} | "
              f"评分={item.rating} | {item.estimated_duration_minutes}min")

    # 步骤 3：交通矩阵
    matrix = await RouteService(provider=MockRouteProvider()).build_matrix(
        city=requirements.destination,
        attractions=attraction_response.items,
        mode=TransportMode.WALK,
    )
    _section("步骤 3：交通矩阵")
    print(f"模式={matrix.mode.value}，节点={len(matrix.attraction_ids)}，边={len(matrix.entries)}")
    for entry in sorted(matrix.entries, key=lambda e: e.duration_minutes)[:3]:
        print(f"  - {entry.from_id} -> {entry.to_id}: {entry.duration_minutes}min")

    # 步骤 4：天气
    weather = await WeatherService(provider=MockWeatherProvider()).get_forecast(
        city=requirements.destination,
        start_date=requirements.start_date,
        days=requirements.duration_days,
    )
    _section("步骤 4：天气")
    print(f"天数：{len(weather.days)}")
    for day in weather.days:
        print(f"  - {day.date}: {day.condition.value}，"
              f"{day.temp_min_c:.0f}~{day.temp_max_c:.0f}℃，"
              f"降水概率 {day.precipitation_probability}%")

    # 步骤 5：餐厅
    restaurant_response = await RestaurantService(provider=MockRestaurantProvider()).search(
        city=requirements.destination,
        limit=requirements.duration_days,
    )
    _section("步骤 5：餐厅")
    print(f"候选数：{restaurant_response.total}")
    for restaurant in restaurant_response.items:
        print(f"  - {restaurant.name} | {'/'.join(restaurant.cuisines)} | "
              f"评分={restaurant.rating} | 人均={restaurant.average_price.amount} "
              f"{restaurant.average_price.currency}")

    # 步骤 6：排程
    outcome = await RuleBasedItineraryPlanner().plan(
        requirements=requirements,
        candidates=attraction_response.items,
        matrix=matrix,
        weather=weather,
    )
    _section("步骤 6：排程草稿")
    print(f"来源：{outcome.source}，天数：{len(outcome.draft.days)}")
    for day in outcome.draft.days:
        print(f"  - Day {day.day_index}（{day.theme}）：{day.attraction_ids}")

    # 步骤 7：时间轴组装（含午餐）
    itinerary = build_itinerary(
        requirements,
        outcome.draft,
        attraction_response.items,
        matrix,
        restaurants=restaurant_response.items,
    )
    _section("步骤 7：时间轴")
    for day in itinerary.days:
        print(f"  Day {day.day_index} {day.date}")
        for item in day.items:
            cost = (
                f"{item.cost.amount} {item.cost.currency}" if item.cost else "无价格"
            )
            print(f"    {item.seq}. [{item.type.value}] {item.name} "
                  f"{item.start_time:%H:%M}-{item.end_time:%H:%M} cost={cost}")

    # 步骤 8：预算
    budget_outcome = BudgetService().estimate(requirements, itinerary)
    _section("步骤 8：预算")
    for line in budget_outcome.breakdown.lines:
        print(f"  {line.category.value}: {line.amount.amount} {line.amount.currency}")
        print(f"    依据：{line.note}")
    print(f"  合计：{budget_outcome.breakdown.total.amount} "
          f"{budget_outcome.breakdown.currency}")
    print(f"  是否超支：{budget_outcome.breakdown.over_budget}")

    _section("Warnings")
    for warning in [*itinerary.warnings, *budget_outcome.warnings]:
        print(f"  - {warning}")

    # 收尾断言：链路确实产出了可用的行程
    assert len(itinerary.days) == requirements.duration_days
    assert all(
        any(item.type is ItineraryItemType.RESTAURANT for item in day.items)
        for day in itinerary.days
    )
    assert budget_outcome.breakdown.total.amount > 0


def test_complete_input_runs_full_pipeline() -> None:
    """信息完整：应一路走到预算，且每天都有午餐。"""
    print(f"\n用户输入：{COMPLETE_INPUT}")

    _section("步骤 0：需求提取")
    extraction = asyncio.run(_service().extract(COMPLETE_INPUT))
    print(f"extracted_by={extraction.extracted_by}，is_ready={extraction.is_ready}")
    print(f"draft={extraction.draft.model_dump()}")
    print(f"缺失字段={extraction.draft.missing_fields}")
    print(f"追问={extraction.clarifying_questions}")

    assert extraction.is_ready is True
    assert extraction.clarifying_questions == []

    requirements = extraction.to_requirements()
    asyncio.run(_run_pipeline(requirements))


def test_incomplete_input_returns_questions() -> None:
    """信息缺失：只返回追问，不进入取数与排程。"""
    print(f"\n用户输入：{INCOMPLETE_INPUT}")

    _section("步骤 0：需求提取")
    extraction = asyncio.run(_service().extract(INCOMPLETE_INPUT))
    print(f"extracted_by={extraction.extracted_by}，is_ready={extraction.is_ready}")
    print(f"draft={extraction.draft.model_dump()}")
    print(f"缺失字段={extraction.draft.missing_fields}")
    print(f"追问={extraction.clarifying_questions}")

    assert extraction.is_ready is False
    assert extraction.draft.destination == "Tokyo"
    assert extraction.draft.duration_days is None
    assert extraction.clarifying_questions

    print("\n结论：需求不完整，流程在此终止，不调用景点/路线/天气/排程。")
