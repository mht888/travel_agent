"""第 7 阶段：Itinerary Planner 测试。

分两组：
- test_framework_*：验证骨架结构，现在就能过（不需要填 TODO）
- test_flow_*：验证完整规划链路，填完 TODO 后才会通过，作为验收标准

运行（在 travel_agent/ 目录下）：
    uv run pytest backend/tests/test_itinerary_planner.py -v
"""

import asyncio
from collections.abc import Sequence
from datetime import date

import pytest

from backend.agents.planner_agent import (
    DAILY_LIMITS,
    PlannerOutcome,
    RuleBasedItineraryPlanner,
)
from backend.providers.mock_provider import (
    MockAttractionProvider,
    MockRestaurantProvider,
    MockRouteProvider,
    MockWeatherProvider,
)
from backend.schemas.attraction import Attraction
from backend.schemas.common import TransportMode
from backend.schemas.itinerary import ItineraryItemType
from backend.schemas.planner import DayPlan, ItineraryDraft
from backend.schemas.route import RouteMatrix
from backend.schemas.trip import TripRequirements
from backend.services.attraction_service import AttractionService
from backend.services.budget_service import BudgetService
from backend.services.itinerary_service import ItineraryService
from backend.services.restaurant_service import RestaurantService
from backend.services.route_service import RouteService
from backend.services.weather_service import WeatherService


@pytest.fixture()
def requirements() -> TripRequirements:
    return TripRequirements(
        destination="Tokyo",
        duration_days=2,
        start_date=date(2026, 4, 1),
        travelers=1,
        interests=["food", "culture"],
        travel_style="relaxed",
    )


class _FixedPlanner:
    """固定排程器：每天取候选里的连续两段，避免测试依赖 LLM。"""

    async def plan(
        self,
        *,
        requirements: TripRequirements,
        candidates: Sequence[Attraction],
        matrix: RouteMatrix,
        weather=None,
        today=None,
    ) -> PlannerOutcome:
        ids = [item.id for item in candidates]
        return PlannerOutcome(
            draft=ItineraryDraft(
                days=[
                    DayPlan(day_index=1, attraction_ids=ids[:2], theme="老城文化"),
                    DayPlan(day_index=2, attraction_ids=ids[2:4], theme="美食购物"),
                ]
            ),
            source="fake",
        )


# ---------------------------- 框架结构测试（现在就能过）----------------------------


def test_framework_imports_work() -> None:
    """关键模块可以 import，不依赖 TODO 是否填充。"""
    from backend.services.itinerary_builder import build_day, build_itinerary

    assert callable(build_day)
    assert callable(build_itinerary)
    assert callable(ItineraryService)


def test_framework_daily_limits_cover_all_styles() -> None:
    """每种节奏都必须有每日上限，否则 planner 会 KeyError。"""
    assert DAILY_LIMITS["relaxed"] < DAILY_LIMITS["balanced"] < DAILY_LIMITS["intensive"]


def test_framework_draft_rejects_duplicate_within_day() -> None:
    """同一天内重复景点应在 schema 层直接拒绝（这条已实现）。"""
    with pytest.raises(ValueError, match="不允许重复"):
        DayPlan(day_index=1, attraction_ids=["a", "a"])


def test_framework_rule_planner_splits_by_daily_limit(requirements: TripRequirements) -> None:
    """规则排程：8 个候选、relaxed 每天 2 个，应排出 2 天共 4 个。"""
    candidates = [
        Attraction(
            id=f"sight-{index}",
            name=f"Sight {index}",
            city="Tokyo",
            category="culture",
            location={"lat": 35.0 + index / 100, "lng": 139.0 + index / 100},
            estimated_duration_minutes=60,
            provenance={
                "source": "test",
                "fetched_at": "2026-01-01T00:00:00Z",
                "status": "mock",
            },
        )
        for index in range(8)
    ]
    matrix = RouteMatrix(
        city="Tokyo",
        mode=TransportMode.WALK,
        attraction_ids=[item.id for item in candidates],
        entries=[],
    )

    outcome = asyncio.run(
        RuleBasedItineraryPlanner().plan(
            requirements=requirements,
            candidates=candidates,
            matrix=matrix,
        )
    )

    assert outcome.source == "rule"
    assert len(outcome.draft.days) == 2
    assert [day.day_index for day in outcome.draft.days] == [1, 2]
    assert all(len(day.attraction_ids) == DAILY_LIMITS["relaxed"] for day in outcome.draft.days)


# ---------------------------- 完整流程测试（填完 TODO 后才会过）----------------------------


def _build_service(planner) -> ItineraryService:
    return ItineraryService(
        attraction_service=AttractionService(provider=MockAttractionProvider()),
        route_service=RouteService(provider=MockRouteProvider()),
        weather_service=WeatherService(provider=MockWeatherProvider()),
        planner=planner,
        budget_service=BudgetService(),
        restaurant_service=RestaurantService(provider=MockRestaurantProvider()),
    )


def test_flow_plan_produces_days_with_transport(requirements: TripRequirements) -> None:
    """端到端：2 天行程，每天 2 个景点，第二项开始必须有交通耗时。"""
    itinerary = asyncio.run(_build_service(_FixedPlanner()).plan(requirements))

    assert len(itinerary.days) == 2
    for day in itinerary.days:
        assert [item.seq for item in day.items] == list(range(1, len(day.items) + 1))
        assert day.items[0].transport_from_previous is None
        for previous, current in zip(day.items, day.items[1:]):
            assert current.transport_from_previous is not None
            assert current.start_time >= previous.end_time
            assert current.end_time > current.start_time


def test_flow_plan_assigns_dates_and_theme(requirements: TripRequirements) -> None:
    """每天应落到具体日期，并带上 LLM 给的主题。"""
    itinerary = asyncio.run(_build_service(_FixedPlanner()).plan(requirements))

    assert itinerary.days[0].date == date(2026, 4, 1)
    assert itinerary.days[1].date == date(2026, 4, 2)
    assert itinerary.days[0].theme == "老城文化"


def test_flow_plan_skips_unknown_attraction_ids(requirements: TripRequirements) -> None:
    """LLM 编造的 id 不能被静默塞进行程，必须跳过并留 warning。"""

    class _HallucinatingPlanner(_FixedPlanner):
        async def plan(self, *, requirements, candidates, matrix, weather=None, today=None):
            ids = [item.id for item in candidates]
            return PlannerOutcome(
                draft=ItineraryDraft(
                    days=[
                        DayPlan(day_index=1, attraction_ids=[ids[0], "made-up-id"]),
                        DayPlan(day_index=2, attraction_ids=ids[2:4]),
                    ]
                ),
                source="fake",
            )

    itinerary = asyncio.run(_build_service(_HallucinatingPlanner()).plan(requirements))

    # 每天会固定插入一顿午餐，所以这里只数景点项
    attractions = [
        item
        for item in itinerary.days[0].items
        if item.type is ItineraryItemType.ATTRACTION
    ]
    assert len(attractions) == 1
    assert any("候选集外" in warning for warning in itinerary.warnings)
