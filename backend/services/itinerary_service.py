"""行程规划编排（第 7 阶段）。

这一层只做编排，不做业务计算：

    AttractionService   取候选景点（已按兴趣 + 评分排序）
    RestaurantService   取餐厅，供每天一顿午餐（确定性，不走 LLM）
    RouteService        取交通时间矩阵
    WeatherService      取天气（用户给了出发日期才有）
    ItineraryPlanner    决定分天与顺序（LLM，失败降级规则）
    itinerary_builder   把 id 序列展开成带时间轴的 Itinerary
    BudgetService       把行程翻译成六类费用（第 8 阶段）

第 9~11 阶段的约束检查、路线优化都会挂在这一层之后，不修改它。
"""

import logging
from functools import lru_cache

from backend.agents.planner_agent import (
    DAILY_LIMITS,
    ItineraryPlanner,
    LlmItineraryPlanner,
    RuleBasedItineraryPlanner,
    create_planner_runnable,
)
from backend.config import get_settings
from backend.providers.mock_provider import (
    MockAttractionProvider,
    MockRestaurantProvider,
    MockRouteProvider,
    MockWeatherProvider,
)
from backend.schemas.common import TransportMode
from backend.schemas.itinerary import Itinerary
from backend.schemas.trip import TripRequirements
from backend.services.attraction_service import AttractionService
from backend.services.budget_service import BudgetService
from backend.services.itinerary_builder import build_itinerary
from backend.services.restaurant_service import RestaurantService
from backend.services.route_service import RouteService
from backend.services.weather_service import WeatherService

logger = logging.getLogger(__name__)


class ItineraryService:
    """一次完整规划：取数 → 排程 → 组装。"""

    # 第一版统一步行，后续按用户偏好或距离阈值切换
    DEFAULT_MODE = TransportMode.WALK
    # 每天要 N 个景点，至少准备 2N 个候选，否则「挑」没有意义
    CANDIDATE_MULTIPLIER = 2

    def __init__(
        self,
        attraction_service: AttractionService,
        route_service: RouteService,
        weather_service: WeatherService,
        planner: ItineraryPlanner,
        budget_service: BudgetService,
        restaurant_service: RestaurantService,
    ) -> None:
        self._attractions = attraction_service
        self._routes = route_service
        self._weather = weather_service
        self._planner = planner
        self._budget = budget_service
        self._restaurants = restaurant_service

    async def plan(self, requirements: TripRequirements) -> Itinerary:
        """按需求生成完整行程：取数 → 排程 → 组装时间轴。

        步骤：
        1. daily_limit = DAILY_LIMITS[requirements.travel_style]
           candidate_limit = min(
               daily_limit * requirements.duration_days * self.CANDIDATE_MULTIPLIER,
               RouteService.MAX_ATTRACTIONS,
           )
           （矩阵有 20 个景点的硬上限，候选数必须和它对齐，否则 RouteService 会抛错。）

        2. 取候选：
           response = await self._attractions.search(
               city=requirements.destination,
               interests=requirements.interests or None,
               limit=candidate_limit,
           )
           候选为空时不要抛异常：返回
           Itinerary(trip=requirements, days=[], warnings=["未找到符合条件的目的地景点..."])，
           让前端展示空状态（和 /attractions/search 的约定一致）。

        3. 取矩阵：
           matrix = await self._routes.build_matrix(
               city=requirements.destination,
               attractions=response.items,
               mode=self.DEFAULT_MODE,
           )

        4. 取天气（可选）：
           只有 requirements.start_date 不为 None 才查；查询天数用 duration_days。
           天气失败不能拖垮排程：try/except 包住，把原因 append 到 warnings，
           weather 置为 None 继续往下走。

        5. 取餐厅（可选）：
           按 duration_days 取餐厅，第 N 天用第 N 条；失败只降级为 warning。
           餐厅不交给 LLM 选择，也不暴露成 Tool。

        6. 排程：
           outcome = await self._planner.plan(
               requirements=requirements,
               candidates=response.items,
               matrix=matrix,
               weather=weather,
           )

        7. 组装：
           itinerary = build_itinerary(..., restaurants=restaurants)

        8. 合并 warning（顺序：降级原因 → 组装阶段问题）：
           itinerary.warnings = [*outcome.warnings, *itinerary.warnings]
           不要覆盖，也不要只保留其中一边。

        9. return itinerary

        约束：
        - 这个函数里不写任何时间/距离/费用计算，那些属于 builder 和后续 Service。
        - 只有「取候选失败」应该让请求失败；天气、矩阵缺边都降级为 warning。
        - warnings 顺序固定：天气降级 → planner 降级 → 组装问题，便于排查。
        """
        daily_limit = DAILY_LIMITS[requirements.travel_style]
        candidate_limit = min(
            daily_limit * requirements.duration_days * self.CANDIDATE_MULTIPLIER,
            RouteService.MAX_ATTRACTIONS,
        )

        # 1. 候选景点：为空是正常业务结果（城市没收录），返回空行程而不是 500
        response = await self._attractions.search(
            city=requirements.destination,
            interests=requirements.interests or None,
            limit=candidate_limit,
        )
        if not response.items:
            return Itinerary(
                trip=requirements,
                days=[],
                warnings=[
                    f"未找到「{requirements.destination}」的候选景点："
                    "请确认城市名，或去掉部分兴趣标签后重试"
                ],
            )

        # 2. 交通矩阵：缺边由 builder 兜底，这里失败说明数据源本身有问题
        matrix = await self._routes.build_matrix(
            city=requirements.destination,
            attractions=response.items,
            mode=self.DEFAULT_MODE,
        )

        # 3. 天气是增强信息，失败不能拖垮排程
        warnings: list[str] = []
        weather = None
        if requirements.start_date is not None:
            try:
                weather = await self._weather.get_forecast(
                    city=requirements.destination,
                    start_date=requirements.start_date,
                    days=requirements.duration_days,
                )
            except Exception as exc:  # noqa: BLE001 - 天气失败必须降级
                logger.warning("天气查询失败，继续排程：%s", exc)
                warnings.append(
                    f"天气查询失败，行程未考虑天气因素：{type(exc).__name__}"
                )

        # 4. 餐厅：每天一顿，按天顺序取用；失败只降级，不影响景点排程
        restaurants = []
        try:
            restaurant_response = await self._restaurants.search(
                city=requirements.destination,
                limit=requirements.duration_days,
            )
            restaurants = restaurant_response.items
        except Exception as exc:  # noqa: BLE001 - 餐厅失败必须降级
            logger.warning("餐厅查询失败，行程不含午餐安排：%s", exc)
            warnings.append(
                f"餐厅查询失败，行程未安排午餐：{type(exc).__name__}"
            )

        # 5. 排程：LLM 决定分天与顺序，失败时由 planner 内部降级
        outcome = await self._planner.plan(
            requirements=requirements,
            candidates=response.items,
            matrix=matrix,
            weather=weather,
        )

        # 6. 组装时间轴；各类 warning 都要保留，不能互相覆盖
        itinerary = build_itinerary(
            requirements,
            outcome.draft,
            response.items,
            matrix,
            restaurants=restaurants,
        )
        itinerary.warnings = [*warnings, *outcome.warnings, *itinerary.warnings]

        # 6. 预算：纯计算，失败也不能拖垮行程（和天气同级别的增强信息）
        try:
            budget_outcome = self._budget.estimate(requirements, itinerary)
        except Exception as exc:  # noqa: BLE001 - 预算失败必须降级
            logger.warning("预算计算失败，行程将不含费用：%s", exc)
            itinerary.warnings.append(
                f"预算计算失败，行程未包含费用：{type(exc).__name__}"
            )
        else:
            itinerary.budget = budget_outcome.breakdown
            itinerary.warnings.extend(budget_outcome.warnings)

        return itinerary


@lru_cache(maxsize=1)
def get_itinerary_service() -> ItineraryService:
    """FastAPI 依赖。

    有 OPENAI_API_KEY 时走 LLM 排程（失败自动降级）；
    没有时直接用规则排程，保证开发和测试不被 API Key 阻塞。
    """
    settings = get_settings()
    fallback = RuleBasedItineraryPlanner()

    planner: ItineraryPlanner = fallback
    if settings.openai_api_key:
        planner = LlmItineraryPlanner(
            runnable=create_planner_runnable(settings),
            fallback=fallback,
        )
    else:
        logger.info("未配置 OPENAI_API_KEY，行程生成使用规则排程")

    return ItineraryService(
        attraction_service=AttractionService(provider=MockAttractionProvider()),
        route_service=RouteService(provider=MockRouteProvider()),
        weather_service=WeatherService(provider=MockWeatherProvider()),
        planner=planner,
        budget_service=BudgetService(),
        restaurant_service=RestaurantService(provider=MockRestaurantProvider()),
    )
