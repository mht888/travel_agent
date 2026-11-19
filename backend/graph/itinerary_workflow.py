"""用景点、酒店和餐厅 Agent 的结果生成完整行程。"""

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TypeVar, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic_core import to_json

from backend.agents.planner_agent import (
    DAILY_LIMITS,
    LlmItineraryPlanner,
    PlannerOutcome,
    RuleBasedItineraryPlanner,
    create_planner_runnable,
)
from backend.cache.pool_cache import SelectedAttraction
from backend.config import get_settings
from backend.graph.hotel_workflow import HotelSearchOutcome, compact_anchors, run_hotel_search
from backend.graph.restaurant_workflow import RestaurantSearchOutcome, run_restaurant_search
from backend.graph.selection_workflow import SelectionOutcome, run_attraction_selection
from backend.schemas.itinerary import Itinerary
from backend.schemas.route import RouteMatrix
from backend.schemas.trip import TripRequirements
from backend.schemas.weather import WeatherForecast
from backend.services.budget_service import BudgetService
from backend.services.fx_service import FXRateUnavailableError
from backend.services.hotel_assembly import select_top_hotels
from backend.services.itinerary_builder import build_itinerary
from backend.services.itinerary_logging import configure_itinerary_logging
from backend.services.meal_planning import add_breakfast_and_dinner
from backend.services.route_service import RouteService
from backend.services.trip_data import GaodeTripData
from backend.tools.gaode_mcp_tools import create_gaode_mcp_tool

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ItineraryWorkflowState(TypedDict, total=False):
    requirements: TripRequirements
    run_id: str
    selection: SelectionOutcome
    restaurants: RestaurantSearchOutcome | None
    hotels: HotelSearchOutcome | None
    matrix: RouteMatrix | None
    weather: WeatherForecast | None
    draft: PlannerOutcome | None
    selection_warnings: list[str]
    restaurant_warnings: list[str]
    hotel_warnings: list[str]
    route_warnings: list[str]
    weather_warnings: list[str]
    itinerary: Itinerary


async def _optional(label: str, operation: Awaitable[T]) -> tuple[T | None, list[str]]:
    try:
        return await operation, []
    except Exception as exc:
        logger.warning("%s失败：%s", label, type(exc).__name__)
        return None, [f"{label}失败（{type(exc).__name__}）"]


def _log_node(
    name: str, node: Callable[[ItineraryWorkflowState], Awaitable[dict]]
) -> Callable[[ItineraryWorkflowState], Awaitable[dict]]:
    async def run(state: ItineraryWorkflowState) -> dict:
        run_id = state.get("run_id", "-")
        started = perf_counter()
        try:
            output = await node(state)
        except Exception as exc:
            logger.error("run=%s node=%s failed=%s", run_id, name, type(exc).__name__)
            raise
        details = output
        if name == "routes" and (matrix := output["matrix"]) is not None:
            details = {
                **output,
                "matrix": {
                    "attraction_ids": matrix.attraction_ids,
                    "route_count": len(matrix.entries),
                    "sample_routes": matrix.entries[:5],
                },
            }
        logger.info(
            "run=%s node=%s elapsed=%.2fs output=%s",
            run_id, name, perf_counter() - started,
            to_json(details, fallback=str).decode(),
        )
        return output

    return run


def build_itinerary_workflow(data: GaodeTripData):
    async def select(state: ItineraryWorkflowState) -> dict:
        trip = state["requirements"]
        top_k = min(RouteService.MAX_ATTRACTIONS, trip.duration_days * DAILY_LIMITS[trip.travel_style])
        outcome, warnings = await _optional(
            "景点初选",
            run_attraction_selection(
                trip.destination,
                trip.interests or ["经典景点"],
                notes=trip.notes,
                start_date=trip.start_date,
                top_k=top_k,
            ),
        )
        if outcome is None:
            outcome = SelectionOutcome(ranking=None, summary="", source="unavailable")
        # 使用同一批有坐标的景点计算路线和排程。
        located = {item.id: item for item in outcome.attractions if item.location is not None}
        missing = sum(item.location is None for item in outcome.attractions)
        outcome.attractions = list(located.values())
        if missing:
            warnings.append(f"{missing} 个景点缺少坐标，未进入排程")
        return {"selection": outcome, "selection_warnings": warnings + outcome.warnings}

    async def restaurants(state: ItineraryWorkflowState) -> dict:
        trip = state["requirements"]
        outcome, warnings = await _optional(
            "餐厅搜索", run_restaurant_search(trip.destination, notes=trip.notes)
        )
        return {
            "restaurants": outcome,
            "restaurant_warnings": warnings + (outcome.warnings if outcome else []),
        }

    async def hotels(state: ItineraryWorkflowState) -> dict:
        selection = state["selection"]
        ranked = {item.id: item for item in selection.ranking.items} if selection.ranking else {}
        entries = [
            SelectedAttraction(
                attraction=item.model_dump(mode="json"),
                score=ranked[item.id].score if item.id in ranked else 0.0,
                reason=ranked[item.id].reason if item.id in ranked else "",
            )
            for item in selection.attractions
        ]
        anchors, warnings = compact_anchors(entries)
        if not anchors:
            return {"hotels": None, "hotel_warnings": warnings}
        outcome, errors = await _optional(
            "酒店搜索", run_hotel_search(state["requirements"].destination, anchors)
        )
        return {
            "hotels": outcome,
            "hotel_warnings": warnings + errors + (outcome.warnings if outcome else []),
        }

    async def routes(state: ItineraryWorkflowState) -> dict:
        attractions = state["selection"].attractions
        if not attractions:
            return {"matrix": None}
        matrix, warnings = await _optional(
            "交通矩阵查询", data.matrix(state["requirements"].destination, attractions)
        )
        if matrix is not None:
            missing = len(attractions) * (len(attractions) - 1) - len(matrix.entries)
            if missing:
                warnings.append(f"{missing} 条景点间步行路线未查到")
        return {"matrix": matrix, "route_warnings": warnings}

    async def weather(state: ItineraryWorkflowState) -> dict:
        trip = state["requirements"]
        start, end = trip.start_date, trip.end_date
        if start is None or end is None:
            return {"weather": None}
        forecast, warnings = await _optional(
            "天气查询", data.forecast(trip.destination, start, end)
        )
        if forecast is not None and len(forecast.days) < trip.duration_days:
            warnings.append(f"天气预报仅覆盖行程中的 {len(forecast.days)} 天")
        return {"weather": forecast, "weather_warnings": warnings}

    async def plan(state: ItineraryWorkflowState) -> dict:
        matrix = state["matrix"]
        if matrix is None:
            return {"draft": None}
        settings = get_settings()
        fallback = RuleBasedItineraryPlanner()
        planner = (
            LlmItineraryPlanner(create_planner_runnable(settings), fallback)
            if settings.openai_api_key
            else fallback
        )
        outcome = await planner.plan(
            requirements=state["requirements"],
            candidates=state["selection"].attractions,
            matrix=matrix,
            weather=state["weather"],
        )
        return {"draft": outcome}

    async def assemble(state: ItineraryWorkflowState) -> dict:
        trip = state["requirements"]
        warnings = [
            warning
            for key in (
                "selection_warnings", "restaurant_warnings", "hotel_warnings",
                "route_warnings", "weather_warnings",
            )
            for warning in state.get(key, [])
        ]
        outcome, matrix = state["draft"], state["matrix"]
        if outcome is None or matrix is None:
            reason = state["selection"].summary or "景点或路线查询没有产出可用数据"
            itinerary = Itinerary(trip=trip, warnings=[*warnings, reason])
        else:
            candidates = state["selection"].attractions
            itinerary = build_itinerary(trip, outcome.draft, candidates, matrix, require_routes=True)
            restaurants = state["restaurants"]
            candidates_for_meals = [
                *state["selection"].restaurants,
                *(restaurants.restaurants if restaurants else []),
            ]
            await add_breakfast_and_dinner(
                itinerary, candidates, candidates_for_meals, data
            )
            itinerary.warnings[:0] = [*warnings, *outcome.warnings]
            try:
                budget = BudgetService().estimate(trip, itinerary)
            except FXRateUnavailableError as exc:
                itinerary.warnings.append(f"预算计算失败：{exc}")
            except Exception as exc:
                logger.warning("预算计算失败：%s", type(exc).__name__)
                itinerary.warnings.append(f"预算计算失败（{type(exc).__name__}）")
            else:
                itinerary.budget = budget.breakdown
                itinerary.warnings.extend(budget.warnings)
        hotels = state["hotels"]
        if hotels is not None:
            scheduled_ids = {
                item.attraction_id for day in itinerary.days for item in day.items
                if item.attraction_id is not None
            }
            points = [
                item.location for item in state["selection"].attractions
                if item.id in scheduled_ids and item.location is not None
            ]
            itinerary.hotels = select_top_hotels(hotels.hotels, points)
            if itinerary.hotels:
                criteria = "用户评分和到行程景点的距离" if points else "用户评分"
                itinerary.hotel_summary = f"按{criteria}推荐 {len(itinerary.hotels)} 家酒店。"
            else:
                itinerary.hotel_summary = "未找到可推荐的酒店。"
        return {"itinerary": itinerary}

    graph = StateGraph(ItineraryWorkflowState)
    for name, node in {
        "select": select,
        "search_restaurants": restaurants,
        "fetch_weather": weather,
        "search_hotels": hotels,
        "routes": routes,
        "plan": plan,
        "assemble": assemble,
    }.items():
        graph.add_node(name, _log_node(name, node))
    for node in ("select", "search_restaurants", "fetch_weather"):
        graph.add_edge(START, node)
    graph.add_edge("select", "search_hotels")
    graph.add_edge("select", "routes")
    graph.add_edge(["routes", "fetch_weather"], "plan")
    graph.add_edge(["plan", "search_hotels", "search_restaurants"], "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


async def run_itinerary_workflow(requirements: TripRequirements) -> Itinerary:
    configure_itinerary_logging(logger)
    run_id = uuid4().hex
    logger.info(
        "run=%s start requirements=%s",
        run_id, requirements.model_dump_json(exclude={"notes"}),
    )
    settings = get_settings()
    tool = (
        create_gaode_mcp_tool()
        if settings.gaode_maps_enabled and (settings.gaode_maps_ak or "").strip()
        else None
    )
    data = GaodeTripData(tool, concurrency=settings.gaode_mcp_max_concurrency)
    graph = build_itinerary_workflow(data)
    result = await graph.ainvoke({"requirements": requirements, "run_id": run_id})
    return result["itinerary"]
