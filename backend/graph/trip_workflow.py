"""景点初选 → 酒店搜索的 LangGraph 编排。

    START
      │
      ▼
    selection ──(selection_id 非空)──▶ hotel ──▶ END
      │
      └──(selection_id 为空)──▶ fallback ──▶ END

设计要点：
    - selection node 只是 run_attraction_selection 的薄包装，
      只负责把 SelectionOutcome 摊平成 state 字段，不重复初选逻辑；
    - 「酒店 Agent 该干活了」的信号就是条件边路由到 hotel node，
      不需要 Agent 之间对话、也不需要轮询；
    - hotel node 先凭 selection_id 从缓存取回入选景点并压缩字段，
      再调用酒店 Agent；酒店实体由 hotel_workflow 的收口节点统一
      对齐 Hotel schema，本节点只负责搬运和记录状态。
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from backend.cache.pool_cache import AttractionPoolCache
from backend.graph.hotel_workflow import (
    HotelSearchOutcome,
    compact_anchors,
    run_hotel_search,
)
from backend.graph.selection_workflow import (
    SelectionOutcome,
    run_attraction_selection,
)
from backend.graph.trip_state import TripPlanningState
from backend.services.attraction_assembly import AttractionAssembler

logger = logging.getLogger(__name__)

# 把初选的具体依赖（agent/tools/store/assembler）挡在图定义之外，
# 图本身只认识「城市 + 兴趣 → SelectionOutcome」这个契约。
SelectionRunner = Callable[[str, list[str]], Awaitable[SelectionOutcome]]

# 酒店阶段同理：图只认识「城市 + 压缩景点 → HotelSearchOutcome」。
HotelRunner = Callable[
    [str, list[dict[str, Any]]], Awaitable[HotelSearchOutcome]
]


def create_selection_node(
    selection_runner: SelectionRunner,
) -> Callable[[TripPlanningState], Awaitable[dict[str, Any]]]:
    """把初选结果收口成 state 更新。"""

    async def selection_node(state: TripPlanningState) -> dict[str, Any]:
        outcome = await selection_runner(
            state["city"], list(state.get("interests") or [])
        )
        return {
            "selection_id": outcome.selection_id,
            "selection_summary": outcome.summary,
            "warnings": outcome.warnings,
        }

    return selection_node


def route_after_selection(
    state: TripPlanningState,
) -> Literal["hotel", "fallback"]:
    """条件边：没有 selection_id 就没有酒店阶段。

    下游酒店阶段要靠 selection_id 去缓存取景点坐标；
    id 为空说明初选没产出可选景点（或装配失败），直接走兜底，
    避免酒店节点拿到空 id 后误判为「缓存过期」。
    """
    return "hotel" if state.get("selection_id") else "fallback"


def create_hotel_node(
    *,
    store: AttractionPoolCache,
    runner: HotelRunner | None = None,
) -> Callable[[TripPlanningState], Awaitable[dict[str, Any]]]:
    """酒店节点：selection_id → 缓存取景 → 压缩 → 跑酒店 Agent。

    酒店实体在 hotel_workflow 的收口节点里生成，
    这里只把结果落进 trip state，并区分「跑完有结果 / 跑完没结果 /
    跳过 / 工具不可用」，方便上层决定是否继续排程。
    """
    hotel_runner = runner or run_hotel_search

    async def hotel_node(state: TripPlanningState) -> dict[str, Any]:
        warnings = list(state.get("warnings") or [])
        selection_id = state.get("selection_id")
        entries = store.load_selection(selection_id) if selection_id else None
        if entries is None:
            warnings.append(
                f"selection_id={selection_id} 取不到入选景点（可能已过期）"
            )
            return {
                "hotel_status": "skipped",
                "hotels": [],
                "hotel_summary": "入选景点缓存已失效，跳过酒店搜索。",
                "warnings": warnings,
            }

        anchors, anchor_warnings = compact_anchors(entries)
        warnings.extend(anchor_warnings)
        if not anchors:
            return {
                "hotel_status": "skipped",
                "hotels": [],
                "hotel_summary": "入选景点都缺少坐标，无法分析住宿区域。",
                "warnings": warnings,
            }

        outcome = await hotel_runner(state["city"], anchors)
        warnings.extend(outcome.warnings)
        hotels = [hotel.model_dump(mode="json") for hotel in outcome.hotels]
        if hotels:
            status = "done"
        elif outcome.source == "unavailable":
            status = "unavailable"
        else:
            status = "empty"

        return {
            "hotel_status": status,
            "hotels": hotels,
            "hotel_summary": outcome.summary,
            "warnings": warnings,
        }

    return hotel_node


def fallback_node(state: TripPlanningState) -> dict[str, Any]:
    """初选没产出 selection_id 时的兜底出口，不进入酒店阶段。"""
    logger.info("初选未产出 selection_id，跳过酒店阶段")
    return {
        "hotel_status": "skipped",
        "hotel_summary": "景点初选未产出可用结果，已跳过酒店搜索。",
    }


def build_trip_planning_graph(
    *,
    selection_runner: SelectionRunner,
    store: AttractionPoolCache,
    hotel_runner: HotelRunner | None = None,
):
    """构建并编译「初选 → 酒店」图。

    selection_runner / hotel_runner 由调用方注入：生产用默认实现，
    测试可以传直接返回结果的假实现。
    store 必须与写 selection_id 的缓存是同一个实例，
    否则酒店节点取不到入选景点。
    """
    graph = StateGraph(TripPlanningState)
    graph.add_node("selection", create_selection_node(selection_runner))
    graph.add_node(
        "hotel", create_hotel_node(store=store, runner=hotel_runner)
    )
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "selection")
    graph.add_conditional_edges(
        "selection",
        route_after_selection,
        {"hotel": "hotel", "fallback": "fallback"},
    )
    graph.add_edge("hotel", END)
    graph.add_edge("fallback", END)
    return graph.compile()


async def run_trip_planning(
    city: str,
    interests: Sequence[str] | None = None,
    *,
    store: AttractionPoolCache | None = None,
    agent: Runnable | None = None,
    tools: list[BaseTool] | None = None,
    assembler: AttractionAssembler | None = None,
    hotel_runner: HotelRunner | None = None,
) -> TripPlanningState:
    """跑一次「初选 → 酒店」编排。

    store 必须由这里统一持有并传给初选：selection_id 是写进这个
    store 的，酒店阶段接入后要复用同一个实例才取得到景点。
    """
    pool_store = store or AttractionPoolCache()

    async def selection_runner(
        run_city: str, run_interests: list[str]
    ) -> SelectionOutcome:
        return await run_attraction_selection(
            run_city,
            run_interests,
            agent=agent,
            tools=tools,
            store=pool_store,
            assembler=assembler,
        )

    graph = build_trip_planning_graph(
        selection_runner=selection_runner,
        store=pool_store,
        hotel_runner=hotel_runner,
    )
    initial_state: TripPlanningState = {
        "city": city,
        "interests": list(interests or []),
        "hotel_status": "pending",
        "hotels": [],
    }
    return await graph.ainvoke(initial_state)
