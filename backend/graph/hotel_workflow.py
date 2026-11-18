"""入选景点 → 住宿区域 → 酒店 的 Agent 工作流。

    START → agent ──(有 tool_calls)──▶ tools ──▶ agent
              │
              (无 tool_calls)
              ▼
           finalize → END

收口节点只做两件事：
    1. 从所有成功的 gaode_mcp.around_search（poi_kind=hotel）结果里抽酒店，
       按 id 去重并转成 Hotel schema（Python 对齐，不靠 LLM 输出字段）；
    2. 把 Agent 最后一段文字作为区域分析摘要。

这样即使 LLM 在文字里写错酒店名或价格，也不会污染结构化结果。
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from backend.agents.hotel_agent import (
    build_hotel_system_prompt,
    create_hotel_agent,
)
from backend.cache.pool_cache import SelectedAttraction
from backend.graph.hotel_state import HotelSearchState
from backend.graph.nodes import create_agent_node
from backend.schemas.hotel import Hotel
from backend.services.hotel_assembly import hotel_from_summary
from backend.tools.gaode_mcp_tools import HotelSummary
from backend.tools.registry import ToolContext, build_tools

logger = logging.getLogger(__name__)

_HOTEL_TOOL = "gaode_mcp"
_HOTEL_ACTION = "around_search"
_HOTEL_POI_KIND = "hotel"
_HOTEL_TOOL_NAMES = (
    "gaode_mcp",
    "mcp_geocode",
    "mcp_web_search",
    "mcp_route_matrix",
)
_DEFAULT_RADIUS = 2000
_MAX_DESCRIPTION_CHARS = 80


def should_continue(state: HotelSearchState) -> Literal["tools", "finalize"]:
    """和景点初选图一致：最后一条 AI 消息还有 tool_calls 就继续执行工具。"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "finalize"


# ---------------------------- 输入压缩 ----------------------------


def _location_payload(value: Any) -> dict[str, Any] | None:
    """兼容 Attraction.location 是 dict 或 GeoPoint 两种情况。"""
    if value is None:
        return None
    if isinstance(value, dict):
        lng, lat = value.get("lng"), value.get("lat")
    else:
        lng, lat = getattr(value, "lng", None), getattr(value, "lat", None)
    if lng is None or lat is None:
        return None
    return {"lng": lng, "lat": lat}


def _truncate(text: Any, limit: int = _MAX_DESCRIPTION_CHARS) -> str | None:
    if not isinstance(text, str):
        return None
    clean = text.strip()
    if not clean:
        return None
    return clean if len(clean) <= limit else f"{clean[:limit]}…"


def compact_anchors(
    entries: Sequence[SelectedAttraction],
) -> tuple[list[dict[str, Any]], list[str]]:
    """把入选景点裁成酒店 Agent 需要的字段。

    完整 Attraction 里的开放时间、票价、provenance 对「住哪」没有帮助，
    留在提示词里只会撑大上下文。缺坐标的景点无法参与地理分析，直接跳过并告警。
    """
    anchors: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in entries:
        attraction = entry.attraction or {}
        name = str(attraction.get("name") or "").strip()
        location = _location_payload(attraction.get("location"))
        if location is None:
            warnings.append(f"{name or '未命名景点'} 缺少坐标，未参与酒店区域分析")
            continue
        anchors.append(
            {
                "attraction_id": str(attraction.get("id") or ""),
                "name": name,
                "category": attraction.get("category"),
                "address": attraction.get("address"),
                "location": location,
                "score": entry.score,
                "reason": entry.reason,
                "interests": entry.interests,
                "description": _truncate(attraction.get("description")),
            }
        )
    return anchors, warnings


# ---------------------------- 结果解析 ----------------------------


def _coerce_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _tool_payloads(
    messages: Sequence[BaseMessage],
    tool_name: str,
    *,
    action: str | None = None,
    poi_kind: str | None = None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != tool_name:
            continue
        payload = _coerce_payload(message.content)
        if payload is None or not payload.get("ok"):
            continue
        if action is not None and payload.get("action") != action:
            continue
        if poi_kind is not None and payload.get("poi_kind") != poi_kind:
            continue
        payloads.append(payload)
    return payloads


def _last_ai_summary(messages: Sequence[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _hotels_from_messages(
    messages: Sequence[BaseMessage],
    *,
    city: str,
) -> list[Hotel]:
    """按调用顺序汇总酒店，按 id 去重（先出现的区域优先）。"""
    hotels: dict[str, Hotel] = {}
    for payload in _tool_payloads(
        messages,
        _HOTEL_TOOL,
        action=_HOTEL_ACTION,
        poi_kind=_HOTEL_POI_KIND,
    ):
        for row in payload.get("items") or []:
            if not isinstance(row, dict):
                continue
            try:
                summary = HotelSummary.model_validate(row)
            except ValidationError as exc:
                logger.debug("跳过无法解析的酒店条目：%s", exc)
                continue
            hotel = hotel_from_summary(summary, city=city)
            hotels.setdefault(hotel.id, hotel)
    return list(hotels.values())


# ---------------------------- 收口节点 ----------------------------


def create_finalize_node():
    async def finalize(state: HotelSearchState) -> dict[str, Any]:
        messages = state["messages"]
        warnings: list[str] = []
        hotels = _hotels_from_messages(messages, city=state.get("city", ""))
        if not hotels:
            warnings.append(
                "Agent 没有成功调用 gaode_mcp（action=around_search, "
                "poi_kind=hotel），未获得可用酒店"
            )

        return {
            "hotels": [hotel.model_dump(mode="json") for hotel in hotels],
            "hotel_summary": _last_ai_summary(messages),
            "warnings": warnings,
        }

    return finalize


# ---------------------------- 图构建 ----------------------------


def build_hotel_graph(
    *,
    agent: Runnable,
    tools: list[BaseTool],
):
    graph = StateGraph(HotelSearchState)
    graph.add_node("agent", create_agent_node(agent))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finalize", create_finalize_node())

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


# ---------------------------- 入口 ----------------------------


@dataclass
class HotelSearchOutcome:
    """一次酒店搜索的结果。"""

    hotels: list[Hotel]
    summary: str
    source: Literal["agent", "unavailable"]
    warnings: list[str] = field(default_factory=list)


async def run_hotel_search(
    city: str,
    anchors: Sequence[dict[str, Any]],
    *,
    radius_meters: int = _DEFAULT_RADIUS,
    agent: Runnable | None = None,
    tools: list[BaseTool] | None = None,
    tool_context: ToolContext | None = None,
) -> HotelSearchOutcome:
    """运行一次「区域发现 → 酒店搜索」。

    anchors 由 upper 层用 compact_anchors 生成，已是精简字段；
    tools / agent 可注入，测试无需真实网络与 API Key。
    """
    normalized_city = city.strip()
    if not normalized_city:
        raise ValueError("city 不能为空白")
    if not anchors:
        return HotelSearchOutcome(
            hotels=[],
            summary="没有可用的景点坐标，无法分析住宿区域。",
            source="unavailable",
            warnings=["入选景点缺少坐标"],
        )

    if tools is None:
        try:
            tools = build_tools(_HOTEL_TOOL_NAMES, tool_context or ToolContext())
        except ValueError as exc:
            logger.warning("酒店工具不可用：%s", exc)
            tools = []

        if not any(tool.name == _HOTEL_TOOL for tool in tools):
            return HotelSearchOutcome(
                hotels=[],
                summary="高德酒店搜索不可用，请检查 GAODE_MAPS_AK。",
                source="unavailable",
                warnings=["缺少 gaode_mcp（高德 AK 未配置或不可用）"],
            )

    if agent is None:
        try:
            agent = create_hotel_agent(tools)
        except ValueError as exc:
            logger.warning("酒店 Agent 不可用：%s", exc)
            return HotelSearchOutcome(
                hotels=[],
                summary=f"酒店 Agent 不可用：{exc}",
                source="unavailable",
                warnings=["缺少 OPENAI_API_KEY 或酒店工具"],
            )

    graph = build_hotel_graph(agent=agent, tools=tools)
    initial_state: HotelSearchState = {
        "messages": [
            build_hotel_system_prompt(
                normalized_city,
                anchors,
                radius_meters=radius_meters,
                tools=tools,
            ),
            HumanMessage(content="请开始分析住宿区域并搜索酒店"),
        ],
        "city": normalized_city,
        "radius_meters": radius_meters,
        "anchors": list(anchors),
        "hotels": [],
        "hotel_summary": None,
        "warnings": [],
    }
    final_state = await graph.ainvoke(
        initial_state,
        # 区域验证 + 多次搜索 + 比较，留足轮次；同时防跑飞
        config={"recursion_limit": 40},
    )

    hotels = [
        Hotel.model_validate(item) for item in final_state.get("hotels") or []
    ]
    summary = final_state.get("hotel_summary")
    warnings = list(final_state.get("warnings") or [])
    if not summary:
        summary = (
            f"已完成住宿区域分析，搜索到 {len(hotels)} 家酒店。"
            if hotels
            else "未获得可用酒店结果。"
        )

    return HotelSearchOutcome(
        hotels=hotels,
        summary=summary,
        source="agent",
        warnings=warnings,
    )


__all__ = [
    "HotelSearchOutcome",
    "build_hotel_graph",
    "compact_anchors",
    "create_finalize_node",
    "run_hotel_search",
    "should_continue",
]
