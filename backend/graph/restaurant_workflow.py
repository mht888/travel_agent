"""城市 / 区域 → 美食类别 → 餐厅 → 商家核实 的 Agent 工作流。

    START → agent ──(有 tool_calls)──▶ tools ──▶ agent
              │
              (无 tool_calls)
              ▼
           finalize → END

收口节点只做两件事：
    1. 从所有成功的 search_restaurants_around 结果里抽餐厅，
       按 id 去重并转成 Restaurant schema（Python 对齐，不靠 LLM 输出字段）；
    2. 把 Agent 最后一段文字作为美食与餐厅分析摘要。

这样即使 LLM 在文字里写错餐厅名或价格，也不会污染结构化结果。
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from backend.agents.restaurant_agent import (
    build_restaurant_system_prompt,
    create_restaurant_agent,
)
from backend.graph.nodes import create_agent_node
from backend.graph.restaurant_state import RestaurantSearchState
from backend.schemas.common import Money
from backend.schemas.restaurant import Restaurant
from backend.services.restaurant_assembly import restaurant_from_summary
from backend.tools.gaode_restaurant_tools import RestaurantSummary
from backend.tools.registry import ToolContext, build_tools

logger = logging.getLogger(__name__)

_RESTAURANT_TOOL = "search_restaurants_around"
_RESTAURANT_TOOL_NAMES = (
    "search_restaurants_around",
    "mcp_web_search",
    "mcp_geocode",
)
_DISCOVERY_TOOL = "mcp_web_search"
# 四个阶段都要检索，轮次比酒店更多，同时保留跑飞保护
_RECURSION_LIMIT = 50


def should_continue(
    state: RestaurantSearchState,
) -> Literal["tools", "finalize"]:
    """最后一条 AI 消息还有 tool_calls 就继续执行工具。"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "finalize"


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


def _restaurants_from_messages(
    messages: Sequence[BaseMessage],
    *,
    city: str,
) -> list[Restaurant]:
    """按调用顺序汇总餐厅，按 id 去重（先出现的区域优先）。"""
    restaurants: dict[str, Restaurant] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name != _RESTAURANT_TOOL:
            continue
        payload = _coerce_payload(message.content)
        if payload is None or not payload.get("ok"):
            continue
        for row in payload.get("items") or []:
            if not isinstance(row, dict):
                continue
            try:
                summary = RestaurantSummary.model_validate(row)
            except ValidationError as exc:
                logger.debug("跳过无法解析的餐厅条目：%s", exc)
                continue
            restaurant = restaurant_from_summary(summary, city=city)
            if restaurant is not None:
                restaurants.setdefault(restaurant.id, restaurant)
    return list(restaurants.values())


def _tool_warnings(messages: Sequence[BaseMessage]) -> list[str]:
    """保留工具失败原因，方便上层判断这次推荐的可靠程度。"""
    warnings: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = _coerce_payload(message.content)
        if payload is not None and payload.get("ok") is False:
            error = payload.get("error") or payload.get("message") or "未知错误"
            warnings.append(f"{message.name or '检索工具'} 调用失败：{error}")
    return warnings


def _has_tool_result(messages: Sequence[BaseMessage]) -> bool:
    return any(isinstance(message, ToolMessage) for message in messages)


# ---------------------------- 收口节点 ----------------------------


def create_finalize_node():
    async def finalize(state: RestaurantSearchState) -> dict[str, Any]:
        messages = state["messages"]
        warnings = _tool_warnings(messages)
        if not _has_tool_result(messages):
            warnings.append("Agent 没有调用任何检索工具，推荐缺少数据依据")

        restaurants = _restaurants_from_messages(
            messages, city=state.get("city", "")
        )
        if not restaurants:
            warnings.append(
                "Agent 没有成功调用 search_restaurants_around，未获得结构化餐厅"
            )

        summary = _last_ai_summary(messages)
        if not summary:
            warnings.append("Agent 没有输出最终推荐")

        return {
            "restaurants": [
                restaurant.model_dump(mode="json") for restaurant in restaurants
            ],
            "restaurant_summary": summary,
            "warnings": warnings,
        }

    return finalize


# ---------------------------- 图构建 ----------------------------


def build_restaurant_graph(
    *,
    agent: Runnable,
    tools: list[BaseTool],
):
    graph = StateGraph(RestaurantSearchState)
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
class RestaurantSearchOutcome:
    """一次餐厅搜索的结果。

    restaurants 只来自 search_restaurants_around 的工具结果；
    summary 是 Agent 的文字分析，不参与结构化字段。
    """

    restaurants: list[Restaurant]
    summary: str
    source: Literal["agent", "unavailable"]
    warnings: list[str] = field(default_factory=list)


async def run_restaurant_search(
    city: str,
    *,
    area: str | None = None,
    cuisines: Sequence[str] | None = None,
    budget: Money | None = None,
    notes: str | None = None,
    agent: Runnable | None = None,
    tools: list[BaseTool] | None = None,
    tool_context: ToolContext | None = None,
) -> RestaurantSearchOutcome:
    """运行一次「美食发现 → 餐厅发现 → 商家核实 → 排序」。

    tools / agent 可注入，测试无需真实网络与 API Key。
    """
    normalized_city = city.strip()
    if not normalized_city:
        raise ValueError("city 不能为空白")

    if tools is None:
        try:
            tools = build_tools(
                _RESTAURANT_TOOL_NAMES, tool_context or ToolContext()
            )
        except ValueError as exc:
            logger.warning("餐厅工具不可用：%s", exc)
            tools = []

        available = {tool.name for tool in tools}
        if not available & {_DISCOVERY_TOOL, _RESTAURANT_TOOL}:
            return RestaurantSearchOutcome(
                restaurants=[],
                summary="网页检索与餐厅搜索都不可用，无法发现当地美食与餐厅。",
                source="unavailable",
                warnings=[
                    f"缺少 {_DISCOVERY_TOOL} 与 {_RESTAURANT_TOOL}"
                    "（BOCHA_API_KEY 或 GAODE_MAPS_AK 未配置）"
                ],
            )

    if agent is None:
        try:
            agent = create_restaurant_agent(tools)
        except ValueError as exc:
            logger.warning("餐厅 Agent 不可用：%s", exc)
            return RestaurantSearchOutcome(
                restaurants=[],
                summary=f"餐厅 Agent 不可用：{exc}",
                source="unavailable",
                warnings=["缺少 OPENAI_API_KEY 或检索工具"],
            )

    graph = build_restaurant_graph(agent=agent, tools=tools)
    initial_state: RestaurantSearchState = {
        "messages": [
            build_restaurant_system_prompt(
                normalized_city,
                area=area,
                cuisines=list(cuisines or []),
                budget=budget,
                notes=notes,
                tools=tools,
            ),
            HumanMessage(content="请开始发现当地美食并推荐餐厅"),
        ],
        "city": normalized_city,
        "area": area,
        "cuisines": list(cuisines or []),
        "budget": budget.model_dump(mode="json") if budget else None,
        "notes": notes,
        "restaurants": [],
        "restaurant_summary": None,
        "warnings": [],
    }
    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": _RECURSION_LIMIT},
    )

    restaurants = [
        Restaurant.model_validate(item)
        for item in final_state.get("restaurants") or []
    ]
    summary = final_state.get("restaurant_summary")
    warnings = list(final_state.get("warnings") or [])
    if not summary:
        summary = (
            f"已完成美食与餐厅分析，搜索到 {len(restaurants)} 家餐厅。"
            if restaurants
            else "未获得可用的餐厅推荐。"
        )

    return RestaurantSearchOutcome(
        restaurants=restaurants,
        summary=summary,
        source="agent",
        warnings=warnings,
    )


__all__ = [
    "RestaurantSearchOutcome",
    "build_restaurant_graph",
    "create_finalize_node",
    "run_restaurant_search",
    "should_continue",
]
