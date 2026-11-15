"""景点初选 LangGraph 工作流。

    START
      │
      ▼
    agent ──(有 tool_calls)──▶ tools
      ▲                          │
      └──────────────────────────┘
      │
      (无 tool_calls)
      ▼
   finalize ──▶ END

Agent 绑定三个工具：
    mcp_web_search：每个兴趣只发一条「城市 兴趣 景点 推荐」；
    extract_web_places：从网页结果里抽地点，写入轻量候选池，返回 search_id；
        纯 Python 统计每个兴趣的数量，不足 10 个返回 insufficient_places，
        Agent 用更大的 count 补检索后重新抽取（抽取会话累计去重）；
    rank_attraction_pool：按 search_id 取回候选池，交给 LLM 打分并截断 top_k。

网页地点只有名称/介绍/分类，坐标等事实数据等入选后再补。
不再有百度候选池降级链路：博查或 LLM 不可用时直接返回空结果。

finalize 只做收口：
    1. 优先解析最后一条 rank_attraction_pool 的结构化结果；
    2. LLM 的最终回答只作为 summary，不作为数据来源。
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

from backend.agents.attraction_selection_agent import (
    build_selection_system_prompt,
    create_attraction_selection_agent,
)
from backend.cache.pool_cache import AttractionPoolCache, WebSearchResultCache
from backend.graph.nodes import create_agent_node
from backend.graph.selection_state import AttractionSelectionState
from backend.schemas.ranking import AttractionRanking, RankingItem
from backend.tools.registry import ToolContext, build_tools

logger = logging.getLogger(__name__)

_RANK_TOOL = "rank_attraction_pool"
_DEFAULT_TOP_K = 8


def should_continue(
    state: AttractionSelectionState,
) -> Literal["tools", "finalize"]:
    """最后一条 AI 消息还有 tool_calls 就继续执行工具，否则进入收口节点。"""
    messages: list[BaseMessage] = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "finalize"


# ---------------------------- 消息解析 ----------------------------


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
) -> list[dict[str, Any]]:
    """收集指定工具的成功返回，保持调用顺序。"""
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != tool_name:
            continue
        payload = _coerce_payload(message.content)
        if payload is not None and payload.get("ok"):
            payloads.append(payload)
    return payloads


def _last_ai_summary(messages: Sequence[BaseMessage]) -> str | None:
    """取最后一条不带 tool_calls 的 AI 文本作为总结。"""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _ranking_from_rank_payload(
    payload: dict[str, Any],
    top_k: int,
) -> AttractionRanking | None:
    """把 rank_attraction_pool 的 JSON 还原成 AttractionRanking。"""
    items: list[RankingItem] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        try:
            items.append(
                RankingItem(
                    id=str(item["id"]),
                    score=float(item["score"]),
                    reason=str(item.get("reason") or ""),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not items:
        return None
    # 网页链路已按兴趣各自截断，这里不能再做全局截断，否则会丢掉分组名额
    if not payload.get("by_interest"):
        items = items[:top_k]
    return AttractionRanking(ranked_ids=[item.id for item in items], items=items)


def _ranking_from_messages(
    messages: Sequence[BaseMessage],
    top_k: int,
) -> AttractionRanking | None:
    """取最后一次成功的 rank_attraction_pool 结果。"""
    for payload in reversed(_tool_payloads(messages, _RANK_TOOL)):
        ranking = _ranking_from_rank_payload(payload, top_k)
        if ranking is not None:
            return ranking
    return None


# ---------------------------- 收口节点 ----------------------------


def create_finalize_node(
    *,
    top_k: int = _DEFAULT_TOP_K,
):
    """收口节点：优先用 Agent 的排序结果，缺失时给出警告。"""

    async def finalize(
        state: AttractionSelectionState,
    ) -> dict[str, Any]:
        messages = state["messages"]
        warnings: list[str] = []

        ranking = _ranking_from_messages(messages, top_k)
        if ranking is None:
            warnings.append("Agent 没有调用 rank_attraction_pool，没有可用的排序结果")

        return {
            "ranking": ranking.model_dump(mode="json") if ranking else None,
            "selection_summary": _last_ai_summary(messages),
            "warnings": warnings,
        }

    return finalize


def build_selection_graph(
    *,
    agent: Runnable,
    tools: list[BaseTool],
    top_k: int = _DEFAULT_TOP_K,
):
    """构建并编译「检索 → 排序」的图。"""
    graph = StateGraph(AttractionSelectionState)
    graph.add_node("agent", create_agent_node(agent))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node(
        "finalize",
        create_finalize_node(top_k=top_k),
    )

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
class SelectionOutcome:
    """一次初选的结果 + 它走的是哪条路径。"""

    ranking: AttractionRanking | None
    summary: str
    source: Literal["agent", "unavailable"]
    warnings: list[str] = field(default_factory=list)


async def run_attraction_selection(
    city: str,
    interests: Sequence[str] | None = None,
    *,
    notes: str | None = None,
    top_k: int = _DEFAULT_TOP_K,
    agent: Runnable | None = None,
    tools: list[BaseTool] | None = None,
    store: AttractionPoolCache | None = None,
) -> SelectionOutcome:
    """运行一次景点初选。

    参数可注入，方便测试：
    - tools 不传时按 registry 现场组装 mcp_web_search + extract_web_places
      + rank_attraction_pool；
    - store 是候选池缓存，不传则新建。
    """
    normalized_city = city.strip()
    if not normalized_city:
        raise ValueError("city 不能为空白")

    normalized_interests = [
        interest.strip() for interest in (interests or []) if interest.strip()
    ]
    if not normalized_interests:
        raise ValueError("interests 不能为空：初选需要按兴趣排序")

    store = store or AttractionPoolCache()
    if tools is None:
        web_ctx = ToolContext(
            bocha_result_cache=WebSearchResultCache(),
            selection_cache=store,
            strict_web_query=True,
        )
        try:
            tools = build_tools(
                ["mcp_web_search", "extract_web_places", "rank_attraction_pool"],
                web_ctx,
            )
        except ValueError as exc:
            logger.warning("网页初选工具不可用：%s", exc)
            tools = []

        if not tools:
            return SelectionOutcome(
                ranking=None,
                summary="博查网页检索或地点排序不可用，请检查 BOCHA_APIKEY 与 OPENAI_API_KEY。",
                source="unavailable",
                warnings=["网页检索链路不可用（缺少博查或 LLM 配置）"],
            )

    if agent is None:
        try:
            agent = create_attraction_selection_agent(tools)
        except ValueError as exc:
            logger.warning("景点初选 Agent 不可用：%s", exc)
            return SelectionOutcome(
                ranking=None,
                summary=f"景点初选 Agent 不可用：{exc}",
                source="unavailable",
                warnings=["缺少 OPENAI_API_KEY，无法进行网页地点初选"],
            )

    graph = build_selection_graph(
        agent=agent,
        tools=tools,
        top_k=top_k,
    )
    initial_state: AttractionSelectionState = {
        "messages": [
            build_selection_system_prompt(
                normalized_city,
                normalized_interests,
                notes,
                tools=tools,
            ),
            HumanMessage(content="请开始筛选景点"),
        ],
        "city": normalized_city,
        "interests": normalized_interests,
        "ranking": None,
        "selection_summary": None,
        "warnings": [],
    }
    final_state = await graph.ainvoke(
        initial_state,
        # 每轮「补检索 → 重新抽取」至少 4 步；4 轮抽取 + 排序留足余量
        config={"recursion_limit": 60},
    )

    raw_ranking = final_state.get("ranking")
    ranking = (
        AttractionRanking.model_validate(raw_ranking) if raw_ranking else None
    )
    summary = final_state.get("selection_summary")
    warnings = list(final_state.get("warnings") or [])

    if ranking is None:
        return SelectionOutcome(
            ranking=None,
            summary="Agent 未产出可用排序，请重试或检查检索结果。",
            source="agent",
            warnings=warnings,
        )

    if not summary:
        summary = (
            f"已按 {len(normalized_interests)} 个兴趣完成景点排序，"
            f"前 {len(ranking.items)} 名见 ranking。"
        )

    return SelectionOutcome(
        ranking=ranking,
        summary=summary,
        source="agent",
        warnings=warnings,
    )


__all__ = [
    "SelectionOutcome",
    "build_selection_graph",
    "create_finalize_node",
    "run_attraction_selection",
    "should_continue",
]
