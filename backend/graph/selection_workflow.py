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

finalize 的作用：
    直接读取最后一条 rank_attraction_pool 的 ToolMessage，把结构化结果写进 state。
    不让 LLM 在最终回答里回吐数据——LLM 复述 id / 分数时容易改写事实。

三层降级（保证主流程永远有结果）：
    1. LLM 排序失败 → 排序器内部回退规则；
    2. Agent 整个不可用（无 Key）→ run_attraction_selection 走纯函数流程；
    3. Agent 没调用 rank 工具 → finalize 用缓存的候选池 + 规则排序兜底。
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
from backend.graph.nodes import create_agent_node
from backend.graph.selection_state import AttractionSelectionState
from backend.providers.base import AttractionProvider
from backend.providers.baidu_provider import get_baidu_attraction_provider
from backend.schemas.ranking import AttractionRanking
from backend.services.attraction_ranking import (
    AttractionRanker,
    create_attraction_ranker,
)
from backend.tools.attraction_selection_tools import (
    AttractionPoolCache,
    create_attraction_selection_tools,
)

logger = logging.getLogger(__name__)

_SEARCH_TOOL = "search_attraction_pool"
_RANK_TOOL = "rank_attraction_pool"


def should_continue(
    state: AttractionSelectionState,
) -> Literal["tools", "finalize"]:
    """最后一条 AI 消息还有 tool_calls 就继续执行工具，否则进入收口节点。"""
    messages: list[BaseMessage] = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "finalize"


def _find_last_tool_payload(
    messages: Sequence[BaseMessage],
    tool_name: str,
) -> dict[str, Any] | None:
    """从后往前找指定工具的最后一条成功结果。"""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage) or message.name != tool_name:
            continue
        if not isinstance(message.content, str):
            return None
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _ranking_from_payload(payload: dict[str, Any] | None) -> AttractionRanking | None:
    """把 rank 工具返回的 JSON 还原成 AttractionRanking。"""
    if not payload or not payload.get("ok"):
        return None
    raw_items = payload.get("items") or []
    try:
        return AttractionRanking(
            ranked_ids=[item["id"] for item in raw_items],
            items=[
                {
                    "id": item["id"],
                    "score": item.get("score", 0.0),
                    "reason": item.get("reason", ""),
                }
                for item in raw_items
            ],
            strategy_used=payload.get("strategy_used", "rule"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("rank 工具结果无法解析：%s", exc)
        return None


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


def create_finalize_node(
    *,
    store: AttractionPoolCache,
    ranker: AttractionRanker,
    top_k: int = 8,
):
    """收口节点：解析排序结果；Agent 漏调 rank 时用缓存兜底。"""

    async def finalize(
        state: AttractionSelectionState,
    ) -> dict[str, Any]:
        messages = state["messages"]

        ranking = _ranking_from_payload(_find_last_tool_payload(messages, _RANK_TOOL))
        if ranking is not None and len(ranking.items) > top_k:
            # Agent 可能给工具传了更大的 top_k，最终数量以工作流配置为准
            ranking = AttractionRanking(
                ranked_ids=ranking.ranked_ids[:top_k],
                items=ranking.items[:top_k],
                strategy_used=ranking.strategy_used,
            )

        if ranking is None:
            # Agent 可能只搜没排：复用缓存里的候选池，再按兴趣排一次
            search_payload = _find_last_tool_payload(messages, _SEARCH_TOOL)
            search_id = (search_payload or {}).get("search_id")
            cached = store.get(search_id) if search_id else None
            if cached is not None:
                city, attractions = cached
                logger.info("Agent 未产出排序结果，使用候选池兜底排序：%s", search_id)
                ranking = await ranker.rank(
                    city=city,
                    attractions=attractions,
                    interests=state.get("interests") or [],
                    top_k=top_k,
                )

        return {
            "ranking": ranking.model_dump(mode="json") if ranking else None,
            "selection_summary": _last_ai_summary(messages),
        }

    return finalize


def build_selection_graph(
    *,
    agent: Runnable,
    tools: list[BaseTool],
    store: AttractionPoolCache,
    ranker: AttractionRanker,
    top_k: int = 8,
):
    """构建并编译「检索 → 排序 → 收口」的图。"""
    graph = StateGraph(AttractionSelectionState)
    graph.add_node("agent", create_agent_node(agent))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node(
        "finalize",
        create_finalize_node(store=store, ranker=ranker, top_k=top_k),
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


@dataclass
class SelectionOutcome:
    """一次初选的结果 + 它走的是哪条路径。"""

    ranking: AttractionRanking | None
    summary: str
    source: Literal["agent", "fallback"]
    warnings: list[str] = field(default_factory=list)


async def _run_fallback(
    *,
    city: str,
    interests: list[str],
    provider: AttractionProvider,
    ranker: AttractionRanker,
    top_k: int,
    reason: str,
) -> SelectionOutcome:
    """纯函数流程：直接检索 + 排序，不经过 LLM 编排。"""
    attractions = await provider.search_attractions(
        city=city,
        interests=interests or None,
        limit=max(20, top_k),
    )
    if not attractions:
        return SelectionOutcome(
            ranking=None,
            summary=f"{city} 没有检索到候选景点。",
            source="fallback",
            warnings=[reason],
        )

    ranking = await ranker.rank(
        city=city,
        attractions=attractions,
        interests=interests,
        top_k=top_k,
    )
    names = [
        next(
            (attraction.name for attraction in attractions if attraction.id == item.id),
            item.id,
        )
        for item in ranking.items
    ]
    summary = (
        f"共检索到 {len(attractions)} 个候选景点，"
        f"按兴趣选出前 {len(ranking.items)} 个：{'、'.join(names)}。"
    )
    return SelectionOutcome(
        ranking=ranking,
        summary=summary,
        source="fallback",
        warnings=[reason],
    )


async def run_attraction_selection(
    city: str,
    interests: Sequence[str] | None = None,
    *,
    notes: str | None = None,
    top_k: int = 8,
    agent: Runnable | None = None,
    tools: list[BaseTool] | None = None,
    provider: AttractionProvider | None = None,
    ranker: AttractionRanker | None = None,
    store: AttractionPoolCache | None = None,
) -> SelectionOutcome:
    """运行一次景点初选。

    参数可注入，方便测试与 A/B 对比：
    - 传 ranker=RuleBasedAttractionRanker() 就是纯规则；
    - 不传则按 ATTRACTION_RANKING_STRATEGY 决定 rule / llm。
    """
    normalized_city = city.strip()
    if not normalized_city:
        raise ValueError("city 不能为空白")

    normalized_interests = [
        interest.strip() for interest in (interests or []) if interest.strip()
    ]
    provider = provider or get_baidu_attraction_provider()
    ranker = ranker or create_attraction_ranker()
    store = store or AttractionPoolCache()

    if agent is None:
        tools = tools or create_attraction_selection_tools(
            provider=provider,
            ranker=ranker,
            pool_cache=store,
        )
        try:
            agent = create_attraction_selection_agent(tools)
        except ValueError as exc:
            logger.warning("景点初选 Agent 不可用，降级为纯函数流程：%s", exc)
            return await _run_fallback(
                city=normalized_city,
                interests=normalized_interests,
                provider=provider,
                ranker=ranker,
                top_k=top_k,
                reason=f"Agent 不可用，已降级为纯函数流程：{exc}",
            )
    else:
        tools = tools or create_attraction_selection_tools(
            provider=provider,
            ranker=ranker,
            pool_cache=store,
        )

    graph = build_selection_graph(
        agent=agent,
        tools=tools,
        store=store,
        ranker=ranker,
        top_k=top_k,
    )
    initial_state: AttractionSelectionState = {
        "messages": [
            build_selection_system_prompt(normalized_city, normalized_interests, notes),
            HumanMessage(content="请开始筛选景点"),
        ],
        "city": normalized_city,
        "interests": normalized_interests,
        "ranking": None,
        "selection_summary": None,
    }
    final_state = await graph.ainvoke(initial_state)

    ranking_payload = final_state.get("ranking")
    ranking = (
        AttractionRanking.model_validate(ranking_payload)
        if ranking_payload
        else None
    )
    summary = final_state.get("selection_summary")

    if ranking is None:
        fallback = await _run_fallback(
            city=normalized_city,
            interests=normalized_interests,
            provider=provider,
            ranker=ranker,
            top_k=top_k,
            reason="Agent 未产出可用排序结果，已降级为纯函数流程",
        )
        if summary:
            fallback.summary = summary
        return fallback

    return SelectionOutcome(
        ranking=ranking,
        summary=summary or "",
        source="agent",
        warnings=[],
    )
