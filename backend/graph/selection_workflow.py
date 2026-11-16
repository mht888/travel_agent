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

图跑完后，入选的 Attraction 连同 score/reason/interests 写入候选池缓存
（kind=selected_attraction），SelectionOutcome 返回 selection_id，
后续排程可用 store.load_selection(selection_id) 取回。
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
from backend.cache.pool_cache import (
    AttractionPoolCache,
    SelectedAttraction,
    WebSearchResultCache,
)
from backend.graph.nodes import create_agent_node
from backend.graph.selection_state import AttractionSelectionState
from backend.schemas.attraction import Attraction, AttractionCategory
from backend.schemas.ranking import AttractionRanking, RankingItem
from backend.services.attraction_assembly import (
    AttractionAssembler,
    create_attraction_assembler,
)
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


def _coerce_category(value: Any) -> AttractionCategory:
    try:
        return AttractionCategory(str(value or "").strip().lower())
    except ValueError:
        return AttractionCategory.OTHER


def _ranking_item_from_row(row: dict[str, Any]) -> RankingItem | None:
    """把工具返回的一行还原成 RankingItem，保留组装 Attraction 需要的字段。"""
    try:
        return RankingItem(
            id=str(row["id"]),
            score=float(row["score"]),
            reason=str(row.get("reason") or ""),
            name=str(row.get("name") or ""),
            category=_coerce_category(row.get("category")),
            description=str(row.get("description") or ""),
            source_url=row.get("source_url"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _ranking_from_rank_payload(
    payload: dict[str, Any],
    top_k: int,
) -> AttractionRanking | None:
    """把 rank_attraction_pool 的 JSON 还原成 AttractionRanking。"""
    items: list[RankingItem] = []
    for row in payload.get("items") or []:
        if not isinstance(row, dict):
            continue
        item = _ranking_item_from_row(row)
        if item is not None:
            items.append(item)
    if not items:
        return None
    # 网页链路已按兴趣各自截断，这里不能再做全局截断，否则会丢掉分组名额
    if not payload.get("by_interest"):
        items = items[:top_k]

    by_interest: dict[str, list[RankingItem]] = {}
    raw_by_interest = payload.get("by_interest")
    if isinstance(raw_by_interest, dict):
        for interest, rows in raw_by_interest.items():
            if not isinstance(rows, list):
                continue
            parsed: list[RankingItem] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = _ranking_item_from_row(row)
                if item is not None:
                    parsed.append(item)
            if parsed:
                by_interest[str(interest)] = parsed

    search_id = payload.get("search_id")
    return AttractionRanking(
        ranked_ids=[item.id for item in items],
        items=items,
        search_id=str(search_id) if search_id else None,
        by_interest=by_interest,
    )


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
    assembler: AttractionAssembler | None = None,
):
    """收口节点：优先用 Agent 的排序结果，并组装成 Attraction。"""

    async def finalize(
        state: AttractionSelectionState,
    ) -> dict[str, Any]:
        messages = state["messages"]
        warnings: list[str] = []

        ranking = _ranking_from_messages(messages, top_k)
        if ranking is None:
            warnings.append("Agent 没有调用 rank_attraction_pool，没有可用的排序结果")

        attractions: list[Attraction] = []
        if ranking is not None:
            if assembler is None:
                warnings.append(
                    "未配置景点组装器，本次只返回 ranking，未组装 Attraction"
                )
            else:
                attractions, assemble_warnings = await assembler(
                    city=state.get("city", ""),
                    items=ranking.items,
                )
                warnings.extend(assemble_warnings)

        return {
            "ranking": ranking.model_dump(mode="json") if ranking else None,
            "attractions": [
                attraction.model_dump(mode="json") for attraction in attractions
            ],
            "selection_summary": _last_ai_summary(messages),
            "warnings": warnings,
        }

    return finalize


def build_selection_graph(
    *,
    agent: Runnable,
    tools: list[BaseTool],
    top_k: int = _DEFAULT_TOP_K,
    assembler: AttractionAssembler | None = None,
):
    """构建并编译「检索 → 排序 → 组装」的图。"""
    graph = StateGraph(AttractionSelectionState)
    graph.add_node("agent", create_agent_node(agent))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node(
        "finalize",
        create_finalize_node(top_k=top_k, assembler=assembler),
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


def _selected_attractions(
    attractions: Sequence[Attraction],
    ranking: AttractionRanking,
) -> list[SelectedAttraction]:
    """把入选 Attraction 与它命中的兴趣、排序理由合并成缓存条目。"""
    reasons = {item.id: item for item in ranking.items}
    interests_by_id: dict[str, list[str]] = {}
    for interest, items in ranking.by_interest.items():
        for item in items:
            interests_by_id.setdefault(item.id, []).append(interest)

    selected: list[SelectedAttraction] = []
    for attraction in attractions:
        ranked = reasons.get(attraction.id)
        selected.append(
            SelectedAttraction(
                attraction=attraction.model_dump(mode="json"),
                score=ranked.score if ranked is not None else 0.0,
                reason=ranked.reason if ranked is not None else "",
                interests=interests_by_id.get(attraction.id, []),
            )
        )
    return selected


@dataclass
class SelectionOutcome:
    """一次初选的结果 + 它走的是哪条路径。"""

    ranking: AttractionRanking | None
    summary: str
    source: Literal["agent", "unavailable"]
    attractions: list[Attraction] = field(default_factory=list)
    # 入选景点已写入 store 时给出 id，可凭它取回 Attraction + 排序理由
    selection_id: str | None = None
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
    assembler: AttractionAssembler | None = None,
) -> SelectionOutcome:
    """运行一次景点初选。

    参数可注入，方便测试：
    - tools 不传时按 registry 现场组装 mcp_web_search + extract_web_places
      + rank_attraction_pool；
    - store 是候选池缓存（同时用来保存入选景点），不传则新建；
    - assembler 不传时按配置创建（缺百度 AK 则为 None，只返回 ranking）。
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
    assembler = assembler or create_attraction_assembler()
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
        assembler=assembler,
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
        "attractions": [],
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
    attractions = [
        Attraction.model_validate(item)
        for item in final_state.get("attractions") or []
    ]
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
            f"前 {len(ranking.items)} 名见 ranking，"
            f"已组装 {len(attractions)} 个景点。"
        )

    # 入选景点连同排序理由写入候选池缓存，后续排程按 selection_id 取回
    selection_id = (
        store.save_selection(
            normalized_city, _selected_attractions(attractions, ranking)
        )
        if attractions
        else None
    )

    return SelectionOutcome(
        ranking=ranking,
        summary=summary,
        source="agent",
        attractions=attractions,
        selection_id=selection_id,
        warnings=warnings,
    )


__all__ = [
    "SelectionOutcome",
    "build_selection_graph",
    "create_finalize_node",
    "run_attraction_selection",
    "should_continue",
]
