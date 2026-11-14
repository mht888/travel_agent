"""景点初选工作流测试：完整循环、收口兜底、无 LLM 降级。"""

import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable

from backend.graph.selection_workflow import (
    SelectionOutcome,
    build_selection_graph,
    run_attraction_selection,
    should_continue,
)
from backend.providers.mock_provider import MockAttractionProvider
from backend.services.attraction_ranking import RuleBasedAttractionRanker
from backend.tools.attraction_selection_tools import (
    AttractionPoolCache,
    create_attraction_selection_tools,
)


def _last_tool_payload(messages: list[BaseMessage], name: str) -> dict[str, Any] | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == name:
            return json.loads(message.content)
    return None


class _FakeAgent(Runnable):
    """按剧本模拟 LLM：搜索 → 排序 → 总结；mode=search_only 时跳过排序。"""

    def __init__(self, mode: str = "full") -> None:
        self.mode = mode
        self.calls = 0

    def invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        return asyncio.run(self.ainvoke(*args, **kwargs))

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_attraction_pool",
                        "args": {"city": "Tokyo"},
                        "id": "call_search",
                    }
                ],
            )

        search_payload = _last_tool_payload(messages, "search_attraction_pool")
        if self.calls == 2 and self.mode == "full" and search_payload:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rank_attraction_pool",
                        "args": {
                            "search_id": search_payload["search_id"],
                            "interests": ["food"],
                            "top_k": 4,
                        },
                        "id": "call_rank",
                    }
                ],
            )

        return AIMessage(content="已完成景点初选，按兴趣优先给出结果。")


def _build_graph(mode: str = "full", top_k: int = 4):
    store = AttractionPoolCache()
    tools = create_attraction_selection_tools(
        provider=MockAttractionProvider(),
        ranker=RuleBasedAttractionRanker(),
        pool_cache=store,
    )
    ranker = RuleBasedAttractionRanker()
    graph = build_selection_graph(
        agent=_FakeAgent(mode),
        tools=tools,
        store=store,
        ranker=ranker,
        top_k=top_k,
    )
    return graph


def _initial_state(interests: list[str]) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="请开始筛选景点")],
        "city": "Tokyo",
        "interests": interests,
        "ranking": None,
        "selection_summary": None,
    }


# ---------------------------- 节点行为 ----------------------------


def test_should_continue_routes_to_tools_and_finalize() -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_attraction_pool", "args": {"city": "Tokyo"}, "id": "c1"}
        ],
    )
    assert should_continue({"messages": [tool_call]}) == "tools"  # type: ignore[typeddict-item]

    assert should_continue(  # type: ignore[typeddict-item]
        {"messages": [AIMessage(content="完成")]}
    ) == "finalize"


# ---------------------------- 完整流程 ----------------------------


def test_full_loop_produces_ranking_and_summary() -> None:
    final_state = asyncio.run(_build_graph("full").ainvoke(_initial_state(["food"])))

    ranking = final_state["ranking"]
    assert ranking is not None
    assert ranking["strategy_used"] == "rule"
    assert ranking["items"][0]["id"] == "tokyo-tsukiji-outer-market"
    assert final_state["selection_summary"] == "已完成景点初选，按兴趣优先给出结果。"


def test_finalize_trims_ranking_to_workflow_top_k() -> None:
    # 假 Agent 给工具传 top_k=4，工作流只允许 2 条
    final_state = asyncio.run(
        _build_graph("full", top_k=2).ainvoke(_initial_state(["food"]))
    )

    ranking = final_state["ranking"]
    assert ranking is not None
    assert len(ranking["items"]) == 2


def test_finalize_ranks_from_cache_when_agent_skips_rank_tool() -> None:
    final_state = asyncio.run(
        _build_graph("search_only").ainvoke(_initial_state(["nature"]))
    )

    ranking = final_state["ranking"]
    assert ranking is not None
    # 明治神宫与新宿御苑同为自然类且评分相同，按 id 稳定排序；只验证兴趣确实生效
    assert ranking["items"][0]["id"] in {"tokyo-meiji-jingu", "tokyo-shinjuku-gyoen"}
    assert "nature" in ranking["items"][0]["reason"]
    assert ranking["strategy_used"] == "rule"


# ---------------------------- 无 LLM 降级 ----------------------------


def test_run_selection_falls_back_when_agent_unavailable(
    monkeypatch: Any,
) -> None:
    import backend.graph.selection_workflow as workflow_module

    def _raise(*_: Any, **__: Any) -> None:
        raise ValueError("missing key")

    monkeypatch.setattr(
        workflow_module, "create_attraction_selection_agent", _raise
    )

    outcome: SelectionOutcome = asyncio.run(
        run_attraction_selection(
            "Tokyo",
            ["food"],
            provider=MockAttractionProvider(),
            ranker=RuleBasedAttractionRanker(),
            top_k=3,
        )
    )

    assert outcome.source == "fallback"
    assert outcome.ranking is not None
    assert outcome.ranking.items[0].id == "tokyo-tsukiji-outer-market"
    assert outcome.warnings
    assert "Tsukiji" in outcome.summary
