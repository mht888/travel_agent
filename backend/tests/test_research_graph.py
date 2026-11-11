"""第 6 阶段：LangGraph Tool Calling Loop 测试。

分为两组：
- test_framework_*：验证框架结构（现在就能过，不需要填 TODO）
- test_flow_*：验证完整循环（填完 TODO 后才会过，作为验收标准）

运行：
    uv run pytest backend/tests/test_research_graph.py -v
"""

import asyncio
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.agents.research_agent import build_system_prompt
from backend.graph.state import ResearchState
from backend.schemas.common import Money
from backend.schemas.trip import TripRequirements


# ---------------------------- fixtures ----------------------------


@pytest.fixture()
def sample_requirements() -> TripRequirements:
    return TripRequirements(
        destination="Tokyo",
        duration_days=5,
        start_date=date(2026, 4, 1),
        travelers=2,
        budget=Money(amount=Decimal("1000"), currency="USD"),
        interests=["food", "culture"],
        travel_style="relaxed",
    )


# 需要 Decimal 构造金额
from decimal import Decimal  # noqa: E402


# ---------------------------- 框架结构测试（现在就能过）----------------------------


def test_framework_state_can_hold_messages() -> None:
    """ResearchState 应该能正常存储消息列表。"""
    state: ResearchState = {
        "messages": [HumanMessage(content="你好")],
        "research_summary": None,
    }
    assert len(state["messages"]) == 1
    assert state["research_summary"] is None


def test_framework_system_prompt_contains_requirements(sample_requirements: TripRequirements) -> None:
    """系统提示应该把用户需求结构化地塞进去。"""
    prompt = build_system_prompt(sample_requirements)

    assert isinstance(prompt, SystemMessage)
    content = prompt.content
    assert "Tokyo" in content
    assert "5" in content  # 天数
    assert "food" in content
    assert "culture" in content


def test_framework_imports_work() -> None:
    """关键模块应该能正常 import，不依赖 TODO 是否填充。"""
    from backend.graph.nodes import create_agent_node, create_tool_node
    from backend.graph.state import ResearchState
    from backend.graph.workflow import build_research_graph, run_research, should_continue

    # 函数存在且可调用
    assert callable(create_agent_node)
    assert callable(create_tool_node)
    assert callable(build_research_graph)
    assert callable(run_research)
    assert callable(should_continue)


# ---------------------------- 完整流程测试（填完 TODO 后才会过）----------------------------


def _make_ai_message_with_tool_calls() -> AIMessage:
    """构造一条带 tool_calls 的 AIMessage，模拟 LLM 决定调用工具。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_attractions",
                "args": {"city": "Tokyo", "interests": ["food"]},
                "id": "call_1",
            }
        ],
    )


def test_should_continue_returns_tools_when_tool_calls_exist() -> None:
    """有 tool_calls 时应该返回 'tools'。"""
    from backend.graph.workflow import should_continue

    state: ResearchState = {"messages": [_make_ai_message_with_tool_calls()]}
    assert should_continue(state) == "tools"


def test_should_continue_returns_end_when_no_tool_calls() -> None:
    """没有 tool_calls 时应该返回 END。"""
    from langgraph.graph import END

    from backend.graph.workflow import should_continue

    state: ResearchState = {"messages": [AIMessage(content="调研完成，这是摘要")]}
    assert should_continue(state) == END


def test_build_research_graph_compiles_without_error() -> None:
    """build_research_graph 应该能正常编译，不抛异常。"""
    from backend.graph.workflow import build_research_graph

    graph = build_research_graph()
    assert graph is not None
    # 编译后的图有 get_graph 方法，可以查看节点结构
    assert hasattr(graph, "get_graph")


def test_run_research_returns_summary_string(
    sample_requirements: TripRequirements, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端到端：run_research 应该返回一段字符串摘要。

    这里用一个假的 agent 来避免真实 API 调用。
    假 agent 的行为：第一轮返回带 tool_calls 的消息，第二轮返回纯文本摘要。
    """
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool

    from backend.graph.workflow import run_research

    class _FakeAgent(Runnable):
        """模拟 LLM 的两轮行为：先调工具，再给摘要。"""

        def __init__(self) -> None:
            self._call_count = 0

        def invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
            return asyncio.run(self.ainvoke(*args, **kwargs))

        async def ainvoke(
            self, messages: list[BaseMessage], **kwargs: Any
        ) -> AIMessage:
            self._call_count += 1
            if self._call_count == 1:
                # 第一轮：模拟 LLM 决定调用 search_attractions
                return _make_ai_message_with_tool_calls()
            # 第二轮：看到工具结果后，输出摘要
            return AIMessage(content="调研完成：推荐筑地市场、浅草寺等景点")

        def bind_tools(self, tools: list[BaseTool]) -> "_FakeAgent":
            # 假 agent 不需要真正绑定工具
            return self

    # 替换 create_research_agent，让它返回假 agent
    import backend.graph.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "create_research_agent",
        lambda tools=None: _FakeAgent(),
    )

    summary = asyncio.run(run_research(sample_requirements))
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "调研" in summary or "景点" in summary
