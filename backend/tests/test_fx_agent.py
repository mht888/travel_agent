r"""FX Rate Agent 测试。

覆盖四件事：
1. 图结构与条件边：有 tool_calls 走 tools，无 tool_calls 结束；
2. LLM 循环：fetch → validate → publish 后能真正落盘；
3. 校验门禁：LLM 即使强行调用 publish，异常汇率也发布不了；
4. 降级：没有 LLM 时走确定性流程；所有源失败时保留旧快照。

全部使用临时目录和假数据源，不联网、不写生产数据。

运行：
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_fx_agent.py -v
"""

import asyncio
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from backend.graph.fx_workflow import (
    build_fx_graph,
    run_fx_agent,
    should_continue,
)
from backend.providers.fx_provider import ExchangeRateProviderError
from backend.schemas.exchange_rate import FXSnapshot
from backend.services.fx_service import FXService
from backend.tools.fx_tools import get_fx_tools

CORE_CODES = ("USD", "EUR", "JPY", "CNY", "GBP", "CHF", "AUD", "CAD")


# ---------------------------- 测试替身 ----------------------------


def _rates(**overrides: Decimal) -> dict[str, Decimal]:
    """一份能通过硬校验的汇率表，仅用于测试。"""
    rates = {code: Decimal("1") for code in CORE_CODES}
    rates.update({"THB": Decimal("36"), "SGD": Decimal("1.35")})
    rates.update(overrides)
    return rates


class _StubProvider:
    """可切换成功/失败的数据源。"""

    def __init__(
        self,
        name: str,
        rates: dict[str, Decimal] | None = None,
        business_date: date = date(2026, 11, 12),
    ) -> None:
        self.name = name
        self.rates = dict(rates or _rates())
        self.business_date = business_date
        self.error: Exception | None = None

    async def fetch_latest(self, base: str = "USD") -> FXSnapshot:
        if self.error is not None:
            raise self.error
        return FXSnapshot(
            base=base,
            rates=dict(self.rates),
            fetched_at=datetime.now(timezone.utc),
            business_date=self.business_date,
            source=self.name,
        )


class _FakeAgent:
    """按脚本返回 tool_calls，最后返回纯文本摘要。"""

    def __init__(self, steps: list[list[dict[str, Any]]], final: str) -> None:
        self._steps = list(steps)
        self._final = final

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        if self._steps:
            return AIMessage(content="", tool_calls=self._steps.pop(0))
        return AIMessage(content=self._final)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _service(tmp_path: Path, **providers: _StubProvider) -> FXService:
    return FXService(data_root=tmp_path, providers=providers)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------- 图结构 ----------------------------


def test_should_continue_routes_to_tools_when_tool_calls_exist() -> None:
    state = {"messages": [AIMessage(content="", tool_calls=[_tool_call("validate_snapshot", {}, "1")])]}
    assert should_continue(state) == "tools"


def test_should_continue_ends_without_tool_calls() -> None:
    state = {"messages": [AIMessage(content="同步完成")]}
    assert should_continue(state) == END


def test_build_fx_graph_compiles(tmp_path: Path) -> None:
    service = _service(tmp_path)
    graph = build_fx_graph(
        agent=_FakeAgent([], "完成"),
        tools=get_fx_tools(service),
    )
    nodes = set(graph.get_graph().nodes)
    assert {"agent", "tools", "__start__", "__end__"} <= nodes


def test_registers_expected_tools(tmp_path: Path) -> None:
    names = {tool.name for tool in get_fx_tools(_service(tmp_path))}
    assert names == {
        "fetch_fx_rates",
        "get_last_snapshot",
        "validate_snapshot",
        "publish_snapshot",
        "get_supported_currencies",
    }


# ---------------------------- LLM 主流程 ----------------------------


def test_agent_publishes_snapshot_after_validation(tmp_path: Path) -> None:
    ecb = _StubProvider("ecb")
    service = _service(tmp_path, ecb=ecb)
    tools = get_fx_tools(service)
    agent = _FakeAgent(
        steps=[
            [_tool_call("fetch_fx_rates", {"provider": "ecb"}, "1")],
            [_tool_call("validate_snapshot", {}, "2")],
            [_tool_call("publish_snapshot", {}, "3")],
        ],
        final="已发布 ecb 汇率",
    )

    outcome = _run(run_fx_agent(agent=agent, tools=tools, service=service))

    assert outcome.mode == "llm"
    assert outcome.published is True
    assert outcome.source == "ecb"
    assert outcome.business_date == date(2026, 11, 12)
    assert outcome.currency_count == len(_rates())
    assert outcome.summary == "已发布 ecb 汇率"

    latest = service.load_latest()
    assert latest is not None
    assert latest.rates["THB"] == Decimal("36")
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history" / "2026-11-12.json").exists()


# ---------------------------- 校验门禁 ----------------------------


def test_publish_is_blocked_when_rate_deviation_is_too_large(tmp_path: Path) -> None:
    ecb = _StubProvider("ecb")
    service = _service(tmp_path, ecb=ecb)

    # 先发布一份正常快照，作为偏离比较基准
    baseline = _run(service.fetch_candidate("ecb"))
    service.publish(baseline)

    # 让 LLM 抓到一份 JPY 暴涨 10 倍的坏数据，并故意跳过判断直接发布
    ecb.rates = _rates(JPY=Decimal("10"))
    tools = get_fx_tools(service)
    agent = _FakeAgent(
        steps=[
            [_tool_call("fetch_fx_rates", {"provider": "ecb"}, "1")],
            [_tool_call("validate_snapshot", {}, "2")],
            [_tool_call("publish_snapshot", {}, "3")],
        ],
        final="尝试发布",
    )

    outcome = _run(run_fx_agent(agent=agent, tools=tools, service=service))

    assert outcome.published is False
    # latest 仍是上一份正常快照
    latest = service.load_latest()
    assert latest is not None
    assert latest.rates["JPY"] == Decimal("1")
    # 候选快照里确实是被污染的数据
    candidate = service.load_candidate()
    assert candidate is not None
    assert candidate.rates["JPY"] == Decimal("10")


# ---------------------------- 降级流程 ----------------------------


def test_falls_back_to_deterministic_when_llm_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ecb = _StubProvider("ecb")
    er_api = _StubProvider("er_api")
    ecb.error = ExchangeRateProviderError("ecb 超时")
    service = _service(tmp_path, ecb=ecb, er_api=er_api)

    import backend.graph.fx_workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "create_fx_agent",
        lambda tools=None: (_ for _ in ()).throw(ValueError("没有 API Key")),
    )

    outcome = _run(run_fx_agent(service=service))

    assert outcome.mode == "deterministic"
    assert outcome.published is True
    assert service.load_latest() is not None
    assert service.load_latest().source == "er_api"


def test_keeps_old_snapshot_when_all_sources_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ecb = _StubProvider("ecb")
    er_api = _StubProvider("er_api")
    service = _service(tmp_path, ecb=ecb, er_api=er_api)

    baseline = _run(service.fetch_candidate("ecb"))
    service.publish(baseline)
    published_at = service.load_latest().fetched_at  # type: ignore[union-attr]

    ecb.error = ExchangeRateProviderError("ecb 超时")
    er_api.error = ExchangeRateProviderError("er_api 超时")

    import backend.graph.fx_workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "create_fx_agent",
        lambda tools=None: (_ for _ in ()).throw(ValueError("没有 API Key")),
    )

    outcome = _run(run_fx_agent(service=service))

    assert outcome.mode == "deterministic"
    assert outcome.published is False
    assert "保留旧快照" in outcome.summary
    # 旧快照没有被覆盖
    assert service.load_latest().fetched_at == published_at  # type: ignore[union-attr]
