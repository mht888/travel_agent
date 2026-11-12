"""FX Rate Agent 的 LangGraph 工作流。

图结构与 Research Agent 一致：agent 与 tools 之间循环，直到 LLM 不再发起 tool_calls。

    START
      │
      ▼
    agent ──(有 tool_calls)──▶ tools
      ▲                          │
      └──────────────────────────┘
      │
      (无 tool_calls)
      ▼
     END

安全边界：
    LLM 只决定「用哪个源、是否重试、要不要发布」。
    publish_snapshot 工具内部会再次校验，LLM 无法绕过。
    没有 OPENAI_API_KEY 或 LLM 调用失败时，降级为确定性流程：
    ecb → er_api 依次尝试，仍然失败则保留旧快照。

调度层（每天 6 点）只需要调用 run_fx_agent()，不依赖图内部细节。
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agents.fx_agent import build_system_prompt, create_fx_agent
from backend.graph.fx_state import FXAgentState
from backend.graph.nodes import create_agent_node
from backend.providers.fx_provider import ExchangeRateProviderError
from backend.services.fx_service import (
    FXService,
    FXStorageError,
    FXValidationError,
    UnknownFXProviderError,
    get_fx_service,
)
from backend.tools.fx_tools import get_fx_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FXAgentOutcome:
    """一次同步的结构化结果，供调度任务记录日志或告警。"""

    summary: str
    published: bool
    mode: Literal["llm", "deterministic"]
    source: str | None = None
    business_date: date | None = None
    currency_count: int = 0


def should_continue(state: FXAgentState) -> Literal["tools", "__end__"]:
    """条件边：最后一条 AI 消息是否还需要调用工具。"""
    messages: list[BaseMessage] = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_fx_graph(
    agent: Runnable | None = None,
    tools: list[Any] | None = None,
):
    """构建并编译汇率 Agent 的 LangGraph。"""
    tools = tools or get_fx_tools()
    agent = agent or create_fx_agent(tools)

    graph = StateGraph(FXAgentState)
    graph.add_node("agent", create_agent_node(agent))
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------- 结果提取 ----------------------------


def _extract_summary(messages: list[BaseMessage]) -> str:
    """取最后一条不带 tool_calls 的 AI 消息作为运维摘要。"""
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or getattr(message, "tool_calls", None):
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = "".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
                if isinstance(block, (str, dict))
            )
            if text.strip():
                return text.strip()
    return ""


def _load_latest_safely(service: FXService):
    """读取当前快照。文件损坏时返回 None，让同步流程继续尝试覆盖它。"""
    try:
        return service.load_latest()
    except FXStorageError:
        logger.exception("读取当前汇率快照失败，按没有历史快照处理")
        return None


def _did_publish(before: Any, after: Any) -> bool:
    if after is None:
        return False
    if before is None:
        return True
    return after.fetched_at > before.fetched_at


# ---------------------------- 降级流程 ----------------------------


async def _run_deterministic(service: FXService) -> FXAgentOutcome:
    """不依赖 LLM 的同步流程：按数据源顺序依次尝试。

    这是 Agent 不可用时的保底路径，不追求覆盖率，只保证：
    有可用数据就发布，没有就保留旧快照并说明原因。
    """
    errors: list[str] = []

    for provider_name in service.available_providers:
        try:
            candidate = await service.fetch_candidate(provider_name)
            report = service.validate(candidate)
            if not report.passed:
                errors.append(
                    f"{provider_name} 校验未通过：{'；'.join(report.errors)}"
                )
                continue

            service.publish(candidate)
            summary = (
                f"确定性流程已发布 {provider_name} 汇率："
                f"business_date={candidate.business_date}，"
                f"{len(candidate.rates)} 种币"
            )
            if report.warnings:
                summary += f"；告警：{'；'.join(report.warnings)}"
            return FXAgentOutcome(
                summary=summary,
                published=True,
                mode="deterministic",
                source=candidate.source,
                business_date=candidate.business_date,
                currency_count=len(candidate.rates),
            )
        except (
            ExchangeRateProviderError,
            FXStorageError,
            FXValidationError,
            UnknownFXProviderError,
        ) as exc:
            errors.append(f"{provider_name}：{exc}")

    latest = _load_latest_safely(service)
    summary = (
        "未发布新汇率，保留旧快照"
        if latest is not None
        else "未发布新汇率，当前没有可用快照"
    )
    if errors:
        summary += f"；失败原因：{' | '.join(errors)}"

    return FXAgentOutcome(
        summary=summary,
        published=False,
        mode="deterministic",
        source=latest.source if latest else None,
        business_date=latest.business_date if latest else None,
        currency_count=len(latest.rates) if latest else 0,
    )


# ---------------------------- 入口 ----------------------------


async def run_fx_agent(
    *,
    agent: Runnable | None = None,
    tools: list[Any] | None = None,
    service: FXService | None = None,
) -> FXAgentOutcome:
    """执行一次汇率同步。

    参数均可注入，便于测试：
        agent   自定义 LLM Runnable（测试可传假 agent）
        tools   自定义工具集
        service 自定义 FXService（隔离数据目录/数据源）

    降级规则：
        - 没有 OPENAI_API_KEY：直接走确定性流程
        - LLM 调用异常：记录日志后走确定性流程
        - Agent 正常结束但没有发布：如实返回 published=False
    """
    service = service or get_fx_service()
    tools = tools or get_fx_tools(service)
    before = _load_latest_safely(service)

    try:
        fx_agent = agent or create_fx_agent(tools)
    except ValueError as exc:
        logger.warning("FX Agent 未启用 LLM：%s；改用确定性流程", exc)
        return await _run_deterministic(service)

    graph = build_fx_graph(agent=fx_agent, tools=tools)
    initial_state: FXAgentState = {
        "messages": [
            SystemMessage(content=build_system_prompt()),
            HumanMessage(content="请开始今天的汇率同步。"),
        ]
    }

    try:
        final_state = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 12},
        )
        summary = _extract_summary(final_state.get("messages", []))
    except Exception as exc:  # noqa: BLE001 - LLM/工具异常都必须降级而不是中断任务
        logger.exception("FX Agent 执行失败，降级为确定性流程")
        fallback = await _run_deterministic(service)
        return FXAgentOutcome(
            summary=(
                f"Agent 执行失败（{type(exc).__name__}），已降级："
                f"{fallback.summary}"
            ),
            published=fallback.published,
            mode=fallback.mode,
            source=fallback.source,
            business_date=fallback.business_date,
            currency_count=fallback.currency_count,
        )

    after = _load_latest_safely(service)
    published = _did_publish(before, after)
    if not summary:
        summary = (
            "本次已发布新汇率" if published else "本次未发布新汇率，保留旧快照"
        )

    return FXAgentOutcome(
        summary=summary,
        published=published,
        mode="llm",
        source=after.source if after else None,
        business_date=after.business_date if after else None,
        currency_count=len(after.rates) if after else 0,
    )


__all__ = [
    "FXAgentOutcome",
    "build_fx_graph",
    "run_fx_agent",
    "should_continue",
]
