"""LangGraph 节点定义（第 6 阶段）。

两个核心节点：
    agent_node：调用 LLM（已 bind_tools），产出 AIMessage（可能带 tool_calls）
    tool_node：执行 AIMessage 里的 tool_calls，产出 ToolMessage

为什么 tool_node 用 langgraph.prebuilt.ToolNode：
    ToolNode 已经处理好了「从最后一条 AI 消息提取 tool_calls → 逐个执行 →
    组装 ToolMessage 列表」的全套逻辑，还包括错误处理和超时。
    第一版不需要自己造轮子；等需要自定义工具执行策略时再替换。
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langgraph.prebuilt import ToolNode

from backend.graph.state import ResearchState

logger = logging.getLogger(__name__)


def create_agent_node(agent: Runnable):
    """创建 agent 节点。

    闭包的原因：LangGraph 的节点签名是 (state) -> state_partial，
    不能直接传 agent。所以用闭包把 agent 包进去，返回一个只接收 state 的函数。

    TODO(你)：完成 agent_node 的实现。

    逻辑：
    1. 从 state 取出 messages 列表。
    2. 调用 agent.ainvoke(messages)，得到 AIMessage。
       （agent 是 bind_tools 后的 Runnable，它会自动把工具定义发给 LLM。）
    3. 返回 {"messages": [ai_message]}。
       注意是列表！LangGraph 的 add_messages reducer 期望收到列表，
       即使只有一条消息也要包成 [ai_message]。

    提示：
        - agent.ainvoke 是异步的，所以这个节点函数必须是 async。
        - 不需要自己处理 tool_calls，LangGraph 的 conditional edge 会判断。
        - 如果 LLM 返回拒绝或空内容，AIMessage 依然是合法的，让上层处理。
    """

    async def agent_node(state: ResearchState) -> dict[str, Any]:
        messages: list[BaseMessage] = state["messages"]

        # TODO: 调用 agent 并返回结果
        # ------------------------------------------------------------------
        # response: AIMessage = await agent.ainvoke(messages)
        # return {"messages": [response]}
        # ------------------------------------------------------------------
        response: AIMessage = await agent.ainvoke(messages)
        return {"messages": [response]}


    return agent_node


def create_tool_node(tools: list[Any]) -> ToolNode:
    """创建工具执行节点。

    这里直接返回 langgraph.prebuilt.ToolNode，它的行为：
    - 输入：state（最后一条消息必须是带 tool_calls 的 AIMessage）
    - 输出：{"messages": [ToolMessage, ...]}
    - 自动处理：工具不存在、参数校验失败、工具抛异常等情况

    不需要写 TODO，因为 ToolNode 是成熟组件。
    但你需要理解它的输入输出契约，否则 conditional edge 接不上。
    """
    return ToolNode(tools)
