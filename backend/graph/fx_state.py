"""FX Rate Agent 的 LangGraph 状态。

与 ResearchState 的差异：
    Research Agent 把「调研摘要」写进 state 供后续规划使用；
    汇率 Agent 的最终产物已经落盘（latest.json），state 只保留循环所需的消息，
    运维摘要直接由 run_fx_agent 从最后一条 AI 消息提取后返回。

messages 必须挂 add_messages reducer：
    Agent 与 ToolNode 之间是循环，工具结果要追加而不是覆盖历史。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class FXAgentState(TypedDict, total=False):
    """汇率 Agent 的循环状态。"""

    # 对话历史：系统提示 + AI 消息 + Tool 消息
    messages: Annotated[list[BaseMessage], add_messages]
