"""LangGraph 状态定义（第 6 阶段）。

为什么用 TypedDict 而不是 Pydantic：
    LangGraph 的 State 默认就是 TypedDict，字段上可以挂 reducer（如 add_messages）。
    Pydantic 模型虽然也能用，但 reducer 的写法更绕，第一版先用 TypedDict 跑通。

messages 字段的 reducer 是关键：
    - 没有 reducer：每次节点返回 {"messages": [...]} 会整体覆盖
    - add_messages：LangGraph 会把新消息追加到列表末尾，保留完整对话历史
    这个行为正好对应 Tool Calling Loop：agent 产出 AI 消息 → tools 产出 Tool 消息 →
    下一轮 agent 看到的是历史消息 + 工具结果。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    """Research Agent 的循环状态。

    total=False 表示所有字段都是可选的：初始状态只传 messages 即可，
    research_summary 在 agent 决定结束时才填充。
    """

    # 对话历史：AI 消息 + 工具消息。add_messages 保证每次只追加不覆盖。
    messages: Annotated[list[BaseMessage], add_messages]

    # 最终调研摘要。agent 不再调用工具时，把结论写进这里，workflow 据此返回。
    research_summary: str | None
