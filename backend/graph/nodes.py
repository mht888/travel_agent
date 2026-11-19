"""Tool Calling 工作流共用的模型调用节点。"""

from typing import Any, Mapping

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable


def create_agent_node(agent: Runnable):
    """将模型调用包装为 LangGraph 节点，返回待追加的消息。"""

    async def agent_node(state: Mapping[str, list[BaseMessage]]) -> dict[str, Any]:
        response: AIMessage = await agent.ainvoke(state["messages"])
        return {"messages": [response]}

    return agent_node
