"""Research Agent 的 LangGraph 工作流（第 6 阶段）。

这是 Tool Calling Loop 的完整编排：

    START
      │
      ▼
    agent ──(有 tool_calls)──▶ tools
      ▲                          │
      │                          │
      └──────────────────────────┘
      │
      (无 tool_calls)
      ▼
     END

关键：
    - agent 和 tools 之间是「环」，不是单向边。
    - 条件判断 should_continue 决定走 tools 还是 END。
    - 整个循环的终止条件是「LLM 不再发起 tool_calls」。

这个文件是第 6 阶段的核心，有 4 处 TODO 需要你完成。
"""

import logging
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agents.research_agent import build_system_prompt, create_research_agent
from backend.graph.nodes import create_agent_node
from backend.graph.state import ResearchState
from backend.schemas.trip import TripRequirements
from backend.tools.travel_tools import get_travel_tools

logger = logging.getLogger(__name__)


def should_continue(state: ResearchState) -> Literal["tools", "__end__"]:
    """条件边：判断 agent 是否还需要调用工具。

    TODO(你)：实现这个函数。

    逻辑：
    1. 取 state["messages"] 的最后一条消息。
    2. 如果它是 AIMessage 且带有 tool_calls（非空列表），返回 "tools"。
    3. 否则返回 END（即 "__end__"）。

    提示：
        - AIMessage 有 .tool_calls 属性，类型是 list[dict]，没有时是空列表。
        - 用 getattr(last_message, "tool_calls", None) 更安全，
          因为最后一条消息可能不是 AIMessage（虽然正常流程里应该是）。
        - END 是 langgraph.graph 导出的常量，值就是 "__end__"。
    """
    messages: list[BaseMessage] = state["messages"]
    # TODO: 实现判断逻辑
    # ------------------------------------------------------------------
    # last_message = messages[-1]
    # if getattr(last_message, "tool_calls", None):
    #     return "tools"
    # return END
    # ------------------------------------------------------------------
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_research_graph(
    agent: Runnable | None = None,
    tools: list[Any] | None = None,
):
    """构建 Research Agent 的 LangGraph。

    TODO(你)：完成图的构建。

    步骤：
    1. 如果 agent 为 None，调用 create_research_agent(tools) 创建。
    2. 如果 tools 为 None，调用 get_travel_tools() 获取。
    3. 创建 StateGraph(ResearchState)。
    4. 添加节点 "agent" = create_agent_node(agent)
    5. 添加节点 "tools" = create_tool_node(tools) —— 注意返回的是 ToolNode 实例
    6. 添加边 START → "agent"
    7. 添加条件边 "agent" → should_continue，映射 {"tools": "tools", END: END}
    8. 添加边 "tools" → "agent"（形成循环）
    9. 调用 graph.compile() 返回编译后的图

    提示：
        - StateGraph 的泛型参数是 State 的 TypedDict。
        - add_conditional_edges 的第三个参数是映射 dict，key 是 should_continue 的返回值。
        - compile() 返回的是 CompiledGraph，可以直接 .ainvoke()。
    """
    tools = tools or get_travel_tools()
    agent = agent or create_research_agent(tools)

    # TODO: 构建并编译图
    # ------------------------------------------------------------------
    # graph = StateGraph(ResearchState)
    # graph.add_node("agent", create_agent_node(agent))
    # graph.add_node("tools", ToolNode(tools))
    # graph.add_edge(START, "agent")
    # graph.add_conditional_edges(
    #     "agent",
    #     should_continue,
    #     {"tools": "tools", END: END},
    # )
    # graph.add_edge("tools", "agent")
    # return graph.compile()
    # ------------------------------------------------------------------
    graph = StateGraph(ResearchState)
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


async def run_research(
    requirements: TripRequirements,
    *,
    agent: Runnable | None = None,
    tools: list[Any] | None = None,
) -> str:
    """运行一次完整的调研循环，返回调研摘要。

    TODO(你)：实现这个入口函数。

    步骤：
    1. 调用 build_research_graph(agent, tools) 得到编译后的图。
    2. 构造初始 state：messages = [build_system_prompt(requirements), HumanMessage(content="请开始调研")]
    3. 调用 graph.ainvoke(state) 得到最终 state。
    4. 从最终 state 的 messages 里取最后一条 AIMessage 的 content 作为摘要返回。
       （循环结束时，最后一条消息一定是不带 tool_calls 的 AIMessage，content 就是摘要。）

    提示：
        - graph.ainvoke 是异步的。
        - 初始消息要包含系统提示，否则 agent 不知道自己该干什么。
        - HumanMessage 的 content 可以是一句简单的触发语，比如 "请开始调研"。
        - 取最后一条消息时，确认它是 AIMessage 类型再取 .content。
    """
    # TODO: 实现运行逻辑
    # ------------------------------------------------------------------
    # graph = build_research_graph(agent=agent, tools=tools)
    # initial_state: ResearchState = {
    #     "messages": [
    #         build_system_prompt(requirements),
    #         HumanMessage(content="请根据需求开始调研"),
    #     ],
    # }
    # final_state = await graph.ainvoke(initial_state)
    # last_message = final_state["messages"][-1]
    # return last_message.content
    # ------------------------------------------------------------------
    graph = build_research_graph(agent=agent, tools=tools)
    initial_state: ResearchState = {
        "messages": [
            build_system_prompt(requirements),
            HumanMessage(content="请根据需求开始调研"),
        ],
    }
    final_state = await graph.ainvoke(initial_state)
    last_message = final_state["messages"][-1]
    return last_message.content
    
