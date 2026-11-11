"""Research Agent（第 6 阶段）。

职责：
    拿到旅行需求后，自主决定调用哪些工具（景点搜索、路线矩阵、天气），
    收集足够信息后输出一段调研摘要，供后续 Itinerary Planner 使用。

它本身不排行程、不算预算，只做「情报收集」。
真正的规划能力在第 7~11 阶段逐步叠加到外层 Workflow。

关键概念：
    llm.bind_tools(tools)
        把工具定义（name / args_schema / description）注入到 LLM 的 function calling 协议里。
        调用后的 AIMessage 会带 tool_calls 字段，LangGraph 据此判断要不要进 ToolNode。

    为什么用 Runnable 而不是直接调 LLM：
        bind_tools 返回的是一个 Runnable，agent_node 里直接 .ainvoke(messages) 即可。
        这样节点代码不需要知道「用的是哪个模型、有没有绑工具」。
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

from backend.config import get_settings
from backend.schemas.trip import TripRequirements
from backend.tools.travel_tools import get_travel_tools

logger = logging.getLogger(__name__)


def build_system_prompt(requirements: TripRequirements) -> SystemMessage:
    """根据旅行需求生成系统提示。

    提示要点：
    1. 明确告诉 agent 它的角色是「情报收集者」，不是规划者。
    2. 把用户的目的地、天数、兴趣、节奏等结构化信息塞进去，避免它反复问。
    3. 给一个「什么时候该停」的标准：信息足够输出摘要时就停，不要无限调工具。
    """
    interests_text = "、".join(requirements.interests) if requirements.interests else "未指定"
    budget_text = (
        f"{requirements.budget.amount} {requirements.budget.currency}"
        if requirements.budget
        else "未指定"
    )

    content = f"""你是旅行规划系统的情报收集 Agent（Research Agent）。

你的任务：根据用户的旅行需求，调用工具收集景点、路线、天气等信息，最后输出一段结构化的调研摘要。

用户需求：
- 目的地：{requirements.destination}
- 天数：{requirements.duration_days} 天
- 兴趣：{interests_text}
- 节奏：{requirements.travel_style.value}
- 预算：{budget_text}
- 人数：{requirements.travelers} 人

可用工具：
- search_attractions：搜索景点
- get_attraction_details：获取景点详情（开放时间、游玩时长、票价）
- get_route_matrix：计算景点间交通时间矩阵
- get_weather：查询目的地天气

工作流程：
1. 先调用 search_attractions 获取候选景点列表。
2. 对感兴趣的景点调用 get_attraction_details 确认开放时间和游玩时长。
3. 如果景点较多，调用 get_route_matrix 看看哪些景点距离近，可以放在同一天。
4. 如果用户给了出发日期，调用 get_weather 看天气，把户外景点安排在晴天。
5. 信息收集充分后，输出一段调研摘要，不要继续调用工具。

输出摘要时请包含：
- 推荐的景点清单（带简短理由）
- 景点之间的交通概况
- 天气对行程的影响（如果查了）
- 任何需要 Planner 注意的约束（如周一闭馆、游玩时长较长等）
"""
    return SystemMessage(content=content)


def create_research_agent(tools: list[BaseTool] | None = None) -> Runnable:
    """创建一个绑定了工具的 LLM Runnable。

    TODO(你)：完成下面三处关键代码。

    1. 从 get_settings() 读取 openai_api_key / openai_base_url / openai_model。
       用 langchain_openai.ChatOpenAI 创建实例。
       注意：api_key 为 None 时应该报错或返回一个明确的提示，因为 Research Agent
       不像 Requirement Agent 那样有规则降级——Tool Calling 必须有 LLM。

    2. 设置 temperature=0（情报收集要确定性，不要随机发挥）。

    3. 调用 llm.bind_tools(tools) 把工具绑上去，返回这个 Runnable。

    提示：
        - ChatOpenAI 的参数名是 api_key / base_url / model，和 openai SDK 一致。
        - bind_tools 返回的是 Runnable，不是新的 LLM 实例。
        - tools 为 None 时用 get_travel_tools() 获取默认工具集。
    """
    settings = get_settings()
    tools = tools or get_travel_tools()

    # TODO: 1. 校验 openai_api_key 是否存在，不存在则抛出 ValueError（提示需要配置 API Key）
    # ------------------------------------------------------------------
    # if not settings.openai_api_key:
    #     raise ValueError("Research Agent 需要 OPENAI_API_KEY，请在 .env 中配置")
    # ------------------------------------------------------------------
    if not settings.openai_api_key:
        raise ValueError("Research Agent 需要 OPENAI_API_KEY，请在 .env 中配置")



    # TODO: 2. 创建 ChatOpenAI 实例
    # ------------------------------------------------------------------
    # from langchain_openai import ChatOpenAI
    # llm: BaseChatModel = ChatOpenAI(
    #     api_key=...,
    #     base_url=...,
    #     model=...,
    #     temperature=0.0,
    # )
    # ------------------------------------------------------------------
    # 下面这行是占位，让模块能 import 通过；你需要用上面的真实 llm 替换
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
    )

    # TODO: 3. 绑定工具并返回
    # ------------------------------------------------------------------
    # return llm.bind_tools(tools)
    # ------------------------------------------------------------------
    return llm.bind_tools(tools)

