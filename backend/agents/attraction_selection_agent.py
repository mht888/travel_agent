"""景点初选 Agent。

职责非常窄：
    拿到城市 + 兴趣，调用两个工具（检索候选池 → 按兴趣排序），
    最后用一小段文字总结结果。

它不处理：
    距离阈值、营业时间、预算、交通方式、分天安排——这些留给后续的分天排序 Agent。
    越权处理会让「按兴趣选景点」和「按约束排行程」两件事互相污染，无法单独评估。

和 Research Agent 的差别：
    Research Agent 自己决定查什么；这里流程固定为「先搜后排」，
    Agent 的价值在于用自然语言理解兴趣，而不是自由探索工具。
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from backend.config import get_settings
from backend.tools.attraction_selection_tools import (
    create_attraction_selection_tools,
)

logger = logging.getLogger(__name__)


def build_selection_system_prompt(
    city: str,
    interests: list[str],
    notes: str | None = None,
) -> SystemMessage:
    """生成固定的初选提示，明确禁止越权处理排程约束。"""
    interests_text = "、".join(interests) if interests else "未指定"
    notes_text = notes or "无"

    content = f"""你是旅行系统的「景点初选 Agent」。

你的唯一任务：从指定城市的候选景点中，按用户兴趣挑出最值得去的若干个景点。

用户信息：
- 城市：{city}
- 兴趣：{interests_text}
- 补充说明：{notes_text}

可用工具：
- search_attraction_pool：检索城市景点候选池，返回 search_id 和精简候选列表
- rank_attraction_pool：按兴趣对候选池排序，返回 score 与理由

工作流程（按顺序执行，不要跳步）：
1. 调用 search_attraction_pool，city 用「{city}」，默认取 20 个候选。
2. 把上一步返回的 search_id 原样传给 rank_attraction_pool，interests 用用户兴趣。
3. 如果候选池为空，直接说明没有检索到，不要编造景点，也不要重试超过一次。
4. 排序完成后，用 2~3 句话总结结果，不要重复罗列全部字段。

边界（必须遵守）：
- 只按兴趣挑景点，不要判断距离、营业时间、门票、预算，也不要做分天安排。
- 不要调用这两个工具以外的工具。
- 不要编造景点名称或 id，一切以工具返回为准。
- 最终回答只是总结，真正的结构化结果由系统读取工具返回。
"""
    return SystemMessage(content=content)


def create_attraction_selection_agent(
    tools: list[BaseTool] | None = None,
) -> Runnable:
    """创建绑定了初选工具的 LLM Runnable。

    和 Research Agent 一样，Tool Calling 必须有 LLM，因此缺少 Key 时直接抛错，
    由上层 run_attraction_selection 决定是否降级为纯函数流程。
    """
    settings = get_settings()
    tools = tools or create_attraction_selection_tools()

    if not settings.openai_api_key:
        raise ValueError("景点初选 Agent 需要 OPENAI_API_KEY，请在 .env 中配置")

    from langchain_openai import ChatOpenAI

    llm: BaseChatModel = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
    )
    return llm.bind_tools(tools)
