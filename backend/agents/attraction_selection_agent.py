"""景点初选 Agent。

职责非常窄：
    拿到城市 + 兴趣，先按兴趣做网页检索，再从网页里抽地点并按兴趣排序。
    排序本身由 rank_attraction_pool 工具完成，Agent 只负责编排这几步。

工具顺序固定：
    mcp_web_search（每个兴趣一条 query，不足时按提示放大 count 补检索）
    → extract_web_places（纯 Python 统计每个兴趣数量，不足 10 个返回
      insufficient_places，Agent 补检索后再抽取，抽取会话累计去重）
    → rank_attraction_pool（凭 search_id 取回并打分）

    网页检索只使用「城市 + 兴趣 + 景点 推荐」这一种写法，不再扩写其他 query。
    博查不可用时直接返回空结果，不降级到百度候选池。

它不处理：
    距离阈值、营业时间、预算、交通方式、分天安排——这些留给后续的分天排序 Agent。
"""

import logging
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from backend.config import get_settings
from backend.tools.registry import get_profile_tools, profile_specs, render_tool_block

logger = logging.getLogger(__name__)


def build_selection_system_prompt(
    city: str,
    interests: list[str],
    notes: str | None = None,
    tools: Sequence[Any] | None = None,
) -> SystemMessage:
    """生成固定的初选提示，明确只允许调用检索与排序两个工具。

    「可用工具」由 tools/registry 的 attraction_selection 画像渲染，
    工具增删只需改注册表，不用改这段提示词。
    """
    interests_text = "、".join(interests) if interests else "未指定"
    notes_text = notes or "无"
    tool_block = render_tool_block(
        tools if tools is not None else profile_specs("attraction_selection")
    )
    query_examples = "\n".join(
        f"  - {city} {interest} 景点 推荐" for interest in interests
    )
    web_search_rule = ""
    extract_rule = ""
    if "mcp_web_search" in tool_block:
        web_search_rule = f"""- 网页检索：每个兴趣用 mcp_web_search 检索，query 固定为「城市 兴趣 景点 推荐」，不得添加其他关键词。首次 count 用 10。例如：
{query_examples}
"""
    if "extract_web_places" in tool_block:
        extract_rule = (
            "- 地点抽取：调用 extract_web_places，把 city 和 interests 传进去；"
            "不要传 web_search_ids，工具会使用本次会话的全部网页结果"
            "（已消费的结果不会重复抽取）。\n"
            "- 数量校验：工具会先把同义/别名地点归并（如「中山路美食街」与「中山路夜市」），"
            "再在 Python 侧统计每个兴趣的地点数。"
            "若返回 error_code=insufficient_places，先不要排序；"
            "按返回的 shortages 对不足的兴趣再次调用 mcp_web_search"
            "（query 不变，count 用返回的 next_count 或更大，最多 50），"
            "然后重新调用 extract_web_places；已抽到的地点会保留。\n"
            "- 重试次数以返回的 rounds_left 为准；rounds_left=0 且仍不足时，"
            "工具会带 warnings 返回 ok=true，此时可以继续排序。\n"
            "- 排序：调用 rank_attraction_pool，传入 search_id 和 interests；"
            "工具会按每个兴趣分别排序，各返回前 7 个（by_interest）。\n"
        )
    else:
        extract_rule = "- 排序：调用 rank_attraction_pool，并传入可用的 search_id。\n"

    content = f"""你是旅行系统的「景点初选 Agent」。

你的唯一任务：为「{city}」按用户兴趣筛出候选景点，并给出排序。

用户信息：
- 城市：{city}
- 兴趣：{interests_text}
- 补充说明：{notes_text}

可用工具：
{tool_block}

调用规则（必须遵守）：
{web_search_rule}{extract_rule}- 只调用上面列出的工具，不要调用任何其他工具。
- search_id 只能使用工具返回的值，不要自己编造；候选池过期时重新检索候选。
- 不要自己编造景点名称，也不要在回答里输出排名或分数；排序结果由工具返回。
- 不判断距离阈值、营业时间、门票、预算和分天安排，这些留给后续的分天排序 Agent。
- 排序完成后，用一句话说明覆盖了哪些兴趣即可。
"""
    return SystemMessage(content=content)


def create_attraction_selection_agent(
    tools: list[BaseTool] | None = None,
) -> Runnable:
    """创建绑定「网页检索 + 地点抽取 + 排序」三个工具的 LLM Runnable。

    缺少 OPENAI_API_KEY 或工具依赖（博查）时直接抛错，
    由上层 run_attraction_selection 转成空结果。
    """
    settings = get_settings()
    tools = tools or get_profile_tools("attraction_selection")

    if not tools:
        raise ValueError(
            "景点初选 Agent 需要 mcp_web_search / extract_web_places / "
            "rank_attraction_pool，请配置博查与 OPENAI_API_KEY"
        )
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
