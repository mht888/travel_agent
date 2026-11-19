"""酒店 Agent：从入选景点出发，先判断住宿区域，再搜索酒店。

和景点初选 Agent 的分工：
    初选 Agent 决定「去哪玩」；酒店 Agent 只决定「住哪」，
    不重新推荐景点，也不评价景点好不好。

提示词按项目里真实可用的工具改写，不保留模板中不存在的
Business Search / Directions 等抽象工具名：
    gaode_mcp             高德 MCP 统一工具（action=around_search, poi_kind=hotel；
                          中心点必须是真实坐标；也提供详情/地理编码/天气/路线）
    mcp_web_search        查区域背景、商圈、住宿环境

关键约束写进提示词：酒店字段只能来自工具返回；
高德的人均价是参考价，不是实时房价；没有房态数据就明确说没有。
"""

import json
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from backend.config import get_settings
from backend.tools.registry import (
    get_profile_tools,
    profile_specs,
    render_tool_block,
)


def build_hotel_system_prompt(
    city: str,
    anchors: Sequence[dict[str, Any]],
    *,
    radius_meters: int = 2000,
    tools: Sequence[Any] | None = None,
) -> SystemMessage:
    """生成酒店 Agent 的固定提示。

    anchors 是压缩后的入选景点，直接以 JSON 展示，
    避免模型再去检索景点或凭记忆补坐标。
    """
    anchor_json = json.dumps(list(anchors), ensure_ascii=False, indent=2)
    tool_block = render_tool_block(
        tools if tools is not None else profile_specs("hotel_search")
    )
    has_gaode = "gaode_mcp" in tool_block
    has_web = "mcp_web_search" in tool_block

    geocode_rule = (
        "- 新区域验证：提出输入里没有的区域时，先用 gaode_mcp（action=geocode）把"
        "「城市 + 区域名」转成坐标（如「南宁市 青秀区 朝阳广场」）。"
        "将返回的 location（经度,纬度）拆成 longitude / latitude；"
        "geocode 失败的区域直接放弃，不要猜坐标。\n"
        if has_gaode
        else "- 本次没有 geocode 工具：只能围绕入选景点坐标搜索，"
        "并在结论里说明「未验证其他区域」。\n"
    )
    web_rule = (
        "- 区域背景：需要验证商圈、滨水区、历史街区等说法时用 mcp_web_search，"
        "query 带上城市与区域名；网页结论只能作为定性依据，不能当酒店事实。\n"
        if has_web
        else ""
    )
    route_rule = (
        "- 交通便利度：使用 gaode_mcp（action=route），用「经度,纬度」传入起终点。"
        "route_mode 支持 driving / walking / bicycling / transit，公交路线还需传城市与目的城市。"
        "没有公交/地铁数据时，写「公交耗时未验证」，不要估算。\n"
        if has_gaode
        else "- 本次没有路线工具：不要输出任何交通耗时，只描述直线距离关系。\n"
    )

    content = f"""你是旅行系统的「酒店区域与酒店搜索 Agent」。

输入已经完成景点初选与排序，你不需要重新推荐景点。
城市：{city}
建议搜索半径：{radius_meters} 米
入选景点（GCJ-02 坐标，可直接用于距离判断）：
{anchor_json}

本轮没有入住日期、人数、实时房态和实时房价数据；
不要假设具体日期，不要把参考价当成实时报价。

可用工具：
{tool_block}

你的任务不是「每个景点附近搜一遍酒店」，而是：
从这些景点出发，先判断合理的住宿区域，再在区域中心找酒店。

Step 1 地理关系判断
- 直接用上面的坐标比较距离，不要重新检索景点。
- 找出：景点集中区、离主城区较远的单点（一日游型）、能覆盖多个景点的中心位置。
- 行政区相同不等于距离近，直线距离不等于通勤时间。

Step 2 生成 2-3 个候选住宿区域
- 至少包含两种类型：
  A 覆盖型：能方便到达多个入选景点；
  B 体验/配套型：商圈、滨水、老城等住宿环境更好的区域。
- 每个区域写清：区域名、由哪些景点或坐标关系推导、推荐理由、潜在缺点。
- 可以提出输入中没有的区域，但必须能解释它和入选景点的关系。

Step 3 验证候选区域
{geocode_rule}{web_rule}{route_rule}- 未经工具验证的关系要标成「推测」，不要写成事实。

Step 4 搜索酒店
- 每个候选区域中心调用一次 gaode_mcp：
  action=around_search，poi_kind=hotel，longitude / latitude 用区域中心，
  radius 用 1500-3000，limit 用 10-25（默认按距离升序返回）。
- keywords 只在确有偏好时填写（如 民宿 / 经济型 / 五星级），否则留空（默认搜「酒店」）。
- 至少覆盖 2 个区域；总搜索次数不超过 6 次；不要为每个景点各搜一遍。
- 工具返回 error_code 时最多重试 1 次；仍失败就降低该区域置信度并说明原因。
- 不要重复调用「同一中心 + 同一参数」的搜索。

Step 5 区域比较与收口
- 比较：到主要景点的距离/耗时、覆盖景点数量、商圈配套、酒店评分分布、参考价。
- 缺少某项数据就写「未验证」，禁止虚构评分、价格、房态、交通时间。
- 工具只展示少量样例和候选总数，完整酒店由系统保存。
- 最终只用 1–3 句列出搜过的住宿区域与尚未核实的事项，合计不超过 120 字。
- 酒店明细与最终排序由系统生成，省略逐家推荐、长篇分析和 JSON。
"""
    return SystemMessage(content=content)


def create_hotel_agent(
    tools: list[BaseTool] | None = None,
) -> Runnable:
    """创建绑定酒店工具集的 LLM Runnable。

    缺高德 AK 或 OPENAI_API_KEY 时直接抛错，
    由 run_hotel_search 转成 unavailable，而不是让图跑到一半失败。
    """
    settings = get_settings()
    tools = tools or get_profile_tools("hotel_search")

    if not tools:
        raise ValueError(
            "酒店 Agent 需要 gaode_mcp，请配置 GAODE_MAPS_AK"
        )
    if not any(tool.name == "gaode_mcp" for tool in tools):
        raise ValueError("酒店 Agent 缺少 gaode_mcp 工具")
    if not settings.openai_api_key:
        raise ValueError("酒店 Agent 需要 OPENAI_API_KEY，请在 .env 中配置")

    from langchain_openai import ChatOpenAI

    llm: BaseChatModel = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
    )
    return llm.bind_tools(tools)
