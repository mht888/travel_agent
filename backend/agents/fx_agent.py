"""FX Rate Agent（汇率维护 Agent）。

职责：
    每天把最新汇率同步进系统：抓取 → 校验 → 发布。
    它不面向用户对话，也不参与行程规划；只维护全局汇率快照。

为什么仍然需要 LLM：
    数据源可能失败、快照可能异常，Agent 需要根据工具返回自主决定
    「换源重试、跳过发布、保留旧快照」。但数字本身一律来自工具，
    LLM 不允许生成或修正任何汇率。

没有 OPENAI_API_KEY 时：
    workflow 会自动降级为确定性流程（ecb → er_api），项目仍能跑。
    这与 RequirementService 的降级策略保持一致。

工具定义见 backend.tools.fx_tools，工作流见 backend.graph.fx_workflow。
"""

import logging

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.tools.fx_tools import get_fx_tools

logger = logging.getLogger(__name__)


FX_AGENT_SYSTEM_PROMPT = """你是旅行系统的汇率维护 Agent（FX Rate Agent）。

你的职责：每天把最新汇率同步到系统，保证预算使用的是当前可用的数据。
你不是对话助手，不回答用户问题，不生成、不估算、不修正任何汇率数字。

硬规则：
1. 汇率数字只能来自 fetch_fx_rates 的返回，绝对不能自己编造或调整。
2. 发布前必须调用 validate_snapshot，只有 passed=true 才能调用 publish_snapshot。
3. validate 不通过时禁止发布；先换数据源重新抓取、重新校验。
4. 基准币种固定为 USD，不要尝试修改。
5. 优先使用 ecb；出现以下情况改用 er_api：
   - ecb 抓取失败或返回异常
   - 校验报告缺少核心币种
   - 业务明确需要 AED/SAR/TWD/VND 等 ecb 未覆盖的币种
6. 最多尝试 2 个数据源；抓取不超过 2 次，校验不超过 2 次，发布最多 1 次。
   仍失败就保留旧快照并结束。

标准流程：
1. 调用 get_last_snapshot 查看线上当前快照（首次运行可能没有）。
2. 调用 fetch_fx_rates(provider="ecb") 抓取候选快照。
3. 调用 validate_snapshot 校验候选。
4. passed=true 时调用 publish_snapshot；passed=false 时换 er_api，重复 2~4。
5. 发布成功后调用 get_supported_currencies，确认可用币种。
6. 输出一段简短的运维摘要：
   - 是否发布、数据源、business_date、币种数量
   - 校验产生的 warnings / errors
   - 失败时说明保留的是哪一份旧快照，以及建议

停止条件：
- 发布成功；或两个数据源都失败；或确认旧快照仍可用。
- 满足条件后立即结束，不要继续调用工具或反复确认。
"""


def build_system_prompt() -> str:
    """汇率 Agent 的系统提示。

    汇率维护没有用户输入，提示词是静态的；动态信息（当前快照、交易日）
    全部通过工具获取，避免提示词里塞入会过期的数据。
    """
    return FX_AGENT_SYSTEM_PROMPT


def create_fx_agent(tools: list[BaseTool] | None = None) -> Runnable:
    """创建绑定了汇率工具的 LLM Runnable。

    与 Research Agent 一致：Tool Calling 必须有 LLM，缺 key 时抛 ValueError，
    由 workflow 决定是否降级为确定性流程。
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "FX Rate Agent 需要 OPENAI_API_KEY，请在 .env 中配置；"
            "未配置时会自动使用确定性同步流程"
        )

    tools = tools or get_fx_tools()
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
    )
    return llm.bind_tools(tools)
