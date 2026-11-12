"""需求提取的多轮对话测试。

多轮的本质：需求接口是**无状态**的。
服务不会自己记住上一轮，必须由调用方（前端 / 测试）把上一轮的
`extraction.draft` 作为 `previous_draft` 回传，服务内部再用 `merge_draft` 做增量合并。

合并规则（决定了下面这些用例该怎么断言）：
    - patch 字段为 None = 本轮没提到 → 保留旧值
    - interests 取并集，支持「我还喜欢购物」
    - 其余字段以本轮为准，支持「预算改成 1500」「改成 4 天」
    - 只有 destination 和 duration_days 是阻塞字段；
      start_date / budget 缺失只追问，不阻止规划

运行（在 travel_agent/ 目录下）：

    # 默认：规则抽取器，稳定、零成本
    .venv\\Scripts\\python.exe -m pytest backend/tests/test_requirement_multiturn.py -v

    # 切到 LLM：需要 .env 里配置 OPENAI_API_KEY
    $env:REQUIREMENT_TEST_MODE="llm"
    .venv\\Scripts\\python.exe -m pytest backend/tests/test_requirement_multiturn.py -v
"""

import asyncio
import os

import pytest

from backend.agents.requirement_agent import RuleBasedRequirementExtractor
from backend.config import get_settings
from backend.schemas.requirement import MissingField
from backend.services.requirement_service import (
    RequirementService,
    get_requirement_service,
)

# 只有显式要求 LLM 模式时才切换，避免日常测试消耗额度和受网络影响
_USE_LLM = os.getenv("REQUIREMENT_TEST_MODE", "rules").strip().lower() == "llm"

pytestmark = pytest.mark.skipif(
    _USE_LLM and not get_settings().openai_api_key,
    reason="REQUIREMENT_TEST_MODE=llm，但未配置 OPENAI_API_KEY",
)


def _service() -> RequirementService:
    """默认注入规则抽取器；REQUIREMENT_TEST_MODE=llm 时走真实的 LLM 抽取器。"""
    if _USE_LLM:
        return get_requirement_service()
    return RequirementService(extractor=RuleBasedRequirementExtractor())


def _turn(message: str, previous_draft=None):
    result = asyncio.run(
        _service().extract(message, previous_draft=previous_draft)
    )
    if _USE_LLM:
        # LLM 调用失败时 RequirementService 会自动降级到规则，
        # 如果不在这里卡住，测试会因为「规则兜底」而假通过。
        assert result.extracted_by == "llm", (
            f"LLM 未生效，已降级为 rules。warnings={result.warnings}"
        )
    return result


def test_three_turns_from_incomplete_to_ready() -> None:
    """第 1 轮缺天数 → 第 2 轮补天数+预算（已可规划）→ 第 3 轮补日期。"""
    turn1 = _turn("我想去东京玩几天")
    assert turn1.is_ready is False
    assert turn1.draft.destination == "Tokyo"
    assert MissingField.DURATION_DAYS in turn1.draft.missing_fields
    assert turn1.clarifying_questions

    turn2 = _turn("计划 5 天，预算 1000 美元", previous_draft=turn1.draft)
    # 第 1 轮的目的地被保留，本轮只补了新字段
    assert turn2.draft.destination == "Tokyo"
    assert turn2.draft.duration_days == 5
    assert turn2.draft.budget is not None
    assert turn2.draft.budget.amount == 1000
    # 日期缺失只追问、不阻塞
    assert MissingField.START_DATE in turn2.draft.missing_fields
    assert turn2.is_ready is True

    turn3 = _turn("12月10日出发", previous_draft=turn2.draft)
    assert turn3.draft.destination == "Tokyo"
    assert turn3.draft.duration_days == 5
    assert turn3.draft.start_date is not None
    assert turn3.draft.missing_fields == []
    assert turn3.is_ready is True

    requirements = turn3.to_requirements()
    assert requirements.destination == "Tokyo"
    assert requirements.duration_days == 5

def test_money() -> None:
    """美元预算被第二轮的人民币预算覆盖，且币种跟随变化。"""
    turn1 = _turn("我想去东京玩几天，预算 1000 美元")
    assert turn1.is_ready is False
    assert turn1.draft.destination == "Tokyo"
    assert MissingField.DURATION_DAYS in turn1.draft.missing_fields
    assert turn1.clarifying_questions
    turn2 = _turn("计划 5 天，预算 1000 日元", previous_draft=turn1.draft)
    # start_date 是非阻塞字段：可以追问，但不影响「能否开始规划」
    assert turn2.is_ready is True
    assert turn2.draft.destination == "Tokyo"
    assert turn2.draft.duration_days == 5
    assert turn2.draft.budget is not None
    assert turn2.draft.budget.amount == 1000
    assert turn2.draft.budget.currency == "JPY"
    assert MissingField.START_DATE in turn2.draft.missing_fields
    print(turn2.draft)
    assert turn2.is_ready is True
def test_interest_is_appended_not_replaced() -> None:
    """「我还喜欢购物」应做并集，不能把上一轮的兴趣覆盖掉。"""
    turn1 = _turn("去东京 5 天，我喜欢美食和文化")
    assert set(turn1.draft.interests) == {"food", "culture"}

    turn2 = _turn("我还喜欢购物", previous_draft=turn1.draft)
    assert set(turn2.draft.interests) == {"food", "culture", "shopping"}


def test_new_value_overrides_but_keeps_other_fields() -> None:
    """「改成 4 天」只改天数，目的地、预算、兴趣都不能丢。"""
    turn1 = _turn("去东京 5 天，预算 1000 美元，喜欢美食")
    turn2 = _turn("改成 4 天", previous_draft=turn1.draft)

    assert turn2.draft.duration_days == 4
    assert turn2.draft.destination == "Tokyo"
    assert turn2.draft.budget is not None
    assert turn2.draft.budget.amount == 1000
    assert turn2.draft.interests == ["food"]


def test_service_is_stateless_so_context_must_be_passed_explicitly() -> None:
    """不传 previous_draft 时，服务不会「记得」上一轮。

    这不是 bug，而是设计：多轮状态由调用方持有（将来是 LangGraph Checkpointer）。
    如果这里意外通过了，说明服务偷偷缓存了状态，反而要警惕。
    """
    _turn("去东京 5 天")
    isolated = _turn("改成 4 天", previous_draft=None)

    assert isolated.draft.destination is None
    assert isolated.draft.duration_days == 4
    assert isolated.is_ready is False
    assert MissingField.DESTINATION in isolated.draft.blocking_missing_fields
