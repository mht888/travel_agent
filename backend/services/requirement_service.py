"""需求提取业务层（第 4 阶段）。

数据流：
    用户一句话
        ↓
    RequirementExtractor（LLM，失败自动降级到规则）
        ↓  RequirementPatch（只含本次提到的字段）
    merge_draft（合并上一轮草稿）
        ↓
    missing_fields（Python 推导，不交给 LLM 数）
        ↓
    RequirementExtraction（草稿 + 追问 + 警告）
"""

import logging
from datetime import date
from functools import lru_cache

from openai import AsyncOpenAI

from backend.agents.requirement_agent import (
    ExtractionOutcome,
    RequirementAgent,
    RequirementExtractor,
    RuleBasedRequirementExtractor,
)
from backend.config import get_settings
from backend.schemas.common import DataStatus, Money
from backend.schemas.requirement import (
    MissingField,
    RequirementExtraction,
    RequirementPatch,
    TripRequirementDraft,
)

logger = logging.getLogger(__name__)

# 追问措辞属于业务层，不放进 Schema；顺序由 missing_fields 决定，保证稳定
_CLARIFYING_QUESTIONS: dict[MissingField, str] = {
    MissingField.DESTINATION: "你想去哪个城市？",
    MissingField.DURATION_DAYS: "计划玩几天？",
    MissingField.START_DATE: "大概什么时候出发？还没定也可以先跳过。",
    MissingField.BUDGET: "总预算大概多少？请带上币种，例如「1000 美元」。",
}


def merge_draft(
    previous: TripRequirementDraft | None,
    patch: RequirementPatch,
) -> TripRequirementDraft:
    """把增量 Patch 合并到上一轮草稿。

    规则：
    - Patch 字段为 None 表示本次没提到，保留旧值；
    - interests 采用并集，支持“我还喜欢购物”这类追加表达；
    - 其余字段以本次为准，支持“预算改成 1500”。
    """
    base = previous or TripRequirementDraft()

    budget = base.budget
    if patch.budget_amount is not None:
        budget = Money(
            amount=patch.budget_amount,
            # agent 已保证有币种；缺失时会按 USD 并写进 warnings
            currency=patch.budget_currency or "USD",
            status=DataStatus.USER_INPUT,
        )

    interests = base.interests
    if patch.interests:
        interests = _merge_interests(base.interests, patch.interests)

    return TripRequirementDraft(
        destination=patch.destination if patch.destination is not None else base.destination,
        duration_days=(
            patch.duration_days if patch.duration_days is not None else base.duration_days
        ),
        start_date=patch.start_date if patch.start_date is not None else base.start_date,
        travelers=patch.travelers if patch.travelers is not None else base.travelers,
        budget=budget,
        interests=interests,
        travel_style=(
            patch.travel_style if patch.travel_style is not None else base.travel_style
        ),
        notes=patch.notes if patch.notes is not None else base.notes,
    )


def _merge_interests(previous: list[str], new: list[str]) -> list[str]:
    """并集，保持旧值在前、新值在后的稳定顺序。"""
    seen = set(previous)
    merged = list(previous)
    for interest in new:
        if interest not in seen:
            seen.add(interest)
            merged.append(interest)
    return merged


class RequirementService:
    """需求提取服务。构造时注入抽取器，便于测试替换。"""

    def __init__(self, extractor: RequirementExtractor) -> None:
        self._extractor = extractor

    async def extract(
        self,
        message: str,
        previous_draft: TripRequirementDraft | None = None,
        today: date | None = None,
    ) -> RequirementExtraction:
        normalized = message.strip()
        if not normalized:
            raise ValueError("message 不能为空白")

        outcome: ExtractionOutcome = await self._extractor.extract(
            normalized,
            previous=previous_draft,
            today=today or date.today(),
        )
        draft = merge_draft(previous_draft, outcome.patch)

        # 缺失字段与追问由 Python 推导：LLM 只负责抽取，不负责数缺了几个字段
        questions = [_CLARIFYING_QUESTIONS[field] for field in draft.missing_fields]

        return RequirementExtraction(
            draft=draft,
            clarifying_questions=questions,
            warnings=outcome.warnings,
            extracted_by=outcome.source,
        )


@lru_cache(maxsize=1)
def get_requirement_service() -> RequirementService:
    """FastAPI 依赖。

    有 OPENAI_API_KEY 走 LLM；没有则直接使用规则解析。
    注意这里不看 mock_mode：mock_mode 控制的是景点/路线等数据源，
    需求抽取属于 LLM 能力，两者不能混为一谈。
    """
    settings = get_settings()
    fallback = RuleBasedRequirementExtractor()

    if not settings.openai_api_key:
        logger.info("未配置 OPENAI_API_KEY，需求提取使用规则解析")
        return RequirementService(extractor=fallback)

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
    )
    return RequirementService(
        extractor=RequirementAgent(
            client=client,
            model=settings.openai_model,
            fallback=fallback,
        )
    )
