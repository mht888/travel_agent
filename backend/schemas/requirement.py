"""需求提取结果（第 4 阶段的输出契约）。

为什么需要「草稿」这一层：
    TripRequirements 是完整、严格的，Planner 可以直接用；
    但用户第一句话经常缺日期、缺预算、甚至缺天数。
    如果直接用 TripRequirements，缺字段就只能抛异常，追问流程无从谈起。

    TripRequirementDraft 允许缺字段；
    RequirementExtraction 在草稿外面加上「缺什么」和「该问什么」；
    字段补齐后调用 to_requirements() 才允许进入 Planner。
"""

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from backend.schemas.common import ALLOWED_CURRENCIES, Money
from backend.schemas.trip import TravelStyle, TripRequirements


class MissingField(str, Enum):
    """用户尚未提供的信息。这里只列「影响规划」的字段，不含 travelers 这类有默认值的字段。"""

    DESTINATION = "destination"
    DURATION_DAYS = "duration_days"
    START_DATE = "start_date"
    BUDGET = "budget"


class TripRequirementDraft(BaseModel):
    """不完整的需求草稿。字段全部可空，但填了就必须合法（天数不能是 0，币种不能乱写）。"""

    destination: str | None = None
    duration_days: int | None = Field(None, ge=1, le=30)
    start_date: date | None = None
    travelers: int = Field(1, ge=1, le=20)
    budget: Money | None = None
    interests: list[str] = Field(default_factory=list)
    travel_style: TravelStyle = TravelStyle.BALANCED
    notes: str | None = None

    @field_validator("destination")
    @classmethod
    def _normalize_destination(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("destination 不能为空白，缺失请传 None")
        return normalized

    @field_validator("interests")
    @classmethod
    def _normalize_interests(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for interest in value:
            clean = interest.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized

    @property
    def missing_fields(self) -> list[MissingField]:
        """按固定顺序返回缺失字段，保证输出稳定、便于测试和前端渲染。"""
        missing: list[MissingField] = []
        if self.destination is None:
            missing.append(MissingField.DESTINATION)
        if self.duration_days is None:
            missing.append(MissingField.DURATION_DAYS)
        if self.start_date is None:
            missing.append(MissingField.START_DATE)
        if self.budget is None:
            missing.append(MissingField.BUDGET)
        return missing

    @property
    def blocking_missing_fields(self) -> list[MissingField]:
        """真正卡住规划的字段。日期和预算缺失可以按默认策略先规划。"""
        blocking: list[MissingField] = []
        if self.destination is None:
            blocking.append(MissingField.DESTINATION)
        if self.duration_days is None:
            blocking.append(MissingField.DURATION_DAYS)
        return blocking

    @property
    def is_ready(self) -> bool:
        return not self.blocking_missing_fields

    def to_requirements(self) -> TripRequirements:
        """转成严格需求。缺阻塞字段时抛 ValueError，把缺什么直接告诉调用方。"""
        if self.destination is None or self.duration_days is None:
            fields = ", ".join(field.value for field in self.blocking_missing_fields)
            raise ValueError(f"需求尚不完整，缺少必填字段：{fields}")

        return TripRequirements(
            destination=self.destination,
            duration_days=self.duration_days,
            start_date=self.start_date,
            travelers=self.travelers,
            budget=self.budget,
            interests=self.interests,
            travel_style=self.travel_style,
            notes=self.notes,
        )


class RequirementPatch(BaseModel):
    """一次抽取出的「增量」，而不是完整需求。

    为什么需要增量：
        多轮对话里用户说「预算改成 1500」，这一句只包含 budget。
        如果抽取结果直接当成完整 TripRequirementDraft，destination / duration 会被默认值覆盖。
        所以用 None 表示「本次没提到」，由 Service 合并到上一轮草稿上。

    interests 单独约定：None = 没提到；[] = 明确说了没有偏好（目前不做特殊处理，等同于没提到）。
    """

    destination: str | None = None
    duration_days: int | None = Field(None, ge=1, le=30)
    start_date: date | None = None
    travelers: int | None = Field(None, ge=1, le=20)
    budget_amount: Decimal | None = Field(None, gt=0)
    budget_currency: str | None = None
    interests: list[str] | None = None
    travel_style: TravelStyle | None = None
    notes: str | None = None

    @field_validator("destination")
    @classmethod
    def _normalize_destination(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("destination 不能为空白，缺失请传 None")
        return normalized

    @field_validator("budget_currency")
    @classmethod
    def _normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized not in ALLOWED_CURRENCIES:
            allowed = ", ".join(sorted(ALLOWED_CURRENCIES))
            raise ValueError(f"暂不支持币种 {normalized}，当前允许：{allowed}")
        return normalized

    @field_validator("interests")
    @classmethod
    def _normalize_interests(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        seen: set[str] = set()
        normalized: list[str] = []
        for interest in value:
            clean = interest.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized


class RequirementExtractionRequest(BaseModel):
    """POST /api/trips/requirements 的请求体。

    previous_draft 由前端回传上一轮结果，实现多轮修改；
    LangGraph Checkpointer（第 13 阶段）接入后，这一层会改成按 thread_id 读取。
    """

    message: str = Field(..., min_length=1, max_length=2000, description="用户自然语言输入")
    previous_draft: TripRequirementDraft | None = Field(
        None, description="多轮修改时带上上一轮草稿；首轮不传"
    )

    @field_validator("message")
    @classmethod
    def _reject_blank_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message 不能为空白")
        return normalized


class RequirementExtraction(BaseModel):
    """一次需求提取的完整结果：草稿 + 缺失项 + 追问。

    clarifying_questions 由第 4 阶段的 LLM/Service 填充（自然语言措辞属于业务层），
    这里只负责携带；missing_fields 是推导出来的，避免 LLM 自己数错。

    warnings / extracted_by 用于透明化降级：LLM 不可用时必须让调用方知道
    这次结果是规则解析出来的，不能伪装成模型输出。
    """

    draft: TripRequirementDraft
    clarifying_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extracted_by: Literal["llm", "rules"] = "rules"

    @computed_field
    @property
    def missing_fields(self) -> list[MissingField]:
        return self.draft.missing_fields

    @computed_field
    @property
    def is_ready(self) -> bool:
        return self.draft.is_ready

    def to_requirements(self) -> TripRequirements:
        return self.draft.to_requirements()
