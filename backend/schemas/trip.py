"""用户旅行需求（Trip）。这是「用户想要什么」，与 Itinerary（系统排出来什么）严格分开。"""

from datetime import date, timedelta
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.schemas.common import Money


class TravelStyle(str, Enum):
    """节奏偏好。第 7 阶段用它换算每日最大游玩分钟数。"""

    RELAXED = "relaxed"
    BALANCED = "balanced"
    INTENSIVE = "intensive"


class TripRequirements(BaseModel):
    """结构化后的旅行需求。第 4 阶段由 Requirement Extraction 产出。

    注意它同时是「多轮修改」的载体：用户说“预算改成 1500”，改的就是这里的字段。
    """

    destination: str = Field(..., min_length=1, description="城市名，如 Tokyo")
    duration_days: int = Field(..., ge=1, le=30)
    start_date: date | None = Field(None, description="未知时允许为空，由前端补")
    travelers: int = Field(1, ge=1, le=20)
    budget: Money | None = Field(None, description="总预算；为空表示用户未给")
    interests: list[str] = Field(default_factory=list)
    travel_style: TravelStyle = TravelStyle.BALANCED
    notes: str | None = Field(
        None, description="未结构化的补充要求，如“不要太紧凑”“不吃生食”"
    )

    @field_validator("destination")
    @classmethod
    def _normalize_destination(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("destination 不能为空白")
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
    def end_date(self) -> date | None:
        """最后一天。start_date 未知时返回 None。"""
        if self.start_date is None:
            return None
        return self.start_date + timedelta(days=self.duration_days - 1)
