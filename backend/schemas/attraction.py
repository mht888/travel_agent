"""景点数据结构。

它是所有 Provider 的统一输出格式：Google Places / OSM / Mock 都必须转换成它，
外部接口的原始字段不允许泄漏到 Service 之上。
"""

from datetime import time
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.schemas.common import GeoPoint, Money, Provenance


def _to_minutes(value: time) -> int:
    """把 time 换成「当天第几分钟」，避免 time 对象不能直接相减的问题。"""
    return value.hour * 60 + value.minute


class AttractionCategory(str, Enum):
    """粗分类。用户兴趣标签直接映射到这里，保持标签体系稳定。"""

    CULTURE = "culture"
    FOOD = "food"
    SHOPPING = "shopping"
    NATURE = "nature"
    OUTDOOR_ACTIVITY = "outdoor_activity"
    NIGHTLIFE = "nightlife"
    ENTERTAINMENT = "entertainment"
    LANDMARKS = "landmark"
    WELLNESS = "wellness"
    ART = "art"
    OTHER = "other"


class OpeningInterval(BaseModel):
    """一条开放时段。一周可以有多条（例如午市/晚市分开）。"""

    weekday: int = Field(..., ge=0, le=6, description="0=周一, 6=周日")
    opens_at: time
    closes_at: time

    @model_validator(mode="after")
    def _validate_order(self) -> "OpeningInterval":
        if _to_minutes(self.closes_at) <= _to_minutes(self.opens_at):
            # 跨午夜（居酒屋、夜店）先不支持：遇到就拆成两条，不要静默放过
            raise ValueError("closes_at 必须晚于 opens_at，跨午夜请拆成两条时段")
        return self


class Attraction(BaseModel):
    """统一后的景点。字段缺失用 None，不要用 0 或空字符串冒充「有数据」。"""

    id: str = Field(..., min_length=1, description="稳定 ID，用于后续引用和去重")
    name: str = Field(..., min_length=1)
    city: str
    country: str | None = None
    category: AttractionCategory = AttractionCategory.OTHER
    tags: list[str] = Field(
        default_factory=list, description="细分标签，如 temple / sushi / ramen"
    )
    description: str | None = None
    location: GeoPoint
    rating: float | None = Field(None, ge=0, le=5)
    review_count: int | None = Field(None, ge=0)
    opening_hours: list[OpeningInterval] = Field(default_factory=list)
    estimated_duration_minutes: int = Field(
        ..., gt=0, le=600, description="预计游玩时长，第 7 阶段排程直接使用"
    )
    ticket_price: Money | None = Field(None, description="门票；免费景点用 0，未知用 None")
    provenance: Provenance

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        """小写 + 去空白 + 去重，避免 "Food" 和 "food" 被算两次兴趣命中。"""
        seen: set[str] = set()
        normalized: list[str] = []
        for tag in value:
            clean = tag.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized

    def matches_interest(self, interest: str) -> bool:
        """兴趣是否命中：粗分类相等，或细分标签相等。大小写不敏感。"""
        target = interest.strip().lower()
        if not target:
            return False
        return target == self.category.value or target in self.tags


class AttractionSearchResponse(BaseModel):
    """搜索接口的响应。回显查询条件，方便前端和日志对齐。"""

    city: str
    interests: list[str] = Field(default_factory=list)
    total: int = Field(..., ge=0, description="实际返回条数，等于 len(items)")
    items: list[Attraction] = Field(default_factory=list)
