"""餐厅数据结构。

为什么单独建模型，而不是复用 Attraction：
    餐厅的核心字段是「菜系」和「人均消费」，景点的核心字段是「游玩时长」和「门票」。
    如果硬塞进 Attraction，`ticket_price` 就会被拿来装人均餐费，
    第 8 阶段的预算拆分和前端展示都会跟着含糊。

字段约定和 Attraction 保持一致：
    - 缺失用 None，不要用 0 或空字符串冒充「有数据」
    - 免费/无需消费用 Money(0)，与「未知」的 None 区分
    - 所有外部数据都要带 Provenance
"""

from pydantic import BaseModel, Field, field_validator

from backend.schemas.attraction import OpeningInterval
from backend.schemas.common import GeoPoint, Money, Provenance


class Restaurant(BaseModel):
    """统一后的餐厅。Provider 负责把外部接口转换成它。"""

    id: str = Field(..., min_length=1, description="稳定 ID，用于后续引用和去重")
    name: str = Field(..., min_length=1)
    city: str
    country: str | None = None
    cuisines: list[str] = Field(
        default_factory=list, description="菜系标签，如 ramen / sushi / tempura"
    )
    description: str | None = None
    location: GeoPoint
    rating: float | None = Field(None, ge=0, le=5)
    review_count: int | None = Field(None, ge=0)
    opening_hours: list[OpeningInterval] = Field(default_factory=list)
    average_price: Money | None = Field(
        None, description="人均消费；未知用 None，不按 0 计算"
    )
    provenance: Provenance

    @field_validator("cuisines")
    @classmethod
    def _normalize_cuisines(cls, value: list[str]) -> list[str]:
        """小写 + 去空白 + 去重，避免 "Ramen" 和 "ramen" 被算成两种菜系。"""
        seen: set[str] = set()
        normalized: list[str] = []
        for cuisine in value:
            clean = cuisine.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized

    def matches_cuisine(self, cuisine: str) -> bool:
        """菜系是否命中。大小写不敏感。"""
        target = cuisine.strip().lower()
        if not target:
            return False
        return target in self.cuisines


class RestaurantSearchResponse(BaseModel):
    """搜索接口的响应。回显查询条件，方便前端和日志对齐。"""

    city: str
    cuisines: list[str] = Field(default_factory=list)
    total: int = Field(..., ge=0, description="实际返回条数，等于 len(items)")
    items: list[Restaurant] = Field(default_factory=list)
