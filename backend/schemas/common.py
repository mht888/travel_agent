"""跨模块复用的基础模型：坐标、金额、数据溯源。

为什么单独一个文件：这些结构会被 attraction / itinerary / budget 同时引用，
放在任意一个业务 Schema 里都会造成循环依赖。
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 第一阶段只支持这几个币种；接入汇率服务时再扩展，不要在业务代码里散落 if/else
ALLOWED_CURRENCIES: frozenset[str] = frozenset({"USD", "JPY", "CNY", "EUR"})


class DataStatus(str, Enum):
    """数据可信度。任何展示给用户的价格/时间都必须带这个标记。"""

    REAL = "real"              # 来自真实 API 的实时数据
    ESTIMATED = "estimated"    # 我们估算的（如餐饮日均消费）
    MOCK = "mock"              # 开发阶段的假数据
    USER_INPUT = "user_input"  # 用户自己说的


class TransportMode(str, Enum):
    """交通方式。路线模块和行程模块都要用，所以放在公共层。

    放在这里而不是 itinerary.py：否则 route.py 会为了一个枚举反向依赖行程模型。
    """

    WALK = "walk"
    TRANSIT = "transit"
    DRIVE = "drive"
    TAXI = "taxi"


class Provenance(BaseModel):
    """外部数据溯源（对应需求文档第六节）。

    value 本体在各业务模型字段上，这里只记录「这条数据从哪来、什么时候拿的、可不可信」。
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="如 mock:tokyo_dataset / google_places / user")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="获取时间，必须带时区",
    )
    status: DataStatus = DataStatus.ESTIMATED
    valid_until: datetime | None = Field(
        None, description="有效期截止；用于判断缓存是否过期"
    )

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source 不能为空")
        return normalized

    @field_validator("fetched_at", "valid_until")
    @classmethod
    def _require_timezone(cls, value: datetime | None) -> datetime | None:
        """不带时区的时间无法和 valid_until 正确比较，直接在入口拦掉。"""
        if value is not None and value.tzinfo is None:
            raise ValueError("时间必须带时区，例如 datetime.now(timezone.utc)")
        return value


class Money(BaseModel):
    """金额。金额和币种必须成对出现，禁止只传一个数字。"""

    amount: Decimal = Field(..., ge=0)
    currency: str = Field(..., description="ISO 4217 三字母代码，如 USD/JPY/CNY")
    status: DataStatus = DataStatus.ESTIMATED

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3:
            raise ValueError("currency 必须是三字母 ISO 4217 代码")
        if normalized not in ALLOWED_CURRENCIES:
            allowed = ", ".join(sorted(ALLOWED_CURRENCIES))
            raise ValueError(f"暂不支持币种 {normalized}，当前允许：{allowed}")
        return normalized


class GeoPoint(BaseModel):
    """经纬度。范围校验交给 Pydantic，非法坐标不允许进入后续路线计算。"""

    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class HealthResponse(BaseModel):
    """健康检查响应。字段少但固定，方便前端和探针直接消费。"""

    status: Literal["ok"] = "ok"
    env: str
    mock_mode: bool
