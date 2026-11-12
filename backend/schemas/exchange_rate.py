"""汇率快照与校验结果。

为什么单独建模：
    汇率是全局共享数据，预算、需求抽取、API 都要消费。
    快照结构确定后，数据源和调度方式可以替换，消费方不用改。

约定：
    - base 固定 USD，rates 表示「1 USD 可兑换多少目标币」。
    - 汇率必须是正的有限 Decimal，不能用 float 参与后续换算。
    - fetched_at 必须带时区；business_date 表示汇率所属交易日。
"""

from datetime import date, datetime, timezone
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from backend.schemas.common import DataStatus


class FXSnapshot(BaseModel):
    """一次抓取的完整汇率快照。"""

    base: str = Field("USD", description="基准币种，当前固定 USD")
    rates: dict[str, Decimal] = Field(
        ..., description="1 个基准币可兑换的目标币数量，必须包含 USD=1"
    )
    fetched_at: datetime = Field(..., description="抓取时间，必须带时区")
    business_date: date = Field(..., description="汇率所属交易日，用于区分周末/节假日")
    source: str = Field(..., min_length=1, description="数据源标识，如 ecb / er_api")
    status: DataStatus = Field(DataStatus.REAL, description="数据可信度标记")
    warnings: list[str] = Field(
        default_factory=list, description="抓取阶段的降级说明，例如跳过了非法汇率"
    )

    @field_validator("base")
    @classmethod
    def _require_usd_base(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized != "USD":
            raise ValueError("当前只支持 USD 作为基准币种")
        return normalized

    @field_validator("rates")
    @classmethod
    def _normalize_rates(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        if not value:
            raise ValueError("rates 不能为空")

        normalized: dict[str, Decimal] = {}
        for code, rate in value.items():
            clean = code.strip().upper()
            if len(clean) != 3 or not clean.isalpha():
                raise ValueError(f"币种代码必须是三个字母：{code!r}")
            if not rate.is_finite() or rate <= 0:
                raise ValueError(f"{clean} 的汇率必须是正的有限数值：{rate}")
            normalized[clean] = rate

        if "USD" not in normalized:
            raise ValueError("rates 必须包含 USD")
        if normalized["USD"] != Decimal("1"):
            raise ValueError("rates['USD'] 必须等于 1")
        return normalized

    @field_validator("fetched_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("fetched_at 必须带时区，例如 datetime.now(timezone.utc)")
        return value

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("source 不能为空")
        return normalized

    def rate_for(self, currency: str) -> Decimal | None:
        """取某个币种相对 USD 的汇率；没有则返回 None。"""
        return self.rates.get(currency.strip().upper())


class ValidationReport(BaseModel):
    """候选快照的校验结果。passed=False 时禁止发布。"""

    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_codes: list[str] = Field(
        default_factory=list, description="缺失的硬性核心币种"
    )
    deviations: dict[str, Decimal] = Field(
        default_factory=dict, description="与上一份快照的偏离比例，0.05 表示 5%"
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    previous_source: str | None = Field(
        None, description="对比基准来自哪个数据源；没有历史快照时为 None"
    )
