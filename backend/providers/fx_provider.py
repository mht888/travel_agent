"""汇率数据源抽象与内置实现。

职责边界：
    Provider 只负责「拉取 + 转成 FXSnapshot」，不做校验、不落盘。
    校验与发布由 FXService 统一处理，保证换源后规则一致。

内置数据源：
    ecb     欧洲央行参考汇率（经 Frankfurter 暴露），约 30 种，权威但覆盖有限
    er_api  open.er-api.com，160+ 种，用于补齐 AED/SAR/TWD/VND 等

为什么不用 LLM 拉汇率：
    汇率必须是确定值。LLM 只负责决定「用哪个源、是否重试」，数字一律来自这里。
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from backend.schemas.common import DataStatus
from backend.schemas.exchange_rate import FXSnapshot


class ExchangeRateProviderError(RuntimeError):
    """数据源不可用或返回结构非法。Agent 可据此换源重试。"""


class ExchangeRateProvider(ABC):
    """汇率数据源接口。实现方只取数，不校验、不发布。"""

    name: str

    @abstractmethod
    async def fetch_latest(self, base: str = "USD") -> FXSnapshot:
        """拉取最新汇率。失败抛 ExchangeRateProviderError。"""


def _parse_rates(
    raw: Any,
    base: str,
    source: str,
    warnings: list[str],
) -> dict[str, Decimal]:
    """把数据源返回的 rates 转成 Decimal，坏数据跳过并记 warning。

    单个币种异常不应该让整份快照不可用，否则一个脏字段会拖垮整个更新任务。
    """
    if not isinstance(raw, dict) or not raw:
        raise ExchangeRateProviderError(f"{source} 返回的 rates 为空或格式非法")

    rates: dict[str, Decimal] = {}
    for code, value in raw.items():
        clean = str(code).strip().upper()
        if len(clean) != 3 or not clean.isalpha():
            warnings.append(f"{source} 跳过非法币种代码 {code!r}")
            continue
        try:
            rate = Decimal(str(value))
        except (InvalidOperation, ValueError):
            warnings.append(f"{source} 跳过无法解析的汇率 {clean}={value!r}")
            continue
        if not rate.is_finite() or rate <= 0:
            warnings.append(f"{source} 跳过非正汇率 {clean}={value!r}")
            continue
        rates[clean] = rate

    rates[base] = Decimal("1")
    return rates


class ECBProvider(ExchangeRateProvider):
    """欧洲央行参考汇率，通过 Frankfurter 公开 API 获取（免费、无需 key）。

    注意：ECB 每个工作日约 16:00 CET 发布一次，周末和欧元区节假日没有新数据。
    6 点任务拿到的是上一交易日汇率，必须用 business_date 表达，而不是抓取时间。
    """

    name = "ecb"

    def __init__(
        self,
        base_url: str = "https://api.frankfurter.dev/v1",
        timeout_seconds: float = 15.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def fetch_latest(self, base: str = "USD") -> FXSnapshot:
        base = base.strip().upper()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self._base_url}/latest", params={"from": base}
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExchangeRateProviderError(f"ecb 拉取失败：{exc}") from exc

        if not isinstance(payload, dict):
            raise ExchangeRateProviderError("ecb 返回格式非法：不是 JSON 对象")

        reported_base = str(payload.get("base", "")).upper()
        if reported_base != base:
            raise ExchangeRateProviderError(
                f"ecb 返回基准币种 {reported_base!r}，与请求 {base!r} 不一致"
            )

        warnings: list[str] = []
        rates = _parse_rates(payload.get("rates"), base, self.name, warnings)

        try:
            business_date = date.fromisoformat(str(payload["date"]))
        except (KeyError, ValueError) as exc:
            raise ExchangeRateProviderError(
                f"ecb 返回的发布日期非法：{payload.get('date')!r}"
            ) from exc

        return FXSnapshot(
            base=base,
            rates=rates,
            fetched_at=datetime.now(timezone.utc),
            business_date=business_date,
            source=self.name,
            status=DataStatus.REAL,
            warnings=warnings,
        )


class ErApiProvider(ExchangeRateProvider):
    """open.er-api.com 免费汇率源，160+ 币种，适合补齐 ECB 未覆盖的币种。"""

    name = "er_api"

    def __init__(
        self,
        base_url: str = "https://open.er-api.com/v6/latest",
        timeout_seconds: float = 15.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def fetch_latest(self, base: str = "USD") -> FXSnapshot:
        base = base.strip().upper()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._base_url}/{base}")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExchangeRateProviderError(f"er_api 拉取失败：{exc}") from exc

        if not isinstance(payload, dict):
            raise ExchangeRateProviderError("er_api 返回格式非法：不是 JSON 对象")

        if payload.get("result") != "success":
            raise ExchangeRateProviderError(
                f"er_api 返回失败：{payload.get('error-type') or payload.get('result')!r}"
            )

        reported_base = str(payload.get("base_code", "")).upper()
        if reported_base != base:
            raise ExchangeRateProviderError(
                f"er_api 返回基准币种 {reported_base!r}，与请求 {base!r} 不一致"
            )

        warnings: list[str] = []
        rates = _parse_rates(payload.get("rates"), base, self.name, warnings)

        timestamp = payload.get("time_last_update_unix")
        if isinstance(timestamp, (int, float)):
            business_date = datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).date()
        else:
            business_date = datetime.now(timezone.utc).date()
            warnings.append("er_api 未返回更新时间，business_date 使用抓取日期")

        return FXSnapshot(
            base=base,
            rates=rates,
            fetched_at=datetime.now(timezone.utc),
            business_date=business_date,
            source=self.name,
            status=DataStatus.REAL,
            warnings=warnings,
        )


_PROVIDER_CLASSES: dict[str, type[ExchangeRateProvider]] = {
    "ecb": ECBProvider,
    "er_api": ErApiProvider,
}


def build_provider(name: str) -> ExchangeRateProvider:
    """按名称创建数据源。未知名称抛 ValueError，由工具层转成可读错误。"""
    key = name.strip().lower()
    provider_class = _PROVIDER_CLASSES.get(key)
    if provider_class is None:
        allowed = ", ".join(sorted(_PROVIDER_CLASSES))
        raise ValueError(f"未知汇率数据源 {name!r}，可选：{allowed}")
    return provider_class()
