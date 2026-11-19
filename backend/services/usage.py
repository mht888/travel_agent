"""按一次行程记录模型用量和外部请求数。"""

from collections import Counter
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, time
from decimal import Decimal
from threading import Lock
from zoneinfo import ZoneInfo

_BEIJING = ZoneInfo("Asia/Shanghai")
_MILLION = Decimal(1_000_000)
# 每百万 token 的人民币单价：缓存命中、缓存未命中、输出。
_DEEPSEEK_PRICES = {
    "off_peak": (Decimal("0.02"), Decimal("1"), Decimal("4")),
    "peak": (Decimal("0.04"), Decimal("2"), Decimal("8")),
}
_FLASH_NAMES = {"deepseek-flash", "deepseek-v4-flash"}


def _price_band(started_at: datetime) -> str | None:
    local = started_at.astimezone(_BEIJING)
    clock = local.time()
    if local.weekday() >= 5 or not (
        time(9) <= clock < time(12) or time(14) <= clock < time(18)
    ):
        return "off_peak"
    try:
        from chinese_calendar import is_holiday

        return "off_peak" if is_holiday(local.date()) else "peak"
    except (ImportError, NotImplementedError):
        # 节假日日历尚未覆盖该年份，避免把法定假日按高峰计费。
        return None


@dataclass(frozen=True)
class ModelUsageRecord:
    call_id: str
    stage: str
    model: str
    started_at: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    price_band: str | None
    estimated_cost_cny: Decimal | None


class UsageCollector:
    def __init__(self, trace_id: str, priced_model: str) -> None:
        self.trace_id = trace_id
        self.priced_model = priced_model.casefold()
        self._records: dict[str, ModelUsageRecord] = {}
        self._requests: Counter[str] = Counter()
        self._lock = Lock()

    def record(
        self, *, call_id: str, stage: str, model: str, started_at: datetime,
        usage: Mapping | None, fallback_usage: Mapping | None = None,
    ) -> ModelUsageRecord:
        raw = usage or {}
        normalized = fallback_usage or {}
        input_tokens = raw.get("prompt_tokens", normalized.get("input_tokens"))
        output_tokens = raw.get("completion_tokens", normalized.get("output_tokens"))
        total_tokens = raw.get("total_tokens", normalized.get("total_tokens"))
        cached = raw.get("prompt_cache_hit_tokens")
        if cached is None:
            cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = (normalized.get("input_token_details") or {}).get("cache_read")
        if cached is None and input_tokens is not None:
            misses = raw.get("prompt_cache_miss_tokens")
            if isinstance(misses, int) and 0 <= misses <= input_tokens:
                cached = input_tokens - misses
        if input_tokens is not None and output_tokens is not None and total_tokens is None:
            total_tokens = input_tokens + output_tokens

        band = _price_band(started_at)
        cost = None
        model_name = model.casefold()
        is_priced = model_name in _FLASH_NAMES and self.priced_model in _FLASH_NAMES
        if (
            is_priced and band is not None
            and input_tokens is not None and output_tokens is not None
            and (input_tokens == 0 or cached is not None and 0 <= cached <= input_tokens)
        ):
            cached = cached or 0
            hit, miss, output = _DEEPSEEK_PRICES[band]
            cost = (cached * hit + (input_tokens - cached) * miss + output_tokens * output) / _MILLION
        record = ModelUsageRecord(
            call_id, stage, model, started_at.astimezone(_BEIJING).isoformat(),
            input_tokens, cached, output_tokens, total_tokens, band, cost,
        )
        with self._lock:
            return self._records.setdefault(call_id, record)

    def count_request(self, provider: str) -> None:
        with self._lock:
            self._requests[provider] += 1

    def summary(self) -> dict:
        with self._lock:
            records = list(self._records.values())
            requests = {provider: self._requests[provider] for provider in ("bocha_mcp", "gaode_mcp")}

        def totals(rows: list[ModelUsageRecord]) -> dict:
            return {
                "model_calls": len(rows),
                "input_tokens": sum(row.input_tokens or 0 for row in rows),
                "cached_input_tokens": sum(row.cached_input_tokens or 0 for row in rows),
                "output_tokens": sum(row.output_tokens or 0 for row in rows),
                "total_tokens": sum(row.total_tokens or 0 for row in rows),
                "known_estimated_cost_cny": sum(
                    (row.estimated_cost_cny for row in rows if row.estimated_cost_cny is not None),
                    Decimal(0),
                ),
                "calls_missing_usage": sum(row.input_tokens is None or row.output_tokens is None for row in rows),
                "calls_missing_cost": sum(row.estimated_cost_cny is None for row in rows),
            }

        stages: dict[str, list[ModelUsageRecord]] = {}
        for record in records:
            stages.setdefault(record.stage, []).append(record)
        return {
            "trace_id": self.trace_id,
            **totals(records),
            "by_stage": {stage: totals(rows) for stage, rows in stages.items()},
            "external_requests": requests,
            "model_calls_detail": [asdict(row) for row in records],
        }


active_usage: ContextVar[UsageCollector | None] = ContextVar("active_usage", default=None)


def count_external_request(provider: str) -> None:
    if collector := active_usage.get():
        collector.count_request(provider)
