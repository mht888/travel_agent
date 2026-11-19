from datetime import datetime, timezone
from decimal import Decimal

from backend.services.usage import UsageCollector, active_usage, count_external_request


def test_usage_prices_deepseek_cache_and_beijing_holidays() -> None:
    collector = UsageCollector("run", "deepseek-flash")
    common = {
        "stage": "Planner",
        "model": "deepseek-v4-flash",
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 100},
    }
    peak = collector.record(
        call_id="1", started_at=datetime(2026, 11, 19, 2, tzinfo=timezone.utc), **common,
    )
    holiday = collector.record(
        call_id="2", started_at=datetime(2026, 10, 1, 2, tzinfo=timezone.utc), **common,
    )
    assert (peak.price_band, peak.estimated_cost_cny) == ("peak", Decimal("0.002604"))
    assert (holiday.price_band, holiday.estimated_cost_cny) == ("off_peak", Decimal("0.001302"))
    assert collector.summary()["total_tokens"] == 2200


def test_usage_deduplicates_calls_and_flags_missing_billing_information() -> None:
    collector = UsageCollector("run", "deepseek-flash")
    args = {
        "call_id": "1", "stage": "PoiMatching", "model": "deepseek-flash",
        "started_at": datetime(2026, 11, 21, 2, tzinfo=timezone.utc),
        "usage": None, "fallback_usage": {"input_tokens": 10, "output_tokens": 4},
    }
    collector.record(**args)
    collector.record(**args)
    collector.record(**{**args, "call_id": "2", "fallback_usage": None})
    token = active_usage.set(collector)
    try:
        count_external_request("gaode_mcp")
        count_external_request("gaode_mcp")
        count_external_request("bocha_mcp")
    finally:
        active_usage.reset(token)

    summary = collector.summary()
    assert summary["model_calls"] == 2
    assert summary["total_tokens"] == 14
    assert summary["calls_missing_usage"] == 1
    assert summary["calls_missing_cost"] == 2
    assert summary["external_requests"] == {"gaode_mcp": 2, "bocha_mcp": 1}
