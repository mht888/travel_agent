"""WeatherService 单元测试。

重点验证三件事：日期连续、结果确定、参数被正确拦截。
"""

import asyncio
from datetime import date, timedelta

import pytest

from backend.providers.mock_provider import MockWeatherProvider
from backend.schemas.common import DataStatus
from backend.services.weather_service import WeatherService


def _service() -> WeatherService:
    return WeatherService(provider=MockWeatherProvider())


def _forecast(city: str, start: date, days: int):
    return asyncio.run(_service().get_forecast(city, start, days))


def test_returns_consecutive_days() -> None:
    start = date(2026, 4, 1)
    forecast = _forecast("Tokyo", start, 5)

    assert forecast.city == "Tokyo"
    assert len(forecast.days) == 5
    assert [day.date for day in forecast.days] == [
        start + timedelta(days=offset) for offset in range(5)
    ]

    for day in forecast.days:
        assert day.provenance.status is DataStatus.MOCK
        assert day.temp_min_c <= day.temp_max_c
        assert 0 <= day.precipitation_probability <= 100


def test_is_deterministic() -> None:
    """同一城市同一天必须返回同样结果，否则测试和 Benchmark 都无法复现。"""
    start = date(2026, 7, 10)

    first = _forecast("Tokyo", start, 3)
    second = _forecast("Tokyo", start, 3)

    assert first == second


def test_city_is_case_insensitive_for_data_and_trimmed_in_response() -> None:
    start = date(2026, 10, 1)

    lower = _forecast("tokyo", start, 2)
    upper = _forecast("  Tokyo  ", start, 2)

    assert upper.city == "Tokyo"
    assert [day.model_dump() for day in lower.days] == [
        day.model_dump() for day in upper.days
    ]


def test_rejects_blank_city() -> None:
    with pytest.raises(ValueError, match="city"):
        _forecast("   ", date(2026, 4, 1), 3)


@pytest.mark.parametrize("days", [0, -1, WeatherService.MAX_FORECAST_DAYS + 1])
def test_rejects_out_of_range_days(days: int) -> None:
    with pytest.raises(ValueError, match="days"):
        _forecast("Tokyo", date(2026, 4, 1), days)
