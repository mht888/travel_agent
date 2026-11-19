import asyncio
import json
from datetime import date
from unittest.mock import AsyncMock, call

import pytest

from backend.schemas.weather import WeatherCondition
from backend.services.trip_data import GaodeTripData
from backend.tools.gaode_mcp_tools import GaodeMcpClient, create_gaode_mcp_tool


@pytest.mark.parametrize("nested_casts", [False, True])
def test_gaode_forecast_matches_trip_dates(nested_casts: bool) -> None:
    casts = [
        {
            "date": f"2026-11-{day}",
            "dayweather": "小雨",
            "daytemp": "17",
            "nighttemp": "12",
        }
        for day in (21, 19, 20, 22)
    ]
    tool = AsyncMock()
    tool.ainvoke.return_value = json.dumps({
        "ok": True,
        "data": {"city": "南宁市", "forecasts": [{"casts": casts}] if nested_casts else casts},
    })
    forecast = asyncio.run(
        GaodeTripData(tool, concurrency=1).forecast(
            "南宁", date(2026, 11, 20), date(2026, 11, 22)
        )
    )

    tool.ainvoke.assert_awaited_once_with({"action": "weather", "city": "南宁"})
    assert [record.date.day for record in forecast.days] == [20, 21, 22]
    assert forecast.days[0].condition == WeatherCondition.RAIN
    assert forecast.days[0].precipitation_probability is None


def test_gaode_reuses_city_code_for_weather_and_poi_search() -> None:
    caller = GaodeMcpClient(api_key="test")
    remote = AsyncMock(side_effect=[
        {"results": [{"level": "市", "adcode": "450100"}]},
        {"forecasts": [{"date": "2026-11-20", "dayweather": "阴", "daytemp": "16", "nighttemp": "12"}]},
        {"pois": []},
        {"pois": []},
    ])
    caller._call_remote = remote
    tool = create_gaode_mcp_tool(caller=caller, default_city="Nanning")

    async def check():
        forecast = await GaodeTripData(tool, concurrency=1).forecast(
            "Nanning", date(2026, 11, 20), date(2026, 11, 22)
        )
        await tool.ainvoke({"action": "text_search", "keywords": "酒店", "poi_kind": "hotel"})
        await tool.ainvoke({"action": "text_search", "keywords": "早餐", "poi_kind": "restaurant"})
        return forecast

    forecast = asyncio.run(check())
    assert [record.date.day for record in forecast.days] == [20]
    assert remote.await_args_list == [
        call("maps_geo", {"address": "Nanning"}),
        call("maps_weather", {"city": "450100"}),
        call("maps_text_search", {"keywords": "酒店", "city": "450100", "citylimit": True}),
        call("maps_text_search", {"keywords": "早餐", "city": "450100", "citylimit": True}),
    ]
