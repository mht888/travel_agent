"""百度 MCP 工具的单元测试。

全部使用 fake caller，不启动 npx、不联网，验证四件事：
1. 只暴露路线矩阵 / 天气 / 地理编码三个工具；
2. 入参 GCJ-02 在调用 MCP 前被转成 BD-09，地理编码结果转回 GCJ-02；
3. 路线矩阵按百度对数上限分批，并把 results 规范成有向边；
4. MCP 故障与参数错误都翻译成 {"ok": false, ...}，不打断 Agent 循环。
"""

import asyncio
import json
from typing import Any

import pytest

from backend.tools import baidu_mcp_tools as mcp


class FakeCaller:
    """模拟 MCP 会话，记录调用参数并按配置返回。"""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = responses

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        response = self._responses[name]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(arguments)
        return response


def _tools(caller: FakeCaller) -> dict[str, Any]:
    return {
        tool.name: tool
        for tool in mcp.create_baidu_mcp_tools(caller=caller)
    }


def _invoke(
    tools: dict[str, Any], name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    raw = asyncio.run(tools[name].ainvoke(payload))
    assert isinstance(raw, str), f"Tool 必须返回 JSON 字符串，实际 {type(raw)}"
    return json.loads(raw)


def test_registers_only_scoped_tools() -> None:
    assert set(_tools(FakeCaller({}))) == {
        "mcp_route_matrix",
        "mcp_weather",
        "mcp_geocode",
    }


def test_coordinate_conversion_round_trip() -> None:
    bd_lat, bd_lng = mcp._gcj02_to_bd09(22.79, 108.39)
    gcj_lat, gcj_lng = mcp._bd09_to_gcj02(bd_lat, bd_lng)

    assert gcj_lat == pytest.approx(22.79, abs=1e-6)
    assert gcj_lng == pytest.approx(108.39, abs=1e-6)


def test_geocode_converts_result_back_to_gcj02() -> None:
    bd_lat, bd_lng = 22.794423933944028, 108.40196412066248
    caller = FakeCaller(
        {
            "map_geocode": {
                "location": {"lat": bd_lat, "lng": bd_lng},
                "level": "旅游景点",
                "confidence": 50,
                "precise": 0,
            }
        }
    )

    result = _invoke(
        _tools(caller), "mcp_geocode", {"address": "南宁市青秀区青秀山"}
    )

    assert result["ok"] is True
    assert result["coord_type"] == "gcj02"
    assert caller.calls == [
        ("map_geocode", {"address": "南宁市青秀区青秀山"})
    ]
    expected_lat, expected_lng = mcp._bd09_to_gcj02(bd_lat, bd_lng)
    assert result["location"]["lat"] == pytest.approx(expected_lat, abs=1e-6)
    assert result["location"]["lng"] == pytest.approx(expected_lng, abs=1e-6)


def test_route_matrix_normalizes_entries_and_uses_bd09() -> None:
    def response(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "elements": [
                        {
                            "duration": {"value": 600, "text": "10分钟"},
                            "distance": {"value": 800, "text": "0.8公里"},
                        }
                        for _ in arguments["destinations"]
                    ]
                }
                for _ in arguments["origins"]
            ]
        }

    caller = FakeCaller({"map_distance_matrix": response})
    result = _invoke(
        _tools(caller),
        "mcp_route_matrix",
        {
            "origins": [{"lat": 22.79, "lng": 108.39}],
            "destinations": [{"lat": 22.817, "lng": 108.332}],
            "mode": "walking",
        },
    )

    assert result["ok"] is True
    assert result["pair_count"] == 1
    assert result["incomplete"] is False
    entry = result["entries"][0]
    assert entry["from_index"] == 0
    assert entry["to_index"] == 0
    assert entry["duration_minutes"] == 10.0
    assert entry["distance_meters"] == 800

    # 传给百度 MCP 的必须是 BD-09，而不是入参的 GCJ-02
    bd_lat, bd_lng = mcp._gcj02_to_bd09(22.79, 108.39)
    assert caller.calls[0][1]["origins"] == [f"{bd_lat:.6f},{bd_lng:.6f}"]


def test_route_matrix_batches_by_pair_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(mcp._PAIR_LIMITS, "walking", 2)

    def response(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "elements": [
                        {"duration": {"value": 60}, "distance": {"value": 100}}
                        for _ in arguments["destinations"]
                    ]
                }
                for _ in arguments["origins"]
            ]
        }

    caller = FakeCaller({"map_distance_matrix": response})
    result = _invoke(
        _tools(caller),
        "mcp_route_matrix",
        {
            "origins": [{"lat": 22.79, "lng": 108.39}, {"lat": 22.80, "lng": 108.40}],
            "destinations": [
                {"lat": 22.81, "lng": 108.41},
                {"lat": 22.82, "lng": 108.42},
                {"lat": 22.83, "lng": 108.43},
            ],
            "mode": "walking",
        },
    )

    assert result["ok"] is True
    assert result["pair_count"] == 6
    assert len(result["entries"]) == 6
    # 上限 2 对：先按终点切 2+1，再按起点切 1+1 / 2
    assert len(caller.calls) == 3


def test_route_matrix_accepts_single_destination_object() -> None:
    caller = FakeCaller(
        {
            "map_distance_matrix": {
                "results": [
                    {
                        "elements": {
                            "duration": {"value": 120},
                            "distance": {"value": 300},
                        }
                    }
                ]
            }
        }
    )

    result = _invoke(
        _tools(caller),
        "mcp_route_matrix",
        {
            "origins": [{"lat": 0.0, "lng": 0.0}],
            "destinations": [{"lat": 0.1, "lng": 0.1}],
        },
    )

    assert result["ok"] is True
    assert result["entries"][0]["duration_minutes"] == 2.0


def test_route_matrix_accepts_flattened_mcp_rows() -> None:
    # 真实 MCP Server 会把 3×2 拍平成 6 行，每行 elements 是单个对象。
    # 旧解析会把它当成 6 个 origin，from_index 涨到 5，调用方按 top[5] 取值时越界。
    rows = [
        {
            "elements": {
                "duration": {"value": 60 * (index + 1)},
                "distance": {"value": 100},
            }
        }
        for index in range(6)
    ]
    caller = FakeCaller({"map_distance_matrix": {"results": rows}})

    result = _invoke(
        _tools(caller),
        "mcp_route_matrix",
        {
            "origins": [
                {"lat": 22.79, "lng": 108.39},
                {"lat": 22.80, "lng": 108.40},
                {"lat": 22.81, "lng": 108.41},
            ],
            "destinations": [
                {"lat": 22.82, "lng": 108.42},
                {"lat": 22.83, "lng": 108.43},
            ],
            "mode": "walking",
        },
    )

    assert result["ok"] is True
    assert result["incomplete"] is False
    assert [
        (entry["from_index"], entry["to_index"])
        for entry in result["entries"]
    ] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]


def test_weather_uses_district_id_and_compacts_fields() -> None:
    caller = FakeCaller(
        {
            "map_weather": {
                "location": {
                    "country": "中国",
                    "province": "广西壮族自治区",
                    "city": "南宁市",
                    "name": "南宁市",
                    "id": "450100",
                },
                "now": {
                    "text": "多云",
                    "temp": 25,
                    "rh": 66,
                    "pm25": 17,
                    "uvi": 2,
                },
                "forecasts": [
                    {
                        "date": "2026-11-15",
                        "week": "星期日",
                        "text_day": "多云",
                        "text_night": "阴",
                        "high": 25,
                        "low": 16,
                        "pressure": 1000,
                    }
                ],
            }
        }
    )

    result = _invoke(
        _tools(caller), "mcp_weather", {"district_id": "450100"}
    )

    assert result["ok"] is True
    assert caller.calls == [("map_weather", {"districtId": "450100"})]
    assert result["now"] == {"text": "多云", "temp": 25, "rh": 66, "uvi": 2}
    assert result["forecasts"] == [
        {
            "date": "2026-11-15",
            "week": "星期日",
            "text_day": "多云",
            "text_night": "阴",
            "high": 25,
            "low": 16,
        }
    ]


def test_weather_location_is_longitude_then_latitude() -> None:
    caller = FakeCaller({"map_weather": {"location": {}, "now": {}, "forecasts": []}})

    result = _invoke(
        _tools(caller),
        "mcp_weather",
        {"latitude": 22.79, "longitude": 108.39},
    )

    assert result["ok"] is True
    bd_lat, bd_lng = mcp._gcj02_to_bd09(22.79, 108.39)
    # 天气接口要求「经度,纬度」，顺序不能和算路一样
    assert caller.calls[0][1] == {"location": f"{bd_lng:.6f},{bd_lat:.6f}"}


def test_weather_requires_district_or_coordinates() -> None:
    result = _invoke(_tools(FakeCaller({})), "mcp_weather", {})

    assert result["ok"] is False
    assert result["error_code"] == "invalid_arguments"


def test_mcp_error_becomes_retryable_result() -> None:
    caller = FakeCaller(
        {"map_weather": mcp.BaiduMcpError("upstream down", retryable=True)}
    )

    result = _invoke(_tools(caller), "mcp_weather", {"district_id": "450100"})

    assert result["ok"] is False
    assert result["error_code"] == "mcp_unavailable"
    assert result["retryable"] is True


def test_invalid_arguments_return_json_instead_of_raising() -> None:
    result = _invoke(
        _tools(FakeCaller({})),
        "mcp_route_matrix",
        {"origins": [], "destinations": [{"lat": 0.0, "lng": 0.0}]},
    )

    assert result["ok"] is False
    assert result["error_code"] == "invalid_arguments"


def test_geocode_without_location_returns_not_found() -> None:
    caller = FakeCaller({"map_geocode": {"confidence": 0}})

    result = _invoke(
        _tools(caller), "mcp_geocode", {"address": "不存在的地址"}
    )

    assert result["ok"] is False
    assert result["error_code"] == "geocode_not_found"
