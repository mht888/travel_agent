"""Travel Tools 单元测试。

不联网、不启动 FastAPI：直接 ainvoke，验证四件事：
1. 成功返回结构与 Service 契约一致；
2. 业务错误被翻译成 {"ok": false, ...}，而不是抛异常；
3. 入参非法走 handle_validation_error，同样返回错误对象；
4. 数据源故障返回可重试标记，Agent 循环不会被打断。
"""

import asyncio
import json
from datetime import date, timedelta
from typing import Any

from backend.tools.travel_tools import create_travel_tools


class _BrokenAttractionService:
    """模拟外部数据源整体不可用。"""

    async def search(self, **_: Any) -> Any:
        raise RuntimeError("upstream down")

    async def get_detail(self, _attraction_id: str) -> Any:
        raise RuntimeError("upstream down")


def _tools(**services: Any) -> dict[str, Any]:
    return {tool.name: tool for tool in create_travel_tools(**services)}


def _invoke(name: str, payload: dict[str, Any], **services: Any) -> dict[str, Any]:
    raw = asyncio.run(_tools(**services)[name].ainvoke(payload))
    assert isinstance(raw, str), f"Tool 必须返回 JSON 字符串，实际 {type(raw)}"
    return json.loads(raw)


def test_registers_expected_tool_names() -> None:
    assert set(_tools()) == {
        "search_attractions",
        "get_attraction_details",
        "get_route_matrix",
        "get_weather",
    }


def test_search_attractions_ranks_by_interest() -> None:
    result = _invoke(
        "search_attractions",
        {"city": "Tokyo", "interests": ["food"]},
    )

    assert result["ok"] is True
    assert result["city"] == "Tokyo"
    assert result["total"] == len(result["items"]) == 2
    assert all(item["category"] == "food" for item in result["items"])
    # 同为兴趣命中时按评分排序：筑地 4.4 高于一兰 4.1
    assert result["items"][0]["id"] == "tokyo-tsukiji-outer-market"
    assert result["items"][0]["provenance"]["status"] == "mock"


def test_search_unknown_city_returns_empty_with_hint() -> None:
    result = _invoke("search_attractions", {"city": "Atlantis"})

    assert result["ok"] is True
    assert result["total"] == 0
    assert result["items"] == []
    assert "hint" in result


def test_get_attraction_details_returns_full_fields() -> None:
    result = _invoke(
        "get_attraction_details",
        {"attraction_id": "tokyo-sensoji"},
    )

    assert result["ok"] is True
    item = result["item"]
    assert item["id"] == "tokyo-sensoji"
    assert item["estimated_duration_minutes"] == 90
    assert len(item["opening_hours"]) == 7
    assert item["location"]["lat"] == 35.7148
    assert item["provenance"]["source"] == "mock:tokyo_dataset"


def test_get_attraction_details_not_found() -> None:
    result = _invoke("get_attraction_details", {"attraction_id": "nope"})

    assert result["ok"] is False
    assert result["error_code"] == "attraction_not_found"
    assert result["retryable"] is False


def test_get_route_matrix_returns_directed_entries() -> None:
    result = _invoke(
        "get_route_matrix",
        {
            "city": "Tokyo",
            "attraction_ids": ["tokyo-sensoji", "tokyo-ueno-park"],
            "mode": "walk",
        },
    )

    assert result["ok"] is True
    assert result["mode"] == "walk"
    assert result["complete"] is True
    assert len(result["entries"]) == 2

    pairs = {(entry["from_id"], entry["to_id"]) for entry in result["entries"]}
    assert pairs == {
        ("tokyo-sensoji", "tokyo-ueno-park"),
        ("tokyo-ueno-park", "tokyo-sensoji"),
    }
    assert all(entry["duration_minutes"] > 0 for entry in result["entries"])


def test_get_route_matrix_reports_missing_ids() -> None:
    result = _invoke(
        "get_route_matrix",
        {
            "city": "Tokyo",
            "attraction_ids": ["tokyo-sensoji", "fake-id"],
        },
    )

    assert result["ok"] is False
    assert result["error_code"] == "attraction_not_found"
    assert result["missing"] == ["fake-id"]


def test_get_route_matrix_rejects_duplicate_ids() -> None:
    result = _invoke(
        "get_route_matrix",
        {
            "city": "Tokyo",
            "attraction_ids": ["tokyo-sensoji", "tokyo-sensoji"],
        },
    )

    assert result["ok"] is False
    assert result["error_code"] == "invalid_argument"
    assert "重复" in result["message"]


def test_invalid_arguments_return_error_instead_of_raising() -> None:
    # 只有 1 个景点：被 args_schema 的 min_length=2 拦下
    result = _invoke(
        "get_route_matrix",
        {"city": "Tokyo", "attraction_ids": ["tokyo-sensoji"]},
    )

    assert result["ok"] is False
    assert result["error_code"] == "invalid_arguments"


def test_get_weather_returns_consecutive_days() -> None:
    start = date(2026, 4, 1)
    result = _invoke(
        "get_weather",
        {"city": "Tokyo", "start_date": start.isoformat(), "days": 3},
    )

    assert result["ok"] is True
    assert result["city"] == "Tokyo"
    assert [day["date"] for day in result["days"]] == [
        (start + timedelta(days=offset)).isoformat() for offset in range(3)
    ]
    assert all(day["provenance"]["status"] == "mock" for day in result["days"])


def test_get_weather_rejects_out_of_range_days() -> None:
    result = _invoke(
        "get_weather",
        {"city": "Tokyo", "start_date": "2026-04-01", "days": 0},
    )

    assert result["ok"] is False
    assert result["error_code"] == "invalid_arguments"


def test_unexpected_source_failure_is_retryable() -> None:
    result = _invoke(
        "search_attractions",
        {"city": "Tokyo"},
        attraction_service=_BrokenAttractionService(),
    )

    assert result["ok"] is False
    assert result["error_code"] == "tool_internal_error"
    assert result["retryable"] is True
