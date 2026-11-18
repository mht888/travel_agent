r"""高德 MCP · 南宁 ↔ 桂林 城际交通导航验证（会消耗配额，默认跳过）。

覆盖 gaode_mcp 的三个 action：
    1. geocode：把「南宁市 / 桂林市」转成坐标；
    2. route + route_mode=driving：城际驾车路线（距离、耗时、主要道路）；
    3. route + route_mode=transit：跨城公共交通（地铁/公交 + 城际火车）。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_GAODE_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_gaode_transport_live.py -s -q

前置条件：
    .env 里配置 GAODE_MAPS_AK（Web 服务 Key），且账户有配额。
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from backend.tools.gaode_mcp_tools import (
    close_gaode_mcp_client,
    create_gaode_mcp_tool,
)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
CITY_FROM = "南宁"
CITY_TO = "桂林"


# ---------------------------- 格式化辅助 ----------------------------


def _to_int(value: Any) -> int | None:
    """高德把距离/耗时都返回成字符串，空值也不少。"""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _km(meters: Any) -> str:
    value = _to_int(meters)
    return f"{value / 1000:.1f} km" if value is not None else "未知"


def _duration(seconds: Any) -> str:
    value = _to_int(seconds)
    if value is None:
        return "未知"
    hours, minutes = divmod(value // 60, 60)
    if hours and minutes:
        return f"{hours} 小时 {minutes} 分钟"
    if hours:
        return f"{hours} 小时"
    return f"{minutes} 分钟"


def _coord(result: dict[str, Any]) -> tuple[str, str]:
    """geocode 返回 {"data": {"results": [{"location": "lng,lat"}]}}。"""
    results = (result.get("data") or {}).get("results") or []
    assert results, f"geocode 没有返回结果：{result}"
    location = results[0].get("location")
    assert isinstance(location, str) and "," in location, f"geocode 坐标异常：{location}"
    return tuple(location.split(",", 1))  # type: ignore[return-value]


# ---------------------------- 打印辅助 ----------------------------


def _print_driving(payload: dict[str, Any]) -> None:
    data = payload.get("data") or {}
    paths = data.get("paths") or []
    assert paths, f"驾车路线为空：{payload}"
    path = paths[0]

    roads: list[str] = []
    for step in path.get("steps") or []:
        road = (step.get("road") or "").strip()
        if road and road not in roads:
            roads.append(road)

    print("\n【驾车】")
    print(f"  全程：{_km(path.get('distance'))}，预计 {_duration(path.get('duration'))}")
    print(f"  主要道路：{' → '.join(roads[:8]) or '未返回'}")
    print(f"  导航步骤：{len(path.get('steps') or [])} 段")
    for step in (path.get("steps") or [])[:3]:
        print(f"    · {step.get('instruction') or '（无描述）'}")


def _leg_labels(transit: dict[str, Any]) -> list[str]:
    """把一段公交方案拆成「步行 → 地铁/公交 → 火车 → ...」。"""
    labels: list[str] = []
    for segment in transit.get("segments") or []:
        walking = segment.get("walking") or {}
        walk_distance = _to_int(walking.get("distance"))
        if walk_distance and walk_distance > 0:
            labels.append(f"步行 {walk_distance} 米")

        for line in (segment.get("bus") or {}).get("buslines") or []:
            start = (line.get("departure_stop") or {}).get("name") or "?"
            end = (line.get("arrival_stop") or {}).get("name") or "?"
            labels.append(f"{line.get('name') or '公交'}（{start} → {end}）")

        railway = segment.get("railway") or {}
        if railway.get("name"):
            trip = railway.get("trip") or ""
            suffix = f"，车次 {trip}" if trip and trip not in railway["name"] else ""
            labels.append(f"城际火车 {railway['name']}{suffix}")
    return labels


def _print_transit(payload: dict[str, Any]) -> None:
    data = payload.get("data") or {}
    transits = data.get("transits") or []
    assert transits, f"跨城公交方案为空：{payload}"

    print("\n【公共交通 / 火车】")
    print(f"  参考里程：{_km(data.get('distance'))}")
    for index, transit in enumerate(transits[:3], start=1):
        labels = _leg_labels(transit)
        transfers = sum(1 for label in labels if "→" in label or "火车" in label)
        print(
            f"  方案 {index}：{_duration(transit.get('duration'))}，"
            f"约 {max(transfers - 1, 0)} 次换乘"
        )
        for label in labels:
            print(f"    · {label}")


# ---------------------------- 测试主体 ----------------------------


async def _run() -> None:
    tool = create_gaode_mcp_tool()
    try:
        print("\n=== 1. geocode：南宁 / 桂林 ===")
        from_lng, from_lat = _coord(
            json.loads(await tool.ainvoke({"action": "geocode", "address": "南宁市", "city": CITY_FROM}))
        )
        to_lng, to_lat = _coord(
            json.loads(await tool.ainvoke({"action": "geocode", "address": "桂林市", "city": CITY_TO}))
        )
        print(f"  南宁市中心：{from_lng},{from_lat}")
        print(f"  桂林市中心：{to_lng},{to_lat}")

        origin = f"{from_lng},{from_lat}"
        destination = f"{to_lng},{to_lat}"

        print("\n=== 2. distance：驾车距离测量 ===")
        distance = json.loads(
            await tool.ainvoke(
                {
                    "action": "distance",
                    "origin": origin,
                    "destination": destination,
                    "distance_mode": "driving",
                }
            )
        )
        print(json.dumps(distance.get("data"), ensure_ascii=False, indent=2))
        assert distance.get("ok"), distance
        # maps_distance 返回 {"results": [{"distance": "...", "duration": "..."}]}
        measured_rows = (distance.get("data") or {}).get("results") or []
        assert measured_rows, f"距离测量没有返回结果：{distance}"
        measured = _to_int(measured_rows[0].get("distance"))
        assert measured is not None and measured > 300_000, "两地驾车距离异常（应大于 300km）"
        print(f"  → 驾车约 {_km(measured)}，{_duration(measured_rows[0].get('duration'))}")

        print("\n=== 3. route：城际驾车导航 ===")
        driving = json.loads(
            await tool.ainvoke(
                {
                    "action": "route",
                    "route_mode": "driving",
                    "origin": origin,
                    "destination": destination,
                }
            )
        )
        assert driving.get("ok"), driving
        driving_path = ((driving.get("data") or {}).get("paths") or [{}])[0]
        assert (_to_int(driving_path.get("distance")) or 0) > 300_000, "驾车路线距离异常"
        assert (_to_int(driving_path.get("duration")) or 0) > 0, "驾车路线耗时异常"
        _print_driving(driving)

        print("\n=== 4. route：跨城公共交通（地铁/公交 + 火车）===")
        transit = json.loads(
            await tool.ainvoke(
                {
                    "action": "route",
                    "route_mode": "transit",
                    "origin": origin,
                    "destination": destination,
                    "city": CITY_FROM,
                    "destination_city": CITY_TO,
                }
            )
        )
        assert transit.get("ok"), transit
        transits = (transit.get("data") or {}).get("transits") or []
        assert transits, "没有返回跨城公共交通方案"
        has_railway = any(
            (segment.get("railway") or {}).get("name")
            for item in transits
            for segment in item.get("segments") or []
        )
        assert has_railway, "公共交通方案里没有城际火车段"
        _print_transit(transit)

        print("\n✅ 南宁 → 桂林 交通导航验证通过")
    finally:
        await close_gaode_mcp_client()


def test_nanning_to_guilin_transport() -> None:
    if not os.getenv("RUN_GAODE_TESTS"):
        pytest.skip("真实高德测试默认跳过，设置 RUN_GAODE_TESTS=1 后运行")

    load_dotenv(ENV_PATH)
    if not os.getenv("GAODE_MAPS_AK") and not os.getenv("gaode_maps_ak"):
        pytest.skip("未配置 GAODE_MAPS_AK")

    asyncio.run(_run())
