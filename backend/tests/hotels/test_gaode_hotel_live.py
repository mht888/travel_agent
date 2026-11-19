r"""高德 MCP · 酒店查询验证（会消耗配额，默认跳过）。

验证统一的 gaode_mcp 工具三条主路径：
    1. around_search + poi_kind=hotel：周边找酒店，返回坐标/评分/人均/距离；
    2. text_search + poi_kind=hotel：按关键词找酒店；
    3. poi_detail：按 POI ID 查完整详情。

注意：走的是高德官方 MCP Server（Streamable HTTP），
.env 里的 GAODE_MAPS_AK 必须是「Web服务」类型 Key。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_GAODE_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/hotels/test_gaode_hotel_live.py -s -q

前置条件：
    1. .env 里配置 GAODE_MAPS_AK；
    2. 账户有 Web 服务 API 配额。
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from backend.tools.gaode_mcp_tools import (
    close_gaode_mcp_client,
    create_gaode_mcp_tool,
)

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
CITY = "南宁市"
TARGET_HOTEL = "南宁饭店"
# 南宁市中心（金湖广场附近），GCJ-02
CENTER_LNG = 108.366543
CENTER_LAT = 22.817999


def _mask(key: str) -> str:
    return f"{key[:6]}****{key[-4:]}" if len(key) > 12 else "****"


def _load(payload: str) -> dict:
    data = json.loads(payload)
    print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])
    return data


async def _run() -> None:
    tool = create_gaode_mcp_tool()
    try:
        print("\n=== around_search · poi_kind=hotel ===")
        around = _load(
            await tool.ainvoke(
                {
                    "action": "around_search",
                    "poi_kind": "hotel",
                    "longitude": CENTER_LNG,
                    "latitude": CENTER_LAT,
                    "radius": 2000,
                    "limit": 10,
                }
            )
        )
        assert around.get("ok"), around
        assert around.get("action") == "around_search"
        assert around.get("poi_kind") == "hotel"
        items = around.get("items") or []
        assert items, "周边搜索没有召回酒店"
        for item in items:
            assert item.get("id") and item.get("name")
            assert item.get("location"), f"{item.get('name')} 缺少坐标"
            assert item.get("distance_meters") is not None
        print(f"召回 {len(items)} 家酒店，距离最近：{items[0].get('name')}")

        print(f"\n=== text_search · poi_kind=hotel ===")
        text = _load(
            await tool.ainvoke(
                {
                    "action": "text_search",
                    "poi_kind": "hotel",
                    "keywords": TARGET_HOTEL,
                    "city": CITY,
                    "city_limit": True,
                    "limit": 5,
                }
            )
        )
        assert text.get("ok"), text
        text_items = text.get("items") or []
        assert text_items, f"关键词搜索没有找到「{TARGET_HOTEL}」"
        matched = next(
            (item for item in text_items if item.get("name") == TARGET_HOTEL),
            text_items[0],
        )

        print(f"\n=== poi_detail：{matched['id']} ===")
        detail = _load(
            await tool.ainvoke(
                {"action": "poi_detail", "poi_id": matched["id"]}
            )
        )
        assert detail.get("ok"), detail
        assert isinstance(detail.get("data"), dict) and detail["data"], "详情为空"
    finally:
        await close_gaode_mcp_client()


def test_gaode_hotel_search() -> None:
    if not os.getenv("RUN_GAODE_TESTS"):
        pytest.skip("真实高德测试默认跳过，设置 RUN_GAODE_TESTS=1 后运行")

    load_dotenv(ENV_PATH)
    key = os.getenv("GAODE_MAPS_AK") or os.getenv("gaode_maps_ak")
    if not key:
        pytest.skip("未配置 GAODE_MAPS_AK")

    print(f"Key: {_mask(key)}")
    asyncio.run(_run())
