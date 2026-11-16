r"""高德 POI 2.0 · 酒店查询验证（会消耗配额，默认跳过）。

验证两条路径：
    1. 关键字搜索：GET https://restapi.amap.com/v5/place/text
       适合「南宁有哪些酒店」这类泛搜；
    2. 输入提示 + POI 详情：
       GET https://restapi.amap.com/v3/assistant/inputtips
       GET https://restapi.amap.com/v5/place/detail
       适合「南宁饭店」这类精确名称查询——关键字搜索按综合权重排序，
       不一定把目标酒店排在前面，必须先拿到 poi_id 再查详情。

注意：这是「Web 服务 API」，.env 里的 GAODE_MAPS_AK 必须是「Web服务」类型 Key。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_GAODE_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_gaode_hotel_live.py -s -q

前置条件：
    1. .env 里配置 GAODE_MAPS_AK；
    2. 账户有 Web 服务 API 配额。
"""

import asyncio
import json
import os
from pathlib import Path

import httpx2
import pytest
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
CITY = "南宁市"
CITY_SHORT = "南宁"

TEXT_ENDPOINT = "https://restapi.amap.com/v5/place/text"
TIPS_ENDPOINT = "https://restapi.amap.com/v3/assistant/inputtips"
DETAIL_ENDPOINT = "https://restapi.amap.com/v5/place/detail"

# 100000 = 住宿服务；100100 = 宾馆酒店；100101~100105 = 星级/经济型；100200 = 民宿
HOTEL_TYPES = "100000"
TARGET_HOTEL = "南宁饭店"


def _mask(key: str) -> str:
    return f"{key[:6]}****{key[-4:]}" if len(key) > 12 else "****"


def _print_poi(index: int, poi: dict) -> None:
    business = poi.get("business") or {}
    photos = poi.get("photos") or []
    address = f"{poi.get('pname', '')}{poi.get('cityname', '')}{poi.get('adname', '')}{poi.get('address', '')}"
    print(f"\n[{index}] {poi.get('name')}  (id={poi.get('id')})")
    print(f"    类型: {poi.get('type')} / {poi.get('typecode')}")
    print(f"    地址: {address}")
    print(f"    坐标: {poi.get('location')}")
    print(
        "    商圈: {area}  评分: {rating}  人均: {cost}".format(
            area=business.get("business_area") or "-",
            rating=business.get("rating") or "-",
            cost=business.get("cost") or "-",
        )
    )
    print(f"    电话: {business.get('tel') or '-'}  营业: {business.get('opentime_today') or '-'}")
    print(f"    图片: {len(photos)} 张")


async def _get_json(client: httpx2.AsyncClient, url: str, params: dict) -> dict:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _search_text(client: httpx2.AsyncClient, key: str) -> dict:
    """泛搜：南宁的住宿类 POI。"""
    params = {
        "key": key,
        "keywords": "酒店",
        "types": HOTEL_TYPES,
        "region": CITY,
        "city_limit": "true",
        "show_fields": "business,photos",
        "page_size": 25,
        "page_num": 1,
        "output": "json",
    }
    print(f"\n=== 关键字搜索：{CITY} 住宿类 POI ===")
    print(f"GET {TEXT_ENDPOINT}")
    payload = await _get_json(client, TEXT_ENDPOINT, params)
    print(
        "status={status} info={info} infocode={infocode} count={count}".format(
            status=payload.get("status"),
            info=payload.get("info"),
            infocode=payload.get("infocode"),
            count=payload.get("count"),
        )
    )
    if payload.get("status") != "1":
        print(f"原始返回: {json.dumps(payload, ensure_ascii=False)}")
        return payload

    pois = payload.get("pois") or []
    for index, poi in enumerate(pois[:5], 1):
        _print_poi(index, poi)
    if len(pois) > 5:
        print(f"\n... 本次返回 {len(pois)} 条，仅展示前 5 条（接口单页上限 25、总量上限 200）")
    return payload


async def _resolve_hotel(client: httpx2.AsyncClient, key: str) -> str:
    """输入提示：把酒店名解析成 poi_id。"""
    params = {
        "key": key,
        "keywords": TARGET_HOTEL,
        "city": CITY_SHORT,
        "citylimit": "true",
        "datatype": "poi",
        # 精确名称查询不要带 type/type=100000：
        # 实测会按类型权重过滤，反而把「南宁饭店」本身挤掉，只返回周边停车场/餐厅。
        "output": "json",
    }
    print(f"\n=== 输入提示：{TARGET_HOTEL} ===")
    print(f"GET {TIPS_ENDPOINT}")
    payload = await _get_json(client, TIPS_ENDPOINT, params)
    print(
        "status={status} info={info} infocode={infocode} count={count}".format(
            status=payload.get("status"),
            info=payload.get("info"),
            infocode=payload.get("infocode"),
            count=payload.get("count"),
        )
    )
    assert payload.get("status") == "1", payload.get("info")

    tips = [
        tip
        for tip in (payload.get("tips") or [])
        if isinstance(tip.get("id"), str) and tip.get("id")
    ]
    for index, tip in enumerate(tips[:5], 1):
        print(
            f"[{index}] {tip.get('name')}  id={tip.get('id')}  "
            f"区县={tip.get('district')}  地址={tip.get('address')}"
        )

    matched = next((tip for tip in tips if (tip.get("name") or "") == TARGET_HOTEL), None)
    if matched is None:
        matched = next((tip for tip in tips if TARGET_HOTEL in (tip.get("name") or "")), None)
    assert matched is not None, f"输入提示没有找到「{TARGET_HOTEL}」"
    return str(matched["id"])


async def _detail(client: httpx2.AsyncClient, key: str, poi_id: str) -> dict:
    """POI 详情：按 poi_id 拿完整字段。"""
    params = {
        "key": key,
        "id": poi_id,
        "show_fields": "business,photos",
        "output": "json",
    }
    print(f"\n=== POI 详情：{poi_id} ===")
    print(f"GET {DETAIL_ENDPOINT}")
    payload = await _get_json(client, DETAIL_ENDPOINT, params)
    print(
        "status={status} info={info} infocode={infocode} count={count}".format(
            status=payload.get("status"),
            info=payload.get("info"),
            infocode=payload.get("infocode"),
            count=payload.get("count"),
        )
    )
    assert payload.get("status") == "1", payload.get("info")

    pois = payload.get("pois") or []
    assert pois, "详情接口没有返回 POI"
    _print_poi(1, pois[0])
    print("\n--- 原始 JSON ---")
    print(json.dumps(pois[0], ensure_ascii=False, indent=2))
    return payload


async def _run(key: str) -> None:
    print(f"Key: {_mask(key)}")
    async with httpx2.AsyncClient(timeout=15.0) as client:
        search_payload = await _search_text(client, key)
        assert search_payload.get("status") == "1", search_payload.get("info")
        assert int(search_payload.get("count") or 0) > 0, "泛搜没有召回酒店"

        poi_id = await _resolve_hotel(client, key)
        await _detail(client, key, poi_id)


def test_gaode_hotel_search() -> None:
    if not os.getenv("RUN_GAODE_TESTS"):
        pytest.skip("真实高德测试默认跳过，设置 RUN_GAODE_TESTS=1 后运行")

    load_dotenv(ENV_PATH)
    key = os.getenv("GAODE_MAPS_AK") or os.getenv("gaode_maps_ak")
    if not key:
        pytest.skip("未配置 GAODE_MAPS_AK")

    asyncio.run(_run(key))
