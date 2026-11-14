r"""真实 Google Places API 验证（会联网，默认跳过）。

目标：只用 Google Places API（New）的 Text Search 查询「中国广西南宁」的景点，
把请求、原始响应和整理后的字段打印到终端，用来判断这个数据源在国内的可用性。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_GOOGLE_PLACES_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_google_places_live.py -s -q

前置条件：
    1. .env 里配置 GOOGLE_MAPS_API_KEY；
    2. 该 Key 已启用 Places API (New)，并绑定可用的结算账号；
    3. 能访问 places.googleapis.com（中国大陆网络通常需要代理）。
"""

import asyncio
import json
import os

import httpx
import pytest

from backend.config import get_settings

SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

# 南宁市中心（南湖一带），半径 50km 覆盖主城区
NANNING_CENTER = {"latitude": 22.8170, "longitude": 108.3665}
SEARCH_RADIUS_METERS = 50_000.0
MAX_RESULTS = 20
QUERY = "南宁 景点"

# 字段掩码决定计费档位，也决定返回内容；缺字段会被网关直接忽略
FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.types",
        "places.primaryType",
        "places.primaryTypeDisplayName",
        "places.regularOpeningHours",
        "places.priceLevel",
        "places.editorialSummary",
        "places.googleMapsUri",
        "places.businessStatus",
    ]
)


def _build_payload() -> dict:
    """Text Search 请求体。locationBias 只影响排序，不限制结果必须在圆内。"""
    return {
        "textQuery": QUERY,
        "languageCode": "zh-CN",
        "regionCode": "CN",
        "maxResultCount": MAX_RESULTS,
        "locationBias": {
            "circle": {
                "center": NANNING_CENTER,
                "radius": SEARCH_RADIUS_METERS,
            }
        },
    }


async def _search(api_key: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0, trust_env=True) as client:
        return await client.post(
            SEARCH_TEXT_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": FIELD_MASK,
            },
            json=_build_payload(),
        )


def _text(value: dict | None) -> str:
    """Places API 的本地化文本都是 {"text": ..., "languageCode": ...}。"""
    if not value:
        return "无"
    return value.get("text") or "无"


def _print_places(places: list[dict]) -> None:
    for index, place in enumerate(places, 1):
        location = place.get("location") or {}
        opening = place.get("regularOpeningHours") or {}
        weekday_text = opening.get("weekdayDescriptions") or []
        rating = place.get("rating")
        reviews = place.get("userRatingCount")

        print(f"\n[{index}] {_text(place.get('displayName'))}")
        print(f"    id          : {place.get('id')}")
        print(f"    地址        : {place.get('formattedAddress') or '无'}")
        print(
            "    坐标        : "
            f"{location.get('latitude')}, {location.get('longitude')}"
        )
        print(
            "    评分        : "
            + (f"{rating}（{reviews} 条评价）" if rating is not None else "无")
        )
        print(f"    主类型      : {_text(place.get('primaryTypeDisplayName'))}")
        print(f"    全部类型    : {', '.join(place.get('types') or []) or '无'}")
        print(f"    价格档位    : {place.get('priceLevel') or '无'}")
        print(f"    营业状态    : {place.get('businessStatus') or '无'}")
        print(
            "    开放时间    : "
            + (" / ".join(weekday_text) if weekday_text else "无")
        )
        print(f"    简介        : {_text(place.get('editorialSummary'))}")
        print(f"    Google Maps : {place.get('googleMapsUri') or '无'}")


def test_google_places_nanning_attractions() -> None:
    if not os.getenv("RUN_GOOGLE_PLACES_TESTS"):
        pytest.skip("真实 Places 测试默认跳过，设置 RUN_GOOGLE_PLACES_TESTS=1 后运行")

    settings = get_settings()
    if not settings.google_maps_api_key:
        pytest.skip("未配置 GOOGLE_MAPS_API_KEY")

    payload = _build_payload()
    print("\n--- 请求 ---")
    print(f"POST {SEARCH_TEXT_URL}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        response = asyncio.run(_search(settings.google_maps_api_key))
    except httpx.RequestError as exc:
        pytest.fail(
            f"无法连接 places.googleapis.com：{type(exc).__name__}: {exc}\n"
            "中国大陆网络通常需要代理（HTTP_PROXY / HTTPS_PROXY）。"
        )

    print(f"\n--- HTTP {response.status_code} ---")
    if response.status_code != 200:
        print(response.text)
        pytest.fail(
            f"Google Places API 返回 {response.status_code}；"
            "403 多为 Key 未启用 Places API (New) / 未绑结算 / 受限，"
            "400 多为字段掩码不合法。"
        )

    body = response.json()
    places = body.get("places") or []
    print(f"共返回 {len(places)} 条景点")

    if not places:
        print("\n--- 原始响应 ---")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print(
            "\n注意：Google Places 在中国大陆的 POI 覆盖非常稀疏，"
            "返回 0 条属于预期结果之一。"
        )
        return

    _print_places(places)

    print("\n--- 原始 JSON（前 2 条）---")
    print(json.dumps(places[:2], ensure_ascii=False, indent=2))
