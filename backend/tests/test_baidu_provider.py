"""百度 Place Provider 测试：用 httpx.MockTransport，不发真实请求。"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from backend.errors import ProviderUnavailableError
from backend.providers.baidu_provider import BaiduAttractionProvider
from backend.schemas.attraction import AttractionCategory

_ITEM = {
    "uid": "abc123",
    "name": "青秀山风景区",
    "location": {"lat": 22.795, "lng": 108.393},
    "detail_info": {
        "overall_rating": "4.6",
        "comment_num": "1200",
        "tag": "旅游景点;公园",
        "shop_hours": "08:00-18:00",
        "price": "20",
    },
}


def _provider(handler: Any) -> BaiduAttractionProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BaiduAttractionProvider(ak="test-ak", client=client)


def _ok(data: dict) -> httpx.Response:
    return httpx.Response(200, json={"status": 0, "message": "ok", **data})


def test_search_maps_fields_and_deduplicates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        query = request.url.params.get("query")
        if query == "景点":
            return _ok({"results": [_ITEM, {**_ITEM, "uid": "dup", "name": "重复景点"}]})
        # 第二个关键词返回和第一页相同的 uid，验证去重
        return _ok({"results": [{**_ITEM, "name": "别名重复"}, {**_ITEM, "uid": "xyz789", "name": "南湖公园"}]})

    provider = _provider(handler)
    attractions = asyncio.run(
        provider.search_by_keywords("南宁", keywords=["景点", "公园"], limit=10)
    )

    assert [item.id for item in attractions] == ["baidu:abc123", "baidu:dup", "baidu:xyz789"]
    assert attractions[0].name == "青秀山风景区"
    assert attractions[0].city == "南宁"
    assert attractions[0].category is AttractionCategory.NATURE
    assert attractions[0].tags == ["旅游景点", "公园"]
    assert attractions[0].rating == 4.6
    assert attractions[0].review_count == 1200
    assert len(attractions[0].opening_hours) == 7
    assert attractions[0].ticket_price is not None
    assert str(attractions[0].ticket_price.amount) == "20"
    assert attractions[0].ticket_price.currency == "CNY"
    assert attractions[0].provenance.source == "baidu:place_api"
    assert attractions[0].provenance.status.value == "real"
    assert {r.url.params.get("query") for r in requests} == {"景点", "公园"}


def test_search_maps_interests_to_keywords() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("query", ""))
        return _ok({"results": []})

    provider = _provider(handler)
    asyncio.run(
        provider.search_attractions("南宁", interests=["food", "nature"], limit=5)
    )

    assert seen == ["景点", "美食", "小吃", "公园", "自然风光"]


def test_search_skips_items_without_location() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(
            {
                "results": [
                    {**_ITEM, "location": {}},
                    {**_ITEM, "uid": "ok-id"},
                ]
            }
        )

    provider = _provider(handler)
    attractions = asyncio.run(provider.search_attractions("南宁", limit=10))

    assert [item.id for item in attractions] == ["baidu:ok-id"]


def test_all_day_hours_become_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        item = {
            **_ITEM,
            "detail_info": {**_ITEM["detail_info"], "shop_hours": "00:00-24:00"},
        }
        return _ok({"results": [item]})

    provider = _provider(handler)
    attractions = asyncio.run(provider.search_attractions("南宁", limit=5))

    assert attractions[0].opening_hours == []


def test_non_zero_status_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"status": 5, "message": "AK参数不存在"}).encode(),
            headers={"content-type": "application/json"},
        )

    provider = _provider(handler)

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(provider.search_attractions("南宁", limit=5))


def test_get_attraction_returns_none_when_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok({"result": None})

    provider = _provider(handler)
    assert asyncio.run(provider.get_attraction("baidu:missing")) is None


def test_get_attraction_parses_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("uid") == "abc123"
        return _ok({"result": {**_ITEM, "city": "南宁市"}})

    provider = _provider(handler)
    attraction = asyncio.run(provider.get_attraction("baidu:abc123"))

    assert attraction is not None
    assert attraction.id == "baidu:abc123"
    assert attraction.city == "南宁市"
