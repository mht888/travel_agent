"""网页地点初选链路的离线测试。

不联网、不调用 LLM：
    - 博查用 _FakeBochaCaller 返回固定网页块；
    - 抽取用 _StubExtractor，按兴趣返回固定地点；
    - 排序用 _StubWebRanker，按 id 稳定排序。

覆盖：
    mcp_web_search → extract_web_places → rank_attraction_pool
    以及工作流 finalize 从 rank 结果收口；
    「每个兴趣不足 10 个 → Agent 补检索 → 重新抽取」的循环。
"""

import asyncio
import json
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable

from backend.cache.pool_cache import AttractionPoolCache, WebSearchResultCache
from backend.graph.selection_workflow import build_selection_graph
from backend.schemas.attraction import AttractionCategory
from backend.schemas.ranking import AttractionRanking, RankingItem
from backend.schemas.web_place import ExtractedPlace
from backend.tools.bocha_mcp_tools import create_bocha_mcp_tools
from backend.tools.rank_attraction_pool_tools import (
    create_rank_attraction_pool_tool,
)
from backend.tools.web_place_extract_tools import (
    build_extract_web_places_tool,
    make_place_id,
)

CITY = "南宁"
INTERESTS = ["美食", "自然"]
TOP_K = 3
PER_INTEREST_LIMIT = 15


def test_attraction_assembler_batches_llm_matches_and_checks_candidate_ids() -> None:
    from unittest.mock import AsyncMock

    from backend.schemas.common import GeoPoint
    from backend.services.attraction_assembly import AttractionAssembler

    async def call_tool(name, arguments):
        if name == "maps_search_detail":
            assert arguments == {"id": "relics"}
            return {"id": "relics", "location": "108.335445,22.811786"}
        candidates = {
            "广西民族文物苑": [{"id": "relics", "name": "民族文物苑", "typecode": "110200"}],
            "青秀山": [{"id": "scenic", "name": "青秀山风景区", "typecode": "110202"}],
            "同名地点": [{"id": "ambiguous", "name": "同名地点"}],
        }
        return {"pois": candidates.get(arguments["keywords"], [])}

    caller = AsyncMock()
    caller.call_tool.side_effect = call_tool
    matcher = AsyncMock()
    matcher.ainvoke.return_value = {
        "matches": [
            {"attraction_id": "a", "poi_id": "relics", "place_kind": "attraction", "reason": "省名前缀与简称"},
            {"attraction_id": "b", "poi_id": "relics", "place_kind": "attraction", "reason": "其他景点候选"},
            {"attraction_id": "c", "poi_id": None, "place_kind": "unknown", "reason": "无法区分"},
        ]
    }
    items = [
        RankingItem(id="a", name="广西民族文物苑", score=1),
        RankingItem(id="b", name="青秀山公园", score=0.9),
        RankingItem(id="c", name="同名地点", score=0.8),
    ]

    assembled = asyncio.run(
        AttractionAssembler(matcher, caller=caller)(city="南宁", items=items)
    )
    attractions, warnings = assembled.attractions, assembled.warnings

    matcher.ainvoke.assert_awaited_once()
    assert attractions[0].location == GeoPoint(lng=108.335445, lat=22.811786)
    assert [item.location for item in attractions[1:]] == [None, None]
    assert any("不在该景点的候选列表" in warning for warning in warnings)
    assert any("模型未确认同一地点" in warning for warning in warnings)
    details = [call.args[1] for call in caller.call_tool.await_args_list if call.args[0] == "maps_search_detail"]
    assert details == [{"id": "relics"}]


def test_web_extractor_reuses_source_without_model_generated_description() -> None:
    from unittest.mock import AsyncMock

    from backend.tools.web_place_extract_tools import LLMWebPlaceExtractor

    model = AsyncMock()
    model.ainvoke.return_value = {"items": [
        {"name": "青秀山", "category": "nature", "source_index": 2},
        {"name": "其他景点", "category": "nature", "source_index": 100},
    ]}
    places = asyncio.run(LLMWebPlaceExtractor(model).extract(
        city="南宁", interest="自然", limit=5,
        results=[
            {"url": "https://example.com/1", "snippet": "无关内容"},
            {"url": "https://example.com/2", "snippet": "青秀山有植物园与步道。"},
        ],
    ))
    assert [(place.name, place.description, place.source_url) for place in places] == [
        ("青秀山", "青秀山有植物园与步道。", "https://example.com/2"),
    ]


def test_poi_timeout_retries_only_affected_batch() -> None:
    from unittest.mock import AsyncMock

    from backend.services.attraction_assembly import AttractionAssembler

    async def call_tool(name, arguments):
        if name == "maps_text_search":
            return {"pois": [{"id": arguments["keywords"], "name": arguments["keywords"]}]}
        return {"location": "104.06,30.66"}

    failed_batches = []

    async def match(messages):
        attractions = json.loads(messages[-1].content)["attractions"]
        if len(attractions) == 1:
            failed_batches.append(attractions[0]["attraction_id"])
            raise TimeoutError
        return {"matches": [
            {"attraction_id": item["attraction_id"], "poi_id": item["name"], "place_kind": "attraction"}
            for item in attractions
        ]}

    items = [RankingItem(id=str(index), name=f"景点{index}", score=1) for index in range(8)]
    matcher = AsyncMock()
    matcher.ainvoke.side_effect = match
    caller = AsyncMock()
    caller.call_tool.side_effect = call_tool
    result = asyncio.run(AttractionAssembler(matcher, caller=caller)(city="成都", items=items))

    assert failed_batches == ["7", "7"]
    assert [item.id for item in result.attractions if item.location] == [str(i) for i in range(7)]
    assert any("第 2 批失败" in warning for warning in result.warnings)


def test_attraction_assembler_uses_city_code_and_skips_closed_poi() -> None:
    from unittest.mock import AsyncMock

    from backend.services.attraction_assembly import AttractionAssembler
    from backend.tools.gaode_mcp_tools import GaodeMcpClient

    async def call_tool(name, arguments):
        if name == "maps_geo":
            assert arguments == {"address": "Chengdu"}
            return {"results": [{"adcode": "510100", "level": "市"}]}
        if name == "maps_text_search":
            assert arguments["city"] == "510100"
            keyword = arguments["keywords"]
            return {"pois": [{
                "id": keyword, "name": f"{keyword}(暂停开放)" if keyword == "金沙遗址博物馆" else keyword,
            }]}
        details = {
            "金沙遗址博物馆": {},
            "成都博物馆": {"name": "成都博物馆(临时闭馆)"},
            "自然博物馆": {"business_status": "暂停开放", "reopen_date": "2026-11-20"},
        }
        return {"location": "104.012,30.681", **details[arguments["id"]]}

    caller = GaodeMcpClient(api_key="test")
    remote = AsyncMock(side_effect=call_tool)
    caller._call_remote = remote
    matcher = AsyncMock()
    names = ["金沙遗址博物馆", "成都博物馆", "自然博物馆"]
    matcher.ainvoke.return_value = {"matches": [
        {"attraction_id": name, "poi_id": name, "place_kind": "attraction", "reason": "同一景点"}
        for name in names
    ]}
    assembled = asyncio.run(AttractionAssembler(matcher, caller=caller)(
        city="Chengdu",
        items=[RankingItem(id=name, name=name, score=1) for name in names],
        start_date=date(2026, 11, 20),
    ))
    attractions, warnings = assembled.attractions, assembled.warnings

    assert [item.name for item in attractions] == ["自然博物馆"]
    assert len(warnings) == 2
    assert all("已排除" in warning for warning in warnings)
    assert sum(call.args[0] == "maps_geo" for call in remote.await_args_list) == 1


def test_food_venues_route_to_meals_and_food_district_stays_in_sightseeing() -> None:
    from unittest.mock import AsyncMock

    from backend.services.attraction_assembly import AttractionAssembler

    async def call_tool(name, arguments):
        if name == "maps_search_detail":
            return {"id": arguments["id"], "location": "104.012,30.681"}
        keyword = arguments["keywords"]
        return {"pois": [{
            "id": keyword, "name": keyword,
            "typecode": "050101" if keyword == "烤匠麻辣烤鱼(环球中心店)" else "110202|050300",
        }]}

    caller = AsyncMock()
    caller.call_tool.side_effect = call_tool
    matcher = AsyncMock()
    items = [
        RankingItem(id="fish", name="烤匠麻辣烤鱼(环球中心店)", score=1, category="food"),
        RankingItem(id="street", name="锦里", score=1, category="food"),
    ]
    matcher.ainvoke.return_value = {"matches": [
        {"attraction_id": item.id, "poi_id": item.name, "place_kind": kind, "reason": "同一地点"}
        for item, kind in zip(items, ("attraction", "restaurant"))
    ]}

    assembled = asyncio.run(AttractionAssembler(matcher, caller=caller)(city="成都", items=items))

    assert [item.id for item in assembled.attractions] == ["street"]
    assert [item.name for item in assembled.restaurants] == [items[0].name]


# ---------------------------- 替身 ----------------------------


class _FakeBochaCaller:
    """按 query 返回两条网页结果，格式与博查 MCP 文本一致。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        query = arguments["query"]
        self.queries.append(query)
        slug = query.replace(" ", "-")
        return (
            f"Title: {query} 榜单\n"
            f"URL: https://example.com/{slug}/a\n"
            "Description: 中山路小吃街、青秀山、建政路夜市都值得去。\n\n"
            f"Title: {query} 相关攻略\n"
            f"URL: https://example.com/{slug}/b\n"
            "Description: 这篇攻略提到复记老友粉和南湖公园。"
        )


class _StubExtractor:
    """按兴趣返回固定数量的地点，记录每次抽取的入参。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def extract(
        self,
        *,
        city: str,
        interest: str,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[ExtractedPlace]:
        self.calls.append((interest, len(results)))
        count = 2 if interest == "美食" else 1
        category = (
            AttractionCategory.FOOD
            if interest == "美食"
            else AttractionCategory.NATURE
        )
        return [
            ExtractedPlace(
                id=f"web:{city}:{interest}:{index}",
                name=f"{city}{interest}地点{index}",
                description=f"{interest}候选 {index}",
                city=city,
                category=category,
                source_url=f"https://example.com/{interest}/{index}",
            )
            for index in range(count)
        ]


class _IdentityAliasResolver:
    """默认不归并，保持抽取结果原样。"""

    async def resolve(
        self,
        *,
        city: str,
        places: list[ExtractedPlace],
    ) -> dict[str, str]:
        return {}


class _MergingAliasResolver:
    """把指定的两个名称视为同一地点，其余保持独立。"""

    def __init__(self, alias_name: str, canonical_name: str) -> None:
        self._alias_name = alias_name
        self._canonical_name = canonical_name
        self.calls = 0

    async def resolve(
        self,
        *,
        city: str,
        places: list[ExtractedPlace],
    ) -> dict[str, str]:
        self.calls += 1
        by_name = {place.name: place.id for place in places}
        alias = by_name.get(self._alias_name)
        canonical = by_name.get(self._canonical_name)
        if alias and canonical and alias != canonical:
            return {alias: canonical}
        return {}


class _GrowingBochaCaller:
    """同一 query 按请求的 count 返回互不相同的网页块，模拟补检索。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        query = arguments["query"]
        count = int(arguments.get("count") or 5)
        self.calls.append((query, count))
        slug = query.replace(" ", "-")
        blocks = [
            (
                f"Title: {query} 第{index}篇\n"
                f"URL: https://example.com/{slug}/{index}\n"
                f"Description: 这是 {query} 的第 {index} 个地点线索。"
            )
            for index in range(count)
        ]
        return "\n\n".join(blocks)


class _ResultExtractor:
    """一条网页结果对应一个地点，名称由 URL 决定，便于验证跨轮去重。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def extract(
        self,
        *,
        city: str,
        interest: str,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[ExtractedPlace]:
        self.calls.append((interest, len(results)))
        category = (
            AttractionCategory.FOOD
            if interest == "美食"
            else AttractionCategory.NATURE
        )
        places: list[ExtractedPlace] = []
        for item in results[:limit]:
            suffix = str(item.get("url") or "").rsplit("/", 1)[-1]
            name = f"{city}{interest}地点{suffix}"
            places.append(
                ExtractedPlace(
                    id=make_place_id(city, name),
                    name=name,
                    description=f"{interest}候选 {suffix}",
                    city=city,
                    category=category,
                    source_url=str(item.get("url") or "") or None,
                )
            )
        return places


class _StubWebRanker:
    """按 id 排序，验证排序工具确实消费了轻量候选池。"""

    def __init__(self) -> None:
        self.seen = 0

    async def rank(
        self,
        *,
        city: str,
        interests: list[str],
        places: list[ExtractedPlace],
        top_k: int,
    ) -> AttractionRanking:
        self.seen = len(places)
        ordered = sorted(places, key=lambda place: place.id)[:top_k]
        items = [
            RankingItem(
                id=place.id,
                score=max(0.0, 1.0 - index * 0.1),
                reason=f"命中兴趣：{'、'.join(interests)}",
            )
            for index, place in enumerate(ordered)
        ]
        return AttractionRanking(
            ranked_ids=[item.id for item in items],
            items=items,
        )


class _ScriptedAgent(Runnable):
    """按 网页搜索 → 抽取 → 排序 → 总结 的顺序发调用。"""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        return asyncio.run(self.ainvoke(*args, **kwargs))

    async def ainvoke(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "mcp_web_search",
                        "args": {"query": "南宁 美食 景点 推荐"},
                        "id": "call_web_food",
                    },
                    {
                        "name": "mcp_web_search",
                        "args": {"query": "南宁 自然 景点 推荐"},
                        "id": "call_web_nature",
                    },
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "extract_web_places",
                        "args": {
                            "city": CITY,
                            "interests": INTERESTS,
                            "limit_per_interest": PER_INTEREST_LIMIT,
                        },
                        "id": "call_extract",
                    }
                ],
            )
        if self.calls == 3:
            search_id = None
            for message in reversed(messages):
                if (
                    isinstance(message, ToolMessage)
                    and message.name == "extract_web_places"
                ):
                    search_id = json.loads(message.content).get("search_id")
                    break
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rank_attraction_pool",
                        "args": {
                            "search_id": search_id,
                            "interests": INTERESTS,
                            "top_k": TOP_K,
                        },
                        "id": "call_rank",
                    }
                ],
            )
        return AIMessage(content="已完成网页地点初选。")


# ---------------------------- 工具搭建 ----------------------------


def _build_tools(
    *,
    min_places_per_interest: int = 1,
    max_rounds: int = 4,
    caller: Any | None = None,
    extractor: Any | None = None,
    alias_resolver: Any | None = None,
) -> tuple[
    dict[str, Any], WebSearchResultCache, Any, _StubWebRanker
]:
    web_cache = WebSearchResultCache()
    pool_cache = AttractionPoolCache()
    extractor = extractor or _StubExtractor()
    web_ranker = _StubWebRanker()
    caller = caller or _FakeBochaCaller()
    alias_resolver = alias_resolver or _IdentityAliasResolver()

    tools = [
        *create_bocha_mcp_tools(
            caller,
            result_cache=web_cache,
            strict_query=True,
        ),
        build_extract_web_places_tool(
            pool_cache,
            web_cache,
            extractor=extractor,
            alias_resolver=alias_resolver,
            min_places_per_interest=min_places_per_interest,
            max_rounds=max_rounds,
        ),
        create_rank_attraction_pool_tool(
            pool_cache,
            web_place_ranker=web_ranker,
        ),
    ]
    return {tool.name: tool for tool in tools}, web_cache, extractor, web_ranker


def _call(tool: Any, **arguments: Any) -> dict[str, Any]:
    return json.loads(asyncio.run(tool.ainvoke(arguments)))


# ---------------------------- 测试 ----------------------------


def test_web_search_returns_web_search_id() -> None:
    tools, web_cache, _, _ = _build_tools()

    payload = _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐")

    assert payload["ok"] is True
    assert payload["count"] == 2
    assert payload["web_search_id"]
    # 返回给 Agent 的精简结果不含摘要，但缓存里必须保留，抽取工具要用
    assert "snippet" not in payload["results"][0]
    cached = web_cache.get(payload["web_search_id"])
    assert cached is not None
    assert cached[1][0]["snippet"]


def test_web_search_rejects_expanded_query_in_strict_mode() -> None:
    tools, web_cache, _, _ = _build_tools()

    rejected = _call(
        tools["mcp_web_search"],
        query="南宁 美食 必吃 老友粉 中山路夜市 攻略",
    )

    assert rejected["ok"] is False
    assert rejected["error_code"] == "invalid_query_format"
    assert rejected["retryable"] is True
    assert web_cache.all() == []

    accepted = _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐")
    assert accepted["ok"] is True
    assert accepted["query"] == "南宁 美食 景点 推荐"
    assert accepted["web_search_id"]


def test_extract_web_places_builds_lightweight_pool() -> None:
    tools, _, extractor, _ = _build_tools()
    _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐")
    _call(tools["mcp_web_search"], query="南宁 自然 景点 推荐")

    payload = _call(
        tools["extract_web_places"],
        city=CITY,
        interests=INTERESTS,
    )

    assert payload["ok"] is True
    assert payload["kind"] == "web_place"
    assert payload["search_id"]
    assert payload["total"] == 3
    assert payload["by_interest"] == {"美食": 2, "自然": 1}
    assert {item["name"] for item in payload["items"]} == {
        "南宁美食地点0",
        "南宁美食地点1",
        "南宁自然地点0",
    }
    # 每个兴趣各抽取一次，且只拿到带对应兴趣词的网页结果
    assert extractor.calls == [("美食", 2), ("自然", 2)]


def test_extract_merges_alias_places_before_counting() -> None:
    resolver = _MergingAliasResolver("南宁美食地点1", "南宁美食地点0")
    tools, _, _, _ = _build_tools(alias_resolver=resolver)
    _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐")
    _call(tools["mcp_web_search"], query="南宁 自然 景点 推荐")

    payload = _call(tools["extract_web_places"], city=CITY, interests=INTERESTS)

    assert payload["ok"] is True
    # 合并前 2 个，合并后 1 个；计数按合并后的结果
    assert payload["raw_counts"] == {"美食": 2, "自然": 1}
    assert payload["by_interest"] == {"美食": 1, "自然": 1}
    assert payload["total"] == 2

    # 代表 id 由代码按固定规则选取，这里只断言同组只剩一个
    assert len(payload["alias_merged"]) == 1
    alias, canonical = next(iter(payload["alias_merged"].items()))
    assert {alias, canonical} == {"南宁美食地点0", "南宁美食地点1"}
    names = {item["name"] for item in payload["items"]}
    assert canonical in names
    assert alias not in names

    # 每轮抽取后归并一次
    assert resolver.calls == 1


def test_extract_without_web_results_returns_error() -> None:
    tools, _, _, _ = _build_tools()

    payload = _call(
        tools["extract_web_places"],
        city=CITY,
        interests=INTERESTS,
    )

    assert payload["ok"] is False
    assert payload["error_code"] == "no_web_results"
    assert payload["retryable"] is True


def test_web_search_escalates_count_on_repeat() -> None:
    tools, _, _, _ = _build_tools(caller=_GrowingBochaCaller())

    first = _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐", count=5)
    second = _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐", count=5)
    third = _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐", count=5)

    # 首次至少 10 条；同一 query 重复检索时自动放大，给补检索留下新结果
    assert (first["count"], second["count"], third["count"]) == (10, 20, 40)
    assert (first["attempt"], second["attempt"], third["attempt"]) == (1, 2, 3)


def test_extract_requests_more_search_when_places_insufficient() -> None:
    caller = _GrowingBochaCaller()
    extractor = _ResultExtractor()
    tools, _, _, _ = _build_tools(
        min_places_per_interest=15,
        max_rounds=4,
        caller=caller,
        extractor=extractor,
    )

    # 首次每个兴趣只搜到 10 篇，抽到 10 个地点，低于目标 15
    _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐", count=5)
    _call(tools["mcp_web_search"], query="南宁 自然 景点 推荐", count=5)

    first = _call(tools["extract_web_places"], city=CITY, interests=INTERESTS)
    assert first["ok"] is False
    assert first["error_code"] == "insufficient_places"
    assert first["retryable"] is True
    assert first["by_interest"] == {"美食": 10, "自然": 10}
    assert first["shortages"] == {"美食": 10, "自然": 10}
    assert first["round"] == 1
    assert first["rounds_left"] == 3
    assert first["next_count"] == 20

    # Agent 按提示用同一 query 补检索；工具自动把 count 放大到 20
    again_1 = _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐", count=5)
    again_2 = _call(tools["mcp_web_search"], query="南宁 自然 景点 推荐", count=5)
    assert again_1["count"] == 20
    assert again_2["count"] == 20

    second = _call(tools["extract_web_places"], city=CITY, interests=INTERESTS)
    assert second["ok"] is True
    assert second["by_interest"] == {"美食": 15, "自然": 15}
    assert second["shortages"] == {}
    assert second["round"] == 2
    # 第二轮只把新增的 10 篇交给 LLM，已消费的 10 篇不会重复抽取
    assert extractor.calls == [
        ("美食", 10),
        ("自然", 10),
        ("美食", 10),
        ("自然", 10),
    ]


def test_extract_returns_ok_after_round_limit_with_warnings() -> None:
    caller = _GrowingBochaCaller()
    extractor = _ResultExtractor()
    tools, _, _, _ = _build_tools(
        min_places_per_interest=15,
        max_rounds=1,
        caller=caller,
        extractor=extractor,
    )

    _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐", count=5)
    _call(tools["mcp_web_search"], query="南宁 自然 景点 推荐", count=5)

    payload = _call(tools["extract_web_places"], city=CITY, interests=INTERESTS)

    assert payload["ok"] is True
    assert payload["shortages"] == {"美食": 10, "自然": 10}
    assert payload["warnings"]
    assert "轮次上限" in payload["warnings"][0]


def test_rank_tool_handles_web_place_pool() -> None:
    tools, _, _, web_ranker = _build_tools()
    _call(tools["mcp_web_search"], query="南宁 美食 景点 推荐")
    _call(tools["mcp_web_search"], query="南宁 自然 景点 推荐")
    extracted = _call(
        tools["extract_web_places"],
        city=CITY,
        interests=INTERESTS,
    )

    payload = _call(
        tools["rank_attraction_pool"],
        search_id=extracted["search_id"],
        interests=INTERESTS,
        top_k=TOP_K,
    )

    assert payload["ok"] is True
    assert payload["kind"] == "web_place"
    assert payload["total"] == TOP_K
    assert web_ranker.seen == 3
    assert set(payload["by_interest"]) == set(INTERESTS)
    assert payload["top_k_per_interest"] == 7
    assert all(len(rows) <= 7 for rows in payload["by_interest"].values())
    scores = [item["score"] for item in payload["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all("name" in item and "description" in item for item in payload["items"])


def test_workflow_finalize_reads_rank_tool_result() -> None:
    web_cache = WebSearchResultCache()
    pool_cache = AttractionPoolCache()
    extractor = _StubExtractor()
    web_ranker = _StubWebRanker()
    tools = [
        *create_bocha_mcp_tools(
            _FakeBochaCaller(),
            result_cache=web_cache,
            strict_query=True,
        ),
        build_extract_web_places_tool(
            pool_cache,
            web_cache,
            extractor=extractor,
            alias_resolver=_IdentityAliasResolver(),
            min_places_per_interest=1,
        ),
        create_rank_attraction_pool_tool(
            pool_cache,
            web_place_ranker=web_ranker,
        ),
    ]
    graph = build_selection_graph(
        agent=_ScriptedAgent(),
        tools=tools,
        top_k=TOP_K,
    )

    final_state = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="请开始筛选景点")],
                "city": CITY,
                "interests": INTERESTS,
                "ranking": None,
                "selection_summary": None,
            }
        )
    )

    ranking = final_state["ranking"]
    assert ranking is not None
    assert 0 < len(ranking["items"]) <= TOP_K
    assert ranking["ranked_ids"] == [item["id"] for item in ranking["items"]]
    assert final_state["selection_summary"] == "已完成网页地点初选。"
