r"""景点初选 + 酒店搜索真实链路（会联网 + 调用 LLM，默认跳过）。

阶段一：景点初选与排序
    mcp_web_search（每个兴趣一条 query，首次 count=10）
    → extract_web_places（从网页结果抽地点，并统计每个兴趣数量；
      不足 10 个会返回 insufficient_places，Agent 用更大的 count 补检索）
    → rank_attraction_pool（LLM 打分排序）
    → finalize 组装 Attraction，写入候选池得到 selection_id

阶段二：酒店搜索
    selection_id → compact_anchors（只保留酒店 Agent 需要的字段）
    → 酒店 Agent（区域发现 → geocode / 网页验证）
    → search_hotels_around（每个候选区域一次）
    → finalize 转成 Hotel schema（按 id 去重）

会打印：
    1. 两个阶段完整的 system prompt；
    2. 每个节点的输入输出（tool_call / 工具返回 / 排序结果）；
    3. 抽取、排序调用里发给 LLM 的完整提示词；
    4. 最终入选景点与 Hotel schema 字段。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_WEB_PLACE_SELECTION_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_web_place_selection_live.py -s -q

前置条件：
    BOCHA_APIKEY、OPENAI_API_KEY、GAODE_MAPS_AK、百度地图 AK 都已配置。
    百度 AK 用于给入选景点补坐标，没有坐标酒店阶段无法开始。
"""

import asyncio
import json
import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from backend.agents.attraction_selection_agent import (
    build_selection_system_prompt,
    create_attraction_selection_agent,
)
from backend.agents.hotel_agent import (
    build_hotel_system_prompt,
    create_hotel_agent,
)
from backend.cache.pool_cache import AttractionPoolCache, WebSearchResultCache
from backend.config import get_settings
from backend.graph.hotel_workflow import build_hotel_graph, compact_anchors
from backend.graph.selection_workflow import (
    _selected_attractions,
    build_selection_graph,
)
from backend.schemas.attraction import Attraction
from backend.schemas.ranking import AttractionRanking
from backend.services.attraction_assembly import create_attraction_assembler
from backend.tools.bocha_mcp_tools import create_bocha_mcp_tools
from backend.tools.registry import get_profile_tools
from backend.tools.rank_attraction_pool_tools import (
    LLMWebPlaceRanker,
    _build_rank_runnable,
    create_rank_attraction_pool_tool,
)
from backend.tools.web_place_extract_tools import (
    LLMPlaceAliasResolver,
    LLMWebPlaceExtractor,
    _build_alias_runnable,
    _build_extract_runnable,
    build_extract_web_places_tool,
)

CITY = "南宁"
INTERESTS = ["美食", "自然"]
TOP_K = 7
HOTEL_RADIUS = 2000


class _EchoRunnable:
    """打印 LLM 实际收到的提示词，再原样转发。"""

    def __init__(self, inner: Any, label: str) -> None:
        self._inner = inner
        self._label = label

    async def ainvoke(self, prompt: str) -> Any:
        print("\n" + "=" * 24 + f" [{self._label}] 发给 LLM 的输入 " + "=" * 24)
        print(prompt)
        print("=" * 78 + "\n")
        return await self._inner.ainvoke(prompt)


def _build_tools(settings: Any) -> tuple[list[Any], AttractionPoolCache]:
    web_cache = WebSearchResultCache()
    pool_cache = AttractionPoolCache()

    extractor = LLMWebPlaceExtractor(
        _EchoRunnable(_build_extract_runnable(settings), "地点抽取"),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )
    ranker = LLMWebPlaceRanker(
        _EchoRunnable(_build_rank_runnable(settings), "网页地点排序"),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )
    alias_resolver = LLMPlaceAliasResolver(
        _EchoRunnable(_build_alias_runnable(settings), "地点归并"),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )

    tools = [
        *create_bocha_mcp_tools(result_cache=web_cache, strict_query=True),
        build_extract_web_places_tool(
            pool_cache,
            web_cache,
            extractor=extractor,
            alias_resolver=alias_resolver,
            min_places_per_interest=15,
            max_rounds=4,
        ),
        create_rank_attraction_pool_tool(
            pool_cache,
            web_place_ranker=ranker,
        ),
    ]
    return tools, pool_cache


def _pretty(payload: str, limit: int = 5) -> str:
    """工具返回是 JSON 字符串：紧凑打印关键字段，避免刷屏。"""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return payload[:500]

    if data.get("ok") is False:
        return f"tool error: {json.dumps(data, ensure_ascii=False)}"

    # search_hotels_around：中心点 + 酒店列表
    if data.get("center") is not None and data.get("items") is not None:
        center = data["center"]
        lines = [
            f"center=({center.get('longitude')}, {center.get('latitude')}) "
            f"radius={data.get('radius_meters')} total={data.get('total')}"
        ]
        for item in data["items"][:15]:
            lines.append(
                f"  - {item.get('name')}  rating={item.get('rating')}  "
                f"cost={item.get('average_cost_cny')}  "
                f"distance={item.get('distance_meters')}m  "
                f"area={item.get('business_area')}"
            )
        return "\n".join(lines)

    if data.get("results") is not None:
        lines = [
            f"query={data.get('query')} count={data.get('count')} "
            f"web_search_id={data.get('web_search_id')}"
        ]
        for item in data["results"][:limit]:
            snippet = str(item.get("snippet") or "")[:120]
            lines.append(
                f"  - {item.get('title')}\n"
                f"    {item.get('url')}\n"
                f"    site={item.get('site')} published_at={item.get('published_at')}"
                + (f"\n    {snippet}" if snippet else "")
            )
        return "\n".join(lines)

    if (
        data.get("kind") == "web_place"
        and data.get("by_interest") is not None
        and data.get("strategy_used") is None
    ):
        lines = [
            f"search_id={data.get('search_id')} total={data.get('total')} "
            f"by_interest={data.get('by_interest')} round={data.get('round')}"
        ]
        if data.get("raw_counts") != data.get("by_interest"):
            lines.append(
                f"  别名归并前 raw_counts={data.get('raw_counts')} "
                f"alias_merged={data.get('alias_merged')}"
            )
        for item in data.get("items", [])[:15]:
            lines.append(
                f"  - {item.get('name')} [{item.get('category')}] "
                f"id={item.get('id')}\n"
                f"    {str(item.get('description') or '')[:100]}"
            )
        if data.get("warnings"):
            lines.append(f"  warnings={data['warnings']}")
        return "\n".join(lines)

    if data.get("by_interest") is not None:
        lines = [
            f"search_id={data.get('search_id')} kind={data.get('kind')} "
            f"total={data.get('total')} "
            f"top_k_per_interest={data.get('top_k_per_interest')}"
        ]
        for interest, rows in data["by_interest"].items():
            lines.append(f"  [{interest}] {len(rows)} 个")
            for index, item in enumerate(rows, start=1):
                lines.append(
                    f"    {index}. {item.get('name')}  score={item.get('score')}  "
                    f"{item.get('reason')}"
                )
        return "\n".join(lines)

    if data.get("items") is not None and "search_id" in data:
        lines = [
            f"search_id={data.get('search_id')} kind={data.get('kind')} "
            f"total={data.get('total')}"
        ]
        for index, item in enumerate(data["items"], start=1):
            lines.append(
                f"  {index}. {item.get('name')}  score={item.get('score')}  "
                f"{item.get('reason')}"
            )
        return "\n".join(lines)

    return json.dumps(data, ensure_ascii=False)[:800]


def _print_ai(message: AIMessage) -> None:
    if message.content:
        print(f"文本：{message.content}")
    for call in message.tool_calls:
        print(f"tool_call：{call['name']}  args={call['args']}")


async def _run() -> None:
    settings = get_settings()
    tools, pool_cache = _build_tools(settings)
    # 没有 assembler 就拿不到坐标，酒店阶段无法开始
    assembler = create_attraction_assembler(settings=settings)

    prompt = build_selection_system_prompt(CITY, INTERESTS, tools=tools)
    print("\n" + "#" * 30 + " SYSTEM PROMPT " + "#" * 30)
    print(prompt.content)
    print("#" * 76)

    agent = create_attraction_selection_agent(tools)
    graph = build_selection_graph(
        agent=agent,
        tools=tools,
        top_k=TOP_K,
        assembler=assembler,
    )

    initial_state = {
        "messages": [
            prompt,
            HumanMessage(
                content=f"请为「{CITY}」筛选兴趣相关的景点：{'、'.join(INTERESTS)}"
            ),
        ],
        "city": CITY,
        "interests": INTERESTS,
        "ranking": None,
        "attractions": [],
        "selection_summary": None,
        "warnings": [],
    }

    final_ranking: dict[str, Any] | None = None
    final_attractions: list[dict[str, Any]] = []
    final_summary: str | None = None

    async for update in graph.astream(initial_state, stream_mode="updates"):
        for node, values in update.items():
            print(f"\n---------- 节点：{node} ----------")
            if not values:
                print("（无输出）")
                continue

            messages: list[BaseMessage] = values.get("messages", [])
            for message in messages:
                if isinstance(message, AIMessage):
                    _print_ai(message)
                elif isinstance(message, ToolMessage):
                    print(f"工具返回 [{message.name}]：")
                    print(_pretty(str(message.content)))

            if values.get("ranking") is not None:
                final_ranking = values["ranking"]
                print(f"ranking={json.dumps(final_ranking, ensure_ascii=False)}")
            if values.get("attractions") is not None:
                final_attractions = values["attractions"]
                print(f"assembled_attractions={len(final_attractions)}")
                for item in final_attractions:
                    location = item.get("location") or {}
                    print(
                        f"  - {item.get('name')} [{item.get('category')}] "
                        f"lng={location.get('lng')} lat={location.get('lat')}"
                    )
            if values.get("selection_summary"):
                final_summary = values["selection_summary"]
                print(f"summary={final_summary}")
            if values.get("warnings"):
                print(f"warnings={values['warnings']}")

    print("\n" + "=" * 30 + " 初选最终结果 " + "=" * 30)
    if final_ranking is None:
        raise AssertionError("初选阶段没有产出 ranking，酒店阶段没有输入")
    for index, item in enumerate(final_ranking.get("items", []), start=1):
        print(f"{index}. {item.get('id')}  score={item.get('score')}  {item.get('reason')}")
    print(f"ranked_ids={final_ranking.get('ranked_ids')}")
    print(f"summary={final_summary}")

    # 把入选景点写进候选池，模拟 selection_id 交接
    ranking_model = AttractionRanking.model_validate(final_ranking)
    attractions = [Attraction.model_validate(item) for item in final_attractions]
    entries = _selected_attractions(attractions, ranking_model)
    if not entries:
        raise AssertionError("初选没有组装出 Attraction，无法进入酒店阶段")
    selection_id = pool_cache.save_selection(CITY, entries)
    print(f"\nselection_id={selection_id}  assembled={len(attractions)}")

    # ---------------------------- 阶段二：酒店搜索 ----------------------------
    hotel_tools = get_profile_tools("hotel_search")
    if not any(tool.name == "search_hotels_around" for tool in hotel_tools):
        raise AssertionError("酒店工具不可用，请检查 GAODE_MAPS_AK 是否配置")

    anchors, anchor_warnings = compact_anchors(entries)
    if not anchors:
        raise AssertionError(f"入选景点都缺少坐标，酒店阶段无输入：{anchor_warnings}")
    if anchor_warnings:
        print(f"酒店阶段输入告警：{anchor_warnings}")
    print(
        f"\n酒店阶段工具：{'、'.join(tool.name for tool in hotel_tools)}  "
        f"anchors={len(anchors)}  radius={HOTEL_RADIUS}m"
    )

    hotel_prompt = build_hotel_system_prompt(
        CITY,
        anchors,
        radius_meters=HOTEL_RADIUS,
        tools=hotel_tools,
    )
    print("\n" + "#" * 30 + " 酒店 SYSTEM PROMPT " + "#" * 30)
    print(hotel_prompt.content)
    print("#" * 76)

    hotel_agent = create_hotel_agent(hotel_tools)
    hotel_graph = build_hotel_graph(agent=hotel_agent, tools=hotel_tools)
    hotel_state = {
        "messages": [
            hotel_prompt,
            HumanMessage(content="请开始分析住宿区域并搜索酒店"),
        ],
        "city": CITY,
        "radius_meters": HOTEL_RADIUS,
        "anchors": anchors,
        "hotels": [],
        "hotel_summary": None,
        "warnings": [],
    }

    final_hotels: list[dict[str, Any]] = []
    final_hotel_summary: str | None = None
    hotel_warnings: list[str] = []
    hotel_finalized = False

    async for update in hotel_graph.astream(
        hotel_state,
        stream_mode="updates",
        config={"recursion_limit": 40},
    ):
        for node, values in update.items():
            print(f"\n---------- 酒店节点：{node} ----------")
            if not values:
                print("（无输出）")
                continue

            for message in values.get("messages", []):
                if isinstance(message, AIMessage):
                    _print_ai(message)
                elif isinstance(message, ToolMessage):
                    print(f"工具返回 [{message.name}]：")
                    print(_pretty(str(message.content)))

            if values.get("hotels") is not None:
                final_hotels = values["hotels"]
                print(f"hotels={len(final_hotels)}")
            if values.get("hotel_summary"):
                final_hotel_summary = values["hotel_summary"]
                print(f"hotel_summary={final_hotel_summary}")
            if values.get("warnings"):
                hotel_warnings = values["warnings"]
                print(f"warnings={hotel_warnings}")
            if node == "finalize":
                hotel_finalized = True

    assert hotel_finalized, "酒店 finalize 节点没有执行"

    print("\n" + "=" * 30 + " 酒店最终结果（Hotel schema） " + "=" * 30)
    print(f"共 {len(final_hotels)} 家酒店")
    for index, item in enumerate(final_hotels, start=1):
        price = (item.get("reference_price") or {}).get("amount")
        provenance = item.get("provenance") or {}
        print(
            f"{index}. {item.get('name')} [{item.get('category')}] "
            f"star={item.get('star_rating')} rating={item.get('rating')} "
            f"price={price} area={item.get('business_area')} "
            f"tags={item.get('tags')}"
        )
        print(
            f"   address={item.get('address')} tel={item.get('tel')} "
            f"source={provenance.get('source')} "
            f"status={provenance.get('status')} "
            f"photos={len(item.get('photo_urls') or [])}"
        )
    print(f"hotel_summary={final_hotel_summary}")
    print(f"warnings={hotel_warnings}")


def test_web_place_selection_live() -> None:
    if not os.getenv("RUN_WEB_PLACE_SELECTION_TESTS"):
        pytest.skip(
            "真实链路默认跳过，设置 RUN_WEB_PLACE_SELECTION_TESTS=1 后运行"
        )

    settings = get_settings()
    if not settings.bocha_apikey:
        pytest.skip("缺少 BOCHA_APIKEY")
    if not settings.openai_api_key:
        pytest.skip("缺少 OPENAI_API_KEY")
    if not settings.gaode_maps_ak:
        pytest.skip("缺少 GAODE_MAPS_AK，酒店阶段无法搜索")
    if not settings.baidu_mcp_enabled or not settings.baidu_maps_ak:
        pytest.skip("缺少百度地图 AK/MCP，入选景点无法补齐坐标")

    asyncio.run(_run())
