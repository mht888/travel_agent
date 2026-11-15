r"""网页地点初选真实链路（会联网 + 调用 LLM，默认跳过）。

流程：
    mcp_web_search（每个兴趣一条 query，首次 count=10）
    → extract_web_places（从网页结果抽地点，并统计每个兴趣数量；
      不足 10 个会返回 insufficient_places，Agent 用更大的 count 补检索）
    → rank_attraction_pool（LLM 打分排序）
    → finalize 收口

会打印：
    1. 完整的 system prompt；
    2. 每个节点的输入输出（tool_call / 工具返回 / 排序结果）；
    3. 抽取与排序调用里发给 LLM 的完整提示词。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_WEB_PLACE_SELECTION_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_web_place_selection_live.py -s -q

前置条件：
    BOCHA_APIKEY 与 OPENAI_API_KEY 都已配置。
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
from backend.cache.pool_cache import AttractionPoolCache, WebSearchResultCache
from backend.config import get_settings
from backend.graph.selection_workflow import build_selection_graph
from backend.tools.bocha_mcp_tools import create_bocha_mcp_tools
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
    tools, _ = _build_tools(settings)

    prompt = build_selection_system_prompt(CITY, INTERESTS, tools=tools)
    print("\n" + "#" * 30 + " SYSTEM PROMPT " + "#" * 30)
    print(prompt.content)
    print("#" * 76)

    agent = create_attraction_selection_agent(tools)
    graph = build_selection_graph(
        agent=agent,
        tools=tools,
        top_k=TOP_K,
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
        "selection_summary": None,
    }

    final_ranking: dict[str, Any] | None = None
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
            if values.get("selection_summary"):
                final_summary = values["selection_summary"]
                print(f"summary={final_summary}")

    print("\n" + "=" * 30 + " 最终结果 " + "=" * 30)
    if final_ranking is None:
        print("没有产出 ranking")
    else:
        for index, item in enumerate(final_ranking.get("items", []), start=1):
            print(f"{index}. {item.get('id')}  score={item.get('score')}  {item.get('reason')}")
        print(f"ranked_ids={final_ranking.get('ranked_ids')}")
    print(f"summary={final_summary}")


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

    asyncio.run(_run())
