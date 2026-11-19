r"""博查 MCP（Streamable HTTP）真实调用验证（会联网、会计费，默认跳过）。

目标：用 MCP 的 Streamable HTTP 传输连上博查托管服务，先列出它暴露的工具，
再用「南宁 夜市 热门」调用搜索工具，把工具列表和原始返回打印到终端。
不经过项目的 Provider / Service / Tool 三层，只验证连接和返回结构。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_BOCHA_MCP_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/tools/test_bocha_mcp_live.py -s -q

前置条件：
    1. .env 里配置 bocha_apikey（或 BOCHA_API_KEY）；
    2. 能访问 mcp.bochaai.com；
    3. 账户有可用额度（Web Search 约 0.036 元/次）。
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx2
import pytest
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ENDPOINT = os.getenv("BOCHA_MCP_URL", "https://mcp.bochaai.com/mcp")
QUERY = "南宁必吃美食推荐店铺"
# 可选值：noLimit / oneDay / oneWeek / oneMonth / oneYear / YYYY-MM-DD / YYYY-MM-DD..YYYY-MM-DD
FRESHNESS = os.getenv("BOCHA_FRESHNESS", "noLimit")
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# 工具名按优先级猜测，实际以 list_tools 结果为准
PREFERRED_TOOL_NAMES = ("bocha_web_search", "bocha_search", "web_search", "search")
QUERY_FIELD_NAMES = ("query", "q", "keyword", "search_query")
COUNT_FIELD_NAMES = ("count", "top_k", "num", "limit")


def _mask(key: str) -> str:
    return f"{key[:6]}****{key[-4:]}" if len(key) > 12 else "****"


def _pick_tool(tools: list) -> object:
    by_name = {tool.name: tool for tool in tools}
    for name in PREFERRED_TOOL_NAMES:
        if name in by_name:
            return by_name[name]
    for tool in tools:
        lowered = tool.name.lower()
        # 避开 AI Search（0.06 元/次，比 Web Search 贵）
        if "search" in lowered and "ai" not in lowered:
            return tool
    return tools[0]


def _build_arguments(schema: dict) -> dict:
    properties = (schema or {}).get("properties") or {}
    query_field = next((name for name in QUERY_FIELD_NAMES if name in properties), "query")
    arguments: dict = {query_field: QUERY}
    count_field = next((name for name in COUNT_FIELD_NAMES if name in properties), None)
    if count_field:
        arguments[count_field] = 5
    if "freshness" in properties:
        arguments["freshness"] = FRESHNESS
    return arguments


def _print_content(result) -> None:
    for index, item in enumerate(result.content or [], 1):
        text = getattr(item, "text", None)
        print(f"\n[content {index}] {getattr(item, 'type', 'unknown')}")
        print(text if text is not None else item)


async def _run(api_key: str) -> None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json, text/event-stream",
    }
    print("--- 连接 ---")
    print(f"Endpoint : {ENDPOINT}")
    print(f"Key      : {_mask(api_key)}")
    print(f"Query    : {QUERY}")
    print(f"Freshness: {FRESHNESS}")

    async with httpx2.AsyncClient(headers=headers, timeout=60.0) as http_client:
        async with streamable_http_client(ENDPOINT, http_client=http_client) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                started = time.perf_counter()
                await session.initialize()
                print(f"\n--- initialize 成功（{time.perf_counter() - started:.2f}s）---")

                tools = (await session.list_tools()).tools
                print(f"\n--- 工具列表（共 {len(tools)} 个）---")
                for tool in tools:
                    schema = getattr(tool, "input_schema", None) or getattr(
                        tool, "inputSchema", None
                    )
                    print(f"\n* {tool.name}")
                    print(f"  description: {tool.description}")
                    print("  inputSchema: " + json.dumps(schema, ensure_ascii=False))

                if not tools:
                    pytest.fail("博查 MCP 未返回任何工具")

                tool = _pick_tool(tools)
                schema = getattr(tool, "input_schema", None) or getattr(
                    tool, "inputSchema", None
                )
                arguments = _build_arguments(schema or {})
                print(f"\n--- 调用 {tool.name} ---")
                print(json.dumps(arguments, ensure_ascii=False, indent=2))

                started = time.perf_counter()
                result = await session.call_tool(tool.name, arguments)
                elapsed = time.perf_counter() - started
                is_error = getattr(result, "is_error", getattr(result, "isError", False))
                print(f"\n--- 调用返回（{elapsed:.2f}s, isError={is_error}）---")
                _print_content(result)

                print("\n--- 原始返回 ---")
                print(
                    json.dumps(
                        result.model_dump(exclude_none=True),
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                )


def test_bocha_mcp_web_search() -> None:
    if not os.getenv("RUN_BOCHA_MCP_TESTS"):
        pytest.skip("真实博查测试默认跳过，设置 RUN_BOCHA_MCP_TESTS=1 后运行")

    load_dotenv(ENV_PATH)
    api_key = os.getenv("bocha_apikey") or os.getenv("BOCHA_API_KEY")
    if not api_key:
        pytest.skip("未配置 bocha_apikey / BOCHA_API_KEY")

    asyncio.run(_run(api_key))
