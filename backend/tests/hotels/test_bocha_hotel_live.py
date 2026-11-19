r"""博查网页工具 · 住宿线索验证（会联网、会计费，默认跳过）。

目标：调用项目里真正给 Agent 使用的 mcp_web_search（博查 MCP），
搜索「南宁 酒店 推荐」，把工具返回的 JSON 打印出来，
用来判断网页结果能否支撑住宿建议（酒店名、区域、价位、口碑）。

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_BOCHA_MCP_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/hotels/test_bocha_hotel_live.py -s -q

前置条件：
    1. .env 里配置 bocha_apikey（或 BOCHA_API_KEY）；
    2. 能访问 mcp.bochaai.com；
    3. 账户有可用额度（Web Search 约 0.036 元/次）。
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from backend.tools.bocha_mcp_tools import (
    close_bocha_mcp_client,
    create_bocha_mcp_tools,
)

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
QUERY = "南宁 朝阳广场 酒店"


def _print_payload(raw: str) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print("\n--- 工具返回（非 JSON）---")
        print(raw)
        return

    print("\n--- 工具返回摘要 ---")
    print(f"ok    : {payload.get('ok')}")
    print(f"query : {payload.get('query')}")
    print(f"count : {payload.get('count')}")
    if payload.get("error_code"):
        print(f"error : {payload.get('error_code')} / {payload.get('message')}")
        return

    for index, item in enumerate(payload.get("results") or [], 1):
        print(f"\n[{index}] {item.get('title')}")
        print(f"    站点: {item.get('site')}  时间: {item.get('published_at')}")
        print(f"    URL : {item.get('url')}")
        print(f"    摘要: {item.get('snippet')}")

    print("\n--- 原始 JSON ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


async def _run() -> None:
    tool = create_bocha_mcp_tools()[0]
    print(f"工具 : {tool.name}")
    print(f"查询 : {QUERY}")
    try:
        raw = await tool.ainvoke({"query": QUERY, "freshness": "noLimit"})
        _print_payload(raw)
    finally:
        await close_bocha_mcp_client()


def test_bocha_hotel_search() -> None:
    if not os.getenv("RUN_BOCHA_MCP_TESTS"):
        pytest.skip("真实博查测试默认跳过，设置 RUN_BOCHA_MCP_TESTS=1 后运行")

    load_dotenv(ENV_PATH)
    if not (os.getenv("bocha_apikey") or os.getenv("BOCHA_API_KEY")):
        pytest.skip("未配置 bocha_apikey / BOCHA_API_KEY")

    asyncio.run(_run())
