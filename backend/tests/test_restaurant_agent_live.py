r"""餐厅 Agent 单跑（会联网 + 调用 LLM，默认跳过）。

链路：
    城市 / 口味 / 预算 → 四阶段提示词
    → Agent 调 gaode_mcp（action=around_search, poi_kind=restaurant）、
      mcp_web_search、mcp_geocode
    → finalize 从工具结果解析 Restaurant（按 id 去重）

会打印：
    1. 完整 system prompt；
    2. 每个节点的 AI 文本与 tool_call；
    3. 工具返回摘要；
    4. 最终 Restaurant 字段、美食摘要与 warnings。

运行方式（PowerShell，工作目录 travel_agent/）：

    # 方式一：pytest
    $env:RUN_RESTAURANT_AGENT_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_restaurant_agent_live.py -s -q

    # 方式二：直接单跑
    .\.venv\Scripts\python.exe -m backend.tests.test_restaurant_agent_live

前置条件：
    OPENAI_API_KEY 必填；GAODE_MAPS_AK 用于结构化餐厅搜索；
    BOCHA_APIKEY 可选，但缺失时只能搜周边、做不了「当地特色发现」。
"""

import asyncio
import json
import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from backend.agents.restaurant_agent import (
    build_restaurant_system_prompt,
    create_restaurant_agent,
)
from backend.config import get_settings
from backend.graph.restaurant_workflow import build_restaurant_graph
from backend.tools.registry import get_profile_tools

CITY = "南宁"
AREA: str | None = None
# 留空表示让 Agent 先做「当地特色美食发现」；想吃特定菜系就填 ["老友粉"]
CUISINES: list[str] = []
# 想按人均筛就填 Money(amount=80, currency="CNY")
BUDGET = None
NOTES = "想找美食店"


def _print_ai(message: AIMessage) -> None:
    if message.content:
        print(f"文本：{message.content}")
    for call in message.tool_calls:
        print(f"tool_call：{call['name']}  args={call['args']}")


def _pretty(payload: str) -> str:
    """工具返回是 JSON 字符串：只打印关键字段，避免刷屏。"""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return str(payload)[:500]

    if data.get("ok") is False:
        return f"tool error: {json.dumps(data, ensure_ascii=False)}"

    # gaode_mcp around_search（poi_kind=restaurant）
    if data.get("center") is not None and data.get("items") is not None:
        center = data["center"]
        lines = [
            f"center=({center.get('longitude')}, {center.get('latitude')}) "
            f"radius={data.get('radius_meters')} total={data.get('total')}"
        ]
        for item in data["items"]:
            lines.append(
                f"  - {item.get('name')}  type={item.get('type')}  "
                f"rating={item.get('rating')}  cost={item.get('average_cost_cny')}  "
                f"distance={item.get('distance_meters')}m  "
                f"open={item.get('opening_hours_today')}"
            )
        return "\n".join(lines)

    # mcp_web_search
    if data.get("results") is not None:
        lines = [f"query={data.get('query')} count={data.get('count')}"]
        for item in data["results"][:5]:
            snippet = str(item.get("snippet") or "")[:100]
            lines.append(
                f"  - {item.get('title')}  {item.get('url')}"
                + (f"\n    {snippet}" if snippet else "")
            )
        return "\n".join(lines)

    return json.dumps(data, ensure_ascii=False)[:800]


async def _run() -> None:
    settings = get_settings()
    tools = get_profile_tools("restaurant_search")
    names = [tool.name for tool in tools]
    print(
        f"\n可用工具：{'、'.join(names)}\n"
        f"openai_model={settings.openai_model} "
        f"bocha={'有' if settings.bocha_apikey else '无'} "
        f"gaode={'有' if settings.gaode_maps_ak else '无'}"
    )
    if "gaode_mcp" not in names:
        raise AssertionError("餐厅周边搜索不可用，请检查 GAODE_MAPS_AK 是否已配置")
    if "mcp_web_search" not in names:
        print("警告：没有网页检索，只能做周边搜索，做不了特色美食发现")

    prompt = build_restaurant_system_prompt(
        CITY,
        area=AREA,
        cuisines=CUISINES,
        budget=BUDGET,
        notes=NOTES,
        tools=tools,
    )
    print("\n" + "#" * 30 + " 餐厅 SYSTEM PROMPT " + "#" * 30)
    print(prompt.content)
    print("#" * 76)

    graph = build_restaurant_graph(
        agent=create_restaurant_agent(tools), tools=tools
    )
    initial_state = {
        "messages": [
            prompt,
            HumanMessage(content=f"请推荐「{CITY}」值得吃的当地美食和具体餐厅"),
        ],
        "city": CITY,
        "area": AREA,
        "cuisines": CUISINES,
        "budget": None,
        "notes": NOTES,
        "restaurants": [],
        "restaurant_summary": None,
        "warnings": [],
    }

    final_restaurants: list[dict[str, Any]] = []
    final_summary: str | None = None
    warnings: list[str] = []
    finalized = False

    async for update in graph.astream(
        initial_state,
        stream_mode="updates",
        config={"recursion_limit": 50},
    ):
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

            if values.get("restaurants") is not None:
                final_restaurants = values["restaurants"]
                print(f"restaurants={len(final_restaurants)}")
            if values.get("restaurant_summary"):
                final_summary = values["restaurant_summary"]
                print(f"restaurant_summary={final_summary}")
            if values.get("warnings"):
                warnings = values["warnings"]
                print(f"warnings={warnings}")
            if node == "finalize":
                finalized = True

    assert finalized, "finalize 节点没有执行"

    print("\n" + "=" * 30 + " 最终餐厅（Restaurant schema） " + "=" * 30)
    print(f"共 {len(final_restaurants)} 家")
    for index, item in enumerate(final_restaurants, start=1):
        location = item.get("location") or {}
        price = (item.get("average_price") or {}).get("amount")
        provenance = item.get("provenance") or {}
        print(
            f"{index}. {item.get('name')}  cuisines={item.get('cuisines')} "
            f"rating={item.get('rating')} price={price} "
            f"lng={location.get('lng')} lat={location.get('lat')}"
        )
        print(
            f"   city={item.get('city')} source={provenance.get('source')} "
            f"status={provenance.get('status')}"
        )
    print(f"\n美食摘要：{final_summary}")
    print(f"warnings={warnings}")


def test_restaurant_agent_live() -> None:
    if not os.getenv("RUN_RESTAURANT_AGENT_TESTS"):
        pytest.skip(
            "真实链路默认跳过，设置 RUN_RESTAURANT_AGENT_TESTS=1 后运行"
        )
    asyncio.run(_run())


if __name__ == "__main__":
    asyncio.run(_run())
