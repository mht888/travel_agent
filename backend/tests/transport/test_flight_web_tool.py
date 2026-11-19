"""KAYAK Tool 独立测试。用 pytest -s 查看提示词、模型回复和查询结果。"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.config import get_settings
from backend.tools import flight_web_tools


class PrintModelTrace(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        for batch in messages:
            self.calls += 1
            print(f"\n=== 第 {self.calls} 轮 LLM 提示词 ===")
            for message in batch:
                print(f"[{message.type}]\n{message.content}")

    def on_llm_end(self, response, **kwargs) -> None:
        print("\n=== LLM 回复 ===")
        for batch in response.generations:
            for generation in batch:
                print(generation.text)


def _run_tool(query: dict, trace: PrintModelTrace) -> dict:
    print("\n=== Tool 输入 ===")
    print(json.dumps(query, ensure_ascii=False, indent=2))
    result = asyncio.run(
        flight_web_tools.search_flights_web.ainvoke(
            {"query": query}, config={"callbacks": [trace]}
        )
    )
    print("\n=== Tool 查询结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def test_flight_web_tool_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    quote = "香港快運\nHK$2,171\n經濟艙"

    class FakePage:
        async def goto(self, url, **kwargs):
            self.url = url
            return SimpleNamespace(status=200)

        def locator(self, selector):
            assert selector == "body"
            return self

        async def inner_text(self, **kwargs):
            return quote

    @asynccontextmanager
    async def fake_browser(**kwargs):
        yield FakePage()

    monkeypatch.setattr(flight_web_tools, "open_browser_page", fake_browser)
    monkeypatch.setattr(
        flight_web_tools,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test", openai_base_url=None, openai_model="test"),
    )
    monkeypatch.setattr(
        langchain_openai,
        "ChatOpenAI",
        lambda **kwargs: FakeListChatModel(
            responses=[json.dumps({"action": "finish", "evidence": [quote]}, ensure_ascii=False)]
        ),
    )
    trace = PrintModelTrace()
    result = _run_tool(
        {"origin": "HKG", "destination": "KIX", "departure": "2026-12-10"}, trace
    )
    assert trace.calls == 1
    assert result["status"] == "ok"
    assert result["excerpts"] == [quote]


@pytest.mark.skipif(
    os.getenv("RUN_FLIGHT_WEB_LIVE_TESTS") != "1",
    reason="设置 RUN_FLIGHT_WEB_LIVE_TESTS=1 后运行 KAYAK 实站测试",
)
def test_flight_web_tool_live() -> None:
    assert get_settings().openai_api_key, "请先配置 OPENAI_API_KEY"
    departure = datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=30)
    trace = PrintModelTrace()
    result = _run_tool(
        {
            "origin": "HKG",
            "destination": "KIX",
            "departure": departure.isoformat(),
            "return_date": (departure + timedelta(days=5)).isoformat(),
            "adults": 1,
            "cabin": "economy",
            "direct_only": False,
        },
        trace,
    )
    assert result["status"] == "ok", (
        f"KAYAK 未返回报价，状态 {result['status']}，错误码 {result['error_code']}"
    )
    assert trace.calls > 0
    assert result["excerpts"]
