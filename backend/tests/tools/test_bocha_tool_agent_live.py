r"""博查工具在 Agent 里的真实调用：打印 Agent 到底收到了什么（会联网、会计费，默认跳过）。

和 test_bocha_mcp_live.py 的区别：
    那个文件绕开项目代码，直接连 MCP 看原始返回；
    本文件走项目自己的 LangChain Tool（registry 注册的 mcp_web_search），
    并且跑一个真实的 tool-calling 循环，把 Agent 每轮看到的内容原样打印：

        工具 schema → LLM 决定的 tool_calls → ToolMessage.content 原文
        → 内容长度 / 粗估 token → 解析后的字段 → 下一轮 messages 的角色与长度

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_BOCHA_AGENT_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/tools/test_bocha_tool_agent_live.py -s -q

前置条件：
    bocha_apikey、OPENAI_API_KEY 都已配置；账户有博查额度（约 0.036 元/次）。
"""

import asyncio
import json
import os
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.tools.registry import build_tools

TOOL_NAME = "mcp_web_search"
QUERY = "南宁 景点 临时闭馆 2026 年 11 月"
MAX_ROUNDS = 4


def _banner(text: str) -> None:
    print("\n" + "=" * 24 + f" {text} " + "=" * 24)


def _print_tool_definition(tool) -> None:
    _banner("Agent 看到的工具定义")
    print(f"name: {tool.name}")
    print(f"description: {tool.description}")
    print("args: " + json.dumps(tool.args, ensure_ascii=False, indent=2))


def _print_ai_message(message: AIMessage, round_no: int) -> None:
    print(f"\n[第 {round_no} 轮 LLM 输出]")
    content = message.content
    if isinstance(content, str) and content.strip():
        print(f"  content: {content.strip()}")
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        print("  tool_calls: 无 → 循环结束")
        return
    for call in calls:
        print(f"  tool_call → {call.get('name')}(id={call.get('id')})")
        print("    args: " + json.dumps(call.get("args") or {}, ensure_ascii=False))


def _print_tool_message(message: ToolMessage) -> dict:
    _banner("Agent 收到的 ToolMessage.content（原文，未截断）")
    content = message.content if isinstance(message.content, str) else str(message.content)
    print(content)

    _banner("这条消息的体量")
    print(f"tool_call_id: {message.tool_call_id}")
    print(f"字符数: {len(content)}")
    print(f"粗估 token: {len(content) // 2}（中文约 2 字符/token，仅供参考）")

    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        print("\n⚠ 返回内容不是 JSON")
        return {}

    _banner("解析后的结构（也就是 Agent 真正能读到的字段）")
    print(f"ok={payload.get('ok')}  query={payload.get('query')}  "
          f"freshness={payload.get('freshness')}  count={payload.get('count')}")
    for index, item in enumerate(payload.get("results") or [], start=1):
        snippet = item.get("snippet") or ""
        print(f"\n  {index}. {item.get('title')}")
        print(f"     url: {item.get('url')}")
        print(f"     published_at: {item.get('published_at')}  site: {item.get('site')}")
        print(f"     snippet（{len(snippet)} 字符）: {snippet}")
    print(f"\n  hint: {payload.get('hint')}")
    return payload


def _print_history(messages: list) -> None:
    _banner("下一轮 LLM 实际看到的 messages")
    for index, message in enumerate(messages, start=1):
        content = message.content if isinstance(message.content, str) else str(message.content)
        role = type(message).__name__
        calls = getattr(message, "tool_calls", None) or []
        suffix = f"  tool_calls={[call.get('name') for call in calls]}" if calls else ""
        print(f"  [{index}] {role:<14} {len(content):>6} 字符{suffix}")


async def _run() -> None:
    settings = get_settings()
    tools = build_tools([TOOL_NAME])
    assert tools, "注册表没有构建出 mcp_web_search（检查 BOCHA_APIKEY / bocha_mcp_enabled）"
    tool = tools[0]
    _print_tool_definition(tool)

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
    )
    agent = llm.bind_tools(tools)
    messages: list = [
        SystemMessage(
            content=(
                "你是测试助手。请调用 mcp_web_search 查询下面这个主题，"
                "只调用一次，拿到结果后直接用一句话总结，不要调用其他工具。\n"
                f"主题：{QUERY}"
            )
        ),
        HumanMessage(content="开始搜索"),
    ]

    called: list[dict] = []
    payload: dict = {}
    for round_no in range(1, MAX_ROUNDS + 1):
        started = time.perf_counter()
        response = await agent.ainvoke(messages)
        print(f"\n（第 {round_no} 轮 LLM 耗时 {time.perf_counter() - started:.2f}s）")
        _print_ai_message(response, round_no)
        messages.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            called.append(call)
            started = time.perf_counter()
            content = await tool.ainvoke(call.get("args") or {})
            elapsed = time.perf_counter() - started
            print(f"\n（工具执行耗时 {elapsed:.2f}s）")

            tool_message = ToolMessage(
                content=content,
                name=call.get("name", TOOL_NAME),
                tool_call_id=call.get("id", ""),
            )
            payload = _print_tool_message(tool_message) or payload
            messages.append(tool_message)

    _print_history(messages)

    assert called, "Agent 没有调用 mcp_web_search"
    assert payload.get("ok") is True, f"工具返回失败：{payload}"


def test_bocha_tool_in_agent_prints_received_content() -> None:
    if not os.getenv("RUN_BOCHA_AGENT_TESTS"):
        pytest.skip("真实 Agent 测试默认跳过，设置 RUN_BOCHA_AGENT_TESTS=1 后运行")

    settings = get_settings()
    if not settings.bocha_apikey:
        pytest.skip("缺少 bocha_apikey / BOCHA_API_KEY，mcp_web_search 不会注册")
    if not settings.openai_api_key:
        pytest.skip("缺少 OPENAI_API_KEY，Agent 无法启动")

    asyncio.run(_run())
