"""对照同一批博查摘要在加入网页正文前后的门票提取结果。"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.tests.attractions.test_chengdu_search_providers_live import (
    PROVIDERS,
    _connect,
    _error_label,
    _extract,
    _read,
    _verified,
)
from backend.tests.attractions.test_chengdu_ticket_prices_live import (
    ATTRACTIONS,
    PROMPT,
    TicketReport,
    _hostname,
)
from backend.tools.bocha_mcp_tools import _result_payload, create_bocha_mcp_tools

# 上一轮核对用的普通成人票价和官方站点，仅用于评分和选择读取页面。
# 价格不传给提取模型，也不要求它必须找到这四个值。
PLACES = dict(zip(ATTRACTIONS[:4], (
    (0, ("cdmuseum.com",)),
    (50, ("wuhouci.net.cn", "cdwuhouci.cn")),
    (0, ("cdjinli.com", "wuhouci.net.cn")),
    (55, ("panda.org.cn",)),
), strict=True))

OFFICIAL_PAGES = {
    "成都博物馆": "https://www.cdmuseum.com/xinwen/202501/3893.html",
    "武侯祠": "https://whpwm.cdwuhouci.cn/index",
    "锦里古街": "https://www.cdjinli.com/",
    "成都大熊猫繁育研究基地": "https://m.panda.org.cn/cn/service/ticket/",
}

BODY_PROMPT = PROMPT.replace("博查搜索摘要", "博查搜索摘要及可用的网页正文") + """
读取正文时优先采用正文中与景点、票种和普通游客匹配的说明。
遇到不同年份的预约公告，优先采用有明确日期、适用于普通参观的较新官方说明，
不要把节假日活动规则当作平日政策。引用依据可以摘自搜索摘要或正文，须逐字复制。"""


class BochaSession:
    def __init__(self, session):
        self.session = session

    async def call_tool(self, name, arguments):
        response = await self.session.call_tool(name, arguments=arguments)
        if response.is_error:
            raise RuntimeError("博查搜索失败")
        return _result_payload(response)


def _official_pages(name, rows):
    domains = PLACES[name][1]
    fallback = OFFICIAL_PAGES.get(name)
    for row in rows:
        host = _hostname(row["url"])
        official = any(host == domain or host.endswith(f".{domain}") for domain in domains)
        relevant = any(word in row["title"] for word in ("票务", "门票", "参观", "预约"))
        other_place = (
            (name == "锦里古街" and host != "cdjinli.com" and "锦里" not in row["title"])
            or (name == "成都大熊猫繁育研究基地" and
                ("熊猫谷" in row["title"] or "/pandavalley/" in row["url"]))
        )
        if official and relevant and not other_place:
            yield row
            if fallback and row["url"].rstrip("/") == fallback.rstrip("/"):
                return
            break
    if fallback:
        yield {"title": f"{name}官方参观或票务页面", "url": fallback}


async def _search_summaries(search, query):
    result = json.loads(await search.ainvoke({"query": query}))
    return result.get("results", [])[:5]


async def _run(bocha_key, reader_name, reader_key):
    settings = get_settings()
    reader_url, _, read_tool = PROVIDERS[reader_name]
    summaries = {}
    bodies = {}
    async with _connect(settings.bocha_mcp_url, bocha_key) as session:
        search = create_bocha_mcp_tools(BochaSession(session))[0]
        for name in PLACES:
            summaries[name] = await _search_summaries(
                search, f"成都 {name} 官方 参观须知 门票价格 预约"
            )
            print(f"{name} | 博查摘要 {len(summaries[name])} 条", flush=True)

        pages_by_name = {name: list(_official_pages(name, rows)) for name, rows in summaries.items()}
        if any(pages_by_name.values()):
            try:
                async with _connect(reader_url, reader_key) as reader:
                    for name, pages in pages_by_name.items():
                        for page in pages:
                            try:
                                body = await _read(reader, reader_name, read_tool, page["url"])
                            except Exception as exc:
                                print(f"{name} | 正文读取失败 | {page['url']} | {_error_label(exc)}", flush=True)
                                continue
                            print(f"{name} | 官网正文 {len(body)} 字 | {page['url']}", flush=True)
                            if body:
                                bodies[name] = {**page, "snippet": body, "evidence_kind": "正文"}
                                break
            except Exception as exc:
                print(f"官网正文读取服务异常 | {_error_label(exc)}", flush=True)

        for name in PLACES:
            if name in bodies:
                continue
            direct = await _search_summaries(search, f"成都 {name} 门票价格 参观预约")
            print(f"{name} | 未取得官网正文，博查直接搜索 {len(direct)} 条", flush=True)
            summaries[name] = list({
                (row["url"], row["snippet"]): row for row in [*summaries[name], *direct]
            }.values())

    if not any(summaries.values()) and not bodies:
        raise RuntimeError("没有可用的搜索摘要或正文")
    enriched = {
        name: [*rows, bodies[name]] if name in bodies else list(rows)
        for name, rows in summaries.items()
    }
    async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
        model = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=0,
            timeout=60,
            max_retries=1,
            http_async_client=http,
        )
        before = await _extract(model, TicketReport, BODY_PROMPT, summaries)
        if not bodies:
            print("本次仅有搜索摘要，未形成正文对照。")
            return before, before, summaries, enriched, 0
        after = await _extract(model, TicketReport, BODY_PROMPT, enriched)
    return before, after, summaries, enriched, len(bodies)


def _print_comparison(before, after, summaries, enriched, pages_read):
    correct = [0, 0]
    bookings = [0, 0]
    body_citations = 0
    for name, (expected, _) in PLACES.items():
        print(f"\n{name} | 参考成人票价 {expected} 元")
        for index, (label, tickets, sources) in enumerate((
            ("摘要", before, summaries),
            ("摘要+正文", after, enriched),
        )):
            ticket = tickets.get(name)
            price, reservation = _verified(ticket, sources[name])
            correct[index] += int(price is not None and price[0] == expected)
            bookings[index] += int(reservation is not None)
            print(f"  {label} | 票价 {price or '未知'} | 预约 {reservation or '未知'}")
            if price:
                kind = "正文" if ticket.source_index > len(summaries[name]) else "摘要"
                body_citations += int(kind == "正文")
                print(f"    票价引文（{kind}）{ticket.quote}")
            if reservation:
                kind = "正文" if ticket.reservation_source_index > len(summaries[name]) else "摘要"
                body_citations += int(kind == "正文")
                print(f"    预约引文（{kind}）{ticket.reservation_quote}")
    print(
        f"\n读取正文 {pages_read}/4 | 匹配参考票价 {correct[0]}→{correct[1]}/4 | "
        f"有引文的预约规则 {bookings[0]}→{bookings[1]}/4 | 直接引用正文 {body_citations} 条"
    )


def test_bocha_ticket_body_live():
    if os.getenv("RUN_BOCHA_TICKET_BODY") != "1":
        pytest.skip("设置 RUN_BOCHA_TICKET_BODY=1 后运行")
    settings = get_settings()
    env = dotenv_values(Path(__file__).resolve().parents[3] / ".env")
    tavily = os.getenv("TAVILY_API_KEY") or env.get("tavily_apikey") or env.get("TAVILY_API_KEY")
    metaso = os.getenv("METASO_API_KEY") or env.get("metaso_apikey") or env.get("mesota_api")
    reader_name, reader_key = ("Tavily", tavily) if tavily else ("秘塔", metaso)
    assert settings.bocha_apikey and settings.openai_api_key, "请配置博查和模型密钥"
    assert reader_key, "请配置 Tavily 或秘塔的网页读取密钥"
    try:
        comparison = asyncio.run(_run(settings.bocha_apikey, reader_name, reader_key))
    except Exception as exc:
        pytest.fail(_error_label(exc), pytrace=False)
    _print_comparison(*comparison)
