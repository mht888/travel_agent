"""用同一组成都景点对照 Tavily 和秘塔 MCP。真实调用默认跳过。"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import httpx
import httpx2
import pytest
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from backend.config import get_settings
from backend.tests.attractions.test_chengdu_ticket_prices_live import (
    ATTRACTIONS,
    PROMPT,
    SITE_PROMPT,
    OfficialSites,
    TicketReport,
    _hostname,
    _source_at,
    _supports_quote,
)

PROVIDERS = {
    "Tavily": ("https://mcp.tavily.com/mcp/", "tavily_search", "tavily_extract"),
    "秘塔": ("https://metaso.cn/api/mcp", "metaso_web_search", "metaso_web_reader"),
}


def _error_label(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return ", ".join(_error_label(child) for child in exc.exceptions)
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return f"{type(exc).__name__} HTTP {status}" if status else type(exc).__name__


@asynccontextmanager
async def _connect(url, key):
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream"}
    async with httpx2.AsyncClient(headers=headers, timeout=25, trust_env=False) as client:
        async with streamable_http_client(url, http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session


def _payloads(result):
    if result.is_error:
        raise RuntimeError("MCP tool returned an error")
    if result.structured_content:
        return [result.structured_content]
    payloads = []
    for item in result.content:
        if item.type != "text":
            continue
        try:
            payloads.append(json.loads(item.text))
        except ValueError:
            payloads.append(item.text)
    return payloads


def _search_rows(payload):
    if isinstance(payload, list):
        return [row for item in payload for row in _search_rows(item)]
    if not isinstance(payload, dict):
        return []
    url = payload.get("url") or payload.get("link")
    if not url:
        return _search_rows(
            payload.get("results") or payload.get("webpages") or payload.get("data") or []
        )
    return [{
        "url": url,
        "title": payload.get("title") or "",
        "snippet": str(payload.get("content") or payload.get("snippet") or payload.get("summary") or "")[:1400],
    }]


async def _search(session, name, tool, query, domain=""):
    args = {"query": query, "max_results": 5, "search_depth": "basic"}
    if name == "Tavily" and domain:
        args["include_domains"] = [domain]
    if name == "秘塔":
        args = {"q": f"site:{domain} {query}" if domain else query, "scope": "webpage", "size": 5}
    async with asyncio.timeout(35):
        rows = _search_rows(_payloads(await session.call_tool(tool, arguments=args)))
    return [
        row for row in rows if _hostname(row["url"]) and (
            not domain or _hostname(row["url"]) == domain
            or _hostname(row["url"]).endswith(f".{domain}")
        )
    ][:5]


async def _read(session, name, tool, url):
    args = {"urls": [url]} if name == "Tavily" else {"url": url}
    async with asyncio.timeout(35):
        payloads = _payloads(await session.call_tool(tool, arguments=args))
    for payload in payloads:
        if isinstance(payload, str):
            return payload[:9000]
        if not isinstance(payload, dict):
            continue
        if name == "Tavily":
            results = payload.get("results") or []
            return (results[0].get("raw_content") or "")[:9000] if results else ""
        data = payload.get("data") or payload
        if isinstance(data, dict):
            return (data.get("content") or data.get("markdown") or "")[:9000]
        if isinstance(data, str):
            return data[:9000]
    return ""


async def _extract(model, schema, prompt, evidence):
    if not evidence or not any(evidence.values()):
        return {}
    result = await model.with_structured_output(schema, method="json_mode").ainvoke(
        [("system", prompt), ("human", json.dumps(evidence, ensure_ascii=False))]
    )
    if schema is OfficialSites:
        return {item.attraction: item for item in result.sites}
    return {item.attraction: item for item in result.tickets}


def _verified(ticket, rows):
    if ticket is None:
        return None, None
    price = _source_at(rows, ticket.source_index)
    reservation = _source_at(rows, ticket.reservation_source_index)
    return (
        (ticket.price_cny, price["url"])
        if ticket.price_cny is not None and _supports_quote(price, ticket.quote.strip()) else None,
        (ticket.reservation, reservation["url"])
        if ticket.reservation and _supports_quote(reservation, ticket.reservation_quote.strip()) else None,
    )


async def _benchmark(name, key, model):
    url, search_tool, read_tool = PROVIDERS[name]
    started = perf_counter()
    failures = []
    reads = []
    search_seconds = 0.0
    read_seconds = 0.0

    async def judge(stage, schema, prompt, rows):
        try:
            return await _extract(model, schema, prompt, rows)
        except Exception as exc:
            failures.append(f"{stage} {_error_label(exc)}")
            return {}

    async with _connect(url, key) as session:
        async def query(place, text, domain=""):
            nonlocal search_seconds
            call_started = perf_counter()
            try:
                return await _search(session, name, search_tool, text, domain)
            except Exception as exc:
                failures.append(f"{place} {_error_label(exc)}")
                return []
            finally:
                search_seconds += perf_counter() - call_started

        discovery = {place: await query(place, f"成都 {place} 官方网站 官网") for place in ATTRACTIONS}
        print(f"{name} | 官网搜索完成 | {sum(bool(rows) for rows in discovery.values())}/7 有候选", flush=True)
        sites = await judge("官网判定", OfficialSites, SITE_PROMPT, discovery)
        domains = {}
        for place, site in sites.items():
            if place in discovery:
                source = _source_at(discovery[place], site.source_index)
                domains[place] = _hostname(source.get("url") or "")
        evidence = {
            place: await query(place, f"成都 {place} 门票价格 参观预约", domains.get(place, ""))
            for place in ATTRACTIONS
        }
        prompt = PROMPT.replace("博查搜索摘要", "搜索摘要或网页正文")
        print(f"{name} | 票价搜索完成 | {sum(bool(rows) for rows in evidence.values())}/7 有候选", flush=True)
        snippet_tickets = await judge("摘要提取", TicketReport, prompt, evidence)
        snippets = {place: _verified(snippet_tickets.get(place), rows) for place, rows in evidence.items()}
        full_pages = {}
        for place, rows in evidence.items():
            if not rows or all(snippets[place]):
                continue
            source = next((row for row in rows if any(
                word in row["title"] for word in ("票务", "门票", "参观", "预约")
            )), rows[0])
            read_started = perf_counter()
            try:
                content = await _read(session, name, read_tool, source["url"])
            except Exception as exc:
                failures.append(f"{place} 正文 {_error_label(exc)}")
                content = ""
            read_seconds += perf_counter() - read_started
            reads.append((place, bool(content)))
            if content:
                full_pages[place] = [*rows, {**source, "snippet": content}]
        full_tickets = await judge("正文提取", TicketReport, prompt, full_pages)

    final = {}
    for place in ATTRACTIONS:
        price, reservation = snippets[place]
        full_price, full_reservation = _verified(full_tickets.get(place), full_pages.get(place, []))
        final[place] = (price or full_price, reservation or full_reservation)
    return {
        "domains": domains,
        "discovery": {place: len(rows) for place, rows in discovery.items()},
        "counts": {place: len(rows) for place, rows in evidence.items()},
        "evidence": evidence,
        "snippets": snippets,
        "final": final,
        "reads": reads,
        "failures": failures,
        "search_seconds": search_seconds,
        "read_seconds": read_seconds,
        "elapsed": perf_counter() - started,
    }


async def _run(keys):
    settings = get_settings()
    async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
        model = ChatOpenAI(
            api_key=settings.openai_api_key, base_url=settings.openai_base_url,
            model=settings.openai_model, temperature=0, timeout=90, max_retries=1,
            http_async_client=client,
        )
        return await asyncio.gather(
            *(_benchmark(name, key, model) for name, key in keys.items()), return_exceptions=True,
        )


def test_chengdu_search_providers_live():
    if os.getenv("RUN_CHENGDU_SEARCH_COMPARE") != "1":
        pytest.skip("设置 RUN_CHENGDU_SEARCH_COMPARE=1 后调用两个真实 MCP")
    env = dotenv_values(Path(__file__).resolve().parents[3] / ".env")
    keys = {
        "Tavily": os.getenv("TAVILY_API_KEY") or env.get("tavily_apikey") or env.get("TAVILY_API_KEY"),
        "秘塔": os.getenv("METASO_API_KEY") or env.get("metaso_apikey") or env.get("METASO_API_KEY")
        or env.get("mesota_api"),
    }
    assert all(keys.values()), "请配置 tavily_apikey 和 metaso_apikey"
    assert get_settings().openai_api_key, "请配置 OPENAI_API_KEY"

    issues = []
    for name, report in zip(keys, asyncio.run(_run(keys)), strict=True):
        if isinstance(report, BaseException):
            issues.append(f"{name} {_error_label(report)}")
            print(f"{name} | 失败 | {_error_label(report)}")
            continue
        print(f"\n--- {name} ---")
        for place, (price, reservation) in report["final"].items():
            print(
                f"{place} | 官网 {report['domains'].get(place) or '未知'} | "
                f"候选 {report['discovery'][place]}/{report['counts'][place]} | "
                f"票价 {price or '未知'} | 预约 {reservation or '未知'}"
            )
        snippets, final = report["snippets"], report["final"]
        official_count = (
            "判定失败" if any(failure.startswith("官网判定 ") for failure in report["failures"])
            else f"{sum(bool(domain) for domain in report['domains'].values())}/7"
        )
        print(
            f"汇总 | 官网 {official_count} | "
            f"票价 {sum(bool(p) for p, _ in snippets.values())}→{sum(bool(p) for p, _ in final.values())}/7 | "
            f"预约 {sum(bool(r) for _, r in snippets.values())}→{sum(bool(r) for _, r in final.values())}/7 | "
            f"正文 {sum(ok for _, ok in report['reads'])}/{len(report['reads'])} | "
            f"搜索 {report['search_seconds']:.1f}s | 正文 {report['read_seconds']:.1f}s | "
            f"总耗时 {report['elapsed']:.1f}s | 请求异常 {report['failures']}"
        )
        if not any(report["discovery"].values()) and not any(report["counts"].values()):
            issues.append(f"{name} 没有返回可用搜索结果")
        if any(failure.startswith(("官网判定 ", "摘要提取 ", "正文提取 ")) for failure in report["failures"]):
            issues.append(f"{name} 有模型判定阶段失败")
            for place, rows in report["evidence"].items():
                for row in rows[:2]:
                    candidate = {field: row[field] for field in ("title", "url")}
                    print(f"{place} | 候选 {json.dumps(candidate, ensure_ascii=False)}")
    if issues:
        pytest.fail(" / ".join(issues), pytrace=False)
