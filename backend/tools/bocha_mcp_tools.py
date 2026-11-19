"""博查 Web Search MCP（Streamable HTTP）的 LangChain Tool：只用于时效性网页信息。

边界（必须守住）：
    只暴露 bocha_web_search，不暴露 bocha_ai_search：
      - 两者入参完全相同，AI Search 贵一倍（0.06 vs 0.036 元/次）；
      - 同时 bind 两个语义重复的工具，模型很容易选错，成本悄悄翻倍。
    MCP 返回的是一整段 "Title:/URL:/Description:/Published date:/Site name:" 文本，
    不是 JSON，所以这里负责解析成结构化 results，并做去重与截断。

与其他工具的分工：
    城市地点线索（景点/街区/市场等）→ 本工具，由景点初选 Agent 按兴趣检索；
    坐标、路线、天气 → 高德 MCP。

传输：
    博查托管的是 Streamable HTTP（https://mcp.bochaai.com/mcp），没有本地子进程，
    鉴权走 Authorization: Bearer。断线比 stdio 频繁，所以会话失效后下次调用重连。
"""

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.services.usage import count_external_request

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://mcp.bochaai.com/mcp"
_WEB_SEARCH_TOOL = "bocha_web_search"

# 通用链路固定返回 5 条：不暴露给模型选，避免一次要 10 条把上下文翻倍。
_MAX_RESULTS = 5

# 景点初选链路：博查不支持翻页，只能靠放大 count 拿新结果。
# 首次 10 条，同一 query 重复检索时按 10 / 20 / 40 / 50 递增。
_STRICT_MIN_COUNT = 10
_MAX_FETCH_RESULTS = 50

# 单条摘要上限。5 条 × 600 字 ≈ 3k 字符：中文约 200 字，足够读完结论，
# 又不会让多次搜索把上下文顶爆。
_SNIPPET_LIMIT = 500

# freshness 只接受这几种写法；合法区间用 .. 连接，不要发明 "lastFiveYears"。
_FRESHNESS_RE = re.compile(
    r"^(?:noLimit|oneDay|oneWeek|oneMonth|oneYear|"
    r"\d{4}-\d{2}-\d{2}(?:\.\.\d{4}-\d{2}-\d{2})?)$"
)

_FIELD_RE = {
    name: re.compile(rf"^{re.escape(name)}:\s*(.*)$", re.MULTILINE)
    for name in ("Title", "URL", "Published date", "Site name")
}


# ---------------------------- MCP 会话 ----------------------------


class BochaMcpError(RuntimeError):
    """MCP 不可用 / 返回错误。retryable 决定 Agent 是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class McpToolCaller(Protocol):
    """Tool 只依赖这一个方法，测试可以注入 fake，不发起网络请求。"""

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Any: ...


class BochaMcpClient:
    """长驻一个 Streamable HTTP 会话，所有 Tool 共享。"""

    def __init__(
        self,
        *,
        api_key: str,
        url: str = _DEFAULT_URL,
        timeout: float = 20.0,
        startup_timeout: float = 20.0,
        max_concurrency: int = 3,
    ) -> None:
        self._api_key = api_key.strip()
        self._url = url.strip() or _DEFAULT_URL
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._session: Any = None
        self._task: asyncio.Task[None] | None = None
        self._startup_error: BochaMcpError | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._start_lock = asyncio.Lock()

    async def _serve(self) -> None:
        try:
            import httpx2
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - 只在缺依赖时触发
            self._startup_error = BochaMcpError(
                "缺少 mcp / httpx2 包，请执行 uv pip install -r requirements.txt",
                retryable=False,
            )
            self._ready.set()
            logger.exception("博查 MCP 依赖缺失")
            return

        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json, text/event-stream",
            }
            # read 超时留大：SSE 长连接可能被服务端持有较久
            async with httpx2.AsyncClient(headers=headers, timeout=60.0) as http_client:
                async with streamable_http_client(
                    self._url, http_client=http_client
                ) as streams:
                    read_stream, write_stream = streams[0], streams[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        self._session = session
                        self._startup_error = None
                        self._ready.set()
                        await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - 启动失败要翻译成 Tool 错误
            if not self._ready.is_set():
                self._startup_error = BochaMcpError(
                    f"博查 MCP 连接失败：{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                self._ready.set()
            else:
                logger.warning("博查 MCP 会话异常退出：%s", exc)
        finally:
            self._session = None

    async def _ensure_started(self) -> None:
        if self._session is not None:
            return
        async with self._start_lock:
            if self._session is not None:
                return
            if self._task is None or self._task.done():
                self._ready.clear()
                self._stop.clear()
                self._startup_error = None
                self._task = asyncio.create_task(
                    self._serve(), name="bocha-mcp-session"
                )
            try:
                await asyncio.wait_for(self._ready.wait(), self._startup_timeout)
            except asyncio.TimeoutError as exc:
                raise BochaMcpError(
                    f"博查 MCP 连接超时（{self._startup_timeout:.0f}s）",
                    retryable=True,
                ) from exc
            if self._startup_error is not None:
                raise self._startup_error
            if self._session is None:
                raise BochaMcpError("博查 MCP 会话未就绪", retryable=True)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._api_key:
            raise BochaMcpError("bocha_apikey 未配置，无法调用博查 MCP")
        await self._ensure_started()
        session = self._session
        if session is None:
            raise BochaMcpError("博查 MCP 会话不可用", retryable=True)

        async with self._semaphore:
            count_external_request("bocha_mcp")
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        name,
                        arguments,
                        read_timeout_seconds=self._timeout,
                    ),
                    self._timeout + 5,
                )
            except asyncio.TimeoutError as exc:
                raise BochaMcpError(
                    f"博查 MCP {name} 超时（{self._timeout:.0f}s）",
                    retryable=True,
                ) from exc
            except Exception as exc:  # noqa: BLE001
                # 远程会话可能已经断开，下一次调用重新建立
                self._session = None
                raise BochaMcpError(
                    f"博查 MCP {name} 调用失败：{type(exc).__name__}: {exc}",
                    retryable=True,
                ) from exc

        payload = _result_payload(result)
        if getattr(result, "is_error", False):
            raise BochaMcpError(
                f"博查 MCP {name} 返回错误：{payload}", retryable=False
            )
        return payload

    async def aclose(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        self._session = None
        if task is None:
            return
        try:
            await asyncio.wait_for(task, 10)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


def _result_payload(result: Any) -> Any:
    """博查没有 structuredContent，正常会走到 content 文本这条分支。"""
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured:
        return structured

    chunks: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    raw = "\n".join(chunks)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


_client: BochaMcpClient | None = None


def get_bocha_mcp_client() -> BochaMcpClient:
    """进程级单例。创建时不连接，第一次调用 Tool 才建立会话。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = BochaMcpClient(
            api_key=settings.bocha_apikey or "",
            url=settings.bocha_mcp_url,
            timeout=settings.bocha_mcp_timeout_seconds,
            max_concurrency=settings.bocha_mcp_max_concurrency,
        )
    return _client


async def close_bocha_mcp_client() -> None:
    """FastAPI 退出时调用，显式结束远程会话。"""
    global _client
    client, _client = _client, None
    if client is not None:
        await client.aclose()


# ---------------------------- 序列化与错误 ----------------------------


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(
    error_code: str,
    message: str,
    *,
    retryable: bool = False,
    **extra: Any,
) -> str:
    return _dump(
        {
            "ok": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            **extra,
        }
    )


def _handle_validation_error(error: Exception) -> str:
    return _error("invalid_arguments", f"参数校验失败：{error}")


# ---------------------------- 返回文本解析 ----------------------------


def _line(block: str, name: str) -> str:
    match = _FIELD_RE[name].search(block)
    return match.group(1).strip() if match else ""


def _description(block: str) -> str:
    """Description 跨多行，截到 Published date 之前。"""
    parts = block.split("Description:", 1)
    if len(parts) != 2:
        return ""
    rest = parts[1]
    stop = re.search(r"^Published date:", rest, re.MULTILINE)
    if stop:
        rest = rest[: stop.start()]
    return " ".join(rest.split())


def _parse_items(text: str) -> list[dict[str, str]]:
    """把一整段多记录文本切成结构化条目。"""
    if not text.strip():
        return []
    items: list[dict[str, str]] = []
    for block in re.split(r"\n(?=Title:\s)", text.strip()):
        title = _line(block, "Title")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "url": _line(block, "URL"),
                "snippet": _description(block),
                "published_at": _line(block, "Published date"),
                "site": _line(block, "Site name"),
            }
        )
    return items


def _canonical_key(item: dict[str, str]) -> str:
    """同一篇文章常以 m.163.com / c.m.163.com 两个地址返回。

    URL 归一化去不掉，只能取路径最后一段做文章 ID（如 KB8U0MKF0530QRMB）。
    """
    path = item["url"].split("?", 1)[0].split("#", 1)[0].rstrip("/")
    tail = path.rsplit("/", 1)[-1]
    stem = tail.rsplit(".", 1)[0] if "." in tail else tail
    if len(stem) >= 8 and re.fullmatch(r"[A-Za-z0-9_-]+", stem):
        return stem.lower()
    return path.lower() or item["title"].lower()


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = _canonical_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _truncate(text: str, limit: int = _SNIPPET_LIMIT) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


# ---------------------------- 入参 Schema ----------------------------


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="搜索关键词，建议带上城市、主题和时间，如「南宁 夜市 热门」",
    )
    freshness: str = Field(
        "noLimit",
        description=(
            "结果发布时间范围：noLimit / oneDay / oneWeek / oneMonth / oneYear / "
            "YYYY-MM-DD / YYYY-MM-DD..YYYY-MM-DD；没有 fiveYears 这类写法"
        ),
    )


class SelectionWebSearchInput(BaseModel):
    """景点初选专用：query 只允许固定模板，避免模型自行扩写关键词。"""

    query: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description=(
            "固定格式：{city} {interest} 景点 推荐，例如「南宁 美食 景点 推荐」。"
            "必须原样使用，不得添加其他关键词、地名或调整顺序；"
            "城市与兴趣内部不要含空格。"
        ),
    )
    freshness: str = Field(
        "noLimit",
        description=(
            "保持 noLimit 即可；该链路只按兴趣发现地点，不做时效过滤"
        ),
    )
    count: int = Field(
        10,
        ge=5,
        le=50,
        description=(
            "本次抓取的网页条数：首次 10；extract_web_places 返回 "
            "insufficient_places 时，用同一 query 传更大的 count"
            "（20/30/40/50）重试；同一 query 重复调用时工具也会自动放大"
        ),
    )


# 严格模式：城市 + 兴趣 + 景点 + 推荐，四段且顺序固定
_SELECTION_QUERY_RE = re.compile(r"^\S+\s+\S+\s+景点\s+推荐$")


# ---------------------------- Web Search ----------------------------


class WebSearchResultSink(Protocol):
    """网页结果缓存接口：Tool 只负责写入，如何存储由调用方决定。"""

    def save(self, query: str, results: Sequence[dict[str, Any]]) -> str: ...


def _build_web_search(
    caller: McpToolCaller,
    result_cache: WebSearchResultSink | None = None,
    *,
    strict_query: bool = False,
) -> BaseTool:
    # 严格模式下同一 query 的调用次数：用于自动放大 count。
    attempts: dict[str, int] = {}

    async def _run(
        query: str,
        freshness: str = "noLimit",
        count: int = 10,
    ) -> str:
        normalized = " ".join(query.split())
        if strict_query and not _SELECTION_QUERY_RE.match(normalized):
            return _error(
                "invalid_query_format",
                "query 必须严格为「城市 兴趣 景点 推荐」，"
                "例如「南宁 美食 景点 推荐」；"
                f"实际收到「{normalized}」，请去掉额外关键词后重试",
                retryable=True,
            )
        if not _FRESHNESS_RE.match(freshness.strip()):
            return _error(
                "invalid_arguments",
                f"freshness 不支持「{freshness}」；"
                "可用 noLimit / oneDay / oneWeek / oneMonth / oneYear / "
                "YYYY-MM-DD / YYYY-MM-DD..YYYY-MM-DD，"
                "例如近 5 年写 2021-11-15..2026-11-15",
            )

        if strict_query:
            # 博查不支持翻页：同一 query 再搜时只能提高 count 才能拿到新结果。
            # 显式传入更大的 count 时以调用方为准，否则按 10 / 20 / 40 / 50 递增。
            key = normalized.casefold()
            attempt = attempts.get(key, 0) + 1
            attempts[key] = attempt
            requested = max(_STRICT_MIN_COUNT, min(_MAX_FETCH_RESULTS, count))
            escalated = _STRICT_MIN_COUNT * (2 ** (attempt - 1))
            fetch_count = min(_MAX_FETCH_RESULTS, max(requested, escalated))
            result_limit = fetch_count
        else:
            attempt = 0
            fetch_count = min(_MAX_FETCH_RESULTS, _MAX_RESULTS + 2)
            result_limit = _MAX_RESULTS

        # 通用链路多取 2 条抵消重复；景点初选按 count 全量取，供抽取工具消费
        arguments = {
            "query": normalized,
            "count": fetch_count,
            "freshness": freshness.strip(),
        }
        try:
            payload = await caller.call_tool(_WEB_SEARCH_TOOL, arguments)
        except BochaMcpError as exc:
            return _error("mcp_unavailable", str(exc), retryable=exc.retryable)
        except (TypeError, ValueError, KeyError) as exc:
            return _error(
                "upstream_bad_response",
                f"博查 MCP 返回结构异常：{exc}",
                retryable=True,
            )

        text = payload if isinstance(payload, str) else _dump(payload or {})
        items = _dedupe(_parse_items(text))[:result_limit]

        # 缓存始终保留完整摘要，抽取工具需要正文线索；
        # 景点初选只把标题/URL 回给 Agent，避免补检索把上下文顶爆。
        cached_results: list[dict[str, Any]] = [
            {
                "title": item["title"],
                "url": item["url"],
                "snippet": _truncate(item["snippet"]),
                "published_at": item["published_at"] or None,
                "site": item["site"] or None,
            }
            for item in items
        ]
        if strict_query:
            results = [
                {
                    "title": row["title"],
                    "url": row["url"],
                    "published_at": row["published_at"],
                    "site": row["site"],
                }
                for row in cached_results
            ]
        else:
            results = cached_results

        if not results:
            empty: dict[str, Any] = {
                "ok": True,
                "query": normalized,
                "freshness": freshness.strip(),
                "count": 0,
                "results": [],
                "hint": "没有检索到结果，可以换更宽泛的关键词或放宽 freshness",
            }
            if strict_query:
                empty["attempt"] = attempt
            return _dump(empty)

        web_search_id = (
            result_cache.save(normalized, cached_results)
            if result_cache is not None and cached_results
            else None
        )

        hint = (
            "results 是网页摘要，不是权威结构化数据；"
            "关键结论请带上 url 与 published_at，冲突时以官方站点为准"
        )
        if strict_query and web_search_id:
            hint = (
                f"已抓取 {len(results)} 条并写入缓存；把 web_search_id 交给 extract_web_places。"
                "若它返回 insufficient_places，用同一 query 再次检索"
                "（count 会自动放大，也可显式传更大的 count），然后重新抽取。"
            )
        elif web_search_id:
            hint = (
                "results 是网页摘要，不是权威结构化数据；"
                "如需从这些结果中抽取地点，把 web_search_id 交给 extract_web_places；"
                "关键结论请带上 url 与 published_at，冲突时以官方站点为准"
            )

        response: dict[str, Any] = {
            "ok": True,
            "query": normalized,
            "freshness": freshness.strip(),
            "count": len(results),
            "results": results,
            "web_search_id": web_search_id,
            "hint": hint,
        }
        if strict_query:
            response["attempt"] = attempt
        return _dump(response)

    description = (
        "博查网页搜索：按关键词检索网页，返回标题、URL、摘要、发布时间。"
        "可用于发现城市地点线索（景点、街区、市场、博物馆等）、近期口碑与攻略，"
        "也可查临时闭馆、节假日活动、票价与预约政策。"
        "坐标、距离、天气仍需专用工具。"
        "结果是网页摘要而非权威数据，引用时必须附上 URL 与发布时间。"
    )
    if strict_query:
        description = (
            "博查网页搜索（景点初选用）：query 必须是"
            "「城市 兴趣 景点 推荐」固定格式，例如「南宁 美食 景点 推荐」。"
            "返回标题、URL、发布时间和 web_search_id；网页正文留在缓存。"
            "同一 query 重复调用会自动放大抓取条数（最多 50），"
            "用于地点不足时补充检索。"
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="mcp_web_search",
        description=description,
        args_schema=SelectionWebSearchInput if strict_query else WebSearchInput,
        handle_validation_error=_handle_validation_error,
    )


# ---------------------------- 组装 ----------------------------


def create_bocha_mcp_tools(
    caller: McpToolCaller | None = None,
    *,
    result_cache: WebSearchResultSink | None = None,
    strict_query: bool = False,
) -> list[BaseTool]:
    """只返回一个工具：不暴露 AI Search，避免模型选到贵一倍的那个。

    result_cache 由调用方注入时，搜索成功会顺带返回 web_search_id，
    供抽取工具复用网页结果，而不是把摘要重新塞回上下文。

    strict_query 供景点初选链路使用：query 会被校验为
    「城市 兴趣 景点 推荐」，不符合直接返回错误让 Agent 重试；
    同一 query 重复检索会自动放大 count（10→20→40→50），
    让「地点不足 → 补检索」循环能拿到新网页；
    其他画像保持自由关键词。
    """
    caller = caller or get_bocha_mcp_client()
    return [
        _build_web_search(
            caller,
            result_cache,
            strict_query=strict_query,
        )
    ]
