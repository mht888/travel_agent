"""百度地图 MCP 的 LangChain Tool：只用于路线矩阵、天气、地理编码。

边界（必须守住）：
    本模块把百度官方 MCP Server 包装成 Agent 可直接调用的工具，但只有三件事：
        1. 路线矩阵   map_distance_matrix
        2. 天气       map_weather
        3. 地理编码   map_geocode
    景点的检索与详情仍然走 Place API v3（backend/providers/baidu_provider.py），
    不要把 map_search_places / map_place_details 也包装进来：
      - MCP 的地点检索固定返回 10 条，没有 page_size / scope=2，也没有评分标签；
      - 接进来会绕开 v3 已经做好的候选池与字段转换。
    其余 MCP 工具（路况、IP 定位、POI 标注）同样不暴露给 Agent。

坐标系：
    REST v3 通过 ret_coordtype=gcj02ll 返回 GCJ-02；
    百度 MCP 底层是旧 REST，默认 BD-09。两侧混用会有几百米偏移，
    所以这里的入参与出参统一为 GCJ-02，调用 MCP 前转换成 BD-09。
    coord_type="bd09" 只给「调用方明确知道自己是 BD-09」的场景用。

进程模型：
    npx 启动的 MCP Server 是长驻子进程，按进程复用一个 ClientSession；
    调用并发由信号量限制，避免 Agent 并行调工具时把同一个 AK 的配额打满。
"""

import asyncio
import json
import logging
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import suppress
from typing import Any, Literal, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.config import get_settings

logger = logging.getLogger(__name__)

_MCP_PACKAGE = "@baidumap/mcp-server-baidu-map"

# 百度批量算路的对数上限。超过上限会被接口直接拒绝，所以按对数分批。
_PAIR_LIMITS: dict[str, int] = {
    "driving": 100,
    "walking": 50,
    "riding": 50,
    "motorcycle": 50,
}

_X_PI = math.pi * 3000.0 / 180.0


# ---------------------------- 坐标系转换 ----------------------------


def _gcj02_to_bd09(lat: float, lng: float) -> tuple[float, float]:
    """GCJ-02 → BD-09（百度接口需要的坐标系）。"""
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * _X_PI)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * _X_PI)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lat, bd_lng


def _bd09_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    """BD-09 → GCJ-02。地理编码返回的是 BD-09，转回来才能和景点坐标对齐。"""
    x = lng - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * _X_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * _X_PI)
    return z * math.sin(theta), z * math.cos(theta)


def bd09_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    """公开入口：服务层组装景点时也要做同样的坐标系转换。"""
    return _bd09_to_gcj02(lat, lng)



# ---------------------------- MCP 会话 ----------------------------


class BaiduMcpError(RuntimeError):
    """MCP 不可用 / 返回错误。retryable 决定 Agent 是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class McpToolCaller(Protocol):
    """Tool 只依赖这一个方法，测试可以注入 fake，不启动 npx。"""

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Any: ...


def _server_params(ak: str):
    """npx 启动百度 MCP Server；Windows 上 npx 是 .cmd，必须经 cmd /c。"""
    from mcp import StdioServerParameters

    env = dict(os.environ)
    env["BAIDU_MAP_API_KEY"] = ak
    if os.name == "nt":
        return StdioServerParameters(
            command="cmd",
            args=["/c", "npx", "-y", _MCP_PACKAGE],
            env=env,
        )
    return StdioServerParameters(
        command="npx", args=["-y", _MCP_PACKAGE], env=env
    )


def _result_payload(result: Any) -> Any:
    """MCP 返回可能是 structured_content，也可能是一段 JSON 文本。"""
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


class BaiduMcpClient:
    """长驻一个 stdio MCP 会话，所有 Tool 共享。"""

    def __init__(
        self,
        *,
        ak: str,
        timeout: float = 30.0,
        startup_timeout: float = 60.0,
        max_concurrency: int = 4,
        package: str = _MCP_PACKAGE,
    ) -> None:
        self._ak = ak.strip()
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._package = package
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._session: Any = None
        self._task: asyncio.Task[None] | None = None
        self._startup_error: BaiduMcpError | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._start_lock = asyncio.Lock()

    async def _serve(self) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - 只在缺依赖时触发
            self._startup_error = BaiduMcpError(
                "缺少 mcp 包，请执行 uv pip install -r requirements.txt",
                retryable=False,
            )
            self._ready.set()
            logger.exception("百度 MCP 依赖缺失")
            return

        try:
            async with stdio_client(_server_params(self._ak)) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._startup_error = None
                    self._ready.set()
                    await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - 启动失败要翻译成 Tool 错误
            if not self._ready.is_set():
                self._startup_error = BaiduMcpError(
                    f"百度 MCP Server 启动失败：{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                self._ready.set()
            else:
                logger.warning("百度 MCP 会话异常退出：%s", exc)
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
                    self._serve(), name="baidu-mcp-session"
                )
            try:
                await asyncio.wait_for(self._ready.wait(), self._startup_timeout)
            except asyncio.TimeoutError as exc:
                raise BaiduMcpError(
                    f"百度 MCP Server 启动超时（{self._startup_timeout:.0f}s）",
                    retryable=True,
                ) from exc
            if self._startup_error is not None:
                raise self._startup_error
            if self._session is None:
                raise BaiduMcpError("百度 MCP Server 未就绪", retryable=True)

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Any:
        if not self._ak:
            raise BaiduMcpError(
                "BAIDU_MAPS_AK 未配置，无法调用百度 MCP", retryable=False
            )
        await self._ensure_started()
        session = self._session
        if session is None:
            raise BaiduMcpError("百度 MCP 会话不可用", retryable=True)

        async with self._semaphore:
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
                raise BaiduMcpError(
                    f"百度 MCP {name} 超时（{self._timeout:.0f}s）",
                    retryable=True,
                ) from exc
            except Exception as exc:  # noqa: BLE001
                # 会话可能已经断开，下一次调用重新拉起
                self._session = None
                raise BaiduMcpError(
                    f"百度 MCP {name} 调用失败：{type(exc).__name__}: {exc}",
                    retryable=True,
                ) from exc

        payload = _result_payload(result)
        if getattr(result, "is_error", False):
            raise BaiduMcpError(
                f"百度 MCP {name} 返回错误：{payload}", retryable=False
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


_client: BaiduMcpClient | None = None


def get_baidu_mcp_client() -> BaiduMcpClient:
    """进程级单例。创建时不启动子进程，第一次调用 Tool 才连接。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = BaiduMcpClient(
            ak=settings.baidu_maps_ak or "",
            timeout=settings.baidu_mcp_timeout_seconds,
            startup_timeout=settings.baidu_mcp_startup_timeout_seconds,
            max_concurrency=settings.baidu_mcp_max_concurrency,
        )
    return _client


async def close_baidu_mcp_client() -> None:
    """FastAPI 退出时调用，避免 npx 子进程被留在后台。"""
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


# ---------------------------- 入参 Schema ----------------------------


class McpCoordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="纬度")
    lng: float = Field(..., ge=-180, le=180, description="经度")


class RouteMatrixInput(BaseModel):
    origins: list[McpCoordinate] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="起点坐标列表，最多 20 个；顺序与返回的 from_index 对应",
    )
    destinations: list[McpCoordinate] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="终点坐标列表，最多 20 个；顺序与返回的 to_index 对应",
    )
    mode: Literal["driving", "walking", "riding", "motorcycle"] = Field(
        "walking",
        description="交通方式；百度批量算路不支持 transit",
    )
    coord_type: Literal["gcj02", "bd09"] = Field(
        "gcj02",
        description="入参坐标系；默认 gcj02，与景点检索结果一致",
    )


class WeatherInput(BaseModel):
    district_id: str | None = Field(
        None,
        description="6 位行政区划代码，如 南宁 450100；与经纬度二选一",
    )
    latitude: float | None = Field(None, ge=-90, le=90, description="纬度")
    longitude: float | None = Field(None, ge=-180, le=180, description="经度")
    coord_type: Literal["gcj02", "bd09"] = Field(
        "gcj02", description="经纬度坐标系；默认 gcj02"
    )


class GeocodeInput(BaseModel):
    address: str = Field(
        ...,
        min_length=1,
        description="结构化地址，如 南宁市青秀区青秀山；地址越完整越准确",
    )


# ---------------------------- 路线矩阵 ----------------------------


def _format_matrix_point(
    point: McpCoordinate, coord_type: str
) -> str:
    """批量算路要求「纬度,经度」。"""
    lat, lng = point.lat, point.lng
    if coord_type == "gcj02":
        lat, lng = _gcj02_to_bd09(lat, lng)
    return f"{lat:.6f},{lng:.6f}"


def _format_weather_location(
    latitude: float, longitude: float, coord_type: str
) -> str:
    """天气接口要求「经度,纬度」，和算路正好相反，别写反。"""
    lat, lng = latitude, longitude
    if coord_type == "gcj02":
        lat, lng = _gcj02_to_bd09(lat, lng)
    return f"{lng:.6f},{lat:.6f}"


def _chunk_with_offset(
    items: Sequence[str], size: int
) -> Iterator[tuple[int, list[str]]]:
    for start in range(0, len(items), size):
        yield start, list(items[start : start + size])


def _edge_from_element(
    element: Any, from_index: int, to_index: int
) -> dict[str, Any] | None:
    if not isinstance(element, dict):
        return None
    duration = element.get("duration") or {}
    distance = element.get("distance") or {}
    seconds = duration.get("value")
    meters = distance.get("value")
    if seconds is None and meters is None:
        return None
    entry: dict[str, Any] = {
        "from_index": from_index,
        "to_index": to_index,
    }
    if seconds is not None:
        entry["duration_seconds"] = seconds
        entry["duration_minutes"] = round(float(seconds) / 60, 1)
    if meters is not None:
        entry["distance_meters"] = meters
    if duration.get("text"):
        entry["duration_text"] = duration["text"]
    if distance.get("text"):
        entry["distance_text"] = distance["text"]
    return entry


def _matrix_entries(
    payload: Any,
    *,
    origin_count: int,
    destination_count: int,
    origin_offset: int,
    destination_offset: int,
) -> list[dict[str, Any]]:
    """兼容两种 results 形状。

    1. 百度文档形状：results 每个元素对应一个 origin，elements 是 destination 数组；
    2. MCP Server 实际形状：把 origin×destination 拍平成一行一对，
       每行的 elements 是单个对象（顺序为 row-major）。
    早先只按第 1 种解析，遇到 3×2 会把 from_index 涨到 5，越界后才在调用方炸掉。
    """
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("缺少 results 数组")

    entries: list[dict[str, Any]] = []

    nested = len(rows) == origin_count and all(
        isinstance(row.get("elements"), list)
        for row in rows
        if isinstance(row, dict)
    )
    if nested:
        for row_index, row in enumerate(rows):
            elements = row.get("elements") if isinstance(row, dict) else None
            if not isinstance(elements, list):
                continue
            for column_index, element in enumerate(elements):
                entry = _edge_from_element(
                    element,
                    origin_offset + row_index,
                    destination_offset + column_index,
                )
                if entry is not None:
                    entries.append(entry)
        return entries

    if len(rows) == origin_count * destination_count:
        for flat_index, row in enumerate(rows):
            element = row.get("elements") if isinstance(row, dict) else None
            entry = _edge_from_element(
                element,
                origin_offset + flat_index // destination_count,
                destination_offset + flat_index % destination_count,
            )
            if entry is not None:
                entries.append(entry)
        return entries

    raise ValueError(
        f"results 行数 {len(rows)} 与 {origin_count}×{destination_count} 不匹配"
    )


async def _fetch_matrix(
    caller: McpToolCaller,
    origins: Sequence[McpCoordinate],
    destinations: Sequence[McpCoordinate],
    mode: str,
    coord_type: str,
) -> tuple[list[dict[str, Any]], int]:
    """按百度对数上限分批调用，返回合并后的有向边。"""
    limit = _PAIR_LIMITS[mode]
    origin_points = [_format_matrix_point(point, coord_type) for point in origins]
    destination_points = [
        _format_matrix_point(point, coord_type) for point in destinations
    ]

    entries: list[dict[str, Any]] = []
    requests = 0
    for destination_offset, destination_chunk in _chunk_with_offset(
        destination_points, limit
    ):
        # 每批的 origins 数量 × destinations 数量必须 <= 对数上限
        origin_chunk_size = max(1, limit // len(destination_chunk))
        for origin_offset, origin_chunk in _chunk_with_offset(
            origin_points, origin_chunk_size
        ):
            payload = await caller.call_tool(
                "map_distance_matrix",
                {
                    "origins": origin_chunk,
                    "destinations": destination_chunk,
                    "mode": mode,
                },
            )
            requests += 1
            entries.extend(
                _matrix_entries(
                    payload,
                    origin_count=len(origin_chunk),
                    destination_count=len(destination_chunk),
                    origin_offset=origin_offset,
                    destination_offset=destination_offset,
                )
            )
    return entries, requests


def _build_route_matrix(caller: McpToolCaller) -> BaseTool:
    async def _run(
        origins: list[McpCoordinate],
        destinations: list[McpCoordinate],
        mode: str = "walking",
        coord_type: str = "gcj02",
    ) -> str:
        try:
            entries, requests = await _fetch_matrix(
                caller, origins, destinations, mode, coord_type
            )
        except BaiduMcpError as exc:
            return _error("mcp_unavailable", str(exc), retryable=exc.retryable)
        except (TypeError, ValueError, KeyError) as exc:
            return _error(
                "upstream_bad_response",
                f"百度 MCP 路线矩阵返回结构异常：{exc}",
                retryable=True,
            )

        expected = len(origins) * len(destinations)
        return _dump(
            {
                "ok": True,
                "coord_type": coord_type,
                "mode": mode,
                "pair_count": expected,
                "requests": requests,
                "incomplete": len(entries) < expected,
                "entries": entries,
                "hint": (
                    "entries 是有向边 from_index -> to_index；"
                    "索引分别对应入参 origins / destinations 的列表下标"
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="mcp_route_matrix",
        description=(
            "百度 MCP 路线矩阵：输入两组 GCJ-02 坐标，返回任意起终点之间的耗时与距离。"
            "仅用于路线矩阵，不用于景点检索或详情（那两条继续用 search_attractions / "
            "get_attraction_details）。mode 只支持 driving / walking / riding / motorcycle，"
            "不支持 transit。超过百度对数上限会自动分批，无需自己拆。"
        ),
        args_schema=RouteMatrixInput,
        handle_validation_error=_handle_validation_error,
    )


# ---------------------------- 天气 ----------------------------


def _compact_now(now: Any) -> dict[str, Any] | None:
    if not isinstance(now, dict):
        return None
    keys = (
        "text",
        "temp",
        "feels_like",
        "rh",
        "wind_class",
        "wind_dir",
        "prec_1h",
        "aqi",
        "uvi",
    )
    return {key: now[key] for key in keys if key in now}


def _compact_forecasts(forecasts: Any) -> list[dict[str, Any]]:
    if not isinstance(forecasts, list):
        return []
    keys = ("date", "week", "text_day", "text_night", "high", "low")
    return [
        {key: item[key] for key in keys if key in item}
        for item in forecasts
        if isinstance(item, dict)
    ]


def _build_weather(caller: McpToolCaller) -> BaseTool:
    async def _run(
        district_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        coord_type: str = "gcj02",
    ) -> str:
        if district_id and (latitude is not None or longitude is not None):
            return _error(
                "invalid_arguments",
                "district_id 与 latitude/longitude 只能二选一",
            )
        if district_id:
            arguments: dict[str, Any] = {"districtId": district_id.strip()}
        elif latitude is not None and longitude is not None:
            arguments = {
                "location": _format_weather_location(
                    latitude, longitude, coord_type
                )
            }
        else:
            return _error(
                "invalid_arguments",
                "必须提供 district_id，或同时提供 latitude 与 longitude",
            )

        try:
            payload = await caller.call_tool("map_weather", arguments)
        except BaiduMcpError as exc:
            return _error("mcp_unavailable", str(exc), retryable=exc.retryable)
        except (TypeError, ValueError, KeyError) as exc:
            return _error(
                "upstream_bad_response",
                f"百度 MCP 天气返回结构异常：{exc}",
                retryable=True,
            )

        if not isinstance(payload, dict):
            return _error(
                "upstream_bad_response", "百度 MCP 天气没有返回对象", retryable=True
            )
        return _dump(
            {
                "ok": True,
                "coord_type": coord_type if not district_id else None,
                "location": payload.get("location"),
                "now": _compact_now(payload.get("now")),
                "forecasts": _compact_forecasts(payload.get("forecasts")),
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="mcp_weather",
        description=(
            "百度 MCP 天气：按 6 位行政区划代码或经纬度查询实时天气与未来 5 天预报。"
            "仅用于天气，不用于景点检索或详情。经纬度按 GCJ-02 传入即可，"
            "内部会处理与百度 BD-09 的差异。"
        ),
        args_schema=WeatherInput,
        handle_validation_error=_handle_validation_error,
    )


# ---------------------------- 地理编码 ----------------------------


def _build_geocode(caller: McpToolCaller) -> BaseTool:
    async def _run(address: str) -> str:
        try:
            payload = await caller.call_tool(
                "map_geocode", {"address": address.strip()}
            )
        except BaiduMcpError as exc:
            return _error("mcp_unavailable", str(exc), retryable=exc.retryable)
        except (TypeError, ValueError, KeyError) as exc:
            return _error(
                "upstream_bad_response",
                f"百度 MCP 地理编码返回结构异常：{exc}",
                retryable=True,
            )

        location = payload.get("location") if isinstance(payload, dict) else None
        if not isinstance(location, dict):
            return _error(
                "geocode_not_found",
                f"没有解析出「{address}」的坐标，请补充更完整的地址",
            )
        try:
            bd_lat = float(location["lat"])
            bd_lng = float(location["lng"])
        except (KeyError, TypeError, ValueError):
            return _error(
                "upstream_bad_response",
                "百度 MCP 地理编码返回的坐标不可解析",
                retryable=True,
            )

        lat, lng = _bd09_to_gcj02(bd_lat, bd_lng)
        return _dump(
            {
                "ok": True,
                "address": address.strip(),
                "location": {"lat": round(lat, 6), "lng": round(lng, 6)},
                "coord_type": "gcj02",
                "level": payload.get("level"),
                "confidence": payload.get("confidence"),
                "precise": payload.get("precise"),
                "hint": "返回坐标已转成 GCJ-02，可直接用于路线矩阵与天气工具",
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="mcp_geocode",
        description=(
            "百度 MCP 地理编码：把结构化地址解析为 GCJ-02 坐标。"
            "仅用于地理编码，不用于景点检索或详情；"
            "返回的 location 可直接传给 mcp_route_matrix / mcp_weather。"
        ),
        args_schema=GeocodeInput,
        handle_validation_error=_handle_validation_error,
    )


# ---------------------------- 组装 ----------------------------


def create_baidu_mcp_tools(
    caller: McpToolCaller | None = None,
) -> list[BaseTool]:
    """只返回三个工具，顺序固定，方便测试与提示词保持一致。"""
    caller = caller or get_baidu_mcp_client()
    return [
        _build_route_matrix(caller),
        _build_weather(caller),
        _build_geocode(caller),
    ]
