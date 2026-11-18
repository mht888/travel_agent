"""高德地图 MCP 的统一工具：只用 gaode_mcp 一个工具覆盖高德 MCP Server 的全部能力。

背景：
    之前酒店/餐厅各有一个直连 Web 服务 API（restapi.amap.com/v5/place/around）的工具；
    现在统一改走高德官方 MCP Server（Streamable HTTP）：
        https://mcp.amap.com/mcp?key=<GAODE_MAPS_AK>
    Agent 只绑定一个工具 gaode_mcp，通过 action 选择具体能力。

action 与高德 MCP 工具的对应：
    around_search   maps_around_search
    text_search     maps_text_search
    poi_detail      maps_search_detail
    geocode         maps_geo
    regeocode       maps_regeocode
    weather         maps_weather
    distance        maps_distance
    route           maps_direction_walking / driving / bicycling / transit_integrated
    ip_location     maps_ip_location
    personal_map    maps_schema_personal_map
    navigate        maps_schema_navi
    taxi            maps_schema_take_taxi

酒店 / 餐厅检索的关键处理：
    MCP 的周边搜索只返回 id / name / address / typecode / photo，
    没有坐标、评分、人均、营业时间；下游 Hotel / Restaurant 组装需要这些字段。
    因此 poi_kind=hotel / restaurant 时，本工具会：
        1. 用 typecode 前缀过滤类别（100* 住宿，05* 餐饮）；
        2. 对前 limit 条逐条调用 maps_search_detail 补齐字段；
        3. 用中心点与详情坐标计算直线距离，并按距离升序返回。
    高德 MCP 详情接口不返回电话与区县，对应字段保持 None，不猜数据。

坐标系：
    高德 MCP 使用 GCJ-02，与项目其他高德/百度转换后的坐标一致。

错误协议与其他 Tool 一致：
    成功 {"ok": true, ...}；失败 {"ok": false, "error_code", "message", "retryable"}。
"""

import asyncio
import json
import logging
import math
from contextlib import suppress
from typing import Any, Literal, Protocol

import httpx2
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.schemas.common import GeoPoint
from backend.tools.tool_results import (
    dump_payload,
    error_result,
    handle_validation_error,
    unexpected_error,
)

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://mcp.amap.com/mcp"

# 高德对免费 Key 有 QPS 限制；详情补齐是 N+1 调用，命中限流时退避重试。
_QPS_HINT = "CUQPS_HAS_EXCEEDED_THE_LIMIT"
_MAX_QPS_RETRIES = 2
# 详情是 N+1 调用，按固定间隔串行补齐，避免短时间超过高德 QPS 限制。
_DETAIL_INTERVAL_SECONDS = 0.25

# typecode 前缀：100* = 住宿服务，05* = 餐饮服务
_POI_KIND_PREFIX = {"hotel": "100", "restaurant": "05"}
_DEFAULT_KEYWORDS = {"hotel": "酒店", "restaurant": "餐厅"}

# 单次返回条数上限（周边搜索原始召回通常为 20 条）
_MAX_LIMIT = 25

Action = Literal[
    "around_search",
    "text_search",
    "poi_detail",
    "geocode",
    "regeocode",
    "weather",
    "distance",
    "route",
    "ip_location",
    "personal_map",
    "navigate",
    "taxi",
]


# ---------------------------- MCP 会话 ----------------------------


class GaodeMcpError(RuntimeError):
    """MCP 不可用 / 返回错误。retryable 决定 Agent 是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class McpToolCaller(Protocol):
    """Tool 只依赖这一个方法，测试可以注入 fake，不发起网络请求。"""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class GaodeMcpClient:
    """长驻一个 Streamable HTTP 会话，所有 action 共享。"""

    def __init__(
        self,
        *,
        api_key: str,
        url: str = _DEFAULT_URL,
        timeout: float = 30.0,
        startup_timeout: float = 30.0,
        max_concurrency: int = 2,
    ) -> None:
        self._api_key = api_key.strip()
        self._url = url.strip() or _DEFAULT_URL
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._session: Any = None
        self._task: asyncio.Task[None] | None = None
        self._startup_error: GaodeMcpError | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._start_lock = asyncio.Lock()
        # 详情接口单独做全局限速：并发搜索时也不会瞬间打满 QPS
        self._detail_lock = asyncio.Lock()
        self._detail_last_call = 0.0

    async def _pace_detail(self) -> None:
        async with self._detail_lock:
            loop = asyncio.get_running_loop()
            wait = _DETAIL_INTERVAL_SECONDS - (loop.time() - self._detail_last_call)
            if wait > 0:
                await asyncio.sleep(min(wait, _DETAIL_INTERVAL_SECONDS))
            self._detail_last_call = loop.time()

    def _endpoint(self) -> str:
        """高德托管端点把 key 放在 query 上，不是 Authorization 头。"""
        separator = "&" if "?" in self._url else "?"
        return f"{self._url}{separator}key={self._api_key}"

    async def _serve(self) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - 只在缺依赖时触发
            self._startup_error = GaodeMcpError(
                "缺少 mcp / httpx2 包，请执行 uv pip install -r requirements.txt",
                retryable=False,
            )
            self._ready.set()
            logger.exception("高德 MCP 依赖缺失")
            return

        try:
            headers = {"Accept": "application/json, text/event-stream"}
            async with httpx2.AsyncClient(headers=headers, timeout=60.0) as http:
                async with streamable_http_client(
                    self._endpoint(), http_client=http
                ) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        self._session = session
                        self._startup_error = None
                        self._ready.set()
                        await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - 启动失败要翻译成 Tool 错误
            if not self._ready.is_set():
                self._startup_error = GaodeMcpError(
                    f"高德 MCP 连接失败：{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                self._ready.set()
            else:
                logger.warning("高德 MCP 会话异常退出：%s", exc)
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
                    self._serve(), name="gaode-mcp-session"
                )
            try:
                await asyncio.wait_for(self._ready.wait(), self._startup_timeout)
            except asyncio.TimeoutError as exc:
                raise GaodeMcpError(
                    f"高德 MCP 连接超时（{self._startup_timeout:.0f}s）",
                    retryable=True,
                ) from exc
            if self._startup_error is not None:
                raise self._startup_error
            if self._session is None:
                raise GaodeMcpError("高德 MCP 会话未就绪", retryable=True)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._api_key:
            raise GaodeMcpError("GAODE_MAPS_AK 未配置，无法调用高德 MCP")
        await self._ensure_started()
        session = self._session
        if session is None:
            raise GaodeMcpError("高德 MCP 会话不可用", retryable=True)

        for attempt in range(_MAX_QPS_RETRIES + 1):
            if name == "maps_search_detail":
                await self._pace_detail()
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
                    raise GaodeMcpError(
                        f"高德 MCP {name} 超时（{self._timeout:.0f}s）",
                        retryable=True,
                    ) from exc
                except Exception as exc:  # noqa: BLE001
                    # 会话是否健康由 _serve 的退出逻辑判断；这里只把错误翻译给 Agent。
                    raise GaodeMcpError(
                        f"高德 MCP {name} 调用失败：{type(exc).__name__}: {exc}",
                        retryable=True,
                    ) from exc

            payload = _result_payload(result)
            if not getattr(result, "is_error", False):
                return payload

            message = str(payload)
            # 高德的 QPS 是硬限流：退避后重试，比把空字段交给下游更有价值。
            if _QPS_HINT in message and attempt < _MAX_QPS_RETRIES:
                await asyncio.sleep(1.0 + attempt * 0.5)
                continue
            raise GaodeMcpError(
                f"高德 MCP {name} 返回错误：{message}",
                retryable=_QPS_HINT in message,
            )

        raise GaodeMcpError(f"高德 MCP {name} 调用失败", retryable=True)

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
    """高德 MCP 返回 JSON 文本；兼容 structuredContent 的客户端实现。"""
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


_client: GaodeMcpClient | None = None


def get_gaode_mcp_client() -> GaodeMcpClient:
    """进程级单例。创建时不连接，第一次调用 Tool 才建立会话。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = GaodeMcpClient(
            api_key=settings.gaode_maps_ak or "",
            url=settings.gaode_mcp_url,
            timeout=settings.gaode_mcp_timeout_seconds,
            startup_timeout=settings.gaode_mcp_startup_timeout_seconds,
            max_concurrency=settings.gaode_mcp_max_concurrency,
        )
    return _client


async def close_gaode_mcp_client() -> None:
    """FastAPI 退出时调用，显式结束远程会话。"""
    global _client
    client, _client = _client, None
    if client is not None:
        await client.aclose()


# ---------------------------- 归一化 DTO ----------------------------


class PoiSummary(BaseModel):
    """归一化后的 POI 条目。金额按项目约定用字符串，避免浮点精度漂移。"""

    id: str
    name: str
    type: str | None = Field(None, description="高德 POI 类型，如 住宿服务;宾馆酒店")
    typecode: str | None = None
    address: str | None = None
    city: str | None = None
    district: str | None = None
    location: GeoPoint | None = None
    distance_meters: int | None = Field(None, description="与中心点的直线距离（米）")
    rating: float | None = Field(None, ge=0, le=5)
    average_cost_cny: str | None = Field(
        None, description="高德返回的人均/参考价（元，字符串）"
    )
    business_area: str | None = None
    tel: str | None = None
    opening_hours_today: str | None = None
    photo_url: str | None = None


# 下游组装层按业务实体命名引用；两者字段一致，共享同一个模型。
HotelSummary = PoiSummary
RestaurantSummary = PoiSummary


# ---------------------------- 字段解析 ----------------------------


def _to_float(value: Any) -> float | None:
    if value is None or value == "" or value == []:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_lng_lat(value: Any) -> GeoPoint | None:
    """高德坐标是 "lng,lat" 字符串；解析失败按未知处理。"""
    if not isinstance(value, str) or "," not in value:
        return None
    lng_text, lat_text = value.split(",", 1)
    try:
        return GeoPoint(lng=float(lng_text), lat=float(lat_text))
    except (TypeError, ValueError):
        return None


def _photo_url(poi: dict[str, Any]) -> str | None:
    photo = poi.get("photo")
    if isinstance(photo, str) and photo.strip():
        return photo.strip()
    photos = poi.get("photos")
    if isinstance(photos, list):
        for item in photos:
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                url = item["url"].strip()
                if url:
                    return url
    return None


def _haversine_meters(center: GeoPoint, point: GeoPoint) -> int:
    """两点球面距离（米）。高德 MCP 不返回 distance，只能自己按坐标算。"""
    radius = 6371000.0
    lat1, lat2 = math.radians(center.lat), math.radians(point.lat)
    delta_lat = lat2 - lat1
    delta_lng = math.radians(point.lng - center.lng)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    )
    return int(round(2 * radius * math.asin(math.sqrt(a))))


def _matches_kind(poi: dict[str, Any], poi_kind: str) -> bool:
    prefix = _POI_KIND_PREFIX.get(poi_kind)
    if prefix is None:
        return True
    typecode = poi.get("typecode")
    if not isinstance(typecode, str) or not typecode:
        return True  # 缺 typecode 时不硬过滤，交给详情结果判断
    # 高德会返回 "100102|050100" 这类多类目拼接，任一段命中即可。
    return any(code.strip().startswith(prefix) for code in typecode.split("|"))


def _light_summary(poi: dict[str, Any]) -> PoiSummary | None:
    poi_id = poi.get("id")
    name = str(poi.get("name") or "").strip()
    if not isinstance(poi_id, str) or not poi_id.strip() or not name:
        return None
    return PoiSummary(
        id=poi_id,
        name=name,
        typecode=str(poi.get("typecode") or "").strip() or None,
        address=str(poi.get("address") or "").strip() or None,
        photo_url=_photo_url(poi),
    )


def _detail_summary(
    poi: dict[str, Any],
    detail: dict[str, Any] | None,
    center: GeoPoint | None,
) -> PoiSummary | None:
    """用周边搜索的轻量条目 + 详情结果拼成完整条目。"""
    base = _light_summary(poi)
    if base is None:
        return None
    if not isinstance(detail, dict):
        return base

    location = _parse_lng_lat(detail.get("location")) or base.location
    rating = _to_float(detail.get("rating"))
    if rating is not None and not 0 <= rating <= 5:
        rating = None
    cost = detail.get("cost")
    average_cost = str(cost).strip() if cost not in (None, "", []) else None
    opening = detail.get("opentime2") or detail.get("open_time")
    tel = detail.get("tel")

    return PoiSummary(
        id=base.id,
        name=str(detail.get("name") or base.name).strip() or base.name,
        type=str(detail.get("type") or "").strip() or None,
        typecode=base.typecode,
        address=str(detail.get("address") or base.address or "").strip() or None,
        city=str(detail.get("city") or "").strip() or None,
        district=str(detail.get("district") or detail.get("adname") or "").strip()
        or None,
        location=location,
        distance_meters=(
            _haversine_meters(center, location)
            if center is not None and location is not None
            else None
        ),
        rating=rating,
        average_cost_cny=average_cost,
        business_area=str(detail.get("business_area") or "").strip() or None,
        tel=str(tel).strip() if tel not in (None, "", []) else None,
        opening_hours_today=str(opening).strip() if opening not in (None, "", []) else None,
        photo_url=_photo_url(detail) or base.photo_url,
    )


# ---------------------------- 入参 Schema ----------------------------


class GaodeMcpInput(BaseModel):
    """gaode_mcp 的统一入参：action 决定用哪些字段，其余留空即可。"""

    action: Action = Field(
        ...,
        description=(
            "要调用的高德能力：around_search / text_search / poi_detail / geocode / "
            "regeocode / weather / distance / route / ip_location / personal_map / "
            "navigate / taxi"
        ),
    )
    poi_kind: Literal["hotel", "restaurant", "any"] = Field(
        "any",
        description=(
            "around_search / text_search 的类别过滤：hotel=住宿(100*)、"
            "restaurant=餐饮(05*)、any=不过滤。酒店/餐厅检索必须显式传 hotel/restaurant"
        ),
    )
    keywords: str | None = Field(
        None,
        description=(
            "around_search / text_search 的关键词，如 酒店 / 民宿 / 餐厅 / 老友粉 / 川菜；"
            "around_search 的 hotel/restaurant 不填时默认 酒店 / 餐厅"
        ),
    )
    longitude: float | None = Field(
        None,
        ge=-180,
        le=180,
        description="中心点/终点经度（GCJ-02）；around_search、regeocode、navigate 必填",
    )
    latitude: float | None = Field(
        None,
        ge=-90,
        le=90,
        description="中心点/终点纬度（GCJ-02）；around_search、regeocode、navigate 必填",
    )
    radius: int = Field(
        2000, ge=100, le=50000, description="around_search 搜索半径（米），默认 2000"
    )
    limit: int = Field(
        10, ge=1, le=_MAX_LIMIT, description="around_search 返回条数上限，默认 10"
    )
    city: str | None = Field(
        None, description="城市名或 adcode；text_search / geocode / weather / transit 用"
    )
    city_limit: bool = Field(
        False, description="text_search 是否限制在城市范围内，默认不限制"
    )
    poi_id: str | None = Field(
        None, description="poi_detail 的高德 POI ID（来自 around_search / text_search）"
    )
    address: str | None = Field(
        None, description="geocode 的结构化地址，如「南宁市青秀区金湖广场」"
    )
    origin: str | None = Field(
        None, description="route / distance / taxi 的起点，格式「经度,纬度」"
    )
    destination: str | None = Field(
        None, description="route / distance / taxi 的终点，格式「经度,纬度」"
    )
    route_mode: Literal["walking", "driving", "bicycling", "transit"] = Field(
        "walking", description="route 的交通方式；transit 需要 city 与 destination_city"
    )
    distance_mode: Literal["driving", "walking", "straight"] = Field(
        "driving", description="distance 的测量方式"
    )
    destination_city: str | None = Field(
        None, description="transit 路径规划的终点城市"
    )
    ip: str | None = Field(None, description="ip_location 的 IP 地址")
    org_name: str | None = Field(
        None, description="personal_map 的专属地图名称"
    )
    line_list: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "personal_map 的行程列表：每项 {title, pointInfoList:[{name,lon,lat,poiId}]}"
        ),
    )
    origin_name: str | None = Field(
        None, description="taxi 的起点名称（可选）"
    )
    destination_name: str | None = Field(
        None, description="taxi 的终点名称，action=taxi 时必填"
    )


# ---------------------------- 参数 → MCP 调用 ----------------------------


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"缺少必填参数 {name}")
    return value.strip() if isinstance(value, str) else value


def _require_lng_lat(longitude: float | None, latitude: float | None) -> str:
    if longitude is None or latitude is None:
        raise ValueError("缺少必填参数 longitude / latitude")
    return f"{longitude:.6f},{latitude:.6f}"


def _build_call(request: GaodeMcpInput) -> tuple[str, dict[str, Any]]:
    """把统一入参翻译成高德 MCP 的工具名与 arguments。"""
    action = request.action

    if action == "around_search":
        location = _require_lng_lat(request.longitude, request.latitude)
        keywords = (request.keywords or "").strip() or _DEFAULT_KEYWORDS.get(
            request.poi_kind, ""
        )
        if not keywords:
            raise ValueError("around_search 需要 keywords（poi_kind=any 时）")
        return "maps_around_search", {
            "keywords": keywords,
            "location": location,
            "radius": str(request.radius),
            "strategy": 0,
        }

    if action == "text_search":
        keywords = _require(request.keywords, "keywords")
        arguments: dict[str, Any] = {"keywords": keywords}
        if request.city:
            arguments["city"] = request.city.strip()
        arguments["citylimit"] = bool(request.city_limit)
        return "maps_text_search", arguments

    if action == "poi_detail":
        return "maps_search_detail", {"id": _require(request.poi_id, "poi_id")}

    if action == "geocode":
        arguments = {"address": _require(request.address, "address")}
        if request.city:
            arguments["city"] = request.city.strip()
        return "maps_geo", arguments

    if action == "regeocode":
        return "maps_regeocode", {
            "location": _require_lng_lat(request.longitude, request.latitude)
        }

    if action == "weather":
        return "maps_weather", {"city": _require(request.city, "city")}

    if action == "distance":
        distance_type = {"driving": "1", "straight": "0", "walking": "3"}
        return "maps_distance", {
            "origins": _require(request.origin, "origin"),
            "destination": _require(request.destination, "destination"),
            "type": distance_type[request.distance_mode],
        }

    if action == "route":
        origin = _require(request.origin, "origin")
        destination = _require(request.destination, "destination")
        if request.route_mode == "walking":
            return "maps_direction_walking", {
                "origin": origin,
                "destination": destination,
            }
        if request.route_mode == "driving":
            return "maps_direction_driving", {
                "origin": origin,
                "destination": destination,
            }
        if request.route_mode == "bicycling":
            return "maps_direction_bicycling", {
                "origin": origin,
                "destination": destination,
            }
        city = _require(request.city, "city")
        cityd = _require(request.destination_city, "destination_city")
        return "maps_direction_transit_integrated", {
            "origin": origin,
            "destination": destination,
            "city": city,
            "cityd": cityd,
        }

    if action == "ip_location":
        return "maps_ip_location", {"ip": _require(request.ip, "ip")}

    if action == "personal_map":
        return "maps_schema_personal_map", {
            "orgName": _require(request.org_name, "org_name"),
            "lineList": _require(request.line_list, "line_list"),
        }

    if action == "navigate":
        if request.longitude is None or request.latitude is None:
            raise ValueError("navigate 需要 longitude / latitude")
        return "maps_schema_navi", {
            "lon": f"{request.longitude:.6f}",
            "lat": f"{request.latitude:.6f}",
        }

    if action == "taxi":
        origin = _parse_lng_lat(_require(request.origin, "origin"))
        destination = _parse_lng_lat(_require(request.destination, "destination"))
        if origin is None or destination is None:
            raise ValueError("taxi 的 origin / destination 必须是「经度,纬度」")
        arguments = {
            "dlon": f"{destination.lng:.6f}",
            "dlat": f"{destination.lat:.6f}",
            "dname": _require(request.destination_name, "destination_name"),
            "slon": f"{origin.lng:.6f}",
            "slat": f"{origin.lat:.6f}",
        }
        if request.origin_name:
            arguments["sname"] = request.origin_name.strip()
        return "maps_schema_take_taxi", arguments

    raise ValueError(f"不支持 action：{action}")


# ---------------------------- around_search 处理 ----------------------------


async def _detail_or_none(
    caller: McpToolCaller, poi_id: str
) -> dict[str, Any] | None:
    """单条详情失败不影响整批结果。"""
    try:
        payload = await caller.call_tool("maps_search_detail", {"id": poi_id})
    except GaodeMcpError as exc:
        logger.debug("高德 POI 详情不可用（%s）：%s", poi_id, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - 单条脏数据不拖垮整批
        logger.debug("高德 POI 详情解析失败（%s）：%s", poi_id, exc)
        return None
    return payload if isinstance(payload, dict) else None


async def _around_response(
    caller: McpToolCaller,
    request: GaodeMcpInput,
    payload: Any,
) -> str:
    pois = payload.get("pois") if isinstance(payload, dict) else None
    if not isinstance(pois, list):
        return error_result(
            "upstream_bad_response",
            "高德 MCP 周边搜索没有返回 pois 数组",
            retryable=True,
        )

    filtered = [
        poi
        for poi in pois
        if isinstance(poi, dict) and _matches_kind(poi, request.poi_kind)
    ][: request.limit]

    center = (
        GeoPoint(lng=request.longitude, lat=request.latitude)
        if request.longitude is not None and request.latitude is not None
        else None
    )

    summaries: list[PoiSummary] = []
    if request.poi_kind == "any":
        for poi in filtered:
            summary = _light_summary(poi)
            if summary is not None:
                summaries.append(summary)
    else:
        # 详情由 client 做全局限速；这里串行拼接，保证条目顺序与距离排序稳定。
        details: list[dict[str, Any] | None] = []
        for poi in filtered:
            poi_id = poi.get("id")
            detail = (
                await _detail_or_none(caller, str(poi_id))
                if isinstance(poi_id, str)
                else None
            )
            details.append(detail)
        for poi, detail in zip(filtered, details):
            summary = _detail_summary(poi, detail, center)
            if summary is not None:
                summaries.append(summary)

    # 详情补齐后按距离升序；缺坐标的排最后，保持原有相对顺序。
    summaries.sort(
        key=lambda item: (
            item.distance_meters is None,
            item.distance_meters or 0,
        )
    )

    items = [
        summary.model_dump(mode="json", exclude_none=True)
        for summary in summaries
    ]
    return dump_payload(
        {
            "ok": True,
            "action": "around_search",
            "poi_kind": request.poi_kind,
            "center": {
                "longitude": request.longitude,
                "latitude": request.latitude,
                "coord_type": "gcj02",
            },
            "radius_meters": request.radius,
            "total": len(items),
            "items": items,
            "source": "amap_mcp",
        }
    )


def _text_response(request: GaodeMcpInput, payload: Any) -> str:
    pois = payload.get("pois") if isinstance(payload, dict) else None
    if not isinstance(pois, list):
        return error_result(
            "upstream_bad_response",
            "高德 MCP 关键词搜索没有返回 pois 数组",
            retryable=True,
        )

    items: list[dict[str, Any]] = []
    for poi in pois:
        if not isinstance(poi, dict) or not _matches_kind(poi, request.poi_kind):
            continue
        summary = _light_summary(poi)
        if summary is not None:
            items.append(summary.model_dump(mode="json", exclude_none=True))
        if len(items) >= request.limit:
            break

    return dump_payload(
        {
            "ok": True,
            "action": "text_search",
            "poi_kind": request.poi_kind,
            "keywords": request.keywords,
            "city": request.city,
            "total": len(items),
            "items": items,
            "source": "amap_mcp",
        }
    )


# ---------------------------- Tool 实现 ----------------------------


def create_gaode_mcp_tool(
    *,
    caller: McpToolCaller | None = None,
    settings: Settings | None = None,
) -> BaseTool:
    """构建唯一的 gaode_mcp 工具。

    caller 由外部注入时（测试 / 复用会话）不再读 AK；
    否则按 settings.gaode_maps_ak 创建默认实现，缺 AK 直接报错。
    """
    if caller is None:
        settings = settings or get_settings()
        ak = (settings.gaode_maps_ak or "").strip()
        if not ak:
            raise ValueError("高德 MCP 工具需要 GAODE_MAPS_AK，请在 .env 中配置")
        caller = get_gaode_mcp_client()

    async def _run(
        action: str,
        poi_kind: str = "any",
        keywords: str | None = None,
        longitude: float | None = None,
        latitude: float | None = None,
        radius: int = 2000,
        limit: int = 10,
        city: str | None = None,
        city_limit: bool = False,
        poi_id: str | None = None,
        address: str | None = None,
        origin: str | None = None,
        destination: str | None = None,
        route_mode: str = "walking",
        distance_mode: str = "driving",
        destination_city: str | None = None,
        ip: str | None = None,
        org_name: str | None = None,
        line_list: list[dict[str, Any]] | None = None,
        origin_name: str | None = None,
        destination_name: str | None = None,
    ) -> str:
        request = GaodeMcpInput(
            action=action,
            poi_kind=poi_kind,
            keywords=keywords,
            longitude=longitude,
            latitude=latitude,
            radius=radius,
            limit=limit,
            city=city,
            city_limit=city_limit,
            poi_id=poi_id,
            address=address,
            origin=origin,
            destination=destination,
            route_mode=route_mode,
            distance_mode=distance_mode,
            destination_city=destination_city,
            ip=ip,
            org_name=org_name,
            line_list=line_list,
            origin_name=origin_name,
            destination_name=destination_name,
        )

        try:
            mcp_name, arguments = _build_call(request)
        except ValueError as exc:
            return error_result("invalid_arguments", str(exc))

        try:
            payload = await caller.call_tool(mcp_name, arguments)
        except GaodeMcpError as exc:
            return error_result("mcp_unavailable", str(exc), retryable=exc.retryable)
        except (TypeError, ValueError, KeyError) as exc:
            return error_result(
                "upstream_bad_response",
                f"高德 MCP 返回结构异常：{exc}",
                retryable=True,
            )
        except Exception as exc:  # noqa: BLE001 - Tool 不能把异常抛回 Agent 循环
            return unexpected_error("gaode_mcp", exc)

        if request.action == "around_search":
            return await _around_response(caller, request, payload)
        if request.action == "text_search":
            return _text_response(request, payload)

        return dump_payload(
            {
                "ok": True,
                "action": request.action,
                "source": "amap_mcp",
                "data": payload,
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="gaode_mcp",
        description=(
            "高德地图 MCP 统一入口，一个工具覆盖高德 MCP Server 全部能力。"
            "action 取值：around_search（周边 POI）、text_search（关键词搜）、"
            "poi_detail（POI 详情）、geocode（地址转坐标）、regeocode（坐标转地址）、"
            "weather（天气）、distance（距离测量）、route（步行/驾车/骑行/公交路径规划）、"
            "ip_location（IP 定位）、personal_map（生成专属地图）、"
            "navigate（导航唤端链接）、taxi（打车唤端链接）。"
            "周边/关键词搜索酒店时必须传 poi_kind=hotel，找餐厅必须传 "
            "poi_kind=restaurant；around_search 会自动按类别过滤、补齐坐标与评分并按距离排序。"
            "坐标一律为 GCJ-02。"
        ),
        args_schema=GaodeMcpInput,
        handle_validation_error=handle_validation_error,
    )


def create_gaode_mcp_tools(
    *,
    caller: McpToolCaller | None = None,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """registry 入口：与其他分组保持「工厂返回列表」的形状。"""
    return [create_gaode_mcp_tool(caller=caller, settings=settings)]


__all__ = [
    "GaodeMcpClient",
    "GaodeMcpError",
    "GaodeMcpInput",
    "HotelSummary",
    "McpToolCaller",
    "PoiSummary",
    "RestaurantSummary",
    "close_gaode_mcp_client",
    "create_gaode_mcp_tool",
    "create_gaode_mcp_tools",
    "get_gaode_mcp_client",
]
