"""高德 POI 2.0 · 周边搜索：以某个中心点找周边酒店。

职责边界（这一阶段只做 Tool，不接 Agent）：
    中心点坐标由上游（景点初选结果）提供，本工具不负责查景点；
    工具只做「参数校验 → 调高德 → 字段归一化 → 返回 JSON」，
    不判断距离是否合适、也不替用户选酒店。

坐标系：
    高德 Web 服务 API 使用 GCJ-02，和景点组装阶段（百度 geocode 转 GCJ-02）一致，
    不需要再做坐标转换。

接口：
    GET https://restapi.amap.com/v5/place/around
    - types=100000 表示「住宿服务」，够覆盖宾馆酒店与民宿；
    - page_size 上限 25，一页足够给 Agent 做初筛；
    - show_fields=business,photos 才会返回评分、人均、电话与图片。

错误协议与其他 Tool 一致：
    成功 {"ok": true, ...}；失败 {"ok": false, "error_code", "message", "retryable"}。
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import httpx
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

_AROUND_URL = "https://restapi.amap.com/v5/place/around"

# 100000 = 住宿服务（宾馆酒店 / 星级酒店 / 经济型 / 民宿等）
_HOTEL_TYPES = "100000"

# 这些 infocode 属于限流，重试可能恢复；其余（KEY 无效、无权限、配额用尽）重试无意义。
_RETRYABLE_INFOCODES = frozenset({"10004", "10021", "10045"})


class GaodeApiError(RuntimeError):
    """高德接口不可用或业务返回失败。retryable 决定 Agent 是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class HotelSummary(BaseModel):
    """归一化后的酒店条目。金额按项目约定用字符串，避免浮点精度漂移。"""

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


# ---------------------------- 高德客户端 ----------------------------


class HotelSearcher(Protocol):
    """Tool 只依赖这一个方法，测试可以注入假实现，不发起网络请求。"""

    async def search_around(
        self,
        *,
        longitude: float,
        latitude: float,
        radius: int,
        limit: int,
        keywords: str | None,
        sortrule: Literal["distance", "weight"],
    ) -> list[HotelSummary]: ...


class AmapHotelSearcher:
    """高德周边搜索实现。"""

    def __init__(
        self,
        ak: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not ak.strip():
            raise ValueError("高德地图 ak 不能为空")
        self._ak = ak.strip()
        self._timeout = timeout
        self._client = client

    async def search_around(
        self,
        *,
        longitude: float,
        latitude: float,
        radius: int,
        limit: int,
        keywords: str | None,
        sortrule: Literal["distance", "weight"] = "distance",
    ) -> list[HotelSummary]:
        params: dict[str, Any] = {
            "key": self._ak,
            "location": f"{longitude:.6f},{latitude:.6f}",
            "types": _HOTEL_TYPES,
            "radius": radius,
            "sortrule": sortrule,
            "page_size": min(max(limit, 1), 25),
            "page_num": 1,
            "show_fields": "business,photos",
            "output": "json",
        }
        if keywords and keywords.strip():
            params["keywords"] = keywords.strip()

        data = await self._request(params)
        pois = data.get("pois") or []
        hotels: list[HotelSummary] = []
        for poi in pois:
            if not isinstance(poi, dict):
                continue
            try:
                hotel = _to_hotel(poi)
            except (TypeError, ValueError) as exc:
                # 单条脏数据不拖垮整批结果
                logger.debug("跳过无法解析的高德酒店条目：%s", exc)
                continue
            if hotel is not None:
                hotels.append(hotel)
        return hotels[:limit]

    async def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        owned = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.get(_AROUND_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GaodeApiError(
                f"高德周边搜索请求失败：{type(exc).__name__}",
                retryable=True,
            ) from exc
        finally:
            if owned:
                await client.aclose()

        if not isinstance(data, dict):
            raise GaodeApiError("高德返回格式不是 JSON 对象")
        if data.get("status") != "1":
            infocode = str(data.get("infocode") or "")
            message = str(data.get("info") or f"infocode={infocode}")
            raise GaodeApiError(
                f"高德周边搜索失败：{message}（infocode={infocode}）",
                retryable=infocode in _RETRYABLE_INFOCODES,
            )
        return data


# ---------------------------- 字段解析 ----------------------------


def _to_float(value: Any) -> float | None:
    if value is None or value == "" or value == []:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "" or value == []:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _parse_location(value: Any) -> GeoPoint | None:
    """高德坐标是 "lng,lat" 字符串；解析失败按未知处理。"""
    if not isinstance(value, str) or "," not in value:
        return None
    lng_text, lat_text = value.split(",", 1)
    try:
        return GeoPoint(lng=float(lng_text), lat=float(lat_text))
    except (TypeError, ValueError):
        return None


def _first_photo_url(poi: dict[str, Any]) -> str | None:
    photos = poi.get("photos")
    if not isinstance(photos, list):
        return None
    for photo in photos:
        if isinstance(photo, dict) and isinstance(photo.get("url"), str):
            url = photo["url"].strip()
            if url:
                return url
    return None


def _to_hotel(poi: dict[str, Any]) -> HotelSummary | None:
    poi_id = poi.get("id")
    name = str(poi.get("name") or "").strip()
    if not isinstance(poi_id, str) or not poi_id.strip() or not name:
        return None

    business = poi.get("business") if isinstance(poi.get("business"), dict) else {}
    rating = _to_float(business.get("rating"))
    if rating is not None and not 0 <= rating <= 5:
        rating = None
    cost = business.get("cost")
    average_cost = str(cost).strip() if cost not in (None, "", []) else None

    return HotelSummary(
        id=poi_id,
        name=name,
        type=str(poi.get("type") or "").strip() or None,
        typecode=str(poi.get("typecode") or "").strip() or None,
        address=str(poi.get("address") or "").strip() or None,
        city=str(poi.get("cityname") or "").strip() or None,
        district=str(poi.get("adname") or "").strip() or None,
        location=_parse_location(poi.get("location")),
        distance_meters=_to_int(poi.get("distance")),
        rating=rating,
        average_cost_cny=average_cost,
        business_area=str(business.get("business_area") or "").strip() or None,
        tel=str(business.get("tel") or "").strip() or None,
        opening_hours_today=str(business.get("opentime_today") or "").strip() or None,
        photo_url=_first_photo_url(poi),
    )


# ---------------------------- 入参 Schema ----------------------------


class SearchHotelsAroundInput(BaseModel):
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="中心点经度（GCJ-02），来自已入选景点的坐标",
    )
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="中心点纬度（GCJ-02），来自已入选景点的坐标",
    )
    radius: int = Field(2000, ge=100, le=50000, description="搜索半径（米），默认 2000")
    limit: int = Field(10, ge=1, le=25, description="返回酒店数量上限，单页最多 25")
    keywords: str | None = Field(
        None, description="可选关键词，如 民宿 / 经济型 / 五星级；不填则按住宿类型搜索"
    )
    sortrule: Literal["distance", "weight"] = Field(
        "distance", description="排序方式：distance=由近到远，weight=高德综合排序"
    )


# ---------------------------- Tool 实现 ----------------------------


def create_gaode_hotel_tool(
    *,
    searcher: HotelSearcher | None = None,
    settings: Settings | None = None,
) -> BaseTool:
    """构建 search_hotels_around。

    searcher 由外部注入时（测试 / 复用连接池）不再读 AK；
    否则按 settings.gaode_maps_ak 创建默认实现，缺 AK 直接报错。
    """
    if searcher is None:
        settings = settings or get_settings()
        ak = (settings.gaode_maps_ak or "").strip()
        if not ak:
            raise ValueError("酒店周边搜索需要 GAODE_MAPS_AK，请在 .env 中配置")
        searcher = AmapHotelSearcher(
            ak, timeout=settings.gaode_place_timeout_seconds
        )

    async def _run(
        longitude: float,
        latitude: float,
        radius: int = 2000,
        limit: int = 10,
        keywords: str | None = None,
        sortrule: Literal["distance", "weight"] = "distance",
    ) -> str:
        try:
            hotels: Sequence[HotelSummary] = await searcher.search_around(
                longitude=longitude,
                latitude=latitude,
                radius=radius,
                limit=limit,
                keywords=keywords,
                sortrule=sortrule,
            )
        except GaodeApiError as exc:
            return error_result(
                "gaode_api_error",
                str(exc),
                retryable=exc.retryable,
            )
        except Exception as exc:  # noqa: BLE001 - Tool 不能把异常抛回 Agent 循环
            return unexpected_error("search_hotels_around", exc)

        items = [
            hotel.model_dump(mode="json", exclude_none=True) for hotel in hotels
        ]
        return dump_payload(
            {
                "ok": True,
                "center": {
                    "longitude": longitude,
                    "latitude": latitude,
                    "coord_type": "gcj02",
                },
                "radius_meters": radius,
                "sortrule": sortrule,
                "total": len(items),
                "items": items,
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="search_hotels_around",
        description=(
            "以景点坐标为中心搜索周边酒店（高德 POI 2.0 周边搜索）。"
            "返回酒店名称、地址、距离、评分、人均价格与电话；"
            "中心点坐标必须来自上游已入选景点，不要自行编造。"
        ),
        args_schema=SearchHotelsAroundInput,
        handle_validation_error=handle_validation_error,
    )


def create_gaode_hotel_tools(
    *,
    searcher: HotelSearcher | None = None,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """registry 入口：与其它分组保持「工厂返回列表」的形状。"""
    return [create_gaode_hotel_tool(searcher=searcher, settings=settings)]


__all__ = [
    "AmapHotelSearcher",
    "GaodeApiError",
    "HotelSearcher",
    "HotelSummary",
    "SearchHotelsAroundInput",
    "create_gaode_hotel_tool",
    "create_gaode_hotel_tools",
]
