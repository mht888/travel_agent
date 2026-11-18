"""高德 POI 2.0 · 周边搜索：以某个中心点找周边餐厅。

职责边界：
    中心点坐标由 Agent 提供（景点坐标或 geocode 结果），本工具不负责查景点；
    工具只做「参数校验 → 调高德 → 字段归一化 → 返回 JSON」，
    不判断口味是否合适、也不替用户选餐厅。

与 gaode_hotel_tools 的差异：
    types=050000（餐饮服务），返回的 type 会带上菜系分段（如 餐饮服务;中餐厅;川菜），
    由组装层解析成 Restaurant.cuisines；
    这里同样不解析营业时间——opentime_today 不含星期，
    强行映射到 weekday 会伪造数据，留给后续接入营业时间接口时处理。

坐标系：
    高德 Web 服务 API 使用 GCJ-02，与 geocode 结果一致，不需要转换。

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

# 050000 = 餐饮服务（中餐厅 / 外国餐厅 / 快餐厅 / 咖啡厅 / 甜品店等）
_RESTAURANT_TYPES = "050000"

# 这些 infocode 属于限流，重试可能恢复；其余（KEY 无效、无权限、配额用尽）重试无意义。
_RETRYABLE_INFOCODES = frozenset({"10004", "10021", "10045"})


class GaodeApiError(RuntimeError):
    """高德接口不可用或业务返回失败。retryable 决定 Agent 是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class RestaurantSummary(BaseModel):
    """归一化后的餐厅条目。金额按项目约定用字符串，避免浮点精度漂移。"""

    id: str
    name: str
    type: str | None = Field(None, description="高德 POI 类型，如 餐饮服务;中餐厅;川菜")
    typecode: str | None = None
    address: str | None = None
    city: str | None = None
    district: str | None = None
    location: GeoPoint | None = None
    distance_meters: int | None = Field(None, description="与中心点的直线距离（米）")
    rating: float | None = Field(None, ge=0, le=5)
    average_cost_cny: str | None = Field(
        None, description="高德返回的人均消费（元，字符串）"
    )
    business_area: str | None = None
    tel: str | None = None
    opening_hours_today: str | None = None
    photo_url: str | None = None


# ---------------------------- 高德客户端 ----------------------------


class RestaurantSearcher(Protocol):
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
    ) -> list[RestaurantSummary]: ...


class AmapRestaurantSearcher:
    """高德周边搜索实现（餐饮服务）。"""

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
    ) -> list[RestaurantSummary]:
        params: dict[str, Any] = {
            "key": self._ak,
            "location": f"{longitude:.6f},{latitude:.6f}",
            "types": _RESTAURANT_TYPES,
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
        restaurants: list[RestaurantSummary] = []
        for poi in pois:
            if not isinstance(poi, dict):
                continue
            try:
                restaurant = _to_restaurant(poi)
            except (TypeError, ValueError) as exc:
                # 单条脏数据不拖垮整批结果
                logger.debug("跳过无法解析的高德餐厅条目：%s", exc)
                continue
            if restaurant is not None:
                restaurants.append(restaurant)
        return restaurants[:limit]

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


def _to_restaurant(poi: dict[str, Any]) -> RestaurantSummary | None:
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

    return RestaurantSummary(
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


class SearchRestaurantsAroundInput(BaseModel):
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="中心点经度（GCJ-02），来自景点坐标或 geocode 结果",
    )
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="中心点纬度（GCJ-02），来自景点坐标或 geocode 结果",
    )
    radius: int = Field(2000, ge=100, le=50000, description="搜索半径（米），默认 2000")
    limit: int = Field(10, ge=1, le=25, description="返回餐厅数量上限，单页最多 25")
    keywords: str | None = Field(
        None,
        description="可选关键词，如 老友粉 / 川菜 / 烧烤 / 米粉；不填则按餐饮类型搜索",
    )
    sortrule: Literal["distance", "weight"] = Field(
        "distance", description="排序方式：distance=由近到远，weight=高德综合排序"
    )


# ---------------------------- Tool 实现 ----------------------------


def create_gaode_restaurant_tool(
    *,
    searcher: RestaurantSearcher | None = None,
    settings: Settings | None = None,
) -> BaseTool:
    """构建 search_restaurants_around。

    searcher 由外部注入时（测试 / 复用连接池）不再读 AK；
    否则按 settings.gaode_maps_ak 创建默认实现，缺 AK 直接报错。
    """
    if searcher is None:
        settings = settings or get_settings()
        ak = (settings.gaode_maps_ak or "").strip()
        if not ak:
            raise ValueError("餐厅周边搜索需要 GAODE_MAPS_AK，请在 .env 中配置")
        searcher = AmapRestaurantSearcher(
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
            restaurants: Sequence[RestaurantSummary] = await searcher.search_around(
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
            return unexpected_error("search_restaurants_around", exc)

        items = [
            restaurant.model_dump(mode="json", exclude_none=True)
            for restaurant in restaurants
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
        name="search_restaurants_around",
        description=(
            "以坐标为中心搜索周边餐厅（高德 POI 2.0 周边搜索）。"
            "返回餐厅名称、菜系（POI 类型）、地址、距离、评分、人均消费、"
            "今日营业时间与电话；中心点坐标必须来自真实数据，不要自行编造。"
        ),
        args_schema=SearchRestaurantsAroundInput,
        handle_validation_error=handle_validation_error,
    )


def create_gaode_restaurant_tools(
    *,
    searcher: RestaurantSearcher | None = None,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """registry 入口：与其它分组保持「工厂返回列表」的形状。"""
    return [create_gaode_restaurant_tool(searcher=searcher, settings=settings)]


__all__ = [
    "AmapRestaurantSearcher",
    "GaodeApiError",
    "RestaurantSearcher",
    "RestaurantSummary",
    "SearchRestaurantsAroundInput",
    "create_gaode_restaurant_tool",
    "create_gaode_restaurant_tools",
]
