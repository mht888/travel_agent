"""百度地图 Place API v3 景点数据源。

为什么用百度而不是 Google Places：
    目标是国内城市（如南宁），百度 Place API 覆盖更完整，且返回中文名称与标签，
    可以直接交给 LLM 判断兴趣，不需要额外的中英映射表。

边界：
    Provider 只做「检索 + 字段转换」，不理解用户兴趣，也不排序。
    兴趣排序属于 AttractionRanker，两者分开才能独立做 A/B 对比。

坐标说明：
    请求固定 ret_coordtype=gcj02ll，返回的是 GCJ-02 坐标。
    与高德/腾讯地图一致；若要和 WGS-84 数据混合，必须先转换。
"""

import logging
import re
from collections.abc import Sequence
from datetime import datetime, time, timezone
from decimal import Decimal

import httpx

from backend.config import get_settings
from backend.errors import ProviderUnavailableError
from backend.providers.base import AttractionProvider
from backend.schemas.attraction import Attraction, AttractionCategory, OpeningInterval
from backend.schemas.common import DataStatus, GeoPoint, Money, Provenance

logger = logging.getLogger(__name__)

_PLACE_SEARCH_URL = "https://api.map.baidu.com/api_place_pro/v1/region"
_PLACE_DETAIL_URL = "https://api.map.baidu.com/place/v2/detail"

# Place API v3 的 page_size 只接受 10~20；免费配额翻页有限，3 页足够组成候选池
_MIN_PAGE_SIZE = 10
_MAX_PAGE_SIZE = 20
_MAX_PAGES = 3

_DEFAULT_KEYWORDS: tuple[str, ...] = ("景点",)

# 兴趣 -> 百度检索词。只用来扩大候选池，命中与否由 Ranker 决定。
_INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "food": ("美食", "小吃"),
    "culture": ("博物馆", "古迹"),
    "nature": ("公园", "自然风光"),
    "shopping": ("购物", "商业街"),
    "nightlife": ("夜景", "酒吧"),
    "entertainment": ("游乐园", "水族馆"),
    "landmark": ("地标", "广场"),
    "art": ("美术馆", "艺术"),
    "wellness": ("温泉",),
    "outdoor_activity": ("登山", "徒步"),
}

# 分类关键词按顺序匹配，先命中先返回。
# 顺序经过调整：公园/温泉优先于「寺」，避免「温泉寺」被判成文化景点。
_CATEGORY_KEYWORDS: tuple[tuple[AttractionCategory, tuple[str, ...]], ...] = (
    (AttractionCategory.WELLNESS, ("温泉", "spa", "疗养")),
    (AttractionCategory.NATURE, ("公园", "植物园", "湿地", "森林", "山", "湖", "风景", "nature", "park")),
    (AttractionCategory.ART, ("美术馆", "画廊", "艺术", "展览", "art", "gallery")),
    (AttractionCategory.CULTURE, ("博物馆", "寺", "庙", "故居", "古迹", "遗址", "纪念馆", "古镇", "文化", "museum", "temple")),
    (AttractionCategory.SHOPPING, ("购物", "商场", "步行街", "百货", "免税", "商业街", "shopping", "mall")),
    (AttractionCategory.FOOD, ("美食", "小吃", "餐厅", "夜市", "food", "market")),
    (AttractionCategory.ENTERTAINMENT, ("乐园", "游乐", "水族馆", "动物园", "剧院", "演出", "影城", "entertainment")),
    (AttractionCategory.NIGHTLIFE, ("酒吧", "夜景", "夜游", "nightlife")),
    (AttractionCategory.OUTDOOR_ACTIVITY, ("徒步", "登山", "骑行", "漂流", "露营地", "outdoor")),
    (AttractionCategory.LANDMARKS, ("地标", "塔", "广场", "大桥", "观景", "landmark", "tower")),
)

# 百度不提供建议游玩时长，按分类给一个保守默认值，后续可由用户行程反馈修正。
_DURATION_BY_CATEGORY: dict[AttractionCategory, int] = {
    AttractionCategory.NATURE: 120,
    AttractionCategory.ENTERTAINMENT: 120,
    AttractionCategory.SHOPPING: 90,
    AttractionCategory.CULTURE: 90,
    AttractionCategory.OUTDOOR_ACTIVITY: 150,
    AttractionCategory.WELLNESS: 120,
}
_DEFAULT_DURATION_MINUTES = 90

_HOURS_PATTERN = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*[-~至]\s*(\d{1,2}):(\d{2})\s*$"
)
_TAG_SPLIT_PATTERN = re.compile(r"[;；,，]")


def _to_float(value: object) -> float | None:
    """百度把评分放在字符串里（如 "4.7"），解析失败按未知处理。"""
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _category_from(name: str, tags: Sequence[str]) -> AttractionCategory:
    haystack = f"{name} {' '.join(tags)}".casefold()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return category
    return AttractionCategory.OTHER


def _parse_price(detail: dict) -> Money | None:
    """百度 price 可能是数字、"50元" 或 "免费"；无法识别时返回 None。"""
    raw = detail.get("price")
    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    if "免费" in text:
        return Money(amount=0, currency="CNY", status=DataStatus.REAL)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return Money(
        amount=Decimal(match.group()),
        currency="CNY",
        status=DataStatus.REAL,
    )


def _parse_opening_hours(raw: object) -> list[OpeningInterval]:
    """只解析 "08:00-18:00" 这类单段格式，其余保持为空（未知）。

    跨午夜或「周一闭馆」这类复杂规则百度没有结构化字段，强行猜会误导排程。
    """
    if not isinstance(raw, str):
        return []
    match = _HOURS_PATTERN.match(raw)
    if match is None:
        return []

    open_hour, open_minute, close_hour, close_minute = map(int, match.groups())
    open_total = open_hour * 60 + open_minute
    close_total = close_hour * 60 + close_minute

    # 百度用 "00:00-24:00" 或 "00:00-00:00" 表示全天营业；
    # 与 Mock 约定一致，用空列表表示「不限时段」。
    if close_total >= 24 * 60 or (open_total == 0 and close_total == 0):
        return []
    if open_hour > 23 or close_hour > 23 or close_total <= open_total:
        return []

    opens_at = time(open_hour, open_minute)
    closes_at = time(close_hour, close_minute)
    return [
        OpeningInterval(weekday=weekday, opens_at=opens_at, closes_at=closes_at)
        for weekday in range(7)
    ]


def _collect_tags(detail: dict) -> list[str]:
    """合并 tag / classified_poi_tag，保留中文原值，不做中英映射。"""
    tags: list[str] = []
    for key in ("tag", "classified_poi_tag"):
        value = detail.get(key)
        if not isinstance(value, str):
            continue
        for part in _TAG_SPLIT_PATTERN.split(value):
            clean = part.strip().lower()
            if clean and clean not in tags:
                tags.append(clean)
    return tags


def _to_attraction(item: dict, city: str) -> Attraction | None:
    """把一条百度结果转换成统一的 Attraction；缺关键字段就丢弃。"""
    uid = item.get("uid") or item.get("id")
    name = str(item.get("name") or "").strip()
    location = item.get("location") or {}
    lat, lng = location.get("lat"), location.get("lng")
    if not uid or not name or lat is None or lng is None:
        return None

    detail = item.get("detail_info") or {}
    tags = _collect_tags(detail)
    category = _category_from(name, tags)
    rating = _to_float(detail.get("overall_rating") or detail.get("rating"))
    if rating is not None and not 0 <= rating <= 5:
        rating = None

    review_count = _to_int(detail.get("comment_num"))
    if review_count is not None and review_count < 0:
        review_count = None

    try:
        geo = GeoPoint(lat=float(lat), lng=float(lng))
    except (TypeError, ValueError):
        return None

    return Attraction(
        id=f"baidu:{uid}",
        name=name,
        city=city,
        country="China",
        category=category,
        tags=tags,
        description=None,
        location=geo,
        rating=rating,
        review_count=review_count,
        opening_hours=_parse_opening_hours(detail.get("shop_hours")),
        estimated_duration_minutes=_DURATION_BY_CATEGORY.get(
            category, _DEFAULT_DURATION_MINUTES
        ),
        ticket_price=_parse_price(detail),
        provenance=Provenance(
            source="baidu:place_api",
            fetched_at=datetime.now(timezone.utc),
            status=DataStatus.REAL,
        ),
    )


class BaiduAttractionProvider(AttractionProvider):
    """百度 Place API v3 实现。

    interests 在这里只被翻译成检索词，不做偏好过滤：
    候选池要尽量全，否则后续排序（无论 rule 还是 llm）都会因为「没得选」失真。
    """

    def __init__(
        self,
        ak: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not ak.strip():
            raise ValueError("百度地图 ak 不能为空")
        self._ak = ak.strip()
        self._timeout = timeout
        self._client = client

    async def search_attractions(
        self,
        city: str,
        interests: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Attraction]:
        return await self.search_by_keywords(
            city=city,
            keywords=self._keywords_for(interests),
            limit=limit,
        )

    async def search_by_keywords(
        self,
        city: str,
        keywords: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Attraction]:
        """按多个中文关键词检索，按 uid 去重后截断到 limit。"""
        if limit <= 0:
            return []

        queries = [keyword.strip() for keyword in (keywords or _DEFAULT_KEYWORDS)]
        queries = [query for query in queries if query] or list(_DEFAULT_KEYWORDS)

        merged: list[Attraction] = []
        seen: set[str] = set()
        for query in queries:
            remaining = limit - len(merged)
            if remaining <= 0:
                break
            raw_items = await self._search_pages(city, query, remaining)
            for item in raw_items:
                try:
                    attraction = _to_attraction(item, city)
                except (TypeError, ValueError) as exc:
                    # 外部数据可能有脏值，单条解析失败不应拖垮整个候选池
                    logger.debug("跳过无法解析的百度景点条目：%s", exc)
                    continue
                if attraction is None or attraction.id in seen:
                    continue
                seen.add(attraction.id)
                merged.append(attraction)
                if len(merged) >= limit:
                    break
        return merged

    async def get_attraction(self, attraction_id: str) -> Attraction | None:
        uid = attraction_id.removeprefix("baidu:").strip()
        if not uid:
            return None
        data = await self._request(
            _PLACE_DETAIL_URL,
            {
                "uid": uid,
                "output": "json",
                "scope": 2,
                "ret_coordtype": "gcj02ll",
                "ak": self._ak,
            },
        )
        result = data.get("result")
        if not isinstance(result, dict):
            return None
        try:
            return _to_attraction(result, str(result.get("city") or ""))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _keywords_for(interests: Sequence[str] | None) -> list[str]:
        keywords = list(_DEFAULT_KEYWORDS)
        for interest in interests or []:
            for keyword in _INTEREST_KEYWORDS.get(interest.strip().lower(), ()):
                if keyword not in keywords:
                    keywords.append(keyword)
        return keywords

    async def _search_pages(
        self, city: str, query: str, limit: int
    ) -> list[dict]:
        results: list[dict] = []
        for page_num in range(_MAX_PAGES):
            remaining = limit - len(results)
            if remaining <= 0:
                break
            page_size = min(_MAX_PAGE_SIZE, max(_MIN_PAGE_SIZE, remaining))
            data = await self._request(
                _PLACE_SEARCH_URL,
                {
                    "query": query,
                    "region": city,
                    "region_limit": "true",
                    "output": "json",
                    "scope": 2,
                    "page_size": page_size,
                    "page_num": page_num,
                    "ret_coordtype": "gcj02ll",
                    "ak": self._ak,
                },
            )
            batch = data.get("results") or []
            results.extend(item for item in batch if isinstance(item, dict))
            if len(batch) < page_size or len(results) >= limit:
                break
        return results[:limit]

    async def _request(self, url: str, params: dict) -> dict:
        """统一处理网络与百度业务错误，全部翻译成 ProviderUnavailableError。"""
        client = self._client
        owned = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailableError("百度地图", f"请求失败：{type(exc).__name__}") from exc
        finally:
            if owned:
                await client.aclose()

        if not isinstance(data, dict):
            raise ProviderUnavailableError("百度地图", "返回格式不是 JSON 对象")
        status = data.get("status", 0)
        if status != 0:
            message = data.get("message") or f"status={status}"
            raise ProviderUnavailableError("百度地图", f"接口返回错误：{message}")
        return data


def get_baidu_attraction_provider() -> BaiduAttractionProvider:
    """默认 Provider。缺少 AK 时抛出可读错误，由上层决定是否降级。"""
    settings = get_settings()
    if not settings.baidu_maps_ak:
        raise ValueError("景点候选池需要 BAIDU_MAPS_AK，请在 .env 中配置")
    return BaiduAttractionProvider(
        ak=settings.baidu_maps_ak,
        timeout=settings.baidu_place_timeout_seconds,
    )
