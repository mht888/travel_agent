"""HotelSummary（Tool DTO）→ Hotel（业务实体）的转换。

只搬运工具已经返回的事实，缺什么就留 None：
    不猜星级、不补价格、不编电话，也不把 typecode 当类别——
    实测高德出现过 typecode=100102 但 type 写「五星级宾馆」的情况，
    所以类别和星级统一从 type 文本解析。

分类映射按 type 的分段关键词匹配（如 住宿服务;宾馆酒店;五星级宾馆），
typecode 只保留在 Tool 层用于排查，不进入业务实体。
"""

import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from backend.schemas.common import DataStatus, Money, Provenance
from backend.schemas.hotel import Hotel, HotelCategory
from backend.tools.gaode_mcp_tools import HotelSummary

_STAR_PATTERN = re.compile(r"([一二三四五1-5])星级")
_STAR_VALUES = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

# 顺序有意义：先匹配更具体的「民宿 / 青年旅舍」，再落到泛化的「酒店」。
_CATEGORY_KEYWORDS: tuple[tuple[str, HotelCategory], ...] = (
    ("民宿", HotelCategory.HOMESTAY),
    ("青年旅舍", HotelCategory.HOSTEL),
    ("度假村", HotelCategory.RESORT),
    ("公寓", HotelCategory.APARTMENT),
    ("招待所", HotelCategory.INN),
    ("宾馆", HotelCategory.HOTEL),
    ("酒店", HotelCategory.HOTEL),
)


def _parse_category(type_text: str | None) -> HotelCategory:
    if not type_text:
        return HotelCategory.OTHER
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in type_text:
            return category
    return HotelCategory.OTHER


def _parse_star_rating(type_text: str | None) -> int | None:
    if not type_text:
        return None
    match = _STAR_PATTERN.search(type_text)
    return _STAR_VALUES.get(match.group(1)) if match else None


def _parse_tags(type_text: str | None) -> list[str]:
    """跳过一级类目「住宿服务」，只保留更具体的两段。"""
    if not type_text:
        return []
    segments = [segment.strip() for segment in type_text.split(";") if segment.strip()]
    return segments[1:] if len(segments) > 1 else segments


def _parse_price(text: str | None) -> Money | None:
    """高德的人均价/参考价，标 ESTIMATED：它不是实时房价。"""
    if not text:
        return None
    try:
        amount = Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        return None
    if amount < 0:
        return None
    return Money(amount=amount, currency="CNY", status=DataStatus.ESTIMATED)


def hotel_from_summary(
    summary: HotelSummary,
    *,
    city: str,
    fetched_at: datetime | None = None,
    source: str = "gaode:mcp_around",
) -> Hotel:
    """把一条酒店 DTO 转成业务实体。

    city 由调用方兜底：个别 POI 的 cityname 为空，
    但这次搜索本身就有明确的所属城市，不必因此丢掉整条数据。
    """
    return Hotel(
        id=summary.id,
        name=summary.name,
        city=(summary.city or city).strip(),
        district=summary.district,
        address=summary.address,
        business_area=summary.business_area,
        location=summary.location,
        category=_parse_category(summary.type),
        tags=_parse_tags(summary.type),
        star_rating=_parse_star_rating(summary.type),
        rating=summary.rating,
        # 高德该接口不返回评价数，保持 None
        review_count=None,
        reference_price=_parse_price(summary.average_cost_cny),
        tel=summary.tel,
        photo_urls=[summary.photo_url] if summary.photo_url else [],
        provenance=Provenance(
            source=source,
            fetched_at=fetched_at or datetime.now(timezone.utc),
            status=DataStatus.REAL,
        ),
    )


__all__ = ["hotel_from_summary"]
