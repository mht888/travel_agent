"""RestaurantSummary（Tool DTO）→ Restaurant（业务实体）的转换。

只搬运工具已经返回的事实，缺什么就留 None：
    不猜菜系、不补评分、不编营业时间，也不把 typecode 当菜系——
    菜系统一从高德 type 文本的分段里取（如 餐饮服务;中餐厅;川菜）。

营业时间暂不映射：
    高德 business.opentime_today 只给「今天」的时段，不带星期，
    直接塞进 OpeningInterval 会伪造 weekday，因此保持空列表，
    原始字符串保留在 Tool 层，等接入营业时间接口后再对齐。
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from backend.schemas.common import DataStatus, Money, Provenance
from backend.schemas.restaurant import Restaurant
from backend.tools.gaode_restaurant_tools import RestaurantSummary

# 一级类目，对「吃什么」没有信息量，不进入 cuisines
_BROAD_TYPES = frozenset({"餐饮服务"})


def _parse_cuisines(type_text: str | None) -> list[str]:
    """按分号拆高德 POI 类型，跳过一级类目，保留菜系与品类。"""
    if not type_text:
        return []
    segments = [segment.strip() for segment in type_text.split(";") if segment.strip()]
    if len(segments) > 1:
        segments = segments[1:]
    return [segment for segment in segments if segment not in _BROAD_TYPES]


def _parse_price(text: str | None) -> Money | None:
    """高德的人均消费，标 ESTIMATED：它不是实时菜单价格。"""
    if not text:
        return None
    try:
        amount = Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        return None
    if amount < 0:
        return None
    return Money(amount=amount, currency="CNY", status=DataStatus.ESTIMATED)


def restaurant_from_summary(
    summary: RestaurantSummary,
    *,
    city: str,
    fetched_at: datetime | None = None,
    source: str = "gaode:poi_around",
) -> Restaurant | None:
    """把一条餐厅 DTO 转成业务实体。

    location 是必填字段：高德偶尔缺坐标，此时返回 None，
    由调用方跳过——没有坐标的餐厅无法参与后续顺路计算。
    city 由调用方兜底：个别 POI 的 cityname 为空，但本次搜索本身有明确城市。
    """
    if summary.location is None:
        return None

    return Restaurant(
        id=summary.id,
        name=summary.name,
        city=(summary.city or city).strip(),
        cuisines=_parse_cuisines(summary.type),
        location=summary.location,
        rating=summary.rating,
        # 高德该接口不返回评价数，保持 None
        review_count=None,
        average_price=_parse_price(summary.average_cost_cny),
        provenance=Provenance(
            source=source,
            fetched_at=fetched_at or datetime.now(timezone.utc),
            status=DataStatus.REAL,
        ),
    )


__all__ = ["restaurant_from_summary"]
