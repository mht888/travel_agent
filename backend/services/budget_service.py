"""预算计算（第 8 阶段）。

职责边界：
    ItineraryService   取数、排程、组装时间轴（第 7 阶段）
    BudgetService      只做「把行程翻译成钱」：六类费用、汇总、超支判断
    Evaluator          消费 BudgetBreakdown 做约束检查（第 9 阶段）

三条硬规则：
1. 只给估算，不给假报价：所有条目 status 都是 ESTIMATED，note 写清算式。
2. 单一币种账本：内部统一按 USD 计算，最后折算成用户预算币种。
3. 未知不等于免费：门票价格为 None 的景点不计入 0，只记 warning。

为什么这里不用 async：
    整个估算是纯函数式计算，不访问网络和数据库。保持同步更容易测试，
    也避免调用方误以为这里有 I/O 需要等待。
    ItineraryService 在组装完之后同步调用它，不增加额外延迟。
"""

import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from backend.schemas.budget import BudgetBreakdown, BudgetCategory, BudgetLineItem
from backend.schemas.common import DataStatus, Money, TransportMode
from backend.schemas.itinerary import Itinerary, ItineraryItemType
from backend.schemas.trip import TravelStyle, TripRequirements

logger = logging.getLogger(__name__)

DEFAULT_CURRENCY = "USD"

# 固定汇率（1 USD 可兑换多少目标币种）。第一版不接汇率服务：
# 数值只用于把 Mock 的日元门票折算进美元账本，全部标记为 ESTIMATED。
_FX_PER_USD: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "JPY": Decimal("150"),
    "CNY": Decimal("7.2"),
    "EUR": Decimal("0.92"),
}

_CENT = Decimal("0.01")
_ZERO = Decimal("0")


@dataclass(frozen=True)
class CostProfile:
    """目的地成本档位，全部以 USD 计价。

    接入真实机票/酒店 Provider 后，这里的默认值会被真实报价替换，
    所以业务代码只依赖这个结构，不依赖具体数字。
    """

    flight_per_person: Decimal
    hotel_per_night: Decimal
    food_per_person_day: Decimal


# 第一版只有东京有独立档位，其余城市走默认档。
# 档位是「量级正确」的估算，不是报价，note 里会写清楚。
_COST_PROFILES: dict[str, CostProfile] = {
    "tokyo": CostProfile(
        flight_per_person=Decimal("450"),
        hotel_per_night=Decimal("140"),
        food_per_person_day=Decimal("55"),
    ),
}
_DEFAULT_PROFILE = CostProfile(
    flight_per_person=Decimal("500"),
    hotel_per_night=Decimal("120"),
    food_per_person_day=Decimal("45"),
)

# 节奏影响餐饮：relaxed 吃得简单、intensive 外食更多。
_FOOD_STYLE_FACTOR: dict[TravelStyle, Decimal] = {
    TravelStyle.RELAXED: Decimal("0.85"),
    TravelStyle.BALANCED: Decimal("1"),
    TravelStyle.INTENSIVE: Decimal("1.2"),
}

# 市内交通单价（USD）。步行永远 0，是唯一不需要估的费用。
_TRANSIT_FARE = Decimal("2")
_DRIVE_PER_KM = Decimal("0.55")
_TAXI_PER_KM = Decimal("2.2")
_DRIVE_MIN_FARE = Decimal("3")
_TAXI_MIN_FARE = Decimal("6")
# 矩阵缺距离时，用耗时反推里程；20km/h 是城市内混行的保守值
_FALLBACK_SPEED_KMH = Decimal("20")

# 其他开销（纪念品、寄存、临时打车）按前五类的比例计提
_OTHER_RATE = Decimal("0.08")

# 已安排餐厅时，日均餐费中剩下早/晚餐的比例（午餐约占 40%）
_REMAINING_MEAL_RATIO = Decimal("0.6")


@dataclass
class BudgetOutcome:
    """预算结果 + 估算过程中产生的提示。

    warnings 必须透传给用户：未知门票、汇率折算这类信息不能静默吞掉。
    """

    breakdown: BudgetBreakdown
    warnings: list[str] = field(default_factory=list)


def _fmt(value: Decimal) -> str:
    """去掉多余的尾零：450.00 -> 450，46.75 -> 46.75。"""
    return format(value.quantize(_CENT).normalize(), "f")


def _profile_for(destination: str) -> CostProfile:
    return _COST_PROFILES.get(destination.strip().casefold(), _DEFAULT_PROFILE)


def _typed_item_cost_usd(
    itinerary: Itinerary, item_type: ItineraryItemType
) -> tuple[Decimal, list[str], int, int]:
    """按行程项类型汇总费用：分组币种 → 折算成 USD。

    返回 (美元总额, 原始金额说明片段, 价格未知的项数, 计入的项数)。
    门票和餐厅共用这套逻辑，但必须按类型分开统计，否则会互相串账。
    """
    per_currency: dict[str, Decimal] = {}
    unknown = 0
    counted = 0

    for day in itinerary.days:
        for item in day.items:
            if item.type is not item_type:
                continue
            if item.cost is None:
                # None 表示「价格未知」，不是「免费」。免费项在数据层是 Money(0)。
                unknown += 1
                continue
            counted += 1
            per_currency[item.cost.currency] = (
                per_currency.get(item.cost.currency, _ZERO) + item.cost.amount
            )

    total_usd = _ZERO
    parts: list[str] = []
    for currency, amount in sorted(per_currency.items()):
        total_usd += amount / _FX_PER_USD[currency]
        parts.append(f"{_fmt(amount)} {currency}")

    return total_usd, parts, unknown, counted


def _transport_cost_usd(
    itinerary: Itinerary,
) -> tuple[Decimal, dict[str, int], int]:
    """按行程里的交通段计价。

    返回 (美元总额, 各方式段数, 缺距离的段数)。
    距离缺失时用耗时反推，但必须把次数透传出去记 warning。
    """
    total = _ZERO
    counts: dict[str, int] = {"walk": 0, "transit": 0, "drive": 0, "taxi": 0}
    missing_distance = 0

    for day in itinerary.days:
        for item in day.items:
            leg = item.transport_from_previous
            if leg is None:
                continue

            counts[leg.mode.value] += 1
            if leg.mode is TransportMode.WALK:
                continue
            if leg.mode is TransportMode.TRANSIT:
                total += _TRANSIT_FARE
                continue

            if leg.distance_meters is not None:
                kilometers = Decimal(leg.distance_meters) / 1000
            else:
                kilometers = (
                    Decimal(leg.duration_minutes) / 60 * _FALLBACK_SPEED_KMH
                )
                missing_distance += 1

            if leg.mode is TransportMode.DRIVE:
                total += max(_DRIVE_MIN_FARE, kilometers * _DRIVE_PER_KM)
            else:  # TAXI
                total += max(_TAXI_MIN_FARE, kilometers * _TAXI_PER_KM)

    return total, counts, missing_distance


def _transport_note(counts: dict[str, int], missing_distance: int) -> str:
    labels = {
        "walk": "步行",
        "transit": "公共交通",
        "drive": "驾车",
        "taxi": "出租车",
    }
    parts = [
        f"{labels[mode]} {count} 段"
        for mode, count in counts.items()
        if count > 0
    ]
    note = "、".join(parts) if parts else "行程内没有景点间交通"
    if missing_distance:
        note += f"；{missing_distance} 段缺少距离，按时长反推里程"
    return note


class BudgetService:
    """六类费用估算 + 汇总。无状态，可以反复调用。"""

    def estimate(
        self,
        requirements: TripRequirements,
        itinerary: Itinerary,
    ) -> BudgetOutcome:
        """把需求与行程翻译成一本账。

        任何一类数据缺失都只降级为 warning，不抛异常：
        预算算不完整不是「请求失败」，而是「结果要标注」。
        """
        ledger = (
            requirements.budget.currency if requirements.budget else DEFAULT_CURRENCY
        )
        rate = _FX_PER_USD[ledger]
        profile = _profile_for(requirements.destination)
        warnings: list[str] = []

        travelers = requirements.travelers
        duration = requirements.duration_days
        # 最后一天退房，所以住宿是「天数 - 1」晚；一日游没有住宿
        nights = max(duration - 1, 0)
        rooms = (travelers + 1) // 2  # 两人一间，向上取整

        flights_usd = profile.flight_per_person * travelers
        hotel_usd = profile.hotel_per_night * nights * rooms
        daily_food = (
            profile.food_per_person_day
            * _FOOD_STYLE_FACTOR[requirements.travel_style]
        )

        tickets_usd, ticket_parts, unknown_tickets, _ = _typed_item_cost_usd(
            itinerary, ItineraryItemType.ATTRACTION
        )
        restaurant_usd, restaurant_parts, unknown_restaurants, restaurant_count = (
            _typed_item_cost_usd(itinerary, ItineraryItemType.RESTAURANT)
        )

        # 已排餐厅按实际人均消费计入，其余餐食仍按日均估算，
        # 比例取 0.6 代表早/晚餐，避免与午餐重复计算。
        if restaurant_count:
            remaining_food = (
                daily_food * _REMAINING_MEAL_RATIO * duration * travelers
            )
            food_usd = restaurant_usd + remaining_food
            food_note = (
                f"已安排 {restaurant_count} 顿餐厅（原价 "
                f"{'、'.join(restaurant_parts) if restaurant_parts else '未知'}）"
                f"+ 其余餐食按 {_fmt(daily_food * _REMAINING_MEAL_RATIO)} USD/人/天"
                f" × {duration} 天 × {travelers} 人"
            )
        else:
            food_usd = daily_food * duration * travelers
            food_note = (
                f"按 {_fmt(daily_food)} USD/人/天 × {duration} 天 × {travelers} 人"
                f"（{requirements.travel_style.value} 档）"
            )

        transport_usd, transport_counts, missing_distance = _transport_cost_usd(
            itinerary
        )

        subtotal = flights_usd + hotel_usd + transport_usd + tickets_usd + food_usd
        other_usd = subtotal * _OTHER_RATE

        if unknown_tickets:
            warnings.append(
                f"{unknown_tickets} 个景点门票价格未知，未计入预算（未知不等于免费）"
            )
        if unknown_restaurants:
            warnings.append(
                f"{unknown_restaurants} 家餐厅人均消费未知，未计入预算（未知不等于免费）"
            )
        if ticket_parts and any(
            not part.endswith(ledger) for part in ticket_parts
        ):
            warnings.append(
                f"门票原始价为 {'、'.join(ticket_parts)}，"
                f"已按固定汇率折算为 {ledger}，金额为估算值"
            )
        if restaurant_parts and any(
            not part.endswith(ledger) for part in restaurant_parts
        ):
            warnings.append(
                f"餐厅原始价为 {'、'.join(restaurant_parts)}，"
                f"已按固定汇率折算为 {ledger}，金额为估算值"
            )
        if missing_distance:
            warnings.append(
                f"{missing_distance} 段交通缺少距离数据，已按平均车速反推里程计费"
            )

        lines = [
            self._line(
                BudgetCategory.FLIGHTS,
                "往返机票（估算）",
                flights_usd,
                ledger,
                rate,
                f"按 {_fmt(profile.flight_per_person)} USD/人 × {travelers} 人；"
                "未提供出发地，按目的地档位估算",
            ),
            self._line(
                BudgetCategory.ACCOMMODATION,
                "住宿（估算）",
                hotel_usd,
                ledger,
                rate,
                f"按 {_fmt(profile.hotel_per_night)} USD/晚 × {nights} 晚 × {rooms} 间",
            ),
            self._line(
                BudgetCategory.TRANSPORT,
                "市内交通（估算）",
                transport_usd,
                ledger,
                rate,
                _transport_note(transport_counts, missing_distance),
            ),
            self._line(
                BudgetCategory.TICKETS,
                "景点门票（估算）",
                tickets_usd,
                ledger,
                rate,
                (
                    f"门票原始价 {'、'.join(ticket_parts)}"
                    if ticket_parts
                    else "行程中没有收费景点"
                ),
            ),
            self._line(
                BudgetCategory.FOOD,
                "餐饮（估算）",
                food_usd,
                ledger,
                rate,
                food_note,
            ),
            self._line(
                BudgetCategory.OTHER,
                "其他开销（估算）",
                other_usd,
                ledger,
                rate,
                f"前五类合计 {_fmt(subtotal)} USD 的 {_fmt(_OTHER_RATE * 100)}%",
            ),
        ]

        breakdown = BudgetBreakdown(
            lines=lines,
            # 用户没给预算就传 None：不替用户假设一个上限
            budget_limit=requirements.budget,
        )
        logger.info(
            "预算估算完成：%d 类，合计 %s %s",
            len(lines),
            _fmt(breakdown.total.amount) if breakdown.total else "0",
            breakdown.currency,
        )
        return BudgetOutcome(breakdown=breakdown, warnings=warnings)

    @staticmethod
    def _line(
        category: BudgetCategory,
        label: str,
        amount_usd: Decimal,
        ledger: str,
        rate: Decimal,
        note: str,
    ) -> BudgetLineItem:
        """生成一条预算条目：统一折算到账本币种，并标注汇率。"""
        amount = (amount_usd * rate).quantize(_CENT, rounding=ROUND_HALF_UP)
        if ledger != "USD":
            note = f"{note}；按 1 USD={_fmt(rate)} {ledger} 折算"
        return BudgetLineItem(
            category=category,
            label=label,
            amount=Money(amount=amount, currency=ledger, status=DataStatus.ESTIMATED),
            note=note,
        )
