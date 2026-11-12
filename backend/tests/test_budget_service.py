"""第 8 阶段：预算计算测试。

覆盖四类行为：
1. 六类费用齐全，总额严格等于明细之和
2. 住宿按「天数 - 1」晚计算，超支判断正确
3. 价格未知的门票不计入 0，只记 warning
4. 外币门票（JPY）折算进美元账本，所有条目同币种

运行（在 travel_agent/ 目录下）：
    .venv\\Scripts\\python.exe -m pytest backend/tests/test_budget_service.py -v
"""

import asyncio
from datetime import date, time
from decimal import Decimal

import pytest

from backend.agents.planner_agent import RuleBasedItineraryPlanner
from backend.providers.mock_provider import (
    MockAttractionProvider,
    MockRestaurantProvider,
    MockRouteProvider,
    MockWeatherProvider,
)
from backend.schemas.budget import BudgetCategory
from backend.schemas.common import DataStatus, Money, Provenance, TransportMode
from backend.schemas.itinerary import (
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
    TransportLeg,
)
from backend.schemas.trip import TravelStyle, TripRequirements
from backend.services.attraction_service import AttractionService
from backend.services.budget_service import BudgetService
from backend.services.itinerary_service import ItineraryService
from backend.services.restaurant_service import RestaurantService
from backend.services.route_service import RouteService
from backend.services.weather_service import WeatherService


def _money(amount: str, currency: str, status: DataStatus = DataStatus.MOCK) -> Money:
    return Money(amount=Decimal(amount), currency=currency, status=status)


def _requirements(
    duration_days: int = 3,
    budget: Money | None = None,
    travelers: int = 1,
) -> TripRequirements:
    return TripRequirements(
        destination="Tokyo",
        duration_days=duration_days,
        start_date=date(2026, 4, 1),
        travelers=travelers,
        budget=budget
        if budget is not None
        else _money("1000", "USD", DataStatus.USER_INPUT),
        interests=["food", "culture"],
        travel_style=TravelStyle.BALANCED,
    )


def _attraction_item(
    seq: int,
    name: str,
    cost: Money | None,
    transport: TransportLeg | None = None,
) -> ItineraryItem:
    """构造一个最小合法行程项：时间递增，第二项起带交通。"""
    start_hour = 8 + seq
    return ItineraryItem(
        seq=seq,
        type=ItineraryItemType.ATTRACTION,
        name=name,
        start_time=time(start_hour, 0),
        end_time=time(start_hour + 1, 0),
        cost=cost,
        transport_from_previous=transport,
    )


def _walk_leg() -> TransportLeg:
    return TransportLeg(
        mode=TransportMode.WALK,
        duration_minutes=20,
        distance_meters=1500,
        provenance=Provenance(source="mock:haversine", status=DataStatus.MOCK),
    )


def _itinerary(
    requirements: TripRequirements,
    ticket: Money | None = None,
) -> Itinerary:
    """第 1 天两个景点（一个收费一个免费），其余天为空，便于精确断言金额。"""
    first = _attraction_item(1, "Tokyo Tower", ticket or _money("1200", "JPY"))
    second = _attraction_item(2, "Ueno Park", _money("0", "JPY"), _walk_leg())
    days = [ItineraryDay(day_index=1, items=[first, second])]
    days.extend(
        ItineraryDay(day_index=index, items=[])
        for index in range(2, requirements.duration_days + 1)
    )
    return Itinerary(trip=requirements, days=days)


def _by_category(breakdown) -> dict[BudgetCategory, object]:
    return {line.category: line for line in breakdown.lines}


def test_six_categories_and_total() -> None:
    """六类费用齐全，total 等于六条明细之和。"""
    requirements = _requirements()
    outcome = BudgetService().estimate(requirements, _itinerary(requirements))
    breakdown = outcome.breakdown

    assert {line.category for line in breakdown.lines} == set(BudgetCategory)
    assert breakdown.total is not None
    assert breakdown.total.amount == sum(
        (line.amount.amount for line in breakdown.lines), Decimal("0")
    )
    # 机票 450 + 住宿 280 + 门票 8 + 餐饮 165 + 其他 72.24 + 交通 0
    assert breakdown.total.amount == Decimal("975.24")
    assert breakdown.currency == "USD"
    assert breakdown.over_budget is False
    assert breakdown.remaining is not None
    assert breakdown.remaining.amount == Decimal("24.76")


def test_all_lines_are_estimated() -> None:
    """没有实时报价时，任何一条都不能伪装成真实价格。"""
    requirements = _requirements()
    breakdown = BudgetService().estimate(requirements, _itinerary(requirements)).breakdown

    assert all(line.amount.status == DataStatus.ESTIMATED for line in breakdown.lines)


def test_hotel_charges_nights_not_days() -> None:
    """3 天行程是 2 晚住宿，写成 3 晚会让所有总额偏大。"""
    requirements = _requirements(duration_days=3)
    breakdown = BudgetService().estimate(requirements, _itinerary(requirements)).breakdown

    hotel = _by_category(breakdown)[BudgetCategory.ACCOMMODATION]
    assert hotel.amount.amount == Decimal("280")  # 140 × 2 晚 × 1 间
    assert "2 晚" in hotel.note


def test_over_budget_amount() -> None:
    """预算 500 时应当判超支，并给出超出金额；remaining 与超支互斥。"""
    requirements = _requirements(budget=_money("500", "USD", DataStatus.USER_INPUT))
    breakdown = BudgetService().estimate(requirements, _itinerary(requirements)).breakdown

    assert breakdown.over_budget is True
    assert breakdown.over_budget_amount is not None
    assert breakdown.over_budget_amount.amount == Decimal("475.24")
    assert breakdown.remaining is None


def test_unknown_ticket_is_not_zero() -> None:
    """门票价格为 None 时不计入金额，但必须留下 warning。"""
    requirements = _requirements()
    itinerary = _itinerary(requirements, ticket=None)
    # 把第一个景点的门票改成「未知」
    itinerary.days[0].items[0].cost = None

    outcome = BudgetService().estimate(requirements, itinerary)
    tickets = _by_category(outcome.breakdown)[BudgetCategory.TICKETS]

    assert tickets.amount.amount == Decimal("0")  # 只剩免费景点
    assert any("门票价格未知" in warning for warning in outcome.warnings)


def test_foreign_currency_is_converted() -> None:
    """门票是 JPY，账本是 USD：全部条目同币种，并在 warning 里说明折算。"""
    requirements = _requirements()
    outcome = BudgetService().estimate(requirements, _itinerary(requirements))

    assert all(line.amount.currency == "USD" for line in outcome.breakdown.lines)
    assert any("折算" in warning for warning in outcome.warnings)
    tickets = _by_category(outcome.breakdown)[BudgetCategory.TICKETS]
    assert "JPY" in tickets.note


def test_no_budget_means_no_over_budget() -> None:
    """用户没给预算时不替他假设上限：over_budget 恒为 False。"""
    requirements = _requirements()
    requirements = requirements.model_copy(update={"budget": None})
    breakdown = BudgetService().estimate(requirements, _itinerary(requirements)).breakdown

    assert breakdown.budget_limit is None
    assert breakdown.over_budget is False
    assert breakdown.remaining is None


def test_empty_itinerary_still_priced() -> None:
    """候选景点为空时，机票住宿餐饮照算，不崩在 days[0]。"""
    requirements = _requirements()
    itinerary = Itinerary(trip=requirements, days=[])

    breakdown = BudgetService().estimate(requirements, itinerary).breakdown

    assert breakdown.total is not None
    assert breakdown.total.amount > 0
    assert _by_category(breakdown)[BudgetCategory.TICKETS].amount.amount == 0


def test_itinerary_service_fills_budget() -> None:
    """端到端：ItineraryService 组装完行程后必须填上 budget。"""
    service = ItineraryService(
        attraction_service=AttractionService(provider=MockAttractionProvider()),
        route_service=RouteService(provider=MockRouteProvider()),
        weather_service=WeatherService(provider=MockWeatherProvider()),
        planner=RuleBasedItineraryPlanner(),
        budget_service=BudgetService(),
        restaurant_service=RestaurantService(provider=MockRestaurantProvider()),
    )
    requirements = _requirements(duration_days=2)

    itinerary = asyncio.run(service.plan(requirements))

    assert itinerary.budget is not None
    assert itinerary.budget.total is not None
    assert itinerary.budget.total.amount > 0
