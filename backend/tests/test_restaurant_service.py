"""餐厅功能测试（Service + 行程插入 + 预算联动）。

运行（在 travel_agent/ 目录下）：
    .venv\\Scripts\\python.exe -m pytest backend/tests/test_restaurant_service.py -v
"""

import asyncio
from datetime import date, time
from decimal import Decimal

import pytest

from backend.providers.mock_provider import (
    MockRestaurantProvider,
    TOKYO_ATTRACTIONS,
    TOKYO_RESTAURANTS,
)
from backend.schemas.attraction import Attraction, AttractionCategory, OpeningInterval
from backend.schemas.budget import BudgetCategory
from backend.schemas.common import DataStatus, GeoPoint, Money, Provenance, TransportMode
from backend.schemas.itinerary import (
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
)
from backend.schemas.planner import DayPlan, ItineraryDraft
from backend.schemas.route import RouteMatrix
from backend.schemas.trip import TripRequirements
from backend.services.budget_service import BudgetService
from backend.services.itinerary_builder import build_day, build_itinerary
from backend.services.restaurant_service import RestaurantService


def _service() -> RestaurantService:
    return RestaurantService(provider=MockRestaurantProvider())


def _attraction(index: int, duration: int = 90) -> Attraction:
    return Attraction(
        id=f"sight-{index}",
        name=f"Sight {index}",
        city="Tokyo",
        category=AttractionCategory.CULTURE,
        location=GeoPoint(lat=35.0 + index / 100, lng=139.0 + index / 100),
        estimated_duration_minutes=duration,
        opening_hours=[
            OpeningInterval(weekday=weekday, opens_at=time(9, 0), closes_at=time(21, 0))
            for weekday in range(7)
        ],
        provenance=Provenance(source="test", status=DataStatus.MOCK),
    )


def _requirements(duration_days: int = 2) -> TripRequirements:
    return TripRequirements(
        destination="Tokyo",
        duration_days=duration_days,
        start_date=date(2026, 4, 10),
        travelers=1,
        interests=["food"],
        travel_style="relaxed",
    )


# ---------------------------- Service ----------------------------


def test_search_returns_tokyo_restaurants_sorted_by_rating() -> None:
    response = asyncio.run(_service().search(city="Tokyo", limit=5))

    assert response.city == "Tokyo"
    assert response.total == 5
    ratings = [item.rating or 0 for item in response.items]
    assert ratings == sorted(ratings, reverse=True)


def test_search_filters_by_cuisine() -> None:
    response = asyncio.run(_service().search(city="tokyo", cuisines=["sushi"], limit=10))

    assert response.items
    assert all(item.matches_cuisine("sushi") for item in response.items)


def test_search_unknown_city_returns_empty() -> None:
    response = asyncio.run(_service().search(city="Atlantis", limit=5))

    assert response.total == 0
    assert response.items == []


def test_search_rejects_blank_city() -> None:
    with pytest.raises(ValueError, match="city"):
        asyncio.run(_service().search(city="   "))


def test_search_rejects_out_of_range_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(_service().search(city="Tokyo", limit=0))


# ---------------------------- 行程插入 ----------------------------


def test_build_day_inserts_lunch_after_noon_attraction() -> None:
    """第二项景点跨过 12:00 后应插入 60 分钟午餐，且 seq 连续。"""
    first, second = _attraction(0), _attraction(1)
    by_id = {item.id: item for item in (first, second)}
    matrix = RouteMatrix(
        city="Tokyo",
        mode=TransportMode.WALK,
        attraction_ids=[first.id, second.id],
        entries=[],
    )
    restaurant = TOKYO_RESTAURANTS[0]

    day, _ = build_day(
        attraction_ids=[first.id, second.id],
        attractions_by_id=by_id,
        matrix=matrix,
        day_index=1,
        restaurant=restaurant,
    )

    types = [item.type for item in day.items]
    assert types == [
        ItineraryItemType.ATTRACTION,
        ItineraryItemType.ATTRACTION,
        ItineraryItemType.RESTAURANT,
    ]
    assert [item.seq for item in day.items] == [1, 2, 3]
    lunch = day.items[2]
    assert lunch.name == restaurant.name
    assert lunch.cost == restaurant.average_price
    assert lunch.transport_from_previous is not None
    assert (
        lunch.transport_from_previous.provenance.source
        == "estimated:lunch-transfer"
    )


def test_build_itinerary_without_restaurants_keeps_attractions_only() -> None:
    """没传餐厅时行为不变，避免影响已有调用方和测试。"""
    attractions = TOKYO_ATTRACTIONS[:2]
    matrix = RouteMatrix(
        city="Tokyo",
        mode=TransportMode.WALK,
        attraction_ids=[item.id for item in attractions],
        entries=[],
    )
    draft = ItineraryDraft(
        days=[DayPlan(day_index=1, attraction_ids=[item.id for item in attractions])]
    )

    itinerary = build_itinerary(_requirements(1), draft, attractions, matrix)

    assert all(
        item.type is ItineraryItemType.ATTRACTION
        for item in itinerary.days[0].items
    )


def test_build_itinerary_assigns_one_restaurant_per_day() -> None:
    attractions = TOKYO_ATTRACTIONS[:4]
    matrix = RouteMatrix(
        city="Tokyo",
        mode=TransportMode.WALK,
        attraction_ids=[item.id for item in attractions],
        entries=[],
    )
    draft = ItineraryDraft(
        days=[
            DayPlan(day_index=1, attraction_ids=[attractions[0].id, attractions[1].id]),
            DayPlan(day_index=2, attraction_ids=[attractions[2].id, attractions[3].id]),
        ]
    )

    itinerary = build_itinerary(
        _requirements(2),
        draft,
        attractions,
        matrix,
        restaurants=TOKYO_RESTAURANTS[:2],
    )

    for day in itinerary.days:
        lunches = [
            item for item in day.items if item.type is ItineraryItemType.RESTAURANT
        ]
        assert len(lunches) == 1


# ---------------------------- 预算联动 ----------------------------


def test_budget_counts_restaurant_meals_and_remaining_food() -> None:
    """FOOD = 已排餐厅实际人均 + 其余餐食（日均 60%），避免与午餐重复计算。"""
    restaurant = TOKYO_RESTAURANTS[0]
    lunch = ItineraryItem(
        seq=2,
        type=ItineraryItemType.RESTAURANT,
        name=restaurant.name,
        start_time=time(12, 10),
        end_time=time(13, 10),
        cost=Money(amount=Decimal("1500"), currency="JPY", status=DataStatus.MOCK),
    )
    attraction = ItineraryItem(
        seq=1,
        type=ItineraryItemType.ATTRACTION,
        name="Sight",
        start_time=time(9, 0),
        end_time=time(10, 30),
        cost=Money(amount=Decimal("0"), currency="JPY", status=DataStatus.MOCK),
    )
    requirements = _requirements(1)
    itinerary = Itinerary(
        trip=requirements,
        days=[ItineraryDay(day_index=1, date=date(2026, 4, 10), items=[attraction, lunch])],
    )

    outcome = BudgetService().estimate(requirements, itinerary)

    food = next(
        line for line in outcome.breakdown.lines if line.category is BudgetCategory.FOOD
    )
    # 1500 JPY / 150 = 10 USD；其余餐食 46.75 × 0.6 × 1 天 × 1 人 = 28.05
    assert food.amount.amount == Decimal("38.05")
    assert "顿餐厅" in food.note
