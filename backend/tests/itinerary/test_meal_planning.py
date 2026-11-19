import asyncio
from datetime import date, time
from unittest.mock import AsyncMock

import pytest

from backend.schemas.attraction import Attraction
from backend.schemas.common import GeoPoint, Provenance, TransportMode
from backend.schemas.itinerary import Itinerary, ItineraryDay, ItineraryItem, TransportLeg
from backend.schemas.restaurant import Restaurant
from backend.schemas.trip import TripRequirements
from backend.services.meal_planning import add_breakfast_and_dinner
from backend.services.trip_data import GaodeTripData


@pytest.mark.parametrize(("restaurant_count", "use_nearby"), [(3, False), (4, False), (4, True)])
def test_meals_use_unique_restaurants_across_days(restaurant_count: int, use_nearby: bool) -> None:
    provenance = Provenance(source="test")
    anchor = GeoPoint(lat=22.8, lng=108.3)
    attractions = [
        Attraction(
            id=f"a{index}", name=f"景点{index}", city="南宁", location=anchor,
            estimated_duration_minutes=60, provenance=provenance,
        )
        for index in (1, 2)
    ]
    itinerary = Itinerary(
        trip=TripRequirements(destination="南宁", duration_days=2),
        days=[
            ItineraryDay(day_index=index, date=date(2026, 11, 19 + index), items=[
                ItineraryItem(
                    seq=1, type="attraction", name=attraction.name, attraction_id=attraction.id,
                    start_time=time(9), end_time=time(10),
                )
            ])
            for index, attraction in enumerate(attractions, start=1)
        ],
    )
    restaurants = [
        Restaurant(
            id=f"r{index}", name=f"餐厅{index}", city="南宁",
            location=GeoPoint(lat=anchor.lat, lng=anchor.lng + (index + 1) / 1000),
            provenance=provenance,
        )
        for index in range(restaurant_count)
    ]
    leg = TransportLeg(
        mode=TransportMode.WALK, duration_minutes=10, distance_meters=600,
        provenance=provenance,
    )
    data = AsyncMock(spec=GaodeTripData)
    data.walking_leg.return_value = leg
    data.nearby_restaurants.return_value = restaurants if use_nearby else []
    initial = restaurants[:3] if use_nearby else restaurants
    asyncio.run(add_breakfast_and_dinner(itinerary, attractions, [*initial, initial[0]], data))

    items = [item for day in itinerary.days for item in day.items if item.restaurant_id]
    assert [item.restaurant_id for item in items] == [restaurant.id for restaurant in restaurants]
    assert [item.meal for item in items] == ["breakfast", "dinner", "breakfast", "dinner"][:restaurant_count]
    first = itinerary.days[0].items
    assert first[0].end_time == time(8, 50)
    assert first[1].transport_from_previous == leg
    assert first[2].start_time == time(17, 30)
    assert first[2].transport_from_previous == leg
    assert any("第 2 天" in warning and "晚餐" in warning for warning in itinerary.warnings) == (
        restaurant_count == 3
    )
    if use_nearby:
        data.nearby_restaurants.assert_awaited_once_with("南宁", anchor, keywords="餐厅")
