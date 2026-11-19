"""按每天首末景点为整趟行程分配不同的早餐和晚餐餐厅。"""

import logging
from collections.abc import Sequence
from datetime import date, time
from typing import Literal

from backend.schemas.attraction import Attraction
from backend.schemas.common import GeoPoint
from backend.schemas.itinerary import Itinerary, ItineraryDay, ItineraryItem, ItineraryItemType, TransportLeg
from backend.schemas.restaurant import Restaurant
from backend.services.trip_data import GaodeTripData
from backend.tools.gaode_mcp_tools import haversine_meters

logger = logging.getLogger(__name__)
Meal = Literal["breakfast", "dinner"]


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _meal_slot(
    restaurant: Restaurant, day: date | None, meal: Meal, anchor: ItineraryItem, leg: TransportLeg
) -> tuple[int, int] | None:
    if meal == "breakfast":
        earliest = 7 * 60
        latest = min(9 * 60, _minutes(anchor.start_time) - leg.duration_minutes)
        duration = 45
    else:
        earliest = max(17 * 60 + 30, _minutes(anchor.end_time) + leg.duration_minutes)
        latest, duration = 21 * 60 + 30, 60
    windows = [(earliest, latest)]
    if day and restaurant.opening_hours:
        windows = [
            (max(earliest, _minutes(interval.opens_at)), min(latest, _minutes(interval.closes_at)))
            for interval in restaurant.opening_hours
            if interval.weekday == day.weekday()
        ]
    valid = [(start, end) for start, end in windows if end - start >= duration]
    if not valid:
        return None
    if meal == "breakfast":
        end = max(end for _, end in valid)
        return end - duration, end
    start = min(start for start, _ in valid)
    return start, start + duration


async def _choose_meal(
    day: ItineraryDay,
    meal: Meal,
    anchor: ItineraryItem,
    point: GeoPoint,
    candidates: Sequence[Restaurant],
    data: GaodeTripData,
) -> tuple[ItineraryItem, TransportLeg] | None:
    nearby = sorted(candidates, key=lambda restaurant: haversine_meters(point, restaurant.location))
    breakfast = meal == "breakfast"
    for offset in range(0, len(nearby), 3):
        options: list[tuple[int, str, ItineraryItem, TransportLeg]] = []
        for restaurant in nearby[offset : offset + 3]:
            origin, destination = (restaurant.location, point) if breakfast else (point, restaurant.location)
            try:
                leg = await data.walking_leg(origin, destination)
            except Exception as exc:
                logger.warning("餐厅 %s 路线查询失败（%s）", restaurant.id, type(exc).__name__)
                continue
            if leg is None or leg.distance_meters is None:
                continue
            slot = _meal_slot(restaurant, day.date, meal, anchor, leg)
            if slot is None:
                continue
            start, end = slot
            item = ItineraryItem(
                seq=1,
                type=ItineraryItemType.RESTAURANT,
                name=restaurant.name,
                restaurant_id=restaurant.id,
                meal=meal,
                start_time=time(*divmod(start, 60)),
                end_time=time(*divmod(end, 60)),
                cost=restaurant.average_price,
                transport_from_previous=None if breakfast else leg,
            )
            options.append((leg.distance_meters, restaurant.id, item, leg))
        if options:
            _, _, item, leg = min(options, key=lambda option: option[:2])
            return item, leg
    return None


async def add_breakfast_and_dinner(
    itinerary: Itinerary,
    attractions: Sequence[Attraction],
    candidates: Sequence[Restaurant],
    data: GaodeTripData,
) -> None:
    """依日期分配，餐厅一经选中，立即从剩余候选中移除。"""
    locations = {item.id: item.location for item in attractions}
    remaining = {restaurant.id: restaurant for restaurant in candidates}
    used_ids: set[str] = set()
    for day in itinerary.days:
        items = list(day.items)
        for meal, label, anchor_index in (("breakfast", "早餐", 0), ("dinner", "晚餐", -1)):
            anchor = day.items[anchor_index] if day.items else None
            point = locations.get(anchor.attraction_id) if anchor and anchor.attraction_id else None
            choice = (
                await _choose_meal(day, meal, anchor, point, list(remaining.values()), data)
                if anchor is not None and point is not None
                else None
            )
            if choice is None and anchor is not None and point is not None:
                try:
                    nearby = await data.nearby_restaurants(
                        itinerary.trip.destination, point, keywords=label if meal == "breakfast" else "餐厅"
                    )
                except Exception as exc:
                    logger.warning("第 %s 天%s补查失败（%s）", day.day_index, label, type(exc).__name__)
                else:
                    extra = {
                        restaurant.id: restaurant for restaurant in nearby
                        if restaurant.id not in used_ids
                    }
                    choice = await _choose_meal(day, meal, anchor, point, list(extra.values()), data)
            if choice is None:
                itinerary.warnings.append(f"第 {day.day_index} 天未找到符合路线和时段的{label}餐厅")
                continue
            item, leg = choice
            if item.restaurant_id is not None:
                used_ids.add(item.restaurant_id)
                remaining.pop(item.restaurant_id, None)
            if meal == "breakfast":
                items[0] = items[0].model_copy(update={"transport_from_previous": leg})
                items.insert(0, item)
            else:
                items.append(item)
        day.items = [
            item.model_copy(update={"seq": index}) for index, item in enumerate(items, start=1)
        ]
