"""将高德路线和天气工具结果转为排程所需的数据。"""

import asyncio
import json
from datetime import date
from itertools import permutations
from math import ceil

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from backend.schemas.attraction import Attraction
from backend.schemas.common import DataStatus, GeoPoint, Provenance, TransportMode
from backend.schemas.itinerary import TransportLeg
from backend.schemas.restaurant import Restaurant
from backend.schemas.route import RouteMatrix, RouteMatrixEntry
from backend.schemas.weather import DailyWeather, WeatherCondition, WeatherForecast
from backend.services.restaurant_assembly import restaurant_from_summary
from backend.tools.gaode_mcp_tools import RestaurantSummary


class GaodeTripData:
    def __init__(self, tool: BaseTool | None, *, concurrency: int) -> None:
        self._tool = tool
        self._limit = asyncio.Semaphore(concurrency)
        self._provenance = Provenance(source="amap_mcp", status=DataStatus.REAL)

    async def _invoke(self, **arguments) -> dict:
        if self._tool is None:
            return {}
        async with self._limit:
            result = await self._tool.ainvoke(arguments)
        payload = json.loads(result)
        return payload if isinstance(payload, dict) and payload.get("ok") else {}

    async def _query(self, **arguments) -> dict:
        data = (await self._invoke(**arguments)).get("data")
        return data if isinstance(data, dict) else {}

    async def nearby_restaurants(
        self, city: str, point: GeoPoint, *, keywords: str
    ) -> list[Restaurant]:
        payload = await self._invoke(
            action="around_search", poi_kind="restaurant", keywords=keywords,
            longitude=point.lng, latitude=point.lat, radius=1800, limit=12,
        )
        restaurants = []
        for row in payload.get("items") or []:
            try:
                summary = RestaurantSummary.model_validate(row)
            except ValidationError:
                continue
            restaurant = restaurant_from_summary(summary, city=city)
            if restaurant is not None:
                restaurants.append(restaurant)
        return restaurants

    async def walking_leg(self, origin: GeoPoint, destination: GeoPoint) -> TransportLeg | None:
        data = await self._query(
            action="route",
            route_mode="walking",
            origin=f"{origin.lng},{origin.lat}",
            destination=f"{destination.lng},{destination.lat}",
        )
        route = data.get("route") or data
        for path in route.get("paths") or []:
            try:
                seconds = float(path["duration"])
                distance = path.get("distance")
                return TransportLeg(
                    mode=TransportMode.WALK,
                    duration_minutes=ceil(seconds / 60),
                    distance_meters=int(distance) if distance not in (None, "") else None,
                    provenance=self._provenance,
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return None

    async def matrix(self, city: str, attractions: list[Attraction]) -> RouteMatrix:
        located = [(item.id, item.location) for item in attractions if item.location is not None]

        async def entry(left: tuple[str, GeoPoint], right: tuple[str, GeoPoint]):
            origin_id, origin = left
            destination_id, destination = right
            leg = await self.walking_leg(origin, destination)
            if leg is None:
                return None
            return RouteMatrixEntry(
                from_id=origin_id, to_id=destination_id, **leg.model_dump()
            )

        entries = await asyncio.gather(
            *(entry(left, right) for left, right in permutations(located, 2))
        )
        return RouteMatrix(
            city=city,
            mode=TransportMode.WALK,
            attraction_ids=[item.id for item in attractions],
            entries=[item for item in entries if item is not None],
        )

    async def forecast(self, city: str, start: date, end: date) -> WeatherForecast:
        data = await self._query(action="weather", city=city)
        forecasts = data.get("forecasts") or [data]
        days: dict[date, DailyWeather] = {}
        for forecast in forecasts:
            if not isinstance(forecast, dict):
                continue
            # 高德 MCP 直接返回逐日记录，部分调用方仍使用 forecasts[].casts。
            for cast in forecast.get("casts") or [forecast]:
                if not isinstance(cast, dict):
                    continue
                try:
                    day = date.fromisoformat(cast["date"])
                    if not start <= day <= end:
                        continue
                    description = str(cast["dayweather"])
                    condition = WeatherCondition.CLOUDY
                    if "雪" in description:
                        condition = WeatherCondition.SNOW
                    elif "雨" in description:
                        condition = WeatherCondition.RAIN
                    elif description == "晴":
                        condition = WeatherCondition.SUNNY
                    days[day] = DailyWeather(
                        date=day,
                        condition=condition,
                        temp_min_c=cast["nighttemp"],
                        temp_max_c=cast["daytemp"],
                        provenance=self._provenance,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return WeatherForecast(city=city, days=[days[day] for day in sorted(days)])
