"""开发阶段的内存数据源：不联网、结果可复现。

为什么 Mock 也要字段齐全：第 9 阶段 Evaluator 需要开放时间、游玩时长、坐标、票价，
第 10 阶段 Optimizer 需要坐标算距离。现在偷懒留空，后面必然返工。
"""

import math
import random
import zlib
from collections.abc import Sequence
from datetime import date as Date
from datetime import datetime, time, timedelta, timezone

from backend.providers.base import (
    AttractionProvider,
    RestaurantProvider,
    RouteProvider,
    WeatherProvider,
)
from backend.schemas.attraction import Attraction, AttractionCategory, OpeningInterval
from backend.schemas.common import (
    DataStatus,
    GeoPoint,
    Money,
    Provenance,
    TransportMode,
)
from backend.schemas.restaurant import Restaurant
from backend.schemas.route import RouteMatrixEntry
from backend.schemas.weather import DailyWeather, WeatherCondition

# 固定时间戳，保证 Mock 数据在测试里完全可复现（而不是每次 now()）
_MOCK_FETCHED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _mock_provenance(source: str = "mock:tokyo_dataset") -> Provenance:
    """所有 Mock 数据共用的溯源信息。source 按数据类型区分，便于排查。"""
    return Provenance(
        source=source,
        fetched_at=_MOCK_FETCHED_AT,
        status=DataStatus.MOCK,
        valid_until=None,
    )


def _everyday(opens_at: time, closes_at: time) -> list[OpeningInterval]:
    """一周七天同一时段。"""
    return [
        OpeningInterval(weekday=weekday, opens_at=opens_at, closes_at=closes_at)
        for weekday in range(7)
    ]


def _free() -> Money:
    """免费景点用 0 表示「确认免费」，与「未知」的 None 区分开。"""
    return Money(amount=0, currency="JPY", status=DataStatus.MOCK)


# 东京最小数据集：10 条，覆盖 culture / food / shopping / nature 四类，
# 坐标分散在浅草、涩谷、银座、秋叶原、新宿、上野、台场，供第 7 阶段做地理分组。
TOKYO_ATTRACTIONS: list[Attraction] = [
    Attraction(
        id="tokyo-sensoji",
        name="Senso-ji Temple",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.CULTURE,
        tags=["temple", "asakusa", "history"],
        description="东京最古老的寺庙，浅草地标。",
        location=GeoPoint(lat=35.7148, lng=139.7967),
        rating=4.6,
        review_count=82000,
        opening_hours=_everyday(time(6, 0), time(17, 0)),
        estimated_duration_minutes=90,
        ticket_price=_free(),
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-meiji-jingu",
        name="Meiji Jingu",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.CULTURE,
        tags=["shrine", "forest", "shibuya"],
        description="明治神宫，紧邻原宿的森林神社。",
        location=GeoPoint(lat=35.6764, lng=139.6993),
        rating=4.5,
        review_count=61000,
        opening_hours=_everyday(time(6, 0), time(17, 0)),
        estimated_duration_minutes=75,
        ticket_price=_free(),
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-tokyo-tower",
        name="Tokyo Tower",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.CULTURE,
        tags=["landmark", "viewpoint", "minato"],
        description="东京经典地标，主观景台可俯瞰市区。",
        location=GeoPoint(lat=35.6586, lng=139.7454),
        rating=4.3,
        review_count=72000,
        opening_hours=_everyday(time(9, 0), time(22, 30)),
        estimated_duration_minutes=60,
        ticket_price=Money(amount=1200, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-teamlab-planets",
        name="teamLab Planets",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.CULTURE,
        tags=["digital_art", "immersive", "toyosu"],
        description="丰洲的沉浸式数字艺术馆，需要提前预约时段。",
        location=GeoPoint(lat=35.6480, lng=139.7915),
        rating=4.4,
        review_count=39000,
        opening_hours=_everyday(time(9, 0), time(22, 0)),
        estimated_duration_minutes=90,
        ticket_price=Money(amount=3800, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-tsukiji-outer-market",
        name="Tsukiji Outer Market",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.FOOD,
        tags=["sushi", "seafood", "street_food"],
        description="筑地场外市场，早餐和海鲜小吃集中地。",
        location=GeoPoint(lat=35.6655, lng=139.7708),
        rating=4.4,
        review_count=45000,
        opening_hours=_everyday(time(6, 0), time(14, 0)),
        estimated_duration_minutes=90,
        ticket_price=None,
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-ichiran-shibuya",
        name="Ichiran Shibuya",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.FOOD,
        tags=["ramen", "solo_friendly", "shibuya"],
        description="一兰拉面涩谷店，24 小时营业，适合单人快速用餐。",
        location=GeoPoint(lat=35.6614, lng=139.7010),
        rating=4.1,
        review_count=21000,
        # 24 小时营业，用空列表表示「不限时段」，Evaluator 需要处理这个分支
        opening_hours=[],
        estimated_duration_minutes=60,
        ticket_price=None,
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-ginza-six",
        name="Ginza Six",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.SHOPPING,
        tags=["department_store", "luxury", "ginza"],
        description="银座大型商业综合体，购物和屋顶花园。",
        location=GeoPoint(lat=35.6699, lng=139.7636),
        rating=4.2,
        review_count=18000,
        opening_hours=_everyday(time(10, 30), time(20, 30)),
        estimated_duration_minutes=120,
        ticket_price=_free(),
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-akihabara-electric-town",
        name="Akihabara Electric Town",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.SHOPPING,
        tags=["electronics", "anime", "akihabara"],
        description="秋叶原电器街，动漫与电子产品集中区。",
        location=GeoPoint(lat=35.6984, lng=139.7730),
        rating=4.3,
        review_count=52000,
        opening_hours=_everyday(time(10, 0), time(20, 0)),
        estimated_duration_minutes=90,
        ticket_price=None,
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-shinjuku-gyoen",
        name="Shinjuku Gyoen",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.NATURE,
        tags=["garden", "park", "shinjuku"],
        description="新宿御苑，市中心的大型庭园，适合慢行。",
        location=GeoPoint(lat=35.6852, lng=139.7100),
        rating=4.5,
        review_count=34000,
        opening_hours=_everyday(time(9, 0), time(16, 0)),
        estimated_duration_minutes=120,
        ticket_price=Money(amount=500, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance(),
    ),
    Attraction(
        id="tokyo-ueno-park",
        name="Ueno Park",
        city="Tokyo",
        country="Japan",
        category=AttractionCategory.NATURE,
        tags=["park", "museum", "ueno"],
        description="上野公园，博物馆群与赏樱地，全天开放。",
        location=GeoPoint(lat=35.7141, lng=139.7774),
        rating=4.4,
        review_count=48000,
        opening_hours=_everyday(time(5, 0), time(23, 0)),
        estimated_duration_minutes=90,
        ticket_price=_free(),
        provenance=_mock_provenance(),
    ),
]


class MockAttractionProvider(AttractionProvider):
    """内存数据源。city 匹配大小写不敏感。"""

    def __init__(self, dataset: list[Attraction] | None = None) -> None:
        # 允许测试注入自定义数据集，避免测试依赖全局常量
        self._dataset: list[Attraction] = (
            list(dataset) if dataset is not None else TOKYO_ATTRACTIONS
        )

    async def search_attractions(
        self,
        city: str,
        interests: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Attraction]:
        city_key = city.strip().casefold()
        wanted = {
            interest.strip().casefold()
            for interest in (interests or [])
            if interest.strip()
        }

        matched: list[Attraction] = []
        for attraction in self._dataset:
            if attraction.city.casefold() != city_key:
                continue
            if wanted and not (wanted & {attraction.category.value, *attraction.tags}):
                continue
            matched.append(attraction)

        # Provider 只做兜底排序（评分降序、id 升序保证稳定），兴趣计分属于 Service
        matched.sort(key=lambda item: (-(item.rating or 0.0), item.id))

        if limit <= 0:
            return []
        return matched[:limit]

    async def get_attraction(self, attraction_id: str) -> Attraction | None:
        for attraction in self._dataset:
            if attraction.id == attraction_id:
                return attraction
        return None


# ---------------------------- 餐厅 Mock ----------------------------

# 东京最小餐厅数据集：12 条，覆盖 ramen / sushi / tempura / soba / tonkatsu /
# cafe / vegan 七类菜系，坐标贴近景点，供「每天一顿午餐」的确定性插入使用。
TOKYO_RESTAURANTS: list[Restaurant] = [
    Restaurant(
        id="tokyo-ichiran-shibuya",
        name="Ichiran Shibuya",
        city="Tokyo",
        country="Japan",
        cuisines=["ramen"],
        description="连锁豚骨拉面，单人隔间座位，适合快速午餐。",
        location=GeoPoint(lat=35.6614, lng=139.7006),
        rating=4.4,
        review_count=21000,
        opening_hours=_everyday(time(10, 0), time(23, 0)),
        average_price=Money(amount=1200, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-afuri-ebisu",
        name="Afuri Ebisu",
        city="Tokyo",
        country="Japan",
        cuisines=["ramen"],
        description="柚子盐拉面，汤头清爽。",
        location=GeoPoint(lat=35.6467, lng=139.7101),
        rating=4.5,
        review_count=9800,
        opening_hours=_everyday(time(11, 0), time(23, 0)),
        average_price=Money(amount=1400, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-nakiryu-otsuka",
        name="Nakiryu",
        city="Tokyo",
        country="Japan",
        cuisines=["ramen"],
        description="米其林推荐的担担面专门店，饭点排队较长。",
        location=GeoPoint(lat=35.7185, lng=139.7280),
        rating=4.6,
        review_count=7600,
        opening_hours=_everyday(time(11, 30), time(21, 0)),
        average_price=Money(amount=1300, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-sushi-dai-toyosu",
        name="Sushi Dai",
        city="Tokyo",
        country="Japan",
        cuisines=["sushi"],
        description="丰洲市场人气寿司店，需要早到取号。",
        location=GeoPoint(lat=35.6455, lng=139.7866),
        rating=4.7,
        review_count=15400,
        opening_hours=_everyday(time(5, 30), time(14, 0)),
        average_price=Money(amount=4000, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-uobei-shibuya",
        name="Uobei Shibuya",
        city="Tokyo",
        country="Japan",
        cuisines=["sushi"],
        description="高速传送带寿司，性价比高，翻台快。",
        location=GeoPoint(lat=35.6595, lng=139.6987),
        rating=4.2,
        review_count=11200,
        opening_hours=_everyday(time(11, 0), time(23, 0)),
        average_price=Money(amount=1500, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-tsukiji-sushisen",
        name="Tsukiji Sushisen",
        city="Tokyo",
        country="Japan",
        cuisines=["sushi"],
        description="筑地场外市场的寿司老店，午市套餐实惠。",
        location=GeoPoint(lat=35.6654, lng=139.7707),
        rating=4.3,
        review_count=6300,
        opening_hours=_everyday(time(9, 0), time(21, 0)),
        average_price=Money(amount=3500, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-tempura-kondo",
        name="Tempura Kondo",
        city="Tokyo",
        country="Japan",
        cuisines=["tempura"],
        description="银座天妇罗名店，午市套餐相对友好。",
        location=GeoPoint(lat=35.6717, lng=139.7649),
        rating=4.6,
        review_count=5200,
        opening_hours=_everyday(time(12, 0), time(21, 0)),
        average_price=Money(amount=8000, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-kanda-matsuya",
        name="Kanda Matsuya",
        city="Tokyo",
        country="Japan",
        cuisines=["soba"],
        description="神田老字号荞麦面，木造店铺。",
        location=GeoPoint(lat=35.6960, lng=139.7660),
        rating=4.4,
        review_count=8700,
        opening_hours=_everyday(time(11, 0), time(20, 0)),
        average_price=Money(amount=1000, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-maisen-omotesando",
        name="Tonkatsu Maisen",
        city="Tokyo",
        country="Japan",
        cuisines=["tonkatsu"],
        description="表参道炸猪排，黑豚定食是招牌。",
        location=GeoPoint(lat=35.6654, lng=139.7120),
        rating=4.4,
        review_count=13500,
        opening_hours=_everyday(time(11, 0), time(22, 0)),
        average_price=Money(amount=1800, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-bills-omotesando",
        name="bills Omotesando",
        city="Tokyo",
        country="Japan",
        cuisines=["cafe", "western"],
        description="澳式早午餐，适合安排轻松的一天。",
        location=GeoPoint(lat=35.6670, lng=139.7100),
        rating=4.3,
        review_count=10400,
        opening_hours=_everyday(time(8, 30), time(22, 0)),
        average_price=Money(amount=2200, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-ain-soph-shinjuku",
        name="Ain Soph Journey",
        city="Tokyo",
        country="Japan",
        cuisines=["vegan", "cafe"],
        description="新宿素食餐厅，薄饼和套餐都受欢迎。",
        location=GeoPoint(lat=35.6938, lng=139.7034),
        rating=4.5,
        review_count=4100,
        opening_hours=_everyday(time(11, 0), time(21, 0)),
        average_price=Money(amount=2000, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
    Restaurant(
        id="tokyo-ginza-kagari",
        name="Ginza Kagari",
        city="Tokyo",
        country="Japan",
        cuisines=["ramen"],
        description="鸡白汤拉面，银座午市排队店。",
        location=GeoPoint(lat=35.6712, lng=139.7655),
        rating=4.5,
        review_count=6900,
        opening_hours=_everyday(time(11, 0), time(21, 0)),
        average_price=Money(amount=1600, currency="JPY", status=DataStatus.MOCK),
        provenance=_mock_provenance("mock:tokyo_restaurants"),
    ),
]


class MockRestaurantProvider(RestaurantProvider):
    """内存餐厅数据源。city 匹配大小写不敏感，cuisines 取交集。"""

    def __init__(self, dataset: list[Restaurant] | None = None) -> None:
        self._dataset: list[Restaurant] = (
            list(dataset) if dataset is not None else TOKYO_RESTAURANTS
        )

    async def search_restaurants(
        self,
        city: str,
        cuisines: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Restaurant]:
        city_key = city.strip().casefold()
        wanted = {
            cuisine.strip().casefold()
            for cuisine in (cuisines or [])
            if cuisine.strip()
        }

        matched: list[Restaurant] = []
        for restaurant in self._dataset:
            if restaurant.city.casefold() != city_key:
                continue
            if wanted and not (wanted & set(restaurant.cuisines)):
                continue
            matched.append(restaurant)

        # Provider 只做兜底排序（评分降序、id 升序保证稳定），偏好排序属于 Service
        matched.sort(key=lambda item: (-(item.rating or 0.0), item.id))

        if limit <= 0:
            return []
        return matched[:limit]


# ---------------------------- 路线 Mock ----------------------------

_EARTH_RADIUS_METERS = 6_371_000

# 估算参数：速度决定耗时，绕行系数把直线距离放大成「实际走法」，
# 固定开销覆盖候车、停车这类无法用距离表达的成本。
_ROUTE_SPEED_KMH: dict[TransportMode, float] = {
    TransportMode.WALK: 4.5,
    TransportMode.TRANSIT: 18.0,
    TransportMode.DRIVE: 25.0,
    TransportMode.TAXI: 25.0,
}

_ROUTE_DETOUR_FACTOR: dict[TransportMode, float] = {
    TransportMode.WALK: 1.2,
    TransportMode.TRANSIT: 1.3,
    TransportMode.DRIVE: 1.35,
    TransportMode.TAXI: 1.35,
}

_ROUTE_OVERHEAD_MINUTES: dict[TransportMode, int] = {
    TransportMode.WALK: 0,
    TransportMode.TRANSIT: 5,
    TransportMode.DRIVE: 3,
    TransportMode.TAXI: 3,
}


def _haversine_meters(origin: GeoPoint, destination: GeoPoint) -> float:
    """两点球面距离（米）。只用于 Mock 估算，真实路线必须走 Routes API。"""
    lat1, lng1, lat2, lng2 = map(
        math.radians, (origin.lat, origin.lng, destination.lat, destination.lng)
    )
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    h = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_METERS * math.asin(math.sqrt(h))


class MockRouteProvider(RouteProvider):
    """直线距离 + 分模式速度的估算型路线源。

    它是估算而不是导航：distance 乘了绕行系数，duration 加了固定开销，
    所以 provenance.status 必须是 MOCK，不能被当成真实路况。
    """

    async def get_route_legs(
        self,
        city: str,
        attractions: Sequence[Attraction],
        mode: TransportMode,
    ) -> list[RouteMatrixEntry]:
        entries: list[RouteMatrixEntry] = []
        for origin in attractions:
            for destination in attractions:
                if origin.id == destination.id:
                    continue

                straight_meters = _haversine_meters(
                    origin.location, destination.location
                )
                estimated_meters = straight_meters * _ROUTE_DETOUR_FACTOR[mode]
                minutes = (
                    math.ceil(
                        estimated_meters / 1000 / _ROUTE_SPEED_KMH[mode] * 60
                    )
                    + _ROUTE_OVERHEAD_MINUTES[mode]
                )

                entries.append(
                    RouteMatrixEntry(
                        from_id=origin.id,
                        to_id=destination.id,
                        mode=mode,
                        duration_minutes=max(0, minutes),
                        distance_meters=int(round(estimated_meters)),
                        provenance=_mock_provenance("mock:haversine"),
                    )
                )
        return entries


# ---------------------------- 天气 Mock ----------------------------

# 东京各月基准气温，仅用于让模拟数据看起来合理，不参与任何业务判断。
_MONTHLY_BASE_TEMP_C: dict[int, int] = {
    1: 5,
    2: 6,
    3: 10,
    4: 15,
    5: 20,
    6: 23,
    7: 27,
    8: 29,
    9: 25,
    10: 19,
    11: 13,
    12: 8,
}


class MockWeatherProvider(WeatherProvider):
    """确定性天气源：同一城市同一天永远返回同一结果。

    用 crc32 而不是 hash() 做种子，因为 str 的 hash 受 PYTHONHASHSEED 影响，
    跨进程不稳定，测试会随机失败。
    """

    async def get_forecast(
        self,
        city: str,
        start_date: Date,
        days: int,
    ) -> list[DailyWeather]:
        forecasts: list[DailyWeather] = []
        for offset in range(days):
            current = start_date + timedelta(days=offset)
            rng = random.Random(
                zlib.crc32(f"{city.casefold()}:{current.isoformat()}".encode())
            )

            base_temp = _MONTHLY_BASE_TEMP_C[current.month] + rng.randint(-3, 3)
            condition = self._pick_condition(rng, current.month)

            if condition is WeatherCondition.SUNNY:
                precipitation = 0
            elif condition is WeatherCondition.CLOUDY:
                precipitation = rng.randint(10, 40)
            else:
                precipitation = rng.randint(50, 90)

            forecasts.append(
                DailyWeather(
                    date=current,
                    condition=condition,
                    temp_min_c=base_temp - 4,
                    temp_max_c=base_temp + 3,
                    precipitation_probability=precipitation,
                    provenance=_mock_provenance("mock:weather"),
                )
            )
        return forecasts

    @staticmethod
    def _pick_condition(rng: random.Random, month: int) -> WeatherCondition:
        """按固定概率抽天气；只有冬季才可能下雪。"""
        roll = rng.randint(0, 99)
        if roll < 55:
            return WeatherCondition.SUNNY
        if roll < 80:
            return WeatherCondition.CLOUDY
        if roll < 92:
            return WeatherCondition.RAIN
        return WeatherCondition.SNOW if month in (12, 1, 2) else WeatherCondition.RAIN
