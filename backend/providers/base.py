"""外部数据源抽象。

业务层只依赖这组接口，不依赖任何厂商 SDK。换成 Google Places 时，
Service 与 API 一行都不用改。
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

from backend.schemas.attraction import Attraction
from backend.schemas.common import TransportMode
from backend.schemas.route import RouteMatrixEntry
from backend.schemas.weather import DailyWeather


class AttractionProvider(ABC):
    """景点数据源。

    实现方只负责「取数 + 转换」，不做用户偏好排序——那是 Service 的职责。
    """

    @abstractmethod
    async def search_attractions(
        self,
        city: str,
        interests: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Attraction]:
        """按城市检索景点候选集。

        约定：
        - 城市大小写不敏感
        - interests 为 None/空 表示不过滤
        - 返回空列表表示「查到了但没有匹配」，不要抛异常
        - 抛 ProviderError（后续新增）才表示数据源故障
        """

    @abstractmethod
    async def get_attraction(self, attraction_id: str) -> Attraction | None:
        """按 id 取景点详情。找不到返回 None。"""


class RestaurantProvider(ABC):
    """餐厅数据源。

    第一版按需求「作为景点搜索的扩展」实现，接口先占位，
    第 5 阶段再决定是否真的接入独立 Provider。
    """

    @abstractmethod
    async def search_restaurants(
        self,
        city: str,
        cuisine: str | None = None,
        limit: int = 20,
    ) -> list[Attraction]:
        """按城市 / 菜系检索餐厅。返回结构复用 Attraction（category=food）。"""


class RouteProvider(ABC):
    """路线数据源。

    只负责「按给定景点和交通方式算出每条有向边的耗时/距离」，
    矩阵组装、去重与规模限制属于 RouteService。
    """

    @abstractmethod
    async def get_route_legs(
        self,
        city: str,
        attractions: Sequence[Attraction],
        mode: TransportMode,
    ) -> list[RouteMatrixEntry]:
        """返回 attractions 中所有「不同景点之间」的有向边。

        约定：
        - 不允许自环（from_id != to_id）
        - 查不到某条边时直接不返回，稀疏矩阵由上层处理
        - 数据源故障抛 ProviderUnavailableError
        """


class WeatherProvider(ABC):
    """天气数据源。只返回逐日概况，不做行程相关的任何判断。"""

    @abstractmethod
    async def get_forecast(
        self,
        city: str,
        start_date: date,
        days: int,
    ) -> list[DailyWeather]:
        """返回从 start_date 起连续 days 天的天气，调用方负责校验 days 范围。"""
