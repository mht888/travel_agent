"""路线与交通矩阵业务层。

职责边界：
    Provider  只负责算出每条有向边的耗时/距离
    Service   校验输入、限制规模、组装 RouteMatrix、处理边界情况
    Schema    只负责「矩阵长什么样、什么样的矩阵算合法」
"""

from collections.abc import Sequence

from backend.providers.base import RouteProvider
from backend.schemas.attraction import Attraction
from backend.schemas.common import TransportMode
from backend.schemas.route import RouteMatrix


class RouteService:
    """构造景点间交通时间矩阵，供第 7 阶段排程与第 10 阶段优化使用。"""

    # N 个景点会产生 N*(N-1) 条有向边，超过这个规模应先做景点筛选
    MAX_ATTRACTIONS = 20

    def __init__(self, provider: RouteProvider) -> None:
        self._provider = provider

    async def build_matrix(
        self,
        city: str,
        attractions: Sequence[Attraction],
        mode: TransportMode = TransportMode.WALK,
    ) -> RouteMatrix:
        """为给定景点集合构造一个交通方式一致的有向矩阵。

        参数错误直接抛 ValueError，由调用方（Router / Graph Node）决定如何呈现；
        这里不引入 HTTP 概念，保证 Service 可以脱离 FastAPI 独立测试。
        """
        normalized_city = city.strip()
        if not normalized_city:
            raise ValueError("city 不能为空白")

        attraction_ids = [attraction.id for attraction in attractions]
        if not attraction_ids:
            raise ValueError("至少需要一个景点才能构造交通矩阵")
        if len(set(attraction_ids)) != len(attraction_ids):
            raise ValueError("景点 id 不允许重复")
        if len(attraction_ids) > self.MAX_ATTRACTIONS:
            raise ValueError(
                f"单次矩阵最多支持 {self.MAX_ATTRACTIONS} 个景点，"
                f"当前 {len(attraction_ids)} 个"
            )

        # 单景点没有「不同景点之间」的边，直接返回合法空矩阵，不必调用 Provider
        if len(attraction_ids) == 1:
            return RouteMatrix(
                city=normalized_city,
                mode=mode,
                attraction_ids=attraction_ids,
                entries=[],
            )

        entries = await self._provider.get_route_legs(
            city=normalized_city,
            attractions=list(attractions),
            mode=mode,
        )
        return RouteMatrix(
            city=normalized_city,
            mode=mode,
            attraction_ids=attraction_ids,
            entries=entries,
        )
