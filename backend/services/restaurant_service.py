"""餐厅搜索服务。

职责边界：
    - Provider 只负责取数（可能来自 Google Places / Tabelog / Mock）
    - 本服务负责参数校验、菜系过滤、排序和响应封装
    - 不负责路线、不负责排进哪一天——那是 ItineraryService / builder 的事

注意：这个服务目前**不暴露成 LLM Tool**。
餐厅是固定流程的一部分（每天一顿午餐），不需要模型自主决定是否查询。
"""

from collections.abc import Sequence

from backend.providers.base import RestaurantProvider
from backend.schemas.restaurant import RestaurantSearchResponse

MAX_LIMIT = 30
DEFAULT_LIMIT = 10


class RestaurantService:
    """把 RestaurantProvider 包装成业务可用的搜索服务。"""

    def __init__(self, provider: RestaurantProvider) -> None:
        self._provider = provider

    async def search(
        self,
        *,
        city: str,
        cuisines: Sequence[str] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> RestaurantSearchResponse:
        """按城市和菜系搜索餐厅。

        参数：
            city: 城市名，大小写不敏感；空字符串直接报错
            cuisines: 菜系白名单，None/空表示不过滤
            limit: 返回条数，1..MAX_LIMIT

        返回：
            RestaurantSearchResponse，items 已按「菜系命中数降序 → 评分降序 → id」排序

        异常：
            ValueError: 参数不合法（这是调用方 bug，不应该静默返回空）
        """
        city_key = city.strip()
        if not city_key:
            raise ValueError("city 不能为空")
        if limit < 1 or limit > MAX_LIMIT:
            raise ValueError(f"limit 必须在 1..{MAX_LIMIT} 之间，收到 {limit}")

        wanted = self._normalize_cuisines(cuisines)
        items = await self._provider.search_restaurants(
            city_key, wanted or None, limit=limit
        )

        # Provider 的排序不作为契约：不同数据源的默认排序差异很大，
        # 这里统一成「命中菜系越多越靠前，再按评分，最后按 id 保证稳定」。
        def sort_key(restaurant: tuple[int, object]) -> tuple[int, float, str]:
            index, item = restaurant  # type: ignore[misc]
            hits = len(set(wanted) & set(item.cuisines)) if wanted else 0  # type: ignore[attr-defined]
            return (-hits, -(item.rating or 0.0), item.id)  # type: ignore[attr-defined]

        ranked = [item for _, item in sorted(enumerate(items), key=sort_key)]

        return RestaurantSearchResponse(
            city=city_key,
            cuisines=wanted,
            total=len(ranked),
            items=ranked,
        )

    @staticmethod
    def _normalize_cuisines(cuisines: Sequence[str] | None) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for cuisine in cuisines or []:
            clean = cuisine.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized


def get_restaurant_service() -> RestaurantService:
    """组合根：切换真实数据源时只改这里。"""
    from backend.providers.mock_provider import MockRestaurantProvider

    return RestaurantService(MockRestaurantProvider())
