"""景点搜索业务层。

职责边界（这条线很重要，后面所有 Service 都照这个来）：
    Provider  只取数、只做字段转换
    Service   做参数归一化、筛选、打分、排序、组装响应
    Router    只做 HTTP 参数与状态码
"""

from collections.abc import Sequence
from functools import lru_cache

from backend.providers.base import AttractionProvider
from backend.providers.mock_provider import MockAttractionProvider
from backend.schemas.attraction import Attraction, AttractionSearchResponse


class AttractionService:
    """景点搜索。构造时注入 Provider，便于测试替换。"""

    # 打分权重先硬编码，第 7 阶段改成可配置（不同旅行风格用不同权重）
    INTEREST_WEIGHT = 1.0
    RATING_WEIGHT = 0.3

    def __init__(self, provider: AttractionProvider) -> None:
        self._provider = provider

    async def search(
        self,
        city: str,
        interests: Sequence[str] | None,
        limit: int,
    ) -> AttractionSearchResponse:
        """搜索并打分排序景点。

        排序优先级：兴趣命中数 → 评分 → id（最后一级保证结果稳定、可断言）。
        """
        normalized_city = city.strip()
        normalized_interests = self._normalize_interests(interests)

        # 多取一些候选，避免 Provider 的兜底排序把低评分但高匹配的条目截掉
        candidates = await self._provider.search_attractions(
            city=normalized_city,
            interests=normalized_interests or None,
            limit=max(limit * 2, limit),
        )

        ranked = sorted(
            candidates,
            key=lambda attraction: (
                -self._score(attraction, normalized_interests),
                -(attraction.rating or 0.0),
                attraction.id,
            ),
        )
        items = ranked[:limit]

        return AttractionSearchResponse(
            city=normalized_city,
            interests=normalized_interests,
            total=len(items),
            items=items,
        )

    @staticmethod
    def _normalize_interests(interests: Sequence[str] | None) -> list[str]:
        """去空白、转小写、去重，保持用户输入顺序（便于回显和测试断言）。"""
        seen: set[str] = set()
        normalized: list[str] = []
        for interest in interests or []:
            clean = interest.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized

    def _score(self, attraction: Attraction, interests: list[str]) -> float:
        """兴趣命中数为主，评分为辅。interests 为空时退化为纯评分排序。"""
        hits = sum(1 for interest in interests if attraction.matches_interest(interest))
        return hits * self.INTEREST_WEIGHT + (attraction.rating or 0.0) / 5 * self.RATING_WEIGHT


@lru_cache(maxsize=1)
def get_attraction_service() -> AttractionService:
    """FastAPI 依赖。

    第一阶段固定用 Mock；第 14 阶段改为按 settings.mock_mode 选择真实 Provider。
    测试里用 app.dependency_overrides[get_attraction_service] 替换。
    """
    return AttractionService(provider=MockAttractionProvider())
