"""景点初选 Agent 的两个 Tool：候选池检索 + 兴趣排序。

为什么单独一个文件而不是塞进 travel_tools：
    travel_tools 是第 5~6 阶段的通用工具箱（搜索/详情/路线/天气），
    这里只服务「景点初选」这一条链路，缓存与排序器依赖都不同，混在一起会互相牵连。

两步走的原因：
    Agent 先搜到 20 个候选，再按兴趣排序。搜索与排序拆开，
    才能在同一个候选池上切换 rule / llm 两套排序做 A/B 对比，不用重复调百度。

search_id：
    候选池留在进程内缓存，Tool 只把 id 交给 LLM。
    这样 LLM 回吐的只有「哪些 id 排前面」，完整字段由代码合并，不会被改写。
"""

import json
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.providers.base import AttractionProvider
from backend.providers.baidu_provider import get_baidu_attraction_provider
from backend.schemas.attraction import Attraction
from backend.services.attraction_ranking import (
    AttractionRanker,
    create_attraction_ranker,
)

logger = logging.getLogger(__name__)

# 候选池默认保留 30 分钟：足够一次 Agent 循环，又不会长期占内存。
DEFAULT_POOL_TTL_SECONDS = 1800.0
_MAX_POOL_ENTRIES = 64


# ---------------------------- 候选池缓存 ----------------------------


class AttractionPoolCache:
    """进程内候选池缓存。

    单实例部署够用；多实例部署时 search_id 会落在不同进程，
    届时应换成 Redis 之类的共享存储，这里刻意不提前引入。
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_POOL_TTL_SECONDS,
        max_entries: int = _MAX_POOL_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, tuple[float, str, list[Attraction]]] = {}

    def save(self, city: str, attractions: Sequence[Attraction]) -> str:
        self._purge()
        while len(self._entries) >= self._max_entries:
            # dict 保持插入顺序，最早写入的条目优先淘汰
            self._entries.pop(next(iter(self._entries)), None)
        search_id = uuid.uuid4().hex[:12]
        self._entries[search_id] = (
            time.monotonic() + self._ttl,
            city,
            list(attractions),
        )
        return search_id

    def get(self, search_id: str) -> tuple[str, list[Attraction]] | None:
        self._purge()
        entry = self._entries.get(search_id)
        if entry is None:
            return None
        _, city, attractions = entry
        return city, list(attractions)

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if entry[0] <= now]
        for key in expired:
            self._entries.pop(key, None)


# ---------------------------- 序列化与错误 ----------------------------


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(
    error_code: str,
    message: str,
    *,
    retryable: bool = False,
    **extra: Any,
) -> str:
    return _dump(
        {
            "ok": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            **extra,
        }
    )


def _handle_validation_error(error: Exception) -> str:
    return _error("invalid_arguments", f"参数校验失败：{error}")


def _unexpected_error(tool_name: str, error: Exception) -> str:
    logger.exception("Tool %s 执行失败", tool_name)
    return _error(
        "tool_internal_error",
        f"{tool_name} 执行失败：{type(error).__name__}",
        retryable=True,
    )


def _compact(attraction: Attraction) -> dict[str, Any]:
    """给 LLM 看的精简候选：保留判断兴趣所需字段，不塞开放时间等大对象。"""
    return {
        "id": attraction.id,
        "name": attraction.name,
        "category": attraction.category.value,
        "tags": attraction.tags[:8],
        "rating": attraction.rating,
        "description": (attraction.description or "")[:120] or None,
        "location": attraction.location.model_dump(mode="json"),
    }


# ---------------------------- 入参 Schema ----------------------------


class SearchAttractionPoolInput(BaseModel):
    city: str = Field(..., min_length=1, description="城市中文名，如 南宁")
    keywords: list[str] | None = Field(
        None,
        description=(
            "补充检索词（中文），如 公园 / 博物馆 / 美食；"
            "不传时使用默认的「景点」，一般不需要传"
        ),
    )
    limit: int = Field(20, ge=5, le=40, description="候选池大小，默认 20")


class RankAttractionPoolInput(BaseModel):
    search_id: str = Field(
        ..., min_length=1, description="search_attraction_pool 返回的候选池 id"
    )
    interests: list[str] = Field(
        default_factory=list,
        description="用户兴趣，可用中文或英文：美食/自然/历史/culture 等",
    )
    top_k: int = Field(8, ge=1, le=20, description="返回前多少个景点")


# ---------------------------- Tool 实现 ----------------------------


def _build_search_pool(
    provider: AttractionProvider,
    cache: AttractionPoolCache,
) -> BaseTool:
    async def _run(
        city: str,
        keywords: list[str] | None = None,
        limit: int = 20,
    ) -> str:
        try:
            normal_city = city.strip()
            if not normal_city:
                return _error("invalid_argument", "city 不能为空白")

            # 百度实现支持多关键词；其他 Provider 退化为按 interests 过滤
            search_by_keywords = getattr(provider, "search_by_keywords", None)
            if callable(search_by_keywords):
                attractions = await search_by_keywords(
                    city=normal_city,
                    keywords=keywords,
                    limit=limit,
                )
            else:
                attractions = await provider.search_attractions(
                    city=normal_city,
                    interests=keywords,
                    limit=limit,
                )
        except ValueError as exc:
            return _error("invalid_argument", str(exc))
        except Exception as exc:  # noqa: BLE001 - 外部数据源失败必须转成结果
            return _unexpected_error("search_attraction_pool", exc)

        if not attractions:
            return _dump(
                {
                    "ok": True,
                    "search_id": None,
                    "city": normal_city,
                    "total": 0,
                    "items": [],
                    "hint": "没有检索到候选景点：可换一个城市名，或补充更宽泛的关键词",
                }
            )

        search_id = cache.save(normal_city, attractions)
        return _dump(
            {
                "ok": True,
                "search_id": search_id,
                "city": normal_city,
                "total": len(attractions),
                "items": [_compact(attraction) for attraction in attractions],
                "hint": "下一步调用 rank_attraction_pool，并把 search_id 和用户兴趣传进去",
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="search_attraction_pool",
        description=(
            "检索城市的景点候选池（默认 20 个），返回 search_id 与精简候选列表。"
            "只负责取候选，不按兴趣排序；下一步必须调用 rank_attraction_pool。"
        ),
        args_schema=SearchAttractionPoolInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_rank_pool(
    ranker: AttractionRanker,
    cache: AttractionPoolCache,
) -> BaseTool:
    async def _run(
        search_id: str,
        interests: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        cached = cache.get(search_id)
        if cached is None:
            return _error(
                "search_id_not_found",
                "候选池不存在或已过期，请重新调用 search_attraction_pool",
                retryable=True,
            )

        city, attractions = cached
        normalized_interests = [
            interest.strip() for interest in (interests or []) if interest.strip()
        ]
        try:
            ranking = await ranker.rank(
                city=city,
                attractions=attractions,
                interests=normalized_interests,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("rank_attraction_pool", exc)

        by_id = {attraction.id: attraction for attraction in attractions}
        items: list[dict[str, Any]] = []
        for item in ranking.items:
            attraction = by_id.get(item.id)
            if attraction is None:
                continue
            items.append(
                {
                    "id": attraction.id,
                    "name": attraction.name,
                    "category": attraction.category.value,
                    "tags": attraction.tags[:8],
                    "rating": attraction.rating,
                    "score": item.score,
                    "reason": item.reason,
                }
            )

        return _dump(
            {
                "ok": True,
                "search_id": search_id,
                "city": city,
                "interests": normalized_interests,
                "strategy_used": ranking.strategy_used,
                "total": len(items),
                "items": items,
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="rank_attraction_pool",
        description=(
            "对 search_attraction_pool 的候选池按用户兴趣排序，返回带 score 与理由的景点清单。"
            "不判断距离、营业时间、门票与分天安排。"
        ),
        args_schema=RankAttractionPoolInput,
        handle_validation_error=_handle_validation_error,
    )


# ---------------------------- 组装 ----------------------------


def create_attraction_selection_tools(
    *,
    provider: AttractionProvider | None = None,
    ranker: AttractionRanker | None = None,
    pool_cache: AttractionPoolCache | None = None,
) -> list[BaseTool]:
    """组装景点初选的两个 Tool，共享同一个候选池缓存。"""
    provider = provider or get_baidu_attraction_provider()
    ranker = ranker or create_attraction_ranker()
    cache = pool_cache or AttractionPoolCache()

    return [
        _build_search_pool(provider, cache),
        _build_rank_pool(ranker, cache),
    ]
