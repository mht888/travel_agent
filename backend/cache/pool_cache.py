"""景点初选链路用到的两个进程内缓存。

为什么单独放 cache 层：
    候选池由编排层（graph）创建并注入工具，缓存放 tools 会让 graph 反向依赖 tools；
    这里只依赖 stdlib，tools 与 graph 都向下依赖它。

多实例部署时 search_id / web_search_id 会落在不同进程，
届时应换成 Redis 之类的共享存储，这里刻意不提前引入。
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# 默认保留 30 分钟：足够一次 Agent 循环，又不会长期占内存。
DEFAULT_POOL_TTL_SECONDS = 1800.0
_MAX_POOL_ENTRIES = 64

PoolKind = Literal["attraction", "web_place", "selected_attraction"]


@dataclass(frozen=True)
class PoolEntry:
    """缓存条目：kind 区分网页地点候选池 / 初选入选结果等用途。"""

    kind: PoolKind
    city: str
    items: list[Any]


@dataclass(frozen=True)
class SelectedAttraction:
    """初选入选景点：Attraction 的 JSON 快照 + 入选时的排序理由。

    只依赖 stdlib，避免 cache 层反向依赖 schemas；
    调用方按 attraction["id"] 还原 Attraction。
    """

    attraction: dict[str, Any]
    score: float
    reason: str
    interests: list[str] = field(default_factory=list)


class AttractionPoolCache:
    """进程内候选池缓存：Tool 只把 search_id 交给 LLM，完整字段留在进程里。"""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_POOL_TTL_SECONDS,
        max_entries: int = _MAX_POOL_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, tuple[float, PoolEntry]] = {}

    def save(
        self,
        city: str,
        items: Sequence[Any],
        *,
        kind: PoolKind = "attraction",
    ) -> str:
        self._purge()
        while len(self._entries) >= self._max_entries:
            # dict 保持插入顺序，最早写入的条目优先淘汰
            self._entries.pop(next(iter(self._entries)), None)
        search_id = uuid.uuid4().hex[:12]
        self._entries[search_id] = (
            time.monotonic() + self._ttl,
            PoolEntry(kind=kind, city=city, items=list(items)),
        )
        return search_id

    def save_selection(
        self,
        city: str,
        items: Sequence[SelectedAttraction],
    ) -> str:
        """保存初选入选景点，返回 selection_id；过期时间与候选池一致。"""
        return self.save(city, items, kind="selected_attraction")

    def load_selection(self, selection_id: str) -> list[SelectedAttraction] | None:
        """按 selection_id 取回入选景点；不存在、已过期或类型不符返回 None。"""
        entry = self.get_entry(selection_id)
        if entry is None or entry.kind != "selected_attraction":
            return None
        return [item for item in entry.items if isinstance(item, SelectedAttraction)]

    def get_entry(self, search_id: str) -> PoolEntry | None:
        self._purge()
        entry = self._entries.get(search_id)
        return entry[1] if entry is not None else None

    def get(self, search_id: str) -> tuple[str, list[Any]] | None:
        """兼容旧调用：返回 (city, items)。"""
        entry = self.get_entry(search_id)
        if entry is None:
            return None
        return entry.city, list(entry.items)

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if entry[0] <= now]
        for key in expired:
            self._entries.pop(key, None)


class WebSearchResultCache:
    """博查网页结果缓存。

    抽取工具凭 web_search_id 取原文，避免把整段网页摘要再塞回 Agent 上下文。
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_POOL_TTL_SECONDS,
        max_entries: int = _MAX_POOL_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, tuple[float, str, list[dict[str, Any]]]] = {}

    def save(self, query: str, results: Sequence[dict[str, Any]]) -> str:
        self._purge()
        while len(self._entries) >= self._max_entries:
            self._entries.pop(next(iter(self._entries)), None)
        web_search_id = uuid.uuid4().hex[:12]
        self._entries[web_search_id] = (
            time.monotonic() + self._ttl,
            query,
            [dict(item) for item in results],
        )
        return web_search_id

    def get(self, web_search_id: str) -> tuple[str, list[dict[str, Any]]] | None:
        self._purge()
        entry = self._entries.get(web_search_id)
        if entry is None:
            return None
        _, query, results = entry
        return query, [dict(item) for item in results]

    def all(self) -> list[tuple[str, list[dict[str, Any]]]]:
        self._purge()
        return [
            (query, [dict(item) for item in results])
            for _, query, results in self._entries.values()
        ]

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if entry[0] <= now]
        for key in expired:
            self._entries.pop(key, None)
