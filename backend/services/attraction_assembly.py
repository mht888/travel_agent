"""把排序结果组装成 Attraction。

坐标只有分天排程前的交通矩阵需要，所以初选阶段只做一次地理编码：
成功就写入 GCJ-02 坐标，失败把 location 留空并记 warning，
不编造坐标，也不阻断排序结果返回。

职责边界：
    Geocoder    只负责「地址 -> 坐标」，不处理重试与并发；
    Assembler   负责并发限流、失败降级、必填字段补齐。
"""

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol

from backend.config import Settings, get_settings
from backend.schemas.attraction import (
    Attraction,
    default_duration_minutes,
)
from backend.schemas.common import DataStatus, GeoPoint, Provenance
from backend.schemas.ranking import RankingItem
from backend.tools.baidu_mcp_tools import (
    McpToolCaller,
    bd09_to_gcj02,
    get_baidu_mcp_client,
)

logger = logging.getLogger(__name__)


class Geocoder(Protocol):
    """地址解析为 GCJ-02 坐标；解析不出来返回 None。"""

    async def geocode(self, address: str) -> GeoPoint | None: ...


class BaiduMcpGeocoder:
    """复用百度 MCP 的 map_geocode，返回值统一转成 GCJ-02。"""

    def __init__(self, caller: McpToolCaller | None = None) -> None:
        self._caller = caller

    async def geocode(self, address: str) -> GeoPoint | None:
        caller = self._caller or get_baidu_mcp_client()
        payload = await caller.call_tool(
            "map_geocode", {"address": address.strip()}
        )
        location = payload.get("location") if isinstance(payload, dict) else None
        if not isinstance(location, dict):
            return None
        try:
            bd_lat = float(location["lat"])
            bd_lng = float(location["lng"])
        except (KeyError, TypeError, ValueError):
            return None
        lat, lng = bd09_to_gcj02(bd_lat, bd_lng)
        return GeoPoint(lat=round(lat, 6), lng=round(lng, 6))


class AttractionAssembler:
    """按排序结果补齐必填字段，组装成 Attraction。"""

    def __init__(self, geocoder: Geocoder, *, max_concurrency: int = 4) -> None:
        self._geocoder = geocoder
        self._max_concurrency = max(1, max_concurrency)

    async def __call__(
        self,
        *,
        city: str,
        items: Sequence[RankingItem],
    ) -> tuple[list[Attraction], list[str]]:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _assemble(
            item: RankingItem,
        ) -> tuple[Attraction | None, str | None]:
            name = item.name.strip()
            if not name:
                return None, f"排序结果缺少景点名称，已跳过组装（id={item.id}）"

            location: GeoPoint | None = None
            warning: str | None = None
            try:
                async with semaphore:
                    location = await self._geocoder.geocode(f"{city} {name}")
            except Exception as exc:  # noqa: BLE001 - 单条失败不能拖垮整批
                logger.warning("「%s」地理编码失败：%s", name, exc)
                warning = f"「{name}」地理编码失败，坐标留空：{type(exc).__name__}"
            else:
                if location is None:
                    warning = f"「{name}」未解析出坐标，坐标留空"

            return (
                Attraction(
                    id=item.id,
                    name=name,
                    city=city,
                    category=item.category,
                    description=item.description.strip() or None,
                    location=location,
                    estimated_duration_minutes=default_duration_minutes(
                        item.category
                    ),
                    provenance=Provenance(
                        source="bocha+llm+baidu_geocode"
                        if location is not None
                        else "bocha+llm",
                        fetched_at=datetime.now(timezone.utc),
                        status=DataStatus.ESTIMATED,
                    ),
                ),
                warning,
            )

        results = await asyncio.gather(
            *(_assemble(item) for item in items)
        )
        attractions = [attraction for attraction, _ in results if attraction]
        warnings = [warning for _, warning in results if warning]
        return attractions, warnings


def create_attraction_assembler(
    *,
    settings: Settings | None = None,
    geocoder: Geocoder | None = None,
) -> AttractionAssembler | None:
    """没有百度 AK 时返回 None：调用方只记 warning，不影响排序结果。"""
    settings = settings or get_settings()
    if geocoder is None:
        if not settings.baidu_mcp_enabled:
            return None
        if not (settings.baidu_maps_ak or "").strip():
            return None
        geocoder = BaiduMcpGeocoder()
    return AttractionAssembler(
        geocoder, max_concurrency=settings.baidu_mcp_max_concurrency
    )
