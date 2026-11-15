"""网页地点排序的 Tool：按兴趣给候选池打分。

为什么单独一个文件：
    排序的提示词、解析、LLM 实现和工具编排都收在这里，
    和抽取工具（web_place_extract_tools）对称；候选池只来自
    extract_web_places，不再有百度候选池分支。

按「每个兴趣各排一次、各取前 N 个」返回 by_interest，
避免热门兴趣占满全部名额。

search_id 只交给 LLM，完整字段由代码从 backend.cache.pool_cache 合并回来，
不会被模型改写。
"""

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.cache.pool_cache import AttractionPoolCache
from backend.config import Settings, get_settings
from backend.schemas.ranking import AttractionRanking, RankingItem
from backend.schemas.web_place import ExtractedPlace
from backend.tools.tool_results import (
    dump_payload,
    error_result,
    handle_validation_error,
    unexpected_error,
)

logger = logging.getLogger(__name__)


def _loads_json(text: str) -> Any:
    """容忍 LLM 把 JSON 包在 ```json 代码块或前后废话里。"""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fenced is not None:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型输出里没有 JSON 对象")
    return json.loads(cleaned[start : end + 1])


# ---------------------------- 网页地点排序器 ----------------------------


class WebPlaceRanker(Protocol):
    """对轻量地点按兴趣排序。"""

    async def rank(
        self,
        *,
        city: str,
        interests: Sequence[str],
        places: Sequence[ExtractedPlace],
        top_k: int,
    ) -> AttractionRanking: ...


class _RawRankingItem(BaseModel):
    id: str = ""
    score: float = 0.0
    reason: str = ""


class _RawRanking(BaseModel):
    items: list[_RawRankingItem] = Field(default_factory=list)


def build_web_place_ranking_prompt(
    city: str,
    places: Sequence[ExtractedPlace],
    interests: Sequence[str],
    top_k: int,
) -> str:
    interest_text = "、".join(interests) if interests else "未指定（按综合相关性排序）"
    lines = [
        f"- {place.id} | {place.name} | 分类：{place.category.value} | "
        f"简介：{(place.description or '').strip()[:80] or '无'}"
        for place in places
    ]

    return f"""你是地点兴趣匹配排序器，只判断「地点与用户兴趣的相关程度」。

## 城市
{city}

## 用户兴趣
{interest_text}

## 候选地点（共 {len(places)} 个）
{chr(10).join(lines)}

## 排序规则
1. score 取 0~1，越匹配兴趣越高；不要考虑距离、门票、营业时间、天气。
2. id 必须原样取自候选列表，禁止编造或改写。
3. 每个地点最多出现一次；按 score 从高到低排列，最多输出 {top_k} 个。
4. reason 用一句中文，说明命中了哪些兴趣特征。

## 输出格式
只输出一个 JSON 对象，不要 markdown 代码块、不要解释文字：
{{"items": [{{"id": "地点id", "score": 0.95, "reason": "理由"}}]}}
"""


class LLMWebPlaceRanker:
    """LLM 排序轻量地点：只返回 id/score/reason。"""

    def __init__(self, runnable: Runnable, *, timeout_seconds: float = 30.0) -> None:
        self._runnable = runnable
        self._timeout = timeout_seconds

    async def rank(
        self,
        *,
        city: str,
        interests: Sequence[str],
        places: Sequence[ExtractedPlace],
        top_k: int,
    ) -> AttractionRanking:
        if not places or top_k <= 0:
            return AttractionRanking(strategy_used="llm")

        prompt = build_web_place_ranking_prompt(city, places, interests, top_k)
        try:
            raw = await asyncio.wait_for(
                self._runnable.ainvoke(prompt),
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("网页地点排序失败：%s", exc)
            raise RuntimeError(
                f"网页地点排序失败：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            items = self._merge(self._parse(raw), places, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("网页地点排序结果无法解析：%s", exc)
            raise RuntimeError(f"网页地点排序结果无法解析：{exc}") from exc

        if not items:
            raise RuntimeError("网页地点排序结果中没有可用 id")

        return AttractionRanking(
            ranked_ids=[item.id for item in items],
            items=items,
            strategy_used="llm",
        )

    @staticmethod
    def _parse(raw: Any) -> list[_RawRankingItem]:
        if isinstance(raw, _RawRanking):
            return raw.items
        if isinstance(raw, BaseModel):
            data = raw.model_dump()
        elif isinstance(raw, str):
            data = _loads_json(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError(f"不支持的 LLM 返回类型：{type(raw).__name__}")
        return _RawRanking.model_validate(data).items

    @staticmethod
    def _merge(
        raw_items: Sequence[_RawRankingItem],
        places: Sequence[ExtractedPlace],
        top_k: int,
    ) -> list[RankingItem]:
        valid_ids = {place.id for place in places}
        seen: set[str] = set()
        merged: list[RankingItem] = []
        for raw in raw_items:
            if raw.id not in valid_ids or raw.id in seen:
                continue
            seen.add(raw.id)
            merged.append(
                RankingItem(
                    id=raw.id,
                    score=max(0.0, min(1.0, raw.score)),
                    reason=raw.reason.strip() or "模型未给出理由",
                )
            )
        merged.sort(key=lambda item: -item.score)
        return merged[:top_k]


def _build_rank_runnable(settings: Settings) -> Runnable:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
        timeout=settings.attraction_ranking_timeout_seconds,
    )
    return llm.with_structured_output(_RawRanking, method="json_mode")


def create_web_place_ranker(
    *,
    settings: Settings | None = None,
) -> WebPlaceRanker:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError("缺少 OPENAI_API_KEY，无法对网页地点排序")
    return LLMWebPlaceRanker(
        _build_rank_runnable(settings),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )


# ---------------------------- 入参 Schema ----------------------------


class RankAttractionPoolInput(BaseModel):
    search_id: str = Field(
        ...,
        min_length=1,
        description="候选池 id（extract_web_places 返回）",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="用户兴趣，可用中文或英文：美食/自然/历史/culture 等",
    )
    top_k: int = Field(8, ge=1, le=20, description="返回前多少个景点")
    top_k_per_interest: int = Field(
        7,
        ge=1,
        le=15,
        description="每个兴趣各取前多少个，默认 7",
    )


# ---------------------------- Tool 实现 ----------------------------


def create_rank_attraction_pool_tool(
    cache: AttractionPoolCache,
    *,
    web_place_ranker: WebPlaceRanker | None = None,
    default_top_k_per_interest: int = 7,
) -> BaseTool:
    """构建 rank_attraction_pool：对网页地点候选池按兴趣排序。"""

    async def _run(
        search_id: str,
        interests: list[str] | None = None,
        top_k: int = 8,
        top_k_per_interest: int = default_top_k_per_interest,
    ) -> str:
        entry = cache.get_entry(search_id)
        if entry is None:
            return error_result(
                "search_id_not_found",
                "候选池不存在或已过期，请重新检索候选",
                retryable=True,
            )
        if entry.kind != "web_place":
            return error_result(
                "unsupported_pool_kind",
                f"不支持的候选池类型：{entry.kind}",
            )
        if web_place_ranker is None:
            return error_result(
                "web_place_ranker_missing",
                "没有配置网页地点排序器",
            )

        normalized_interests = [
            interest.strip() for interest in (interests or []) if interest.strip()
        ]
        city = entry.city
        places = [item for item in entry.items if isinstance(item, ExtractedPlace)]
        by_place = {place.id: place for place in places}
        by_interest: dict[str, list[dict[str, Any]]] = {}

        def _row(place: ExtractedPlace, ranked: Any) -> dict[str, Any]:
            return {
                "id": place.id,
                "name": place.name,
                "category": place.category.value,
                "description": place.description,
                "source_url": place.source_url,
                "score": ranked.score,
                "reason": ranked.reason,
            }

        items: list[dict[str, Any]] = []
        if not normalized_interests:
            try:
                ranking = await web_place_ranker.rank(
                    city=city,
                    interests=[],
                    places=places,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                return unexpected_error("rank_attraction_pool", exc)
            items = [
                _row(by_place[item.id], item)
                for item in ranking.items
                if item.id in by_place
            ]
        else:
            # 每个兴趣各自排一遍、各自截断，避免热门兴趣占满全部名额
            seen_ids: set[str] = set()
            for interest in normalized_interests:
                try:
                    interest_ranking = await web_place_ranker.rank(
                        city=city,
                        interests=[interest],
                        places=places,
                        top_k=top_k_per_interest,
                    )
                except Exception as exc:  # noqa: BLE001
                    return unexpected_error("rank_attraction_pool", exc)
                bucket: list[dict[str, Any]] = []
                for ranked in interest_ranking.items:
                    place = by_place.get(ranked.id)
                    if place is None:
                        continue
                    row = _row(place, ranked)
                    bucket.append(row)
                    # 同一地点可能同时命中多个兴趣：分组里各留一份，平铺只留首次
                    if place.id not in seen_ids:
                        seen_ids.add(place.id)
                        items.append(row)
                by_interest[interest] = bucket

        payload: dict[str, Any] = {
            "ok": True,
            "search_id": search_id,
            "city": city,
            "interests": normalized_interests,
            "kind": entry.kind,
            "strategy_used": "llm",
            "total": len(items),
            "items": items,
        }
        if by_interest:
            payload["by_interest"] = by_interest
            payload["top_k_per_interest"] = top_k_per_interest
        return dump_payload(payload)

    return StructuredTool.from_function(
        coroutine=_run,
        name="rank_attraction_pool",
        description=(
            "按用户兴趣对网页地点候选池排序（候选池来自 extract_web_places）。"
            "按每个兴趣分别排序，各返回前 7 个（见 by_interest）；"
            "不判断距离、营业时间、门票与分天安排。"
        ),
        args_schema=RankAttractionPoolInput,
        handle_validation_error=handle_validation_error,
    )


def create_web_place_rank_tool(
    cache: AttractionPoolCache,
    *,
    web_place_ranker: WebPlaceRanker | None = None,
) -> BaseTool:
    """registry 入口：未注入排序器时按配置创建默认实现。"""
    return create_rank_attraction_pool_tool(
        cache,
        web_place_ranker=web_place_ranker or create_web_place_ranker(),
    )
