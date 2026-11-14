"""景点兴趣排序：两套可切换实现，用于 A/B 对比。

职责边界：
    只回答「这些候选里，哪些更符合用户兴趣」。
    距离、营业时间、门票、预算、分天都不在这里判断——那是后续排程 Agent 的事。

两套实现：
    RuleBasedAttractionRanker  纯 Python 关键词命中 + 评分，零依赖、可解释、可复现
    LLMAttractionRanker        让 LLM 直接读候选名称/标签/简介给出兴趣分

为什么都保留：
    用配置 ATTRACTION_RANKING_STRATEGY 在 rule / llm 之间切换，
    同一条链路可以跑出两组结果做效果与延迟对比，而不是二选一删掉另一个。
"""

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.schemas.attraction import Attraction
from backend.schemas.ranking import AttractionRanking, RankingItem

logger = logging.getLogger(__name__)

# 中英文兴趣别名：只覆盖常见兴趣，保证规则基线可用，不追求完整标签体系。
INTEREST_ALIASES: dict[str, tuple[str, ...]] = {
    "food": ("美食", "小吃", "餐厅", "夜市", "料理", "food", "cuisine"),
    "culture": ("文化", "历史", "博物馆", "寺", "庙", "古迹", "遗址", "故居", "culture", "history"),
    "nature": ("自然", "公园", "山", "湖", "森林", "湿地", "植物园", "nature", "park"),
    "shopping": ("购物", "商场", "商业街", "步行街", "百货", "免税", "shopping", "mall"),
    "nightlife": ("夜景", "酒吧", "夜市", "灯光秀", "nightlife", "bar"),
    "entertainment": ("乐园", "游乐", "主题公园", "演出", "剧院", "水族馆", "动物园", "entertainment"),
    "landmark": ("地标", "塔", "广场", "大桥", "观景", "landmark", "tower"),
    "art": ("艺术", "美术馆", "画廊", "展览", "art", "gallery"),
    "wellness": ("温泉", "疗养", "spa", "wellness"),
    "outdoor_activity": ("户外", "徒步", "登山", "骑行", "漂流", "outdoor", "hiking"),
}


class AttractionRanker(Protocol):
    """排序器接口。rule / llm 两套实现都满足它，上层只依赖这个协议。"""

    async def rank(
        self,
        *,
        city: str,
        attractions: Sequence[Attraction],
        interests: Sequence[str],
        top_k: int,
    ) -> AttractionRanking: ...


def _normalize_interests(interests: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for interest in interests:
        clean = interest.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def _haystack(attraction: Attraction) -> str:
    """名称 + 分类 + 标签 + 简介拼成检索文本，统一小写。"""
    return " ".join(
        [
            attraction.name,
            attraction.category.value,
            " ".join(attraction.tags),
            attraction.description or "",
        ]
    ).casefold()


def _interest_terms(interest: str) -> tuple[str, ...]:
    """把兴趣扩展成同义词组。

    支持用户直接输入中文（如「美食」）：先在英文 canonical key 里找，
    找不到再反查哪个组的别名包含它，保证中英文兴趣都能命中同一组关键词。
    """
    aliases = INTEREST_ALIASES.get(interest)
    if aliases is None:
        aliases = next(
            (values for values in INTEREST_ALIASES.values() if interest in values),
            (),
        )
    return (interest, *aliases)


class RuleBasedAttractionRanker:
    """关键词命中数为主、评分为辅的确定性排序。

    评分公式固定为 0~1：
        interest_score = 命中兴趣数 / 兴趣总数
        score = 0.7 * interest_score + 0.3 * rating / 5
    没有兴趣时退化为纯评分排序。
    """

    INTEREST_WEIGHT = 0.7
    RATING_WEIGHT = 0.3

    async def rank(
        self,
        *,
        city: str,
        attractions: Sequence[Attraction],
        interests: Sequence[str],
        top_k: int,
    ) -> AttractionRanking:
        normalized = _normalize_interests(interests)
        scored = [
            (attraction, self._score(attraction, normalized))
            for attraction in attractions
        ]
        scored.sort(
            key=lambda pair: (
                -pair[1],
                -(pair[0].rating or 0.0),
                pair[0].id,
            )
        )

        items = [
            RankingItem(
                id=attraction.id,
                score=score,
                reason=self._reason(attraction, normalized),
            )
            for attraction, score in scored[: max(top_k, 0)]
        ]
        return AttractionRanking(
            ranked_ids=[item.id for item in items],
            items=items,
            strategy_used="rule",
        )

    def _score(self, attraction: Attraction, interests: list[str]) -> float:
        if not interests:
            return round((attraction.rating or 0.0) / 5, 4)
        hits = self._hits(attraction, interests)
        value = (
            self.INTEREST_WEIGHT * len(hits) / len(interests)
            + self.RATING_WEIGHT * (attraction.rating or 0.0) / 5
        )
        return round(min(1.0, max(0.0, value)), 4)

    def _hits(self, attraction: Attraction, interests: list[str]) -> list[str]:
        text = _haystack(attraction)
        hits: list[str] = []
        for interest in interests:
            # 先看分类与标签是否精确命中，再用别名做模糊匹配
            if interest == attraction.category.value or interest in attraction.tags:
                hits.append(interest)
                continue
            aliases = _interest_terms(interest)
            if any(alias.casefold() in text for alias in aliases):
                hits.append(interest)
        return hits

    def _reason(self, attraction: Attraction, interests: list[str]) -> str:
        if not interests:
            return "未指定兴趣，按评分排序"
        hits = self._hits(attraction, interests)
        if not hits:
            return "未命中兴趣，按评分排序"
        return f"命中兴趣：{'、'.join(hits)}"


class _RawRankingItem(BaseModel):
    """LLM 原始输出条目。score 不做边界校验，交给代码钳制，避免整批报废。"""

    id: str = ""
    score: float = 0.0
    reason: str = ""


class _RawRanking(BaseModel):
    items: list[_RawRankingItem] = Field(default_factory=list)


def build_ranking_prompt(
    city: str,
    attractions: Sequence[Attraction],
    interests: Sequence[str],
    top_k: int,
) -> str:
    """把候选与兴趣拼成排序提示。

    TODO 留给后续迭代的点：兴趣权重、多人偏好冲突、负面偏好（不要什么）。
    当前只要求「按兴趣相关性打分」，不掺杂距离/门票/营业时间。
    """
    interest_text = "、".join(interests) if interests else "未指定（按综合口碑排序）"
    lines: list[str] = []
    for attraction in attractions:
        tags = "、".join(attraction.tags) or "无"
        rating = (
            f"{attraction.rating:.1f}"
            if attraction.rating is not None
            else "未知"
        )
        description = (attraction.description or "").strip()[:80] or "无"
        lines.append(
            f"- {attraction.id} | {attraction.name} | 分类：{attraction.category.value} | "
            f"标签：{tags} | 评分：{rating} | 简介：{description}"
        )

    return f"""你是景点兴趣匹配排序器，只判断「景点与用户兴趣的相关程度」。

## 城市
{city}

## 用户兴趣
{interest_text}

## 候选景点（共 {len(attractions)} 个）
{chr(10).join(lines)}

## 排序规则
1. score 取 0~1，越匹配兴趣越高；不要考虑距离、门票、营业时间、天气。
2. id 必须原样取自候选列表，禁止编造或改写。
3. 每个景点最多出现一次；按 score 从高到低排列，最多输出 {top_k} 个。
4. reason 用一句中文，说明命中了哪些兴趣特征。

## 输出格式
只输出一个 JSON 对象，不要 markdown 代码块、不要解释文字：
{{"items": [{{"id": "景点id", "score": 0.95, "reason": "理由"}}]}}
"""


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


class LLMAttractionRanker:
    """LLM 排序。任何失败（超时、非法 JSON、未知 id 全被丢弃）都回退规则排序。"""

    def __init__(
        self,
        runnable: Runnable,
        fallback: RuleBasedAttractionRanker,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._runnable = runnable
        self._fallback = fallback
        self._timeout = timeout_seconds

    async def rank(
        self,
        *,
        city: str,
        attractions: Sequence[Attraction],
        interests: Sequence[str],
        top_k: int,
    ) -> AttractionRanking:
        normalized = _normalize_interests(interests)
        if not attractions or top_k <= 0:
            return AttractionRanking(strategy_used="llm")

        prompt = build_ranking_prompt(city, attractions, normalized, top_k)
        try:
            raw = await asyncio.wait_for(
                self._runnable.ainvoke(prompt),
                timeout=self._timeout,
            )
            items = self._merge(self._parse(raw), attractions, top_k)
        except Exception as exc:  # noqa: BLE001 - LLM 失败必须转成可运行结果
            logger.warning("LLM 排序失败，回退规则排序：%s", exc)
            return await self._fallback_result(
                city=city,
                attractions=attractions,
                interests=normalized,
                top_k=top_k,
            )

        if not items:
            logger.warning("LLM 排序结果中没有可用 id，回退规则排序")
            return await self._fallback_result(
                city=city,
                attractions=attractions,
                interests=normalized,
                top_k=top_k,
            )

        items = await self._backfill(
            items=items,
            city=city,
            attractions=attractions,
            interests=normalized,
            top_k=top_k,
        )
        return AttractionRanking(
            ranked_ids=[item.id for item in items],
            items=items,
            strategy_used="llm",
        )

    async def _fallback_result(
        self,
        *,
        city: str,
        attractions: Sequence[Attraction],
        interests: list[str],
        top_k: int,
    ) -> AttractionRanking:
        result = await self._fallback.rank(
            city=city,
            attractions=attractions,
            interests=interests,
            top_k=top_k,
        )
        return AttractionRanking(
            ranked_ids=result.ranked_ids,
            items=result.items,
            strategy_used="rule_fallback",
        )

    def _parse(self, raw: Any) -> list[_RawRankingItem]:
        """兼容结构化输出模型、dict、字符串三种返回。"""
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
        attractions: Sequence[Attraction],
        top_k: int,
    ) -> list[RankingItem]:
        """只保留合法 id，钳制分数；未知 id 直接丢弃而不是让整批失败。"""
        valid_ids = {attraction.id for attraction in attractions}
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
        # 稳定排序：分数相同时保留模型给出的先后顺序
        merged.sort(key=lambda item: -item.score)
        return merged[:top_k]

    async def _backfill(
        self,
        *,
        items: list[RankingItem],
        city: str,
        attractions: Sequence[Attraction],
        interests: list[str],
        top_k: int,
    ) -> list[RankingItem]:
        """LLM 只给了一部分时，用规则结果补足 top_k，保证下游数量稳定。"""
        if len(items) >= top_k:
            return items

        rule_result = await self._fallback.rank(
            city=city,
            attractions=attractions,
            interests=interests,
            top_k=top_k,
        )
        existing = {item.id for item in items}
        for rule_item in rule_result.items:
            if len(items) >= top_k:
                break
            if rule_item.id in existing:
                continue
            existing.add(rule_item.id)
            items.append(rule_item)
        return items


def _build_llm_runnable(settings: Settings) -> Runnable:
    """结构化输出 Runnable。

    与 Planner 一致使用 json_mode：部分 OpenAI 兼容端点在 thinking 模式下
    不支持 json_schema / function_calling，json_mode 兼容性最好。
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
        timeout=settings.attraction_ranking_timeout_seconds,
    )
    return llm.with_structured_output(_RawRanking, method="json_mode")


def create_attraction_ranker(
    strategy: str | None = None,
    *,
    settings: Settings | None = None,
) -> AttractionRanker:
    """按配置组装排序器。缺少 API Key 时自动退回规则实现。"""
    settings = settings or get_settings()
    rule = RuleBasedAttractionRanker()

    chosen = (strategy or settings.attraction_ranking_strategy).strip().lower()
    if chosen != "llm":
        return rule

    if not settings.openai_api_key:
        logger.warning(
            "ATTRACTION_RANKING_STRATEGY=llm 但缺少 OPENAI_API_KEY，回退规则排序"
        )
        return rule

    return LLMAttractionRanker(
        _build_llm_runnable(settings),
        rule,
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )
