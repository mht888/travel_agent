"""网页地点抽取的 Tool：从博查网页结果里抽地点，供景点初选使用。

为什么单独一个文件：
    抽取有自己的提示词、解析、别名归并和跨轮会话（补检索时累计去重），
    和百度候选池 / 排序不是一个职责；拆开后排序链路不必知道抽取细节。

数据流：
    mcp_web_search 把网页原文缓存成 web_search_id
    → 本工具凭 id 取原文、按兴趣抽取地点
    → Python 侧统计每个兴趣的数量，不足 min_places_per_interest 时返回
      insufficient_places，Agent 用更大的 count 补检索后再调用
    → 成功后写入候选池，返回 search_id 给 rank_attraction_pool

会话为什么在闭包：
    一次 run 一个工具实例，抽取会话随闭包存活；
    已抽到的地点、已消费的网页都保留，补检索不会从头再来。
"""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.cache.pool_cache import AttractionPoolCache, WebSearchResultCache
from backend.config import Settings, get_settings
from backend.schemas.attraction import AttractionCategory
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

# 每个兴趣至少要抽到多少个地点；不足时由 Agent 补检索。
DEFAULT_MIN_PLACES_PER_INTEREST = 10
# 最多抽取几轮（首轮 + 补检索）。博查 count 上限 50，4 轮足够跑到上限。
DEFAULT_MAX_EXTRACT_ROUNDS = 4

_CATEGORY_VALUES = {item.value for item in AttractionCategory}
_GENERIC_WORDS = ("攻略", "榜单", "排名", "盘点", "推荐", "门票", "交通")


def normalize_name(name: str) -> str:
    """名称归一化：去空白、转小写，用于跨来源去重。"""
    return re.sub(r"\s+", "", name).casefold()


def make_place_id(city: str, name: str) -> str:
    """稳定 id：同一城市同一名称永远得到同一个值。"""
    digest = hashlib.sha1(
        f"{city}:{normalize_name(name)}".encode("utf-8")
    ).hexdigest()[:12]
    return f"web:{digest}"


def _coerce_category(value: str) -> AttractionCategory:
    clean = (value or "").strip().lower()
    if clean in _CATEGORY_VALUES:
        return AttractionCategory(clean)
    return AttractionCategory.OTHER


def _looks_generic(name: str) -> bool:
    return any(word in name for word in _GENERIC_WORDS)


# ---------------------------- 协议 ----------------------------


class WebPlaceExtractor(Protocol):
    """从一组网页结果里抽出地点。"""

    async def extract(
        self,
        *,
        city: str,
        interest: str,
        results: Sequence[dict[str, Any]],
        limit: int,
    ) -> list[ExtractedPlace]: ...


class PlaceAliasResolver(Protocol):
    """判断哪些地点是同一个实体，返回 alias_id -> canonical_id。

    未合并的地点不出现在返回值里；实现方必须保证 id 只来自入参。
    """

    async def resolve(
        self,
        *,
        city: str,
        places: Sequence[ExtractedPlace],
    ) -> dict[str, str]: ...


# ---------------------------- LLM 输出结构 ----------------------------


class _RawPlace(BaseModel):
    name: str = ""
    description: str = ""
    category: str = ""
    source_url: str = ""


class _RawPlaces(BaseModel):
    items: list[_RawPlace] = Field(default_factory=list)


class _RawMergeGroup(BaseModel):
    canonical: str = ""
    aliases: list[str] = Field(default_factory=list)
    reason: str = ""


class _RawMergeGroups(BaseModel):
    merge: list[_RawMergeGroup] = Field(default_factory=list)


# ---------------------------- 抽取 ----------------------------


def build_extract_prompt(
    city: str,
    interest: str,
    results: Sequence[dict[str, Any]],
    limit: int,
) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()[:200]
        url = str(item.get("url") or "").strip()
        lines.append(f"[{index}] {title}\n    {snippet}\n    URL: {url}")

    return f"""你是地点抽取器。从网页摘要里找出「真实、可到访的地点」。

## 城市
{city}

## 兴趣
{interest}

## 网页结果（共 {len(results)} 条）
{chr(10).join(lines)}

## 抽取规则
1. 只输出具体地点名（景点、街区、市场、公园、餐厅、博物馆等），不要输出攻略标题。
2. 标题或摘要里只是榜单、排名、门票信息的，不输出。
3. 不确定是不是真实地点时不要输出，禁止编造。
4. 同一地点只输出一次；最多输出 {limit} 个。
5. category 只能取以下值之一：
   culture / food / shopping / nature / outdoor_activity / nightlife /
   entertainment / landmark / wellness / art / other
6. description 用一句中文概括摘要里与该地点有关的信息，不要照抄整段。

## 输出格式
只输出一个 JSON 对象，不要 markdown 代码块、不要解释文字：
{{"items": [{{"name": "地点名", "description": "一句介绍", "category": "food", "source_url": "https://..."}}]}}
"""


class LLMWebPlaceExtractor:
    """LLM 抽取：输出非法条目直接丢弃，不让整批失败。"""

    def __init__(self, runnable: Runnable, *, timeout_seconds: float = 30.0) -> None:
        self._runnable = runnable
        self._timeout = timeout_seconds

    async def extract(
        self,
        *,
        city: str,
        interest: str,
        results: Sequence[dict[str, Any]],
        limit: int,
    ) -> list[ExtractedPlace]:
        if not results or limit <= 0:
            return []

        prompt = build_extract_prompt(city, interest, results, limit)
        try:
            raw = await asyncio.wait_for(
                self._runnable.ainvoke(prompt),
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("网页地点抽取失败：%s", exc)
            raise RuntimeError(
                f"网页地点抽取失败：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            raw_places = self._parse(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("网页地点抽取结果无法解析：%s", exc)
            raise RuntimeError(f"网页地点抽取结果无法解析：{exc}") from exc

        return self._dedupe(city, raw_places, limit)

    @staticmethod
    def _parse(raw: Any) -> list[_RawPlace]:
        if isinstance(raw, _RawPlaces):
            return raw.items
        if isinstance(raw, BaseModel):
            data = raw.model_dump()
        elif isinstance(raw, str):
            data = _loads_json(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError(f"不支持的 LLM 返回类型：{type(raw).__name__}")
        return _RawPlaces.model_validate(data).items

    @staticmethod
    def _dedupe(
        city: str,
        raw_places: Sequence[_RawPlace],
        limit: int,
    ) -> list[ExtractedPlace]:
        seen: set[str] = set()
        places: list[ExtractedPlace] = []
        for raw in raw_places:
            name = raw.name.strip()
            if not name or _looks_generic(name):
                continue
            key = normalize_name(name)
            if key in seen:
                continue
            seen.add(key)
            places.append(
                ExtractedPlace(
                    id=make_place_id(city, name),
                    name=name,
                    description=raw.description.strip()[:160],
                    city=city,
                    category=_coerce_category(raw.category),
                    source_url=raw.source_url.strip() or None,
                )
            )
            if len(places) >= limit:
                break
        return places


# ---------------------------- 别名归并 ----------------------------


def build_alias_prompt(city: str, places: Sequence[ExtractedPlace]) -> str:
    lines = [
        f"- {place.id} | {place.name} | 分类：{place.category.value} | "
        f"简介：{(place.description or '').strip()[:60] or '无'}"
        for place in places
    ]

    return f"""你是地点实体归并器，只判断候选地点里哪些是同一个真实地点。

## 城市
{city}

## 候选地点（共 {len(places)} 个）
{chr(10).join(lines)}

## 归并规则
1. 只合并同一实体的不同称呼：简称/全称/别名/中英文写法/错别字，或多带了「景区、公园、步行街、夜市、美食街」等后缀。
2. 分店、同一品牌的不同门店、相邻但独立的场所、同名但实际不同的地点，不要合并。
3. 不确定时不要合并。
4. canonical 与 aliases 必须原样取自候选 id，禁止编造；canonical 选信息更完整的一条，aliases 不能包含 canonical 自己。
5. 只输出需要合并的组；未列出的地点视为独立。
6. reason 用一句中文说明为什么是同一地点。

## 输出格式
只输出一个 JSON 对象，不要 markdown 代码块、不要解释文字：
{{"merge": [{{"canonical": "地点id", "aliases": ["地点id"], "reason": "理由"}}]}}
"""


class LLMPlaceAliasResolver:
    """LLM 归并同义地点：只接受入参 id，非法分组直接丢弃，失败退回恒等映射。"""

    def __init__(self, runnable: Runnable, *, timeout_seconds: float = 30.0) -> None:
        self._runnable = runnable
        self._timeout = timeout_seconds

    async def resolve(
        self,
        *,
        city: str,
        places: Sequence[ExtractedPlace],
    ) -> dict[str, str]:
        if len(places) < 2:
            return {}

        prompt = build_alias_prompt(city, places)
        try:
            raw = await asyncio.wait_for(
                self._runnable.ainvoke(prompt),
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            # 归并是增强项，失败不能阻断初选
            logger.warning("地点别名归并失败：%s", exc)
            return {}

        try:
            groups = self._parse(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("地点别名归并结果无法解析：%s", exc)
            return {}

        return self._merge(groups, places)

    @staticmethod
    def _parse(raw: Any) -> list[_RawMergeGroup]:
        if isinstance(raw, _RawMergeGroups):
            return raw.merge
        if isinstance(raw, BaseModel):
            data = raw.model_dump()
        elif isinstance(raw, str):
            data = _loads_json(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError(f"不支持的 LLM 返回类型：{type(raw).__name__}")
        return _RawMergeGroups.model_validate(data).merge

    @staticmethod
    def _merge(
        groups: Sequence[_RawMergeGroup],
        places: Sequence[ExtractedPlace],
    ) -> dict[str, str]:
        valid_ids = {place.id for place in places}
        # canonical 不能同时是别人的 alias，否则会出现链式/环；这类组直接丢弃
        all_aliases = {
            alias.strip() for group in groups for alias in group.aliases
        }

        mapping: dict[str, str] = {}
        for group in groups:
            canonical = group.canonical.strip()
            if canonical not in valid_ids or canonical in all_aliases:
                continue
            for alias in group.aliases:
                clean_alias = alias.strip()
                if (
                    clean_alias not in valid_ids
                    or clean_alias == canonical
                    or clean_alias in mapping
                ):
                    continue
                mapping[clean_alias] = canonical
        return mapping


# ---------------------------- 抽取会话 ----------------------------


@dataclass
class _ExtractionSession:
    """一次初选内、跨多次抽取调用的上下文。

    抽取工具会被 Agent 反复调用（每次先补检索再抽）：
    已抽到的地点、已消费的网页结果都要保留，否则重试等于从头再来。
    放在工具闭包里而不是全局，一次 run 一个会话，互不污染。
    """

    city: str = ""
    rounds: int = 0
    by_interest: dict[str, dict[str, ExtractedPlace]] = field(
        default_factory=dict
    )
    processed: dict[str, set[tuple[str, str]]] = field(default_factory=dict)

    def reset(self, city: str) -> None:
        self.city = city
        self.rounds = 0
        self.by_interest.clear()
        self.processed.clear()


def _result_key(item: dict[str, Any]) -> tuple[str, str]:
    """网页结果的去重键，和 _dedupe_web_results 保持一致。"""
    return (
        str(item.get("title") or "").strip().casefold(),
        str(item.get("url") or "").strip(),
    )


def _dedupe_web_results(
    entries: Sequence[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """同一链接/标题只喂一次，避免抽取时重复。"""
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for _, results in entries:
        for item in results:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title:
                continue
            key = (title.casefold(), url)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _prefer_place(
    base: ExtractedPlace,
    alias: ExtractedPlace,
) -> ExtractedPlace:
    """合并同义地点：名称/分类/链接用代表项，介绍取更长的一份。"""
    if len(alias.description) <= len(base.description):
        return base
    return base.model_copy(update={"description": alias.description})


async def _apply_alias_merge(
    city: str,
    session: _ExtractionSession,
    resolver: PlaceAliasResolver,
    warnings: list[str],
) -> dict[str, str]:
    """把同义地点并到代表 id 上，返回「别名 -> 代表名」供日志查看。"""
    unique: dict[str, ExtractedPlace] = {}
    for bucket in session.by_interest.values():
        for place in bucket.values():
            unique.setdefault(place.id, place)
    if len(unique) < 2:
        return {}

    try:
        raw_mapping = await resolver.resolve(
            city=city,
            places=list(unique.values()),
        )
    except Exception as exc:  # noqa: BLE001
        # 归并是增强项，失败时保留原始结果，不能阻断初选
        logger.warning("地点别名归并失败：%s", exc)
        warnings.append(f"别名归并失败（{type(exc).__name__}），保留原始结果")
        return {}

    mapping = {
        alias: canonical
        for alias, canonical in raw_mapping.items()
        if alias in unique and canonical in unique and alias != canonical
    }
    if not mapping:
        return {}

    # 代表 id 由代码按固定规则决定（介绍更长 > 名称更完整 > id），
    # 避免 LLM 每轮把主副颠倒导致计数抖动。
    grouped: dict[str, set[str]] = {}
    for alias, canonical in mapping.items():
        grouped.setdefault(canonical, {canonical}).add(alias)

    stable: dict[str, str] = {}
    for members in grouped.values():
        candidates = [unique[place_id] for place_id in members if place_id in unique]
        if len(candidates) < 2:
            continue
        representative = max(
            candidates,
            key=lambda place: (
                len(place.description),
                len(place.name),
                place.id,
            ),
        )
        for place in candidates:
            if place.id != representative.id:
                stable[place.id] = representative.id

    if not stable:
        return {}

    for interest, bucket in session.by_interest.items():
        rebuilt: dict[str, ExtractedPlace] = {}
        for place in bucket.values():
            canonical_id = stable.get(place.id, place.id)
            base = unique.get(canonical_id, place)
            rebuilt[canonical_id] = _prefer_place(base, place)
        session.by_interest[interest] = rebuilt

    names = {place.id: place.name for place in unique.values()}
    return {
        names[alias]: names[canonical] for alias, canonical in stable.items()
    }


# ---------------------------- 入参 Schema ----------------------------


class ExtractWebPlacesInput(BaseModel):
    city: str = Field(..., min_length=1, description="城市中文名，如 南宁")
    interests: list[str] = Field(
        ..., min_length=1, description="用户兴趣，按兴趣分别抽取"
    )
    web_search_ids: list[str] | None = Field(
        None,
        description=(
            "mcp_web_search 返回的 web_search_id；"
            "不传时使用本次会话里全部网页检索结果（已消费的网页不会重复抽取）"
        ),
    )
    limit_per_interest: int = Field(
        15,
        ge=1,
        le=30,
        description="每个兴趣最多保留多少个候选；低于最小目标（10）时会自动抬高",
    )


# ---------------------------- 工厂 ----------------------------


def _build_extract_runnable(settings: Settings) -> Runnable:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
        timeout=settings.attraction_ranking_timeout_seconds,
    )
    return llm.with_structured_output(_RawPlaces, method="json_mode")


def _build_alias_runnable(settings: Settings) -> Runnable:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=0.0,
        timeout=settings.attraction_ranking_timeout_seconds,
    )
    return llm.with_structured_output(_RawMergeGroups, method="json_mode")


def create_web_place_extractor(
    *,
    settings: Settings | None = None,
) -> WebPlaceExtractor:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError("缺少 OPENAI_API_KEY，无法从网页结果抽取地点")
    return LLMWebPlaceExtractor(
        _build_extract_runnable(settings),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )


def create_place_alias_resolver(
    *,
    settings: Settings | None = None,
) -> PlaceAliasResolver:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError("缺少 OPENAI_API_KEY，无法归并同义地点")
    return LLMPlaceAliasResolver(
        _build_alias_runnable(settings),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )


def build_extract_web_places_tool(
    pool_cache: AttractionPoolCache,
    web_search_cache: WebSearchResultCache,
    *,
    extractor: WebPlaceExtractor | None = None,
    alias_resolver: PlaceAliasResolver | None = None,
    min_places_per_interest: int = DEFAULT_MIN_PLACES_PER_INTEREST,
    max_rounds: int = DEFAULT_MAX_EXTRACT_ROUNDS,
) -> BaseTool:
    """构建 extract_web_places。

    抽取成功后把轻量候选池写入 pool_cache（kind=web_place），
    返回的 search_id 直接交给 rank_attraction_pool。

    alias_resolver 负责在计数前归并同义地点；缺少 LLM 配置时自动关闭。
    显式传 None 时按默认配置创建，和旧工厂保持一致。
    """
    extractor = extractor or create_web_place_extractor()
    if alias_resolver is None:
        try:
            alias_resolver = create_place_alias_resolver()
        except ValueError:
            # 归并是增强项：没有 LLM key 时关闭，不影响抽取/排序
            alias_resolver = None

    session = _ExtractionSession()

    async def _run(
        city: str,
        interests: list[str],
        web_search_ids: list[str] | None = None,
        limit_per_interest: int = 15,
    ) -> str:
        normalized_interests = [
            interest.strip() for interest in interests if interest.strip()
        ]
        if not normalized_interests:
            return error_result("invalid_interests", "interests 不能为空")

        if web_search_ids:
            entries: list[tuple[str, list[dict[str, Any]]]] = []
            for web_search_id in web_search_ids:
                found = web_search_cache.get(web_search_id)
                if found is None:
                    return error_result(
                        "web_search_id_not_found",
                        "网页检索结果不存在或已过期，请重新调用 mcp_web_search",
                        retryable=True,
                    )
                entries.append(found)
        else:
            entries = web_search_cache.all()

        if not entries:
            return error_result(
                "no_web_results",
                "还没有可用的网页检索结果，请先调用 mcp_web_search",
                retryable=True,
            )

        city_text = city.strip()
        if session.city != city_text:
            session.reset(city_text)

        # 调用方传了更小的 limit 也不能低于最小目标，否则永远无法达标
        target_limit = max(limit_per_interest, min_places_per_interest)
        warnings: list[str] = []
        consumed_new = False

        for interest in normalized_interests:
            matched = [
                entry
                for entry in entries
                if interest.casefold() in entry[0].casefold()
            ]
            # query 里没带兴趣词时退回全部结果，避免整条链路因命名差异断掉
            results = _dedupe_web_results(matched or entries)
            bucket = session.by_interest.setdefault(interest, {})
            if len(bucket) >= target_limit:
                continue

            seen = session.processed.setdefault(interest, set())
            # 补检索只会带来新增网页；已消费的摘要不再重复喂给 LLM
            fresh = [item for item in results if _result_key(item) not in seen]
            if not fresh:
                continue

            try:
                places = await extractor.extract(
                    city=city_text,
                    interest=interest,
                    results=fresh,
                    limit=target_limit,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("兴趣 %s 的地点抽取失败：%s", interest, exc)
                warnings.append(f"{interest}：地点抽取失败（{type(exc).__name__}）")
                continue

            # 只有抽取成功才标记为已消费；失败时留给下一轮重试
            consumed_new = True
            for item in fresh:
                seen.add(_result_key(item))
            for place in places:
                if len(bucket) >= target_limit:
                    break
                existing = bucket.get(place.id)
                if existing is None or (
                    not existing.description and place.description
                ):
                    bucket[place.id] = place

        if consumed_new:
            session.rounds += 1

        # 先按原始名称计数，再让 LLM 归并同义地点；合并后重新计数，
        # 避免「中山路美食街 / 中山路夜市」这种别名把数量撑到门槛。
        raw_counts = {
            interest: len(session.by_interest.get(interest, {}))
            for interest in normalized_interests
        }
        alias_merged: dict[str, str] = {}
        # 只有本轮真的抽到新地点时才需要重新归并；同一批候选重复送 LLM 没有意义
        if alias_resolver is not None and consumed_new and any(raw_counts.values()):
            alias_merged = await _apply_alias_merge(
                city_text,
                session,
                alias_resolver,
                warnings,
            )

        by_interest = {
            interest: len(session.by_interest.get(interest, {}))
            for interest in normalized_interests
        }

        merged: dict[str, ExtractedPlace] = {}
        for places in session.by_interest.values():
            for place in places.values():
                existing = merged.get(place.id)
                if existing is None or (
                    not existing.description and place.description
                ):
                    merged[place.id] = place

        if not merged:
            return error_result(
                "no_places_extracted",
                "网页结果里没有抽到可用地点；请再次调用 mcp_web_search"
                "（用更大的 count）后重新抽取",
                retryable=True,
                by_interest=by_interest,
                raw_counts=raw_counts,
                alias_merged=alias_merged,
                needs_search=not consumed_new,
                rounds_left=max(0, max_rounds - session.rounds),
                warnings=warnings,
            )

        shortages = {
            interest: count
            for interest, count in by_interest.items()
            if count < min_places_per_interest
        }

        if shortages and session.rounds < max_rounds:
            next_count = min(50, 10 * (2 ** session.rounds))
            if consumed_new:
                next_step = (
                    "请对不足的兴趣再次调用 mcp_web_search（query 不变，count 用 "
                    f"{next_count} 或更大），然后重新调用 extract_web_places"
                )
            else:
                next_step = (
                    "本轮没有新的网页结果；请先对不足的兴趣调用 mcp_web_search"
                    f"（query 不变，count 用 {next_count} 或更大），再重新抽取"
                )
            return error_result(
                "insufficient_places",
                "地点数量不足："
                + "、".join(
                    f"{interest} {count} 个" for interest, count in shortages.items()
                )
                + f"，每个兴趣至少需要 {min_places_per_interest} 个；"
                + next_step,
                retryable=True,
                city=city_text,
                kind="web_place",
                by_interest=by_interest,
                raw_counts=raw_counts,
                alias_merged=alias_merged,
                shortages=shortages,
                needs_search=True,
                round=session.rounds,
                rounds_left=max_rounds - session.rounds,
                next_count=next_count,
                warnings=warnings,
            )

        if shortages:
            warnings.append(
                f"达到检索轮次上限，仍有兴趣不足 "
                f"{min_places_per_interest} 个："
                + "、".join(
                    f"{interest} {count} 个" for interest, count in shortages.items()
                )
            )

        search_id = pool_cache.save(
            city_text,
            list(merged.values()),
            kind="web_place",
        )
        return dump_payload(
            {
                "ok": True,
                "search_id": search_id,
                "city": city_text,
                "kind": "web_place",
                "interests": normalized_interests,
                "by_interest": by_interest,
                "raw_counts": raw_counts,
                "alias_merged": alias_merged,
                "total": len(merged),
                "round": session.rounds,
                "shortages": shortages,
                "items": [
                    {
                        "id": place.id,
                        "name": place.name,
                        "category": place.category.value,
                        "description": place.description,
                    }
                    for place in merged.values()
                ],
                "warnings": warnings,
                "hint": "下一步调用 rank_attraction_pool，把 search_id 和用户兴趣传进去",
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="extract_web_places",
        description=(
            "从 mcp_web_search 的网页结果里抽取真实地点，写入轻量候选池并返回 search_id。"
            "抽取后会先归并同义/别名地点（如「中山路美食街」与「中山路夜市」），"
            "再按兴趣统计：任一兴趣不足 "
            f"{min_places_per_interest} 个时返回 insufficient_places，"
            "请让 Agent 用更大的 count 补检索后再次调用本工具；"
            "已抽到的地点与已消费的网页会保留并去重。"
        ),
        args_schema=ExtractWebPlacesInput,
        handle_validation_error=handle_validation_error,
    )
