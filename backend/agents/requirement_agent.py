"""需求抽取 Agent（第 4 阶段）。

职责边界：
    RequirementAgent / RuleBasedRequirementExtractor
        只负责「一句话 → 增量字段 RequirementPatch」，不做合并、不问问题。
    RequirementService
        负责多轮合并、缺失字段推导和追问措辞。
    Schema
        负责「什么样的字段值算合法」。

为什么 LLM 输出不直接复用 TripRequirementDraft：
    1. 草稿字段带默认值（travelers=1、travel_style=balanced），会掩盖「用户没提到」，
       多轮合并时会把旧值覆盖掉；
    2. Money / Decimal 生成的 JSON Schema 在 strict 模式下兼容性较差。
    因此 LLM 只输出扁平的原始类型，校验与转换全部放在 Python。
"""

import dataclasses
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from backend.schemas.common import ALLOWED_CURRENCIES
from backend.schemas.requirement import RequirementPatch, TripRequirementDraft
from backend.schemas.trip import TravelStyle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionOutcome:
    """抽取结果 + 过程元信息。

    warnings 必须透传到接口响应：降级、丢弃非法值这类事不能让调用方猜。
    """

    patch: RequirementPatch
    warnings: list[str] = dataclasses.field(default_factory=list)
    source: Literal["llm", "rules"] = "rules"


class RequirementExtractor(Protocol):
    """抽取器接口。LLM 与规则实现都满足它，Service 只依赖这个协议。"""

    async def extract(
        self,
        message: str,
        *,
        previous: TripRequirementDraft | None = None,
        today: date | None = None,
    ) -> ExtractionOutcome:
        ...


# ---------------------------- 字段归一化 ----------------------------

_CURRENCY_ALIASES: dict[str, str] = {
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "美元": "USD",
    "美金": "USD",
    "jpy": "JPY",
    "yen": "JPY",
    "日元": "JPY",
    "cny": "CNY",
    "rmb": "CNY",
    "yuan": "CNY",
    "人民币": "CNY",
    "元": "CNY",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "欧元": "EUR",
}

# 用户口语 → 与 AttractionCategory / 景点 tags 对齐的标签
_INTEREST_ALIASES: dict[str, str] = {
    "美食": "food",
    "吃": "food",
    "餐厅": "food",
    "food": "food",
    "购物": "shopping",
    "shopping": "shopping",
    "文化": "culture",
    "历史": "culture",
    "寺庙": "culture",
    "神社": "culture",
    "博物馆": "culture",
    "culture": "culture",
    "自然": "nature",
    "公园": "nature",
    "风景": "nature",
    "nature": "nature",
    "夜生活": "nightlife",
    "酒吧": "nightlife",
    "nightlife": "nightlife",
}

_STYLE_ALIASES: dict[str, TravelStyle] = {
    "relaxed": TravelStyle.RELAXED,
    "悠闲": TravelStyle.RELAXED,
    "轻松": TravelStyle.RELAXED,
    "放松": TravelStyle.RELAXED,
    "balanced": TravelStyle.BALANCED,
    "适中": TravelStyle.BALANCED,
    "intensive": TravelStyle.INTENSIVE,
    "紧凑": TravelStyle.INTENSIVE,
    "高强度": TravelStyle.INTENSIVE,
}

_CITY_ALIASES: dict[str, str] = {
    "东京": "Tokyo",
    "tokyo": "Tokyo",
    "大阪": "Osaka",
    "osaka": "Osaka",
    "京都": "Kyoto",
    "kyoto": "Kyoto",
    "首尔": "Seoul",
    "seoul": "Seoul",
    "曼谷": "Bangkok",
    "bangkok": "Bangkok",
    "新加坡": "Singapore",
    "singapore": "Singapore",
    "巴黎": "Paris",
    "paris": "Paris",
    "伦敦": "London",
    "london": "London",
    "纽约": "New York",
    "new york": "New York",
}


def _normalize_currency(value: str | None) -> str | None:
    """把 "dollars" / "美元" / "usd " 统一成 ISO 代码；无法识别返回 None。"""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) == 3 and cleaned.upper() in ALLOWED_CURRENCIES:
        return cleaned.upper()
    return _CURRENCY_ALIASES.get(cleaned.lower())


def _normalize_interests(values: Sequence[str] | None) -> list[str]:
    """映射到统一标签、去重、保持输入顺序。"""
    if not values:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        clean = value.strip().lower()
        if not clean:
            continue
        canonical = _INTEREST_ALIASES.get(clean, clean)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def _normalize_style(value: str | None) -> TravelStyle | None:
    if value is None:
        return None
    return _STYLE_ALIASES.get(value.strip().lower())


# ---------------------------- LLM 抽取 ----------------------------


class _LLMExtraction(BaseModel):
    """给 OpenAI 结构化输出用的扁平原型。所有字段都可空，None 表示没提到。"""

    destination: str | None = None
    duration_days: int | None = None
    start_date: str | None = Field(None, description="ISO 8601 日期，如 2026-04-01")
    travelers: int | None = None
    budget_amount: float | None = None
    budget_currency: str | None = None
    interests: list[str] = Field(default_factory=list)
    travel_style: str | None = None
    notes: str | None = None


_SYSTEM_PROMPT = """你是旅行需求抽取器。把用户的话转换成结构化字段，不要编造用户没说的信息。

规则：
1. 只抽取用户明确表达、或结合上下文可以直接推断的信息；拿不准的字段输出 null。
2. start_date 使用 ISO 格式 YYYY-MM-DD。用户说“下个月”“下周五”这类相对日期时，
   结合给定的今天日期换算；无法可靠换算时输出 null。
3. interests 只能从 food / shopping / culture / nature / nightlife 中选，可多选。
4. travel_style 只能是 relaxed（悠闲、不要太紧凑）、balanced（适中）、intensive（紧凑、高强度）。
5. budget_currency 用 ISO 三字母代码：美元→USD，日元→JPY，人民币→CNY，欧元→EUR。
   用户没说币种时输出 null。
6. notes 只放无法结构化的补充要求（如“不吃生食”），不要重复已有字段。
7. 如果用户在修改已有需求（如“预算改成 1500”），只输出被修改的字段，其余输出 null。

输出格式：只输出一个 JSON 对象，不要 markdown 代码块，不要解释。字段如下（全部必填，未知用 null）：
destination: 城市名或 null
duration_days: 整数天数或 null
start_date: YYYY-MM-DD 或 null
travelers: 整数或 null
budget_amount: 数字或 null
budget_currency: USD/JPY/CNY/EUR 或 null
interests: 标签数组（可为空数组）
travel_style: relaxed/balanced/intensive 或 null
notes: 字符串或 null
"""


def _build_user_prompt(
    message: str,
    previous: TripRequirementDraft | None,
    today: date,
) -> str:
    parts = [f"今天日期：{today.isoformat()}"]
    if previous is not None:
        parts.append(
            "已确认的需求（JSON；本次没有修改的字段不要重复输出）："
            f"{previous.model_dump_json(exclude_none=True)}"
        )
    parts.append(f"用户这次输入：{message}")
    return "\n".join(parts)


def _strip_code_fence(text: str) -> str:
    """兼容部分 OpenAI 兼容端点给 JSON 套 ```json 围栏的行为。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _to_patch(raw: _LLMExtraction) -> tuple[RequirementPatch, list[str]]:
    """把 LLM 原始输出逐字段校验后再组装成 Patch。

    逐字段处理而不是整体 model_validate：
    一个字段非法（例如天数 0）不应该让整句话的抽取结果全部作废。
    """
    warnings: list[str] = []
    data: dict[str, Any] = {}

    if raw.destination and raw.destination.strip():
        data["destination"] = raw.destination.strip()

    if raw.duration_days is not None:
        if 1 <= raw.duration_days <= 30:
            data["duration_days"] = raw.duration_days
        else:
            warnings.append(f"天数 {raw.duration_days} 超出 1~30，已忽略")

    if raw.start_date and raw.start_date.strip():
        try:
            data["start_date"] = date.fromisoformat(raw.start_date.strip())
        except ValueError:
            warnings.append(f"无法解析出发日期「{raw.start_date}」，已忽略")

    if raw.travelers is not None:
        if 1 <= raw.travelers <= 20:
            data["travelers"] = raw.travelers
        else:
            warnings.append(f"人数 {raw.travelers} 超出 1~20，已忽略")

    if raw.budget_amount is not None:
        try:
            amount = Decimal(str(raw.budget_amount))
        except InvalidOperation:
            amount = None
        if amount is not None and amount > 0:
            data["budget_amount"] = amount.quantize(Decimal("0.01"))
            currency = _normalize_currency(raw.budget_currency)
            if currency is None:
                # 不静默猜测：金额保留下来，但把「按 USD 处理」写进 warnings
                data["budget_currency"] = "USD"
                warnings.append("预算未说明币种，已按 USD 处理")
            else:
                data["budget_currency"] = currency
        else:
            warnings.append(f"预算金额「{raw.budget_amount}」不合法，已忽略")

    interests = _normalize_interests(raw.interests)
    if interests:
        data["interests"] = interests

    if raw.travel_style:
        style = _normalize_style(raw.travel_style)
        if style is None:
            warnings.append(f"无法识别旅行风格「{raw.travel_style}」，已忽略")
        else:
            data["travel_style"] = style

    if raw.notes and raw.notes.strip():
        data["notes"] = raw.notes.strip()

    try:
        return RequirementPatch(**data), warnings
    except ValidationError as exc:
        # 理论上前面已逐字段校验，走不到这里；真走到了也不能让接口 500
        logger.warning("需求补丁校验失败: %s", exc)
        return RequirementPatch(), warnings + ["抽取结果未通过校验，已忽略本次增量"]


class RequirementAgent:
    """LLM 抽取器。任何异常都降级到规则解析，保证流程不被外部 API 阻塞。"""

    def __init__(
        self,
        client: Any,
        model: str,
        fallback: RequirementExtractor | None = None,
    ) -> None:
        # client 传 Any 而不是 AsyncOpenAI：测试里用假客户端即可，不必联网
        self._client = client
        self._model = model
        self._fallback = fallback or RuleBasedRequirementExtractor()

    async def extract(
        self,
        message: str,
        *,
        previous: TripRequirementDraft | None = None,
        today: date | None = None,
    ) -> ExtractionOutcome:
        current_day = today or date.today()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(message, previous, current_day),
            },
        ]

        # 第一优先：json_schema 严格结构化输出（OpenAI 官方模型支持）
        try:
            raw = await self._structured_output(messages)
        except Exception as structured_exc:  # noqa: BLE001 - 兼容性探测，失败继续尝试
            logger.info("json_schema 输出不可用，改用 JSON mode: %s", structured_exc)
            try:
                # 第二优先：json_object + Pydantic 校验（DeepSeek 等兼容端点）
                raw = await self._json_mode_output(messages)
            except Exception as json_exc:  # noqa: BLE001 - 外部 API 失败必须降级
                logger.warning("LLM 需求抽取失败，降级为规则解析: %s", json_exc)
                fallback_outcome = await self._fallback.extract(
                    message, previous=previous, today=current_day
                )
                return ExtractionOutcome(
                    patch=fallback_outcome.patch,
                    warnings=[
                        "LLM 抽取失败，已降级为规则解析："
                        f"{type(structured_exc).__name__} / {type(json_exc).__name__}",
                        *fallback_outcome.warnings,
                    ],
                    source="rules",
                )

        patch, warnings = _to_patch(raw)
        return ExtractionOutcome(patch=patch, warnings=warnings, source="llm")

    async def _structured_output(self, messages: list[dict[str, str]]) -> _LLMExtraction:
        """OpenAI 严格结构化输出：服务端按 Pydantic 生成的 JSON Schema 约束。"""
        completion = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=_LLMExtraction,
            temperature=0,
        )
        reply = completion.choices[0].message
        if getattr(reply, "refusal", None):
            raise ValueError(f"模型拒绝回答：{reply.refusal}")
        parsed = getattr(reply, "parsed", None)
        if parsed is None:
            raise ValueError("模型没有返回结构化结果")
        return parsed

    async def _json_mode_output(self, messages: list[dict[str, str]]) -> _LLMExtraction:
        """兼容只支持 json_object 的端点（如 DeepSeek）。

        json_object 模式不保证 schema，所以这里必须再走一遍 Pydantic 校验；
        校验失败会抛异常，由调用方降级到规则解析。
        """
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = getattr(completion.choices[0].message, "content", None)
        if not content or not content.strip():
            raise ValueError("模型没有返回内容")
        return _LLMExtraction.model_validate_json(_strip_code_fence(content))


# ---------------------------- 规则抽取（降级方案） ----------------------------

_DURATION_PATTERN = re.compile(r"(\d+)\s*(?:天|日|days?|nights?)", re.IGNORECASE)
_TRAVELERS_PATTERN = re.compile(
    r"(\d+)\s*(?:人|位|个人|travelers?|people|persons?)", re.IGNORECASE
)
_BUDGET_PATTERN = re.compile(
    r"(?:预算|budget|花费|控制在)[^\d]{0,8}"
    r"(\d+(?:\.\d+)?)\s*"
    r"(美元|美金|日元|人民币|元|欧元|usd|jpy|cny|eur|dollars?|yen|yuan|euros?)?",
    re.IGNORECASE,
)
_BUDGET_WITH_CURRENCY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(美元|美金|日元|人民币|元|欧元|usd|jpy|cny|eur|dollars?|yen|yuan|euros?)",
    re.IGNORECASE,
)
_FULL_DATE_PATTERN = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")
_MONTH_DAY_PATTERN = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")


def _mask(text: str, start: int, end: int) -> str:
    """把已匹配的片段替换成等长空格，避免同一段文字被二次解析。"""
    return f"{text[:start]}{' ' * (end - start)}{text[end:]}"


class RuleBasedRequirementExtractor:
    """不依赖网络的兜底抽取器。

    覆盖开发与测试里最常用的表达（城市、天数、预算、人数、兴趣、节奏、显式日期）。
    它不追求高召回：解析不到的字段留空，由 Service 生成追问。
    """

    async def extract(
        self,
        message: str,
        *,
        previous: TripRequirementDraft | None = None,
        today: date | None = None,
    ) -> ExtractionOutcome:
        current_day = today or date.today()
        text = message.strip()
        warnings: list[str] = []
        data: dict[str, Any] = {}

        # 日期先解析并从文本里挖掉，避免 "3月5日" 被当成 "5 天"
        start_date, text = self._extract_start_date(text, current_day)
        if start_date is not None:
            data["start_date"] = start_date

        lowered = text.lower()

        if match := _DURATION_PATTERN.search(text):
            data["duration_days"] = int(match.group(1))

        if match := _BUDGET_PATTERN.search(text):
            data["budget_amount"] = Decimal(match.group(1))
            currency = _normalize_currency(match.group(2))
        elif match := _BUDGET_WITH_CURRENCY_PATTERN.search(text):
            data["budget_amount"] = Decimal(match.group(1))
            currency = _normalize_currency(match.group(2))
        else:
            currency = None
            match = None

        if "budget_amount" in data:
            if currency is None:
                data["budget_currency"] = "USD"
                warnings.append("预算未说明币种，已按 USD 处理")
            else:
                data["budget_currency"] = currency

        if match := _TRAVELERS_PATTERN.search(text):
            data["travelers"] = int(match.group(1))

        style = self._extract_style(lowered)
        if style is not None:
            data["travel_style"] = style

        interests = self._extract_interests(lowered)
        if interests:
            data["interests"] = interests

        destination = self._extract_destination(lowered)
        if destination is not None:
            data["destination"] = destination

        try:
            patch = RequirementPatch(**data)
        except ValidationError as exc:
            logger.warning("规则抽取结果校验失败: %s", exc)
            return ExtractionOutcome(
                patch=RequirementPatch(),
                warnings=[*warnings, "规则抽取结果未通过校验，已忽略本次增量"],
                source="rules",
            )
        return ExtractionOutcome(patch=patch, warnings=warnings, source="rules")

    @staticmethod
    def _extract_start_date(text: str, today: date) -> tuple[date | None, str]:
        if match := _FULL_DATE_PATTERN.search(text):
            try:
                value = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                return None, text
            return value, _mask(text, *match.span())

        if match := _MONTH_DAY_PATTERN.search(text):
            try:
                value = date(today.year, int(match.group(1)), int(match.group(2)))
            except ValueError:
                return None, text
            # 用户说“3 月 5 日”但今年已经过了，按明年算
            if value < today:
                try:
                    value = value.replace(year=today.year + 1)
                except ValueError:
                    return None, text
            return value, _mask(text, *match.span())

        return None, text

    @staticmethod
    def _extract_style(lowered: str) -> TravelStyle | None:
        # 先匹配「不要太紧凑」，否则会被后面的「紧凑」误判
        if re.search(r"不要太紧凑|悠闲|轻松|放松|relaxed|chill", lowered):
            return TravelStyle.RELAXED
        if re.search(r"紧凑|高强度|intensive|packed", lowered):
            return TravelStyle.INTENSIVE
        if re.search(r"适中|balanced", lowered):
            return TravelStyle.BALANCED
        return None

    @staticmethod
    def _extract_interests(lowered: str) -> list[str]:
        # 按关键词首次出现的位置排序，保持用户表达顺序
        hits: list[tuple[int, str]] = []
        for keyword, canonical in _INTEREST_ALIASES.items():
            index = lowered.find(keyword)
            if index >= 0:
                hits.append((index, canonical))

        seen: set[str] = set()
        interests: list[str] = []
        for _, canonical in sorted(hits):
            if canonical not in seen:
                seen.add(canonical)
                interests.append(canonical)
        return interests

    @staticmethod
    def _extract_destination(lowered: str) -> str | None:
        best: tuple[int, str] | None = None
        for alias, city in _CITY_ALIASES.items():
            index = lowered.find(alias)
            if index >= 0 and (best is None or index < best[0]):
                best = (index, city)
        return best[1] if best else None
