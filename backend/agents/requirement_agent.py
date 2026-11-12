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
from typing import Any, Literal, Protocol, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError

from backend.schemas.attraction import AttractionCategory
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




def _normalize_currency(value: str | None) -> str | None:
    """把 "dollars" / "美元" / "usd " 统一成 ISO 代码；无法识别返回 None。"""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) == 3 and cleaned.upper() in ALLOWED_CURRENCIES:
        return cleaned.upper()
    return None


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
        canonical = clean
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def _normalize_style(value: str | None) -> TravelStyle | None:
    if value is None:
        return None
    try:
        return TravelStyle(value.strip().lower())
    except ValueError:
        return None


def _normalize_destination(value: str | None) -> str | None:
    """城市名只做去空白；提示词已要求 LLM 输出英文名。"""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


# ---------------------------- 提示词字段说明（由 schema 生成） ----------------------------
# 提示词里的「字段 + 可填值」直接从 schema 取值：
#   字段名与顺序 → _LLMExtraction 的字段定义
#   兴趣取值     → AttractionCategory
#   节奏取值     → TravelStyle
#   币种取值     → ALLOWED_CURRENCIES
#   数值范围     → RequirementPatch 的 Field 校验约束
# 新增字段或调整取值范围时只改 schema，提示词和结构化输出 schema 自动同步。

_STYLE_EXTRA_HINTS: dict[TravelStyle, tuple[str, ...]] = {
    TravelStyle.RELAXED: ("不想太累", "不赶时间", "慢慢逛"),
    TravelStyle.BALANCED: (),
    TravelStyle.INTENSIVE: ("赶行程", "多打卡"),
}


def _field_bounds(field_name: str) -> tuple[int | None, int | None]:
    """从 RequirementPatch 的校验约束里取上下限，避免提示词和校验规则各写一份。"""
    metadata = RequirementPatch.model_fields[field_name].metadata
    lower = next((item.ge for item in metadata if getattr(item, "ge", None) is not None), None)
    upper = next((item.le for item in metadata if getattr(item, "le", None) is not None), None)
    return lower, upper


def _interests_hint() -> str:
    """兴趣取值来自 AttractionCategory，排除兜底用的 other。"""
    return " / ".join(
        category.value
        for category in AttractionCategory
        if category is not AttractionCategory.OTHER
    )


def _styles_hint() -> str:
    parts: list[str] = []
    for style in TravelStyle:
        extra = _STYLE_EXTRA_HINTS.get(style, ())
        parts.append(f"{style.value}（{'、'.join(extra)}）" if extra else style.value)
    return " / ".join(parts)


_DURATION_MIN, _DURATION_MAX = _field_bounds("duration_days")
_TRAVELERS_MIN, _TRAVELERS_MAX = _field_bounds("travelers")


# ---------------------------- LLM 抽取 ----------------------------


class _LLMExtraction(BaseModel):
    """给 OpenAI 结构化输出用的扁平原型。所有字段都可空，None 表示没提到。

    每个字段的 description 同时用于结构化输出 schema 和下面的提示词，
    字段说明只维护这一份。
    """

    destination: str | None = Field(
        None,
        description="城市名，用英文名（如 Tokyo、Osaka、Kyoto）。只提到国家、没说城市时输出 null。",
    )
    duration_days: int | None = Field(
        None,
        description=(
            f"整数 {_DURATION_MIN}~{_DURATION_MAX}，单位天。"
            "“玩 5 天”“5 天的行程”→ 5。"
        ),
    )
    start_date: str | None = Field(
        None,
        description=(
            "YYYY-MM-DD。“下个月”“下周五”这类相对日期，按给定的今天日期换算；"
            "无法可靠换算时输出 null。"
        ),
    )
    travelers: int | None = Field(
        None,
        description=(
            f"整数 {_TRAVELERS_MIN}~{_TRAVELERS_MAX}，出行人数。"
            "“我们 4 个人”→ 4；“1000 人民币”里的“人”不是人数。"
        ),
    )
    budget_amount: float | None = Field(
        None,
        description="只填数字，不要带币种符号或千分位。“预算 1 万”→ 10000，“1.5 千”→ 1500。",
    )
    budget_currency: str | None = Field(
        None,
        description=(
            f"只能是 {' / '.join(sorted(ALLOWED_CURRENCIES))}。"
            "美元/美金→USD，日元→JPY，人民币/元/块→CNY，欧元→EUR。"
            "用户没说币种时输出 null，不要猜。"
        ),
    )
    interests: list[str] = Field(
        default_factory=list,
        description=f"判断用户的兴趣满足这个标签列表中的哪些标签,只能从这些标签中选，尽量准确，可多选：{_interests_hint()}。",
    )
    travel_style: str | None = Field(
        None,
        description=f"只能是 {_styles_hint()}。",
    )
    notes: str | None = Field(
        None,
        description="无法结构化的补充要求（如“不吃生食”“带老人小孩”），不要重复上面的字段。",
    )


def _render_field_specs() -> str:
    """按 _LLMExtraction 的字段顺序渲染提示词里的字段清单。"""
    return "\n".join(
        f"- {name}：{info.description}"
        for name, info in _LLMExtraction.model_fields.items()
    )


def _render_output_skeleton() -> str:
    """按 _LLMExtraction 的字段生成 JSON 模板，输出格式不再手写一份字段清单。"""
    body = ",\n".join(
        f'  "{name}": {"[]" if get_origin(info.annotation) is list else "null"}'
        for name, info in _LLMExtraction.model_fields.items()
    )
    return "{\n" + body + "\n}"


def _render_output_types() -> str:
    """字段名 + JSON 类型，帮助模型对齐 JSON 里每个值该写什么。"""
    labels = {str: "string", int: "integer", float: "number"}
    lines: list[str] = []
    for name, info in _LLMExtraction.model_fields.items():
        if get_origin(info.annotation) is list:
            label = "array"
        else:
            base = next(
                (arg for arg in get_args(info.annotation) if arg is not type(None)),
                info.annotation,
            )
            label = labels.get(base, "string")
        lines.append(f"{name}: {label}")
    return "\n".join(lines)


_SYSTEM_PROMPT = f"""你是专业的旅行需求抽取专家。任务：判断用户本轮这句话提到了哪些字段，并把内容映射成下面规定的可填值。

【字段与可填内容】
{_render_field_specs()}

【判断规则】
1. 只输出用户本轮明确提到的字段。没提到的字段一律输出 null,interests 输出 []。
2. 不要用常识补全，不要编造用户没说的信息；拿不准就输出 null。
3. 用户的话只要能对应到上面的可填值，就必须映射成规定取值，不要照抄原话。例如：
   - “旅游不想太累，不赶时间”→ travel_style = relaxed
   - “想多吃点当地小吃”→ interests = ["food"]
   - “预算大概一万块人民币”→ budget_amount = 10000,budget_currency = CNY
   - “我们一家四口”→ travelers = 4
4. 用户在修改已有需求（如“预算改成 1500”）时，只输出被修改的字段，其余输出 null。
5. “不想要 X”这类否定表达不算喜欢 X：例如“不想购物”→ interests 不填 shopping。

【输出格式】
只输出一个 JSON 对象，不要 markdown 代码块，不要解释。必须包含下面全部字段，不能省略；
未提到的字段填 null，interests 填 []。字段与类型：
{_render_output_types()}

JSON 结构：
{_render_output_skeleton()}
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
        data["destination"] = _normalize_destination(raw.destination)

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
                # 不静默猜测：金额保留下来，但把「按 CNY 处理」写进 warnings
                data["budget_currency"] = "CNY"
                warnings.append("预算未说明币种，已按 CNY 处理")
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


def _is_empty_extraction(raw: _LLMExtraction) -> bool:
    """判断 LLM 返回的结构化结果是否等于「什么都没抽到」。

    背景：DeepSeek 等兼容端点调用 completions.parse(response_format=Model) 时，
    可能不抛异常但返回所有字段为 None / 空列表的空对象。此时不能把它当成
    「LLM 成功抽取了空需求」，而应该降级到 json mode 或规则解析。
    """
    return (
        not raw.destination
        and raw.duration_days is None
        and not raw.start_date
        and raw.travelers is None
        and raw.budget_amount is None
        and not raw.budget_currency
        and not raw.interests
        and not raw.travel_style
        and not raw.notes
    )


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
        raw: _LLMExtraction | None = None
        structured_exc: Exception | None = None
        json_exc: Exception | None = None

        try:
            candidate = await self._structured_output(messages)
            # 兼容端点可能返回空对象而不报错，视为失败，继续降级
            if not _is_empty_extraction(candidate):
                raw = candidate
        except Exception as exc:  # noqa: BLE001 - 兼容性探测，失败继续尝试
            structured_exc = exc
            logger.info("json_schema 输出不可用，改用 JSON mode: %s", exc)

        # 第二优先：json_object + Pydantic 校验（DeepSeek 等兼容端点）
        if raw is None:
            try:
                candidate = await self._json_mode_output(messages)
                if not _is_empty_extraction(candidate):
                    raw = candidate
            except Exception as exc:  # noqa: BLE001 - 外部 API 失败必须降级
                json_exc = exc
                logger.info("JSON mode 抽取失败或为空，降级为规则解析: %s", exc)

        # 第三优先：规则解析
        if raw is None:
            logger.warning("LLM 需求抽取未返回有效结果，降级为规则解析")
            fallback_outcome = await self._fallback.extract(
                message, previous=previous, today=current_day
            )
            exc_names = " / ".join(
                type(e).__name__ for e in (structured_exc, json_exc) if e is not None
            )
            return ExtractionOutcome(
                patch=fallback_outcome.patch,
                warnings=[
                    f"LLM 抽取未返回有效结果，已降级为规则解析"
                    + (f"（{exc_names}）" if exc_names else ""),
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
    # 「1000 人民币」里的「人」不是人数，必须排除，否则 1000 会被当成 travelers
    r"(\d+)\s*(?:人(?!民币)|位|个人|travelers?|people|persons?)", re.IGNORECASE
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

# 规则兜底专用的中文关键词表。提示词取值来自 schema，这些表只服务正则解析。
_RULE_CURRENCY_ALIASES: dict[str, str] = {
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

_RULE_INTEREST_KEYWORDS: dict[str, str] = {
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

_RULE_CITY_ALIASES: dict[str, str] = {
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


def _rule_currency(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if len(cleaned) == 3 and cleaned.upper() in ALLOWED_CURRENCIES:
        return cleaned.upper()
    return _RULE_CURRENCY_ALIASES.get(cleaned)


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
            currency = _rule_currency(match.group(2))
        elif match := _BUDGET_WITH_CURRENCY_PATTERN.search(text):
            data["budget_amount"] = Decimal(match.group(1))
            currency = _rule_currency(match.group(2))
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
        for keyword, canonical in _RULE_INTEREST_KEYWORDS.items():
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
        for alias, city in _RULE_CITY_ALIASES.items():
            index = lowered.find(alias)
            if index >= 0 and (best is None or index < best[0]):
                best = (index, city)
        return best[1] if best else None
