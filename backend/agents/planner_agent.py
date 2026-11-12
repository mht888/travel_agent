"""行程排程 Agent（第 7 阶段）。

和 Research Agent 的分工：
    Research Agent    bind_tools + 自主循环        → 文本情报
    Planner Agent     with_structured_output + 单次 → 结构化排程草稿

为什么 Planner 不用 bind_tools：
    排程需要的全部数据（候选景点、交通矩阵、天气）在调用前已经一次性拿到，
    LLM 不需要再决定「查什么」，只需要决定「怎么组合」。
    单次结构化输出更便宜、更可控、也更容易校验。

为什么保留规则排程作为降级：
    项目约定「能力降级链」，LLM 不可用时主流程不能断；
    同时它也是第 8 节 Benchmark 里的 Baseline 对照。
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, time
from typing import Protocol

from langchain_core.runnables import Runnable

from backend.config import Settings
from backend.schemas.attraction import Attraction
from backend.schemas.planner import DayPlan, ItineraryDraft
from backend.schemas.route import RouteMatrix
from backend.schemas.trip import TravelStyle, TripRequirements
from backend.schemas.weather import WeatherForecast

logger = logging.getLogger(__name__)

# 每种节奏的每日景点上限。LLM 提示词与规则降级共用这一份，避免两处口径不一致。
DAILY_LIMITS: dict[TravelStyle, int] = {
    TravelStyle.RELAXED: 4,
    TravelStyle.BALANCED: 5,
    TravelStyle.INTENSIVE: 7,
}


@dataclass
class PlannerOutcome:
    """排程结果 + 它是谁排的。source 用于日志、评估和前端提示。"""

    draft: ItineraryDraft
    source: str  # "llm" | "rule"
    warnings: list[str] = field(default_factory=list)


class ItineraryPlanner(Protocol):
    """排程器接口。LLM 实现与规则实现都满足它，Service 只依赖这个协议。"""

    async def plan(
        self,
        *,
        requirements: TripRequirements,
        candidates: Sequence[Attraction],
        matrix: RouteMatrix,
        weather: WeatherForecast | None = None,
        today: date | None = None,
    ) -> PlannerOutcome: ...


_WEEKDAY_NAMES = ["一", "二", "三", "四", "五", "六", "日"]


def _format_ticket(attraction: Attraction) -> str:
    """门票摘要。None 表示未知，0 表示确认免费，两者不能混。"""
    price = attraction.ticket_price
    if price is None:
        return "未知"
    if price.amount == 0:
        return "免费"
    return f"{price.amount}{price.currency}"


def _format_opening_hours(attraction: Attraction) -> str:
    """把一周的开放时段压成一行，避免提示词被重复数据撑爆。"""
    intervals = attraction.opening_hours
    if not intervals:
        return "未知"

    grouped: dict[tuple[time, time], list[int]] = {}
    for interval in intervals:
        grouped.setdefault((interval.opens_at, interval.closes_at), []).append(
            interval.weekday
        )

    parts: list[str] = []
    for (opens_at, closes_at), weekdays in grouped.items():
        days_text = "、".join(f"周{_WEEKDAY_NAMES[day]}" for day in sorted(weekdays))
        parts.append(f"{days_text} {opens_at:%H:%M}-{closes_at:%H:%M}")
    return "；".join(parts)


def _summarize_matrix(
    matrix: RouteMatrix, valid_ids: set[str], limit: int = 30
) -> str:
    """只给「耗时短」的边：它们才是「适合排同一天」的信号。

    10 个景点就有 90 条有向边，全量贴进提示词既浪费 token 又稀释注意力。
    """
    entries = [
        entry
        for entry in matrix.entries
        if entry.from_id in valid_ids
        and entry.to_id in valid_ids
        and entry.duration_minutes <= 60
    ]
    if not entries:
        # 城市景点分散时可能没有短边，退化为「最短的若干条」，避免信息为 0
        entries = sorted(
            (
                entry
                for entry in matrix.entries
                if entry.from_id in valid_ids and entry.to_id in valid_ids
            ),
            key=lambda entry: entry.duration_minutes,
        )
    entries = entries[:limit]
    if not entries:
        return "暂无（候选景点不足两个，或路线数据缺失）"
    return "\n".join(
        f"- {entry.from_id} -> {entry.to_id}: {entry.duration_minutes} 分钟"
        for entry in entries
    )


def _summarize_weather(weather: WeatherForecast | None) -> str:
    if weather is None or not weather.days:
        return "未查询（用户未提供出发日期）"
    return "\n".join(
        f"- {day.date}: {day.condition.value}，"
        f"{day.temp_min_c:.0f}~{day.temp_max_c:.0f}℃，"
        f"降水概率 {day.precipitation_probability}%"
        for day in weather.days
    )


def build_planner_prompt(
    requirements: TripRequirements,
    candidates: Sequence[Attraction],
    matrix: RouteMatrix,
    weather: WeatherForecast | None = None,
) -> str:
    """把需求 + 候选 + 矩阵 + 天气拼成一条 system prompt。

    TODO(你)：实现这个函数。
    这是第 7 阶段最重要的一段代码——同样的候选集，提示词好坏直接决定
    LLM 会不会编造 id、会不会把相距一小时的景点塞进同一天。

    必须包含的信息（缺一条就容易被 LLM 忽略）：
    1. 角色与任务：你是旅行规划师，只能从候选列表里选景点并分配到每一天。
    2. 用户需求：destination / duration_days / interests / travel_style /
       budget / travelers / notes。
    3. 硬约束，用「必须 / 禁止」写：
       - 正好输出 duration_days 天，day_index 从 1 连续；
       - 每天景点数不超过 DAILY_LIMITS[requirements.travel_style]；
       - attraction_ids 只能取自候选列表，禁止编造；
       - 同一个景点只能出现一次。
    4. 候选景点清单：建议一行一个，字段至少包含 id / name / category /
       rating / estimated_duration_minutes / ticket_price / opening_hours 摘要。
       **id 放最前面**，LLM 最容易抄错的就是 id。
    5. 交通矩阵摘要：从 matrix.entries 里挑关键边写成
       "from_id -> to_id: 12min"；候选较多时只列短耗时边，避免提示词爆炸。
    6. 天气（如果有）：日期 + condition + 降水概率，提示把户外景点排在晴天。
    7. 排序原则（按优先级）：同一天地理集中（参考矩阵耗时） > 兴趣命中 > 评分。
    8. 输出要求：严格按 schema 输出，theme 用一句中文描述当天主题。

    提示：
    - 不需要把 JSON Schema 写进提示词，with_structured_output 会自动带上；
      你要做的是把「决策所需的事实」写清楚。
    - candidates 已经按「兴趣命中 + 评分」排好序（AttractionService 做的），
      不要重复排序，直接用它建立提示词里的顺序直觉。
    - candidates 为空时提前 raise ValueError，不要拼一个空提示词给 LLM。
    - 审计便利：把 matrix.mode 写进提示词，避免 LLM 按「开车」估算步行时间。
    """
    if not candidates:
        raise ValueError("candidates 不能为空")

    daily_limit = DAILY_LIMITS[requirements.travel_style]
    valid_ids = {item.id for item in candidates}
    interests_text = "、".join(requirements.interests) or "未指定"
    budget_text = (
        f"{requirements.budget.amount} {requirements.budget.currency}"
        if requirements.budget
        else "未指定"
    )
    notes_text = requirements.notes or "无"

    candidate_lines = "\n".join(
        f"- {item.id} | {item.name} | {item.category.value} | "
        f"评分 {item.rating if item.rating is not None else '未知'} | "
        f"建议游玩 {item.estimated_duration_minutes} 分钟 | "
        f"门票 {_format_ticket(item)} | 开放时间 {_format_opening_hours(item)}"
        for item in candidates
    )
    day_sequence = list(range(1, requirements.duration_days + 1))

    return f"""你是一名资深旅行规划师，负责把候选景点分配到每一天。

## 用户需求
- 目的地：{requirements.destination}
- 天数：{requirements.duration_days} 天（必须正好输出 {requirements.duration_days} 天）
- 兴趣：{interests_text}
- 节奏：{requirements.travel_style.value}（每天最多 {daily_limit} 个景点）
- 预算：{budget_text}
- 人数：{requirements.travelers} 人
- 补充要求：{notes_text}

## 硬约束（违反任何一条都会导致排程失败）
1. day_index 必须是 {day_sequence}，不能跳号或重复。
2. 每天景点数不超过 {daily_limit} 个，且至少 1 个。
3. attraction_ids 只能从下面的候选清单里选，禁止编造、禁止改写 id。
4. 同一个景点只能安排一次，不能跨天重复。

## 候选景点（已按兴趣命中与评分排序）
{candidate_lines}

## 步行耗时较短的景点对（来自路线矩阵，单位分钟）
{_summarize_matrix(matrix, valid_ids)}

## 天气
{_summarize_weather(weather)}

## 排程原则（按优先级）
1. 同一天尽量安排地理集中的景点：优先把上面「耗时短」的景点对放在同一天。
2. 兴趣匹配优先于评分：用户明确喜欢的类别优先入选。
3. 遵守节奏：{requirements.travel_style.value} 的用户不要塞满，留出休息和用餐时间。
4. 户外与自然类景点优先安排在降水概率低的日期。

## 输出要求
只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字，结构必须是：
{{"days": [{{"day_index": 1, "attraction_ids": ["景点id1", "景点id2"], "theme": "当天主题"}}], "rationale": "一句话说明整体思路"}}
其中 days 必须正好 {requirements.duration_days} 条，attraction_ids 只能用上面的 id。
"""


def create_planner_runnable(settings: Settings) -> Runnable:
    """创建带结构化输出的 LLM Runnable。

    为什么用 method="json_mode" 而不是默认的 json_schema：
        默认的 json_schema 会发 response_format={"type": "json_schema"}，
        而 thinking 模式下的 OpenAI 兼容端点（如 deepseek-flash）会直接返回
        400「This response_format type is unavailable now」；function_calling
        同样被拒（「Thinking mode does not support this tool_choice」）。
        json_mode 只要求 response_format={"type": "json_object"}，兼容性最好，
        代价是必须在提示词里显式给出 JSON 结构（build_planner_prompt 已提供），
        最后再由 ItineraryDraft.model_validate 做严格校验。
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        # 保留一点组合灵活性，但不至于乱排；完全 0 容易每次都同一个顺序
        temperature=0.2,
    )
    return llm.with_structured_output(ItineraryDraft, method="json_mode")


class LlmItineraryPlanner:
    """LLM 排程；失败时降级到规则排程，并把降级原因透传给用户。"""

    def __init__(
        self,
        runnable: Runnable,
        fallback: "RuleBasedItineraryPlanner",
    ) -> None:
        self._runnable = runnable
        self._fallback = fallback

    async def plan(
        self,
        *,
        requirements: TripRequirements,
        candidates: Sequence[Attraction],
        matrix: RouteMatrix,
        weather: WeatherForecast | None = None,
        today: date | None = None,
    ) -> PlannerOutcome:
        prompt = build_planner_prompt(requirements, candidates, matrix, weather)

        try:
            raw = await self._runnable.ainvoke(prompt)
            # 结构化输出正常返回 ItineraryDraft；这里再兜一次，防止不同
            # 兼容端点返回 dict。校验失败同样进入降级。
            draft = ItineraryDraft.model_validate(raw)
        except NotImplementedError:
            # TODO 未完成属于开发期错误，不吞掉，直接往上抛方便你定位
            raise
        except Exception as exc:  # noqa: BLE001 - LLM 失败必须转成可运行结果
            logger.warning("LLM 排程失败，降级为规则排程：%s", exc)
            outcome = await self._fallback.plan(
                requirements=requirements,
                candidates=candidates,
                matrix=matrix,
                weather=weather,
                today=today,
            )
            outcome.warnings.insert(
                0, f"LLM 排程失败，已降级为规则排程：{type(exc).__name__}"
            )
            return outcome

        logger.info("LLM 排程完成：%d 天", len(draft.days))
        return PlannerOutcome(draft=draft, source="llm")


class RuleBasedItineraryPlanner:
    """规则排程：按候选顺序每天切片。

    作为降级路径和 Benchmark 的 Baseline，不做地理聚类
    （那是第 10 阶段 Route Optimizer 的职责）。
    """

    async def plan(
        self,
        *,
        requirements: TripRequirements,
        candidates: Sequence[Attraction],
        matrix: RouteMatrix,
        weather: WeatherForecast | None = None,
        today: date | None = None,
    ) -> PlannerOutcome:
        if not candidates:
            raise ValueError("候选景点为空，无法生成行程")

        daily_limit = DAILY_LIMITS[requirements.travel_style]
        warnings = ["当前使用规则排程，同一天景点的地理集中度不做保证"]

        days: list[DayPlan] = []
        cursor = 0
        for day_index in range(1, requirements.duration_days + 1):
            chunk = list(candidates[cursor : cursor + daily_limit])
            if not chunk:
                break
            cursor += len(chunk)
            days.append(
                DayPlan(
                    day_index=day_index,
                    attraction_ids=[item.id for item in chunk],
                    theme=None,
                )
            )

        dropped = len(candidates) - cursor
        if dropped > 0:
            warnings.append(f"{dropped} 个候选景点因天数/节奏限制未纳入行程")

        return PlannerOutcome(
            draft=ItineraryDraft(days=days),
            source="rule",
            warnings=warnings,
        )
