"""行程数据结构。这是「系统排出来什么」，会被 Evaluator 逐条检查。"""

from collections.abc import Sequence
from datetime import date as Date
from datetime import time
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from backend.schemas.common import DataStatus, Money, Provenance, TransportMode
from backend.schemas.trip import TripRequirements


def _to_minutes(value: time) -> int:
    """当天第几分钟。"""
    return value.hour * 60 + value.minute


def check_no_overlap(items: Sequence["ItineraryItem"]) -> list[str]:
    """检查相邻行程项时间是否重叠。

    纯函数：第 9 阶段 Evaluator 和 ItineraryDay 的校验共用同一份规则，
    避免「校验逻辑只活在 validator 里、无法复用」的问题。
    """
    problems: list[str] = []
    for previous, current in zip(items, items[1:]):
        if _to_minutes(current.start_time) < _to_minutes(previous.end_time):
            problems.append(
                f"第 {previous.seq} 项结束于 {previous.end_time:%H:%M}，"
                f"但第 {current.seq} 项 {current.start_time:%H:%M} 就开始了"
            )
    return problems


class TransportLeg(BaseModel):
    """两个停留点之间的一段交通。第 6 阶段由 Routes API 填充。"""

    mode: TransportMode
    duration_minutes: int = Field(..., ge=0)
    distance_meters: int | None = Field(None, ge=0)
    provenance: Provenance


class ItineraryItemType(str, Enum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    TRANSIT = "transit"
    BREAK = "break"


class ItineraryItem(BaseModel):
    """行程中的一个停留点。时间都是「当天本地时间」。"""

    seq: int = Field(..., ge=1, description="当天内的顺序，从 1 开始")
    type: ItineraryItemType
    name: str
    attraction_id: str | None = Field(None, description="指向 Attraction.id，便于回查")
    start_time: time
    end_time: time
    cost: Money | None = None
    transport_from_previous: TransportLeg | None = Field(
        None, description="第一项为 None"
    )

    @model_validator(mode="after")
    def _validate_time_order(self) -> "ItineraryItem":
        # 约定：只支持当天内的时段，跨午夜一律拆成两条
        if _to_minutes(self.end_time) <= _to_minutes(self.start_time):
            raise ValueError("end_time 必须晚于 start_time（跨午夜请拆成两条）")
        return self


class ItineraryDay(BaseModel):
    day_index: int = Field(..., ge=1)
    date: Date | None = None
    items: list[ItineraryItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_items(self) -> "ItineraryDay":
        if not self.items:
            return self

        seqs = [item.seq for item in self.items]
        expected = list(range(1, len(self.items) + 1))
        if seqs != expected:
            raise ValueError(f"items 的 seq 必须从 1 连续递增，期望 {expected}，实际 {seqs}")

        problems = check_no_overlap(self.items)
        if problems:
            raise ValueError("；".join(problems))
        return self

    @property
    def daily_cost(self) -> Money | None:
        """当天合计花费。没有带价格的项时返回 None。"""
        costs = [item.cost for item in self.items if item.cost is not None]
        if not costs:
            return None

        currencies = {cost.currency for cost in costs}
        if len(currencies) > 1:
            raise ValueError(
                f"第 {self.day_index} 天存在多种币种 {sorted(currencies)}，请先统一换算"
            )

        statuses = {cost.status for cost in costs}
        return Money(
            amount=sum((cost.amount for cost in costs), Decimal("0")),
            currency=costs[0].currency,
            # 全部来源一致就保留原状态，混合来源按最保守的「估算」处理
            status=next(iter(statuses)) if len(statuses) == 1 else DataStatus.ESTIMATED,
        )


class Itinerary(BaseModel):
    """一次完整规划结果。第 7~11 阶段逐步填充。"""

    trip: TripRequirements
    days: list[ItineraryDay] = Field(default_factory=list)
    total_cost: Money | None = None
    warnings: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(
        default_factory=list,
        description="优化达到最大重试次数仍未解决的问题，必须透传给用户，不能静默丢弃",
    )

    @model_validator(mode="after")
    def _check_day_count(self) -> "Itinerary":
        # 选择「允许 + warning」而不是报错：规划过程中行程会被逐步填充，
        # 中途保存半成品不该直接失败；一致性由 Evaluator（第 9 阶段）负责拦截
        if self.days and len(self.days) != self.trip.duration_days:
            self.warnings.append(
                f"行程天数 {len(self.days)} 与需求 {self.trip.duration_days} 天不一致"
            )
        return self
