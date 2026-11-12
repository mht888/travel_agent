"""LLM 排程的输出契约（第 7 阶段）。

这个文件定义「LLM 被允许说什么」，不是「系统最终交付什么」：
    ItineraryDraft   LLM 输出：每天去哪些景点（只有 id 和顺序）
    Itinerary        系统输出：带时间轴、交通、费用的完整行程

为什么 LLM 只输出 id：
    时间轴依赖交通矩阵，属于确定性计算，必须由 Python 完成。
    如果让 LLM 直接写 "09:00-11:00 浅草寺"，它会在第 9 阶段 Evaluator
    里被自己的算术错误反复击穿，而且结果无法复现、无法测试。
"""

from pydantic import BaseModel, Field, field_validator, model_validator


class DayPlan(BaseModel):
    """LLM 对某一天给出的安排。只有景点 id，没有时间。"""

    day_index: int = Field(..., ge=1, description="第几天，从 1 开始连续")
    attraction_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="当天按游玩顺序排列的景点 id，必须来自候选列表",
    )
    theme: str | None = Field(None, description="当天主题，如「浅草老城与下町美食」")

    @field_validator("attraction_ids")
    @classmethod
    def _no_duplicate_within_day(cls, value: list[str]) -> list[str]:
        """同一天内不允许重复景点。这是机械校验，直接给在这里。"""
        if len(set(value)) != len(value):
            raise ValueError(f"同一天内景点 id 不允许重复：{value}")
        return value


class ItineraryDraft(BaseModel):
    """LLM 的完整排程草稿。由 with_structured_output 直接生成。"""

    days: list[DayPlan] = Field(..., description="每天一条，day_index 必须连续")
    rationale: str | None = Field(None, description="整体排程理由，可选")

    @model_validator(mode="after")
    def _validate_day_sequence(self) -> "ItineraryDraft":
        """校验 day_index 连续 + 跨天不重复景点。

        LLM 最常见的两类排错都会在这里被拦下，从而触发 planner_agent 的降级：
        1. day_index 跳号或重复（如 1,3,3）。
        2. 同一个景点被安排进两天。
        """
        if not self.days:
            raise ValueError("days 不能为空")

        expected = list(range(1, len(self.days) + 1))
        actual = [day.day_index for day in self.days]
        if actual != expected:
            raise ValueError(
                f"day_index 必须从 1 连续递增，期望 {expected}，实际 {actual}"
            )

        seen: dict[str, int] = {}
        for day in self.days:
            for attraction_id in day.attraction_ids:
                if attraction_id in seen:
                    raise ValueError(
                        f"景点 {attraction_id} 同时出现在第 {seen[attraction_id]} 天"
                        f"和第 {day.day_index} 天，每个景点只能安排一次"
                    )
                seen[attraction_id] = day.day_index

        return self
