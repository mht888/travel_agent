"""景点排序结果结构。

只描述「谁排在前面、为什么」，不携带景点完整字段：
完整对象留在候选池里，由 Tool 按 id 合并，避免 LLM 回吐时篡改评分、坐标等事实。
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RankingItem(BaseModel):
    """单个景点的排序结果。

    score 统一归一化到 0~1，rule 与 llm 两套实现的分数才具有可比性。
    """

    id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0, le=1)
    reason: str = Field("", description="排序理由，可直接展示给用户")


class AttractionRanking(BaseModel):
    """一次排序的完整结果。"""

    ranked_ids: list[str] = Field(default_factory=list, description="按名次排列的景点 id")
    items: list[RankingItem] = Field(default_factory=list)
    strategy_used: Literal["rule", "llm", "rule_fallback"] = Field(
        "rule",
        description="实际生效的策略：rule 纯规则 / llm 大模型 / rule_fallback 大模型失败后回退",
    )

    @model_validator(mode="after")
    def _sync_ranked_ids(self) -> "AttractionRanking":
        """只给了 items 时自动补 ranked_ids，避免两份顺序不一致。"""
        if not self.ranked_ids and self.items:
            self.ranked_ids = [item.id for item in self.items]
        return self
