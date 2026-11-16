"""景点排序结果结构。

只描述「谁排在前面、为什么」，不携带景点完整字段：
完整对象留在候选池里，由 Tool 按 id 合并，避免 LLM 回吐时篡改评分、坐标等事实。

RankingItem 额外保留名称/分类/简介/来源，是为了让 finalize 能直接组装 Attraction，
不用再回查候选池；这些字段由代码从工具返回值合并进来，LLM 只负责 id。
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.schemas.attraction import AttractionCategory


class RankingItem(BaseModel):
    """单个景点的排序结果。

    score 统一归一化到 0~1。
    """

    id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0, le=1)
    reason: str = Field("", description="排序理由，可直接展示给用户")
    # 以下字段由代码从候选池合并，用于组装 Attraction；缺失时保持默认值
    name: str = Field("", description="景点名称")
    category: AttractionCategory = Field(
        AttractionCategory.OTHER, description="粗分类"
    )
    description: str = Field("", description="网页摘要里的简介")
    source_url: str | None = Field(None, description="来源网页")


class AttractionRanking(BaseModel):
    """一次排序的完整结果。"""

    ranked_ids: list[str] = Field(default_factory=list, description="按名次排列的景点 id")
    items: list[RankingItem] = Field(default_factory=list)
    strategy_used: Literal["llm"] = Field(
        "llm",
        description="实际生效的策略：只使用 llm 评分",
    )
    search_id: str | None = Field(
        None, description="候选池 id，可用于回查完整地点；过期后不可用"
    )
    by_interest: dict[str, list[RankingItem]] = Field(
        default_factory=dict, description="按兴趣分组的名次，平铺 items 只保留首次命中"
    )

    @model_validator(mode="after")
    def _sync_ranked_ids(self) -> "AttractionRanking":
        """只给了 items 时自动补 ranked_ids，避免两份顺序不一致。"""
        if not self.ranked_ids and self.items:
            self.ranked_ids = [item.id for item in self.items]
        return self
