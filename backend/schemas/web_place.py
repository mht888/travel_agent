"""网页抽取出的轻量地点候选。

只保留排序需要的语义字段：名称、介绍、城市、分类、来源。
坐标、营业时间、门票等事实数据要等入选之后再补，
避免对每个候选都提前调用地理编码。
"""

from pydantic import BaseModel, Field

from backend.schemas.attraction import AttractionCategory


class ExtractedPlace(BaseModel):
    """从网页摘要里抽出的地点，尚未补全为 Attraction。"""

    id: str = Field(..., min_length=1, description="稳定 ID，用于去重与排序引用")
    name: str = Field(..., min_length=1, description="地点常用名")
    description: str = Field("", description="网页摘要里的介绍，截断后展示")
    city: str = Field("", description="所属城市")
    category: AttractionCategory = Field(
        AttractionCategory.OTHER,
        description="粗分类，抽取时顺带给出",
    )
    source_url: str | None = Field(None, description="来源网页")
