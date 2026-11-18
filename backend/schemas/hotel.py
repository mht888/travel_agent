"""酒店数据结构。

为什么单独建模型，不塞进 Attraction / Restaurant：
    酒店的决策字段是「星级 / 参考价 / 离景点多远」，
    景点的核心是「游玩时长 / 门票」，餐厅的核心是「菜系 / 人均」。
    硬塞进已有模型会让第 8 阶段的住宿预算和推荐理由都变含糊。

分层约定：
    HotelSummary（tools/gaode_mcp_tools.py）是给 LLM 看的工具 DTO，
    金额按 Tool 约定用字符串；Hotel 是业务层实体，金额用 Money，
    必须带 Provenance。Provider / 组装层负责 DTO → 实体。

三个关键取舍：
    1. reference_price 不是实时房价：高德只返回参考价/人均，
       统一标 DataStatus.ESTIMATED；接入 OTA 实时报价后再升级为 REAL。
    2. 距离不放 Hotel 上：同一个酒店相对不同景点距离不同，
       放进 HotelCandidate，绑定「以哪个景点为中心」的一次搜索。
    3. location 允许 None：个别 POI 缺坐标，缺坐标的酒店不能参与
       路线 / 步行距离计算，由下游服务拦截，而不是在这里丢弃数据。
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.schemas.common import GeoPoint, Money, Provenance


class HotelCategory(str, Enum):
    """粗分类。高德 typecode 由组装层映射到这里，保持分类体系稳定。"""

    HOTEL = "hotel"
    INN = "inn"
    HOSTEL = "hostel"
    RESORT = "resort"
    APARTMENT = "apartment"
    HOMESTAY = "homestay"
    OTHER = "other"


class Hotel(BaseModel):
    """统一后的酒店。字段缺失用 None，不要用 0 或空字符串冒充「有数据」。"""

    id: str = Field(..., min_length=1, description="稳定 ID（高德 POI id），用于去重和引用")
    name: str = Field(..., min_length=1)
    city: str
    district: str | None = Field(None, description="区县，如 青秀区")
    address: str | None = None
    business_area: str | None = Field(
        None, description="商圈，如 朝阳广场；用于按区域聚合和解释推荐理由"
    )
    location: GeoPoint | None = None
    category: HotelCategory = HotelCategory.OTHER
    tags: list[str] = Field(
        default_factory=list,
        description="细分标签，如 五星级 / 经济型 / 民宿；来源是高德 type 字段",
    )
    star_rating: int | None = Field(
        None, ge=1, le=5, description="挂牌星级；经济型酒店和民宿通常没有"
    )
    rating: float | None = Field(None, ge=0, le=5, description="用户评分")
    review_count: int | None = Field(None, ge=0)
    reference_price: Money | None = Field(
        None,
        description="参考价 / 人均价格，不是实时房价；未知用 None，不按 0 计算",
    )
    tel: str | None = None
    photo_urls: list[str] = Field(default_factory=list)
    provenance: Provenance

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        """小写 + 去空白 + 去重，避免「五星级」和「五星级宾馆」重复计数。"""
        seen: set[str] = set()
        normalized: list[str] = []
        for tag in value:
            clean = tag.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)
        return normalized

    @field_validator("photo_urls")
    @classmethod
    def _clean_photo_urls(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for url in value:
            item = url.strip()
            if item and item not in seen:
                seen.add(item)
                cleaned.append(item)
        return cleaned


class HotelCandidate(BaseModel):
    """一次「以某个景点为中心」搜索命中的酒店。

    距离只对本次搜索的中心点有意义，所以不放进 Hotel 本体；
    后续多景点搜索会产生多个候选，由编排层按景点聚合。
    """

    hotel: Hotel
    distance_meters: int | None = Field(
        None, ge=0, description="与中心景点的直线距离（米）；数据源未返回时为 None"
    )


class HotelSearchResponse(BaseModel):
    """搜索响应。回显查询条件，方便前端和日志对齐。"""

    city: str
    anchor_attraction_id: str = Field(..., description="以哪个入选景点为中心搜索")
    anchor_attraction_name: str | None = None
    radius_meters: int = Field(2000, ge=100, le=50000)
    total: int = Field(..., ge=0, description="实际返回条数，等于 len(items)")
    items: list[HotelCandidate] = Field(default_factory=list)
