"""旅行相关的 HTTP 接口。

第 4 阶段新增 /trips/requirements；第 11 阶段加 /trips/plan。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.schemas.attraction import AttractionSearchResponse
from backend.schemas.requirement import RequirementExtraction, RequirementExtractionRequest
from backend.services.attraction_service import AttractionService, get_attraction_service
from backend.services.requirement_service import (
    RequirementService,
    get_requirement_service,
)

router = APIRouter(prefix="/attractions", tags=["attractions"])
requirements_router = APIRouter(prefix="/trips", tags=["requirements"])


@router.get(
    "/search",
    response_model=AttractionSearchResponse,
    summary="按城市和兴趣搜索景点",
)
async def search_attractions(
    city: Annotated[str, Query(min_length=1, description="城市英文名，如 Tokyo")],
    interests: Annotated[
        list[str] | None,
        Query(description="兴趣标签，可重复：?interests=food&interests=culture"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50, description="返回条数上限")] = 10,
    service: AttractionService = Depends(get_attraction_service),
) -> AttractionSearchResponse:
    """搜索景点候选集。

    路由层只做 HTTP 相关的事：参数清洗、调用 Service、异常翻译。
    城市没有数据时返回 200 + 空 items，方便前端展示空状态。
    """
    city_name = city.strip()
    if not city_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="city 不能为空白",
        )

    return await service.search(city=city_name, interests=interests, limit=limit)


@requirements_router.post(
    "/requirements",
    response_model=RequirementExtraction,
    summary="从自然语言中提取结构化旅行需求",
)
async def extract_requirements(
    payload: RequirementExtractionRequest,
    service: RequirementService = Depends(get_requirement_service),
) -> RequirementExtraction:
    """多轮修改时把上一轮的 draft 放回 previous_draft 即可。

    返回里始终带 missing_fields / clarifying_questions，
    前端不需要自己判断「还缺什么」。
    """
    return await service.extract(
        message=payload.message,
        previous_draft=payload.previous_draft,
    )
