"""旅行相关的 HTTP 接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from backend.graph.itinerary_workflow import run_itinerary_workflow
from backend.schemas.attraction import AttractionSearchResponse
from backend.schemas.itinerary import Itinerary
from backend.schemas.requirement import RequirementExtraction, RequirementExtractionRequest
from backend.schemas.restaurant import RestaurantSearchResponse
from backend.schemas.trip import TripRequirements
from backend.services.attraction_service import AttractionService, get_attraction_service
from backend.services.requirement_service import (
    RequirementService,
    get_requirement_service,
)
from backend.services.restaurant_service import RestaurantService, get_restaurant_service

router = APIRouter(prefix="/attractions", tags=["attractions"])
restaurant_router = APIRouter(prefix="/restaurants", tags=["restaurants"])
requirements_router = APIRouter(prefix="/trips", tags=["requirements"])
plan_router = APIRouter(prefix="/trips", tags=["planner"])


class PlanFromTextResponse(BaseModel):
    extraction: RequirementExtraction
    itinerary: Itinerary | None = None


@router.get(
    "/search",
    response_model=AttractionSearchResponse,
    summary="按城市和兴趣搜索景点",
)
async def search_attractions(
    city: Annotated[str, Query(min_length=1, description="城市中文名，如 成都、东京")],
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


@restaurant_router.get(
    "/search",
    response_model=RestaurantSearchResponse,
    summary="按城市和菜系搜索餐厅（不暴露给 LLM，只供行程/前端使用）",
)
async def search_restaurants(
    city: Annotated[str, Query(min_length=1, description="城市中文名，如 成都、东京")],
    cuisines: Annotated[
        list[str] | None,
        Query(description="菜系过滤，可重复：?cuisines=ramen&cuisines=sushi"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=30, description="返回条数上限")] = 10,
    service: RestaurantService = Depends(get_restaurant_service),
) -> RestaurantSearchResponse:
    """搜索餐厅候选。

    排序规则：菜系命中数 → 评分 → id。
    城市没有收录时返回 200 + 空 items，与景点搜索的约定一致。
    """
    city_name = city.strip()
    if not city_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="city 不能为空白",
        )

    try:
        return await service.search(city=city_name, cuisines=cuisines, limit=limit)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


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


@plan_router.post(
    "/plan",
    response_model=Itinerary,
    summary="按结构化需求生成每日行程（LLM 排程 + Python 时间轴）",
)
async def plan_trip(
    requirements: TripRequirements,
) -> Itinerary:
    """使用景点初选和真实地图数据生成行程。"""
    return await run_itinerary_workflow(requirements)


@plan_router.post(
    "/plan-from-text",
    response_model=PlanFromTextResponse,
    summary="从自然语言提取需求并生成行程",
)
async def plan_from_text(
    payload: RequirementExtractionRequest,
    service: RequirementService = Depends(get_requirement_service),
) -> PlanFromTextResponse:
    extraction = await service.extract(payload.message, previous_draft=payload.previous_draft)
    if not extraction.is_ready:
        return PlanFromTextResponse(extraction=extraction)
    itinerary = await run_itinerary_workflow(extraction.to_requirements())
    return PlanFromTextResponse(extraction=extraction, itinerary=itinerary)
