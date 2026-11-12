"""旅行相关的 HTTP 接口。

第 4 阶段新增 /trips/requirements；第 6 阶段新增 /trips/research；
第 7 阶段新增 /trips/plan（生成每日行程）。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.graph.workflow import run_research
from backend.schemas.attraction import AttractionSearchResponse
from backend.schemas.itinerary import Itinerary
from backend.schemas.requirement import RequirementExtraction, RequirementExtractionRequest
from backend.schemas.restaurant import RestaurantSearchResponse
from backend.schemas.trip import TripRequirements
from backend.services.attraction_service import AttractionService, get_attraction_service
from backend.services.itinerary_service import ItineraryService, get_itinerary_service
from backend.services.requirement_service import (
    RequirementService,
    get_requirement_service,
)
from backend.services.restaurant_service import RestaurantService, get_restaurant_service

router = APIRouter(prefix="/attractions", tags=["attractions"])
restaurant_router = APIRouter(prefix="/restaurants", tags=["restaurants"])
requirements_router = APIRouter(prefix="/trips", tags=["requirements"])
research_router = APIRouter(prefix="/trips", tags=["research"])
plan_router = APIRouter(prefix="/trips", tags=["planner"])


class ResearchResponse(BaseModel):
    """/trips/research 的响应。summary 是 Research Agent 输出的调研摘要。"""

    summary: str = Field(..., description="Research Agent 输出的调研摘要")


class ResearchFromTextResponse(BaseModel):
    """/trips/research-from-text 的响应。

    两种情况：
    - 需求完整：ready=True，summary 有值，requirements 是结构化需求。
    - 需求不完整：ready=False，summary 为 None，clarifying_questions 是要追问的问题。
    """

    ready: bool = Field(..., description="需求是否完整到可以开始调研")
    requirements: TripRequirements | None = Field(
        None, description="结构化后的旅行需求（ready=True 时才有值）"
    )
    summary: str | None = Field(None, description="调研摘要（ready=True 时才有值）")
    clarifying_questions: list[str] = Field(
        default_factory=list, description="需要追问用户的问题（ready=False 时）"
    )
    warnings: list[str] = Field(default_factory=list)


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


@restaurant_router.get(
    "/search",
    response_model=RestaurantSearchResponse,
    summary="按城市和菜系搜索餐厅（不暴露给 LLM，只供行程/前端使用）",
)
async def search_restaurants(
    city: Annotated[str, Query(min_length=1, description="城市英文名，如 Tokyo")],
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


@research_router.post(
    "/research",
    response_model=ResearchResponse,
    summary="运行 Research Agent：调用工具收集景点/路线/天气信息并输出调研摘要",
)
async def research_trip(requirements: TripRequirements) -> ResearchResponse:
    """第 6 阶段验证接口。

    传入结构化的旅行需求，Research Agent 会自主调用 search_attractions、
    get_route_matrix、get_weather 等工具，收集信息后返回一段调研摘要。

    使用方式（在 /docs 页面）：
    1. 先调用 POST /trips/requirements 提取需求，拿到 draft。
    2. 把 draft 的字段填到 TripRequirements 里（destination、duration_days 必填）。
    3. 调用 POST /trips/research，等待 Agent 完成工具调用循环。

    注意：这个接口会真实调用 LLM，需要 .env 里配置 OPENAI_API_KEY。
    """
    summary = await run_research(requirements)
    return ResearchResponse(summary=summary)


@research_router.post(
    "/research-from-text",
    response_model=ResearchFromTextResponse,
    summary="一句话输入 → 需求提取 →（完整则）调研 → 返回摘要",
)
async def research_from_text(
    payload: RequirementExtractionRequest,
    service: RequirementService = Depends(get_requirement_service),
) -> ResearchFromTextResponse:
    """串联「需求提取」和「Research Agent」两个阶段。

    流程：
    1. 用 RequirementService 从自然语言提取需求草稿。
    2. 如果草稿不完整（缺目的地或天数），直接返回追问，不调用 LLM 调研。
    3. 如果草稿完整，转成 TripRequirements，调用 run_research 跑工具调用循环。

    这样用户输入一句话就能拿到调研结果，不需要手动在两个接口之间复制字段。
    （完整的多阶段 LangGraph 工作流留到第 11 阶段，这里只是顺序调用两个 Service。）
    """
    extraction = await service.extract(
        message=payload.message,
        previous_draft=payload.previous_draft,
    )

    # 需求不完整：返回追问，不浪费 LLM 调用
    if not extraction.is_ready:
        return ResearchFromTextResponse(
            ready=False,
            clarifying_questions=extraction.clarifying_questions,
            warnings=extraction.warnings,
        )

    # 需求完整：跑 Research Agent
    requirements = extraction.to_requirements()
    summary = await run_research(requirements)
    return ResearchFromTextResponse(
        ready=True,
        requirements=requirements,
        summary=summary,
        warnings=extraction.warnings,
    )


@plan_router.post(
    "/plan",
    response_model=Itinerary,
    summary="按结构化需求生成每日行程（LLM 排程 + Python 时间轴）",
)
async def plan_trip(
    requirements: TripRequirements,
    service: ItineraryService = Depends(get_itinerary_service),
) -> Itinerary:
    """第 7 阶段验证接口。

    数据流：
        需求 → 景点候选（AttractionService）
             → 交通矩阵（RouteService）
             → 天气（WeatherService，有 start_date 才查）
             → LLM 排程（Planner Agent，失败降级规则）
             → Python 时间轴组装（itinerary_builder）
             → Itinerary

    在 /docs 里的使用顺序：
    1. POST /trips/requirements 拿到结构化草稿（可选，也可以手填）。
    2. 把草稿字段填进这个接口的 body，destination 与 duration_days 必填。
    3. 返回的 days[].items[] 就是带 start_time / end_time / 交通耗时的行程。

    注意：配置了 OPENAI_API_KEY 且填写了 planner 的 TODO 后才会真正调用 LLM；
    否则自动使用规则排程，结果里 warnings 会说明原因。
    """
    return await service.plan(requirements)
