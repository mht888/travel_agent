"""旅行相关的 HTTP 接口。

第 4 阶段新增 /trips/requirements；第 6 阶段新增 /trips/research；第 11 阶段加 /trips/plan。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.graph.workflow import run_research
from backend.schemas.attraction import AttractionSearchResponse
from backend.schemas.requirement import RequirementExtraction, RequirementExtractionRequest
from backend.schemas.trip import TripRequirements
from backend.services.attraction_service import AttractionService, get_attraction_service
from backend.services.requirement_service import (
    RequirementService,
    get_requirement_service,
)

router = APIRouter(prefix="/attractions", tags=["attractions"])
requirements_router = APIRouter(prefix="/trips", tags=["requirements"])
research_router = APIRouter(prefix="/trips", tags=["research"])


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
