"""旅行 Agent 的 LangChain Tools（第 5 阶段）。

职责边界：
    Provider  取数
    Service   参数归一化、筛选、排序、矩阵组装
    Tool      只做三件事：Pydantic 入参校验 → 调用 Service → 序列化返回值

为什么返回 JSON 字符串而不是抛异常：
    Tool 的输出会作为 ToolMessage 回灌给 LLM。抛异常会让整个 Agent 循环中断，
    LLM 也无法知道「错在哪、能不能重试」。因此统一返回：
        成功：{"ok": true, ...}
        失败：{"ok": false, "error_code": "...", "message": "...", "retryable": bool}
"""

import json
import logging
from datetime import date
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.providers.mock_provider import MockRouteProvider, MockWeatherProvider
from backend.schemas.common import TransportMode
from backend.services.attraction_service import AttractionService, get_attraction_service
from backend.services.route_service import RouteService
from backend.services.weather_service import WeatherService

logger = logging.getLogger(__name__)


# ---------------------------- 序列化与错误 ----------------------------


def _dump(payload: dict[str, Any]) -> str:
    """统一出口：JSON 字符串，Decimal / datetime 交给 default 兜底。"""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(
    error_code: str,
    message: str,
    *,
    retryable: bool = False,
    **extra: Any,
) -> str:
    """把已知错误翻译成 LLM 能读懂的失败结果。"""
    return _dump(
        {
            "ok": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            **extra,
        }
    )


def _handle_validation_error(error: Exception) -> str:
    """入参不符合 args_schema 时返回错误对象，而不是让工具调用直接抛异常。"""
    return _error("invalid_arguments", f"参数校验失败：{error}")


def _unexpected_error(tool_name: str, error: Exception) -> str:
    """外部数据源或未预期异常：记日志，返回可重试标记，不中断 Agent。"""
    logger.exception("Tool %s 执行失败", tool_name)
    return _error(
        "tool_internal_error",
        f"{tool_name} 执行失败：{type(error).__name__}",
        retryable=True,
    )


# ---------------------------- 入参 Schema ----------------------------


class SearchAttractionsInput(BaseModel):
    city: str = Field(..., min_length=1, description="城市英文名，如 Tokyo")
    interests: list[str] | None = Field(
        None,
        description=(
            "兴趣标签，可多选：food / shopping / culture / nature / nightlife。"
            "不传表示不按兴趣过滤"
        ),
    )
    limit: int = Field(10, ge=1, le=30, description="返回条数上限，建议 8~15")


class GetAttractionDetailsInput(BaseModel):
    attraction_id: str = Field(
        ..., min_length=1, description="景点 id，来自 search_attractions 的返回结果"
    )


class GetRouteMatrixInput(BaseModel):
    city: str = Field(..., min_length=1, description="城市英文名，如 Tokyo")
    attraction_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=RouteService.MAX_ATTRACTIONS,
        description=(
            f"参与计算的景点 id，至少 2 个、最多 {RouteService.MAX_ATTRACTIONS} 个；"
            "id 必须来自 search_attractions 的结果"
        ),
    )
    mode: TransportMode = Field(
        TransportMode.WALK, description="交通方式：walk / transit / drive / taxi"
    )


class GetWeatherInput(BaseModel):
    city: str = Field(..., min_length=1, description="城市英文名，如 Tokyo")
    start_date: date = Field(..., description="起始日期，ISO 格式 YYYY-MM-DD")
    days: int = Field(5, ge=1, le=WeatherService.MAX_FORECAST_DAYS, description="查询天数")


# ---------------------------- Tool 实现 ----------------------------


def _build_search_attractions(service: AttractionService) -> BaseTool:
    async def _run(
        city: str,
        interests: list[str] | None = None,
        limit: int = 10,
    ) -> str:
        try:
            response = await service.search(
                city=city,
                interests=interests,
                limit=limit,
            )
        except ValueError as exc:
            return _error("invalid_argument", str(exc))
        except Exception as exc:  # noqa: BLE001 - 外部数据源失败必须转成结果
            return _unexpected_error("search_attractions", exc)

        payload: dict[str, Any] = {
            "ok": True,
            "city": response.city,
            "interests": response.interests,
            "total": response.total,
            "items": [item.model_dump(mode="json") for item in response.items],
        }
        if response.total == 0:
            payload["hint"] = "没有匹配结果：可去掉部分 interests，或确认城市名是否正确"
        return _dump(payload)

    return StructuredTool.from_function(
        coroutine=_run,
        name="search_attractions",
        description=(
            "搜索指定城市的景点候选，按兴趣命中数和评分排序。"
            "返回景点 id、类别、评分、坐标、预计游玩时长等基础信息。"
            "后续查询详情、计算路线、排行程都必须使用这里返回的 id。"
        ),
        args_schema=SearchAttractionsInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_get_attraction_details(service: AttractionService) -> BaseTool:
    async def _run(attraction_id: str) -> str:
        try:
            attraction = await service.get_detail(attraction_id)
        except ValueError as exc:
            return _error("invalid_argument", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("get_attraction_details", exc)

        if attraction is None:
            return _error(
                "attraction_not_found",
                f"没有找到 id 为「{attraction_id}」的景点，请先调用 search_attractions 获取有效 id",
            )
        return _dump({"ok": True, "item": attraction.model_dump(mode="json")})

    return StructuredTool.from_function(
        coroutine=_run,
        name="get_attraction_details",
        description=(
            "按景点 id 获取完整详情：开放时间、预计游玩时长、门票价格、坐标、评分。"
            "排行程前需要确认开放时间和游玩时长时调用。"
        ),
        args_schema=GetAttractionDetailsInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_get_route_matrix(
    attraction_service: AttractionService,
    route_service: RouteService,
) -> BaseTool:
    async def _run(
        city: str,
        attraction_ids: list[str],
        mode: TransportMode = TransportMode.WALK,
    ) -> str:
        try:
            # 矩阵只认 Attraction 对象：Tool 负责按 id 回查，避免 LLM 自己拼坐标
            attractions = []
            missing: list[str] = []
            for attraction_id in attraction_ids:
                attraction = await attraction_service.get_detail(attraction_id)
                if attraction is None:
                    missing.append(attraction_id)
                else:
                    attractions.append(attraction)

            if missing:
                return _error(
                    "attraction_not_found",
                    "以下景点 id 不存在，请改用 search_attractions 的返回结果",
                    missing=missing,
                )

            matrix = await route_service.build_matrix(
                city=city,
                attractions=attractions,
                mode=mode,
            )
        except ValueError as exc:
            return _error("invalid_argument", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("get_route_matrix", exc)

        return _dump(
            {
                "ok": True,
                "city": matrix.city,
                "mode": matrix.mode.value,
                "attraction_ids": matrix.attraction_ids,
                "complete": matrix.is_complete,
                "entries": [
                    entry.model_dump(mode="json") for entry in matrix.entries
                ],
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="get_route_matrix",
        description=(
            "获取多个景点之间的交通耗时矩阵（有向边 from_id -> to_id）。"
            "用于排程前判断景点距离、优化当天顺序。"
            f"attraction_ids 至少 2 个、最多 {RouteService.MAX_ATTRACTIONS} 个。"
        ),
        args_schema=GetRouteMatrixInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_get_weather(service: WeatherService) -> BaseTool:
    async def _run(city: str, start_date: date, days: int = 5) -> str:
        try:
            forecast = await service.get_forecast(
                city=city,
                start_date=start_date,
                days=days,
            )
        except ValueError as exc:
            return _error("invalid_argument", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("get_weather", exc)

        return _dump(
            {
                "ok": True,
                "city": forecast.city,
                "days": [day.model_dump(mode="json") for day in forecast.days],
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="get_weather",
        description=(
            "查询城市从 start_date 起连续 days 天的天气概况（天气状况、温度区间、降水概率）。"
            "用于把户外景点安排在天气较好的日期。"
        ),
        args_schema=GetWeatherInput,
        handle_validation_error=_handle_validation_error,
    )


# ---------------------------- 组装 ----------------------------


def create_travel_tools(
    *,
    attraction_service: AttractionService | None = None,
    route_service: RouteService | None = None,
    weather_service: WeatherService | None = None,
) -> list[BaseTool]:
    """组装一组共享同一批 Service 的 Tool。

    默认全部使用 Mock Provider（第一阶段约定）；
    Agent 与测试可以注入自己的 Service，无需修改 Tool 代码。
    """
    attraction_service = attraction_service or get_attraction_service()
    route_service = route_service or RouteService(provider=MockRouteProvider())
    weather_service = weather_service or WeatherService(provider=MockWeatherProvider())

    return [
        _build_search_attractions(attraction_service),
        _build_get_attraction_details(attraction_service),
        _build_get_route_matrix(attraction_service, route_service),
        _build_get_weather(weather_service),
    ]


def get_travel_tools() -> list[BaseTool]:
    """第 6 阶段 bind_tools() 使用的默认工具集。每次返回新实例，可安全修改。"""
    return create_travel_tools()
