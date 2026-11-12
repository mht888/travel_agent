"""FastAPI 入口。

启动（在 travel_agent/ 目录下）：
    uv run uvicorn backend.main:app --reload
文档：
    http://127.0.0.1:8000/docs
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.health import router as health_router
from backend.api.trip import requirements_router
from backend.api.trip import research_router
from backend.api.trip import plan_router
from backend.api.trip import restaurant_router
from backend.api.trip import router as trip_router
from backend.config import get_settings
from backend.errors import TravelAgentError

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """应用工厂。测试里可以创建独立实例，避免状态串味。"""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="智能旅行规划 Agent（第一阶段：Mock 数据 + 基础接口）",
        version="0.1.0",
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(trip_router, prefix=settings.api_prefix)
    app.include_router(restaurant_router, prefix=settings.api_prefix)
    app.include_router(requirements_router, prefix=settings.api_prefix)
    app.include_router(research_router, prefix=settings.api_prefix)
    app.include_router(plan_router, prefix=settings.api_prefix)

    @app.exception_handler(TravelAgentError)
    async def handle_travel_agent_error(
        request: Request, exc: TravelAgentError
    ) -> JSONResponse:
        """业务异常统一成 {"error": {"code", "message"}}，前端只依赖这一种形状。"""
        logger.warning("业务异常 %s %s: %s", request.method, request.url.path, exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """兜底异常。prod 不返回细节，避免泄漏内部实现。"""
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        message = str(exc) if settings.debug else "服务器内部错误"
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": message}},
        )

    return app


app = create_app()
