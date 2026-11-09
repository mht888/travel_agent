"""健康检查接口。本地联调、容器探针、前端启动自检都用它。"""

from fastapi import APIRouter, Depends

from backend.config import Settings, get_settings
from backend.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """返回进程存活状态与当前运行模式。

    后续接入数据库 / 外部 API 后，在这里追加各依赖的 readiness 结果。
    """
    return HealthResponse(
        status="ok",
        env=settings.app_env,
        mock_mode=settings.mock_mode,
    )
