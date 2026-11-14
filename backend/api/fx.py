"""汇率状态与手动刷新接口。

用途：
    GET  /api/fx/status   给运维和前端看当前快照、是否过期、下次执行时间
    POST /api/fx/refresh  手工或外部调度补跑；需要 FX_INTERNAL_TOKEN

手动刷新与定时任务共用同一把锁和同一套幂等判断，不会并发发布。
"""

import secrets

from fastapi import APIRouter, Header, HTTPException, Request, status

from backend.config import get_settings
from backend.graph.fx_workflow import FXAgentOutcome
from backend.scheduler import JOB_ID, execute_refresh
from backend.services.fx_service import FXStorageError, get_fx_service

router = APIRouter(prefix="/fx", tags=["fx"])


def _outcome_payload(outcome: FXAgentOutcome | None) -> dict | None:
    if outcome is None:
        return None
    return {
        "summary": outcome.summary,
        "published": outcome.published,
        "mode": outcome.mode,
        "source": outcome.source,
        "business_date": (
            outcome.business_date.isoformat() if outcome.business_date else None
        ),
        "currency_count": outcome.currency_count,
    }


@router.get("/status", summary="汇率快照状态")
def fx_status(request: Request) -> dict:
    """返回当前生效的汇率摘要。没有快照时 available=false。"""
    service = get_fx_service()
    try:
        latest = service.load_latest()
    except FXStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"汇率快照损坏：{exc}",
        ) from exc

    scheduler = getattr(request.app.state, "fx_scheduler", None)
    job = scheduler.get_job(JOB_ID) if scheduler is not None else None
    next_run_at = getattr(job, "next_run_time", None)

    if latest is None:
        return {
            "available": False,
            "scheduler_next_run_at": next_run_at.isoformat() if next_run_at else None,
        }

    return {
        "available": True,
        "base": latest.base,
        "source": latest.source,
        "status": latest.status.value,
        "fetched_at": latest.fetched_at.isoformat(),
        "business_date": latest.business_date.isoformat(),
        "currency_count": len(latest.rates),
        "stale": service.is_stale(latest),
        "scheduler_next_run_at": next_run_at.isoformat() if next_run_at else None,
    }


@router.post("/refresh", summary="手动刷新汇率")
async def fx_refresh(
    x_internal_token: str | None = Header(
        None,
        alias="X-Internal-Token",
        description="与 FX_INTERNAL_TOKEN 配置一致",
    ),
) -> dict:
    """立即执行一次汇率刷新。

    与定时任务的区别：跳过「距上次抓取不足 N 小时」的新鲜度判断，
    但仍然遵守文件锁，另一个任务执行中会返回 409。
    """
    settings = get_settings()
    if not settings.fx_internal_token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="手动刷新未启用：未配置 FX_INTERNAL_TOKEN",
        )
    if not x_internal_token or not secrets.compare_digest(
        x_internal_token, settings.fx_internal_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="内部 token 无效",
        )

    result = await execute_refresh(force=True)
    if result.status == "skipped_locked":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=result.message,
        )

    return {
        "status": result.status,
        "message": result.message,
        "outcome": _outcome_payload(result.outcome),
    }
