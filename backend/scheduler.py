"""汇率定时同步。

每天 6 点（默认 Asia/Shanghai）执行一次 run_fx_agent()，并在应用启动时补跑：
服务在 6 点时没运行、或上次同步失败，启动后能自动恢复，不用人工介入。

三个必须处理的问题：
    幂等  服务重启会触发启动补偿，用 fetched_at 判断，避免重复发布
    并发  多 worker 会各自注册任务，用文件锁保证同一时刻只有一个在执行
    超时  LLM 或数据源可能挂起，用硬超时避免长期占用锁

调度器只是 run_fx_agent() 的调用方，不包含任何汇率业务逻辑。
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Literal
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from backend.config import get_settings
from backend.graph.fx_workflow import FXAgentOutcome, run_fx_agent
from backend.services.fx_service import FXService, FXStorageError, get_fx_service

logger = logging.getLogger(__name__)

JOB_ID = "fx-rate-refresh"
RETRY_JOB_ID = "fx-rate-refresh-retry"
LOCK_FILE_NAME = ".refresh.lock"

RefreshStatus = Literal[
    "ok",
    "failed",
    "timeout",
    "skipped_locked",
    "skipped_fresh",
]
AgentRunner = Callable[..., Awaitable[FXAgentOutcome]]


@dataclass(frozen=True)
class RefreshResult:
    """一次刷新尝试的结果。调度任务、启动补偿和手动接口共用。"""

    status: RefreshStatus
    outcome: FXAgentOutcome | None = None
    message: str = ""


class RefreshLock:
    """基于文件的互斥锁，只适用于单机多进程。

    多主机部署时应改用独立调度进程或外部 cron 调用 /fx/refresh。
    """

    def __init__(self, path: Path, stale_after: timedelta) -> None:
        self.path = path
        self._stale_after = stale_after
        self._held = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clear_stale()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.write(f"acquired_at={datetime.now(timezone.utc).isoformat()}\n")
        self._held = True
        return True

    def release(self) -> None:
        if not self._held:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._held = False

    def _clear_stale(self) -> None:
        """进程崩溃可能留下锁文件；超过阈值后允许接管。"""
        try:
            modified_at = datetime.fromtimestamp(
                self.path.stat().st_mtime, tz=timezone.utc
            )
        except FileNotFoundError:
            return

        if datetime.now(timezone.utc) - modified_at > self._stale_after:
            try:
                self.path.unlink()
            except FileNotFoundError:
                return
            logger.warning("已清理过期的汇率刷新锁：%s", self.path)


def _lock(service: FXService) -> RefreshLock:
    settings = get_settings()
    return RefreshLock(
        service.latest_path.parent / LOCK_FILE_NAME,
        timedelta(minutes=settings.fx_lock_stale_minutes),
    )


def refresh_due(
    service: FXService,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """判断是否需要抓取。

    用 fetched_at 而不是 business_date：ECB 周末不更新，业务日期会停在周五，
    用业务日期判断会导致每次重启都重跑。
    """
    settings = get_settings()
    latest = service.load_latest()
    if latest is None:
        return True, "还没有已发布的汇率快照"

    reference = now or datetime.now(timezone.utc)
    age = reference - latest.fetched_at
    minimum = timedelta(hours=settings.fx_min_refresh_interval_hours)
    if age < minimum:
        return False, f"距离上次抓取仅 {age}，小于阈值 {minimum}"
    return True, f"距离上次抓取已 {age}"


async def execute_refresh(
    service: FXService | None = None,
    *,
    agent_runner: AgentRunner = run_fx_agent,
    force: bool = False,
    now: datetime | None = None,
) -> RefreshResult:
    """执行一次刷新：抢锁 → 判断新鲜度 → 调 Agent → 释放锁。

    force=True 用于手动刷新，跳过新鲜度判断，但仍然需要抢锁。
    """
    settings = get_settings()
    service = service or get_fx_service()
    lock = _lock(service)

    if not lock.acquire():
        logger.info("汇率刷新被跳过：另一个任务正在执行")
        return RefreshResult(
            status="skipped_locked",
            message="另一个汇率刷新任务正在执行",
        )

    try:
        if not force:
            try:
                due, reason = refresh_due(service, now=now)
            except FXStorageError:
                # 快照损坏不应该让任务永久失败，按「需要刷新」处理，尝试覆盖它
                logger.exception("读取汇率快照失败，按需要刷新处理")
                due, reason = True, "当前快照无法读取，尝试重新抓取"
            if not due:
                logger.info("汇率刷新被跳过：%s", reason)
                return RefreshResult(status="skipped_fresh", message=reason)

        try:
            outcome = await asyncio.wait_for(
                agent_runner(service=service),
                timeout=settings.fx_job_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error(
                "汇率刷新超时（> %ss）", settings.fx_job_timeout_seconds
            )
            return RefreshResult(
                status="timeout",
                message=f"刷新超过 {settings.fx_job_timeout_seconds}s 未完成",
            )
        except Exception as exc:  # noqa: BLE001 - 调度任务不能因单次异常退出
            logger.exception("汇率刷新执行失败")
            return RefreshResult(
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )

        if not outcome.published:
            logger.warning("汇率刷新未发布：%s", outcome.summary)
            return RefreshResult(
                status="failed",
                outcome=outcome,
                message=outcome.summary,
            )

        logger.info(
            "汇率刷新成功：source=%s business_date=%s currencies=%d mode=%s",
            outcome.source,
            outcome.business_date,
            outcome.currency_count,
            outcome.mode,
        )
        return RefreshResult(status="ok", outcome=outcome, message=outcome.summary)
    finally:
        lock.release()


def _log_result(label: str, result: RefreshResult) -> None:
    if result.status == "ok":
        logger.info("%s完成：%s", label, result.message)
    elif result.status.startswith("skipped"):
        logger.info("%s跳过：%s", label, result.message)
    else:
        logger.warning("%s失败（%s）：%s", label, result.status, result.message)


def build_scheduler(service: FXService | None = None) -> AsyncIOScheduler:
    """创建调度器并注册每天一次的 cron 任务。

    不在这里 start()，由 lifespan 统一管理启动和关闭。
    """
    settings = get_settings()
    timezone_info = ZoneInfo(settings.fx_timezone)
    scheduler = AsyncIOScheduler(timezone=timezone_info)

    async def retry_job() -> None:
        result = await execute_refresh(service=service)
        _log_result("汇率刷新重试", result)

    async def refresh_job() -> None:
        result = await execute_refresh(service=service)
        _log_result("汇率定时任务", result)
        if result.status in {"failed", "timeout"}:
            retry_at = datetime.now(timezone_info) + timedelta(minutes=15)
            scheduler.add_job(
                retry_job,
                DateTrigger(run_date=retry_at),
                id=RETRY_JOB_ID,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            logger.warning("汇率刷新失败，已安排 %s 重试", retry_at.isoformat())

    scheduler.add_job(
        refresh_job,
        CronTrigger(
            hour=settings.fx_refresh_hour,
            minute=settings.fx_refresh_minute,
            timezone=timezone_info,
        ),
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    return scheduler


async def run_startup_catch_up(service: FXService | None = None) -> None:
    """启动补偿：快照过期就立即补跑一次，不阻塞应用启动。"""
    try:
        result = await execute_refresh(service=service)
    except Exception:  # noqa: BLE001 - 后台任务异常不能冒泡成未处理任务
        logger.exception("汇率启动补偿异常")
        return
    _log_result("汇率启动补偿", result)


__all__ = [
    "JOB_ID",
    "RETRY_JOB_ID",
    "RefreshLock",
    "RefreshResult",
    "build_scheduler",
    "execute_refresh",
    "refresh_due",
    "run_startup_catch_up",
]
