"""汇率定时同步测试。

覆盖：
    文件锁互斥与过期接管
    新鲜度判断（用 fetched_at，不看 business_date）
    执行成功 / 跳过 / 锁冲突 / 超时 / 未发布
    cron 任务注册与执行入口

不联网、不启动调度器、不写生产数据。
"""

import asyncio
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from backend.config import get_settings
from backend.graph.fx_workflow import FXAgentOutcome
from backend.scheduler import (
    JOB_ID,
    LOCK_FILE_NAME,
    RefreshLock,
    RefreshResult,
    build_scheduler,
    execute_refresh,
    refresh_due,
)
from backend.schemas.exchange_rate import FXSnapshot
from backend.services.fx_service import FXService

CORE_CODES = ("USD", "EUR", "JPY", "CNY", "GBP", "CHF", "AUD", "CAD")


def _rates() -> dict[str, Decimal]:
    return {code: Decimal("1") for code in CORE_CODES}


def _outcome(published: bool = True) -> FXAgentOutcome:
    return FXAgentOutcome(
        summary="测试结果" if published else "未发布",
        published=published,
        mode="deterministic",
        source="test",
        business_date=date(2026, 11, 13),
        currency_count=len(CORE_CODES),
    )


def _service(tmp_path: Path) -> FXService:
    return FXService(data_root=tmp_path, providers={})


def _publish(
    service: FXService,
    *,
    fetched_at: datetime | None = None,
) -> None:
    snapshot = FXSnapshot(
        rates=_rates(),
        fetched_at=fetched_at or datetime.now(timezone.utc),
        business_date=date(2026, 11, 13),
        source="test",
    )
    service.publish(snapshot)


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch):
    current = get_settings()
    monkeypatch.setattr(current, "fx_min_refresh_interval_hours", 6.0)
    monkeypatch.setattr(current, "fx_job_timeout_seconds", 5.0)
    monkeypatch.setattr(current, "fx_lock_stale_minutes", 30)
    return current


# ---------------------------- 文件锁 ----------------------------


def test_lock_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILE_NAME
    first = RefreshLock(path, timedelta(minutes=30))
    second = RefreshLock(path, timedelta(minutes=30))

    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    assert second.acquire() is True
    second.release()
    assert not path.exists()


def test_lock_takes_over_stale_file(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILE_NAME
    path.write_text("pid=999\n", encoding="utf-8")
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
    os.utime(path, (old, old))

    lock = RefreshLock(path, timedelta(minutes=30))
    assert lock.acquire() is True
    lock.release()


# ---------------------------- 新鲜度判断 ----------------------------


def test_refresh_due_without_snapshot(tmp_path: Path, settings) -> None:
    due, reason = refresh_due(_service(tmp_path))
    assert due is True
    assert "没有" in reason


def test_refresh_due_skips_recent_snapshot(tmp_path: Path, settings) -> None:
    service = _service(tmp_path)
    _publish(service)

    due, reason = refresh_due(service)
    assert due is False
    assert "小于阈值" in reason


def test_refresh_due_when_snapshot_is_old(tmp_path: Path, settings) -> None:
    service = _service(tmp_path)
    _publish(
        service,
        fetched_at=datetime.now(timezone.utc) - timedelta(hours=10),
    )

    due, _ = refresh_due(service)
    assert due is True


# ---------------------------- 执行流程 ----------------------------


def test_execute_refresh_publishes_and_releases_lock(
    tmp_path: Path, settings
) -> None:
    service = _service(tmp_path)

    async def runner(*, service: FXService) -> FXAgentOutcome:
        return _outcome()

    result = asyncio.run(
        execute_refresh(service=service, agent_runner=runner)
    )

    assert result.status == "ok"
    assert result.outcome is not None
    assert not (service.latest_path.parent / LOCK_FILE_NAME).exists()


def test_execute_refresh_skips_when_fresh(tmp_path: Path, settings) -> None:
    service = _service(tmp_path)
    _publish(service)

    async def runner(*, service: FXService) -> FXAgentOutcome:
        raise AssertionError("快照较新时不应执行 Agent")

    result = asyncio.run(
        execute_refresh(service=service, agent_runner=runner)
    )

    assert result.status == "skipped_fresh"


def test_execute_refresh_force_ignores_freshness(
    tmp_path: Path, settings
) -> None:
    service = _service(tmp_path)
    _publish(service)

    async def runner(*, service: FXService) -> FXAgentOutcome:
        return _outcome()

    result = asyncio.run(
        execute_refresh(service=service, agent_runner=runner, force=True)
    )

    assert result.status == "ok"


def test_execute_refresh_skips_when_lock_held(
    tmp_path: Path, settings
) -> None:
    service = _service(tmp_path)
    lock = RefreshLock(
        service.latest_path.parent / LOCK_FILE_NAME,
        timedelta(minutes=30),
    )
    assert lock.acquire() is True
    try:

        async def runner(*, service: FXService) -> FXAgentOutcome:
            raise AssertionError("锁被占用时不应执行 Agent")

        result = asyncio.run(
            execute_refresh(
                service=service,
                agent_runner=runner,
                force=True,
            )
        )
    finally:
        lock.release()

    assert result.status == "skipped_locked"


def test_execute_refresh_reports_timeout(
    tmp_path: Path, settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "fx_job_timeout_seconds", 0.01)
    service = _service(tmp_path)

    async def slow_runner(*, service: FXService) -> FXAgentOutcome:
        await asyncio.sleep(0.2)
        return _outcome()

    result = asyncio.run(
        execute_refresh(service=service, agent_runner=slow_runner)
    )

    assert result.status == "timeout"
    assert not (service.latest_path.parent / LOCK_FILE_NAME).exists()


def test_execute_refresh_reports_unpublished(
    tmp_path: Path, settings
) -> None:
    service = _service(tmp_path)

    async def runner(*, service: FXService, **kwargs: Any) -> FXAgentOutcome:
        return _outcome(published=False)

    result = asyncio.run(
        execute_refresh(service=service, agent_runner=runner)
    )

    assert result.status == "failed"
    assert result.outcome is not None
    assert result.outcome.published is False


# ---------------------------- 调度器注册 ----------------------------


def test_build_scheduler_registers_daily_job(tmp_path: Path) -> None:
    scheduler = build_scheduler(service=_service(tmp_path))

    job = scheduler.get_job(JOB_ID)
    assert job is not None
    assert str(scheduler.timezone) == get_settings().fx_timezone
    # 未 start 的调度器不会执行任务，这里只验证注册成功
    assert job.id == JOB_ID


# ---------------------------- 手动刷新接口 ----------------------------


def test_status_endpoint_returns_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    import backend.api.fx as fx_api
    from backend.main import create_app

    service = _service(tmp_path)
    _publish(service)
    monkeypatch.setattr(fx_api, "get_fx_service", lambda: service)

    with TestClient(create_app()) as client:
        response = client.get("/api/fx/status")

    payload = response.json()
    assert response.status_code == 200
    assert payload["available"] is True
    assert payload["currency_count"] == len(CORE_CODES)
    assert payload["stale"] is False


def test_refresh_endpoint_is_disabled_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from backend.main import create_app

    monkeypatch.setattr(get_settings(), "fx_internal_token", None)

    with TestClient(create_app()) as client:
        response = client.post("/api/fx/refresh")

    assert response.status_code == 404


def test_refresh_endpoint_rejects_wrong_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from backend.main import create_app

    monkeypatch.setattr(get_settings(), "fx_internal_token", "secret-token")

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/fx/refresh",
            headers={"X-Internal-Token": "wrong"},
        )

    assert response.status_code == 401


def test_refresh_endpoint_returns_execution_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from backend.main import create_app

    monkeypatch.setattr(get_settings(), "fx_internal_token", "secret-token")

    async def fake_refresh(*, force: bool = False) -> RefreshResult:
        assert force is True
        return RefreshResult(status="ok", message="已刷新")

    monkeypatch.setattr("backend.api.fx.execute_refresh", fake_refresh)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/fx/refresh",
            headers={"X-Internal-Token": "secret-token"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
