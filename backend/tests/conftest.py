"""pytest 公共配置。

测试里默认关闭汇率定时任务：
TestClient 会触发 FastAPI lifespan，如果调度器开着，启动补偿会真的抓汇率、
甚至写生产数据。必须在任何 backend 模块导入前把开关关掉。
"""

import os
from datetime import datetime, timezone

import pytest

os.environ["FX_SCHEDULE_ENABLED"] = "false"

from backend.config import get_settings  # noqa: E402

# 防止之前导入过 config 时沿用旧配置
get_settings.cache_clear()


@pytest.fixture
def budget_fx_file(tmp_path, monkeypatch):
    """用固定测试汇率写入临时快照，避免预算测试依赖同步时间。"""
    from backend.schemas.exchange_rate import FXSnapshot
    from backend.services import budget_service
    from backend.services.fx_service import FXService

    now = datetime.now(timezone.utc)
    snapshot = FXSnapshot(
        rates={"USD": "1", "CNY": "7", "JPY": "150", "EUR": "0.9"},
        fetched_at=now,
        business_date=now.date(),
        source="test",
    )
    service = FXService(data_root=tmp_path)
    service.latest_path.write_text(snapshot.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(budget_service, "get_fx_service", lambda: service)
    return service.latest_path
