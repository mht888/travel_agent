"""pytest 公共配置。

测试里默认关闭汇率定时任务：
TestClient 会触发 FastAPI lifespan，如果调度器开着，启动补偿会真的抓汇率、
甚至写生产数据。必须在任何 backend 模块导入前把开关关掉。
"""

import os

os.environ["FX_SCHEDULE_ENABLED"] = "false"

from backend.config import get_settings  # noqa: E402

# 防止之前导入过 config 时沿用旧配置
get_settings.cache_clear()
