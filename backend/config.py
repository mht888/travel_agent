"""应用配置。

约定：
- 所有配置只从环境变量 / .env 读取，其他模块不要直接读 os.environ。
- .env 按「当前工作目录」解析，所以启动命令要在 travel_agent/ 下执行。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。字段名即环境变量名（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 应用 ---
    app_name: str = "Travel Agent"
    app_env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = True
    api_prefix: str = "/api"

    # --- 第一阶段强制 Mock；接入真实 API 后由 .env 覆盖 ---
    mock_mode: bool = True

    # --- 外部服务凭据（第一阶段留空，mock_mode=True 时不应被读取）---
    openai_api_key: str | None = None
    google_maps_api_key: str | None = None

    # --- OpenAI（第 4 阶段：需求提取）---
    # 没填 key 时 RequirementService 自动降级为规则解析，不会阻塞开发与测试
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_timeout_seconds: float = 30.0

    # --- 百度地图 Place API（景点候选池）---
    baidu_maps_ak: str | None = None
    baidu_place_timeout_seconds: float = Field(10.0, gt=0)

    # --- 景点兴趣排序策略（A/B 对比用）---
    # rule：纯 Python 关键词 + 评分；llm：让大模型按兴趣打分。
    # 两套实现都保留，切换只改环境变量，不改代码。
    attraction_ranking_strategy: Literal["rule", "llm"] = "rule"
    attraction_ranking_timeout_seconds: float = Field(30.0, gt=0)

    # --- 数据库（第 3 阶段接入，现在只占位，不建连接）---
    database_url: str | None = None

    # --- 汇率定时同步 ---
    # 关闭后不会注册任何定时任务，也不会执行启动补偿；测试环境默认关闭
    fx_schedule_enabled: bool = True
    fx_refresh_hour: int = Field(6, ge=0, le=23)
    fx_refresh_minute: int = Field(0, ge=0, le=59)
    fx_timezone: str = "Asia/Shanghai"
    fx_job_timeout_seconds: float = Field(180.0, gt=0)
    # 距离上次抓取不足该值时跳过，避免服务重启后重复发布
    fx_min_refresh_interval_hours: float = Field(6.0, ge=0)
    # 锁文件超过该时间视为进程异常退出，可以被接管
    fx_lock_stale_minutes: int = Field(30, ge=1)
    # 手动刷新接口的 token；为空时 POST /fx/refresh 不可用
    fx_internal_token: str | None = None
    # 覆盖默认的 data/fx 目录，测试和部署用
    fx_data_dir: str | None = None

    # --- CORS ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("api_prefix")
    @classmethod
    def _normalize_api_prefix(cls, value: str) -> str:
        """把前缀统一成以 '/' 开头、不以 '/' 结尾的形式。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("api_prefix 不能为空")
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized.rstrip("/") or "/"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置。API 依赖注入与测试共享同一个实例。"""
    return Settings()
