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

    # --- 数据库（第 3 阶段接入，现在只占位，不建连接）---
    database_url: str | None = None

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
