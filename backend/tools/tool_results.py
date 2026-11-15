"""Tool 返回值的统一序列化与错误封装。

多个工具模块共用同一套 ok / error_code / retryable 结构，
放在这里避免各写一份后字段慢慢漂移。
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def dump_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def error_result(
    error_code: str,
    message: str,
    *,
    retryable: bool = False,
    **extra: Any,
) -> str:
    return dump_payload(
        {
            "ok": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            **extra,
        }
    )


def handle_validation_error(error: Exception) -> str:
    return error_result("invalid_arguments", f"参数校验失败：{error}")


def unexpected_error(tool_name: str, error: Exception) -> str:
    logger.exception("Tool %s 执行失败", tool_name)
    return error_result(
        "tool_internal_error",
        f"{tool_name} 执行失败：{type(error).__name__}",
        retryable=True,
    )
