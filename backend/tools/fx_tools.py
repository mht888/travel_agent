"""汇率 Agent 的 LangChain Tools。

职责边界：
    Provider  取数
    Service   校验、暂存、发布、换算
    Tool      只做三件事：参数校验 → 调用 Service → 序列化结果

为什么返回 JSON 字符串而不是抛异常：
    工具输出会回灌给 LLM。异常会中断 Agent 循环，LLM 也无法知道能否重试。
    统一返回：
        成功：{"ok": true, ...}
        失败：{"ok": false, "error_code": "...", "message": "...", "retryable": bool}

工具之间的数据传递：
    抓取到的候选快照暂存在 candidate.json，validate / publish 都读取它。
    不让 LLM 在工具参数里搬运整份汇率 JSON，既容易截断，也无法审计。
"""

import json
import logging
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.providers.fx_provider import ExchangeRateProviderError
from backend.schemas.exchange_rate import FXSnapshot, ValidationReport
from backend.services.fx_service import (
    FXService,
    FXStorageError,
    FXValidationError,
    UnknownFXProviderError,
    get_fx_service,
)

logger = logging.getLogger(__name__)


# ---------------------------- 序列化与错误 ----------------------------


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(
    error_code: str,
    message: str,
    *,
    retryable: bool = False,
    **extra: Any,
) -> str:
    return _dump(
        {
            "ok": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            **extra,
        }
    )


def _handle_validation_error(error: Exception) -> str:
    return _error("invalid_arguments", f"参数校验失败：{error}")


def _unexpected_error(tool_name: str, error: Exception) -> str:
    logger.exception("Tool %s 执行失败", tool_name)
    return _error(
        "tool_internal_error",
        f"{tool_name} 执行失败：{type(error).__name__}",
        retryable=True,
    )


def _snapshot_summary(snapshot: FXSnapshot, service: FXService) -> dict[str, Any]:
    return {
        "base": snapshot.base,
        "source": snapshot.source,
        "status": snapshot.status.value,
        "fetched_at": snapshot.fetched_at.isoformat(),
        "business_date": snapshot.business_date.isoformat(),
        "currency_count": len(snapshot.rates),
        "currencies": sorted(snapshot.rates),
        "stale": service.is_stale(snapshot),
    }


def _report_payload(report: ValidationReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "errors": report.errors,
        "warnings": report.warnings,
        "missing_codes": report.missing_codes,
        "deviations": {
            code: str(ratio)
            for code, ratio in sorted(report.deviations.items())
        },
        "previous_source": report.previous_source,
    }


# ---------------------------- 入参 Schema ----------------------------


class FetchFXRatesInput(BaseModel):
    provider: Literal["ecb", "er_api"] = Field(
        "ecb",
        description=(
            "数据源：ecb 是欧洲央行参考汇率（约 30 种，权威，每个工作日更新）；"
            "er_api 覆盖 160+ 种，用于补齐 AED/SAR/TWD/VND 等。"
        ),
    )


class NoInput(BaseModel):
    """这些工具不需要参数。"""


# ---------------------------- Tool 实现 ----------------------------


def _build_fetch_fx_rates(service: FXService) -> BaseTool:
    async def _run(provider: str = "ecb") -> str:
        try:
            snapshot = await service.fetch_candidate(provider)
        except UnknownFXProviderError as exc:
            return _error("unknown_provider", str(exc))
        except ExchangeRateProviderError as exc:
            return _error(
                "provider_unavailable",
                str(exc),
                retryable=True,
                provider=provider,
            )
        except FXStorageError as exc:
            return _error("storage_error", str(exc), retryable=True)
        except Exception as exc:  # noqa: BLE001 - 外部源失败必须转成结果
            return _unexpected_error("fetch_fx_rates", exc)

        return _dump(
            {
                "ok": True,
                "staged": _snapshot_summary(snapshot, service),
                "warnings": snapshot.warnings,
                "hint": (
                    "候选快照已暂存，尚未生效。下一步调用 validate_snapshot；"
                    "通过后再调用 publish_snapshot。"
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="fetch_fx_rates",
        description=(
            "从指定数据源抓取最新汇率，暂存为候选快照（不会影响线上汇率）。"
            "优先用 ecb；如果失败，或需要 AED/SAR/TWD/VND 等 ECB 未覆盖的币种，"
            "改用 er_api。抓取成功后必须调用 validate_snapshot 校验。"
        ),
        args_schema=FetchFXRatesInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_get_last_snapshot(service: FXService) -> BaseTool:
    async def _run() -> str:
        try:
            snapshot = service.load_latest()
        except FXStorageError as exc:
            return _error("snapshot_corrupt", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("get_last_snapshot", exc)

        if snapshot is None:
            return _error(
                "snapshot_not_found",
                "还没有已发布的汇率快照，请先执行 fetch_fx_rates",
            )

        return _dump({"ok": True, "snapshot": _snapshot_summary(snapshot, service)})

    return StructuredTool.from_function(
        coroutine=_run,
        name="get_last_snapshot",
        description=(
            "读取当前已发布的汇率快照摘要：基准币、数据源、交易日、币种列表、是否过期。"
            "用于确认线上基准，或在抓取失败时判断能否保留旧快照。"
        ),
        args_schema=NoInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_validate_snapshot(service: FXService) -> BaseTool:
    async def _run() -> str:
        try:
            candidate = service.load_candidate()
            if candidate is None:
                return _error(
                    "candidate_not_found",
                    "没有待校验的候选快照，请先调用 fetch_fx_rates",
                )
            report = service.validate(candidate)
        except FXStorageError as exc:
            return _error("storage_error", str(exc), retryable=True)
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("validate_snapshot", exc)

        payload: dict[str, Any] = {
            "ok": True,
            "ready_to_publish": report.passed,
            "report": _report_payload(report),
            "candidate": _snapshot_summary(candidate, service),
        }
        if not report.passed:
            payload["hint"] = (
                "校验未通过，禁止发布。可以换数据源重新 fetch_fx_rates，"
                "或保留旧快照等待下一次更新。"
            )
        return _dump(payload)

    return StructuredTool.from_function(
        coroutine=_run,
        name="validate_snapshot",
        description=(
            "校验暂存的候选汇率快照：核心币种是否齐全、与上一份快照的偏离是否超阈值、"
            "交易日是否回退。返回 passed / errors / warnings / deviations。"
            "passed=false 时禁止调用 publish_snapshot。"
        ),
        args_schema=NoInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_publish_snapshot(service: FXService) -> BaseTool:
    async def _run() -> str:
        try:
            candidate = service.load_candidate()
            if candidate is None:
                return _error(
                    "candidate_not_found",
                    "没有待发布的候选快照，请先调用 fetch_fx_rates",
                )
            latest_path = service.publish(candidate)
        except FXValidationError as exc:
            return _error(
                "validation_failed",
                "候选快照未通过校验，禁止发布",
                report=_report_payload(exc.report),
            )
        except FXStorageError as exc:
            return _error("storage_error", str(exc), retryable=True)
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("publish_snapshot", exc)

        return _dump(
            {
                "ok": True,
                "published": _snapshot_summary(candidate, service),
                "latest_path": str(latest_path),
                "history_path": str(
                    service.history_dir / f"{candidate.business_date.isoformat()}.json"
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="publish_snapshot",
        description=(
            "把通过校验的候选快照发布为正式汇率，写入 latest 和按日历史。"
            "发布前会再次校验；未通过直接拒绝。校验通过后调用一次即可。"
        ),
        args_schema=NoInput,
        handle_validation_error=_handle_validation_error,
    )


def _build_get_supported_currencies(service: FXService) -> BaseTool:
    async def _run() -> str:
        try:
            snapshot = service.load_latest()
        except FXStorageError as exc:
            return _error("snapshot_corrupt", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _unexpected_error("get_supported_currencies", exc)

        if snapshot is None:
            return _error(
                "snapshot_not_found",
                "还没有已发布的汇率快照，暂时无法提供可用币种",
            )

        return _dump(
            {
                "ok": True,
                "base": snapshot.base,
                "currency_count": len(snapshot.rates),
                "currencies": sorted(snapshot.rates),
                "business_date": snapshot.business_date.isoformat(),
                "source": snapshot.source,
                "stale": service.is_stale(snapshot),
            }
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="get_supported_currencies",
        description=(
            "返回当前汇率快照里实际可换算的币种列表。"
            "需求抽取和预算模块应以此为准，不要在代码里写死币种白名单。"
        ),
        args_schema=NoInput,
        handle_validation_error=_handle_validation_error,
    )


def get_fx_tools(service: FXService | None = None) -> list[BaseTool]:
    """构造汇率 Agent 的工具集。测试可注入独立 service。"""
    service = service or get_fx_service()
    return [
        _build_fetch_fx_rates(service),
        _build_get_last_snapshot(service),
        _build_validate_snapshot(service),
        _build_publish_snapshot(service),
        _build_get_supported_currencies(service),
    ]
