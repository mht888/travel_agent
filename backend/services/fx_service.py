"""汇率快照的校验、存储与换算。

职责：
    - 拉取：调用 Provider，把候选快照暂存，不直接生效
    - 校验：核心币种、时间回退、偏离阈值
    - 发布：通过校验后原子写入 latest + 按日历史
    - 消费：给预算和需求抽取提供可用币种与换算能力

为什么先暂存再发布：
    Agent 需要在 fetch 和 publish 之间做判断；直接覆盖 latest 会让一份坏数据
    立刻影响线上预算。candidate.json 是唯一暂存区，latest.json 是生效区。

当前没有数据库，快照落在 data/fx/ 下；接入 DATABASE_URL 后只需替换读写实现。
"""

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from backend.providers.fx_provider import (
    ExchangeRateProvider,
    ExchangeRateProviderError,
    build_provider,
)
from backend.schemas.exchange_rate import FXSnapshot, ValidationReport

logger = logging.getLogger(__name__)

# 硬性核心币种：缺失任意一个就阻断发布；其余币种只告警，不影响发布。
DEFAULT_REQUIRED_CODES: frozenset[str] = frozenset(
    {"USD", "EUR", "JPY", "CNY", "GBP", "CHF", "AUD", "CAD"}
)

# 与上一份快照比较：超过 5% 告警，超过 15% 视为数据源异常并阻断。
DEVIATION_WARN = Decimal("0.05")
DEVIATION_BLOCK = Decimal("0.15")

# 超过 7 天未更新的快照不再参与换算，避免用过期汇率做预算。
SNAPSHOT_MAX_AGE_DAYS = 7

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "fx"


class FXStorageError(RuntimeError):
    """快照文件缺失、损坏或无法写入。"""


class FXValidationError(RuntimeError):
    """候选快照未通过校验。异常携带完整报告，便于工具层原样返回。"""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__("汇率快照未通过校验，禁止发布")
        self.report = report


class FXRateUnavailableError(RuntimeError):
    """没有可用汇率，或目标币种不在当前快照里。"""


class UnknownFXProviderError(RuntimeError):
    """请求了未注册的数据源。"""


def _snapshot_payload(snapshot: FXSnapshot) -> dict[str, Any]:
    """显式序列化：Decimal 一律转字符串，避免浮点精度丢失。"""
    return {
        "base": snapshot.base,
        "rates": {
            code: str(rate) for code, rate in sorted(snapshot.rates.items())
        },
        "fetched_at": snapshot.fetched_at.isoformat(),
        "business_date": snapshot.business_date.isoformat(),
        "source": snapshot.source,
        "status": snapshot.status.value,
        "warnings": list(snapshot.warnings),
    }


def _snapshot_from_payload(payload: dict[str, Any]) -> FXSnapshot:
    return FXSnapshot(
        base=payload["base"],
        rates={
            code: Decimal(str(rate))
            for code, rate in (payload.get("rates") or {}).items()
        },
        fetched_at=datetime.fromisoformat(payload["fetched_at"]),
        business_date=date.fromisoformat(payload["business_date"]),
        source=payload["source"],
        status=payload.get("status", "real"),
        warnings=list(payload.get("warnings") or []),
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """先写临时文件再替换，避免进程中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except OSError as exc:
        raise FXStorageError(f"写入汇率文件失败：{path}：{exc}") from exc


class FXService:
    """汇率快照的读写、校验和换算。"""

    def __init__(
        self,
        *,
        data_root: Path | None = None,
        providers: Mapping[str, ExchangeRateProvider] | None = None,
    ) -> None:
        self._root = Path(data_root) if data_root is not None else _DEFAULT_DATA_ROOT
        self._providers = (
            dict(providers)
            if providers is not None
            else {"ecb": build_provider("ecb"), "er_api": build_provider("er_api")}
        )

    # ---------------------------- 路径与读取 ----------------------------

    @property
    def available_providers(self) -> list[str]:
        return sorted(self._providers)

    @property
    def latest_path(self) -> Path:
        return self._root / "latest.json"

    @property
    def candidate_path(self) -> Path:
        return self._root / "candidate.json"

    @property
    def history_dir(self) -> Path:
        return self._root / "history"

    def _read(self, path: Path) -> FXSnapshot | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FXStorageError(f"读取汇率文件失败：{path}：{exc}") from exc
        if not isinstance(payload, dict):
            raise FXStorageError(f"汇率文件格式非法：{path}")
        try:
            return _snapshot_from_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise FXStorageError(f"汇率文件内容非法：{path}：{exc}") from exc

    def load_latest(self) -> FXSnapshot | None:
        """读取已发布的快照；从未发布过返回 None。"""
        return self._read(self.latest_path)

    def load_candidate(self) -> FXSnapshot | None:
        """读取暂存的候选快照；还没抓取过返回 None。"""
        return self._read(self.candidate_path)

    def save_candidate(self, snapshot: FXSnapshot) -> Path:
        """暂存候选快照。只影响 candidate.json，不影响线上使用的 latest.json。"""
        _atomic_write(self.candidate_path, _snapshot_payload(snapshot))
        return self.candidate_path

    # ---------------------------- 拉取 ----------------------------

    async def fetch_candidate(self, provider_name: str) -> FXSnapshot:
        """调用指定数据源抓取并暂存。数据源错误原样上抛，由工具层降级。"""
        key = provider_name.strip().lower()
        provider = self._providers.get(key)
        if provider is None:
            allowed = ", ".join(self.available_providers)
            raise UnknownFXProviderError(
                f"未知汇率数据源 {provider_name!r}，可选：{allowed}"
            )
        snapshot = await provider.fetch_latest("USD")
        self.save_candidate(snapshot)
        logger.info(
            "汇率候选已暂存：source=%s business_date=%s currencies=%d",
            snapshot.source,
            snapshot.business_date,
            len(snapshot.rates),
        )
        return snapshot

    # ---------------------------- 校验 ----------------------------

    def validate(
        self,
        snapshot: FXSnapshot,
        *,
        previous: FXSnapshot | None = None,
        required_codes: frozenset[str] | set[str] | None = None,
    ) -> ValidationReport:
        """候选快照的硬校验。

        不通过的条件：
            - 缺少硬性核心币种
            - business_date 比上一份更早（时间回退）
            - 单币种偏离超过阻断阈值
            - 币种数量异常少
        """
        previous = previous if previous is not None else self.load_latest()
        required = {
            code.strip().upper()
            for code in (required_codes or DEFAULT_REQUIRED_CODES)
        }

        errors: list[str] = []
        warnings: list[str] = list(snapshot.warnings)
        deviations: dict[str, Decimal] = {}

        missing_codes = sorted(required - snapshot.rates.keys())
        if missing_codes:
            errors.append(f"缺少核心币种：{', '.join(missing_codes)}")

        if previous is None:
            warnings.append("没有历史快照，已跳过偏离校验")
        else:
            if snapshot.business_date < previous.business_date:
                errors.append(
                    "business_date 回退："
                    f"{snapshot.business_date} < {previous.business_date}"
                )

            for code in sorted(snapshot.rates.keys() & previous.rates.keys()):
                previous_rate = previous.rates[code]
                if previous_rate <= 0:
                    continue
                ratio = (snapshot.rates[code] - previous_rate) / previous_rate
                absolute = abs(ratio).quantize(Decimal("0.0001"))
                deviations[code] = absolute
                if absolute > DEVIATION_BLOCK:
                    errors.append(
                        f"{code} 偏离 {absolute:.2%}，超过阻断阈值 "
                        f"{DEVIATION_BLOCK:.0%}"
                    )
                elif absolute > DEVIATION_WARN:
                    warnings.append(
                        f"{code} 偏离 {absolute:.2%}，超过告警阈值 "
                        f"{DEVIATION_WARN:.0%}"
                    )

            added = sorted(snapshot.rates.keys() - previous.rates.keys())
            removed = sorted(previous.rates.keys() - snapshot.rates.keys())
            if added:
                warnings.append(f"新增币种：{', '.join(added)}")
            if removed:
                warnings.append(f"币种减少：{', '.join(removed)}")

        if len(snapshot.rates) < 5:
            errors.append(
                f"币种数量过少（{len(snapshot.rates)}），疑似数据源异常"
            )

        return ValidationReport(
            passed=not errors,
            errors=errors,
            warnings=warnings,
            missing_codes=missing_codes,
            deviations=deviations,
            previous_source=previous.source if previous else None,
        )

    # ---------------------------- 发布 ----------------------------

    def publish(self, snapshot: FXSnapshot) -> Path:
        """发布快照：再次校验，通过后写历史与 latest，返回 latest 路径。"""
        report = self.validate(snapshot)
        if not report.passed:
            raise FXValidationError(report)

        payload = _snapshot_payload(snapshot)
        history_path = self.history_dir / f"{snapshot.business_date.isoformat()}.json"
        _atomic_write(history_path, payload)
        _atomic_write(self.latest_path, payload)

        logger.info(
            "汇率已发布：source=%s business_date=%s currencies=%d",
            snapshot.source,
            snapshot.business_date,
            len(snapshot.rates),
        )
        return self.latest_path

    # ---------------------------- 消费 ----------------------------

    def supported_currencies(self) -> list[str]:
        """当前可换算的币种；还没有发布快照时返回空列表。"""
        snapshot = self.load_latest()
        return sorted(snapshot.rates) if snapshot else []

    def is_stale(
        self,
        snapshot: FXSnapshot,
        *,
        now: datetime | None = None,
        max_age_days: int = SNAPSHOT_MAX_AGE_DAYS,
    ) -> bool:
        """按抓取时间判断是否过期。业务日期由快照自身表达，这里只看数据新鲜度。"""
        reference = now or datetime.now(timezone.utc)
        return reference - snapshot.fetched_at > timedelta(days=max_age_days)

    def convert(
        self,
        amount: Decimal | int | float | str,
        from_currency: str,
        to_currency: str,
    ) -> Decimal:
        """按美元三角换算：amount / rate[from] * rate[to]。

        任何中间结果都不量化，调用方在展示时再按币种精度处理。
        """
        snapshot = self.load_latest()
        if snapshot is None:
            raise FXRateUnavailableError("没有已发布的汇率快照，无法换算")

        source = from_currency.strip().upper()
        target = to_currency.strip().upper()
        missing = [
            code for code in (source, target) if code not in snapshot.rates
        ]
        if missing:
            raise FXRateUnavailableError(
                f"当前汇率快照缺少币种：{', '.join(missing)}"
            )
        if self.is_stale(snapshot):
            raise FXRateUnavailableError(
                f"汇率快照已超过 {SNAPSHOT_MAX_AGE_DAYS} 天未更新，拒绝换算"
            )

        value = Decimal(str(amount))
        return value / snapshot.rates[source] * snapshot.rates[target]


@lru_cache(maxsize=1)
def get_fx_service() -> FXService:
    """进程内单例。数据目录可由 FX_DATA_DIR 覆盖，便于测试和部署。"""
    from backend.config import get_settings

    settings = get_settings()
    data_root = Path(settings.fx_data_dir) if settings.fx_data_dir else None
    return FXService(data_root=data_root)


__all__ = [
    "DEVIATION_BLOCK",
    "DEVIATION_WARN",
    "DEFAULT_REQUIRED_CODES",
    "FXRateUnavailableError",
    "FXService",
    "FXStorageError",
    "FXValidationError",
    "SNAPSHOT_MAX_AGE_DAYS",
    "UnknownFXProviderError",
    "ExchangeRateProviderError",
    "get_fx_service",
]
