"""Schema 契约测试。

目的不是「覆盖率」，而是证明第 2 阶段新增的约束真的会拦住脏数据：
币种、时区、时间顺序、矩阵引用、预算汇总、草稿完整性。
"""

from datetime import datetime, time, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.schemas import (
    BudgetBreakdown,
    BudgetCategory,
    BudgetLineItem,
    DataStatus,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
    MissingField,
    Money,
    OpeningInterval,
    Provenance,
    RequirementExtraction,
    RouteMatrix,
    RouteMatrixEntry,
    TransportMode,
    TripRequirementDraft,
    TripRequirements,
)

_FETCHED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _provenance() -> Provenance:
    return Provenance(
        source="test:fixture", fetched_at=_FETCHED_AT, status=DataStatus.MOCK
    )


def _money(
    amount: str,
    currency: str = "USD",
    status: DataStatus = DataStatus.ESTIMATED,
) -> Money:
    return Money(amount=Decimal(amount), currency=currency, status=status)


def _item(
    seq: int,
    start: time,
    end: time,
    cost: Money | None = None,
) -> ItineraryItem:
    return ItineraryItem(
        seq=seq,
        type=ItineraryItemType.ATTRACTION,
        name=f"P{seq}",
        start_time=start,
        end_time=end,
        cost=cost,
    )


def _route_entry(
    from_id: str,
    to_id: str,
    mode: TransportMode = TransportMode.WALK,
    duration_minutes: int = 12,
) -> RouteMatrixEntry:
    return RouteMatrixEntry(
        from_id=from_id,
        to_id=to_id,
        mode=mode,
        duration_minutes=duration_minutes,
        distance_meters=900,
        provenance=_provenance(),
    )


# ---------------------------- 公共类型 ----------------------------


def test_money_rejects_unsupported_currency() -> None:
    with pytest.raises(ValidationError, match="暂不支持币种"):
        Money(amount=10, currency="GBP")


def test_money_normalizes_currency_case_and_space() -> None:
    assert Money(amount=10, currency=" usd ").currency == "USD"


def test_provenance_rejects_naive_datetime() -> None:
    """不带时区的时间会和 valid_until 比错，必须在入口拦住。"""
    with pytest.raises(ValidationError, match="时区"):
        Provenance(source="test:fixture", fetched_at=datetime(2026, 1, 1))


def test_opening_interval_rejects_cross_midnight() -> None:
    with pytest.raises(ValidationError, match="closes_at"):
        OpeningInterval(weekday=0, opens_at=time(22, 0), closes_at=time(2, 0))


# ---------------------------- 行程 ----------------------------


def test_itinerary_item_rejects_reversed_time() -> None:
    with pytest.raises(ValidationError, match="end_time"):
        _item(1, time(10, 0), time(9, 30))


def test_itinerary_day_rejects_non_continuous_seq() -> None:
    with pytest.raises(ValidationError, match="seq"):
        ItineraryDay(
            day_index=1,
            items=[_item(1, time(9, 0), time(10, 0)), _item(3, time(11, 0), time(12, 0))],
        )


def test_itinerary_day_rejects_time_overlap() -> None:
    with pytest.raises(ValidationError, match="结束于"):
        ItineraryDay(
            day_index=1,
            items=[
                _item(1, time(9, 0), time(10, 30)),
                _item(2, time(10, 0), time(11, 0)),
            ],
        )


def test_itinerary_day_daily_cost_rejects_mixed_currency() -> None:
    day = ItineraryDay(
        day_index=1,
        items=[
            _item(1, time(9, 0), time(10, 0), _money("10", currency="USD")),
            _item(2, time(11, 0), time(12, 0), _money("100", currency="JPY")),
        ],
    )
    with pytest.raises(ValueError, match="多种币种"):
        _ = day.daily_cost


# ---------------------------- 路线矩阵 ----------------------------


def test_route_matrix_lookup_and_completeness() -> None:
    matrix = RouteMatrix(
        city="Tokyo",
        mode=TransportMode.WALK,
        attraction_ids=["a", "b"],
        entries=[
            _route_entry("a", "b"),
            _route_entry("b", "a", duration_minutes=15),
        ],
    )

    assert matrix.is_complete is True
    assert matrix.duration_between("a", "b") == 12
    assert matrix.duration_between("b", "a") == 15


def test_route_matrix_allows_sparse_edges() -> None:
    """真实 API 查不到某条边时返回 None，不要为了凑完整而伪造数据。"""
    matrix = RouteMatrix(
        city="Tokyo",
        mode=TransportMode.WALK,
        attraction_ids=["a", "b"],
        entries=[_route_entry("a", "b")],
    )

    assert matrix.is_complete is False
    assert matrix.duration_between("b", "a") is None
    with pytest.raises(KeyError):
        matrix.duration_between("a", "c")


def test_route_matrix_rejects_unknown_id() -> None:
    with pytest.raises(ValidationError, match="未声明"):
        RouteMatrix(
            city="Tokyo",
            mode=TransportMode.WALK,
            attraction_ids=["a"],
            entries=[_route_entry("a", "b")],
        )


def test_route_matrix_rejects_duplicate_edge() -> None:
    with pytest.raises(ValidationError, match="重复"):
        RouteMatrix(
            city="Tokyo",
            mode=TransportMode.WALK,
            attraction_ids=["a", "b"],
            entries=[_route_entry("a", "b"), _route_entry("a", "b")],
        )


def test_route_matrix_rejects_mixed_mode() -> None:
    with pytest.raises(ValidationError, match="交通方式"):
        RouteMatrix(
            city="Tokyo",
            mode=TransportMode.WALK,
            attraction_ids=["a", "b"],
            entries=[_route_entry("a", "b", mode=TransportMode.DRIVE)],
        )


def test_route_matrix_rejects_self_loop() -> None:
    with pytest.raises(ValidationError, match="自环"):
        RouteMatrix(
            city="Tokyo",
            mode=TransportMode.WALK,
            attraction_ids=["a"],
            entries=[_route_entry("a", "a")],
        )


# ---------------------------- 预算 ----------------------------


def test_budget_total_and_remaining_are_derived() -> None:
    breakdown = BudgetBreakdown(
        lines=[
            BudgetLineItem(
                category=BudgetCategory.FLIGHTS,
                label="往返机票（估算）",
                amount=_money("420"),
            ),
            BudgetLineItem(
                category=BudgetCategory.FOOD,
                label="餐饮（估算）",
                amount=_money("560"),
            ),
        ],
        budget_limit=_money("1000", status=DataStatus.USER_INPUT),
    )

    assert breakdown.currency == "USD"
    assert breakdown.total is not None and breakdown.total.amount == Decimal("980")
    assert breakdown.remaining is not None and breakdown.remaining.amount == Decimal("20")
    assert breakdown.over_budget is False
    assert breakdown.over_budget_amount is None

    # 推导字段必须出现在对外 JSON 里，前端不用自己算
    payload = breakdown.model_dump(mode="json")
    assert payload["total"]["amount"] == "980"
    assert payload["remaining"]["amount"] == "20"


def test_budget_marks_over_budget_without_negative_money() -> None:
    breakdown = BudgetBreakdown(
        lines=[
            BudgetLineItem(
                category=BudgetCategory.FLIGHTS,
                label="机票",
                amount=_money("420"),
            ),
            BudgetLineItem(
                category=BudgetCategory.FOOD,
                label="餐饮",
                amount=_money("560"),
            ),
        ],
        budget_limit=_money("900", status=DataStatus.USER_INPUT),
    )

    assert breakdown.over_budget is True
    assert breakdown.remaining is None
    assert breakdown.over_budget_amount is not None
    assert breakdown.over_budget_amount.amount == Decimal("80")


def test_budget_rejects_mixed_currency() -> None:
    with pytest.raises(ValidationError, match="多种币种"):
        BudgetBreakdown(
            lines=[
                BudgetLineItem(
                    category=BudgetCategory.FLIGHTS,
                    label="机票",
                    amount=_money("420", currency="USD"),
                ),
                BudgetLineItem(
                    category=BudgetCategory.FOOD,
                    label="餐饮",
                    amount=_money("12000", currency="JPY"),
                ),
            ]
        )


def test_budget_rejects_limit_currency_mismatch() -> None:
    with pytest.raises(ValidationError, match="不一致"):
        BudgetBreakdown(
            lines=[
                BudgetLineItem(
                    category=BudgetCategory.TRANSPORT,
                    label="交通",
                    amount=_money("60", currency="USD"),
                )
            ],
            budget_limit=_money("5000", currency="JPY", status=DataStatus.USER_INPUT),
        )


def test_budget_without_lines_has_no_total() -> None:
    breakdown = BudgetBreakdown(budget_limit=_money("1000", status=DataStatus.USER_INPUT))

    assert breakdown.currency == "USD"
    assert breakdown.total is None
    assert breakdown.remaining is None
    assert breakdown.over_budget is False


# ---------------------------- 需求草稿 ----------------------------


def test_draft_reports_missing_fields_in_stable_order() -> None:
    draft = TripRequirementDraft(destination="Tokyo")

    assert draft.missing_fields == [
        MissingField.DURATION_DAYS,
        MissingField.START_DATE,
        MissingField.BUDGET,
    ]
    assert draft.blocking_missing_fields == [MissingField.DURATION_DAYS]
    assert draft.is_ready is False

    with pytest.raises(ValueError, match="duration_days"):
        draft.to_requirements()


def test_draft_converts_when_blocking_fields_present() -> None:
    draft = TripRequirementDraft(
        destination=" Tokyo ",
        duration_days=5,
        budget=_money("1000", status=DataStatus.USER_INPUT),
        interests=["Food", "food", "SHOPPING", " "],
    )

    requirements = draft.to_requirements()

    assert isinstance(requirements, TripRequirements)
    assert requirements.destination == "Tokyo"
    assert requirements.interests == ["food", "shopping"]
    assert requirements.budget is not None
    assert requirements.budget.amount == Decimal("1000")


def test_draft_rejects_blank_destination() -> None:
    with pytest.raises(ValidationError, match="destination"):
        TripRequirementDraft(destination="   ")


def test_requirement_extraction_serializes_missing_fields() -> None:
    extraction = RequirementExtraction(
        draft=TripRequirementDraft(destination="Tokyo"),
        clarifying_questions=["计划玩几天？"],
    )

    assert extraction.missing_fields == [
        MissingField.DURATION_DAYS,
        MissingField.START_DATE,
        MissingField.BUDGET,
    ]
    assert extraction.is_ready is False

    payload = extraction.model_dump(mode="json")
    assert payload["missing_fields"] == ["duration_days", "start_date", "budget"]
    assert payload["is_ready"] is False
