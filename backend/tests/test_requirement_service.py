"""需求提取测试（第 4 阶段）。

全程不联网：
    - 规则解析器直接跑真实逻辑；
    - LLM 路径用假客户端，只验证「结构化输出 → Patch」和失败降级。
"""

import asyncio
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agents.requirement_agent import (
    RequirementAgent,
    RuleBasedRequirementExtractor,
)
from backend.config import get_settings
from backend.main import app
from backend.schemas import (
    DataStatus,
    MissingField,
    Money,
    TripRequirementDraft,
    TravelStyle,
)
from backend.services.requirement_service import (
    RequirementService,
    get_requirement_service,
)

_TOKYO_SENTENCE = (
    "帮我规划一次日本东京 5 日游，预算 1000 美元，"
    "喜欢美食、购物和文化景点，希望行程不要太紧凑"
)


def _service() -> RequirementService:
    return RequirementService(extractor=RuleBasedRequirementExtractor())


def _extract(
    message: str,
    previous: TripRequirementDraft | None = None,
    today: date | None = None,
):
    return asyncio.run(_service().extract(message, previous, today or date(2026, 1, 1)))


# ---------------------------- 规则解析 ----------------------------


def test_extracts_full_requirement_sentence() -> None:
    extraction = _extract(_TOKYO_SENTENCE)
    draft = extraction.draft

    assert draft.destination == "Tokyo"
    assert draft.duration_days == 5
    assert draft.budget is not None
    assert draft.budget.amount == Decimal("1000")
    assert draft.budget.currency == "USD"
    assert draft.budget.status is DataStatus.USER_INPUT
    assert draft.interests == ["food", "shopping", "culture"]
    assert draft.travel_style is TravelStyle.RELAXED

    # 日期非阻塞：可以先生成行程，所以 is_ready 应该为 True
    assert extraction.is_ready is True
    assert extraction.missing_fields == [MissingField.START_DATE]
    assert extraction.clarifying_questions == ["大概什么时候出发？还没定也可以先跳过。"]
    assert extraction.extracted_by == "rules"
    assert extraction.warnings == []


def test_reports_missing_fields_and_questions() -> None:
    extraction = _extract("我想去大阪玩")

    assert extraction.draft.destination == "Osaka"
    assert extraction.is_ready is False
    assert extraction.missing_fields == [
        MissingField.DURATION_DAYS,
        MissingField.START_DATE,
        MissingField.BUDGET,
    ]
    assert len(extraction.clarifying_questions) == 3


def test_budget_without_currency_defaults_to_usd_with_warning() -> None:
    extraction = _extract("东京 3 天，预算 1000")

    assert extraction.draft.budget is not None
    assert extraction.draft.budget.currency == "USD"
    assert any("币种" in warning for warning in extraction.warnings)


def test_explicit_date_is_not_mistaken_for_duration() -> None:
    """「2026年4月1日」里的 1 不能被解析成 1 天。"""
    extraction = _extract("东京 3 天，2026年4月1日出发")

    assert extraction.draft.duration_days == 3
    assert extraction.draft.start_date == date(2026, 4, 1)


def test_month_day_rolls_over_to_next_year() -> None:
    today = date(2026, 11, 9)

    this_year = _extract("东京 3 天，12月5日出发", today=today)
    next_year = _extract("东京 3 天，3月5日出发", today=today)

    assert this_year.draft.start_date == date(2026, 12, 5)
    assert next_year.draft.start_date == date(2027, 3, 5)


# ---------------------------- 多轮合并 ----------------------------


def test_modification_only_updates_mentioned_field() -> None:
    previous = TripRequirementDraft(
        destination="Tokyo",
        duration_days=5,
        budget=Money(
            amount=Decimal("1000"), currency="USD", status=DataStatus.USER_INPUT
        ),
    )

    extraction = _extract("预算改成 1500 美元", previous)
    draft = extraction.draft

    assert draft.destination == "Tokyo"
    assert draft.duration_days == 5
    assert draft.budget is not None
    assert draft.budget.amount == Decimal("1500")


def test_interests_are_merged_not_replaced() -> None:
    previous = TripRequirementDraft(destination="Tokyo", duration_days=5, interests=["food"])

    extraction = _extract("我还喜欢购物", previous)

    assert extraction.draft.interests == ["food", "shopping"]


def test_message_without_new_information_keeps_previous_draft() -> None:
    previous = TripRequirementDraft(destination="Kyoto", duration_days=4)

    extraction = _extract("好的，谢谢", previous)

    assert extraction.draft.destination == "Kyoto"
    assert extraction.draft.duration_days == 4


# ---------------------------- LLM 路径 ----------------------------


class _FakeCompletions:
    def __init__(
        self,
        parsed=None,
        error: Exception | None = None,
        content: str | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self._parsed = parsed
        self._error = error
        self._content = content
        self._create_error = create_error

    async def parse(self, **_: object):
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=self._parsed, refusal=None)
                )
            ]
        )

    async def create(self, **_: object):
        if self._create_error is not None:
            raise self._create_error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeClient:
    def __init__(
        self,
        parsed=None,
        error: Exception | None = None,
        content: str | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(parsed, error, content, create_error)
        )


def _parsed_payload() -> SimpleNamespace:
    return SimpleNamespace(
        destination="Kyoto",
        duration_days=3,
        start_date="2027-04-01",
        travelers=2,
        budget_amount=800.0,
        budget_currency="usd",
        interests=["culture", "food"],
        travel_style="relaxed",
        notes="不吃生食",
    )


def test_llm_output_is_converted_and_validated() -> None:
    agent = RequirementAgent(
        client=_FakeClient(_parsed_payload()),
        model="test-model",
        fallback=RuleBasedRequirementExtractor(),
    )

    outcome = asyncio.run(agent.extract("帮我安排京都", today=date(2026, 1, 1)))
    patch = outcome.patch

    assert outcome.source == "llm"
    assert patch.destination == "Kyoto"
    assert patch.duration_days == 3
    assert patch.start_date == date(2027, 4, 1)
    assert patch.travelers == 2
    assert patch.budget_amount == Decimal("800.00")
    assert patch.budget_currency == "USD"
    assert patch.interests == ["culture", "food"]
    assert patch.travel_style is TravelStyle.RELAXED
    assert patch.notes == "不吃生食"
    assert outcome.warnings == []


def test_structured_output_unavailable_falls_back_to_json_mode() -> None:
    """DeepSeek 不支持 json_schema 时，应改用 json_object 并继续走 LLM。"""
    content = """```json
{"destination": "Tokyo", "duration_days": 5, "start_date": null, "travelers": null,
 "budget_amount": 1000, "budget_currency": "USD",
 "interests": ["food", "shopping"], "travel_style": "relaxed", "notes": null}
```"""
    agent = RequirementAgent(
        client=_FakeClient(
            error=RuntimeError("response_format type is unavailable"),
            content=content,
        ),
        model="deepseek-flash",
    )

    outcome = asyncio.run(agent.extract("东京 5 天", today=date(2026, 1, 1)))

    assert outcome.source == "llm"
    assert outcome.patch.destination == "Tokyo"
    assert outcome.patch.duration_days == 5
    assert outcome.patch.budget_amount == Decimal("1000.00")
    assert outcome.patch.interests == ["food", "shopping"]
    assert outcome.warnings == []


def test_llm_failure_falls_back_to_rules() -> None:
    agent = RequirementAgent(
        client=_FakeClient(error=RuntimeError("boom")),
        model="test-model",
        fallback=RuleBasedRequirementExtractor(),
    )

    outcome = asyncio.run(agent.extract("东京 4 天", today=date(2026, 1, 1)))

    assert outcome.source == "rules"
    assert outcome.patch.destination == "Tokyo"
    assert outcome.patch.duration_days == 4
    assert any("降级" in warning for warning in outcome.warnings)


def test_llm_invalid_field_is_dropped_but_rest_kept() -> None:
    payload = _parsed_payload()
    payload.duration_days = 0
    payload.start_date = "not-a-date"

    agent = RequirementAgent(
        client=_FakeClient(payload),
        model="test-model",
        fallback=RuleBasedRequirementExtractor(),
    )

    outcome = asyncio.run(agent.extract("京都", today=date(2026, 1, 1)))

    assert outcome.source == "llm"
    assert outcome.patch.destination == "Kyoto"
    assert outcome.patch.duration_days is None
    assert outcome.patch.start_date is None
    assert len(outcome.warnings) == 2


# ---------------------------- HTTP 接口 ----------------------------


@pytest.fixture()
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_requirement_service] = _service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_api_returns_structured_requirements(client: TestClient) -> None:
    prefix = get_settings().api_prefix
    response = client.post(
        f"{prefix}/trips/requirements",
        json={"message": _TOKYO_SENTENCE},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["draft"]["destination"] == "Tokyo"
    assert body["draft"]["duration_days"] == 5
    assert body["missing_fields"] == ["start_date"]
    assert body["is_ready"] is True
    assert body["extracted_by"] == "rules"


def test_api_supports_multi_turn_previous_draft(client: TestClient) -> None:
    prefix = get_settings().api_prefix
    response = client.post(
        f"{prefix}/trips/requirements",
        json={
            "message": "预算改成 1500 美元",
            "previous_draft": {"destination": "Osaka", "duration_days": 2},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["draft"]["destination"] == "Osaka"
    assert body["draft"]["duration_days"] == 2
    assert body["draft"]["budget"]["amount"] == "1500"


def test_api_rejects_blank_message(client: TestClient) -> None:
    prefix = get_settings().api_prefix
    response = client.post(f"{prefix}/trips/requirements", json={"message": "   "})

    assert response.status_code == 422
