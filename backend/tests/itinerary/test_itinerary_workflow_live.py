"""用真实配置验收完整行程。

在项目根目录运行：
    .venv/Scripts/python.exe -m pytest -q -s backend/tests/itinerary/test_itinerary_workflow_live.py
"""

import json
from contextvars import ContextVar
from datetime import datetime, time, timedelta, timezone
from functools import wraps
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from pydantic import BaseModel
from pydantic_core import to_json

from backend.config import get_settings
from backend.graph import itinerary_workflow
from backend.main import create_app
from backend.schemas.common import DataStatus
from backend.schemas.itinerary import Itinerary, ItineraryItemType
from backend.schemas.requirement import RequirementExtraction
from backend.services.itinerary_logging import configure_itinerary_logging
from backend.services.requirement_service import RequirementService
from backend.services.usage import UsageCollector, active_usage

logger = itinerary_workflow.logger


class _ToolTrace(AsyncCallbackHandler):
    """记录工具调用的参数和结果，使用 tool_run 关联同一次调用。"""

    run_inline = True

    def __init__(self, collector: UsageCollector, agent: ContextVar[str]) -> None:
        self.collector = collector
        self.trace_id = collector.trace_id
        self.agent = agent
        self.model_calls = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        metadata = metadata or {}
        self.model_calls[run_id] = (
            datetime.now(timezone.utc),
            metadata.get("usage_stage") or self.agent.get(),
            metadata.get("ls_model_name") or self.collector.priced_model,
        )

    async def on_llm_end(self, response, *, run_id, **kwargs):
        message = next((
            generation.message
            for group in response.generations for generation in group
            if hasattr(generation, "message")
        ), None)
        output = response.llm_output or {}
        metadata = getattr(message, "response_metadata", {})
        self._record_model(
            run_id,
            model=output.get("model_name") or metadata.get("model_name"),
            usage=output.get("token_usage") or metadata.get("token_usage"),
            fallback_usage=getattr(message, "usage_metadata", None),
        )

    async def on_llm_error(self, error, *, run_id, **kwargs):
        self._record_model(run_id)

    def _record_model(self, run_id, *, model=None, usage=None, fallback_usage=None):
        started = self.model_calls.pop(run_id, None)
        if started is None:
            return
        started_at, stage, requested_model = started
        record = self.collector.record(
            call_id=str(run_id), stage=stage, model=model or requested_model,
            started_at=started_at, usage=usage, fallback_usage=fallback_usage,
        )
        logger.info("trace=%s event=model_usage detail=%s", self.trace_id, to_json(record).decode())

    async def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        logger.info(
            "trace=%s agent=%s tool=%s tool_run=%s event=tool_start input=%s",
            self.trace_id, self.agent.get(), (serialized or {}).get("name"),
            run_id, to_json(inputs if inputs is not None else input_str, fallback=str).decode(),
        )

    async def on_tool_end(self, output, *, run_id, **kwargs):
        payload = output.content if isinstance(output, ToolMessage) else output
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                pass
        logger.info(
            "trace=%s agent=%s tool=%s tool_run=%s event=tool_end output=%s",
            self.trace_id, self.agent.get(), kwargs.get("name"),
            run_id, to_json(payload, fallback=str).decode(),
        )

    async def on_tool_error(self, error, *, run_id, **kwargs):
        logger.warning(
            "trace=%s agent=%s tool_run=%s event=tool_error error=%s",
            self.trace_id, self.agent.get(), run_id, type(error).__name__,
        )


def _print_output(title: str, value: object) -> None:
    def encode(item: object):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"无法打印 {type(item).__name__}")

    print(f"\n=== {title} ===", flush=True)
    print(json.dumps(value, default=encode, ensure_ascii=False, indent=2), flush=True)


def _trace_workflow(monkeypatch: pytest.MonkeyPatch, collector: UsageCollector) -> None:
    configure_itinerary_logging(logger)
    trace_id = collector.trace_id
    current_agent = ContextVar("live_test_agent", default="workflow")
    tool_trace = _ToolTrace(collector, current_agent)
    build_workflow = itinerary_workflow.build_itinerary_workflow

    def trace_agent(owner, attribute: str, name: str) -> None:
        original = getattr(owner, attribute)

        @wraps(original)
        async def invoke_agent(*args, **kwargs):
            token = current_agent.set(name)
            logger.info("trace=%s agent=%s event=agent_start", trace_id, name)
            try:
                outcome = await original(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "trace=%s agent=%s event=agent_error error=%s",
                    trace_id, name, type(exc).__name__,
                )
                raise
            else:
                source = getattr(outcome, "source", getattr(outcome, "extracted_by", None))
                logger.info("trace=%s agent=%s event=agent_end source=%s", trace_id, name, source)
                return outcome
            finally:
                current_agent.reset(token)

        monkeypatch.setattr(owner, attribute, invoke_agent)

    trace_agent(RequirementService, "extract", "Requirement")
    trace_agent(itinerary_workflow, "run_attraction_selection", "AttractionSelection")
    trace_agent(itinerary_workflow, "run_restaurant_search", "Restaurant")
    trace_agent(itinerary_workflow, "run_hotel_search", "Hotel")
    trace_agent(itinerary_workflow.LlmItineraryPlanner, "plan", "Planner")
    trace_agent(itinerary_workflow.RuleBasedItineraryPlanner, "plan", "RuleBasedPlanner")

    def build_with_trace(data):
        graph = build_workflow(data)

        async def invoke(state):
            logger.info("trace=%s run=%s event=workflow_start", trace_id, state.get("run_id"))
            _print_output("结构化旅行需求", state["requirements"])
            result = {}
            async for update in graph.astream(
                state, config={"callbacks": [tool_trace]}, stream_mode="updates"
            ):
                for node, output in update.items():
                    if node == "select":
                        selection = output["selection"]
                        _print_output("景点初选", {
                            "summary": selection.summary,
                            "attractions": selection.attractions,
                            "warnings": output["selection_warnings"],
                        })
                    elif node in {"search_restaurants", "search_hotels"}:
                        kind = "restaurants" if node == "search_restaurants" else "hotels"
                        outcome = output[kind]
                        warning_key = "restaurant_warnings" if kind == "restaurants" else "hotel_warnings"
                        _print_output("餐厅候选" if kind == "restaurants" else "酒店候选", {
                            "summary": outcome.summary if outcome else None,
                            "items": getattr(outcome, kind, []),
                            "warnings": output.get(warning_key, []),
                        })
                    elif node == "fetch_weather":
                        _print_output("天气预报", {
                            "forecast": output["weather"],
                            "warnings": output.get("weather_warnings", []),
                        })
                    elif node == "routes":
                        matrix = output["matrix"]
                        _print_output("交通矩阵", {
                            "attraction_ids": matrix.attraction_ids if matrix else [],
                            "route_count": len(matrix.entries) if matrix else 0,
                            "sample_routes": matrix.entries[:5] if matrix else [],
                            "warnings": output.get("route_warnings", []),
                        })
                    elif node == "plan":
                        planner = output["draft"]
                        _print_output("Planner 排程草稿", {
                            "source": planner.source if planner else None,
                            "draft": planner.draft if planner else None,
                            "warnings": planner.warnings if planner else [],
                        })
                    elif node == "assemble":
                        itinerary = output["itinerary"]
                        _print_output("最终推荐酒店（最多 5 家）", itinerary.hotels)
                        _print_output("最终时间轴（含早餐、晚餐和实际路段）", itinerary.days)
                        _print_output("预算", itinerary.budget)
                        _print_output("流程告警", itinerary.warnings)
                    result.update(output)
            return result

        return SimpleNamespace(ainvoke=invoke)

    monkeypatch.setattr(itinerary_workflow, "build_itinerary_workflow", build_with_trace)


def test_complete_text_generates_live_itinerary(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    assert settings.openai_api_key, "请在 .env 配置 OPENAI_API_KEY"
    assert settings.bocha_mcp_enabled and settings.bocha_apikey, "请启用博查并配置 BOCHA_APIKEY"
    assert settings.gaode_maps_enabled and settings.gaode_maps_ak, "请启用高德并配置 GAODE_MAPS_AK"
    collector = UsageCollector(uuid4().hex, settings.openai_model)
    _trace_workflow(monkeypatch, collector)

    tomorrow = datetime.now(timezone(timedelta(hours=8))).date() + timedelta(days=1)
    message = (
        f"帮我规划从 {tomorrow.isoformat()} 出发的成都 3 日游，2 人同行，"
        "总预算 3000 元人民币，喜欢历史、自然和本地美食，"
        "行程尽量轻松，并推荐出行方便的酒店。"
    )
    print(f"输入：{message}", flush=True)
    app = create_app()

    @app.middleware("http")
    async def collect_request_usage(request, call_next):
        token = active_usage.set(collector)
        try:
            return await call_next(request)
        finally:
            active_usage.reset(token)

    try:
        with TestClient(app) as client:
            response = client.post(
                f"{settings.api_prefix.rstrip('/')}/trips/plan-from-text",
                json={"message": message},
            )
    finally:
        usage = collector.summary()
        logger.info("trace=%s event=usage_summary output=%s", collector.trace_id, to_json(usage).decode())
        _print_output("本次模型 Token 与费用（人民币）", json.loads(to_json(usage)))

    assert response.status_code == 200, response.text
    payload = response.json()
    extraction = RequirementExtraction.model_validate(payload["extraction"])
    _print_output("需求提取结果", extraction)
    assert extraction.is_ready, extraction.clarifying_questions
    assert payload["itinerary"] is not None
    itinerary = Itinerary.model_validate(payload["itinerary"])
    assert itinerary.trip.destination == "成都"
    
    assert itinerary.trip.start_date == tomorrow
    assert itinerary.trip.travelers == 2
    assert itinerary.trip.duration_days == 3
    assert itinerary.trip.budget is not None
    assert itinerary.trip.budget.amount == 3000
    assert itinerary.trip.budget.currency == "CNY"
    assert [day.day_index for day in itinerary.days] == [1, 2, 3], itinerary.warnings
    assert 1 <= len(itinerary.hotels) <= 5, itinerary.warnings
    assert itinerary.budget is not None, itinerary.warnings

    restaurant_ids = set()
    for day in itinerary.days:
        assert any(item.type == ItineraryItemType.ATTRACTION for item in day.items)
        for item in day.items:
            if item.type == ItineraryItemType.ATTRACTION and "中山路" in item.name:
                assert item.start_time >= time(18), item
        meals = [item for item in day.items if item.type == ItineraryItemType.RESTAURANT]
        assert [item.meal for item in meals] == ["breakfast", "dinner"], itinerary.warnings
        for item in meals:
            assert item.restaurant_id and item.restaurant_id not in restaurant_ids
            restaurant_ids.add(item.restaurant_id)
        assert day.items[0].meal == "breakfast"
        assert day.items[-1].meal == "dinner"
        for previous, current in zip(day.items, day.items[1:]):
            leg = current.transport_from_previous
            assert leg is not None
            assert leg.provenance.source == "amap_mcp"
            assert leg.provenance.status == DataStatus.REAL
            elapsed = (
                current.start_time.hour * 60 + current.start_time.minute
                - previous.end_time.hour * 60 - previous.end_time.minute
            )
            assert elapsed >= leg.duration_minutes
