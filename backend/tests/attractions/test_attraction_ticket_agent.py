import asyncio
import json
from datetime import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.agents import attraction_ticket_agent
from backend.config import Settings
from backend.schemas.attraction import Attraction
from backend.schemas.common import Money, Provenance
from backend.schemas.itinerary import Itinerary, ItineraryDay, ItineraryItem, ItineraryItemType
from backend.schemas.trip import TripRequirements
from backend.services.budget_service import _typed_item_cost_usd
from backend.tools import attraction_ticket_tools as tools


@pytest.mark.parametrize("official_index, body, expected_calls", [
    (1, "成人票免费，须提前实名预约", 2),
    (1, "", 3),
    (None, "", 2),
])
def test_ticket_tool_prefers_official_body_and_falls_back(monkeypatch, official_index, body, expected_calls):
    official = {"title": "参观须知", "url": "https://museum.example/visit", "snippet": "开放时间"}
    third_party = {"title": "参观信息", "url": "https://news.example/museum",
                   "snippet": "成人票免费，须提前实名预约"}
    search = AsyncMock()
    search.ainvoke.side_effect = lambda args: json.dumps({
        "results": [third_party if "官网" not in args["query"] and not args["query"].startswith("site:")
                    else official]
    })
    reader = AsyncMock(return_value=body)
    monkeypatch.setattr(tools, "_read_official", reader)
    site_model = AsyncMock()
    site_model.ainvoke.return_value = tools._OfficialPage(source_index=official_index)
    ticket_model = AsyncMock()

    async def extract(messages):
        rows = json.loads(messages[1][1])["rows"]
        index = next(i for i, row in enumerate(rows, 1) if "成人票免费" in row["snippet"])
        return tools._TicketExtraction(
            adult_price=Money(amount=0, currency="CNY"), price_source_index=index,
            price_quote="成人票免费", reservation="提前实名预约",
            reservation_source_index=index, reservation_quote="须提前实名预约",
        )

    ticket_model.ainvoke.side_effect = extract
    model = Mock()
    model.with_structured_output.side_effect = [site_model, ticket_model]
    tool = tools.create_attraction_ticket_tool(
        {"visit": tools.TicketVisit(city="成都", name="博物馆")}, search, model, Settings(),
    )
    assert "error" in asyncio.run(tool.ainvoke({"visit_id": "not-in-trip"}))
    reply = asyncio.run(tool.ainvoke({"visit_id": "visit"}))
    assert search.ainvoke.await_count == expected_calls
    assert reader.await_count == int(official_index is not None)
    assert reply["adult_price"]["amount"] == "0"
    assert reply["price_source"]["from_body"] is bool(body)
    assert reply["reservation_source"]["official"] is bool(body)


def test_agent_only_executes_distinct_requested_visits_and_prices_per_traveler(monkeypatch):
    itinerary = Itinerary(
        trip=TripRequirements(destination="成都", duration_days=1, travelers=2),
        days=[ItineraryDay(day_index=1, items=[
            ItineraryItem(seq=1, type="attraction", name="博物馆", attraction_id="museum",
                          start_time=time(9), end_time=time(10)),
            ItineraryItem(seq=2, type="restaurant", name="餐厅", attraction_id="restaurant",
                          start_time=time(10), end_time=time(11)),
            ItineraryItem(seq=3, type="attraction", name="公园", attraction_id="park",
                          start_time=time(11), end_time=time(12)),
        ])],
    )
    attractions = [
        Attraction(id=key, name=name, city="成都", category="culture",
                   estimated_duration_minutes=60, provenance=Provenance(source="test"))
        for key, name in [("museum", "博物馆"), ("restaurant", "餐厅"), ("park", "公园"), ("unused", "候选")]
    ]
    tool = SimpleNamespace(
        name="lookup_attraction_ticket",
        ainvoke=AsyncMock(return_value={"adult_price": {"amount": 50, "currency": "CNY"}}),
    )
    model = Mock()
    model.bind_tools.return_value.ainvoke = AsyncMock(return_value=SimpleNamespace(tool_calls=[
        {"name": tool.name, "args": {"visit_id": f"1:{key}"}}
        for key in ("restaurant", "museum", "museum", "unused")
    ]))
    monkeypatch.setattr(attraction_ticket_agent, "ChatOpenAI", lambda **kwargs: model)
    monkeypatch.setattr(attraction_ticket_agent, "get_settings", lambda: Settings(
        openai_api_key="mock", bocha_apikey="mock", bocha_mcp_enabled=True,
    ))
    monkeypatch.setattr(attraction_ticket_agent, "create_bocha_mcp_tools", lambda caller: [Mock()])
    register = Mock(return_value=tool)
    monkeypatch.setattr(attraction_ticket_agent, "create_attraction_ticket_tool", register)

    asyncio.run(attraction_ticket_agent.enrich_itinerary_tickets(itinerary, attractions))

    assert set(register.call_args.args[0]) == {"1:museum", "1:park"}
    tool.ainvoke.assert_awaited_once_with({"visit_id": "1:museum"})
    museum, restaurant, park = itinerary.days[0].items
    assert museum.cost.amount == 50 and museum.ticket.adult_price.amount == 50
    assert restaurant.ticket is None and park.ticket is None
    total, _, unknown, _ = _typed_item_cost_usd(itinerary, ItineraryItemType.ATTRACTION, {"CNY": 5})
    assert total == 20 and unknown == 1
