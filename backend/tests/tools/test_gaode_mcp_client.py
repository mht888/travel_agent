import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.graph.hotel_workflow import _hotels_from_messages
from backend.graph.restaurant_workflow import _restaurants_from_messages
from backend.tools.gaode_mcp_tools import GaodeMcpClient, create_gaode_mcp_tool


def test_gaode_request_limit_is_shared_by_all_actions(monkeypatch) -> None:
    async def check():
        client = GaodeMcpClient(api_key="test", max_concurrency=5)
        timestamps = []

        async def call_tool(*args, **kwargs):
            timestamps.append(asyncio.get_running_loop().time())
            return SimpleNamespace(structured_content={"ok": True})

        client._session = SimpleNamespace(call_tool=call_tool)
        monkeypatch.setattr(client, "_ensure_started", AsyncMock())
        await asyncio.gather(*(
            client.call_tool(name, {})
            for name in ("maps_weather", "maps_text_search", "maps_search_detail", "maps_geo")
        ))
        assert len(timestamps) == 4
        assert timestamps[3] - timestamps[0] >= 0.99

    asyncio.run(check())


def test_gaode_retries_closed_connection_with_new_session(monkeypatch) -> None:
    async def check():
        client = GaodeMcpClient(api_key="test")
        first = SimpleNamespace(call_tool=AsyncMock(side_effect=RuntimeError("Connection closed")))
        second = SimpleNamespace(call_tool=AsyncMock(
            return_value=SimpleNamespace(structured_content={"id": "poi-1"})
        ))
        client._session = first

        async def ensure_started():
            if client._session is None:
                client._session = second

        monkeypatch.setattr(client, "_ensure_started", ensure_started)
        result = await client.call_tool("maps_search_detail", {"id": "poi-1"})

        assert result == {"id": "poi-1"}
        first.call_tool.assert_awaited_once()
        second.call_tool.assert_awaited_once()
        assert len(client._request_times) == 2

    asyncio.run(check())


def test_compact_poi_messages_keep_all_candidates_for_finalize() -> None:
    async def check(kind, code, finalize):
        async def call_tool(name, arguments):
            if name == "maps_around_search":
                return {"pois": [{"id": str(i), "name": f"候选{i}", "typecode": code} for i in range(5)]}
            return {"location": "104.06,30.66", "tel": "保留在完整数据里", "rating": "4.5"}

        caller = SimpleNamespace(call_tool=call_tool)
        tool = create_gaode_mcp_tool(caller=caller, compact_poi_results=True)
        message = await tool.ainvoke({
            "type": "tool_call", "name": "gaode_mcp", "id": f"call-{kind}",
            "args": {
                "action": "around_search", "poi_kind": kind,
                "latitude": 30.66, "longitude": 104.06,
            },
        })
        content = json.loads(message.content)
        assert content["total"] == 5
        assert len(content["items"]) == 3
        assert "location" not in content["items"][0]
        assert len(message.artifact["items"]) == 5
        assert len(finalize([message], city="成都")) == 5

    async def run():
        await check("hotel", "100100", _hotels_from_messages)
        await check("restaurant", "050100", _restaurants_from_messages)

    asyncio.run(run())
