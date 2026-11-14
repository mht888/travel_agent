"""景点初选 Tool 契约测试：候选池缓存、排序合并、错误码。"""

import asyncio
import json
from typing import Any

from backend.providers.mock_provider import MockAttractionProvider
from backend.services.attraction_ranking import RuleBasedAttractionRanker
from backend.tools.attraction_selection_tools import (
    AttractionPoolCache,
    create_attraction_selection_tools,
)


def _tools(**kwargs: Any) -> dict[str, Any]:
    defaults = {
        "provider": MockAttractionProvider(),
        "ranker": RuleBasedAttractionRanker(),
        "pool_cache": AttractionPoolCache(),
    }
    defaults.update(kwargs)
    return {tool.name: tool for tool in create_attraction_selection_tools(**defaults)}


def _invoke(tools: dict[str, Any], name: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = asyncio.run(tools[name].ainvoke(payload))
    assert isinstance(raw, str)
    return json.loads(raw)


def test_search_pool_returns_search_id_and_compact_items() -> None:
    result = _invoke(_tools(), "search_attraction_pool", {"city": "Tokyo"})

    assert result["ok"] is True
    assert result["total"] == len(result["items"]) == 10
    assert result["search_id"]
    item = result["items"][0]
    assert {"id", "name", "category", "tags", "rating", "location"} <= set(item)
    assert "estimated_duration_minutes" not in item


def test_rank_pool_merges_full_objects() -> None:
    tools = _tools()
    search = _invoke(tools, "search_attraction_pool", {"city": "Tokyo"})

    result = _invoke(
        tools,
        "rank_attraction_pool",
        {
            "search_id": search["search_id"],
            "interests": ["food"],
            "top_k": 3,
        },
    )

    assert result["ok"] is True
    assert result["strategy_used"] == "rule"
    assert result["items"][0]["id"] == "tokyo-tsukiji-outer-market"
    assert result["items"][0]["name"] == "Tsukiji Outer Market"
    assert "reason" in result["items"][0]
    assert 0 <= result["items"][0]["score"] <= 1


def test_rank_pool_rejects_unknown_search_id() -> None:
    result = _invoke(
        _tools(),
        "rank_attraction_pool",
        {"search_id": "nope", "interests": ["food"]},
    )

    assert result["ok"] is False
    assert result["error_code"] == "search_id_not_found"
    assert result["retryable"] is True


def test_search_pool_returns_empty_without_search_id() -> None:
    tools = _tools(provider=MockAttractionProvider(dataset=[]))

    result = _invoke(tools, "search_attraction_pool", {"city": "Tokyo"})

    assert result["ok"] is True
    assert result["search_id"] is None
    assert result["total"] == 0
    assert "hint" in result
