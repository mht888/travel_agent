"""景点排序器单元测试：规则、LLM、降级与回填。"""

import asyncio
from typing import Any

from backend.config import Settings
from backend.providers.mock_provider import TOKYO_ATTRACTIONS
from backend.services.attraction_ranking import (
    LLMAttractionRanker,
    RuleBasedAttractionRanker,
    create_attraction_ranker,
)


class _FakeRunnable:
    """最小 Runnable 替身：只实现 ranker 用到的 ainvoke。"""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------- 规则排序 ----------------------------


def test_rule_ranker_prefers_interest_hits() -> None:
    ranking = _run(
        RuleBasedAttractionRanker().rank(
            city="Tokyo",
            attractions=TOKYO_ATTRACTIONS,
            interests=["food"],
            top_k=3,
        )
    )

    assert ranking.strategy_used == "rule"
    assert ranking.ranked_ids[0] == "tokyo-tsukiji-outer-market"
    assert "food" in ranking.items[0].reason
    assert all(0 <= item.score <= 1 for item in ranking.items)


def test_rule_ranker_falls_back_to_rating_without_interests() -> None:
    ranking = _run(
        RuleBasedAttractionRanker().rank(
            city="Tokyo",
            attractions=TOKYO_ATTRACTIONS,
            interests=[],
            top_k=3,
        )
    )

    assert ranking.ranked_ids[0] == "tokyo-sensoji"  # 4.6 分最高
    assert all(item.reason == "未指定兴趣，按评分排序" for item in ranking.items)


def test_rule_ranker_supports_chinese_interest() -> None:
    ranking = _run(
        RuleBasedAttractionRanker().rank(
            city="Tokyo",
            attractions=TOKYO_ATTRACTIONS,
            interests=["美食"],
            top_k=2,
        )
    )

    assert ranking.ranked_ids[0] == "tokyo-tsukiji-outer-market"
    assert "美食" in ranking.items[0].reason


# ---------------------------- LLM 排序 ----------------------------


def test_llm_ranker_parses_fenced_json_and_drops_unknown_ids() -> None:
    runnable = _FakeRunnable(
        '```json\n{"items": ['
        '{"id": "tokyo-ueno-park", "score": 0.9, "reason": "自然公园"},'
        '{"id": "fake-id", "score": 0.99, "reason": "不存在"},'
        '{"id": "tokyo-sensoji", "score": 0.8, "reason": "历史文化"}'
        "]}\n```"
    )
    ranker = LLMAttractionRanker(runnable, RuleBasedAttractionRanker())

    ranking = _run(
        ranker.rank(
            city="Tokyo",
            attractions=TOKYO_ATTRACTIONS,
            interests=["nature", "culture"],
            top_k=2,
        )
    )

    assert ranking.strategy_used == "llm"
    assert ranking.ranked_ids == ["tokyo-ueno-park", "tokyo-sensoji"]
    assert "nature" in runnable.prompts[0]


def test_llm_ranker_falls_back_to_rule_on_failure() -> None:
    runnable = _FakeRunnable(RuntimeError("upstream timeout"))
    ranker = LLMAttractionRanker(runnable, RuleBasedAttractionRanker())

    ranking = _run(
        ranker.rank(
            city="Tokyo",
            attractions=TOKYO_ATTRACTIONS,
            interests=["food"],
            top_k=2,
        )
    )

    assert ranking.strategy_used == "rule_fallback"
    assert ranking.ranked_ids[0] == "tokyo-tsukiji-outer-market"


def test_llm_ranker_backfills_missing_candidates() -> None:
    runnable = _FakeRunnable(
        '{"items": [{"id": "tokyo-sensoji", "score": 0.95, "reason": "历史"}]}'
    )
    ranker = LLMAttractionRanker(runnable, RuleBasedAttractionRanker())

    ranking = _run(
        ranker.rank(
            city="Tokyo",
            attractions=TOKYO_ATTRACTIONS,
            interests=["culture"],
            top_k=3,
        )
    )

    assert ranking.strategy_used == "llm"
    assert len(ranking.items) == 3
    assert ranking.ranked_ids[0] == "tokyo-sensoji"
    assert len(set(ranking.ranked_ids)) == 3


# ---------------------------- 工厂 ----------------------------


def test_factory_uses_rule_when_strategy_is_rule() -> None:
    ranker = create_attraction_ranker(
        "rule",
        settings=Settings(openai_api_key=None, attraction_ranking_strategy="rule"),
    )
    assert isinstance(ranker, RuleBasedAttractionRanker)


def test_factory_falls_back_to_rule_without_api_key() -> None:
    ranker = create_attraction_ranker(
        "llm",
        settings=Settings(openai_api_key=None, attraction_ranking_strategy="llm"),
    )
    assert isinstance(ranker, RuleBasedAttractionRanker)
