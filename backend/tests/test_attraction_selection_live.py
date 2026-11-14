r"""景点初选 A/B 对比（会联网 + 调用 LLM，默认跳过）。

三种跑法共用同一份候选池，保证对比公平：
    A  纯函数 + 规则排序    不走 Agent、不走 LLM，作为基线
    B  Agent + 规则排序     验证 Agent 编排是否可用
    C  Agent + LLM 排序     验证大模型对兴趣的理解，并打印发给 LLM 的完整输入

运行方式（PowerShell，工作目录 travel_agent/）：
    $env:RUN_ATTRACTION_SELECTION_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_attraction_selection_live.py -s -q

前置条件：
    BAIDU_MAPS_AK 与 OPENAI_API_KEY 都已配置。
"""

import asyncio
import os
import time
from typing import Any

import pytest

from backend.config import get_settings
from backend.graph.selection_workflow import SelectionOutcome, run_attraction_selection
from backend.providers.baidu_provider import get_baidu_attraction_provider
from backend.schemas.attraction import Attraction
from backend.schemas.ranking import AttractionRanking
from backend.services.attraction_ranking import (
    LLMAttractionRanker,
    RuleBasedAttractionRanker,
    _build_llm_runnable,
)

CITY = "南宁"
INTERESTS = ["美食", "自然", "历史"]
TOP_K = 8


class _ReplayProvider:
    """把第一次搜索到的候选池原样回放给 Agent。

    这样 A/B/C 面对的是同一批景点，不会因为三次百度检索的波动导致对比失真，
    也避免重复消耗接口配额。
    """

    def __init__(self, attractions: list[Attraction]) -> None:
        self._attractions = attractions

    async def search_attractions(
        self,
        city: str,
        interests: list[str] | None = None,
        limit: int = 20,
    ) -> list[Attraction]:
        return list(self._attractions)[:limit]

    async def get_attraction(self, attraction_id: str) -> Attraction | None:
        return next(
            (item for item in self._attractions if item.id == attraction_id),
            None,
        )


class _EchoRunnable:
    """包一层，在调用 LLM 前打印它实际收到的排序提示词。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def ainvoke(self, prompt: str) -> Any:
        print("\n" + "=" * 20 + " 发给 LLM 的排序输入 " + "=" * 20)
        print(prompt)
        print("=" * 62 + "\n")
        return await self._inner.ainvoke(prompt)


def _print_ranking(
    label: str,
    ranking: AttractionRanking,
    names: dict[str, str],
) -> None:
    print(f"\n[{label}] strategy={ranking.strategy_used}")
    for index, item in enumerate(ranking.items, start=1):
        name = names.get(item.id, "（未知名称）")
        print(
            f"  {index}. {name}  [{item.id}]  "
            f"score={item.score:.2f}  {item.reason}"
        )


async def _load_pool() -> list[Attraction]:
    provider = get_baidu_attraction_provider()
    return await provider.search_attractions(
        CITY, interests=INTERESTS, limit=20
    )


async def _run_rule_baseline(pool: list[Attraction]) -> AttractionRanking:
    return await RuleBasedAttractionRanker().rank(
        city=CITY,
        attractions=pool,
        interests=INTERESTS,
        top_k=TOP_K,
    )


async def _run_agent(
    ranker: Any,
    pool: list[Attraction],
) -> SelectionOutcome:
    return await run_attraction_selection(
        CITY,
        INTERESTS,
        top_k=TOP_K,
        provider=_ReplayProvider(pool),
        ranker=ranker,
    )


def _build_llm_ranker() -> LLMAttractionRanker:
    """手动对比专用：在真实 LLM 外包装一层，打印排序提示词。

    生产链路仍然走 create_attraction_ranker("llm")，不会打印任何东西。
    """
    settings = get_settings()
    return LLMAttractionRanker(
        _EchoRunnable(_build_llm_runnable(settings)),
        RuleBasedAttractionRanker(),
        timeout_seconds=settings.attraction_ranking_timeout_seconds,
    )


def test_compare_rule_and_llm_ranking() -> None:
    if not os.getenv("RUN_ATTRACTION_SELECTION_TESTS"):
        pytest.skip("真实链路对比默认跳过，设置 RUN_ATTRACTION_SELECTION_TESTS=1 后运行")

    settings = get_settings()
    if not settings.baidu_maps_ak:
        pytest.skip("缺少 BAIDU_MAPS_AK")
    if not settings.openai_api_key:
        pytest.skip("缺少 OPENAI_API_KEY")

    started = time.perf_counter()
    pool = asyncio.run(_load_pool())
    pool_seconds = time.perf_counter() - started
    names = {attraction.id: attraction.name for attraction in pool}
    print(f"\n候选池：{len(pool)} 个景点（{CITY}），检索耗时 {pool_seconds:.2f}s")

    started = time.perf_counter()
    rule_ranking = asyncio.run(_run_rule_baseline(pool))
    baseline_seconds = time.perf_counter() - started
    _print_ranking("A 纯函数 + 规则", rule_ranking, names)

    started = time.perf_counter()
    agent_rule = asyncio.run(_run_agent(RuleBasedAttractionRanker(), pool))
    agent_rule_seconds = time.perf_counter() - started
    assert agent_rule.ranking is not None
    _print_ranking("B Agent + 规则", agent_rule.ranking, names)

    started = time.perf_counter()
    agent_llm = asyncio.run(_run_agent(_build_llm_ranker(), pool))
    agent_llm_seconds = time.perf_counter() - started
    assert agent_llm.ranking is not None
    _print_ranking("C Agent + LLM", agent_llm.ranking, names)

    print(
        "\n耗时（秒）："
        f"A={baseline_seconds:.2f}  B={agent_rule_seconds:.2f}  C={agent_llm_seconds:.2f}"
    )
    print(f"路径：B source={agent_rule.source}  C source={agent_llm.source}")
    if agent_llm.ranking.strategy_used == "rule_fallback":
        print("警告：C 的 LLM 排序失败并回退规则，本次对比结果不可用")

    for outcome in (agent_rule, agent_llm):
        ids = outcome.ranking.ranked_ids
        assert ids
        assert len(ids) <= TOP_K
        assert len(ids) == len(set(ids))
