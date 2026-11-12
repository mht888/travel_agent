r"""真实 LLM 需求抽取验证（会联网，默认跳过）。

只验证 RequirementAgent 的抽取链路：发送给模型的 prompt → 模型原始输出 → 归一化后的 patch。
不涉及多轮合并、追问生成和下游规划。

运行方式（PowerShell）：
    $env:RUN_LLM_TESTS = "1"
    .\.venv\Scripts\python.exe -m pytest backend/tests/test_llm_extraction_live.py -s -q
"""

import asyncio
import os
from datetime import date
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

from backend.agents.requirement_agent import RequirementAgent
from backend.config import get_settings

# 一条覆盖所有字段的完整需求：城市、天数、日期、人数、预算、兴趣、节奏、备注
MESSAGE = (
    "下个月5号出发去东京玩5天，我们一家三口，总预算2万人民币，"
    "喜欢美食和文化，也想购物，行程别太赶，另外我不吃生食。"
)
ROUNDS = 1


class _RecordingCompletions:
    """记录实际发给模型的消息和模型原始输出，不改动 agent 的调用逻辑。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.messages: list[dict[str, str]] | None = None
        self.mode: str | None = None
        self.raw: str | None = None

    async def create(self, **kwargs):
        self.messages = kwargs.get("messages")
        self.mode = "json_object"
        completion = await self._inner.create(**kwargs)
        self.raw = completion.choices[0].message.content
        return completion

    async def parse(self, **kwargs):
        self.messages = kwargs.get("messages")
        self.mode = "json_schema"
        completion = await self._inner.parse(**kwargs)
        parsed = completion.choices[0].message.parsed
        self.raw = parsed.model_dump_json() if parsed is not None else None
        return completion


def test_llm_extraction_prints_prompt_and_output() -> None:
    if not os.getenv("RUN_LLM_TESTS"):
        pytest.skip("真实 LLM 测试默认跳过，设置 RUN_LLM_TESTS=1 后运行")

    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("未配置 OPENAI_API_KEY")

    real_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
    )
    recorder = _RecordingCompletions(real_client.chat.completions)
    agent = RequirementAgent(
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
        model=settings.openai_model,
    )

    today = date.today()
    for round_no in range(1, ROUNDS + 1):
        outcome = asyncio.run(agent.extract(MESSAGE, today=today))

        print(f"\n{'=' * 24} 第 {round_no}/{ROUNDS} 轮 {'=' * 24}")
        for message in recorder.messages or []:
            print(f"--- 发送给 LLM 的 {message['role']} prompt ---")
            print(message["content"])
        print(f"--- LLM 原始输出（{recorder.mode}）---")
        print(recorder.raw)
        print("--- 归一化后的 patch ---")
        print(outcome.patch.model_dump_json(exclude_none=True))
        print(f"source={outcome.source} warnings={outcome.warnings}")
