"""景点初选工作流状态。

和 ResearchState 分开，是因为这条链路有明确的输入（city / interests）
和明确的产物（一份按兴趣排好序的 AttractionRanking）。
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AttractionSelectionState(TypedDict, total=False):
    """agent ↔ tools 循环状态 + 一次性排序结果。"""

    messages: Annotated[list[BaseMessage], add_messages]

    # 输入
    city: str
    interests: list[str]

    # 产物：AttractionRanking 的 JSON 形式
    ranking: dict[str, Any] | None
    selection_summary: str | None

    # 排序失败原因集中放在这里
    warnings: list[str]
