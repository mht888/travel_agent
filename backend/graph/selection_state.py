"""景点初选工作流状态。

和 ResearchState 分开，是因为这条链路有明确的输入（city / interests）
和明确的产物（ranking），不需要把「调研摘要」这类字段混进来。
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AttractionSelectionState(TypedDict, total=False):
    """agent ↔ tools 循环状态 + 最终排序结果。"""

    messages: Annotated[list[BaseMessage], add_messages]

    # 输入
    city: str
    interests: list[str]

    # 产物：AttractionRanking 的 JSON 形式，避免 TypedDict 直接依赖 Pydantic 序列化
    ranking: dict[str, Any] | None
    selection_summary: str | None
