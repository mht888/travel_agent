"""酒店搜索工作流状态。

输入是「压缩后的入选景点」，不是完整 Attraction：
Agent 只需要 name / location / category / score / reason 做地理推理，
描述、开放时间、票价、provenance 全部裁掉，避免上下文膨胀。
压缩逻辑见 hotel_workflow.compact_anchors。

hotels 存 Hotel 的 JSON（已对齐 hotel schema），
不是 Agent 的自由文本——酒店字段只从 Tool 结果里取。
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class HotelSearchState(TypedDict, total=False):
    """agent ↔ tools 循环状态 + 收口后的酒店列表。"""

    messages: Annotated[list[BaseMessage], add_messages]

    # 输入
    city: str
    radius_meters: int
    anchors: list[dict[str, Any]]

    # 产物：Hotel.model_dump(mode="json")
    hotels: list[dict[str, Any]]
    hotel_summary: str | None
    # Tool 失败、区域未验证、selection 过期等集中放这里
    warnings: list[str]
