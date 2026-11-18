"""餐厅搜索工作流状态。

和酒店状态的区别：
    酒店只关心「住哪」，输入必须是带坐标的入选景点；
    餐厅关心「吃什么、去哪家」，输入是城市 / 区域 / 口味 / 预算，
    不依赖景点坐标——后续要按天排餐时，再把当天的景点锚点加进来。

restaurants 存 Restaurant 的 JSON（已对齐 restaurant schema），
只来自 search_restaurants_around 的工具结果，不取自 LLM 文本；
restaurant_summary 是 Agent 的最终文字结论。
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RestaurantSearchState(TypedDict, total=False):
    """agent ↔ tools 循环状态 + 收口后的文字摘要。"""

    messages: Annotated[list[BaseMessage], add_messages]

    # 输入
    city: str
    area: str | None
    cuisines: list[str]
    # Money.model_dump(mode="json")；为空表示用户没给预算
    budget: dict[str, Any] | None
    notes: str | None

    # 产物：Restaurant.model_dump(mode="json")
    restaurants: list[dict[str, Any]]
    restaurant_summary: str | None
    # 工具失败、字段未核实、能力缺失等集中放这里
    warnings: list[str]
