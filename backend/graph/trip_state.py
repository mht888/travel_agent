"""「景点初选 → 酒店搜索」编排图的状态。

每个字段只有前一个节点写入，所以不需要 reducer：
    selection node 写 selection_id / selection_summary / warnings；
    hotel 或 fallback node 写 hotel_status / hotel_summary。

hotel_status 区分「还没实现」和「初选失败被跳过」，
入口不用靠 hotel_summary 的文案去猜。
"""

from typing import Any, Literal, TypedDict


class TripPlanningState(TypedDict, total=False):
    # ---- 输入 ----
    city: str
    interests: list[str]

    # ---- selection node 产出 ----
    selection_id: str | None
    selection_summary: str | None
    warnings: list[str]

    # ---- hotel / fallback node 产出 ----
    hotel_status: Literal[
        "pending", "done", "empty", "skipped", "unavailable"
    ]
    hotel_summary: str | None
    # Hotel.model_dump(mode="json") 列表，已对齐 schema
    hotels: list[dict[str, Any]]
