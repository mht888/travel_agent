"""路线与交通耗时数据结构。

职责边界：
    Provider   取数（Google Routes / OSRM / Mock）
    Service    组装矩阵、处理缺失
    本文件     只负责「矩阵长什么样、什么样的矩阵算合法」

它是第 6 阶段取数的统一输出，也是第 10 阶段 OR-Tools 的输入。
"""

from pydantic import BaseModel, Field, model_validator

from backend.schemas.common import Provenance, TransportMode


class RouteLeg(BaseModel):
    """两个点之间的一段交通。与行程里的 TransportLeg 字段一致，但这里挂在矩阵上。"""

    mode: TransportMode
    duration_minutes: int = Field(..., ge=0)
    distance_meters: int | None = Field(None, ge=0)
    provenance: Provenance


class RouteMatrixEntry(RouteLeg):
    """矩阵中的一条有向边：from_id -> to_id。

    有向而不是无向：步行往返耗时相同，但驾车/公共交通可能不同，不能默认对称。
    """

    from_id: str = Field(..., min_length=1)
    to_id: str = Field(..., min_length=1)


class RouteMatrix(BaseModel):
    """景点间交通时间矩阵。

    允许稀疏：真实 API 可能查不到某条路线，缺失边用 duration_between() 返回 None 表示，
    由上层决定「跳过该景点」还是「用直线距离估算」。不要为了凑完整矩阵而伪造数据。
    """

    city: str = Field(..., min_length=1)
    mode: TransportMode
    attraction_ids: list[str] = Field(
        ..., min_length=1, description="本次参与计算的全部景点 id，作为合法边集合"
    )
    entries: list[RouteMatrixEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_entries(self) -> "RouteMatrix":
        # 重复 id 会让「合法边集合」失去意义，也会让矩阵出现多条同起终点
        if len(set(self.attraction_ids)) != len(self.attraction_ids):
            raise ValueError("attraction_ids 不允许重复")

        known = set(self.attraction_ids)
        seen_pairs: set[tuple[str, str]] = set()
        for entry in self.entries:
            if entry.from_id not in known or entry.to_id not in known:
                raise ValueError(
                    f"矩阵边 {entry.from_id}->{entry.to_id} 引用了未声明的景点 id"
                )
            if entry.from_id == entry.to_id:
                raise ValueError(f"不允许自环 {entry.from_id}->{entry.to_id}")

            pair = (entry.from_id, entry.to_id)
            if pair in seen_pairs:
                raise ValueError(f"矩阵边 {entry.from_id}->{entry.to_id} 重复")

            # 一个矩阵只描述一种交通方式，混入其他方式会让总耗时口径失真
            if entry.mode != self.mode:
                raise ValueError(
                    f"矩阵交通方式为 {self.mode.value}，但边 {entry.from_id}->{entry.to_id} "
                    f"是 {entry.mode.value}"
                )
            seen_pairs.add(pair)
        return self

    def duration_between(self, from_id: str, to_id: str) -> int | None:
        """取一条边的耗时。id 未声明抛 KeyError，边缺失返回 None。"""
        if from_id not in self.attraction_ids or to_id not in self.attraction_ids:
            raise KeyError(f"{from_id} 或 {to_id} 不在本矩阵声明的景点集合中")
        for entry in self.entries:
            if entry.from_id == from_id and entry.to_id == to_id:
                return entry.duration_minutes
        return None

    @property
    def is_complete(self) -> bool:
        """所有「不同景点之间的有向对」是否都有数据。单景点时视为完整。"""
        expected = len(self.attraction_ids) * (len(self.attraction_ids) - 1)
        return len(self.entries) == expected
