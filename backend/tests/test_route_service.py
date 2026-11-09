"""RouteService 单元测试。

不启动 FastAPI、不联网：直接用东京 Mock 数据集验证矩阵契约与输入边界。
"""

import asyncio

import pytest

from backend.providers.mock_provider import TOKYO_ATTRACTIONS, MockRouteProvider
from backend.schemas.attraction import Attraction
from backend.schemas.common import DataStatus, TransportMode
from backend.services.route_service import RouteService


def _service() -> RouteService:
    return RouteService(provider=MockRouteProvider())


def _attractions(*attraction_ids: str) -> list[Attraction]:
    by_id = {attraction.id: attraction for attraction in TOKYO_ATTRACTIONS}
    return [by_id[attraction_id] for attraction_id in attraction_ids]


def _build(
    attraction_ids: tuple[str, ...],
    mode: TransportMode = TransportMode.WALK,
    city: str = "Tokyo",
):
    return asyncio.run(
        _service().build_matrix(city, _attractions(*attraction_ids), mode)
    )


def test_builds_complete_directed_matrix() -> None:
    matrix = _build(("tokyo-sensoji", "tokyo-ueno-park", "tokyo-ginza-six"))

    assert matrix.city == "Tokyo"
    assert matrix.mode is TransportMode.WALK
    assert matrix.attraction_ids == [
        "tokyo-sensoji",
        "tokyo-ueno-park",
        "tokyo-ginza-six",
    ]
    assert len(matrix.entries) == 6
    assert matrix.is_complete is True

    assert all(entry.from_id != entry.to_id for entry in matrix.entries)
    assert all(entry.mode is TransportMode.WALK for entry in matrix.entries)
    assert all(entry.provenance.status is DataStatus.MOCK for entry in matrix.entries)
    assert all(entry.duration_minutes > 0 for entry in matrix.entries)

    forward = matrix.duration_between("tokyo-sensoji", "tokyo-ueno-park")
    backward = matrix.duration_between("tokyo-ueno-park", "tokyo-sensoji")
    assert forward is not None and backward is not None


def test_walk_takes_longer_than_drive() -> None:
    """同一对景点，步行耗时应显著高于驾车，否则速度参数就是失效的。"""
    walk = _build(("tokyo-sensoji", "tokyo-ginza-six"), TransportMode.WALK)
    drive = _build(("tokyo-sensoji", "tokyo-ginza-six"), TransportMode.DRIVE)

    walk_minutes = walk.duration_between("tokyo-sensoji", "tokyo-ginza-six")
    drive_minutes = drive.duration_between("tokyo-sensoji", "tokyo-ginza-six")

    assert walk_minutes is not None and drive_minutes is not None
    assert walk_minutes > drive_minutes


def test_single_attraction_returns_valid_empty_matrix() -> None:
    matrix = _build(("tokyo-sensoji",))

    assert matrix.entries == []
    assert matrix.is_complete is True
    assert matrix.duration_between("tokyo-sensoji", "tokyo-sensoji") is None


def test_city_is_trimmed() -> None:
    matrix = _build(("tokyo-sensoji", "tokyo-ueno-park"), city="  Tokyo  ")

    assert matrix.city == "Tokyo"


def test_rejects_empty_attractions() -> None:
    with pytest.raises(ValueError, match="至少需要一个景点"):
        _build(())


def test_rejects_duplicate_attraction_ids() -> None:
    with pytest.raises(ValueError, match="不允许重复"):
        _build(("tokyo-sensoji", "tokyo-sensoji"))


def test_rejects_blank_city() -> None:
    with pytest.raises(ValueError, match="city"):
        _build(("tokyo-sensoji", "tokyo-ueno-park"), city="   ")


def test_rejects_too_many_attractions() -> None:
    with pytest.raises(ValueError, match="最多支持"):
        asyncio.run(
            _service().build_matrix(
                "Tokyo",
                [
                    TOKYO_ATTRACTIONS[0].model_copy(update={"id": f"fake-{index}"})
                    for index in range(RouteService.MAX_ATTRACTIONS + 1)
                ],
            )
        )
