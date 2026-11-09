"""冒烟测试：验证「能启动 + 能返回结构化数据」。

运行（在 travel_agent/ 目录下）：
    uv run pytest -q
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.main import app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    # 用 with 触发 lifespan，后续加启动/关闭逻辑时不用改测试
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    """健康检查应返回 200、status=ok，并处于 Mock 模式。"""
    prefix = get_settings().api_prefix
    response = client.get(f"{prefix}/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "env": "dev", "mock_mode": True}


def test_search_attractions_returns_mock_items(client: TestClient) -> None:
    """Tokyo + food 应返回非空、且来源标记为 mock 的景点列表。"""
    prefix = get_settings().api_prefix
    response = client.get(
        f"{prefix}/attractions/search",
        params={"city": "Tokyo", "interests": ["food"], "limit": 10},
    )

    assert response.status_code == 200
    body = response.json()
    items = body["items"]

    assert items, "Tokyo + food 应该有结果"
    assert body["total"] == len(items)
    assert body["interests"] == ["food"]
    assert all(item["provenance"]["status"] == "mock" for item in items)

    # 这条断言就是 Service 兴趣打分的验收标准：美食需求不该被高评分寺庙挤到第一
    first = items[0]
    assert first["category"] == "food" or "food" in first["tags"]


def test_search_unknown_city_returns_empty(client: TestClient) -> None:
    """未知城市返回 200 + 空列表，而不是 404。"""
    prefix = get_settings().api_prefix
    response = client.get(f"{prefix}/attractions/search", params={"city": "Nowhere"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []
