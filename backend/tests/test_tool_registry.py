"""工具注册表测试：profile 一致性、开关过滤、提示词渲染。

只做静态校验和本地构建，不发起任何网络请求：
    - 每个 profile 引用的名字都必须在 TOOL_SPECS 里注册；
    - 关闭 MCP 开关时，提示词与 build_tools 的结果同时少掉对应工具；
    - Agent 提示词里的「可用工具」来自注册表，而不是手写字符串。
"""

from __future__ import annotations

import pytest

from backend.agents.attraction_selection_agent import build_selection_system_prompt
from backend.agents.fx_agent import build_system_prompt as build_fx_prompt
from backend.tools.registry import (
    TOOL_PROFILES,
    TOOL_SPECS,
    ToolContext,
    build_tools,
    get_profile_tools,
    profile_specs,
    render_tool_block,
)


class _FakeSettings:
    """只实现注册表用到的三个开关字段。"""

    def __init__(
        self,
        *,
        baidu_mcp_enabled: bool = False,
        bocha_mcp_enabled: bool = False,
        bocha_apikey: str | None = None,
    ) -> None:
        self.baidu_mcp_enabled = baidu_mcp_enabled
        self.bocha_mcp_enabled = bocha_mcp_enabled
        self.bocha_apikey = bocha_apikey


def _ctx(**overrides) -> ToolContext:
    defaults = {"settings": _FakeSettings()}
    defaults.update(overrides)
    return ToolContext(**defaults)


def test_every_profile_name_is_registered() -> None:
    for profile, names in TOOL_PROFILES.items():
        for name in names:
            assert name in TOOL_SPECS, f"{profile} 引用了未注册的工具 {name}"


def test_profile_has_no_duplicate_names() -> None:
    for profile, names in TOOL_PROFILES.items():
        assert len(names) == len(set(names)), f"{profile} 有重复工具"


def test_disabled_mcp_tools_are_filtered_from_specs() -> None:
    disabled = profile_specs("research", _ctx())
    disabled_names = {spec.name for spec in disabled}
    assert "search_attractions" in disabled_names
    assert "mcp_geocode" not in disabled_names
    assert "mcp_web_search" not in disabled_names

    enabled_settings = _FakeSettings(
        baidu_mcp_enabled=True,
        bocha_mcp_enabled=True,
        bocha_apikey="test-key",
    )
    enabled_names = {
        spec.name for spec in profile_specs("research", _ctx(settings=enabled_settings))
    }
    assert {"mcp_route_matrix", "mcp_weather", "mcp_geocode"} <= enabled_names
    assert "mcp_web_search" in enabled_names


def test_build_tools_keeps_requested_order() -> None:
    tools = build_tools(["get_weather", "search_attractions"], _ctx())
    assert [tool.name for tool in tools] == ["get_weather", "search_attractions"]


def test_build_tools_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        build_tools(["no_such_tool"], _ctx())


def test_prompt_tools_match_bound_tools() -> None:
    """关闭 MCP 后，提示词里的工具清单必须与真正 bind 的工具一致。"""
    ctx = _ctx()
    prompt_names = {spec.name for spec in profile_specs("research", ctx)}
    bound_names = {tool.name for tool in get_profile_tools("research", ctx)}
    assert prompt_names == bound_names


def test_research_profile_builds_four_local_tools_without_mcp() -> None:
    tools = get_profile_tools("research", _ctx())
    assert [tool.name for tool in tools] == [
        "search_attractions",
        "get_attraction_details",
        "get_route_matrix",
        "get_weather",
    ]


def test_attraction_selection_profile_uses_web_search_extract_and_rank() -> None:
    assert TOOL_PROFILES["attraction_selection"] == (
        "mcp_web_search",
        "extract_web_places",
        "rank_attraction_pool",
    )


def test_selection_prompt_renders_tools_from_registry() -> None:
    prompt = build_selection_system_prompt(
        "南宁",
        ["自然", "历史"],
        tools=[
            "mcp_web_search",
            "extract_web_places",
            "rank_attraction_pool",
        ],
    )
    content = prompt.content

    assert "mcp_web_search" in content
    assert "extract_web_places" in content
    assert "rank_attraction_pool" in content
    # 描述来自注册表的 prompt_hint，而不是旧的手写文案
    assert "按兴趣检索城市地点线索" in content
    assert "从博查网页结果中抽取地点" in content
    assert "按兴趣对候选池排序" in content
    assert "只调用上面列出的工具" in content
    # 网页链路启用时不再引导调用百度候选池
    assert "search_attraction_pool" not in content
    # 每个兴趣只允许一种网页检索写法
    assert "南宁 自然 景点 推荐" in content
    assert "南宁 历史 景点 推荐" in content
    assert "好去处" not in content
    assert "必去" not in content


def test_fx_prompt_renders_all_fx_tools() -> None:
    content = build_fx_prompt()
    for name in TOOL_PROFILES["fx"]:
        assert name in content


def test_render_tool_block_accepts_names_and_tools() -> None:
    from_hints = render_tool_block(["get_weather"])
    assert from_hints == "- get_weather：查询城市天气"

    tools = build_tools(["get_weather"], _ctx())
    assert render_tool_block(tools) == from_hints
