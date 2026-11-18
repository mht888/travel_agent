"""统一工具注册表：所有 Agent 从同一份清单里选工具，提示词不再手写工具名。

参考 s02_tool_use 的 TOOLS + TOOL_HANDLERS：
    s02 用「定义列表 + 分发映射」替代硬编码调用；
    这里把同样的思路移到后端：ToolSpec 是清单，TOOL_GROUPS 是构建器，
    TOOL_PROFILES 是「某类 Agent 选哪几个」。

三层职责：
    ToolSpec     描述一个工具：名字、所属分组、给提示词用的一句话、启用条件。
    TOOL_GROUPS  真正构建工具，Service / Provider / MCP client 都通过 ToolContext 注入。
    TOOL_PROFILES  Agent 只引用 profile 名，不直接 import 各工具工厂。

enabled() 是纯函数，只看 settings，不建连接——提示词渲染和单测都能安全调用。

典型用法：
    from backend.tools.registry import ToolContext, get_profile_tools, render_tool_block

    tools = get_profile_tools("research")
    prompt = f"可用工具：\\n{render_tool_block(tools)}"

新增工具的流程：
    1. 在对应模块写好工厂函数；
    2. 往 TOOL_SPECS 加一条 ToolSpec；
    3. 把名字加进需要的 TOOL_PROFILES；
    4. 在 TOOL_GROUPS 里接上构建逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

from langchain_core.tools import BaseTool

from backend.config import get_settings


def _always(_settings: Any) -> bool:
    return True


def _baidu_mcp_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "baidu_mcp_enabled", False))


def _bocha_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "bocha_mcp_enabled", False)) and bool(
        getattr(settings, "bocha_apikey", None)
    )


def _gaode_mcp_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "gaode_maps_enabled", False)) and bool(
        getattr(settings, "gaode_maps_ak", None)
    )


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的静态描述。不持有实例，也不持有 Service。"""

    name: str
    group: str
    prompt_hint: str
    enabled: Callable[[Any], bool] = _always


@dataclass
class ToolContext:
    """构建工具时可注入的依赖。

    全部可选：不传就由各工厂用默认实现（Mock Service / 默认 Provider）。
    测试可以注入 fake，避免真实网络请求。
    """

    settings: Any = None
    attraction_service: Any = None
    route_service: Any = None
    weather_service: Any = None
    selection_cache: Any = None
    web_place_extractor: Any = None
    web_place_ranker: Any = None
    alias_resolver: Any = None
    bocha_result_cache: Any = None
    strict_web_query: bool = False
    fx_service: Any = None
    baidu_mcp_caller: Any = None
    bocha_caller: Any = None
    gaode_mcp_caller: Any = None


# ---------------------------- 工具清单 ----------------------------

TOOL_SPECS: dict[str, ToolSpec] = {
    # 通用旅行工具（travel_tools）
    "search_attractions": ToolSpec(
        name="search_attractions",
        group="travel",
        prompt_hint="按城市和兴趣搜索景点，返回结构化清单",
    ),
    "get_attraction_details": ToolSpec(
        name="get_attraction_details",
        group="travel",
        prompt_hint="查询景点详情：开放时间、游玩时长、票价",
    ),
    "get_route_matrix": ToolSpec(
        name="get_route_matrix",
        group="travel",
        prompt_hint="计算多个地点之间的交通时间矩阵",
    ),
    "get_weather": ToolSpec(
        name="get_weather",
        group="travel",
        prompt_hint="查询城市天气",
    ),
    # 百度地图 MCP（随 baidu_mcp_enabled 开关）
    "mcp_route_matrix": ToolSpec(
        name="mcp_route_matrix",
        group="baidu_mcp",
        prompt_hint="百度路线矩阵（driving / walking / riding / motorcycle，不支持 transit）",
        enabled=_baidu_mcp_enabled,
    ),
    "mcp_weather": ToolSpec(
        name="mcp_weather",
        group="baidu_mcp",
        prompt_hint="百度天气（行政区划代码或 GCJ-02 坐标）",
        enabled=_baidu_mcp_enabled,
    ),
    "mcp_geocode": ToolSpec(
        name="mcp_geocode",
        group="baidu_mcp",
        prompt_hint="地址或地名转换为 GCJ-02 坐标",
        enabled=_baidu_mcp_enabled,
    ),
    # 博查网页搜索（需要 key）
    "mcp_web_search": ToolSpec(
        name="mcp_web_search",
        group="bocha",
        prompt_hint="网页搜索：按兴趣检索城市地点线索，也可查临时闭馆、活动、票价与近期口碑",
        enabled=_bocha_enabled,
    ),
    # 景点初选（web_place_extract_tools / rank_attraction_pool_tools）
    "extract_web_places": ToolSpec(
        name="extract_web_places",
        group="web_place_extract",
        prompt_hint="从博查网页结果中抽取地点，写入轻量候选池并返回 search_id；数量不足会提示补检索",
        enabled=_bocha_enabled,
    ),
    "rank_attraction_pool": ToolSpec(
        name="rank_attraction_pool",
        group="rank_pool",
        prompt_hint="按兴趣对候选池排序，返回 score 与理由；网页地点按每个兴趣各取前 7 个",
    ),
    # 高德地图 MCP：一个工具覆盖周边/关键词搜索、详情、地理编码、天气、路线等（需要 key）
    "gaode_mcp": ToolSpec(
        name="gaode_mcp",
        group="gaode_mcp",
        prompt_hint=(
            "高德地图 MCP：action=around_search 按坐标找 POI（酒店传 poi_kind=hotel，"
            "餐厅传 poi_kind=restaurant，返回距离/评分/人均/营业时间）；"
            "也支持 text_search、poi_detail、geocode、regeocode、weather、distance、"
            "route、ip_location、personal_map、navigate、taxi"
        ),
        enabled=_gaode_mcp_enabled,
    ),
    # 汇率维护（fx_tools）
    "fetch_fx_rates": ToolSpec(
        name="fetch_fx_rates",
        group="fx",
        prompt_hint="从指定数据源抓取汇率，暂存为候选快照",
    ),
    "get_last_snapshot": ToolSpec(
        name="get_last_snapshot",
        group="fx",
        prompt_hint="读取当前已发布的汇率快照摘要",
    ),
    "validate_snapshot": ToolSpec(
        name="validate_snapshot",
        group="fx",
        prompt_hint="校验候选快照，passed=false 时禁止发布",
    ),
    "publish_snapshot": ToolSpec(
        name="publish_snapshot",
        group="fx",
        prompt_hint="把通过校验的候选快照发布为正式汇率",
    ),
    "get_supported_currencies": ToolSpec(
        name="get_supported_currencies",
        group="fx",
        prompt_hint="返回当前快照里实际可换算的币种列表",
    ),
}


# ---------------------------- Agent 工具画像 ----------------------------

TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "research": (
        "search_attractions",
        "get_attraction_details",
        "get_route_matrix",
        "get_weather",
        "mcp_route_matrix",
        "mcp_weather",
        "mcp_geocode",
        "mcp_web_search",
    ),
    # 景点初选：网页检索 → 地点抽取 → 按兴趣排序
    "attraction_selection": (
        "mcp_web_search",
        "extract_web_places",
        "rank_attraction_pool",
    ),
    # 酒店搜索：先验证候选区域（geocode / 网页），再搜酒店与交通
    "hotel_search": (
        "gaode_mcp",
        "mcp_geocode",
        "mcp_web_search",
    ),
    # 餐厅搜索：网页发现美食与门店 + 高德商家核实 + 地名坐标校验
    "restaurant_search": (
        "gaode_mcp",
        "mcp_web_search",
        "mcp_geocode",
    ),
    "fx": (
        "fetch_fx_rates",
        "get_last_snapshot",
        "validate_snapshot",
        "publish_snapshot",
        "get_supported_currencies",
    ),
}


# ---------------------------- 分组构建器 ----------------------------
# 每个分组只构建一次，再按名字筛选；工厂都延迟 import，避免模块间循环依赖。


def _build_travel(ctx: ToolContext) -> list[BaseTool]:
    from backend.tools.travel_tools import create_travel_tools

    return create_travel_tools(
        attraction_service=ctx.attraction_service,
        route_service=ctx.route_service,
        weather_service=ctx.weather_service,
    )


def _build_baidu_mcp(ctx: ToolContext) -> list[BaseTool]:
    from backend.tools.baidu_mcp_tools import create_baidu_mcp_tools

    return create_baidu_mcp_tools(ctx.baidu_mcp_caller)


def _selection_cache(ctx: ToolContext) -> Any:
    """候选池缓存由各分组共享：谁先构建谁创建，后构建的直接复用。"""
    if ctx.selection_cache is None:
        from backend.cache.pool_cache import AttractionPoolCache

        ctx.selection_cache = AttractionPoolCache()
    return ctx.selection_cache


def _build_bocha(ctx: ToolContext) -> list[BaseTool]:
    from backend.cache.pool_cache import WebSearchResultCache
    from backend.tools.bocha_mcp_tools import create_bocha_mcp_tools

    if ctx.bocha_result_cache is None:
        ctx.bocha_result_cache = WebSearchResultCache()
    return create_bocha_mcp_tools(
        ctx.bocha_caller,
        result_cache=ctx.bocha_result_cache,
        strict_query=ctx.strict_web_query,
    )


def _build_web_place_extract(ctx: ToolContext) -> list[BaseTool]:
    from backend.cache.pool_cache import WebSearchResultCache
    from backend.tools.web_place_extract_tools import build_extract_web_places_tool

    cache = _selection_cache(ctx)
    if ctx.bocha_result_cache is None:
        ctx.bocha_result_cache = WebSearchResultCache()
    return [
        build_extract_web_places_tool(
            cache,
            ctx.bocha_result_cache,
            extractor=ctx.web_place_extractor,
            alias_resolver=ctx.alias_resolver,
        )
    ]


def _build_rank_pool(ctx: ToolContext) -> list[BaseTool]:
    from backend.tools.rank_attraction_pool_tools import create_web_place_rank_tool

    return [
        create_web_place_rank_tool(
            _selection_cache(ctx),
            web_place_ranker=ctx.web_place_ranker,
        )
    ]


def _build_gaode_mcp(ctx: ToolContext) -> list[BaseTool]:
    from backend.tools.gaode_mcp_tools import create_gaode_mcp_tools

    return create_gaode_mcp_tools(
        caller=ctx.gaode_mcp_caller,
        settings=ctx.settings,
    )


def _build_fx(ctx: ToolContext) -> list[BaseTool]:
    from backend.tools.fx_tools import get_fx_tools

    return get_fx_tools(ctx.fx_service)


TOOL_GROUPS: dict[str, Callable[[ToolContext], list[BaseTool]]] = {
    "travel": _build_travel,
    "baidu_mcp": _build_baidu_mcp,
    "bocha": _build_bocha,
    "web_place_extract": _build_web_place_extract,
    "rank_pool": _build_rank_pool,
    "gaode_mcp": _build_gaode_mcp,
    "fx": _build_fx,
}


# ---------------------------- 对外 API ----------------------------


def get_tool_spec(name: str) -> ToolSpec:
    try:
        return TOOL_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"未注册的工具：{name}") from exc


def _enabled_for(spec: ToolSpec, ctx: ToolContext) -> bool:
    """启用判断。显式注入 MCP caller 时跳过开关，方便离线测试。"""
    settings = ctx.settings or get_settings()
    bocha_ready = _bocha_enabled(settings) or ctx.bocha_caller is not None

    if spec.name in ("mcp_web_search", "extract_web_places"):
        return bocha_ready

    if spec.enabled(settings):
        return True
    if spec.group == "baidu_mcp" and ctx.baidu_mcp_caller is not None:
        return True
    if spec.group == "bocha" and ctx.bocha_caller is not None:
        return True
    if spec.group == "gaode_mcp" and ctx.gaode_mcp_caller is not None:
        return True
    return False


def profile_specs(profile: str, ctx: ToolContext | None = None) -> list[ToolSpec]:
    """返回某个 Agent 画像下、当前配置启用的工具描述（不构建实例）。

    提示词渲染用它，不需要网络连接，也不依赖 API Key。
    """
    ctx = ctx or ToolContext()
    try:
        names = TOOL_PROFILES[profile]
    except KeyError as exc:
        raise KeyError(f"未知的工具画像：{profile}") from exc
    return [TOOL_SPECS[name] for name in names if _enabled_for(TOOL_SPECS[name], ctx)]


def build_tools(
    names: Iterable[str],
    ctx: ToolContext | None = None,
) -> list[BaseTool]:
    """按名字构建工具，顺序与传入顺序一致。

    未知名字直接报错（拼错不应静默少工具）；未启用的工具静默跳过，
    这样同一个 profile 在有无 MCP key 的环境下都能用。
    """
    ctx = ctx or ToolContext()
    requested = list(names)

    unknown = [name for name in requested if name not in TOOL_SPECS]
    if unknown:
        raise KeyError(f"未注册的工具：{', '.join(unknown)}")

    specs = [TOOL_SPECS[name] for name in requested if _enabled_for(TOOL_SPECS[name], ctx)]

    grouped: dict[str, list[ToolSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.group, []).append(spec)

    built: dict[str, BaseTool] = {}
    for group in grouped:
        for tool in TOOL_GROUPS[group](ctx):
            built[tool.name] = tool

    return [built[spec.name] for spec in specs if spec.name in built]


def get_profile_tools(profile: str, ctx: ToolContext | None = None) -> list[BaseTool]:
    """按 Agent 画像构建工具集，Agent 代码只需要引用 profile 名。"""
    try:
        names = TOOL_PROFILES[profile]
    except KeyError as exc:
        raise KeyError(f"未知的工具画像：{profile}") from exc
    # 景点初选的网页检索必须使用固定 query 模板
    if profile == "attraction_selection":
        ctx = replace(ctx or ToolContext(), strict_web_query=True)
    return build_tools(names, ctx)


def render_tool_block(items: Sequence[Any]) -> str:
    """把工具描述渲染成提示词里的「可用工具」列表。

    接受 ToolSpec / BaseTool / 工具名字符串；未知工具退化为描述首行。
    """
    lines: list[str] = []
    for item in items:
        if isinstance(item, str):
            name = item
            tool = None
        else:
            name = item.name
            tool = item

        spec = TOOL_SPECS.get(name)
        if spec is not None:
            hint = spec.prompt_hint
        else:
            description = getattr(tool, "description", "") or ""
            hint = description.strip().splitlines()[0][:60] if description.strip() else ""
        lines.append(f"- {name}：{hint}")
    return "\n".join(lines)
