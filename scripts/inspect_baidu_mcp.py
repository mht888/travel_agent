"""手动探查百度地图 MCP Server 的工具与返回结构。

用法（在 travel_agent/ 目录下）：
    .\\.venv\\Scripts\\python.exe scripts\\inspect_baidu_mcp.py --city 南宁 --query 景点

可选：
    --matrix                 查询起终点路线，默认 青秀山风景区 → 青山大桥观景台
    --origin 青秀山风景区     路线起点
    --destination 青山大桥观景台  路线终点
    --mode walking           交通方式：driving/walking/riding/motorcycle/transit
                             其中 transit 走单条路线接口，其余走批量算路
    --district-id 450100     额外调用 map_weather（需要 6 位行政区划代码）
    --no-detail              只搜索，不逐条查详情
    --verbose                打印工具入参 schema 和异常堆栈

首次运行会通过 npx 下载 @baidumap/mcp-server-baidu-map，需要 Node 18+。
AK 默认读取 .env 的 BAIDU_MAPS_AK，也可以用 --ak 覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as exc:  # pragma: no cover - 只在缺依赖时触发
    raise SystemExit(
        "缺少 mcp 包，请先安装："
        r"uv pip install --python .\.venv\Scripts\python.exe 'mcp>=1.0'"
    ) from exc

from backend.config import get_settings

MCP_PACKAGE = "@baidumap/mcp-server-baidu-map"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探查百度地图 MCP Server")
    parser.add_argument("--city", default="南宁", help="城市名，默认 南宁")
    parser.add_argument("--query", default="景点", help="检索词，默认 景点")
    parser.add_argument("--ak", default=None, help="覆盖 BAIDU_MAPS_AK")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次调用超时秒数")
    parser.add_argument("--no-detail", action="store_true", help="不调用 map_place_details")
    parser.add_argument("--matrix", action="store_true", help="查询起终点之间的路线")
    parser.add_argument("--origin", default="青秀山风景区", help="路线起点名称")
    parser.add_argument("--destination", default="青山大桥观景台", help="路线终点名称")
    parser.add_argument(
        "--mode",
        default="walking",
        choices=["driving", "walking", "riding", "motorcycle", "transit"],
        help="交通方式；transit 走 map_directions，其余走 map_distance_matrix",
    )
    parser.add_argument("--district-id", default=None, help="额外调用 map_weather 的行政区划代码")
    parser.add_argument("--verbose", action="store_true", help="打印 schema 与异常堆栈")
    return parser.parse_args()


def _server_params(ak: str) -> StdioServerParameters:
    env = dict(os.environ)
    env["BAIDU_MAP_API_KEY"] = ak
    if os.name == "nt":
        # Windows 上 npx 是 .cmd，必须经过 cmd /c 才能被 CreateProcess 拉起
        return StdioServerParameters(
            command="cmd",
            args=["/c", "npx", "-y", MCP_PACKAGE],
            env=env,
        )
    return StdioServerParameters(command="npx", args=["-y", MCP_PACKAGE], env=env)


def _result_to_obj(result: Any) -> Any:
    """MCP 返回可能是 structuredContent，也可能是一段 JSON 文本。"""
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured:
        return structured

    chunks: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    raw = "\n".join(chunks)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _print_obj(title: str, obj: Any) -> None:
    print(f"\n===== {title} =====")
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    else:
        print(obj)


def _find_places(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        for key in ("places", "results", "pois", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _find_places(value)
                if nested:
                    return nested
    return []


def _find_place(places: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """按名称精确匹配，避免拿到同名区域里的其他 POI。"""
    for place in places:
        if place.get("name") == name:
            return place
    return None


def _point(place: dict[str, Any]) -> str | None:
    location = place.get("location")
    if not isinstance(location, dict):
        return None
    lat, lng = location.get("lat"), location.get("lng")
    if lat is None or lng is None:
        return None
    return f"{float(lat):.6f},{float(lng):.6f}"


async def _resolve_place(
    session: ClientSession,
    places: list[dict[str, Any]],
    name: str,
    city: str,
    timeout: float,
) -> dict[str, Any] | None:
    """先在已有搜索结果里找，找不到再按名称单独检索一次。"""
    found = _find_place(places, name)
    if found:
        return found

    result = await asyncio.wait_for(
        session.call_tool("map_search_places", {"query": name, "region": city}),
        timeout,
    )
    candidates = _find_places(_result_to_obj(result))
    matched = _find_place(candidates, name)
    if matched:
        return matched
    if candidates:
        print(f"未精确匹配「{name}」，改用「{candidates[0].get('name')}」")
        return candidates[0]
    return None


async def _call(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any],
    timeout: float,
) -> Any:
    started = time.perf_counter()
    result = await asyncio.wait_for(session.call_tool(name, arguments), timeout)
    elapsed = time.perf_counter() - started
    obj = _result_to_obj(result)
    is_error = getattr(result, "isError", False)
    _print_obj(f"{name} ({elapsed:.2f}s, is_error={is_error})", obj)
    return obj


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    ak = (args.ak or settings.baidu_maps_ak or "").strip()
    if not ak:
        raise SystemExit("没有 AK：请在 .env 配置 BAIDU_MAPS_AK，或传 --ak")

    async with stdio_client(_server_params(ak)) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), args.timeout)
            tools = await asyncio.wait_for(session.list_tools(), args.timeout)

            print(f"工具数量：{len(tools.tools)}")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
                if args.verbose:
                    schema = getattr(tool, "inputSchema", None) or getattr(
                        tool, "input_schema", None
                    )
                    if schema:
                        print(json.dumps(schema, ensure_ascii=False, indent=2))

            search = await _call(
                session,
                "map_search_places",
                {"query": args.query, "region": args.city},
                args.timeout,
            )
            places = _find_places(search)
            print(f"\n搜索到 {len(places)} 条 POI")
            if not places:
                print("没有结果：可换 --city 或 --query 再试")
                return

            first = places[0]
            uid = first.get("uid") or first.get("id")
            print(f"第一条：{first.get('name')} / uid={uid}")

            if not args.no_detail and uid:
                await _call(
                    session,
                    "map_place_details",
                    {"uid": str(uid), "scope": "2"},
                    args.timeout,
                )

            if args.matrix:
                origin = await _resolve_place(
                    session, places, args.origin, args.city, args.timeout
                )
                destination = await _resolve_place(
                    session, places, args.destination, args.city, args.timeout
                )
                origin_point = _point(origin) if origin else None
                destination_point = _point(destination) if destination else None

                if not origin or not destination or not origin_point or not destination_point:
                    print("\n起终点解析失败，跳过路线查询")
                else:
                    print(
                        f"\n路线：{origin.get('name')} → {destination.get('name')}"
                        f"（mode={args.mode}）"
                    )
                    if args.mode == "transit":
                        # 公交没有批量矩阵接口，只能查单条路线
                        await _call(
                            session,
                            "map_directions",
                            {
                                "origin": origin_point,
                                "destination": destination_point,
                                "mode": "transit",
                            },
                            args.timeout,
                        )
                    else:
                        await _call(
                            session,
                            "map_distance_matrix",
                            {
                                "origins": [origin_point],
                                "destinations": [destination_point],
                                "mode": args.mode,
                            },
                            args.timeout,
                        )

            if args.district_id:
                await _call(
                    session,
                    "map_weather",
                    {"districtId": args.district_id},
                    args.timeout,
                )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as exc:  # noqa: BLE001 - 手动脚本需要看到完整原因
        print(f"\nMCP 调用失败：{type(exc).__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
