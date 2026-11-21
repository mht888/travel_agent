"""聚合数据火车票 API 实站测试。手动启用后仅发一次查询。"""

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import httpx
import pytest
from dotenv import dotenv_values

ENDPOINT = "https://apis.juhe.cn/fapigw/train/query"


@pytest.mark.skipif(
    os.getenv("RUN_JUHE_TRAIN_LIVE_TESTS") != "1",
    reason="设置 RUN_JUHE_TRAIN_LIVE_TESTS=1 后运行聚合数据实站测试",
)
def test_juhe_train_api_live() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    api_key = os.getenv("JUHE_TRAIN_API_KEY") or dotenv_values(env_path).get("JUHE_TRAIN_API_KEY")
    api_key = (api_key or "").strip()
    if not api_key:
        pytest.fail("请在项目 .env 或终端环境变量中配置 JUHE_TRAIN_API_KEY。", pytrace=False)

    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    tomorrow = today + timedelta(days=1)
    departure = date.fromisoformat(os.getenv("JUHE_TRAIN_DEPARTURE", tomorrow.isoformat()))
    origin = os.getenv("JUHE_TRAIN_ORIGIN", "北京").strip()
    destination = os.getenv("JUHE_TRAIN_DESTINATION", "上海").strip()
    assert origin and destination and origin != destination, "请设置不同的出发地和到达地。"
    assert 0 <= (departure - today).days < 15, "测试日期须在北京时间今天起的 15 天内。"

    query = {
        "search_type": "1",
        "departure_station": origin,
        "arrival_station": destination,
        "date": departure.isoformat(),
        "enable_booking": "2",
    }
    started = perf_counter()
    try:
        response = httpx.post(
            ENDPOINT, data={**query, "key": api_key}, timeout=15.0, follow_redirects=False,
        )
    except httpx.RequestError as exc:
        pytest.fail(f"聚合数据连接失败（{type(exc).__name__}）。", pytrace=False)
    elapsed_ms = round((perf_counter() - started) * 1_000)
    if response.status_code != 200:
        pytest.fail(f"聚合数据返回 HTTP {response.status_code}，耗时 {elapsed_ms} ms。", pytrace=False)
    try:
        payload = response.json()
    except ValueError:
        pytest.fail("聚合数据返回的内容无法解析为 JSON。", pytrace=False)
    if not isinstance(payload, dict):
        pytest.fail("聚合数据返回的 JSON 结构不符合文档。", pytrace=False)
    if str(payload.get("error_code")) != "0":
        reason = str(payload.get("reason", "")).replace(api_key, "[已隐藏]")
        pytest.fail(
            f"聚合数据查询失败，业务码 {payload.get('error_code')}，原因 {reason}。",
            pytrace=False,
        )

    trains = payload.get("result")
    assert isinstance(trains, list) and trains, "查询成功，但没有返回车次。"
    priced_trains = [
        train for train in trains
        if isinstance(train.get("prices"), list)
        and any(seat.get("price") not in (None, "") for seat in train["prices"])
    ]
    station_pairs = sorted({
        f"{train.get('departure_station')} → {train.get('arrival_station')}" for train in trains
    })
    print(json.dumps({
        "query": query,
        "elapsed_ms": elapsed_ms,
        "train_count": len(trains),
        "priced_train_count": len(priced_trains),
        "station_pairs": station_pairs,
        "sample_trains": (priced_trains or trains)[:5],
    }, ensure_ascii=False, indent=2))
    assert priced_trains, "查询返回了车次，但未返回任何席别票价。"
