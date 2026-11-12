"""把排程草稿组装成带时间轴的行程（第 7 阶段）。

职责切分：
    planner_agent      决定「每天去哪些景点、什么顺序」
    itinerary_builder  决定「几点到几点、中间留多少交通时间」
    Evaluator（第 9 阶段）决定「这样排是否合法」

第一版时间轴规则（规则越简单越容易测试）：
    - 每天 09:00 开始
    - 每个景点占 estimated_duration_minutes
    - 景点之间插入矩阵里的有向耗时；矩阵缺边时用 15 分钟估算并记 warning
    - 提供餐厅时，在第一个「结束时间 >= 12:00」的景点之后插入一顿 60 分钟午餐
    - 超过 19:00 记 warning，但不报错（是否违规留给第 9 阶段判定）

午餐交通的已知简化：
    矩阵只包含景点之间的边，没有餐厅节点。餐厅前后各按
    LUNCH_TRANSFER_MINUTES 估算，并在 TransportLeg 的 provenance 里标明
    estimated:lunch-transfer，不冒充真实路线数据。
"""

from collections.abc import Mapping, Sequence
from datetime import date, time, timedelta

from backend.schemas.attraction import Attraction
from backend.schemas.common import DataStatus, Provenance
from backend.schemas.itinerary import (
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
    TransportLeg,
)
from backend.schemas.planner import ItineraryDraft
from backend.schemas.restaurant import Restaurant
from backend.schemas.route import RouteMatrix, RouteMatrixEntry
from backend.schemas.trip import TripRequirements

DAY_START = time(9, 0)
DAY_END = time(19, 0)
FALLBACK_TRANSFER_MINUTES = 15
LUNCH_START = time(12, 0)
LUNCH_DURATION_MINUTES = 60
LUNCH_TRANSFER_MINUTES = 10


def _to_minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _to_time(minutes: int) -> time:
    hour, minute = divmod(minutes, 60)
    return time(hour=hour % 24, minute=minute)


def _find_entry(
    matrix: RouteMatrix, from_id: str, to_id: str
) -> RouteMatrixEntry | None:
    """在矩阵里找一条有向边。稀疏矩阵里缺边是正常情况，返回 None。"""
    for entry in matrix.entries:
        if entry.from_id == from_id and entry.to_id == to_id:
            return entry
    return None


def _make_lunch_item(
    restaurant: Restaurant, *, seq: int, start_minutes: int, mode
) -> ItineraryItem:
    """构造午餐项。餐厅不在交通矩阵里，前后统一按估算步行时间处理。"""
    return ItineraryItem(
        seq=seq,
        type=ItineraryItemType.RESTAURANT,
        name=restaurant.name,
        attraction_id=None,
        start_time=_to_time(start_minutes),
        end_time=_to_time(start_minutes + LUNCH_DURATION_MINUTES),
        cost=restaurant.average_price,
        transport_from_previous=TransportLeg(
            mode=mode,
            duration_minutes=LUNCH_TRANSFER_MINUTES,
            distance_meters=None,
            provenance=Provenance(
                source="estimated:lunch-transfer",
                status=DataStatus.ESTIMATED,
            ),
        ),
    )


def build_day(
    *,
    attraction_ids: Sequence[str],
    attractions_by_id: Mapping[str, Attraction],
    matrix: RouteMatrix,
    day_index: int,
    day_date: date | None = None,
    theme: str | None = None,
    restaurant: Restaurant | None = None,
) -> tuple[ItineraryDay, list[str]]:
    """把一天内的景点 id 展开成带时间的行程项。返回 (ItineraryDay, warnings)。

    步骤：
    1. cursor = _to_minutes(DAY_START)，items = []，warnings = []
    2. 遍历 attraction_ids（enumerate 从 0 开始）：
       a. attraction = attractions_by_id[attraction_id]
       b. 第一项没有交通；从第二项开始：
          - entry = _find_entry(matrix, 上一个 id, 当前 id)
          - entry 存在：transfer = entry.duration_minutes，
            transport 直接复用 entry 的 distance 和 provenance
          - entry 缺失：transfer = FALLBACK_TRANSFER_MINUTES，
            transport 的 provenance 用
            Provenance(source="estimated:fallback", status=DataStatus.ESTIMATED)，
            并 append 一条 warning（不要静默补数字）
          - 构造 TransportLeg(mode=matrix.mode, duration_minutes=transfer,
            distance_meters=entry.distance_meters if entry else None,
            provenance=...)
       c. start = cursor + transfer（第一项 transfer=0）
       d. end = start + attraction.estimated_duration_minutes
       e. append ItineraryItem(
              seq=index + 1,
              type=ItineraryItemType.ATTRACTION,
              name=attraction.name,
              attraction_id=attraction.id,
              start_time=_to_time(start),
              end_time=_to_time(end),
              cost=attraction.ticket_price,
              transport_from_previous=transport,
          )
       f. cursor = end
    3. cursor 晚于 _to_minutes(DAY_END) 时 append warning：
       f"第 {day_index} 天行程结束于 {_to_time(cursor):%H:%M}，超过 {DAY_END:%H:%M}"
    4. return ItineraryDay(day_index=day_index, date=day_date, theme=theme,
       items=items), warnings

    注意：
    - ItineraryItem 要求 end_time 严格晚于 start_time。estimated_duration_minutes
      已经是 > 0，正常不会触发；但 _to_time 会把超过 24 点的分钟数绕回 0 点，
      跨午夜的情况第一版不处理，留给第 9 阶段拦截。
    - seq 必须从 1 连续递增，ItineraryDay 的 validator 会检查。
    """
    items: list[ItineraryItem] = []
    warnings: list[str] = []
    cursor = _to_minutes(DAY_START)
    previous_id: str | None = None
    lunch_pending = restaurant is not None

    for index, attraction_id in enumerate(attraction_ids):
        attraction = attractions_by_id[attraction_id]

        transport: TransportLeg | None = None
        transfer = 0
        if previous_id is not None:
            entry = _find_entry(matrix, previous_id, attraction_id)
            if entry is None:
                # 稀疏矩阵里缺边是正常情况：用固定值估算，但必须留痕
                transfer = FALLBACK_TRANSFER_MINUTES
                provenance = Provenance(
                    source="estimated:fallback",
                    status=DataStatus.ESTIMATED,
                )
                distance = None
                warnings.append(
                    f"第 {day_index} 天缺少 {previous_id} -> {attraction_id} 的路线数据，"
                    f"已按 {FALLBACK_TRANSFER_MINUTES} 分钟估算"
                )
            else:
                transfer = entry.duration_minutes
                provenance = entry.provenance
                distance = entry.distance_meters

            transport = TransportLeg(
                mode=matrix.mode,
                duration_minutes=transfer,
                distance_meters=distance,
                provenance=provenance,
            )

        start = cursor + transfer
        end = start + attraction.estimated_duration_minutes

        # 超过 24 点会被 _to_time 绕回 0 点，从而产生 end < start 的非法项；
        # 与其伪造，不如截断当天并明确告知用户
        if end >= 24 * 60:
            warnings.append(
                f"第 {day_index} 天排到「{attraction.name}」时已超过 24 点，"
                f"剩余 {len(attraction_ids) - index} 个景点未安排，请降低每日强度"
            )
            break

        items.append(
            ItineraryItem(
                seq=len(items) + 1,
                type=ItineraryItemType.ATTRACTION,
                name=attraction.name,
                attraction_id=attraction.id,
                start_time=_to_time(start),
                end_time=_to_time(end),
                cost=attraction.ticket_price,
                transport_from_previous=transport,
            )
        )
        cursor = end
        previous_id = attraction_id

        # 第一处「结束时间跨过 12:00」的景点之后安排午餐。
        # 午餐后再去下一个景点，交通仍然取矩阵里的景点间耗时（见模块 docstring 的已知简化）。
        if restaurant is not None and lunch_pending and end >= _to_minutes(LUNCH_START):
            lunch_end = cursor + LUNCH_TRANSFER_MINUTES + LUNCH_DURATION_MINUTES
            if lunch_end <= _to_minutes(DAY_END):
                items.append(
                    _make_lunch_item(
                        restaurant,
                        seq=len(items) + 1,
                        start_minutes=cursor + LUNCH_TRANSFER_MINUTES,
                        mode=matrix.mode,
                    )
                )
                cursor = lunch_end
            else:
                warnings.append(
                    f"第 {day_index} 天行程较满，12:00 后已无合适时段安排餐厅"
                    f"「{restaurant.name}」"
                )
            lunch_pending = False

    # 全天景点都在中午前结束：把午餐放在最后一个景点之后（不晚于 19:00）
    if restaurant is not None and lunch_pending and items:
        lunch_end = cursor + LUNCH_TRANSFER_MINUTES + LUNCH_DURATION_MINUTES
        if lunch_end <= _to_minutes(DAY_END):
            items.append(
                _make_lunch_item(
                    restaurant,
                    seq=len(items) + 1,
                    start_minutes=cursor + LUNCH_TRANSFER_MINUTES,
                    mode=matrix.mode,
                )
            )
            cursor = lunch_end
        else:
            warnings.append(
                f"第 {day_index} 天行程较满，未安排餐厅「{restaurant.name}」"
            )

    if cursor > _to_minutes(DAY_END):
        warnings.append(
            f"第 {day_index} 天行程结束于 {_to_time(cursor):%H:%M}，"
            f"超过 {DAY_END:%H:%M}"
        )

    return (
        ItineraryDay(
            day_index=day_index,
            date=day_date,
            theme=theme,
            items=items,
        ),
        warnings,
    )


def build_itinerary(
    requirements: TripRequirements,
    draft: ItineraryDraft,
    candidates: Sequence[Attraction],
    matrix: RouteMatrix,
    restaurants: Sequence[Restaurant] = (),
) -> Itinerary:
    """把草稿落到 Itinerary：过滤掉候选集之外的 id，保持 LLM 给的顺序。

    restaurants 按天顺序取用：第 N 天用 restaurants[N-1]，用完了当天不排午餐。
    餐厅不参与排程决策（不由 LLM 选择），只是每天固定一顿的确定性安排。
    """
    by_id = {item.id: item for item in candidates}
    warnings: list[str] = []
    days: list[ItineraryDay] = []

    for day_plan in draft.days:
        known = [aid for aid in day_plan.attraction_ids if aid in by_id]
        unknown = [aid for aid in day_plan.attraction_ids if aid not in by_id]
        if unknown:
            warnings.append(
                f"第 {day_plan.day_index} 天引用了候选集外的景点 id："
                f"{'、'.join(unknown)}，已跳过"
            )
        if not known:
            warnings.append(f"第 {day_plan.day_index} 天没有可用景点，已跳过")
            continue

        day_date = (
            requirements.start_date + timedelta(days=day_plan.day_index - 1)
            if requirements.start_date
            else None
        )
        restaurant = (
            restaurants[day_plan.day_index - 1]
            if day_plan.day_index <= len(restaurants)
            else None
        )
        day, day_warnings = build_day(
            attraction_ids=known,
            attractions_by_id=by_id,
            matrix=matrix,
            day_index=day_plan.day_index,
            day_date=day_date,
            theme=day_plan.theme,
            restaurant=restaurant,
        )
        days.append(day)
        warnings.extend(day_warnings)

    return Itinerary(trip=requirements, days=days, warnings=warnings)
