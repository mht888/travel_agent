"""行程使用中文城市名，兼容常用英文城市名和中文全称。"""

import re

CITY_ALIASES = {
    "tokyo": "东京", "東京都": "东京",
    "osaka": "大阪", "大阪市": "大阪",
    "kyoto": "京都", "京都市": "京都",
    "seoul": "首尔", "bangkok": "曼谷",
    "singapore": "新加坡", "paris": "巴黎",
    "london": "伦敦", "new york": "纽约",
    "nanning": "南宁", "南宁市": "南宁",
    "chengdu": "成都", "成都市": "成都",
    "beijing": "北京", "北京市": "北京",
    "shanghai": "上海", "上海市": "上海",
    "guangzhou": "广州", "广州市": "广州",
    "shenzhen": "深圳", "深圳市": "深圳",
    "hangzhou": "杭州", "杭州市": "杭州",
    "chongqing": "重庆", "重庆市": "重庆",
    "wuhan": "武汉", "武汉市": "武汉",
    "nanjing": "南京", "南京市": "南京",
    "xian": "西安", "xi'an": "西安", "西安市": "西安",
    "guilin": "桂林", "桂林市": "桂林",
    "hong kong": "香港", "macau": "澳门",
}

_HAN = re.compile(r"[\u3400-\u9fff]")


def canonical_city_name(value: str) -> str:
    """统一已知别名，保留其他名称供调用方按场景校验。"""
    clean = " ".join(value.split())
    return CITY_ALIASES.get(clean.casefold(), clean)


def require_chinese_city_name(value: str) -> str:
    city = canonical_city_name(value)
    if not city:
        raise ValueError("destination 不能为空白")
    if not _HAN.search(city) or re.search(r"[A-Za-z]", city):
        raise ValueError("无法识别城市的中文名称，请提供中文城市名")
    return city
