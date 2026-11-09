"""天气业务层。

Provider 只负责取数；Service 负责参数归一化、范围校验与响应组装。
"""

from datetime import date

from backend.providers.base import WeatherProvider
from backend.schemas.weather import WeatherForecast


class WeatherService:
    """按城市和日期返回逐日天气。"""

    # 与 TripRequirements.duration_days 的上限保持一致
    MAX_FORECAST_DAYS = 30

    def __init__(self, provider: WeatherProvider) -> None:
        self._provider = provider

    async def get_forecast(
        self,
        city: str,
        start_date: date,
        days: int,
    ) -> WeatherForecast:
        """返回从 start_date 起连续 days 天的天气。

        参数错误直接抛 ValueError；Service 不依赖 FastAPI，可独立测试。
        """
        normalized_city = city.strip()
        if not normalized_city:
            raise ValueError("city 不能为空白")
        if not 1 <= days <= self.MAX_FORECAST_DAYS:
            raise ValueError(f"days 必须在 1 到 {self.MAX_FORECAST_DAYS} 之间")

        items = await self._provider.get_forecast(
            city=normalized_city,
            start_date=start_date,
            days=days,
        )
        # 排序放在 Service：前端不应依赖 Provider 的返回顺序
        return WeatherForecast(
            city=normalized_city,
            days=sorted(items, key=lambda item: item.date),
        )
