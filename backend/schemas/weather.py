"""天气数据结构。

它是第 3 阶段 WeatherService 的输出契约，也是第 5 阶段 get_weather Tool 的返回格式。

约定：
- 只描述「某城市某一天」的概况，不承诺小时级精度。
- 每条数据都带 Provenance，Mock 与真实预报不能混为一谈。
"""

from datetime import date as Date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from backend.schemas.common import Provenance


class WeatherCondition(str, Enum):
    """粗粒度天气状况。真实 Provider 需要先把厂商码映射到这里。"""

    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"


class DailyWeather(BaseModel):
    """单日天气。温度用摄氏度，降水概率用百分比。"""

    date: Date
    condition: WeatherCondition
    temp_min_c: float
    temp_max_c: float
    precipitation_probability: int = Field(..., ge=0, le=100)
    provenance: Provenance

    @model_validator(mode="after")
    def _validate_temperature(self) -> "DailyWeather":
        if self.temp_min_c > self.temp_max_c:
            raise ValueError("temp_min_c 不能高于 temp_max_c")
        return self


class WeatherForecast(BaseModel):
    """一次查询的逐日天气。

    日期必须升序且不重复：Service 负责排序，这里只负责拦截 Provider 的脏数据，
    避免前端拿到的顺序依赖 Provider 实现。
    """

    city: str = Field(..., min_length=1)
    days: list[DailyWeather] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_dates(self) -> "WeatherForecast":
        dates = [day.date for day in self.days]
        if len(set(dates)) != len(dates):
            raise ValueError("天气日期不允许重复")
        if dates != sorted(dates):
            raise ValueError("天气日期必须按升序排列")
        return self
