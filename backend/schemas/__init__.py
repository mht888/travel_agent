"""Schema 包。

统一从这个文件导出常用模型，业务代码只 import backend.schemas 即可，
后面重命名文件时不会到处改 import。
"""

from backend.schemas.attraction import (
    Attraction,
    AttractionCategory,
    AttractionSearchResponse,
    OpeningInterval,
)
from backend.schemas.budget import BudgetBreakdown, BudgetCategory, BudgetLineItem
from backend.schemas.common import (
    DataStatus,
    GeoPoint,
    Money,
    Provenance,
    TransportMode,
)
from backend.schemas.itinerary import (
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
    TransportLeg,
)
from backend.schemas.requirement import (
    MissingField,
    RequirementExtraction,
    RequirementExtractionRequest,
    RequirementPatch,
    TripRequirementDraft,
)
from backend.schemas.restaurant import Restaurant, RestaurantSearchResponse
from backend.schemas.route import RouteLeg, RouteMatrix, RouteMatrixEntry
from backend.schemas.trip import TravelStyle, TripRequirements
from backend.schemas.weather import DailyWeather, WeatherCondition, WeatherForecast

__all__ = [
    "Attraction",
    "AttractionCategory",
    "AttractionSearchResponse",
    "BudgetBreakdown",
    "BudgetCategory",
    "BudgetLineItem",
    "DailyWeather",
    "DataStatus",
    "GeoPoint",
    "Itinerary",
    "ItineraryDay",
    "ItineraryItem",
    "ItineraryItemType",
    "MissingField",
    "Money",
    "OpeningInterval",
    "Provenance",
    "RequirementExtraction",
    "RequirementExtractionRequest",
    "RequirementPatch",
    "Restaurant",
    "RestaurantSearchResponse",
    "RouteLeg",
    "RouteMatrix",
    "RouteMatrixEntry",
    "TransportLeg",
    "TransportMode",
    "TravelStyle",
    "TripRequirementDraft",
    "TripRequirements",
    "WeatherCondition",
    "WeatherForecast",
]
