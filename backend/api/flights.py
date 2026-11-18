"""独立的 KAYAK 香港站浏览器查询入口。"""

from fastapi import APIRouter

from backend.tools.flight_web_tools import FlightQuery, FlightSearchResult, run_flight_search

router = APIRouter(prefix="/flights", tags=["flights"])


@router.post("/search", response_model=FlightSearchResult)
async def search_flights(query: FlightQuery) -> FlightSearchResult:
    return await run_flight_search(query)
