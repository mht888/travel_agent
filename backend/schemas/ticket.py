"""景点门票和预约信息及各自的引用证据。"""

from pydantic import BaseModel, HttpUrl

from backend.schemas.common import Money


class TicketSource(BaseModel):
    url: HttpUrl
    quote: str
    published_at: str | None = None
    official: bool = False
    from_body: bool = False


class AttractionTicket(BaseModel):
    adult_price: Money | None = None
    price_source: TicketSource | None = None
    reservation: str | None = None
    reservation_source: TicketSource | None = None
