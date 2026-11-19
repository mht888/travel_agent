from backend.schemas.common import GeoPoint, Provenance
from backend.schemas.hotel import Hotel
from backend.services.hotel_assembly import select_top_hotels


def test_hotel_ranking_considers_score_and_distance_and_limits_to_five() -> None:
    center = GeoPoint(lat=22.8, lng=108.3)
    farther = GeoPoint(lat=22.9, lng=108.4)
    candidates = [
        Hotel(
            id=str(index),
            name=f"酒店{index}",
            city="南宁",
            rating=rating,
            location=point,
            provenance=Provenance(source="test"),
        )
        for index, (rating, point) in enumerate(
            [(None, center), (4.8, farther), (4.7, center), (4.6, center),
             (4.5, center), (4.4, center), (4.3, center)]
        )
    ]

    ranked = select_top_hotels(candidates, [center])

    assert len(ranked) == 5
    assert ranked[0].id == "2"
    assert all(hotel.rating is not None for hotel in ranked)
    assert select_top_hotels(candidates[:2], []) == [candidates[1], candidates[0]]
