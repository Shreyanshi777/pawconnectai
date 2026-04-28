from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

try:
    import httpx
except Exception:
    httpx = None

from backend.app.core.config import settings


DEFAULT_FALLBACK_LOCATION_NAME = "Francis Institute of Technology (Engineering College)"
DEFAULT_FALLBACK_LOCATION_ADDRESS = (
    "Francis Institute of Technology (Engineering College), Santoshi Mata Road, "
    "Mount Poinsur, Mandapeshwar, Dahisar West, Mumbai, Maharashtra 400103"
)
DEFAULT_FALLBACK_LOCATION_LAT = 19.2553
DEFAULT_FALLBACK_LOCATION_LONG = 72.8665

MAHARASHTRA_LOCATIONS = [
    {"area": "Mumbai", "lat": 19.0760, "long": 72.8777},
    {"area": "Pune", "lat": 18.5204, "long": 73.8567},
    {"area": "Nagpur", "lat": 21.1458, "long": 79.0882},
    {"area": "Nashik", "lat": 19.9975, "long": 73.7898},
    {"area": "Aurangabad", "lat": 19.8762, "long": 75.3433},
    {"area": "Solapur", "lat": 17.6599, "long": 75.9064},
    {"area": "Kolhapur", "lat": 16.7050, "long": 74.2433},
    {"area": "Thane", "lat": 19.2183, "long": 72.9781},
    {"area": "Navi Mumbai", "lat": 19.0330, "long": 73.0297},
    {"area": "Amravati", "lat": 20.9374, "long": 77.7796},
]

MAHARASHTRA_BOUNDS = {
    "lat_min": 15.60,
    "lat_max": 22.10,
    "long_min": 72.60,
    "long_max": 80.95,
}


def infer_area(location_lat: float | None, location_long: float | None) -> str | None:
    if location_lat is None or location_long is None:
        return None
    if not is_within_maharashtra(location_lat, location_long):
        return "Outside Maharashtra"

    best_area = None
    best_distance = None
    for item in MAHARASHTRA_LOCATIONS:
        distance = haversine_km(location_lat, location_long, item["lat"], item["long"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_area = item["area"]
    return best_area


def reverse_geocode(location_lat: float | None, location_long: float | None) -> dict:
    if location_lat is None or location_long is None:
        return {
            "location_name": DEFAULT_FALLBACK_LOCATION_NAME,
            "location_address": DEFAULT_FALLBACK_LOCATION_ADDRESS,
            "location_lat": DEFAULT_FALLBACK_LOCATION_LAT,
            "location_long": DEFAULT_FALLBACK_LOCATION_LONG,
        }

    area = infer_area(location_lat, location_long)
    result = {
        "location_name": area,
        "location_address": None,
    }

    if httpx is None:
        if result["location_name"] is None:
          result["location_name"] = DEFAULT_FALLBACK_LOCATION_NAME
        if result["location_address"] is None:
          result["location_address"] = DEFAULT_FALLBACK_LOCATION_ADDRESS
        result["location_lat"] = DEFAULT_FALLBACK_LOCATION_LAT
        result["location_long"] = DEFAULT_FALLBACK_LOCATION_LONG
        return result

    try:
        response = httpx.get(
            settings.nominatim_url,
            params={
                "lat": location_lat,
                "lon": location_long,
                "format": "jsonv2",
                "addressdetails": 1,
            },
            headers={"User-Agent": settings.app_user_agent},
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        address = payload.get("address", {})
        road = (
            address.get("house_number")
            or address.get("house_name")
            or address.get("road")
            or address.get("pedestrian")
            or address.get("footway")
            or address.get("street")
        )
        locality = (
            address.get("suburb")
            or address.get("neighbourhood")
            or address.get("quarter")
            or address.get("residential")
            or address.get("locality")
            or address.get("city_district")
            or address.get("hamlet")
            or address.get("village")
        )
        city = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("district")
            or address.get("county")
            or address.get("state_district")
            or area
        )
        region = address.get("state") or address.get("region") or address.get("country_state")

        short_parts = [part for part in [locality, city] if part]
        full_parts = [part for part in [road, locality, city, region] if part]

        result["location_name"] = ", ".join(short_parts) if short_parts else (city or area)
        result["location_address"] = ", ".join(full_parts) if full_parts else result["location_name"]
        result["location_lat"] = float(location_lat)
        result["location_long"] = float(location_long)
        if not result["location_name"] and payload.get("display_name"):
            result["location_name"] = payload["display_name"]
        if not result["location_address"] and payload.get("display_name"):
            result["location_address"] = payload["display_name"]
        if area == "Outside Maharashtra" and payload.get("display_name"):
            result["location_name"] = payload.get("display_name")
            result["location_address"] = payload.get("display_name")
        if result.get("location_lat") is None:
            result["location_lat"] = DEFAULT_FALLBACK_LOCATION_LAT
        if result.get("location_long") is None:
            result["location_long"] = DEFAULT_FALLBACK_LOCATION_LONG
        return result
    except Exception:
        if result["location_name"] is None:
            result["location_name"] = DEFAULT_FALLBACK_LOCATION_NAME
        if result["location_address"] is None:
            result["location_address"] = DEFAULT_FALLBACK_LOCATION_ADDRESS
        result["location_lat"] = DEFAULT_FALLBACK_LOCATION_LAT
        result["location_long"] = DEFAULT_FALLBACK_LOCATION_LONG
        return result


def is_within_maharashtra(lat: float, lon: float) -> bool:
    return (
        MAHARASHTRA_BOUNDS["lat_min"] <= lat <= MAHARASHTRA_BOUNDS["lat_max"]
        and MAHARASHTRA_BOUNDS["long_min"] <= lon <= MAHARASHTRA_BOUNDS["long_max"]
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius * c
