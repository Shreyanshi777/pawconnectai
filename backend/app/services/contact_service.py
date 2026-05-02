from __future__ import annotations

import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from difflib import SequenceMatcher
from math import atan2, cos, radians, sin, sqrt
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover - httpx is expected in requirements
    httpx = None

from backend.app.core.config import settings


VET_CATEGORIES = "pet.veterinary,healthcare,pet,service"
RESCUE_CATEGORIES = "pet.service,pet,service,emergency"
DEFAULT_RESULT_LIMIT = 20
NEARBY_CONTACT_RADIUS_METERS = 8000
FAR_CONTACT_RADIUS_METERS = max(25000, int(settings.geoapify_search_radius_meters))
MAPPLS_NEARBY_CONTACT_RADIUS_METERS = 5000
MAPPLS_FAR_CONTACT_RADIUS_METERS = 15000
OVERPASS_NEARBY_RADIUS_METERS = 30000
OVERPASS_FAR_RADIUS_METERS = 60000
CONTACT_CACHE_TTL_SECONDS = 180
AUTOCOMPLETE_CACHE_TTL_SECONDS = 600
_CONTACT_CACHE: dict[tuple[Any, ...], tuple[float, tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {}
_AUTOCOMPLETE_CACHE: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
SUPPORTED_CITY_POINTS = {
    "mumbai": ("Mumbai", 19.0760, 72.8777),
    "thane": ("Thane", 19.2183, 72.9781),
}
SUPPORTED_CITY_ALIASES = {
    "bombay": "mumbai",
    "thana": "thane",
}
SUPPORTED_LOCATION_GROUPS = [
    {
        "label": "Borivali",
        "city": "Mumbai",
        "lat": 19.2290,
        "lon": 72.8560,
        "aliases": [
            "borivali",
            "borivli",
            "borivalli",
            "borivali west",
            "borivali east",
            "borivali station",
            "borivali railway station",
            "borivali west link road",
            "s v road borivali",
            "ic colony",
            "eksar",
            "gorai",
            "yogi nagar",
            "m g road borivali",
        ],
    },
    {
        "label": "Dahisar",
        "city": "Mumbai",
        "lat": 19.2570,
        "lon": 72.8590,
        "aliases": [
            "dahisar",
            "dahisar east",
            "dahisar west",
            "dahisar station",
            "mandapeshwar",
            "ketkipada",
            "shailendra nagar",
            "mhatre wadi",
            "mira bhayandar road",
            "st francis institute of technology",
            "francis institute of technology",
            "mount poinsur",
            "santoshi mata road",
            "st francis dahisar",
        ],
    },
]
SUPPORTED_REGION_RADIUS_KM = 7.0
LOCALITY_VARIANT_SUGGESTIONS = {
    "borivali": [
        "Borivali East",
        "Borivali West",
        "Borivali Station",
        "Borivali Railway Station",
        "Borivali West Link Road",
        "S.V. Road, Borivali West",
        "IC Colony",
        "Eksar",
        "Gorai",
        "Yogi Nagar",
        "M.G. Road, Borivali East",
        "Chikuwadi",
        "Saibaba Nagar",
        "Devki Nagar",
        "Vazira Naka",
        "Shimpoli",
        "Prem Nagar",
        "Kora Kendra",
        "New Link Road, Borivali",
        "Borivali National Park",
        "Mahavir Nagar, Borivali",
    ],
    "dahisar": [
        "Dahisar East",
        "Dahisar West",
        "Dahisar Station",
        "Mandapeshwar",
        "Ketkipada",
        "Shailendra Nagar",
        "Mhatre Wadi",
        "Mira-Bhayandar Road",
        "St. Francis Institute of Technology",
        "St Francis Institute of Technology",
        "Francis Institute of Technology",
        "Mount Poinsur",
        "Santoshi Mata Road",
        "St Francis Dahisar",
        "Anand Nagar",
        "Nancy Colony",
        "Rawalpada",
        "Ashok Van",
        "Ketkipada East",
        "Dahisar Toll Naka",
        "LC Road, Dahisar",
        "SV Road, Dahisar",
    ],
}

VET_CONTACT_KEYWORDS = (
    "vet",
    "veterinary",
    "animal hospital",
    "animal clinic",
    "pet clinic",
    "pet hospital",
    "animal doctor",
)
RESCUE_CONTACT_KEYWORDS = (
    "rescue",
    "shelter",
    "welfare",
    "spca",
    "animal aid",
    "animal welfare",
    "pet rescue",
    "animal rescue",
    "animal shelter",
    "animal help",
    "street animal",
    "animal welfare trust",
    "animal welfare society",
    "foundation",
    "trust",
    "society",
)
VET_CONTACT_CATEGORY_GROUPS = [
    "pet.veterinary",
    "healthcare",
    "healthcare.clinic_or_praxis.general",
    "pet",
    "service",
]
RESCUE_CONTACT_CATEGORY_GROUPS = [
    "pet.service",
    "pet",
    "service",
    "emergency",
]


@dataclass(frozen=True)
class LocationContext:
    label: str
    lat: float
    lon: float
    place_id: str | None = None
    source: str = "query"


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


def distance_label(distance_km: float | None) -> str | None:
    if distance_km is None:
        return None
    if distance_km < 1:
        meters = max(1, round(distance_km * 1000))
        return f"{meters}m"
    rounded = round(distance_km, 1)
    if float(rounded).is_integer():
        rounded = int(rounded)
    return f"{rounded}k"


def map_link(lat: float, lon: float) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}"


def _client() -> httpx.Client:
    if httpx is None:  # pragma: no cover - defensive guard
        raise RuntimeError("httpx is not available.")
    return httpx.Client(
        timeout=3.0,
        trust_env=False,
        headers={
            "Accept-Language": "en",
            "User-Agent": settings.app_user_agent,
        },
    )


def _api_key() -> str:
    key = (settings.geoapify_api_key or "").strip()
    if not key:
        raise RuntimeError("Geoapify API key is missing. Set GEOAPIFY_API_KEY in .env.")
    return key


def _google_api_key() -> str | None:
    key = (settings.google_maps_api_key or "").strip()
    return key or None


def _mappls_access_token() -> str | None:
    key = (settings.mappls_access_token or "").strip()
    return key or None


def _has_mappls() -> bool:
    return bool(_mappls_access_token())


def _has_google_places() -> bool:
    return bool(_google_api_key())


def _google_request_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(url, params=params)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _google_autocomplete_url() -> str:
    return "https://maps.googleapis.com/maps/api/place/autocomplete/json"


def _google_place_details_url() -> str:
    return "https://maps.googleapis.com/maps/api/place/details/json"


def _google_nearby_url() -> str:
    return "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


def _mappls_autosuggest_url() -> str:
    return "https://search.mappls.com/search/places/autosuggest/json"


def _mappls_textsearch_url() -> str:
    return "https://search.mappls.com/search/places/textsearch/json"


def _mappls_geocode_url() -> str:
    return "https://search.mappls.com/search/address/geocode"


def _mappls_reverse_geocode_url() -> str:
    return "https://search.mappls.com/search/address/rev-geocode"


def _mappls_nearby_url() -> str:
    return "https://search.mappls.com/search/places/nearby/json"


def _google_pick_prediction(query: str, predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = _match_text(_normalize_query(query) or "")
    scored: list[tuple[int, dict[str, Any]]] = []
    for index, prediction in enumerate(predictions):
        description = _match_text(prediction.get("description"))
        main_text = _match_text(prediction.get("structured_formatting", {}).get("main_text"))
        score = 0
        if description == normalized:
            score += 1000
        if main_text == normalized:
            score += 900
        if description.startswith(normalized):
            score += 400
        if main_text.startswith(normalized):
            score += 350
        if normalized and normalized in description:
            score += 200
        if normalized and normalized in main_text:
            score += 150
        score -= index
        scored.append((score, prediction))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _google_place_details(client: httpx.Client, place_id: str) -> dict[str, Any] | None:
    key = _google_api_key()
    if not key:
        return None
    payload = _google_request_json(
        client,
        _google_place_details_url(),
        {
            "place_id": place_id,
            "fields": "place_id,name,formatted_address,geometry,formatted_phone_number,international_phone_number,website,url,address_component,types",
            "language": "en",
            "key": key,
        },
    )
    if payload.get("status") != "OK" or not isinstance(payload.get("result"), dict):
        return None
    return payload["result"]


def _google_autocomplete(
    client: httpx.Client,
    query: str,
    *,
    limit: int = 12,
    bias_lat: float | None = None,
    bias_lon: float | None = None,
    bias_radius_m: int = 25000,
) -> list[dict[str, Any]]:
    key = _google_api_key()
    if not key:
        return []
    params: dict[str, Any] = {
        "input": query,
        "language": "en",
        "components": f"country:{settings.geoapify_country_code}",
        "key": key,
    }
    if bias_lat is not None and bias_lon is not None:
        params["location"] = f"{bias_lat},{bias_lon}"
        params["radius"] = str(bias_radius_m)
        params["strictbounds"] = "true"
    payload = _google_request_json(
        client,
        _google_autocomplete_url(),
        params,
    )
    if payload.get("status") == "ZERO_RESULTS":
        return []
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return []
    return [pred for pred in predictions[:limit] if isinstance(pred, dict)]


def _google_location_context(client: httpx.Client, location: str) -> LocationContext | None:
    normalized = _normalize_query(location)
    if not normalized:
        return None
    anchor_item, _ = _supported_location_anchor(normalized)
    bias_lat = float(anchor_item["lat"]) if anchor_item is not None else None
    bias_lon = float(anchor_item["lon"]) if anchor_item is not None else None
    predictions = _google_autocomplete(client, normalized, limit=5, bias_lat=bias_lat, bias_lon=bias_lon)
    if not predictions:
        return None
    prediction = _google_pick_prediction(normalized, predictions) or predictions[0]
    place_id = _extract_text(prediction.get("place_id"))
    if not place_id:
        return None
    details = _google_place_details(client, place_id)
    if not details:
        return None
    geometry = details.get("geometry") or {}
    location_data = geometry.get("location") if isinstance(geometry, dict) else {}
    lat = location_data.get("lat") if isinstance(location_data, dict) else None
    lon = location_data.get("lng") if isinstance(location_data, dict) else None
    if lat is None or lon is None:
        return None
    return LocationContext(
        label=_extract_text(details.get("name")) or _extract_text(prediction.get("description")) or normalized,
        lat=float(lat),
        lon=float(lon),
        place_id=place_id,
        source="query",
    )


def _google_contact_search(client: httpx.Client, origin: LocationContext, kind: str) -> list[dict[str, Any]]:
    key = _google_api_key()
    if not key:
        return []
    keywords = {
        "vet": ["veterinary", "vet", "animal hospital"],
        "rescue": [
            "animal rescue",
            "rescue",
            "shelter",
            "animal shelter",
            "animal welfare",
            "animal aid",
            "animal help",
            "animal welfare trust",
            "animal welfare society",
            "street animal",
            "ngo",
            "foundation",
            "trust",
            "society",
        ],
    }.get(kind, ["vet"])
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keyword in keywords:
        radius_meters = 5000 if kind == "vet" else 15000
        payload = _google_request_json(
            client,
            _google_nearby_url(),
            {
                "location": f"{origin.lat},{origin.lon}",
                "radius": radius_meters,
                "keyword": keyword,
                "language": "en",
                "key": key,
            },
        )
        if payload.get("status") not in {"OK", "ZERO_RESULTS"}:
            continue
        items = payload.get("results")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            place_id = _extract_text(item.get("place_id"))
            if not place_id or place_id in seen:
                continue
            seen.add(place_id)
            details = _google_place_details(client, place_id) or {}
            geometry = details.get("geometry") or item.get("geometry") or {}
            loc = geometry.get("location") if isinstance(geometry, dict) else {}
            lat = loc.get("lat") if isinstance(loc, dict) else None
            lng = loc.get("lng") if isinstance(loc, dict) else None
            if lat is None or lng is None:
                continue
            distance_km = haversine_km(origin.lat, origin.lon, float(lat), float(lng))
            name = _extract_text(details.get("name"), item.get("name")) or ("Nearby Vet" if kind == "vet" else "Nearby Rescue")
            address = _extract_text(
                details.get("formatted_address"),
                item.get("vicinity"),
                item.get("formatted_address"),
            ) or "Unknown area"
            contact_info = details if isinstance(details, dict) else {}
            phone = _extract_text(contact_info.get("formatted_phone_number"), contact_info.get("international_phone_number"))
            website = _extract_text(contact_info.get("website"))
            opening_hours = None
            if isinstance(contact_info.get("opening_hours"), dict):
                opening_hours = _extract_text(contact_info["opening_hours"].get("weekday_text"))
            results.append(
                {
                    "name": name,
                    "address": address,
                    "phone": phone or "Not available",
                    "website": website,
                    "distance_km": round(distance_km, 3),
                    "distance_label": distance_label(distance_km),
                    "maps_link": f"https://www.google.com/maps/search/?api=1&query={float(lat)},{float(lng)}",
                    "opening_hours": opening_hours,
                    "lat": float(lat),
                    "lon": float(lng),
                }
            )
            if len(results) >= DEFAULT_RESULT_LIMIT:
                return results
    results.sort(key=lambda item: item.get("distance_km") if isinstance(item.get("distance_km"), (int, float)) else float("inf"))
    return results[:DEFAULT_RESULT_LIMIT]


def _mappls_request_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict[str, Any]:
    token = _mappls_access_token()
    if not token:
        return {}
    request_params = dict(params)
    request_params["access_token"] = token
    response = client.get(url, params=request_params)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _mappls_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("suggestedLocations", "userAddedLocations", "results", "places"):
        items = payload.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _mappls_pick_item(query: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = _match_text(query)
    if not items:
        return None
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        name = _match_text(item.get("placeName") or item.get("name") or item.get("place_name"))
        address = _match_text(item.get("placeAddress") or item.get("address") or item.get("place_address"))
        score = 0
        if name == normalized:
            score += 100
        if address == normalized:
            score += 90
        if normalized and normalized in name:
            score += 40
        if normalized and normalized in address:
            score += 30
        if name.startswith(normalized[:4]):
            score += 10
        scored.append((score, -index, item))
    scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    return scored[0][2] if scored else items[0]


def _mappls_item_coordinates(item: dict[str, Any]) -> tuple[float | None, float | None]:
    lat_candidates = [
        item.get("latitude"),
        item.get("lat"),
        item.get("entryLatitude"),
    ]
    lon_candidates = [
        item.get("longitude"),
        item.get("lon"),
        item.get("lng"),
        item.get("entryLongitude"),
    ]
    lat = next((candidate for candidate in lat_candidates if candidate is not None), None)
    lon = next((candidate for candidate in lon_candidates if candidate is not None), None)
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None


def _mappls_geocode_text(client: httpx.Client, address: str, *, bias_lat: float | None = None, bias_lon: float | None = None) -> tuple[float | None, float | None]:
    query = (address or "").strip()
    if not query or not _has_mappls():
        return None, None
    params: dict[str, Any] = {
        "address": query,
        "itemCount": 1,
        "bias": 1,
        "region": "IND",
    }
    if bias_lat is not None and bias_lon is not None:
        params["location"] = f"{bias_lat},{bias_lon}"
    try:
        payload = _mappls_request_json(client, _mappls_geocode_url(), params)
    except Exception:
        return None, None
    items = _mappls_items(payload)
    for item in items:
        lat, lon = _mappls_item_coordinates(item)
        if lat is not None and lon is not None:
            return lat, lon
    return None, None


def _mappls_location_context(client: httpx.Client, location: str) -> LocationContext | None:
    normalized = _normalize_query(location)
    if not normalized or not _has_mappls():
        return None
    anchor_item, _ = _supported_location_anchor(normalized)
    bias_lat = float(anchor_item["lat"]) if anchor_item is not None else None
    bias_lon = float(anchor_item["lon"]) if anchor_item is not None else None
    queries = [
        normalized,
        f"{normalized}, Maharashtra, India",
        f"{normalized}, Mumbai, Maharashtra, India",
        f"{normalized}, Borivali, Mumbai, Maharashtra, India",
        f"{normalized}, Dahisar, Mumbai, Maharashtra, India",
    ]
    for query in queries:
        try:
            payload = _mappls_request_json(
                client,
                _mappls_textsearch_url(),
                {
                    "query": query,
                    "region": "IND",
                    **({"location": f"{bias_lat},{bias_lon}"} if bias_lat is not None and bias_lon is not None else {}),
                },
            )
            items = _mappls_items(payload)
            if not items:
                payload = _mappls_request_json(
                    client,
                    _mappls_autosuggest_url(),
                    {
                        "query": query,
                        "location": f"{bias_lat},{bias_lon}" if bias_lat is not None and bias_lon is not None else None,
                    },
                )
                items = _mappls_items(payload)
            picked = _mappls_pick_item(normalized, items)
            if not picked:
                continue
            lat, lon = _mappls_item_coordinates(picked)
            if lat is None or lon is None:
                address = _extract_text(picked.get("placeAddress"), picked.get("place_name"), picked.get("placeName"), picked.get("address"))
                if address:
                    geocode = _mappls_request_json(
                        client,
                        _mappls_geocode_url(),
                        {
                            "address": address,
                            "itemCount": 1,
                            "bias": 1,
                            "podFilter": "hna",
                            "region": "IND",
                        },
                    )
                    geocode_items = _mappls_items(geocode)
                    geocode_picked = geocode_items[0] if geocode_items else None
                    if geocode_picked:
                        lat, lon = _mappls_item_coordinates(geocode_picked)
                        if lat is None or lon is None:
                            lat, lon = _mappls_item_coordinates(picked)
            if lat is None or lon is None:
                continue
            label = _extract_text(
                picked.get("placeName"),
                picked.get("name"),
                picked.get("place_name"),
                picked.get("placeAddress"),
                normalized,
            ) or normalized
            if _is_supported_region(lat, lon) or anchor_item is not None:
                return LocationContext(
                    label=label,
                    lat=lat,
                    lon=lon,
                    place_id=_extract_text(picked.get("eLoc"), picked.get("place_id"), picked.get("placeId")),
                    source="query",
                )
        except Exception:
            continue
    return None


def _mappls_reverse_location(client: httpx.Client, lat: float, lon: float) -> LocationContext | None:
    if not _has_mappls():
        return None
    try:
        payload = _mappls_request_json(
            client,
            _mappls_reverse_geocode_url(),
            {
                "lat": lat,
                "lng": lon,
                "region": "IND",
            },
        )
    except Exception:
        return None
    items = _mappls_items(payload)
    item = items[0] if items else payload
    if not isinstance(item, dict):
        return None
    resolved_lat, resolved_lon = _mappls_item_coordinates(item)
    label = _extract_text(
        item.get("formattedAddress"),
        item.get("placeAddress"),
        item.get("formatted_address"),
        item.get("placeName"),
        item.get("name"),
    ) or "Current location"
    return LocationContext(
        label=label,
        lat=resolved_lat if resolved_lat is not None else lat,
        lon=resolved_lon if resolved_lon is not None else lon,
        place_id=_extract_text(item.get("eLoc"), item.get("place_id"), item.get("placeId")),
        source="gps",
    )


def _mappls_contact_search(client: httpx.Client, origin: LocationContext, kind: str) -> list[dict[str, Any]]:
    if not _has_mappls():
        return []
    origin_text = _match_text(origin.label)
    keyword_sets = {
        "vet": ["veterinary"],
        "rescue": ["animal rescue", "rescue", "animal welfare"],
    }
    fallback_keywords = {
        "vet": ["vet", "animal hospital", "animal clinic", "pet clinic", "pet hospital"],
        "rescue": [
            "rescue",
            "shelter",
            "animal shelter",
            "animal welfare",
            "animal aid",
            "animal help",
            "welfare organization",
            "spca",
            "ngo",
            "foundation",
            "trust",
            "society",
        ],
    }
    keywords = keyword_sets.get(kind, ["vet"])
    fallback = fallback_keywords.get(kind, [])
    results: list[dict[str, Any]] = []

    def fetch_keyword(keyword: str, radius_meters: int) -> list[dict[str, Any]]:
        local_results: list[dict[str, Any]] = []
        local_seen: set[str] = set()
        with _client() as keyword_client:
            try:
                payload = _mappls_request_json(
                    keyword_client,
                    _mappls_nearby_url(),
                    {
                        "keywords": keyword,
                        "refLocation": f"{origin.lat},{origin.lon}",
                        "region": "IND",
                        "radius": radius_meters,
                        "page": 1,
                    },
                )
            except Exception:
                return []
            for item in _mappls_items(payload):
                name = _extract_text(item.get("placeName"), item.get("name"))
                address = _extract_text(item.get("placeAddress"), item.get("address")) or "Unknown area"
                e_loc = _extract_text(item.get("eLoc"), item.get("eloc"), item.get("placeId"))
                key = (e_loc or name or address).casefold()
                if not key or key in local_seen:
                    continue
                local_seen.add(key)
                lat, lon = _mappls_item_coordinates(item)
                distance_km = None
                if lat is not None and lon is not None:
                    distance_km = haversine_km(origin.lat, origin.lon, lat, lon)
                    if distance_km * 1000 > radius_meters:
                        continue
                phone = _extract_text(item.get("mobileNo"), item.get("landlineNo"), item.get("phone"))
                website = _extract_text(item.get("website"))
                maps_target = address or name or origin.label
                local_results.append(
                    {
                        "id": len(local_results) + 1,
                        "name": name or ("Nearby Vet" if kind == "vet" else "Nearby Rescue"),
                        "address": address,
                        "phone": phone or "Not available",
                        "area": address,
                        "distance_km": round(distance_km, 3) if distance_km is not None else None,
                        "distance_label": distance_label(distance_km) if distance_km is not None else None,
                        "website": website,
                        "opening_hours": _extract_text(item.get("hourOfOperation"), item.get("opening_hours")),
                        "maps_link": map_link(lat, lon) if lat is not None and lon is not None else f"https://www.google.com/maps/search/?api=1&query={maps_target}",
                    }
                )
                if len(local_results) >= DEFAULT_RESULT_LIMIT:
                    break
        return local_results

    def fetch_keywords(radius_meters: int, active_keywords: list[str]) -> list[dict[str, Any]]:
        if not active_keywords:
            return []
        local_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(2, len(active_keywords))) as executor:
            futures = [executor.submit(fetch_keyword, keyword, radius_meters) for keyword in active_keywords]
            for future in as_completed(futures):
                try:
                    local_results.extend(future.result() or [])
                except Exception:
                    continue
        deduped: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        for item in local_results:
            key = f"{_match_text(item.get('name'))}|{_match_text(item.get('address'))}|{item.get('maps_link') or ''}"
            if key in seen_items:
                continue
            seen_items.add(key)
            deduped.append(item)
        deduped.sort(key=lambda item: item.get("distance_km") if isinstance(item.get("distance_km"), (int, float)) else float("inf"))
        return deduped[:DEFAULT_RESULT_LIMIT]

    nearby_radius = MAPPLS_NEARBY_CONTACT_RADIUS_METERS
    nearby_results = fetch_keywords(nearby_radius, keywords[:2])
    if nearby_results:
        return nearby_results

    far_results = fetch_keywords(MAPPLS_FAR_CONTACT_RADIUS_METERS, keywords[:2] + fallback[:2])
    return far_results


def _contact_radius_for_origin(origin: LocationContext) -> int:
    origin_text = _match_text(origin.label)
    if any(token in origin_text for token in ("ic colony", "eksar", "gorai", "st francis", "mount poinsur", "shimpoli", "saibaba nagar", "chikuwadi", "vazira naka", "mandapeshwar", "ketkipada", "rawalpada", "nancy colony", "ashok van")):
        return 2200
    if any(token in origin_text for token in ("borivali west", "borivali east", "dahisar west", "dahisar east", "borivali", "dahisar")):
        return 3500
    return 5000


def _cache_get(cache: dict, key: tuple[Any, ...], ttl_seconds: int):
    if ttl_seconds <= 0:
        return None
    import time

    record = cache.get(key)
    if not record:
        return None
    stored_at, value = record
    if time.time() - stored_at > ttl_seconds:
        cache.pop(key, None)
        return None
    return value


def _cache_set(cache: dict, key: tuple[Any, ...], value, ttl_seconds: int):
    if ttl_seconds <= 0:
        return
    import time

    cache[key] = (time.time(), value)


def _extract_features(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    features = payload.get("features")
    if isinstance(features, list):
        return [feature for feature in features if isinstance(feature, dict)]
    results = payload.get("results")
    if isinstance(results, list):
        return [feature for feature in results if isinstance(feature, dict)]
    return []


def _feature_properties(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties")
    if isinstance(props, dict):
        return props
    return feature


def _feature_coordinates(feature: dict[str, Any]) -> tuple[float | None, float | None]:
    props = _feature_properties(feature)
    lat = props.get("lat")
    lon = props.get("lon")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        try:
            return float(coordinates[1]), float(coordinates[0])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _extract_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _match_text(value: Any) -> str:
    return str(value or "").casefold().strip()


def _contact_keywords(kind: str) -> tuple[str, ...]:
    return VET_CONTACT_KEYWORDS if kind == "vet" else RESCUE_CONTACT_KEYWORDS


def _matches_contact_kind(kind: str, *values: Any) -> bool:
    text = " ".join(_match_text(value) for value in values if value is not None)
    if not text:
        return False
    return any(keyword in text for keyword in _contact_keywords(kind))


def _format_address(props: dict[str, Any]) -> str:
    return _extract_text(
        props.get("formatted"),
        props.get("address_line1"),
        " ".join(
            part
            for part in [
                props.get("housenumber"),
                props.get("street"),
                props.get("city"),
            ]
            if part
        ),
        props.get("city"),
        props.get("state"),
        props.get("name"),
    ) or "Unknown area"


def _format_location_label(props: dict[str, Any]) -> str:
    return _extract_text(
        props.get("formatted"),
        props.get("address_line2"),
        props.get("city"),
        props.get("suburb"),
        props.get("district"),
        props.get("county"),
        props.get("name"),
    ) or "Current location"


def _extract_contact_data(props: dict[str, Any]) -> dict[str, Any]:
    contact = props.get("contact")
    if not isinstance(contact, dict):
        contact = {}
    phone = _extract_text(
        contact.get("phone"),
        *(contact.get("phone_other") or []) if isinstance(contact.get("phone_other"), list) else [],
        props.get("phone"),
    )
    website = _extract_text(
        contact.get("website"),
        props.get("website"),
    )
    email = _extract_text(
        contact.get("email"),
        props.get("email"),
    )
    opening_hours = _extract_text(
        props.get("opening_hours"),
        contact.get("opening_hours"),
    )
    return {
        "phone": phone,
        "website": website,
        "email": email,
        "opening_hours": opening_hours,
    }


def _normalize_query(location: str | None) -> str | None:
    if not location:
        return None
    text = location.strip()
    if not text:
        return None

    near_match = re.search(r"(?i)\b(?:near|nearby|around|close to|closest to|in|at)\b\s+(.+)$", text)
    if near_match:
        text = near_match.group(1).strip()

    text = re.sub(
        r"(?i)\b(?:show|get|find|search|locate|nearest|nearby|near|around|vet|vets|veterinary|clinic|clinics|rescue|rescues|animal rescue|animal contacts?|contact|contacts|numbers?|please)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,.-")

    generic = {
        "",
        "vet",
        "vets",
        "veterinary",
        "clinic",
        "clinics",
        "rescue",
        "rescues",
        "animal rescue",
        "animal",
        "contact",
        "contacts",
        "nearby vet",
        "nearby rescue",
    }
    if text.lower() in generic:
        return None
    return text or None


def _locality_variants_for_query(query: str | None) -> list[str]:
    normalized = _normalize_query(query)
    if not normalized:
        return []
    text = _match_text(normalized)
    variants: list[str] = []
    for key, values in LOCALITY_VARIANT_SUGGESTIONS.items():
        key_text = _match_text(key)
        if (
            re.search(rf"\b{re.escape(key_text)}\b", text)
            or text.startswith(key_text[:4])
            or key_text.startswith(text)
            or any(_match_text(alias).startswith(text) for alias in values)
            or any(text.startswith(_match_text(alias).split()[0]) for alias in values if _match_text(alias))
            or SequenceMatcher(None, text, key_text).ratio() >= 0.78
            or any(SequenceMatcher(None, text, _match_text(alias)).ratio() >= 0.78 for alias in values)
        ):
            variants.extend(values)
    seen: set[str] = set()
    ordered: list[str] = []
    for value in variants:
        cleaned = value.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(cleaned)
    return ordered


def _candidate_location_queries(normalized: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for query in [
        normalized,
        f"{normalized}, Maharashtra, India",
        f"{normalized}, Mumbai, Maharashtra, India",
        f"{normalized}, Thane, Maharashtra, India",
    ]:
        cleaned = query.strip().strip(",")
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        queries.append(cleaned)
    return queries


def _supported_location_anchor(location: str | None) -> tuple[dict[str, Any] | None, str | None]:
    normalized = _normalize_query(location)
    if not normalized:
        return None, None
    text = _match_text(normalized)
    candidates: list[tuple[int, dict[str, Any], str]] = []
    fuzzy_candidates: list[tuple[float, int, dict[str, Any], str]] = []
    for item in SUPPORTED_LOCATION_GROUPS:
        aliases = item.get("aliases") or []
        for alias in aliases:
            alias_text = _match_text(alias)
            if not alias_text:
                continue
            if re.search(rf"\b{re.escape(alias_text)}\b", text):
                candidates.append((len(alias_text), item, alias_text))
                break
            similarity = SequenceMatcher(None, text, alias_text).ratio()
            if similarity >= 0.78 or alias_text.startswith(text[:4]) or text.startswith(alias_text[:4]):
                fuzzy_candidates.append((similarity, len(alias_text), item, alias_text))
    if not candidates:
        if fuzzy_candidates:
            _, _, item, alias = sorted(fuzzy_candidates, key=lambda pair: (pair[0], pair[1]), reverse=True)[0]
            return item, alias
        return None, None
    _, item, alias = sorted(candidates, key=lambda pair: pair[0], reverse=True)[0]
    return item, alias


def _try_precise_geocode_location(client: httpx.Client, location: str) -> LocationContext | None:
    normalized = _normalize_query(location)
    if not normalized:
        return None
    for query in _candidate_location_queries(normalized):
        resolved = _geocode_location(client, query)
        if resolved and _is_supported_region(resolved.lat, resolved.lon):
            return resolved
    return None


def _resolve_supported_city_query(location: str | None) -> LocationContext | None:
    normalized = _normalize_query(location)
    if not normalized:
        return None
    text = normalized.lower()
    anchor_item, anchor_alias = _supported_location_anchor(normalized)
    if anchor_item is not None:
        label = str(anchor_item["label"])
        city = str(anchor_item["city"])
        if anchor_alias:
            geocode_queries = [
                f"{normalized}, Maharashtra, India",
                f"{anchor_alias}, {city}, Maharashtra, India",
                f"{label}, {city}, Maharashtra, India",
            ]
            try:
                if httpx is not None:
                    with _client() as client:
                        for query in geocode_queries:
                            resolved = _geocode_location(client, query)
                            if resolved and _is_supported_region(resolved.lat, resolved.lon):
                                return LocationContext(
                                    label=resolved.label or label,
                                    lat=resolved.lat,
                                    lon=resolved.lon,
                                    place_id=resolved.place_id,
                                    source="query",
                                )
            except Exception:
                pass
        return LocationContext(label=label, lat=float(anchor_item["lat"]), lon=float(anchor_item["lon"]), source="query")

    if httpx is not None:
        try:
            with _client() as client:
                resolved = _try_precise_geocode_location(client, normalized)
                if resolved:
                    return resolved
        except Exception:
            pass
    candidates: list[tuple[int, dict[str, Any]]] = []
    for item in SUPPORTED_LOCATION_GROUPS:
        aliases = item.get("aliases") or []
        for alias in aliases:
            alias_text = str(alias).lower().strip()
            if not alias_text:
                continue
            pattern = rf"\b{re.escape(alias_text)}\b"
            if re.search(pattern, text):
                candidates.append((len(alias_text), item))
                break
    if candidates:
        _, item = sorted(candidates, key=lambda pair: pair[0], reverse=True)[0]
        label = str(item["label"])
        city = str(item["city"])
        geocode_queries = [f"{normalized}, Maharashtra, India"]
        if normalized != label.lower():
            geocode_queries.append(f"{normalized}, {label}, {city}, Maharashtra, India")
            geocode_queries.append(f"{normalized}, {city}, Maharashtra, India")
        geocode_queries.append(f"{label}, {city}, Maharashtra, India")
        try:
            if httpx is not None:
                with _client() as client:
                    for query in geocode_queries:
                        resolved = _geocode_location(client, query)
                        if resolved and _is_supported_region(resolved.lat, resolved.lon):
                            return LocationContext(
                                label=resolved.label or label,
                                lat=resolved.lat,
                                lon=resolved.lon,
                                place_id=resolved.place_id,
                                source="query",
                            )
        except Exception:
            pass
        return LocationContext(label=label, lat=float(item["lat"]), lon=float(item["lon"]), source="query")

    if httpx is not None and len(text) >= 3:
        try:
            with _client() as client:
                for query in _candidate_location_queries(normalized):
                    resolved = _geocode_location(client, query)
                    if resolved and _is_supported_region(resolved.lat, resolved.lon):
                        return resolved
        except Exception:
            pass
    return None


def _is_supported_region(lat: float, lon: float) -> bool:
    for item in SUPPORTED_LOCATION_GROUPS:
        distance = haversine_km(lat, lon, float(item["lat"]), float(item["lon"]))
        if distance <= SUPPORTED_REGION_RADIUS_KM:
            return True
    return False


def _request_json(client: httpx.Client, url: str, params: dict[str, Any], *, method: str = "GET", data: str | None = None) -> dict[str, Any]:
    response = client.request(method, url, params=params, content=data)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _request_overpass(client: httpx.Client, query: str) -> dict[str, Any]:
    payload = _request_json(
        client,
        settings.overpass_url,
        {},
        method="POST",
        data=query,
    )
    return payload if isinstance(payload, dict) else {}


def _geocode_location(client: httpx.Client, query: str) -> LocationContext | None:
    payload = _request_json(
        client,
        settings.geoapify_geocode_url,
        {
            "text": query,
            "format": "geojson",
            "lang": "en",
            "filter": f"countrycode:{settings.geoapify_country_code}",
            "apiKey": _api_key(),
        },
    )
    features = _extract_features(payload)
    if not features:
        return None
    feature = features[0]
    props = _feature_properties(feature)
    lat, lon = _feature_coordinates(feature)
    if lat is None or lon is None:
        return None
    return LocationContext(
        label=_format_location_label(props),
        lat=lat,
        lon=lon,
        place_id=_extract_text(props.get("place_id")),
        source="query",
    )


def _reverse_geocode(client: httpx.Client, lat: float, lon: float) -> LocationContext | None:
    payload = _request_json(
        client,
        settings.geoapify_reverse_geocode_url,
        {
            "lat": lat,
            "lon": lon,
            "format": "geojson",
            "lang": "en",
            "apiKey": _api_key(),
        },
    )
    features = _extract_features(payload)
    if not features:
        return None
    feature = features[0]
    props = _feature_properties(feature)
    resolved_lat, resolved_lon = _feature_coordinates(feature)
    return LocationContext(
        label=_format_location_label(props),
        lat=resolved_lat if resolved_lat is not None else lat,
        lon=resolved_lon if resolved_lon is not None else lon,
        place_id=_extract_text(props.get("place_id")),
        source="gps",
    )


def _ip_geolocate(client: httpx.Client, client_ip: str | None) -> LocationContext | None:
    params: dict[str, Any] = {
        "apiKey": _api_key(),
    }
    if client_ip:
        params["ip"] = client_ip
    payload = _request_json(client, settings.geoapify_ip_geolocation_url, params)
    location = payload.get("location")
    if not isinstance(location, dict):
        return None
    lat = location.get("lat")
    lon = location.get("lng") or location.get("lon")
    if lat is None or lon is None:
        return None
    label_parts = [
        _extract_text(location.get("city")),
        _extract_text(location.get("region")),
    ]
    label = ", ".join(part for part in label_parts if part) or "Current location"
    try:
        return LocationContext(label=label, lat=float(lat), lon=float(lon), source="ip")
    except (TypeError, ValueError):
        return None


def _resolve_location(client: httpx.Client, location: str | None, lat: float | None, lon: float | None, client_ip: str | None) -> LocationContext | None:
    if lat is not None and lon is not None:
        try:
            if _has_mappls():
                resolved = _mappls_reverse_location(client, float(lat), float(lon))
                if resolved:
                    return resolved
            if settings.geoapify_api_key:
                resolved = _reverse_geocode(client, float(lat), float(lon))
                if resolved:
                    return resolved
        except Exception:
            pass
        return LocationContext(label="Current location", lat=float(lat), lon=float(lon), source="gps")

    if location:
        anchor_item, _ = _supported_location_anchor(location)
        if anchor_item is not None:
            normalized = _normalize_query(location) or str(anchor_item["label"])
            normalized_text = _match_text(normalized)
            label = normalized if " " in normalized_text else str(anchor_item["label"])
            return LocationContext(
                label=label,
                lat=float(anchor_item["lat"]),
                lon=float(anchor_item["lon"]),
                source="query",
            )
        try:
            if _has_mappls():
                mappls_resolved = _mappls_location_context(client, location)
                if mappls_resolved:
                    return mappls_resolved
        except Exception:
            pass

        try:
            if _has_google_places():
                google_resolved = _google_location_context(client, location)
                if google_resolved:
                    return google_resolved
        except Exception:
            pass

        try:
            resolved = _try_precise_geocode_location(client, location)
            if resolved:
                return resolved
        except Exception:
            pass

        supported = _resolve_supported_city_query(location)
        if supported:
            return supported

        return None

    try:
        return _ip_geolocate(client, client_ip)
    except Exception:
        return None


def preview_location_resolution(
    location: str | None,
    lat: float | None,
    lon: float | None,
    *,
    client_ip: str | None = None,
) -> dict[str, Any]:
    status = "unknown"
    message: str | None = None
    resolved_label: str | None = None
    resolved_lat: float | None = None
    resolved_lon: float | None = None

    if lat is not None and lon is not None:
        status = "current_location"
        message = "Current location detected."
        resolved_label = "Current location"
        resolved_lat = float(lat)
        resolved_lon = float(lon)
        return {
            "status": status,
            "message": message,
            "label": resolved_label,
            "lat": resolved_lat,
            "lon": resolved_lon,
        }

    normalized = _normalize_query(location)
    if normalized:
        anchor_item, _ = _supported_location_anchor(normalized)
        if anchor_item is not None:
            normalized_text = _match_text(normalized)
            label = normalized if " " in normalized_text else str(anchor_item["label"])
            return {
                "status": "nearest_supported_match",
                "message": f"Nearest supported match: {label}",
                "label": label,
                "lat": float(anchor_item["lat"]),
                "lon": float(anchor_item["lon"]),
            }
        if _has_mappls():
            try:
                with _client() as client:
                    mappls_resolved = _mappls_location_context(client, normalized)
                    if mappls_resolved:
                        exact = normalized.lower() == (mappls_resolved.label or "").strip().lower()
                        status = "exact_place_found" if exact else "nearest_supported_match"
                        message = (
                            f"Exact place found: {mappls_resolved.label}"
                            if exact
                            else f"Nearest supported match: {mappls_resolved.label}"
                        )
                        return {
                            "status": status,
                            "message": message,
                            "label": mappls_resolved.label,
                            "lat": mappls_resolved.lat,
                            "lon": mappls_resolved.lon,
                        }
            except Exception:
                pass

        if _has_google_places():
            try:
                with _client() as client:
                    google_resolved = _google_location_context(client, normalized)
                    if google_resolved:
                        exact = normalized.lower() == (google_resolved.label or "").strip().lower()
                        status = "exact_place_found" if exact else "nearest_supported_match"
                        message = (
                            f"Exact place found: {google_resolved.label}"
                            if exact
                            else f"Nearest supported match: {google_resolved.label}"
                        )
                        return {
                            "status": status,
                            "message": message,
                            "label": google_resolved.label,
                            "lat": google_resolved.lat,
                            "lon": google_resolved.lon,
                        }
            except Exception:
                pass

        if httpx is not None:
            try:
                with _client() as client:
                    precise = _try_precise_geocode_location(client, normalized)
                    if precise:
                        exact = normalized.lower() == (precise.label or "").strip().lower()
                        status = "exact_place_found" if exact else "nearest_supported_match"
                        message = (
                            f"Exact place found: {precise.label}"
                            if exact
                            else f"Nearest supported match: {precise.label}"
                        )
                        return {
                            "status": status,
                            "message": message,
                            "label": precise.label,
                            "lat": precise.lat,
                            "lon": precise.lon,
                        }
            except Exception:
                pass

        supported = _resolve_supported_city_query(location)
        if supported:
            status = "nearest_supported_match"
            message = f"Nearest supported match: {supported.label}"
            return {
                "status": status,
                "message": message,
                "label": supported.label,
                "lat": supported.lat,
                "lon": supported.lon,
            }

        status = "location_not_found"
        message = "Location not found accurately, showing nearest match."
        return {
            "status": status,
            "message": message,
            "label": normalized,
            "lat": resolved_lat,
            "lon": resolved_lon,
        }

    if httpx is not None:
        try:
            with _client() as client:
                ip_location = _ip_geolocate(client, client_ip)
                if ip_location:
                    return {
                        "status": "current_location",
                        "message": "Current location detected.",
                        "label": ip_location.label,
                        "lat": ip_location.lat,
                        "lon": ip_location.lon,
                    }
        except Exception:
            pass

    return {
        "status": status,
        "message": message,
        "label": resolved_label,
        "lat": resolved_lat,
        "lon": resolved_lon,
    }


def _place_search(
    client: httpx.Client,
    origin: LocationContext,
    categories: str,
    *,
    name: str | None = None,
    use_place_filter: bool = True,
    limit: int = DEFAULT_RESULT_LIMIT,
    radius_meters: int = FAR_CONTACT_RADIUS_METERS,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "categories": categories,
        "lang": "en",
        "limit": limit,
        "apiKey": _api_key(),
        "bias": f"proximity:{origin.lon},{origin.lat}",
    }
    if name:
        params["name"] = name
    if use_place_filter and origin.place_id:
        params["filter"] = f"place:{origin.place_id}"
    else:
        params["filter"] = f"circle:{origin.lon},{origin.lat},{radius_meters}"

    payload = _request_json(client, settings.geoapify_places_url, params)
    return _extract_features(payload)


def _osm_element_coordinates(element: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    center = element.get("center")
    if isinstance(center, dict):
        lat = center.get("lat")
        lon = center.get("lon")
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except (TypeError, ValueError):
                pass
    return None, None


def _osm_element_tags(element: dict[str, Any]) -> dict[str, Any]:
    tags = element.get("tags")
    if isinstance(tags, dict):
        return tags
    return {}


def _overpass_query(kind: str, origin: LocationContext, radius_meters: int) -> str:
    if kind == "vet":
        filters = [
            'node["amenity"="veterinary"](around:{radius},{lat},{lon});',
            'way["amenity"="veterinary"](around:{radius},{lat},{lon});',
            'relation["amenity"="veterinary"](around:{radius},{lat},{lon});',
            'node["healthcare"="veterinary"](around:{radius},{lat},{lon});',
            'way["healthcare"="veterinary"](around:{radius},{lat},{lon});',
            'relation["healthcare"="veterinary"](around:{radius},{lat},{lon});',
            'node["office"="veterinary"](around:{radius},{lat},{lon});',
            'way["office"="veterinary"](around:{radius},{lat},{lon});',
            'relation["office"="veterinary"](around:{radius},{lat},{lon});',
            'node["shop"="pet"]["name"~"(vet|veterinary|animal hospital|animal clinic|pet clinic|pet hospital)",i](around:{radius},{lat},{lon});',
            'way["shop"="pet"]["name"~"(vet|veterinary|animal hospital|animal clinic|pet clinic|pet hospital)",i](around:{radius},{lat},{lon});',
            'relation["shop"="pet"]["name"~"(vet|veterinary|animal hospital|animal clinic|pet clinic|pet hospital)",i](around:{radius},{lat},{lon});',
            'node["amenity"="clinic"]["name"~"(vet|veterinary|animal hospital|pet clinic)",i](around:{radius},{lat},{lon});',
            'way["amenity"="clinic"]["name"~"(vet|veterinary|animal hospital|pet clinic)",i](around:{radius},{lat},{lon});',
            'relation["amenity"="clinic"]["name"~"(vet|veterinary|animal hospital|pet clinic)",i](around:{radius},{lat},{lon});',
            'node["healthcare"="clinic"]["name"~"(vet|veterinary|animal hospital|pet clinic)",i](around:{radius},{lat},{lon});',
            'way["healthcare"="clinic"]["name"~"(vet|veterinary|animal hospital|pet clinic)",i](around:{radius},{lat},{lon});',
            'relation["healthcare"="clinic"]["name"~"(vet|veterinary|animal hospital|pet clinic)",i](around:{radius},{lat},{lon});',
        ]
    else:
        filters = [
            'node["amenity"="animal_shelter"](around:{radius},{lat},{lon});',
            'way["amenity"="animal_shelter"](around:{radius},{lat},{lon});',
            'relation["amenity"="animal_shelter"](around:{radius},{lat},{lon});',
            'node["social_facility"="shelter"](around:{radius},{lat},{lon});',
            'way["social_facility"="shelter"](around:{radius},{lat},{lon});',
            'relation["social_facility"="shelter"](around:{radius},{lat},{lon});',
            'node["office"="animal_welfare"](around:{radius},{lat},{lon});',
            'way["office"="animal_welfare"](around:{radius},{lat},{lon});',
            'relation["office"="animal_welfare"](around:{radius},{lat},{lon});',
            'node["shop"="pet"]["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare|pet rescue)",i](around:{radius},{lat},{lon});',
            'way["shop"="pet"]["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare|pet rescue)",i](around:{radius},{lat},{lon});',
            'relation["shop"="pet"]["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare|pet rescue)",i](around:{radius},{lat},{lon});',
            'node["office"="ngo"]["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare)",i](around:{radius},{lat},{lon});',
            'way["office"="ngo"]["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare)",i](around:{radius},{lat},{lon});',
            'relation["office"="ngo"]["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare)",i](around:{radius},{lat},{lon});',
            'node["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare)",i](around:{radius},{lat},{lon});',
            'way["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare)",i](around:{radius},{lat},{lon});',
            'relation["name"~"(rescue|shelter|welfare|spca|animal aid|animal welfare)",i](around:{radius},{lat},{lon});',
        ]
    wrapped = "\n".join(filters)
    return f"""
[out:json][timeout:20];
(
{wrapped.format(radius=radius_meters, lat=origin.lat, lon=origin.lon)}
);
out center tags;
""".strip()


def _search_kind_overpass(client: httpx.Client, origin: LocationContext, kind: str, *, radius_meters: int) -> list[dict[str, Any]]:
    try:
        payload = _request_overpass(client, _overpass_query(kind, origin, radius_meters))
    except Exception:
        return []
    elements = payload.get("elements")
    if not isinstance(elements, list):
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in elements:
        if not isinstance(element, dict):
            continue
        tags = _osm_element_tags(element)
        lat, lon = _osm_element_coordinates(element)
        if lat is None or lon is None:
            continue
        name = _extract_text(tags.get("name"), tags.get("operator"), tags.get("brand"))
        if not _matches_contact_kind(kind, name, tags.get("operator"), tags.get("brand"), tags.get("amenity"), tags.get("office"), tags.get("shop")):
            continue
        key = _extract_text(tags.get("name"), tags.get("operator"), tags.get("brand"), tags.get("wikidata"))
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "properties": {
                    "name": name or ("Nearby Vet" if kind == "vet" else "Nearby Rescue"),
                    "formatted": _extract_text(tags.get("addr:full"), tags.get("addr:street"), tags.get("addr:city"), tags.get("addr:suburb"), name) or "Unknown area",
                    "contact": {
                        "phone": _extract_text(tags.get("contact:phone"), tags.get("phone"), tags.get("telephone")),
                        "website": _extract_text(tags.get("contact:website"), tags.get("website")),
                        "opening_hours": _extract_text(tags.get("opening_hours")),
                    },
                    "distance": int(round(haversine_km(origin.lat, origin.lon, lat, lon) * 1000)),
                    "lat": lat,
                    "lon": lon,
                }
            }
        )
        if len(results) >= DEFAULT_RESULT_LIMIT:
            break
    return results


def _expand_search_terms(kind: str) -> list[str | None]:
    if kind == "vet":
        return [
            None,
            "vet",
            "veterinary",
            "animal hospital",
            "animal clinic",
            "pet clinic",
            "pet hospital",
            "animal doctor",
        ]
    return [
        None,
        "rescue",
        "animal rescue",
        "shelter",
        "animal shelter",
        "welfare",
        "spca",
        "animal aid",
        "pet rescue",
    ]


def _location_specific_search_terms(origin: LocationContext, kind: str) -> list[str | None]:
    origin_label = _match_text(origin.label)
    terms: list[str | None] = []

    if kind == "vet":
        if "borivali" in origin_label:
            terms.extend(
                [
                    "Vetic",
                    "Sai Pet Clinic",
                    "The Kozy Vet",
                    "Pet Animal Clinic",
                    "borivali vet",
                    "borivali veterinary",
                    "borivali animal hospital",
                    "borivali animal clinic",
                    "borivali pet clinic",
                    "borivali west vet",
                    "borivali east vet",
                    "ic colony vet",
                    "eksar vet",
                    "gorai vet",
                    "saibaba nagar vet",
                    "shimpoli vet",
                    "new link road vet",
                    "sv road borivali vet",
                    "borivali station vet",
                ]
            )
        if "dahisar" in origin_label or "mount poinsur" in origin_label or "st francis" in origin_label:
            terms.extend(
                [
                    "Blue7 Vets",
                    "Dr Vikram Veterinary Clinic",
                    "Pets Clinic",
                    "dahisar vet",
                    "dahisar veterinary",
                    "dahisar animal hospital",
                    "dahisar animal clinic",
                    "dahisar pet clinic",
                    "dahisar east vet",
                    "dahisar west vet",
                    "st francis institute of technology vet",
                    "mount poinsur vet",
                    "mandapeshwar vet",
                    "ketkipada vet",
                    "rawalpada vet",
                    "nancy colony vet",
                    "ashok van vet",
                    "sv road dahisar vet",
                ]
            )
    else:
        if "borivali" in origin_label:
            terms.extend(
                [
                    "borivali rescue",
                    "borivali animal rescue",
                    "borivali shelter",
                    "borivali spca",
                ]
            )
        if "dahisar" in origin_label or "mount poinsur" in origin_label or "st francis" in origin_label:
            terms.extend(
                [
                    "dahisar rescue",
                    "dahisar animal rescue",
                    "dahisar shelter",
                    "dahisar spca",
                ]
            )

    seen: set[str] = set()
    ordered: list[str | None] = []
    for term in terms:
        lowered = (term or "").strip().lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(term)
    return ordered


def _search_kind(client: httpx.Client, origin: LocationContext, kind: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    origin_text = _match_text(origin.label)
    is_targeted_region = any(token in origin_text for token in ("borivali", "dahisar"))

    def collect(features: list[dict[str, Any]], *, max_distance_km: float | None = None) -> None:
        for feature in features:
            props = _feature_properties(feature)
            if not _matches_contact_kind(kind, props.get("name"), props.get("formatted"), props.get("address_line1"), props.get("address_line2"), props.get("city"), props.get("suburb"), props.get("district"), props.get("county")):
                continue
            place_id = _extract_text(props.get("place_id"))
            key = place_id or _extract_text(props.get("name"), props.get("formatted"))
            if not key or key in seen:
                continue
            lat, lon = _feature_coordinates(feature)
            if lat is None or lon is None:
                continue
            distance_km = haversine_km(origin.lat, origin.lon, lat, lon)
            if max_distance_km is not None and distance_km > max_distance_km:
                continue
            seen.add(key)
            results.append(feature)
            if len(results) >= DEFAULT_RESULT_LIMIT:
                return

    search_terms = _expand_search_terms(kind)
    search_terms = _location_specific_search_terms(origin, kind) + search_terms
    category_groups = VET_CONTACT_CATEGORY_GROUPS if kind == "vet" else RESCUE_CONTACT_CATEGORY_GROUPS
    if is_targeted_region:
        search_terms = search_terms[:4]
        category_groups = category_groups[:2]

    def search_round(*, max_distance_km: float | None, radius_meters: int) -> None:
        use_place_filters = (False,) if is_targeted_region else (False, True)
        for categories in category_groups:
            if len(results) >= DEFAULT_RESULT_LIMIT:
                return
            for term in search_terms:
                if len(results) >= DEFAULT_RESULT_LIMIT:
                    return
                for use_place_filter in use_place_filters:
                    if len(results) >= DEFAULT_RESULT_LIMIT:
                        return
                    try:
                        features = _place_search(
                            client,
                            origin,
                            categories,
                            name=term,
                            use_place_filter=use_place_filter,
                            limit=max(20, DEFAULT_RESULT_LIMIT),
                            radius_meters=radius_meters,
                        )
                    except Exception:
                        features = []
                    collect(features, max_distance_km=max_distance_km)
                    if results:
                        return
                    if len(results) >= DEFAULT_RESULT_LIMIT:
                        return

    search_round(max_distance_km=5.0, radius_meters=NEARBY_CONTACT_RADIUS_METERS)
    if results:
        return results

    search_round(max_distance_km=None, radius_meters=FAR_CONTACT_RADIUS_METERS)
    if results:
        return results

    overpass_results = _search_kind_overpass(
        client,
        origin,
        kind,
        radius_meters=OVERPASS_NEARBY_RADIUS_METERS,
    )
    collect(overpass_results, max_distance_km=6.0)
    if results:
        return results

    overpass_results = _search_kind_overpass(
        client,
        origin,
        kind,
        radius_meters=OVERPASS_FAR_RADIUS_METERS,
    )
    collect(overpass_results)
    return results


def _fetch_place_details(client: httpx.Client, feature: dict[str, Any]) -> dict[str, Any]:
    return {}


def _build_contact(
    client: httpx.Client,
    feature: dict[str, Any],
    origin: LocationContext,
    kind: str,
    index: int,
) -> dict[str, Any] | None:
    props = _feature_properties(feature)
    lat, lon = _feature_coordinates(feature)
    if lat is None or lon is None:
        return None

    details = {}
    detail_fields = {}
    place_fields = _extract_contact_data(props)

    name = _extract_text(
        props.get("name"),
        details.get("name") if isinstance(details, dict) else None,
        props.get("address_line1"),
        props.get("formatted"),
    ) or ("Nearby Vet" if kind == "vet" else "Nearby Rescue")

    formatted_address = _extract_text(
        details.get("formatted") if isinstance(details, dict) else None,
        props.get("formatted"),
        props.get("address_line1"),
        props.get("address_line2"),
        props.get("city"),
        props.get("suburb"),
        props.get("district"),
        props.get("county"),
    ) or "Unknown area"

    if lat is None or lon is None:
        lookup_query = formatted_address if formatted_address != "Unknown area" else name
        resolved_contact_location = None
        if lookup_query and httpx is not None:
            try:
                if _has_mappls():
                    resolved_contact_location = _mappls_location_context(client, lookup_query)
                if resolved_contact_location is None and _has_google_places():
                    resolved_contact_location = _google_location_context(client, lookup_query)
                if resolved_contact_location is None:
                    resolved_contact_location = _try_precise_geocode_location(client, lookup_query)
            except Exception:
                resolved_contact_location = None
        if resolved_contact_location is not None:
            lat = resolved_contact_location.lat
            lon = resolved_contact_location.lon

    distance = props.get("distance")
    distance_km = None
    if isinstance(distance, (int, float)):
        distance_km = float(distance) / 1000.0
    else:
        distance_km = haversine_km(origin.lat, origin.lon, lat, lon)

    contact_data = detail_fields or place_fields
    phone = contact_data.get("phone") or "Not available"
    website = contact_data.get("website")
    opening_hours = contact_data.get("opening_hours")
    email = contact_data.get("email")

    if kind == "vet":
        return {
            "id": index,
            "name": name,
            "address": formatted_address,
            "phone": phone,
            "area": formatted_address,
            "distance_km": round(distance_km, 3),
            "distance_label": distance_label(distance_km),
            "website": website,
            "opening_hours": opening_hours,
            "maps_link": map_link(lat, lon),
        }

    return {
        "id": index,
        "name": name,
        "phone": phone,
        "email": email,
        "area": formatted_address,
        "address": formatted_address,
        "distance_km": round(distance_km, 3),
        "distance_label": distance_label(distance_km),
        "website": website,
        "opening_hours": opening_hours,
        "maps_link": map_link(lat, lon),
    }


def get_contacts_for_area(
    db,
    location: str | None,
    lat: float | None,
    lng: float | None,
    *,
    client_ip: str | None = None,
    resolved_origin: LocationContext | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if httpx is None:
        return [], []

    def dedupe_contacts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        for item in items:
            key = "|".join(
                [
                    _match_text(item.get("name")),
                    _match_text(item.get("address") or item.get("area")),
                    _match_text(item.get("maps_link")),
                ]
            )
            if not key.strip("|"):
                continue
            if key in seen_items:
                continue
            seen_items.add(key)
            deduped.append(item)
        return deduped

    cache_key = (
        (_normalize_query(location) or "").lower(),
        round(float(lat), 4) if lat is not None else None,
        round(float(lng), 4) if lng is not None else None,
        (client_ip or "").strip(),
    )
    cached = _cache_get(_CONTACT_CACHE, cache_key, CONTACT_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    def load_fallback_contacts(kind: str, origin: LocationContext) -> list[dict[str, Any]]:
        features: list[dict[str, Any]] = []
        try:
            features = _search_kind(client, origin, kind)
        except Exception:
            features = []
        contacts: list[dict[str, Any]] = []
        for index, feature in enumerate(features, start=1):
            try:
                contact = _build_contact(client, feature, origin, kind, index)
            except Exception:
                contact = None
            if contact:
                contacts.append(contact)
        return contacts

    def cache_and_return(rescue_result: list[dict[str, Any]], vet_result: list[dict[str, Any]]):
        rescue_result = dedupe_contacts(rescue_result)
        vet_result = dedupe_contacts(vet_result)
        rescue_result.sort(key=lambda item: item.get("distance_km") if isinstance(item.get("distance_km"), (int, float)) else float("inf"))
        vet_result.sort(key=lambda item: item.get("distance_km") if isinstance(item.get("distance_km"), (int, float)) else float("inf"))
        result = (rescue_result, vet_result)
        _cache_set(_CONTACT_CACHE, cache_key, result, CONTACT_CACHE_TTL_SECONDS)
        return result

    rescue_contacts: list[dict[str, Any]] = []
    vet_contacts: list[dict[str, Any]] = []

    try:
        with _client() as client:
            origin = resolved_origin or _resolve_location(client, location, lat, lng, client_ip)
            if origin is None:
                return [], []

            if _has_mappls():
                try:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        rescue_future = executor.submit(_mappls_contact_search, client, origin, "rescue")
                        vet_future = executor.submit(_mappls_contact_search, client, origin, "vet")
                        try:
                            rescue_contacts = rescue_future.result()
                        except Exception:
                            rescue_contacts = []
                        try:
                            vet_contacts = vet_future.result()
                        except Exception:
                            vet_contacts = []
                except Exception:
                    rescue_contacts = []
                    vet_contacts = []
                if rescue_contacts or vet_contacts:
                    return cache_and_return(rescue_contacts, vet_contacts)

            if _has_google_places():
                try:
                    rescue_contacts = _google_contact_search(client, origin, "rescue")
                except Exception:
                    rescue_contacts = []
                try:
                    vet_contacts = _google_contact_search(client, origin, "vet")
                except Exception:
                    vet_contacts = []
                if rescue_contacts or vet_contacts:
                    return cache_and_return(rescue_contacts, vet_contacts)

            try:
                rescue_features = _search_kind(client, origin, "rescue")
            except Exception:
                rescue_features = []
            try:
                vet_features = _search_kind(client, origin, "vet")
            except Exception:
                vet_features = []

            for index, feature in enumerate(rescue_features, start=1):
                try:
                    contact = _build_contact(client, feature, origin, "rescue", index)
                except Exception:
                    contact = None
                if contact:
                    rescue_contacts.append(contact)

            for index, feature in enumerate(vet_features, start=1):
                try:
                    contact = _build_contact(client, feature, origin, "vet", index)
                except Exception:
                    contact = None
                if contact:
                    vet_contacts.append(contact)
    except Exception:
        return [], []

    return cache_and_return(rescue_contacts, vet_contacts)


def autocomplete_locations(query: str, *, limit: int = 12) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    if not normalized:
        return []
    cache_key = (_match_text(normalized), int(limit))
    cached = _cache_get(_AUTOCOMPLETE_CACHE, cache_key, AUTOCOMPLETE_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    suggestions: list[dict[str, Any]] = []
    query_text = _match_text(normalized)
    precise_terms = {"road", "lane", "street", "st", "st.", "society", "complex", "tower", "building", "bldg", "floor", "flat", "apartment", "near", "opposite", "next to", "opp", "hno", "house"}
    should_prefer_precise = any(term in query_text for term in precise_terms) or len(query_text.split()) > 1
    anchor_item, _ = _supported_location_anchor(normalized)

    locality_variants = _locality_variants_for_query(normalized)
    if locality_variants:
        for variant in locality_variants[:limit]:
            variant_label = variant
            variant_lat = None
            variant_lon = None
            variant_secondary = None
            variant_city = str(anchor_item["city"]) if anchor_item is not None else "Mumbai"
            if httpx is not None:
                try:
                    with _client() as client:
                        variant_queries = [variant]
                        if anchor_item is not None:
                            variant_queries.insert(0, f"{variant}, {anchor_item['city']}, Maharashtra, India")
                        for variant_query in variant_queries:
                            resolved = _mappls_location_context(client, variant_query) if _has_mappls() else None
                            if resolved is None and _has_google_places():
                                resolved = _google_location_context(client, variant_query)
                            if resolved is None:
                                resolved = _try_precise_geocode_location(client, variant_query)
                            if resolved and _is_supported_region(resolved.lat, resolved.lon):
                                variant_label = resolved.label or variant
                                variant_lat = round(float(resolved.lat), 6)
                                variant_lon = round(float(resolved.lon), 6)
                                variant_secondary = f"{variant}, {variant_city}, Maharashtra"
                                break
                except Exception:
                    pass
            if variant_lat is None and anchor_item is not None:
                variant_lat = round(float(anchor_item["lat"]), 6)
                variant_lon = round(float(anchor_item["lon"]), 6)
                variant_secondary = f"{variant}, {variant_city}, Maharashtra"
            if variant_lat is None and anchor_item is None:
                variant_secondary = f"{variant}, Mumbai, Maharashtra"
            suggestions.append(
                {
                    "label": variant_label,
                    "main_text": variant_label,
                    "secondary_text": variant_secondary or f"{variant_city}, Maharashtra",
                    "lat": variant_lat,
                    "lon": variant_lon,
                    "place_id": None,
                    "address": variant_label,
                }
            )

    if _has_mappls():
        try:
            with _client() as client:
                bias_item, _ = _supported_location_anchor(normalized)
                bias_lat = float(bias_item["lat"]) if bias_item is not None else None
                bias_lon = float(bias_item["lon"]) if bias_item is not None else None
                candidate_queries = [
                    normalized,
                    f"{normalized}, Borivali, Mumbai, Maharashtra, India",
                    f"{normalized}, Dahisar, Mumbai, Maharashtra, India",
                ]
                for candidate_query in candidate_queries:
                    payload = _mappls_request_json(
                        client,
                        _mappls_autosuggest_url(),
                        {
                            "query": candidate_query,
                            **({"location": f"{bias_lat},{bias_lon}"} if bias_lat is not None and bias_lon is not None else {}),
                        },
                    )
                    items = _mappls_items(payload)
                    if not items:
                        payload = _mappls_request_json(
                            client,
                            _mappls_textsearch_url(),
                            {
                                "query": candidate_query,
                                "region": "IND",
                                **({"location": f"{bias_lat},{bias_lon}"} if bias_lat is not None and bias_lon is not None else {}),
                            },
                        )
                        items = _mappls_items(payload)
                    for item in items[: max(5, limit)]:
                        label = _extract_text(item.get("placeName"), item.get("place_name"), item.get("name"), candidate_query) or candidate_query
                        address = _extract_text(item.get("placeAddress"), item.get("address"), item.get("formattedAddress"), label) or label
                        lat, lon = _mappls_item_coordinates(item)
                        if lat is None or lon is None:
                            geocode_payload = _mappls_request_json(
                                client,
                                _mappls_geocode_url(),
                                {
                                    "address": address,
                                    "itemCount": 1,
                                    "bias": 1,
                                    "region": "IND",
                                },
                            )
                            geocode_items = _mappls_items(geocode_payload)
                            geocode_item = geocode_items[0] if geocode_items else None
                            if geocode_item:
                                lat, lon = _mappls_item_coordinates(geocode_item)
                                if geocode_item.get("placeName"):
                                    label = _extract_text(geocode_item.get("placeName"), label) or label
                                if geocode_item.get("placeAddress"):
                                    address = _extract_text(geocode_item.get("placeAddress"), address) or address
                        if lat is None or lon is None:
                            continue
                        if not _is_supported_region(lat, lon):
                            continue
                        suggestions.append(
                            {
                                "label": label,
                                "main_text": label,
                                "secondary_text": address,
                                "lat": round(float(lat), 6),
                                "lon": round(float(lon), 6),
                                "place_id": _extract_text(item.get("eLoc"), item.get("eloc"), item.get("placeId")),
                                "address": address,
                        }
                    )
        except Exception:
            pass

    if _has_mappls() and suggestions and len(suggestions) >= max(5, min(limit, 8)):
        deduped: list[dict[str, Any]] = []
        seen_entries: set[tuple[str, str, str]] = set()
        for item in suggestions:
            label = str(item.get("label") or "").strip()
            lat = "" if item.get("lat") is None else str(item.get("lat"))
            lon = "" if item.get("lon") is None else str(item.get("lon"))
            key = (label.lower(), lat, lon)
            if not label or key in seen_entries:
                continue
            seen_entries.add(key)
            deduped.append(item)
        _cache_set(_AUTOCOMPLETE_CACHE, cache_key, deduped[:limit], AUTOCOMPLETE_CACHE_TTL_SECONDS)
        return deduped[:limit]

    if _has_google_places() and not _has_mappls():
        try:
            with _client() as client:
                bias_item, _ = _supported_location_anchor(normalized)
                bias_lat = float(bias_item["lat"]) if bias_item is not None else None
                bias_lon = float(bias_item["lon"]) if bias_item is not None else None
                google_predictions = _google_autocomplete(
                    client,
                    normalized,
                    limit=limit,
                    bias_lat=bias_lat,
                    bias_lon=bias_lon,
                    bias_radius_m=40000,
                )
                for prediction in google_predictions:
                    place_id = _extract_text(prediction.get("place_id"))
                    structured = prediction.get("structured_formatting") if isinstance(prediction.get("structured_formatting"), dict) else {}
                    label = _extract_text(
                        structured.get("main_text"),
                        prediction.get("description"),
                    ) or normalized.title()
                    address = _extract_text(structured.get("secondary_text"), prediction.get("description")) or label
                    lat = None
                    lon = None
                    if place_id:
                        details = _google_place_details(client, place_id)
                        if details:
                            geometry = details.get("geometry") or {}
                            location_data = geometry.get("location") if isinstance(geometry, dict) else {}
                            if isinstance(location_data, dict):
                                lat = location_data.get("lat")
                                lon = location_data.get("lng")
                            label = _extract_text(details.get("name")) or label
                            address = _extract_text(details.get("formatted_address")) or address
                    suggestions.append(
                        {
                            "label": label,
                            "main_text": label,
                            "secondary_text": address,
                            "lat": round(float(lat), 6) if lat is not None else None,
                            "lon": round(float(lon), 6) if lon is not None else None,
                            "place_id": place_id,
                            "address": address,
                        }
                    )
        except Exception:
            pass

    if httpx is not None and should_prefer_precise:
        try:
            with _client() as client:
                precise = _mappls_location_context(client, normalized) if _has_mappls() else _try_precise_geocode_location(client, normalized)
                if precise:
                    suggestions.append(
                        {
                            "label": precise.label or normalized.title(),
                            "main_text": precise.label or normalized.title(),
                            "secondary_text": _format_location_label({"formatted": precise.label or normalized.title()}),
                            "lat": round(precise.lat, 6),
                            "lon": round(precise.lon, 6),
                            "place_id": precise.place_id,
                            "address": precise.label or normalized.title(),
                        }
                    )
        except Exception:
            pass

    if not locality_variants:
        for item in SUPPORTED_LOCATION_GROUPS:
            aliases = [_match_text(alias) for alias in item.get("aliases", [])]
            if any(alias and (alias in query_text or query_text in alias) for alias in aliases):
                suggestions.append(
                    {
                        "label": f'{item["label"]}, {item["city"]}',
                        "lat": round(float(item["lat"]), 6),
                        "lon": round(float(item["lon"]), 6),
                        "place_id": None,
                    }
                )

    if not suggestions and httpx is not None and len(query_text) >= 3:
        try:
            with _client() as client:
                resolved = _mappls_location_context(client, normalized) if _has_mappls() else _try_precise_geocode_location(client, normalized)
                if resolved:
                    suggestions.append(
                        {
                            "label": resolved.label or normalized.title(),
                            "main_text": resolved.label or normalized.title(),
                            "secondary_text": resolved.label or normalized.title(),
                            "lat": round(resolved.lat, 6),
                            "lon": round(resolved.lon, 6),
                            "place_id": resolved.place_id,
                            "address": resolved.label or normalized.title(),
                        }
                    )
        except Exception:
            pass

    deduped: list[dict[str, Any]] = []
    seen_entries: set[tuple[str, str, str]] = set()
    for item in suggestions:
        label = str(item.get("label") or "").strip()
        lat = "" if item.get("lat") is None else str(item.get("lat"))
        lon = "" if item.get("lon") is None else str(item.get("lon"))
        key = (label.lower(), lat, lon)
        if not label or key in seen_entries:
            continue
        seen_entries.add(key)
        deduped.append(item)

    _cache_set(_AUTOCOMPLETE_CACHE, cache_key, deduped[:limit], AUTOCOMPLETE_CACHE_TTL_SECONDS)
    return deduped[:limit]
