"""Explicitly attributed weather providers for the dashboard and API.

Open-Meteo current conditions are model-derived, not station observations.
IMD responses are used only when an operator supplies an access token.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import os
import time
from typing import Any


DELHI = {"name": "Delhi", "latitude": 28.6139, "longitude": 77.2090}
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
IMD_URL = "https://api.imd.gov.in/api/v1"


class WeatherSourceError(RuntimeError):
    """Provider unavailable or returned an invalid response."""


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.4,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "WeatherIntelligenceResearch/1.0"})
    return session


def _get_json(url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> dict:
    import requests
    try:
        response = _session().get(url, params=params, headers=headers, timeout=(5, 20))
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise WeatherSourceError(f"Weather provider request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise WeatherSourceError("Weather provider returned an unexpected response.")
    return payload


@lru_cache(maxsize=16)
def _open_meteo_cached(city: str, lat: float, lon: float, hour_bucket: int) -> dict:
    del hour_bucket  # cache key bounds repeat requests to at most one per hour
    payload = _get_json(OPEN_METEO_URL, {
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability,precipitation,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code",
        "forecast_days": 7,
    })
    if not isinstance(payload.get("current"), dict) or not isinstance(payload.get("daily"), dict):
        raise WeatherSourceError("Open-Meteo response is missing current or daily data.")
    return {"city": city, "provider": "Open-Meteo", "product": "numerical weather model",
            "attribution": "Open-Meteo Forecast API", "source_url": "https://open-meteo.com/en/docs",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "timezone": payload.get("timezone"),
            "current": payload["current"], "hourly": payload.get("hourly", {}), "daily": payload["daily"]}


def open_meteo_forecast(city: str = "Delhi", latitude: float = DELHI["latitude"],
                        longitude: float = DELHI["longitude"]) -> dict:
    """Return current model conditions and seven-day forecast, cached hourly."""
    return _open_meteo_cached(city, float(latitude), float(longitude), int(time.time() // 3600))


def imd_current_weather(station_id: str | None = None) -> dict:
    """Fetch official IMD current weather when IMD_API_TOKEN is configured.

    Set IMD_API_AUTH_HEADER, IMD_API_AUTH_VALUE and IMD_STATION_ID using the
    exact auth scheme and credentials specified in the account's IMD portal.
    The public reference lists the endpoint but does not document an auth scheme.
    """
    auth_name = os.getenv("IMD_API_AUTH_HEADER")
    auth_value = os.getenv("IMD_API_AUTH_VALUE")
    station = station_id or os.getenv("IMD_STATION_ID")
    if not auth_name or not auth_value or not station:
        raise WeatherSourceError("IMD current weather is not configured; set IMD_API_AUTH_HEADER, IMD_API_AUTH_VALUE and IMD_STATION_ID per your IMD account.")
    payload = _get_json(f"{IMD_URL}/current_wx", {"id": station}, {auth_name: auth_value})
    return {"provider": "India Meteorological Department", "product": "station observation",
            "attribution": "IMD Current Weather API", "source_url": "https://api.imd.gov.in/public/api_reference.html",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "data": payload}

