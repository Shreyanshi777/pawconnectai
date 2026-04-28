from __future__ import annotations

from pathlib import Path

from PIL import ExifTags, Image


GPS_TAG = next((key for key, value in ExifTags.TAGS.items() if value == "GPSInfo"), None)


def extract_gps_from_image(image_path: str | Path) -> tuple[float | None, float | None]:
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        if not exif or GPS_TAG not in exif:
            return None, None

        gps_info_raw = exif.get(GPS_TAG)
        gps_info = {}
        for key, value in gps_info_raw.items():
            gps_info[ExifTags.GPSTAGS.get(key, key)] = value

        lat = _convert_gps(gps_info.get("GPSLatitude"), gps_info.get("GPSLatitudeRef"))
        lon = _convert_gps(gps_info.get("GPSLongitude"), gps_info.get("GPSLongitudeRef"))
        return lat, lon
    except Exception:
        return None, None


def _convert_gps(value, ref) -> float | None:
    if not value or not ref:
        return None

    try:
        degrees = float(value[0][0]) / float(value[0][1])
        minutes = float(value[1][0]) / float(value[1][1])
        seconds = float(value[2][0]) / float(value[2][1])
        result = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in {"S", "W"}:
            result *= -1
        return result
    except Exception:
        return None
