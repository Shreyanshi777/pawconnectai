from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Report:
    report_id: int
    user_id: int | None
    image_path: str
    analysis_status: str
    animal_type: str | None
    health_status: str
    confidence_score: float
    detection_confidence: float
    bbox_x1: int | None
    bbox_y1: int | None
    bbox_x2: int | None
    bbox_y2: int | None
    guidance: str
    detected_conditions: str | None = None
    animal_reports_json: str | None = None
    location_name: str | None = None
    location_address: str | None = None
    rescue_requested: bool = False
    rescue_status: str = "not_requested"
    location_lat: float | None = None
    location_long: float | None = None
    animal_name: str | None = None
    created_at: datetime | None = None

    @property
    def detected_conditions_list(self) -> list[str]:
        if not self.detected_conditions:
            return []
        try:
            value = json.loads(self.detected_conditions)
            return value if isinstance(value, list) else []
        except Exception:
            return []

    @property
    def animal_reports_list(self) -> list[dict[str, Any]]:
        if not self.animal_reports_json:
            return []
        try:
            value = json.loads(self.animal_reports_json)
            return value if isinstance(value, list) else []
        except Exception:
            return []


@dataclass(slots=True)
class RescueContact:
    rescue_id: int
    name: str
    phone: str
    email: str | None
    area: str


@dataclass(slots=True)
class VetContact:
    vet_id: int
    name: str
    address: str
    phone: str
    area: str
