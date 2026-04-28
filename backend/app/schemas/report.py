from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DetectionBox(BaseModel):
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None


class RescueContactOut(BaseModel):
    id: int
    name: str
    phone: str
    email: Optional[str] = None
    area: str
    address: Optional[str] = None
    distance_km: Optional[float] = None
    distance_label: Optional[str] = None
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    maps_link: Optional[str] = None


class VetContactOut(BaseModel):
    id: int
    name: str
    address: str
    phone: str
    area: str
    distance_km: Optional[float] = None
    distance_label: Optional[str] = None
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    maps_link: Optional[str] = None


class SearchedLocationOut(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class NearbyContactsResponse(BaseModel):
    location_name: Optional[str] = None
    location_status: str = "unknown"
    location_message: Optional[str] = None
    searched_location: Optional[SearchedLocationOut] = None
    rescue_contacts: list[RescueContactOut] = Field(default_factory=list)
    vet_contacts: list[VetContactOut] = Field(default_factory=list)


class ReportCreate(BaseModel):
    user_id: Optional[int] = None
    image_path: str
    analysis_status: str = "animal_detected"
    animal_type: Optional[str] = None
    animal_name: Optional[str] = None
    health_status: str = "NotApplicable"
    confidence_score: float = 0.0
    detection_confidence: float = 0.0
    guidance: str
    detected_conditions: list[str] = Field(default_factory=list)
    animal_reports: list[dict] = Field(default_factory=list)
    location_name: Optional[str] = None
    location_address: Optional[str] = None
    rescue_requested: bool = False
    rescue_status: str = "not_requested"
    location_lat: Optional[float] = None
    location_long: Optional[float] = None
    bbox: DetectionBox = Field(default_factory=DetectionBox)


class ReportUpdate(BaseModel):
    rescue_status: Optional[str] = None
    animal_name: Optional[str] = None


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    report_id: int
    user_id: Optional[int]
    image_path: str
    analysis_status: str
    animal_type: Optional[str]
    animal_name: Optional[str] = None
    health_status: str
    confidence_score: float
    detection_confidence: float
    bbox_x1: Optional[int]
    bbox_y1: Optional[int]
    bbox_x2: Optional[int]
    bbox_y2: Optional[int]
    guidance: str
    detected_conditions: list[str] = Field(default_factory=list)
    animal_reports: list[dict] = Field(default_factory=list)
    animal_detected: Optional[str] = None
    location_name: Optional[str]
    location_address: Optional[str]
    rescue_requested: bool
    rescue_status: str
    location_lat: Optional[float]
    location_long: Optional[float]
    health_score: int = 0
    urgency_level: str = "none"
    urgency_label: str = "No action needed"
    primary_issues: list[str] = Field(default_factory=list)
    visible_symptoms: list[str] = Field(default_factory=list)
    body_condition: str = ""
    animal_description: str = ""
    injury_description: str = ""
    breed_guess: Optional[str] = None
    what_is_wrong: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    needs_rescue: bool = False
    help_type: str = "none"
    triage_reasoning: str = ""
    emergency_plan: dict = Field(default_factory=dict)
    avoid_steps: list[str] = Field(default_factory=list)
    contact_priority: str = ""
    health_status_code: str = "unknown"
    created_at: datetime


class PredictionResponse(BaseModel):
    report_id: int
    image_path: str
    image_url: str
    analysis_status: str
    is_animal: bool
    animal_type: Optional[str]
    animal_detected: Optional[str] = None
    animal_name: Optional[str] = None
    location_name: Optional[str]
    location_address: Optional[str]
    location_lat: Optional[float] = None
    location_long: Optional[float] = None
    detection_confidence: float
    health_status: str
    health_status_code: str = "unknown"
    health_confidence: float
    health_score: int = 0
    urgency_level: str = "none"
    urgency_label: str = "No action needed"
    bounding_box: DetectionBox
    guidance: str
    health_summary: str
    condition_summary: str
    primary_issues: list[str] = Field(default_factory=list)
    visible_symptoms: list[str] = Field(default_factory=list)
    body_condition: str = ""
    animal_description: str = ""
    injury_description: str = ""
    breed_guess: Optional[str] = None
    what_is_wrong: str = ""
    recommended_actions: list[str]
    needs_rescue: bool
    help_type: str = "none"
    triage_reasoning: str = ""
    emergency_plan: dict = Field(default_factory=dict)
    avoid_steps: list[str] = Field(default_factory=list)
    contact_priority: str = ""
    needs_help: bool = False
    detected_conditions: list[str]
    rescue_prompt: str
    rescue_contacts: list[RescueContactOut]
    vet_contacts: list[VetContactOut]
    animal_reports: list[dict]
    other_detections: list[dict]
