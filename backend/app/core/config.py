from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Animal Health Detection API"
    sqlite_db_path: Path = Field(
        default=Path("database/animal_health.db"),
        validation_alias=AliasChoices("DATABASE_PATH", "SQLITE_DB_PATH"),
    )
    upload_dir: str = "uploads"
    frontend_origin: str = "*"
    custom_yolo_model_path: str = "models/domestic_animals_yolov8n.pt"
    yolo_model_path: str = "yolov8n.pt"
    health_model_path: str = "models/health_classifier.pt"
    species_model_path: str = "models/species_classifier.pt"
    detector_imgsz: int = 640
    detector_confidence_threshold: float = 0.30
    detector_max_detections: int = 8
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-5-mini"
    openai_vision_model: str = "gpt-4o-mini"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_chat_model: str = "llama-3.1-8b-instant"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    geoapify_api_key: str | None = None
    geoapify_geocode_url: str = "https://api.geoapify.com/v1/geocode/search"
    geoapify_reverse_geocode_url: str = "https://api.geoapify.com/v1/geocode/reverse"
    geoapify_places_url: str = "https://api.geoapify.com/v2/places"
    geoapify_place_details_url: str = "https://api.geoapify.com/v2/place-details"
    geoapify_ip_geolocation_url: str = "https://api.geoapify.com/v1/ipinfo"
    geoapify_search_radius_meters: int = 15000
    geoapify_country_code: str = "in"
    google_maps_api_key: str | None = None
    upi_vpa: str | None = None
    upi_payee_name: str = "Paw Connect"
    upi_note: str = "Support street animal care"
    nominatim_url: str = "https://nominatim.openstreetmap.org/reverse"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    app_user_agent: str = "PawConnect/1.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def upload_path(self) -> Path:
        return self.project_root / self.upload_dir

    @property
    def database_path(self) -> Path:
        if self.sqlite_db_path.is_absolute():
            return self.sqlite_db_path
        return self.project_root / self.sqlite_db_path

    @property
    def legacy_sqlite_path(self) -> Path:
        return self.database_path


settings = Settings()
