from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile

from backend.app.core.config import settings
from backend.app.schemas.chat import MedicalChatRequest, MedicalChatResponse
from backend.app.db.session import get_db
from backend.app.schemas.report import DetectionBox, NearbyContactsResponse, PredictionResponse, ReportCreate, ReportOut, ReportUpdate, SearchedLocationOut
from backend.app.services.ai_pipeline import SUPPORTED_DOMESTIC_ANIMALS, pipeline
from backend.app.services.contact_service import autocomplete_locations, get_contacts_for_area, preview_location_resolution
from backend.app.services.crud import create_report, delete_all_reports, delete_report, get_report, get_reports, update_report
from backend.app.services.geo_service import reverse_geocode
from backend.app.services.guidance import guidance_for_status, health_summary_for_status
from backend.app.services.medical_chat import medical_chat_reply
from backend.app.services.reporting import build_case_report
from backend.app.services.vision_report import generate_vision_case_report
from backend.app.utils.image_metadata import extract_gps_from_image
from backend.app.utils.file_utils import save_upload_file

router = APIRouter()


def health_status_code_from_score(score: int) -> str:
    if score >= 85:
        return "healthy"
    if score >= 65:
        return "mild_injury"
    if score >= 40:
        return "moderate_injury"
    if score >= 20:
        return "severe_injury"
    return "critical"


def report_to_schema(report) -> ReportOut:
    case_report = build_case_report(
        animal_type=report.animal_type,
        health_status=report.health_status,
        detected_conditions=report.detected_conditions_list,
        location_address=report.location_address,
        confidence=report.confidence_score,
    )
    first_animal_report = report.animal_reports_list[0] if report.animal_reports_list else {}
    first_case_report = build_case_report(
        animal_type=first_animal_report.get("animal_type", report.animal_type),
        health_status=first_animal_report.get("health_status", report.health_status),
        detected_conditions=first_animal_report.get("detected_conditions", report.detected_conditions_list),
        location_address=report.location_address,
        confidence=first_animal_report.get("health_confidence", report.confidence_score),
    )
    animal_reports = [{
        **first_animal_report,
        "animal_type": first_animal_report.get("animal_type", report.animal_type),
        "animal_name": first_animal_report.get("animal_name", report.animal_name),
        "breed_guess": first_animal_report.get("breed_guess"),
        "animal_description": first_animal_report.get("animal_description", ""),
        "injury_description": first_animal_report.get("injury_description", ""),
        "health_status": first_animal_report.get("health_status", report.health_status),
        "health_score": first_case_report["health_score"],
        "urgency_level": first_case_report["urgency_level"],
        "urgency_label": first_case_report["urgency_label"],
        "primary_issues": first_case_report["primary_issues"],
        "visible_symptoms": first_case_report["visible_symptoms"],
        "body_condition": first_case_report["body_condition"],
        "what_is_wrong": first_case_report["what_is_wrong"],
        "help_type": first_case_report["help_type"],
        "triage_reasoning": first_case_report["triage_reasoning"],
        "emergency_plan": first_case_report["emergency_plan"],
        "avoid_steps": first_case_report["avoid_steps"],
        "contact_priority": first_case_report["contact_priority"],
    }] if report.animal_reports_list else []

    return ReportOut(
        report_id=report.report_id,
        user_id=report.user_id,
        image_path=report.image_path,
        analysis_status=report.analysis_status,
        animal_type=report.animal_type,
        animal_name=report.animal_name,
        animal_detected=report.animal_type,
        health_status=report.health_status,
        confidence_score=report.confidence_score,
        detection_confidence=report.detection_confidence,
        bbox_x1=report.bbox_x1,
        bbox_y1=report.bbox_y1,
        bbox_x2=report.bbox_x2,
        bbox_y2=report.bbox_y2,
        guidance=report.guidance,
        detected_conditions=report.detected_conditions_list,
        animal_reports=animal_reports,
        health_score=case_report["health_score"],
        urgency_level=case_report["urgency_level"],
        urgency_label=case_report["urgency_label"],
        primary_issues=case_report["primary_issues"],
        visible_symptoms=case_report["visible_symptoms"],
        body_condition=case_report["body_condition"],
        animal_description=case_report.get("animal_description", ""),
        injury_description=case_report.get("injury_description", ""),
        breed_guess=case_report.get("breed_guess"),
        what_is_wrong=case_report["what_is_wrong"],
        recommended_actions=case_report["recommended_actions"],
        needs_rescue=case_report["needs_rescue"],
        help_type=case_report["help_type"],
        triage_reasoning=case_report["triage_reasoning"],
        emergency_plan=case_report["emergency_plan"],
        avoid_steps=case_report["avoid_steps"],
        contact_priority=case_report["contact_priority"],
        health_status_code=health_status_code_from_score(case_report["health_score"]),
        location_name=report.location_name,
        location_address=report.location_address,
        rescue_requested=report.rescue_requested,
        rescue_status=report.rescue_status,
        location_lat=report.location_lat,
        location_long=report.location_long,
        created_at=report.created_at,
    )


def to_url_path(path_value: str) -> str:
    return Path(str(path_value)).as_posix().lstrip("/")


@router.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}


@router.get("/public-config")
def public_config():
    return {
        "upi_vpa": settings.upi_vpa,
        "upi_payee_name": settings.upi_payee_name,
        "upi_note": settings.upi_note,
    }


@router.get("/contacts/nearby", response_model=NearbyContactsResponse)
def nearby_contacts(
    request: Request,
    response: Response,
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)
    location_preview = preview_location_resolution(location, lat, lng, client_ip=client_ip)
    rescue_contacts, vet_contacts = get_contacts_for_area(None, location, lat, lng, client_ip=client_ip)
    resolved_name = location_preview.get("label") or location or ("Current location" if lat is not None and lng is not None else None)
    if rescue_contacts or vet_contacts:
        location_status = location_preview.get("status") or "unknown"
        location_message = location_preview.get("message")
    else:
        location_status = "no_live_contacts"
        location_message = location_preview.get("message") or "No live contacts found within range."
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return NearbyContactsResponse(
        location_name=resolved_name,
        location_status=location_status,
        location_message=location_message,
        searched_location=SearchedLocationOut(
            name=location_preview.get("label") or resolved_name,
            address=location_preview.get("message") if location_preview.get("status") == "location_not_found" else None,
            lat=location_preview.get("lat"),
            lng=location_preview.get("lon"),
        ),
        rescue_contacts=rescue_contacts,
        vet_contacts=vet_contacts,
    )


@router.get("/locations/autocomplete")
def autocomplete_locations_route(response: Response, q: Optional[str] = None):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return {"suggestions": autocomplete_locations(q or "")}


@router.get("/db-health")
def db_health(db = Depends(get_db)):
    try:
        cursor = db.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM reports")
            report_count = cursor.fetchone()[0]
        finally:
            cursor.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {exc}") from exc
    return {
        "connected": True,
        "database": str(settings.database_path.name),
        "report_count": report_count,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_animal_health(
    image: UploadFile = File(...),
    user_id: Optional[int] = Form(default=None),
    animal_name: Optional[str] = Form(default=None),
    location_lat: Optional[float] = Form(default=None),
    location_long: Optional[float] = Form(default=None),
    area: Optional[str] = Form(default=None),
    contact_rescue: bool = Form(default=False),
    prefer_current_location: bool = Form(default=False),
    db = Depends(get_db),
):
    saved_file = save_upload_file(image)
    exif_lat, exif_long = extract_gps_from_image(saved_file)
    if prefer_current_location:
        effective_lat = location_lat
        effective_long = location_long
    else:
        effective_lat = location_lat if location_lat is not None else exif_lat
        effective_long = location_long if location_long is not None else exif_long
    prediction = pipeline.predict(saved_file)
    location_info = reverse_geocode(effective_lat, effective_long)
    if effective_lat is None or effective_long is None:
        effective_lat = location_info.get("location_lat", effective_lat)
        effective_long = location_info.get("location_long", effective_long)
    inferred_area = area or location_info["location_name"]
    rescue_contacts, vet_contacts = get_contacts_for_area(db, inferred_area, effective_lat, effective_long)

    base_case_report = build_case_report(
        animal_type=prediction["animal_type"],
        health_status=prediction["health_status"],
        detected_conditions=prediction["detected_conditions"],
        location_address=location_info["location_address"],
        confidence=prediction["health_confidence"],
    )
    vision_case_report = None
    if prediction["analysis_status"] == "animal_detected" and prediction["animal_type"]:
        vision_case_report = generate_vision_case_report(
            saved_file,
            base_case_report,
            image_hint=prediction["animal_type"],
        )
    case_report = {**base_case_report, **vision_case_report} if vision_case_report else base_case_report
    if vision_case_report:
        vision_animal = vision_case_report.get("animal_detected")
        if vision_animal in SUPPORTED_DOMESTIC_ANIMALS:
            prediction["animal_type"] = vision_animal
        if vision_case_report.get("health_status"):
            prediction["health_status"] = vision_case_report["health_status"]
        if vision_case_report.get("detection_confidence") is not None:
            prediction["detection_confidence"] = float(vision_case_report["detection_confidence"])
        if vision_case_report.get("health_score") is not None:
            prediction["health_confidence"] = max(prediction["health_confidence"], min(0.99, float(vision_case_report["health_score"]) / 100.0))
        if vision_case_report.get("detected_conditions"):
            prediction["detected_conditions"] = vision_case_report["detected_conditions"]
        if vision_case_report.get("medical_alert"):
            prediction["medical_alert"] = vision_case_report["medical_alert"]
    primary_report = {
        "animal_type": prediction["animal_type"],
        "animal_name": animal_name,
        "detection_confidence": prediction["detection_confidence"],
        "health_status": prediction["health_status"],
        "health_confidence": prediction["health_confidence"],
        "health_score": case_report["health_score"],
        "urgency_level": case_report["urgency_level"],
        "urgency_label": case_report["urgency_label"],
        "bounding_box": prediction["bbox"],
        "detected_conditions": case_report["structured_findings"],
        "medical_alert": prediction["medical_alert"],
        "guidance": case_report["guidance"],
        "condition_summary": case_report["condition_summary"],
        "recommended_actions": case_report["recommended_actions"],
        "needs_rescue": case_report["needs_rescue"],
        "health_summary": case_report["health_summary"],
        "likely_issue": case_report["likely_issue"],
        "urgency": case_report["urgency"],
        "primary_issues": case_report["primary_issues"],
        "visible_symptoms": case_report["visible_symptoms"],
        "body_condition": case_report["body_condition"],
        "animal_description": case_report.get("animal_description", ""),
        "injury_description": case_report.get("injury_description", ""),
        "breed_guess": case_report.get("breed_guess"),
        "what_is_wrong": case_report["what_is_wrong"],
        "help_type": case_report["help_type"],
        "triage_reasoning": case_report["triage_reasoning"],
        "emergency_plan": case_report["emergency_plan"],
        "avoid_steps": case_report["avoid_steps"],
        "contact_priority": case_report["contact_priority"],
    }
    guidance = case_report["guidance"] or guidance_for_status(prediction["health_status"])
    health_summary = case_report["health_summary"] or health_summary_for_status(prediction["health_status"], prediction["health_confidence"])
    should_offer_contacts = prediction["analysis_status"] == "animal_detected" and prediction["health_status"] != "Healthy"
    rescue_prompt = (
        "Do you want to contact a rescue team?"
        if prediction["analysis_status"] == "animal_detected"
        else prediction["medical_alert"]
    )

    report = create_report(
        db,
        ReportCreate(
            user_id=user_id,
            image_path=saved_file.relative_to(settings.project_root).as_posix(),
            analysis_status=prediction["analysis_status"],
            animal_type=prediction["animal_type"],
            animal_name=animal_name,
            health_status=prediction["health_status"],
            confidence_score=prediction["health_confidence"],
            detection_confidence=prediction["detection_confidence"],
            guidance=guidance,
            detected_conditions=case_report["structured_findings"],
            location_name=inferred_area,
            location_address=location_info["location_address"],
            rescue_requested=contact_rescue if prediction["analysis_status"] == "animal_detected" else False,
            rescue_status="pending" if contact_rescue and prediction["analysis_status"] == "animal_detected" else "not_requested",
            location_lat=effective_lat,
            location_long=effective_long,
            bbox=DetectionBox(**prediction["bbox"]),
            animal_reports=[primary_report],
        ),
    )

    return PredictionResponse(
        report_id=report.report_id,
        image_path=report.image_path,
        image_url=f"/{to_url_path(report.image_path)}",
        analysis_status=prediction["analysis_status"],
        is_animal=prediction["is_animal"],
        animal_type=prediction["animal_type"],
        animal_detected=prediction["animal_type"],
        animal_name=report.animal_name,
        location_name=inferred_area,
        location_address=location_info["location_address"],
        location_lat=effective_lat,
        location_long=effective_long,
        detection_confidence=prediction["detection_confidence"],
        health_status=prediction["health_status"],
        health_status_code=health_status_code_from_score(case_report["health_score"]),
        health_confidence=prediction["health_confidence"],
        health_score=case_report["health_score"],
        urgency_level=case_report["urgency_level"],
        urgency_label=case_report["urgency_label"],
        bounding_box=DetectionBox(**prediction["bbox"]),
        guidance=guidance,
        health_summary=health_summary,
        condition_summary=case_report["condition_summary"],
        animal_description=case_report.get("animal_description", ""),
        injury_description=case_report.get("injury_description", ""),
        breed_guess=case_report.get("breed_guess"),
        primary_issues=case_report["primary_issues"],
        visible_symptoms=case_report["visible_symptoms"],
        body_condition=case_report["body_condition"],
        what_is_wrong=case_report["what_is_wrong"],
        recommended_actions=case_report["recommended_actions"],
        needs_rescue=case_report["needs_rescue"],
        help_type=case_report["help_type"],
        triage_reasoning=case_report["triage_reasoning"],
        emergency_plan=case_report["emergency_plan"],
        avoid_steps=case_report["avoid_steps"],
        contact_priority=case_report["contact_priority"],
        needs_help=case_report["needs_rescue"],
        detected_conditions=case_report["structured_findings"],
        rescue_prompt=rescue_prompt,
        rescue_contacts=rescue_contacts if contact_rescue or should_offer_contacts else [],
        vet_contacts=vet_contacts if should_offer_contacts else [],
        animal_reports=[primary_report],
        other_detections=[],
    )


@router.post("/medical-chat", response_model=MedicalChatResponse)
def medical_chat(payload: MedicalChatRequest):
    messages = [{"role": message.role, "content": message.content} for message in payload.messages]
    reply, fallback_used, model_used = medical_chat_reply(
        messages=messages,
        animal_type=payload.animal_type,
        health_status=payload.health_status,
        detected_conditions=payload.detected_conditions,
        location_name=payload.location_name,
    )
    return MedicalChatResponse(
        reply=reply,
        model=model_used,
        fallback_used=fallback_used,
    )


@router.get("/reports", response_model=list[ReportOut])
def list_reports(
    animal_type: Optional[str] = None,
    health_status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db = Depends(get_db),
):
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    return [report_to_schema(report) for report in get_reports(db, animal_type=animal_type, health_status=health_status, start_date=start, end_date=end)]


@router.get("/reports/{report_id}", response_model=ReportOut)
def read_report(report_id: int, db = Depends(get_db)):
    report = get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report_to_schema(report)


@router.put("/reports/{report_id}", response_model=ReportOut)
def update_report_status(report_id: int, payload: ReportUpdate, db = Depends(get_db)):
    report = update_report(db, report_id, payload)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report_to_schema(report)


@router.delete("/reports/{report_id}")
def remove_report(report_id: int, db = Depends(get_db)):
    deleted = delete_report(db, report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"deleted": True}


@router.delete("/reports")
def remove_all_reports(db = Depends(get_db)):
    deleted_count = delete_all_reports(db)
    return {"deleted": True, "deleted_count": deleted_count}
