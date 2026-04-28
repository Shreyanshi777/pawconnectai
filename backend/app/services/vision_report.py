from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from backend.app.core.config import settings

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SUPPORTED_ANIMALS = ("dog", "cat", "rabbit", "bird", "cow")
SUPPORTED_SEVERITIES = ("Healthy", "Mild", "Serious")
SUPPORTED_URGENCY_LEVELS = ("none", "monitor", "vet_soon", "urgent", "emergency")


VISION_SYSTEM_PROMPT = """
You are an animal triage vision engine for a rescue web app.
Analyze exactly one domestic animal in the image and return ONLY valid JSON.
Focus on posture, eyes, face, body condition, mobility, weakness, pale appearance, wounds, bleeding, swelling, bruising, broken limb, or distress.

Supported animals:
- dog
- cat
- rabbit
- bird
- cow

Return this JSON shape:
{
  "animal_detected": "dog|cat|rabbit|bird|cow|unknown",
  "detection_confidence": 0.0,
  "breed_guess": "string|unknown",
  "animal_description": "string",
  "injury_description": "string",
  "health_status": "Healthy|Mild|Serious",
  "health_score": 0,
  "urgency_level": "none|monitor|vet_soon|urgent|emergency",
  "urgency_label": "string",
  "primary_issues": ["string"],
  "visible_symptoms": ["string"],
  "body_condition": "string",
  "what_is_wrong": "string",
  "recommended_actions": ["string"],
  "needs_rescue": true,
  "help_type": "none|home_care|vet_checkup|urgent_vet|emergency_vet",
  "triage_reasoning": "string",
  "condition_summary": "string",
  "health_summary": "string",
  "medical_alert": "string",
  "detected_conditions": ["string"],
  "emergency_plan": {
    "level": "none|monitor|vet_soon|urgent|emergency",
    "label": "string",
    "summary": "string",
    "immediate_steps": ["string"],
    "avoid_steps": ["string"],
    "contact_priority": "string",
    "sos_message": "string"
  },
  "avoid_steps": ["string"],
  "contact_priority": "string"
}

Be conservative:
- If the animal looks weak, collapsed, pale, limping, unable to stand, or in severe distress, do NOT call it healthy.
- If unsure, choose Mild or Serious rather than Healthy.
- Do not mention multiple animals.
- If you can infer a likely breed or breed mix from the image, include it in breed_guess; otherwise use "unknown".
- Describe the visible body shape, coat, posture, color, and injury in plain language.
- Keep the wording practical and human-friendly.
""".strip()


def _image_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _load_json_payload(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"\s*```$", "", content).strip()
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                return None
    return None


def _normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
      return []
    cleaned: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _normalize_string(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _normalize_health_status(value: Any) -> str:
    text = _normalize_string(value, "Healthy").title()
    if text not in SUPPORTED_SEVERITIES:
        return "Healthy"
    return text


def _normalize_urgency_level(value: Any) -> str:
    text = _normalize_string(value, "none").lower()
    return text if text in SUPPORTED_URGENCY_LEVELS else "none"


def _normalize_animal(value: Any) -> str:
    text = _normalize_string(value, "unknown").lower()
    return text if text in SUPPORTED_ANIMALS else "unknown"


def _default_emergency_plan(animal_type: str, health_status: str, urgency_level: str, summary: str) -> dict[str, Any]:
    if urgency_level in {"urgent", "emergency"} or health_status == "Serious":
        steps = [
            "Keep the animal as calm and still as possible.",
            "Do not force it to walk or stand if it seems weak or hurt.",
            "Contact a rescue team or veterinarian immediately.",
        ]
        avoid = [
            "Do not give human medicine.",
            "Do not press deep wounds aggressively.",
            "Do not crowd or scare the animal.",
        ]
        label = "Emergency - act now"
        priority = "Contact rescue or a vet immediately"
    elif health_status == "Mild":
        steps = [
            "Keep the area quiet and safe.",
            "Avoid stressing or handling the injured area.",
            "Arrange a veterinary check soon.",
        ]
        avoid = [
            "Do not force movement.",
            "Do not use human medicine.",
            "Do not ignore worsening symptoms.",
        ]
        label = "See vet within 48h"
        priority = "Book a vet check soon"
    else:
        steps = [
            "Observe from a safe distance.",
            "Re-scan if the condition changes.",
            "Keep the animal away from danger.",
        ]
        avoid = [
            "Do not panic or crowd the animal.",
            "Do not handle unless needed for safety.",
        ]
        label = "No action needed"
        priority = "Monitor safely"
    return {
        "level": urgency_level,
        "label": label,
        "summary": summary,
        "immediate_steps": steps,
        "avoid_steps": avoid,
        "contact_priority": priority,
        "sos_message": f"{animal_type.title()} triage: {label}. {summary}".strip(),
    }


def _normalize_payload(payload: dict[str, Any], base_case_report: dict[str, Any]) -> dict[str, Any]:
    animal_detected = _normalize_animal(payload.get("animal_detected") or base_case_report.get("animal_detected") or base_case_report.get("animal_type"))
    breed_guess = _normalize_string(payload.get("breed_guess"), base_case_report.get("breed_guess", "unknown")).strip()
    health_status = _normalize_health_status(payload.get("health_status") or base_case_report.get("health_status"))
    urgency_level = _normalize_urgency_level(payload.get("urgency_level") or base_case_report.get("urgency_level"))
    detection_confidence = float(payload.get("detection_confidence") or 0.0)
    health_score = int(round(payload.get("health_score") or base_case_report.get("health_score") or 0))

    primary_issues = _normalize_list(payload.get("primary_issues")) or _normalize_list(base_case_report.get("primary_issues"))
    visible_symptoms = _normalize_list(payload.get("visible_symptoms")) or _normalize_list(base_case_report.get("visible_symptoms"))
    detected_conditions = _normalize_list(payload.get("detected_conditions")) or _normalize_list(base_case_report.get("structured_findings"))
    recommended_actions = _normalize_list(payload.get("recommended_actions")) or _normalize_list(base_case_report.get("recommended_actions"))
    avoid_steps = _normalize_list(payload.get("avoid_steps")) or _normalize_list(base_case_report.get("avoid_steps"))

    body_condition = _normalize_string(payload.get("body_condition"), base_case_report.get("body_condition", ""))
    animal_description = _normalize_string(payload.get("animal_description"), base_case_report.get("animal_description", ""))
    injury_description = _normalize_string(payload.get("injury_description"), base_case_report.get("injury_description", ""))
    what_is_wrong = _normalize_string(payload.get("what_is_wrong"), base_case_report.get("what_is_wrong", ""))
    help_type = _normalize_string(payload.get("help_type"), base_case_report.get("help_type", "none"))
    triage_reasoning = _normalize_string(payload.get("triage_reasoning"), base_case_report.get("triage_reasoning", ""))
    condition_summary = _normalize_string(payload.get("condition_summary"), base_case_report.get("condition_summary", ""))
    health_summary = _normalize_string(payload.get("health_summary"), base_case_report.get("health_summary", ""))
    medical_alert = _normalize_string(payload.get("medical_alert"), base_case_report.get("guidance", ""))
    contact_priority = _normalize_string(payload.get("contact_priority"), base_case_report.get("contact_priority", ""))

    emergency_plan = payload.get("emergency_plan") if isinstance(payload.get("emergency_plan"), dict) else {}
    if not emergency_plan:
        emergency_plan = _default_emergency_plan(animal_detected, health_status, urgency_level, medical_alert or condition_summary)
    else:
        emergency_plan = {
            "level": _normalize_urgency_level(emergency_plan.get("level") or urgency_level),
            "label": _normalize_string(emergency_plan.get("label"), base_case_report.get("urgency_label", "No action needed")),
            "summary": _normalize_string(emergency_plan.get("summary"), medical_alert or condition_summary),
            "immediate_steps": _normalize_list(emergency_plan.get("immediate_steps")) or recommended_actions,
            "avoid_steps": _normalize_list(emergency_plan.get("avoid_steps")) or avoid_steps,
            "contact_priority": _normalize_string(emergency_plan.get("contact_priority"), contact_priority or "Monitor safely"),
            "sos_message": _normalize_string(emergency_plan.get("sos_message"), medical_alert or condition_summary),
        }

    return {
        "animal_detected": animal_detected,
        "detection_confidence": round(detection_confidence, 4),
        "health_status": health_status,
        "health_score": max(0, min(100, health_score)),
        "urgency_level": urgency_level,
        "urgency_label": _normalize_string(payload.get("urgency_label"), base_case_report.get("urgency_label", "No action needed")),
        "breed_guess": breed_guess or "unknown",
        "primary_issues": primary_issues,
        "visible_symptoms": visible_symptoms,
        "body_condition": body_condition,
        "animal_description": animal_description,
        "injury_description": injury_description,
        "what_is_wrong": what_is_wrong,
        "recommended_actions": recommended_actions,
        "needs_rescue": bool(payload.get("needs_rescue", base_case_report.get("needs_rescue", False))),
        "help_type": help_type or base_case_report.get("help_type", "none"),
        "triage_reasoning": triage_reasoning,
        "condition_summary": condition_summary,
        "health_summary": health_summary,
        "medical_alert": medical_alert,
        "detected_conditions": detected_conditions,
        "emergency_plan": emergency_plan,
        "avoid_steps": avoid_steps,
        "contact_priority": contact_priority or emergency_plan["contact_priority"],
    }


def generate_vision_case_report(
    image_path: Path,
    base_case_report: dict[str, Any],
    image_hint: str | None = None,
) -> dict[str, Any] | None:
    if OpenAI is None:
        return None

    client = None
    model = None
    if settings.groq_api_key:
        client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
        model = settings.groq_vision_model
    elif settings.openai_api_key:
        client = OpenAI(api_key=settings.openai_api_key)
        model = settings.openai_vision_model
    else:
        return None

    try:
        image_url = _image_to_data_url(image_path)
        prompt = (
            f"Animal hint from local model: {image_hint or base_case_report.get('animal_detected') or base_case_report.get('animal_type') or 'unknown'}.\n"
            f"Local findings: {', '.join(base_case_report.get('structured_findings', []) or base_case_report.get('detected_conditions', []) or []) or 'none'}.\n"
            f"Local health score: {base_case_report.get('health_score', 'unknown')}.\n"
            "Write the report for one animal only. Keep it simple and specific.\n"
            "Try to guess the breed or breed mix if visible clues support it. If not, use unknown.\n"
            "Describe the animal's body shape, coat/color, posture, and the visible injury in plain English."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    ],
                },
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else ""
        payload = _load_json_payload(content or "")
        if not isinstance(payload, dict):
            return None
        return _normalize_payload(payload, base_case_report)
    except Exception:
        return None
