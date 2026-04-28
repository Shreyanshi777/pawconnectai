from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from backend.app.services.emergency_engine import build_emergency_plan


SPECIES_HINTS = {
    "dog": "Keep a calm distance, avoid cornering the dog, and offer help only if safe.",
    "cat": "Approach slowly and quietly, and avoid sudden grabbing because scared cats may scratch or hide.",
    "rabbit": "Handle gently and avoid lifting unless necessary because rabbits can injure themselves while struggling.",
    "bird": "Minimize noise and movement, and place the bird in a ventilated box if rescue guidance advises transport.",
    "cow": "Stay to the side, avoid crowding, and get local animal assistance if the cow cannot stand or is bleeding.",
}

REFERENCE_DATASET_PATH = Path(__file__).resolve().parents[3] / "tmp_dataset_inspect" / "animal_injury_dataset_v2.csv"
URGENCY_LABELS = {
    "none": "No action needed",
    "monitor": "Monitor at home",
    "vet_soon": "See vet within 48h",
    "urgent": "Urgent vet visit today",
    "emergency": "Emergency - act now",
}

KEYWORD_FINDINGS = {
    "bleeding": "Visible bleeding or a blood-like region may be present",
    "wound": "A visible wound-like area may be present",
    "abrasion": "A surface abrasion or scrape may be present",
    "cut": "A cut or laceration-like area may be present",
    "skin": "Skin or fur damage may be present",
    "swelling": "Swelling or tissue inflammation may be present",
    "bruise": "Bruising or deeper tissue discoloration may be present",
    "pale": "Pale eyes, gums, or weak circulation may be present",
    "weak": "Weakness, exhaustion, or low-energy posture may be present",
    "collapse": "The animal may be collapsing or unable to stay upright",
    "fracture": "The posture may suggest pain or a possible fracture",
    "broken": "A broken limb or severe bone injury may be present",
    "limp": "The animal may be limping or avoiding normal weight bearing",
    "leg": "A limb or leg problem may be present",
    "posture": "Abnormal posture may indicate pain, weakness, or distress",
    "eye": "Eye irritation, trauma, or partial closure may be present",
    "tissue": "Exposed tissue or a severe lesion pattern may be present",
    "distress": "The animal may be in visible distress",
    "mobility": "Reduced mobility or reluctance to move may be present",
    "drag": "The animal may be dragging a limb or unable to move normally",
}


def normalize_conditions(detected_conditions: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in detected_conditions:
        text = str(item or "").strip()
        if not text:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def derive_structured_findings(health_status: str, detected_conditions: list[str]) -> list[str]:
    findings: list[str] = []
    normalized = [item.lower() for item in normalize_conditions(detected_conditions)]

    for condition in normalized:
        for keyword, message in KEYWORD_FINDINGS.items():
            if keyword in condition and message not in findings:
                findings.append(message)

    if not findings:
        if health_status == "Healthy":
            findings.append("No strong visible injury markers were detected")
        elif health_status == "Mild":
            findings.append("A limited visible issue may be present but does not appear critical")
        elif health_status == "Serious":
            findings.append("Strong visual warning signs suggest a serious injury or severe discomfort")

    if any(term in " ".join(normalized) for term in ("pale", "weak", "collapse", "broken", "drag")):
        if "Weakness, exhaustion, or low-energy posture may be present" not in findings:
            findings.append("Weakness, exhaustion, or low-energy posture may be present")
        if "The animal may be collapsing or unable to stay upright" not in findings and any(term in " ".join(normalized) for term in ("collapse", "fallen", "unable to stand")):
            findings.append("The animal may be collapsing or unable to stay upright")
        if "A broken limb or severe bone injury may be present" not in findings and any(term in " ".join(normalized) for term in ("broken", "fracture", "limb", "leg")):
            findings.append("A broken limb or severe bone injury may be present")

    return findings[:5]


def likely_issue_text(health_status: str, structured_findings: list[str]) -> str:
    joined = " ".join(structured_findings).lower()
    if "bleeding" in joined or "exposed tissue" in joined:
        return "Likely open wound or active tissue injury"
    if "fracture" in joined or "weight bearing" in joined or "limping" in joined:
        return "Likely limb injury, pain, or mobility problem"
    if "eye" in joined:
        return "Likely eye or facial injury"
    if "weakness" in joined or "collapse" in joined or "pale" in joined:
        return "Likely weakness, circulation issue, or severe distress"
    if "swelling" in joined or "bruise" in joined:
        return "Likely swelling, bruising, or soft tissue trauma"
    if health_status == "Healthy":
        return "No major visible injury pattern detected"
    if health_status == "Mild":
        return "Visible mild injury or discomfort pattern detected"
    if health_status == "Serious":
        return "Visible serious injury or emergency pattern detected"
    return "Manual review is recommended"


def urgency_text(health_status: str, structured_findings: list[str]) -> str:
    joined = " ".join(structured_findings).lower()
    if health_status == "Serious" or "bleeding" in joined or "fracture" in joined:
        return "Urgent"
    if health_status == "Mild":
        return "Monitor closely"
    if health_status == "Healthy":
        return "Stable"
    return "Needs review"


def urgency_level(health_status: str, structured_findings: list[str]) -> str:
    joined = " ".join(structured_findings).lower()
    if health_status == "Healthy":
        return "none"
    if health_status == "Mild":
        return "monitor"
    if "bleeding" in joined or "fracture" in joined or "exposed tissue" in joined or "collapse" in joined or "pale" in joined:
        return "emergency" if health_status == "Serious" else "urgent"
    if health_status == "Serious":
        return "urgent"
    return "vet_soon"


def help_type_for_level(level: str) -> str:
    mapping = {
        "none": "none",
        "monitor": "home_care",
        "vet_soon": "vet_checkup",
        "urgent": "urgent_vet",
        "emergency": "emergency_vet",
    }
    return mapping.get(level, "vet_checkup")


def health_score_for_status(health_status: str, confidence: float | None, structured_findings: list[str]) -> int:
    if health_status == "NotApplicable":
        return 0
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.65
    joined = " ".join(structured_findings).lower()
    if health_status == "Healthy":
        if any(term in joined for term in ("weakness", "collapse", "pale", "fracture", "dragging")):
            return max(22, min(45, int(round(42 - (confidence_value * 8)))))
        return max(85, min(100, int(round(84 + (confidence_value * 16)))))
    if health_status == "Mild":
        if any(term in joined for term in ("limping", "weight bearing", "broken", "weakness", "collapse", "pale")):
            return max(55, min(74, int(round(70 - (confidence_value * 10)))))
        return max(65, min(84, int(round(76 - (confidence_value * 10)))))
    if health_status == "Serious":
        if any(term in joined for term in ("bleeding", "fracture", "exposed tissue", "collapse", "pale", "broken", "drag")):
            return max(5, min(24, int(round(22 - (confidence_value * 8)))))
        return max(20, min(39, int(round(36 - (confidence_value * 10)))))
    return 50


def primary_issues_from_findings(structured_findings: list[str]) -> list[str]:
    issues = []
    for finding in structured_findings:
        lowered = finding.lower()
        if "bleeding" in lowered:
            issues.append("Possible bleeding")
        elif "wound" in lowered or "laceration" in lowered or "abrasion" in lowered:
            issues.append("Visible wound or skin damage")
        elif "limping" in lowered or "weight bearing" in lowered or "mobility" in lowered:
            issues.append("Mobility problem or limb pain")
        elif "weakness" in lowered or "collapse" in lowered or "pale" in lowered:
            issues.append("Weakness, poor circulation, or collapse risk")
        elif "fracture" in lowered:
            issues.append("Possible fracture or severe limb injury")
        elif "eye" in lowered:
            issues.append("Eye or facial injury")
        elif "swelling" in lowered or "bruise" in lowered:
            issues.append("Swelling or bruising")
        elif "posture" in lowered or "distress" in lowered:
            issues.append("Pain or distress posture")
    deduped: list[str] = []
    for issue in issues:
        if issue not in deduped:
            deduped.append(issue)
    return deduped[:4]


def body_condition_text(animal_type: str, health_status: str, structured_findings: list[str]) -> str:
    if health_status == "Healthy":
        return f"The overall body condition of the {animal_type} appears stable and visually normal in this image."
    if any("limping" in item.lower() or "weight bearing" in item.lower() for item in structured_findings):
        return f"The {animal_type} appears uncomfortable and may be protecting one limb or avoiding normal movement."
    if any("bleeding" in item.lower() or "wound" in item.lower() for item in structured_findings):
        return f"The {animal_type} shows visible surface trauma and may be in pain or at risk of worsening injury."
    return f"The overall body condition of the {animal_type} appears abnormal enough to justify closer monitoring or veterinary review."


def animal_description_text(animal_type: str, breed_guess: str | None, health_status: str, structured_findings: list[str]) -> str:
    breed = (breed_guess or "").strip().lower()
    breed_phrase = ""
    if breed and breed not in {"unknown", "not sure", animal_type.lower()}:
        breed_phrase = f"likely a {breed} {animal_type}"
    else:
        breed_phrase = f"a {animal_type}"

    finding_hint = ""
    if structured_findings:
        finding_hint = f" Visible cues include {'; '.join(structured_findings[:2]).lower()}."

    condition_hint = {
        "Healthy": "The animal looks alert and visually stable overall.",
        "Mild": "The animal looks mostly stable but may be uncomfortable or guarding a limb.",
        "Serious": "The animal looks visibly distressed and may need urgent care.",
    }.get(health_status, "The animal needs a closer review.")

    return f"The image appears to show {breed_phrase}. {condition_hint}{finding_hint}".strip()


def injury_description_text(animal_type: str, health_status: str, likely_issue: str, structured_findings: list[str]) -> str:
    if health_status == "Healthy":
        return f"No clear visible injury stands out on the {animal_type} in this image."
    if structured_findings:
        detail = "; ".join(structured_findings[:2])
        return f"The {animal_type} shows signs that may fit {likely_issue.lower()}. Visible clues: {detail}."
    return f"The {animal_type} may have a visible problem, but the image is not detailed enough to pinpoint the injury."


def what_is_wrong_text(animal_type: str, health_status: str, likely_issue: str, structured_findings: list[str]) -> str:
    if health_status == "Healthy":
        return f"Nothing major is visibly wrong - the {animal_type} appears healthy in this image."
    if structured_findings:
        return f"The {animal_type} may have a problem. Most likely issue: {likely_issue}. Main visible clues: {'; '.join(structured_findings[:2])}."
    return f"The {animal_type} may have a visible health problem, but the image does not provide enough detail for a more specific explanation."


def triage_reasoning_text(animal_type: str, health_status: str, urgency_level_value: str, structured_findings: list[str]) -> str:
    if health_status == "Healthy":
        return f"The {animal_type} does not show strong visible injury markers, so no urgent response is suggested."
    if urgency_level_value in {"urgent", "emergency"}:
        return f"The urgency is high because the visible findings suggest a serious injury pattern that could worsen without prompt veterinary help."
    if urgency_level_value == "vet_soon":
        return f"The findings suggest a moderate visible issue, so a vet check is recommended soon rather than waiting for the condition to worsen."
    return f"The visible findings appear limited, but the {animal_type} should still be monitored closely in case pain or distress increases."


def build_health_summary(animal_type: str, health_status: str, confidence: float | None, structured_findings: list[str]) -> str:
    confidence_text = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "unknown"
    if health_status == "Healthy":
        return (
            f"The {animal_type} appears visually stable in this image. "
            f"No strong injury markers were detected. Confidence: {confidence_text}."
        )
    if health_status == "Mild":
        return (
            f"The {animal_type} may have a mild visible issue. "
            f"Main cues: {'; '.join(structured_findings[:2])}. Confidence: {confidence_text}."
        )
    if health_status == "Serious":
        return (
            f"The {animal_type} may have a serious visible injury. "
            f"Main cues: {'; '.join(structured_findings[:2])}. Confidence: {confidence_text}. Treat as urgent."
        )
    return f"The system could not form a reliable medical summary. Confidence: {confidence_text}."


def normalize_reference_status(health_status: str) -> str:
    if health_status == "Healthy":
        return "healthy"
    if health_status == "Mild":
        return "mild"
    if health_status == "Serious":
        return "serious"
    return "unknown"


@lru_cache(maxsize=1)
def load_reference_profiles() -> dict[tuple[str, str], dict[str, list[str] | int]]:
    profiles: dict[tuple[str, str], dict[str, list[str] | int]] = {}
    if not REFERENCE_DATASET_PATH.exists():
        return profiles

    with REFERENCE_DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            animal = (row.get("animal_type") or "").strip().lower()
            if animal not in SPECIES_HINTS:
                continue

            severity = (row.get("injury_severity") or "").strip().lower()
            if severity in {"none", ""}:
                status = "healthy"
            elif severity == "mild":
                status = "mild"
            else:
                status = "serious"

            key = (animal, status)
            entry = profiles.setdefault(
                key,
                {
                    "count": 0,
                    "symptoms": [],
                    "behaviors": [],
                    "wounds": [],
                    "mobility": [],
                    "notes": [],
                },
            )
            entry["count"] = int(entry["count"]) + 1
            for source_key, bucket_key in (
                ("visible_symptoms", "symptoms"),
                ("behavior_indicators", "behaviors"),
                ("wound_type", "wounds"),
                ("mobility", "mobility"),
                ("notes", "notes"),
            ):
                value = (row.get(source_key) or "").strip()
                if value and value.lower() not in {"none", "normal"}:
                    bucket = entry[bucket_key]
                    if isinstance(bucket, list) and value not in bucket and len(bucket) < 6:
                        bucket.append(value)
    return profiles


def reference_summary(animal_type: str | None, health_status: str) -> str:
    animal = (animal_type or "").lower()
    status = normalize_reference_status(health_status)
    profile = load_reference_profiles().get((animal, status))
    if not profile:
        return ""

    symptoms = ", ".join(str(item).replace("_", " ") for item in list(profile["symptoms"])[:2])
    behavior = ", ".join(str(item).replace("_", " ") for item in list(profile["behaviors"])[:1])
    wound = ", ".join(str(item).replace("_", " ") for item in list(profile["wounds"])[:1])
    parts = [part for part in [symptoms, behavior, wound] if part]
    if not parts:
        return ""
    return f"Reference cases for similar {animal} situations often mention: {'; '.join(parts)}."


def build_case_report(
    animal_type: str | None,
    health_status: str,
    detected_conditions: list[str],
    location_address: str | None,
    confidence: float | None = None,
):
    animal_label = animal_type or "animal"
    normalized_conditions = normalize_conditions(detected_conditions)
    if health_status == "NotApplicable" or animal_type is None or animal_label.lower() == "unknown":
        return {
            "condition_summary": "The image could not be confirmed as a supported animal case.",
            "recommended_actions": [],
            "needs_rescue": False,
            "guidance": "Not recognized as a supported animal. No medical advice is generated.",
            "health_summary": "The submitted image could not be confirmed as a supported domestic animal case for medical analysis.",
            "structured_findings": ["Not recognized as a supported animal"],
            "likely_issue": "Not recognized",
            "urgency": "Not recognized",
            "urgency_level": "none",
            "urgency_label": "Not recognized",
            "health_score": 0,
            "primary_issues": [],
            "visible_symptoms": [],
            "body_condition": "No supported animal was recognized in this image.",
            "what_is_wrong": "The animal could not be recognized from this image.",
            "help_type": "none",
            "triage_reasoning": "The system could not confirm a supported animal, so no medical guidance is generated.",
            "emergency_plan": {
                "level": "none",
                "label": "Not recognized",
                "summary": "The image could not be confirmed as a supported animal case.",
                "immediate_steps": [],
                "avoid_steps": [],
                "contact_priority": "Retake a clearer image of the animal",
                "sos_message": "Not recognized as a supported animal.",
            },
            "avoid_steps": [],
            "contact_priority": "Retake a clearer image of the animal",
        }
    structured_findings = derive_structured_findings(health_status, normalized_conditions)
    condition_summary = summarize_conditions(animal_label, health_status, structured_findings)
    reference_hint = reference_summary(animal_type, health_status)
    if reference_hint:
        condition_summary = f"{condition_summary} {reference_hint}"
    needs_rescue = health_status == "Serious" or any(
        "bleeding" in marker.lower() or "wound" in marker.lower() or "fracture" in marker.lower()
        for marker in structured_findings
    )
    likely_issue = likely_issue_text(health_status, structured_findings)
    urgency = urgency_text(health_status, structured_findings)
    urgency_level_value = urgency_level(health_status, structured_findings)
    health_score = health_score_for_status(health_status, confidence, structured_findings)
    primary_issues = primary_issues_from_findings(structured_findings)
    body_condition = body_condition_text(animal_label, health_status, structured_findings)
    animal_description = animal_description_text(animal_label, None, health_status, structured_findings)
    injury_description = injury_description_text(animal_label, health_status, likely_issue, structured_findings)
    what_is_wrong = what_is_wrong_text(animal_label, health_status, likely_issue, structured_findings)
    triage_reasoning = triage_reasoning_text(animal_label, health_status, urgency_level_value, structured_findings)
    health_summary = build_health_summary(animal_label, health_status, confidence, structured_findings)
    emergency_plan = build_emergency_plan(animal_type, health_score, structured_findings, health_status)
    actions = list(emergency_plan["immediate_steps"])[:5]
    avoid_steps = list(emergency_plan["avoid_steps"])[:5]

    guidance = f"{emergency_plan['sos_message']} {' '.join(actions[:3])}".strip()
    if location_address:
        guidance += f" Reported location: {location_address}."

    return {
        "condition_summary": condition_summary,
        "recommended_actions": actions,
        "needs_rescue": needs_rescue,
        "guidance": guidance,
        "health_summary": health_summary,
        "structured_findings": structured_findings,
        "likely_issue": likely_issue,
        "urgency": urgency,
        "urgency_level": urgency_level_value,
        "urgency_label": URGENCY_LABELS[urgency_level_value],
        "health_score": health_score,
        "primary_issues": primary_issues,
        "visible_symptoms": structured_findings,
        "body_condition": body_condition,
        "animal_description": animal_description,
        "injury_description": injury_description,
        "what_is_wrong": what_is_wrong,
        "help_type": help_type_for_level(urgency_level_value),
        "triage_reasoning": triage_reasoning,
        "emergency_plan": emergency_plan,
        "avoid_steps": avoid_steps,
        "contact_priority": emergency_plan["contact_priority"],
    }


def summarize_conditions(animal_type: str, health_status: str, detected_conditions: list[str]) -> str:
    base = {
        "Healthy": f"The detected {animal_type} does not show strong visible injury markers in the current image.",
        "Mild": f"The detected {animal_type} appears to have a visible but non-critical issue that should still be checked soon.",
        "Serious": f"The detected {animal_type} appears to have visible signs consistent with a serious injury or severe distress.",
        "NotApplicable": "The image could not be confirmed as a supported animal case.",
    }.get(health_status, "The case needs manual review.")

    if detected_conditions:
        detail = "; ".join(detected_conditions[:3])
        return f"{base} Visible signs noted by the system: {detail}."
    return base


def build_actions(animal_type: str, health_status: str, detected_conditions: list[str]) -> list[str]:
    actions = []
    species_hint = SPECIES_HINTS.get(animal_type)
    if species_hint:
        actions.append(species_hint)

    if health_status == "Healthy":
        actions.extend(
            [
                "Do not force contact if the animal looks stable and alert.",
                "Keep the area safe from traffic or crowding if possible.",
                "Monitor from a distance and recheck only if the condition changes.",
            ]
        )
    elif health_status == "Mild":
        actions.extend(
            [
                "Approach carefully only if the animal is calm and the area is safe.",
                "Avoid touching the injured area and avoid feeding medication or human painkillers.",
                "Contact a nearby vet or rescue volunteer for a timely check, especially if swelling or limping increases.",
            ]
        )
    elif health_status == "Serious":
        actions.extend(
            [
                "Treat this as urgent and contact a rescue team or veterinarian immediately.",
                "Do not force the animal to walk if there may be fracture, deep wound, or severe pain.",
                "If bleeding is visible and it is safe to help, use a clean cloth near the wound without pressing aggressively into deep tissue.",
            ]
        )
    else:
        actions.append("Capture another clear photo and seek manual review from a rescue volunteer or vet.")

    if any("bleeding" in item.lower() for item in detected_conditions):
        actions.append("Visible bleeding may need urgent professional care even if the animal is still standing.")
    if any("posture" in item.lower() for item in detected_conditions):
        actions.append("Abnormal posture can indicate pain, fracture, weakness, or neurological distress.")

    reference_hint = reference_summary(animal_type, health_status)
    if reference_hint:
        actions.append(reference_hint)

    return actions[:5]
