from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmergencyPlan:
    level: str
    label: str
    summary: str
    immediate_steps: list[str]
    avoid_steps: list[str]
    contact_priority: str
    sos_message: str


def _base_species_steps(animal_type: str) -> tuple[list[str], list[str]]:
    animal = (animal_type or "animal").lower()

    if animal == "dog":
        return (
            [
                "Keep the dog calm and away from traffic or other animals.",
                "Speak softly and move slowly so the dog does not panic.",
                "If the dog is cold or shaking, wrap it gently in a towel or blanket.",
            ],
            [
                "Do not corner or chase the dog.",
            ],
        )
    if animal == "cat":
        return (
            [
                "Approach quietly and keep the cat in a small, safe space if possible.",
                "Let the cat move at its own pace and avoid sudden touching.",
                "If the cat is cold, wrap it gently in a towel or blanket to keep it warm.",
            ],
            [
                "Do not grab the cat suddenly or force handling if it is fearful.",
            ],
        )
    if animal == "rabbit":
        return (
            [
                "Keep the rabbit still, quiet, and protected from stress.",
                "Handle only if absolutely needed and support the body fully.",
                "If the rabbit is cold or weak, wrap it lightly in a towel and keep it warm.",
            ],
            [
                "Do not lift the rabbit by the ears or let it struggle freely.",
            ],
        )
    if animal == "bird":
        return (
            [
                "Move the bird to a quiet ventilated box or carrier if it can be done safely.",
                "Reduce noise, light, and movement around the bird.",
                "If the bird looks cold, place a soft cloth or towel around it without covering the face.",
            ],
            [
                "Do not squeeze the bird or keep it in an open, stressful area.",
            ],
        )
    if animal == "cow":
        return (
            [
                "Keep distance and stay to the side so the cow is not crowded.",
                "Use a calm, wide approach and avoid surrounding the animal.",
                "If the cow is weak or lying down, keep it calm and call local livestock help quickly.",
            ],
            [
                "Do not stand directly behind the cow or block its movement.",
            ],
        )
    return ([], [])


def _condition_specific_steps(findings: list[str], level: str) -> tuple[list[str], list[str]]:
    joined = " ".join(findings).lower()
    do_now: list[str] = []
    avoid: list[str] = []

    if any(term in joined for term in ("bleeding", "blood", "open wound", "wound", "laceration")):
        do_now.extend([
            "If there is visible bleeding, place a clean cloth or gauze over the area and apply gentle pressure.",
            "Keep the cloth in place while you arrange help, and replace it only if it becomes soaked through.",
        ])
        avoid.append("Do not press deeply into the wound or remove large clots.")

    if any(term in joined for term in ("fracture", "broken", "limp", "dragging limb", "unable to stand")):
        do_now.extend([
            "Keep the animal as still as possible and avoid making it walk if a fracture or limb injury is possible.",
            "If transport is needed, support the body evenly and avoid bending the injured limb.",
        ])
        avoid.append("Do not try to straighten a limb or force the animal to stand.")

    if any(term in joined for term in ("weakness", "collapse", "pale", "low-energy", "unable to stand")):
        do_now.extend([
            "Keep the animal warm, quiet, and shaded from heat or cold.",
            "Watch breathing and responsiveness closely while help is on the way.",
        ])
        avoid.append("Do not give food if the animal is weak, fainting, or struggling to breathe.")

    if any(term in joined for term in ("eye", "face", "facial")):
        do_now.extend([
            "Keep the animal from rubbing the eye or face against the ground.",
            "Avoid touching the eye itself and keep bright light or dust away if possible.",
        ])
        avoid.append("Do not put drops, ointment, or pressure directly into the eye unless a vet told you to.")

    if any(term in joined for term in ("breathing", "choking", "mouth open", "gasping")):
        do_now.extend([
            "Keep the neck and head in a neutral position so breathing stays as open as possible.",
            "Move the animal only if there is immediate danger, and keep the path calm and clear.",
        ])
        avoid.append("Do not tilt the head back or force water or food into the mouth.")

    if level in {"urgent", "emergency"} and not do_now:
        do_now.extend([
            "Keep the animal in a quiet place away from people, traffic, and other animals.",
            "Stay with the animal until a rescue or vet takes over.",
        ])

    return do_now, avoid


def _level_from_score(score: int | None, findings: list[str]) -> str:
    joined = " ".join(findings).lower()
    if score is not None and score < 30:
        return "emergency"
    if any(term in joined for term in ("bleeding", "fracture", "exposed tissue", "collapse", "pale", "broken limb", "broken bone", "weakness", "unable to stand")):
        return "emergency"
    if score is not None and score < 50:
        return "urgent"
    if any(term in joined for term in ("limping", "swelling", "bruise", "weakness", "dragging limb", "eye irritation")):
        return "vet_soon"
    return "monitor"


def build_emergency_plan(
    animal_type: str | None,
    health_score: int | None,
    structured_findings: list[str],
    health_status: str | None = None,
) -> dict[str, object]:
    animal = (animal_type or "animal").lower()
    level = _level_from_score(health_score, structured_findings)
    species_steps, species_avoid = _base_species_steps(animal)

    do_now: list[str]
    avoid: list[str] = list(species_avoid)
    contact_priority = "Monitor from a safe distance"
    condition_do_now, condition_avoid = _condition_specific_steps(structured_findings, level)

    if level == "emergency":
        do_now = [
            "Call a rescue team or vet immediately.",
            "Keep the animal warm, quiet, and away from further danger.",
            "If the animal is awake and able to swallow, offer a small amount of water without forcing it.",
            "If it is safe, gently wrap the animal in a towel or blanket before transport.",
            "If the area is safe, note the location and visible injury before moving the animal.",
        ]
        avoid.extend(
            [
                "Do not give human medicine or force the animal to walk.",
                "Do not force water into the mouth or feed the animal if it is weak, vomiting, or struggling to breathe.",
            ]
        )
        contact_priority = "Emergency rescue or vet needed now"
    elif level == "urgent":
        do_now = [
            "Arrange a vet or rescue check as soon as possible.",
            "Keep the animal away from traffic, children, and other animals.",
            "If the animal is awake and calm, you may offer a small amount of water but do not force it.",
            "If the animal looks cold or shaky, wrap it gently in a towel or blanket.",
            "Watch for worsening bleeding, swelling, or limping.",
        ]
        avoid.extend(
            [
                "Do not delay if the animal gets weaker or starts bleeding.",
                "Do not feed medication without professional advice.",
                "Do not force food or water into the animal.",
            ]
        )
        contact_priority = "Urgent vet review today"
    elif level == "vet_soon":
        do_now = [
            "Contact a nearby vet or rescue for guidance today.",
            "Keep the animal in a safe, low-stress area.",
            "Offer clean water if the animal is awake and swallowing normally.",
            "Recheck the animal soon for any change in movement or breathing.",
        ]
        avoid.extend(
            [
                "Do not ignore swelling, limping, or repeated distress signals.",
            ]
        )
        contact_priority = "Vet check recommended soon"
    else:
        do_now = [
            "Observe the animal from a safe distance.",
            "Keep the area calm and safe from traffic or hazards.",
            "Re-scan if behavior or appearance changes.",
        ]
        contact_priority = "Safe observation is enough for now"

    if condition_do_now:
        do_now = condition_do_now + do_now
    if condition_avoid:
        avoid.extend(condition_avoid)

    summary_map = {
        "emergency": "Emergency SOS: act now and contact rescue or a vet immediately. Weakness, pale appearance, collapse, or broken-limb signs are treated as critical.",
        "urgent": "Urgent help is recommended and the animal should be checked today. Weakness, limping, or pale appearance can still be serious.",
        "vet_soon": "The animal should be checked soon to avoid the problem getting worse.",
        "monitor": "The animal can be watched safely for now, but keep an eye on changes.",
    }

    label_map = {
        "emergency": "Emergency",
        "urgent": "Urgent",
        "vet_soon": "Vet Soon",
        "monitor": "Monitor",
    }

    return {
        "level": level,
        "label": label_map[level],
        "summary": summary_map[level],
        "immediate_steps": species_steps + do_now,
        "avoid_steps": avoid[:5],
        "contact_priority": contact_priority,
        "sos_message": summary_map[level],
        "health_status": health_status or "",
    }
