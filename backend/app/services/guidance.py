def guidance_for_status(health_status: str) -> str:
    mapping = {
        "Healthy": "Animal seems fine. Leave it safe and monitor from a distance.",
        "Mild": "Consider taking the animal to a vet or contacting a rescue team.",
        "Serious": "URGENT: Contact a rescue team or vet immediately.",
        "NotApplicable": "No medical guidance is available because the image could not be confirmed as a supported animal case.",
    }
    return mapping.get(health_status, "Use caution and contact a professional if unsure.")


def health_summary_for_status(health_status: str, confidence: float) -> str:
    if health_status == "NotApplicable":
        return "The submitted image could not be confirmed as a supported domestic animal case for medical analysis."
    if health_status == "Healthy":
        return f"The animal appears stable with no strong visible signs of injury. AI confidence: {confidence:.0%}."
    if health_status == "Mild":
        return (
            f"The animal may have a minor visible issue such as limping, small wounds, or unusual posture. "
            f"AI confidence: {confidence:.0%}."
        )
    if health_status == "Serious":
        return (
            f"The animal may show severe visible distress, injury, bleeding, or collapse indicators. "
            f"AI confidence: {confidence:.0%}. Treat this as urgent."
        )
    return f"The AI result is uncertain. Confidence: {confidence:.0%}."
