from __future__ import annotations

from backend.app.core.config import settings

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = """
You are Care Bot, an animal medical help assistant for the public.
You only answer about domestic animal first aid, visible symptoms, urgency, safe handling, hydration, transport, wound-care precautions, vet escalation, rescue escalation, and prevention mistakes that first-time pet owners should avoid.
You may answer conversationally and support follow-up questions naturally.
Do not claim to be a veterinarian.
Do not give medication dosages.
If the animal may have serious injury, breathing difficulty, collapse, heavy bleeding, poisoning, fracture, severe burn, seizure, inability to stand, or severe dehydration, say it is urgent and recommend immediate rescue or veterinary care.
Keep the advice practical, calm, and easy for non-experts.
""".strip()


GENERAL_KNOWLEDGE = {
    "bleeding": "If there is visible bleeding, keep the animal as calm as possible, avoid forcing movement, and seek urgent vet or rescue help. If it is safe, place a clean cloth near the wound without pushing into deep tissue.",
    "wound": "For an open wound, do not apply random ointments or human medicine. Keep the area clean, prevent licking if possible, and get veterinary help, especially if the wound is deep, dirty, or bleeding.",
    "fracture": "If you suspect a fracture, do not make the animal walk. Use a flat support surface or carrier if transport is needed and seek urgent veterinary care.",
    "poison": "If poisoning is possible, urgent veterinary help is needed. Do not force food, milk, or home remedies unless a veterinary professional tells you to do so.",
    "vomit": "Vomiting can be mild or urgent depending on frequency, weakness, bloating, poisoning risk, and dehydration. If vomiting repeats, there is blood, or the animal looks weak, seek veterinary care quickly.",
    "diarrhea": "Diarrhea can lead to dehydration. Monitor water intake, avoid risky human foods, and seek vet advice if diarrhea is severe, bloody, prolonged, or paired with weakness or vomiting.",
    "dog": "Dogs in pain may hide it. Watch for limping, whining, panting, swelling, reluctance to move, or bleeding. Keep the dog calm and avoid sudden touching.",
    "cat": "Cats often hide pain. Warning signs include hiding, hissing when touched, trouble walking, rapid breathing, refusal to eat, or visible wounds.",
    "rabbit": "Rabbits are fragile and stress-sensitive. Avoid rough handling, keep them warm and quiet, and seek help quickly if they stop eating, seem weak, or appear injured.",
    "bird": "If a bird is injured, reduce noise, place it in a ventilated box, avoid unnecessary handling, and contact a rescue or vet familiar with birds.",
    "first time": "If this is your first time helping a domestic animal, focus on three things first: stay safe, reduce the animal's stress, and get professional help early when the situation is unclear.",
    "pet": "For pets, avoid guessing with human medicines or force-feeding. Note the symptoms, when they started, and whether the animal is eating, walking, breathing, and responding normally.",
    "eat": "Loss of appetite can be a warning sign if it continues, especially with vomiting, diarrhea, pain, fever, or weakness. Encourage calm and water access, and seek vet advice if it persists.",
    "water": "Hydration matters. If the animal cannot drink, vomits repeatedly, or looks weak or sunken-eyed, it may need urgent veterinary attention.",
    "seizure": "During a seizure, keep the area safe, do not put anything in the mouth, reduce stimulation, and seek urgent veterinary care after the episode.",
    "breathing": "Trouble breathing is urgent. Keep the animal calm, avoid pressure on the neck or chest, and contact emergency veterinary or rescue help immediately.",
}


def medical_chat_reply(
    messages: list[dict],
    animal_type: str | None,
    health_status: str | None,
    detected_conditions: list[str],
    location_name: str | None,
) -> tuple[str, bool, str]:
    groq_api_key = settings.groq_api_key.strip() if settings.groq_api_key else None
    openai_api_key = settings.openai_api_key.strip() if settings.openai_api_key else None
    context_block = (
        f"Animal type: {animal_type or 'unknown'}\n"
        f"Health status: {health_status or 'unknown'}\n"
        f"Detected conditions: {', '.join(detected_conditions) if detected_conditions else 'none'}\n"
        f"Location: {location_name or 'unknown'}"
    )

    if groq_api_key and OpenAI is not None:
        try:
            client = OpenAI(
                api_key=groq_api_key,
                base_url=settings.groq_base_url,
            )
            input_messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCase context:\n" + context_block}]
            input_messages.extend(messages)

            response = client.chat.completions.create(
                model=settings.groq_chat_model,
                messages=input_messages,
            )
            reply = response.choices[0].message.content if response.choices else ""
            return reply or fallback_medical_reply(context_block, messages, animal_type, health_status, detected_conditions), False, f"groq:{settings.groq_chat_model}"
        except Exception:
            return fallback_medical_reply(context_block, messages, animal_type, health_status, detected_conditions), True, "local-fallback"

    if openai_api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=openai_api_key)
            input_messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCase context:\n" + context_block}]
            input_messages.extend(messages)

            response = client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=input_messages,
            )
            reply = response.choices[0].message.content if response.choices else ""
            return reply or fallback_medical_reply(context_block, messages, animal_type, health_status, detected_conditions), False, f"openai:{settings.openai_chat_model}"
        except Exception:
            return fallback_medical_reply(context_block, messages, animal_type, health_status, detected_conditions), True, "local-fallback"

    return fallback_medical_reply(context_block, messages, animal_type, health_status, detected_conditions), True, "local-fallback"


def fallback_medical_reply(
    context_block: str,
    messages: list[dict],
    animal_type: str | None,
    health_status: str | None,
    detected_conditions: list[str],
) -> str:
    latest_user_message = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "").lower()
    matched_advice = []

    for keyword, advice in GENERAL_KNOWLEDGE.items():
        if keyword in latest_user_message:
            matched_advice.append(advice)

    if animal_type and animal_type.lower() in GENERAL_KNOWLEDGE:
        matched_advice.append(GENERAL_KNOWLEDGE[animal_type.lower()])

    urgency_line = urgency_for_status(health_status, detected_conditions)
    if not matched_advice:
        matched_advice.append(
            "Please tell me what you are seeing, for example bleeding, swelling, limping, vomiting, trouble breathing, not eating, weakness, or a wound."
        )

    unique_advice = []
    seen = set()
    for item in matched_advice:
        if item not in seen:
            unique_advice.append(item)
            seen.add(item)

    response_parts = [
        "Care Bot is using local guidance mode right now.",
        f"Case context:\n{context_block}",
        urgency_line,
        "Helpful guidance:",
        "\n".join(f"- {item}" for item in unique_advice[:4]),
        "If you want, ask a follow-up like: 'What should I do before the vet arrives?' or 'Is this urgent for a dog?'",
    ]
    return "\n\n".join(response_parts)


def urgency_for_status(health_status: str | None, detected_conditions: list[str]) -> str:
    if health_status == "Serious":
        return "This looks urgent. Contact a rescue team or veterinarian immediately."
    if any("bleeding" in item.lower() or "open wound" in item.lower() for item in detected_conditions):
        return "Visible injury markers suggest this may need urgent professional help."
    if health_status == "Mild":
        return "This may not be immediately critical, but the animal should still be checked soon."
    return "Share the visible symptoms and I will help you decide the next safe step."
