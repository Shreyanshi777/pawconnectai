from typing import Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class MedicalChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    animal_type: Optional[str] = None
    health_status: Optional[str] = None
    detected_conditions: list[str] = Field(default_factory=list)
    location_name: Optional[str] = None


class MedicalChatResponse(BaseModel):
    reply: str
    model: str
    fallback_used: bool = False
