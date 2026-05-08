from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Speaker(str, Enum):
    patient = "patient"
    provider = "provider"
    system = "system"
    unknown = "unknown"


class UrgencyLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    emergency = "Emergency"


class ChatMessage(BaseModel):
    speaker: Speaker = Speaker.unknown
    text: str = Field(..., min_length=1, max_length=5000)
    timestamp: Optional[datetime] = None

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class AnalyzeConversationRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional external id from the source system.",
        max_length=120,
    )
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=200)
    patient_profile: Optional[dict[str, Any]] = Field(default=None)
    include_embeddings: bool = Field(default=False)


class MedicalEntity(BaseModel):
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: float = Field(ge=0, le=1)
    source: str


class SymptomFinding(BaseModel):
    name: str
    severity: Optional[str] = None
    negated: bool = False
    confidence: float = Field(ge=0, le=1)


class UrgencyPrediction(BaseModel):
    level: UrgencyLevel
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]


class ConversationAnalysis(BaseModel):
    id: UUID
    conversation_id: Optional[str] = None
    summary: str
    symptoms: list[SymptomFinding]
    entities: list[MedicalEntity]
    urgency: UrgencyPrediction
    key_topics: list[str]
    recommendations: list[str]
    model_info: dict[str, str]
    created_at: datetime


class ConversationRecord(BaseModel):
    id: UUID
    conversation_id: Optional[str]
    summary: str
    urgency_level: UrgencyLevel
    urgency_confidence: float
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
