from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base


class ConversationAnalysisModel(Base):
    __tablename__ = "conversation_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[Optional[str]] = mapped_column(String(120), index=True)
    transcript: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    urgency_level: Mapped[str] = mapped_column(String(24), index=True)
    urgency_confidence: Mapped[float] = mapped_column(Float)
    symptoms: Mapped[list] = mapped_column(JSON)
    entities: Mapped[list] = mapped_column(JSON)
    key_topics: Mapped[list] = mapped_column(JSON)
    recommendations: Mapped[list] = mapped_column(JSON)
    model_info: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
