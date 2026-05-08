from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db
from app.models.conversation import ConversationAnalysisModel
from app.nlp.preprocessing import messages_to_transcript
from app.schemas.conversation import (
    AnalyzeConversationRequest,
    ConversationAnalysis,
    ConversationRecord,
    HealthResponse,
)
from app.services import get_analyzer, get_cache

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, version=settings.version)


@router.post(
    "/analyze",
    response_model=ConversationAnalysis,
    status_code=status.HTTP_201_CREATED,
    tags=["analysis"],
)
def analyze_conversation(
    payload: AnalyzeConversationRequest,
    db: Session = Depends(get_db),
) -> ConversationAnalysis:
    cache = get_cache()
    cache_key = cache.key_for(payload.model_dump())
    cached = cache.get(cache_key)
    if cached:
        return ConversationAnalysis.model_validate(cached)

    analyzer = get_analyzer()
    analysis = analyzer.analyze(payload)
    transcript = messages_to_transcript([message.model_dump() for message in payload.messages])
    record = ConversationAnalysisModel(
        id=str(analysis.id),
        conversation_id=analysis.conversation_id,
        transcript=transcript,
        summary=analysis.summary,
        urgency_level=analysis.urgency.level.value,
        urgency_confidence=analysis.urgency.confidence,
        symptoms=[item.model_dump() for item in analysis.symptoms],
        entities=[item.model_dump() for item in analysis.entities],
        key_topics=analysis.key_topics,
        recommendations=analysis.recommendations,
        model_info=analysis.model_info,
    )
    db.add(record)
    db.commit()
    cache.set(cache_key, analysis.model_dump(mode="json"))
    return analysis


@router.get("/conversations", response_model=list[ConversationRecord], tags=["analysis"])
def list_conversations(db: Session = Depends(get_db), limit: int = 25) -> list[ConversationRecord]:
    safe_limit = max(1, min(limit, 100))
    records = db.scalars(
        select(ConversationAnalysisModel)
        .order_by(ConversationAnalysisModel.created_at.desc())
        .limit(safe_limit)
    ).all()
    return [
        ConversationRecord(
            id=UUID(record.id),
            conversation_id=record.conversation_id,
            summary=record.summary,
            urgency_level=record.urgency_level,
            urgency_confidence=record.urgency_confidence,
            created_at=record.created_at,
        )
        for record in records
    ]


@router.get("/conversations/{analysis_id}", response_model=ConversationAnalysis, tags=["analysis"])
def get_conversation(analysis_id: UUID, db: Session = Depends(get_db)) -> ConversationAnalysis:
    record = db.get(ConversationAnalysisModel, str(analysis_id))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    return ConversationAnalysis(
        id=UUID(record.id),
        conversation_id=record.conversation_id,
        summary=record.summary,
        symptoms=record.symptoms,
        entities=record.entities,
        urgency={"level": record.urgency_level, "confidence": record.urgency_confidence, "reasons": []},
        key_topics=record.key_topics,
        recommendations=record.recommendations,
        model_info=record.model_info,
        created_at=record.created_at,
    )
