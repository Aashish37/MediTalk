from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.nlp.entities import extract_entities, extract_symptoms
from app.nlp.model_registry import ModelRegistry
from app.nlp.preprocessing import messages_to_transcript
from app.nlp.summarization import summarize
from app.nlp.topics import extract_key_topics
from app.nlp.urgency import classify_urgency
from app.schemas.conversation import AnalyzeConversationRequest, ConversationAnalysis, UrgencyLevel


class ConversationAnalyzer:
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry

    def analyze(
        self,
        payload: AnalyzeConversationRequest,
        analysis_id: UUID | None = None,
    ) -> ConversationAnalysis:
        transcript = messages_to_transcript([message.model_dump() for message in payload.messages])
        entities = extract_entities(transcript, self.model_registry.spacy_nlp)
        symptoms = extract_symptoms(transcript, entities)
        urgency = classify_urgency(transcript, entities, self.model_registry.zero_shot_classifier)
        summary = summarize(transcript, self.model_registry.summarizer)
        topics = extract_key_topics(transcript)

        return ConversationAnalysis(
            id=analysis_id or uuid4(),
            conversation_id=payload.conversation_id,
            summary=summary,
            symptoms=symptoms,
            entities=entities,
            urgency=urgency,
            key_topics=topics,
            recommendations=_recommendations(urgency.level),
            model_info=self.model_registry.model_info(),
            created_at=datetime.now(timezone.utc),
        )


def _recommendations(level: UrgencyLevel) -> list[str]:
    if level == UrgencyLevel.emergency:
        return [
            "Advise immediate emergency care or local emergency services.",
            "Escalate to a licensed clinician for urgent review.",
        ]
    if level == UrgencyLevel.high:
        return [
            "Recommend same-day clinical evaluation.",
            "Monitor symptom progression and red-flag changes closely.",
        ]
    if level == UrgencyLevel.medium:
        return [
            "Recommend timely provider follow-up.",
            "Collect additional context such as duration, severity, medications, and history.",
        ]
    return [
        "Provide routine guidance and self-monitoring instructions.",
        "Escalate if symptoms worsen or new red flags appear.",
    ]
