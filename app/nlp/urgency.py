from __future__ import annotations

import re

from app.schemas.conversation import MedicalEntity, UrgencyLevel, UrgencyPrediction


EMERGENCY_PATTERNS = {
    "chest pain with breathing difficulty": r"\b(chest pain|crushing pain).{0,80}(shortness of breath|difficulty breathing)\b",
    "stroke warning signs": r"\b(stroke|face drooping|slurred speech|one-sided weakness|can't move)\b",
    "loss of consciousness": r"\b(fainted|unconscious|passed out|loss of consciousness)\b",
    "severe bleeding": r"\b(heavy bleeding|won't stop bleeding|coughing blood|vomiting blood)\b",
    "suicidal ideation": r"\b(suicidal|kill myself|self harm|overdose)\b",
}

HIGH_PATTERNS = {
    "high fever": r"\b(fever|temperature).{0,40}(103|104|105|40 c|41 c|very high)\b",
    "severe pain": r"\b(severe|worst|unbearable|crushing).{0,40}(pain|headache)\b",
    "pregnancy concern": r"\b(pregnant|pregnancy).{0,80}(bleeding|severe pain|dizziness)\b",
    "dehydration risk": r"\b(cannot keep fluids|no urination|dehydrated)\b",
}

MEDIUM_PATTERNS = {
    "persistent symptoms": r"\b(persistent|for three days|for 3 days|getting worse|worsening)\b",
    "infection signs": r"\b(fever|infection|pus|swelling|rash)\b",
    "moderate pain": r"\b(moderate pain|sharp pain|painful)\b",
}


def classify_urgency(
    text: str,
    entities: list[MedicalEntity],
    zero_shot_classifier: object | None = None,
) -> UrgencyPrediction:
    if zero_shot_classifier:
        prediction = _zero_shot_urgency(text, zero_shot_classifier)
        if prediction:
            return prediction

    reasons: list[str] = []
    level = UrgencyLevel.low
    score = 0.25

    emergency_hits = _matched_reasons(text, EMERGENCY_PATTERNS)
    high_hits = _matched_reasons(text, HIGH_PATTERNS)
    medium_hits = _matched_reasons(text, MEDIUM_PATTERNS)

    if emergency_hits:
        level = UrgencyLevel.emergency
        score = 0.94
        reasons.extend(emergency_hits)
    elif high_hits:
        level = UrgencyLevel.high
        score = 0.82
        reasons.extend(high_hits)
    elif medium_hits:
        level = UrgencyLevel.medium
        score = 0.68
        reasons.extend(medium_hits)
    else:
        symptom_count = len([entity for entity in entities if entity.label == "SYMPTOM"])
        if symptom_count >= 3:
            level = UrgencyLevel.medium
            score = 0.6
            reasons.append("multiple reported symptoms")
        else:
            reasons.append("no high-risk clinical language detected")

    return UrgencyPrediction(level=level, confidence=score, reasons=reasons)


def _matched_reasons(text: str, patterns: dict[str, str]) -> list[str]:
    return [
        reason
        for reason, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]


def _zero_shot_urgency(text: str, classifier: object) -> UrgencyPrediction | None:
    try:
        labels = ["Emergency", "High", "Medium", "Low"]
        result = classifier(
            text[:3000],
            candidate_labels=labels,
            hypothesis_template="This medical conversation indicates {} urgency.",
        )
        level = UrgencyLevel(result["labels"][0])
        confidence = float(result["scores"][0])
        return UrgencyPrediction(
            level=level,
            confidence=round(confidence, 3),
            reasons=["transformer zero-shot urgency classification"],
        )
    except Exception:
        return None
