from __future__ import annotations

import re
from collections.abc import Iterable

from app.nlp.preprocessing import sentence_split
from app.schemas.conversation import MedicalEntity, SymptomFinding


MEDICAL_TERMS: dict[str, list[str]] = {
    "SYMPTOM": [
        "chest pain",
        "shortness of breath",
        "fever",
        "cough",
        "headache",
        "dizziness",
        "nausea",
        "vomiting",
        "fatigue",
        "abdominal pain",
        "rash",
        "sore throat",
        "palpitations",
        "weakness",
        "blurred vision",
        "swelling",
        "bleeding",
        "confusion",
        "severe pain",
    ],
    "MEDICATION": [
        "aspirin",
        "ibuprofen",
        "acetaminophen",
        "paracetamol",
        "metformin",
        "insulin",
        "amoxicillin",
        "atorvastatin",
        "lisinopril",
        "albuterol",
        "warfarin",
        "prednisone",
        "omeprazole",
    ],
    "DISEASE": [
        "diabetes",
        "hypertension",
        "asthma",
        "covid",
        "influenza",
        "pneumonia",
        "migraine",
        "stroke",
        "heart attack",
        "myocardial infarction",
        "kidney disease",
        "depression",
        "anxiety",
    ],
    "BODY_PART": [
        "head",
        "chest",
        "abdomen",
        "stomach",
        "arm",
        "leg",
        "back",
        "throat",
        "heart",
        "lung",
        "lungs",
        "knee",
        "shoulder",
        "eye",
        "ear",
    ],
    "CONDITION": [
        "pregnancy",
        "allergy",
        "infection",
        "dehydration",
        "inflammation",
        "fracture",
        "seizure",
        "high blood pressure",
        "low blood sugar",
    ],
}

NEGATION_RE = re.compile(r"\b(no|not|denies|without|negative for|free of)\b", re.IGNORECASE)
SEVERITY_RE = re.compile(r"\b(mild|moderate|severe|worst|unbearable|sharp|crushing)\b", re.IGNORECASE)


def extract_entities(text: str, spacy_nlp: object | None = None) -> list[MedicalEntity]:
    entities = _lexicon_entities(text)
    if spacy_nlp:
        entities.extend(_spacy_entities(text, spacy_nlp))
    return _dedupe_entities(entities)


def extract_symptoms(text: str, entities: Iterable[MedicalEntity]) -> list[SymptomFinding]:
    findings: list[SymptomFinding] = []
    lowered = text.lower()
    for entity in entities:
        if entity.label != "SYMPTOM":
            continue
        window = _context_window(lowered, entity.text.lower())
        severity_match = SEVERITY_RE.search(window)
        findings.append(
            SymptomFinding(
                name=entity.text,
                severity=severity_match.group(1).lower() if severity_match else None,
                negated=_is_negated(lowered, entity.text.lower()),
                confidence=entity.confidence,
            )
        )
    unique: dict[str, SymptomFinding] = {}
    for finding in findings:
        key = finding.name.lower()
        if key not in unique or finding.confidence > unique[key].confidence:
            unique[key] = finding
    return list(unique.values())


def _lexicon_entities(text: str) -> list[MedicalEntity]:
    entities: list[MedicalEntity] = []
    for label, terms in MEDICAL_TERMS.items():
        for term in terms:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(
                    MedicalEntity(
                        text=match.group(0),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.88,
                        source="medical_lexicon",
                    )
                )
    return entities


def _spacy_entities(text: str, spacy_nlp: object) -> list[MedicalEntity]:
    doc = spacy_nlp(text)
    allowed = {"DISEASE", "CHEMICAL", "DRUG", "ORG", "PERSON", "GPE"}
    label_map = {"CHEMICAL": "MEDICATION", "DRUG": "MEDICATION"}
    return [
        MedicalEntity(
            text=ent.text,
            label=label_map.get(ent.label_, ent.label_),
            start=ent.start_char,
            end=ent.end_char,
            confidence=0.72 if ent.label_ in allowed else 0.55,
            source="spacy",
        )
        for ent in doc.ents
        if ent.label_ in allowed
    ]


def _context_window(text: str, term: str, radius: int = 60) -> str:
    index = text.find(term)
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(term) + radius)
    return text[start:end]


def _is_negated(text: str, term: str) -> bool:
    for sentence in sentence_split(text):
        index = sentence.lower().find(term)
        if index < 0:
            continue
        prefix = sentence[:index]
        local_prefix = prefix[-45:]
        if NEGATION_RE.search(local_prefix):
            return True
        if re.search(rf"\bdenies\s+{re.escape(term)}\b", sentence, re.IGNORECASE):
            return True
    return False


def _dedupe_entities(entities: list[MedicalEntity]) -> list[MedicalEntity]:
    unique: dict[tuple[str, str, int | None], MedicalEntity] = {}
    for entity in entities:
        key = (entity.text.lower(), entity.label, entity.start)
        existing = unique.get(key)
        if existing is None or entity.confidence > existing.confidence:
            unique[key] = entity
    return sorted(unique.values(), key=lambda item: (item.start is None, item.start or 0, item.label))
