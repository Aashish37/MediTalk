from app.nlp.entities import extract_entities, extract_symptoms
from app.nlp.urgency import classify_urgency


def test_entity_and_symptom_extraction() -> None:
    text = "Patient reports fever, cough, and asthma. Denies chest pain. Took aspirin."
    entities = extract_entities(text)
    labels = {entity.label for entity in entities}
    symptoms = extract_symptoms(text, entities)

    assert {"SYMPTOM", "DISEASE", "MEDICATION"}.issubset(labels)
    assert any(symptom.name.lower() == "chest pain" and symptom.negated for symptom in symptoms)


def test_urgency_rules() -> None:
    prediction = classify_urgency("Patient fainted and is now confused.", [])
    assert prediction.level == "Emergency"
