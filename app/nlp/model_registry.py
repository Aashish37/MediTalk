from __future__ import annotations

from functools import cached_property
from typing import Any

from app.core.config import Settings


class ModelRegistry:
    """Lazy optional model loader.

    The API runs without heavyweight NLP assets, which keeps local development and
    CI friendly. When enabled, installed transformer/spaCy models enrich output.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def spacy_nlp(self) -> Any | None:
        if not self.settings.enable_spacy:
            return None
        try:
            import spacy

            return spacy.load(self.settings.spacy_model)
        except Exception:
            return None

    @cached_property
    def summarizer(self) -> Any | None:
        if not self.settings.enable_transformers:
            return None
        try:
            from transformers import pipeline

            return pipeline("summarization", model=self.settings.summary_model)
        except Exception:
            return None

    @cached_property
    def zero_shot_classifier(self) -> Any | None:
        if not self.settings.enable_transformers:
            return None
        try:
            from transformers import pipeline

            return pipeline("zero-shot-classification", model=self.settings.zero_shot_model)
        except Exception:
            return None

    @cached_property
    def embedding_model(self) -> Any | None:
        if not self.settings.enable_transformers:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self.settings.embedding_model)
        except Exception:
            return None

    def model_info(self) -> dict[str, str]:
        return {
            "ner": self.settings.spacy_model if self.spacy_nlp else "rule_based_medical_lexicon",
            "summary": self.settings.summary_model if self.summarizer else "extractive_textrank_fallback",
            "urgency": self.settings.zero_shot_model
            if self.zero_shot_classifier
            else "clinical_rules_and_keyword_scoring",
            "embeddings": self.settings.embedding_model if self.embedding_model else "tfidf_keywords",
        }
