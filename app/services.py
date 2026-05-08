from functools import lru_cache

from app.core.cache import AnalysisCache
from app.core.config import get_settings
from app.nlp.analyzer import ConversationAnalyzer
from app.nlp.model_registry import ModelRegistry


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry(get_settings())


@lru_cache
def get_analyzer() -> ConversationAnalyzer:
    return ConversationAnalyzer(get_model_registry())


@lru_cache
def get_cache() -> AnalysisCache:
    return AnalysisCache(get_settings())
