from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MediTalk"
    app_env: str = "development"
    log_level: str = "INFO"
    version: str = "0.1.0"
    database_url: str = "sqlite:///./meditalk.db"
    redis_url: Optional[str] = None
    enable_transformers: bool = False
    enable_spacy: bool = False
    spacy_model: str = "en_core_web_sm"
    summary_model: str = "sshleifer/distilbart-cnn-12-6"
    zero_shot_model: str = "facebook/bart-large-mnli"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
