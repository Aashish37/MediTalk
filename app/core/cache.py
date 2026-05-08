from __future__ import annotations

import hashlib
import json
from typing import Any

from redis import Redis

from app.core.config import Settings


class AnalysisCache:
    def __init__(self, settings: Settings):
        self.client: Redis | None = None
        if settings.redis_url:
            try:
                self.client = Redis.from_url(settings.redis_url, decode_responses=True)
                self.client.ping()
            except Exception:
                self.client = None

    def key_for(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return f"analysis:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.client:
            return None
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 900) -> None:
        if self.client:
            self.client.setex(key, ttl_seconds, json.dumps(value, default=str))
