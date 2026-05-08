from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db import init_db


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        try:
            logger.info("application_started", service=settings.app_name, env=settings.app_env)
        except TypeError:
            logger.info("application_started service=%s env=%s", settings.app_name, settings.app_env)
        yield

    app = FastAPI(
        title="MediTalk Medical Chat Analysis API",
        description="AI/NLP backend for summarizing and classifying medical conversations.",
        version=settings.version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
