"""
FastAPI gateway application.

This is the main entry point for the API.
Includes automatic GEPA optimization integration.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.core.config import settings
from src.db.session import init_db
from src.gateway.routes import tasks, health, agents

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    logger.info("starting_application")
    await init_db()
    logger.info("database_initialized")
    
    # Load optimized prompts from GEPA
    try:
        from src.core.gepa.middleware import GEPAPromptLoader
        results = await GEPAPromptLoader.load_all()
        loaded = sum(1 for v in results.values() if v)
        logger.info("gepa_prompts_loaded", count=loaded, total=len(results))
    except Exception as e:
        logger.warning("gepa_prompt_load_failed", error=str(e))
    
    # Start GEPA automation in background (if enabled)
    if os.getenv("GEPA_AUTO_ENABLED", "true").lower() == "true":
        try:
            from src.core.gepa.middleware import start_gepa_automation_background
            await start_gepa_automation_background()
            logger.info("gepa_automation_enabled")
        except Exception as e:
            logger.warning("gepa_automation_start_failed", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("shutting_down_application")


# Create FastAPI app
app = FastAPI(
    title="Agent Platform",
    description="Modular agent orchestration API",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "Agent Platform",
        "version": "0.1.0",
        "docs": "/docs",
    }
