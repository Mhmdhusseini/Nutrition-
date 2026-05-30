"""
Smart Diet Planner — FastAPI Application

Production-ready REST API for personalised diet plan generation.
Models are loaded once at startup and kept in memory for fast inference.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.plan import router as plan_router
from api.services.model_loader import store

# ── Logging ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load models once at startup ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all trained models into memory before accepting requests."""
    logger.info("Starting model loading …")
    store.load()
    logger.info("Models ready — accepting requests.")
    yield
    logger.info("Shutting down.")


# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Diet Planner API",
    description=(
        "A production-ready REST API that generates personalised daily nutrition "
        "plans. Powered by PyTorch models trained on nutritional datasets.\n\n"
        "### Features\n"
        "- **Calorie prediction** based on age, gender, weight, height & activity level\n"
        "- **Nutrient breakdown** for breakfast, lunch & dinner\n"
        "- **Recipe recommendations** optimised to match predicted targets\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow Flutter & any frontend ─────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────

app.include_router(plan_router)


# ── Utility endpoints ──────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """API landing — basic info and links."""
    return {
        "name": "Smart Diet Planner API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Info"])
async def health():
    """Lightweight health check for load balancers / Railway."""
    return {"status": "healthy"}
