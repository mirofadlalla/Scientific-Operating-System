"""
app/main.py
~~~~~~~~~~~
Entry point for the AI Scientific Operating System API.

This file is intentionally thin — it only wires together the pieces defined
in app/core/ and app/api/.  All business logic lives elsewhere:

  app/core/orchestration.py  ← routing, prompts, composite detection
  app/core/lifespan.py       ← startup / shutdown
  app/core/middleware.py     ← ReadinessMiddleware, MonitoringMiddleware
  app/api/v1/               ← versioned route handlers
  app/schemas/              ← Pydantic models
"""
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.lifespan   import lifespan
from app.core.middleware  import MonitoringMiddleware, ReadinessMiddleware
from app.api.v1.router   import api_router

# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-lixir Scientific Operating System",
    description=(
        "AI Scientific OS for Drug Discovery — Chemical analysis, biomedical mechanisms, "
        "RAG knowledge base, and real-time voice interaction.\n\n"
        "---\n\n"
        "🚀 **[Live Demo (Frontend) →  https://scientific-operating-system.vercel.app/]"
        "(https://scientific-operating-system.vercel.app/)**"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware (order matters: added last = runs first) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ReadinessMiddleware)
app.add_middleware(MonitoringMiddleware)

# ── Versioned API routes ──────────────────────────────────────────────────────
app.include_router(api_router)


# ──────────────────────────────────────────────────────────────────────────────
# Root-level routes (no version prefix — stable forever)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect HF Space root URL to interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
async def health_check():
    """
    Lightweight keep-alive probe.
    The React frontend pings this every 4 minutes to prevent HF Space sleep.
    """
    return {"status": "ok", "timestamp": time.time()}