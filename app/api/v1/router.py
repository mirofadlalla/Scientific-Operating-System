"""
app.api.v1.router
~~~~~~~~~~~~~~~~~
Combines all v1 sub-routers under the /api/v1 prefix.
"""
from fastapi import APIRouter

from app.api.v1.chat       import router as chat_router
from app.api.v1.audio      import router as audio_router
from app.api.v1.rag        import router as rag_router
from app.api.v1.monitoring import router as monitoring_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(chat_router)
api_router.include_router(audio_router)
api_router.include_router(rag_router)
api_router.include_router(monitoring_router)
