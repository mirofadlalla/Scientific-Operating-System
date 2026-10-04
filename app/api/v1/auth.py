from fastapi import APIRouter

from app.controllers import auth_controller
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """
    Authenticate with username + password stored in MongoDB Atlas.

    Returns a Bearer JWT token valid for 60 minutes.
    Include this token in the `Authorization: Bearer <token>` header
    when calling protected endpoints (e.g. POST /api/v1/rag/ingest).
    """
    return await auth_controller.login(body)
