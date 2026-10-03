"""
app.core.auth
~~~~~~~~~~~~~
FastAPI dependency that protects endpoints with a Bearer JWT.

Token creation, password checking and token validation live in
app.services.auth_service.

Usage::

    @router.post("/rag/ingest")
    async def rag_ingest(..., username: str = Depends(verify_token)):
        ...
"""
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services import auth_service

_bearer = HTTPBearer(auto_error=True)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Return the authenticated username, or raise 401 on an invalid/expired token."""
    return auth_service.verify_access_token(credentials.credentials)
