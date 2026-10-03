"""
app.core.exceptions
~~~~~~~~~~~~~~~~~~~
Framework-agnostic application errors.

Services raise these; a single exception handler (registered in main.py)
turns them into HTTP responses, so services never import HTTPException.
"""
from typing import Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        if detail is not None:
            self.detail = detail
        self.headers = headers
        super().__init__(self.detail)


class BadRequestError(AppError):
    status_code = 400


class UnauthorizedError(AppError):
    status_code = 401


class ServiceUnavailableError(AppError):
    status_code = 503


class InternalError(AppError):
    status_code = 500


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )
