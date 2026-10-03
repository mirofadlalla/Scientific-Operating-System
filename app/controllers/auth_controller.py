"""app.controllers.auth_controller"""
from app.schemas.auth import LoginRequest, TokenResponse
from app.services import auth_service


async def login(body: LoginRequest) -> TokenResponse:
    token = await auth_service.login(body.username, body.password)
    return TokenResponse(access_token=token)
