from fastapi import APIRouter, WebSocket

from app.controllers import chat_controller, voice_controller

from app.schemas.chat import UserQuery

router = APIRouter(tags=["Chat"])


@router.post("/orchestrate")
async def process_user_input(query: UserQuery):
    """
    Stream an AI response token-by-token.

    The response is plain text streamed via `text/plain` — the frontend
    appends each received chunk to the message bubble in real time.
    """
    return chat_controller.orchestrate(query)


@router.websocket("/ws/voice")
async def websocket_voice_channel(websocket: WebSocket, session_id: str = "ws_session"):
    """Real-time bi-directional voice channel."""
    await voice_controller.handle_voice_channel(websocket, session_id)
