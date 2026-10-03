"""app.controllers.chat_controller"""
from fastapi.responses import StreamingResponse

from app.schemas.chat import UserQuery
from app.services import chat_service


def orchestrate(query: UserQuery) -> StreamingResponse:
    """Plain-text token stream; the frontend appends each chunk to the message bubble."""
    return StreamingResponse(
        chat_service.stream_reply(query.text_input, query.session_id, query.user_id),
        media_type="text/plain",
    )
