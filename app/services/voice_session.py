"""
app.services.voice_session
~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-connection state for the WebSocket voice channel, plus safe send helpers
that never raise on a closed socket.
"""
import asyncio
import json
import uuid
from typing import List, Optional

from fastapi import WebSocket
from starlette.websockets import WebSocketState


class VoiceSession:
    """Tracks state for a single WebSocket voice session."""

    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.audio_chunks: List[bytes] = []
        self.ai_streaming = False       # AI is currently streaming a response
        self.interrupted = False        # User interrupted AI mid-stream
        self.current_task: Optional[asyncio.Task] = None
        self.turn_id: str = ""
        self.client_vad: Optional[str] = None   # "silero" | "energy" as reported by the browser
        self._closed = False

    @property
    def audio_bytes(self) -> int:
        """Total size of the audio currently buffered for the next turn."""
        return sum(len(c) for c in self.audio_chunks)

    def new_turn(self) -> str:
        self.turn_id = uuid.uuid4().hex[:8]
        self.interrupted = False
        self.ai_streaming = False
        return self.turn_id

    # ── Safe senders: return False once the connection is gone ───────────────

    async def send_text(self, data: str) -> bool:
        if self._closed:
            return False
        try:
            if self.ws.client_state == WebSocketState.DISCONNECTED:
                self._closed = True
                return False
            await self.ws.send_text(data)
            return True
        except Exception:
            self._closed = True
            return False

    async def send_bytes(self, data: bytes) -> bool:
        if self._closed:
            return False
        try:
            if self.ws.client_state == WebSocketState.DISCONNECTED:
                self._closed = True
                return False
            await self.ws.send_bytes(data)
            return True
        except Exception:
            self._closed = True
            return False

    async def send_json(self, obj: dict) -> bool:
        return await self.send_text(json.dumps(obj))
