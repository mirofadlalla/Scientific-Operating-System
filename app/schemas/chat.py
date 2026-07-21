"""Pydantic request / response schemas for the chat endpoints."""
from pydantic import BaseModel


class UserQuery(BaseModel):
    session_id: str
    user_id: str
    text_input: str
