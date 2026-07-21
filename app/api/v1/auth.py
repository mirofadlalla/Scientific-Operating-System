"""
app.api.v1.auth
~~~~~~~~~~~~~~~
POST /api/v1/auth/login  — Exchange credentials for a JWT access token.

The username and password are verified against the MongoDB Atlas `users`
collection. Passwords must be stored as bcrypt hashes in MongoDB.

To create a user in MongoDB Atlas Shell:
  db.users.insertOne({
    username: "omar",
    password: "<bcrypt_hash>"
  })

Generate a hash:
  python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
"""
import logging

from fastapi import APIRouter, HTTPException, status

from app.core.auth import authenticate_user, create_access_token
from app.schemas.auth import LoginRequest, TokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """
    Authenticate with username + password stored in MongoDB Atlas.

    Returns a Bearer JWT token valid for 60 minutes.
    Include this token in the `Authorization: Bearer <token>` header
    when calling protected endpoints (e.g. POST /api/v1/rag/ingest).
    """
    user = await authenticate_user(body.username, body.password)
    if user is None:
        logger.warning("Failed login attempt for username: %s", body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user["username"])
    logger.info("User logged in: %s", user["username"])
    return TokenResponse(access_token=token)
