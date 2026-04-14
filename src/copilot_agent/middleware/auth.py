"""JWT authentication for WebSocket and REST endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt


def create_token(
    user: str,
    session_id: str,
    site: str,
    secret: str,
    algorithm: str = "HS256",
    expiry_hours: int = 24,
) -> str:
    """Create a signed JWT token."""
    payload = {
        "sub": user,
        "sid": session_id,
        "site": site,
        "exp": datetime.now(UTC) + timedelta(hours=expiry_hours),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def verify_token(token: str, secret: str, algorithm: str = "HS256") -> dict:
    """Verify and decode a JWT token. Raises jwt.InvalidTokenError on failure."""
    return jwt.decode(token, secret, algorithms=[algorithm])
