"""JWT authentication for Knowledge Service requests.

Auth is DISABLED by default (RAG_AUTH_ENABLED=false in config) so local dev
works with no setup. Set RAG_AUTH_ENABLED=true and RAG_JWT_SECRET to enforce
it — no other code changes needed.

For service-to-service calls that don't want to mint JWTs, the app layer also
supports static API keys (KNOWLEDGE_API_KEYS) — see app/middleware/auth.py.
Ported from the LLM Gateway's ``features/auth.py`` — same mechanism.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import jwt

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class TokenClaims:
    """Decoded and validated JWT claims."""

    sub: str
    iss: str
    aud: str
    exp: int
    iat: int
    extra: dict[str, Any]


class JWTValidator:
    """Validates JWT tokens for Knowledge Service access."""

    def __init__(
        self,
        secret: str | None = None,
        algorithm: str | None = None,
        issuer: str | None = None,
        audience: str | None = None,
    ) -> None:
        self._secret = secret or settings.jwt_secret
        self._algorithm = algorithm or settings.jwt_algorithm
        self._issuer = issuer or settings.jwt_issuer
        self._audience = audience or settings.jwt_audience

    def validate(self, token: str) -> TokenClaims:
        """Decode and validate a JWT token.

        Raises:
            InvalidTokenError: expired, malformed, or wrong secret/issuer/audience.
        """
        if not self._secret:
            raise InvalidTokenError("JWT secret is not configured")

        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["sub", "exp", "iat", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError:
            raise InvalidTokenError("Token has expired")
        except jwt.InvalidIssuerError:
            raise InvalidTokenError("Invalid token issuer")
        except jwt.InvalidAudienceError:
            raise InvalidTokenError("Invalid token audience")
        except jwt.DecodeError as e:
            raise InvalidTokenError(f"Token decode failed: {e}")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")

        return TokenClaims(
            sub=payload["sub"],
            iss=payload["iss"],
            aud=payload["aud"],
            exp=payload["exp"],
            iat=payload["iat"],
            extra={
                k: v for k, v in payload.items() if k not in ("sub", "iss", "aud", "exp", "iat")
            },
        )

    def create_token(self, subject: str, ttl_seconds: int = 3600, **extra_claims: Any) -> str:
        """Mint a signed JWT (for testing / internal service-to-service use)."""
        now = int(time.time())
        payload = {
            "sub": subject,
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "exp": now + ttl_seconds,
            **extra_claims,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)


class InvalidTokenError(Exception):
    """Raised when JWT validation fails."""
