"""Mint a JWT for local testing when RAG_AUTH_ENABLED=true.

Usage::

    python scripts/mint_token.py --subject ops --role admin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.auth import JWTValidator  # noqa: E402
from rag.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a Knowledge Service JWT")
    parser.add_argument("--subject", default="local-dev", help="Token subject (principal name)")
    parser.add_argument("--role", default="user", choices=["user", "admin"], help="Role claim")
    parser.add_argument("--ttl", type=int, default=3600, help="Time to live, in seconds")
    args = parser.parse_args()

    if not settings.jwt_secret:
        print("RAG_JWT_SECRET is not set — set it in .env before minting a token.", file=sys.stderr)
        raise SystemExit(1)

    validator = JWTValidator()
    token = validator.create_token(args.subject, ttl_seconds=args.ttl, role=args.role)
    print(token)


if __name__ == "__main__":
    main()
