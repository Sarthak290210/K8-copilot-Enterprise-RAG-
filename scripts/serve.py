"""Production entrypoint for the API (used by the Dockerfile).

For local development use `make api`, which runs uvicorn with --reload.
"""

import sys

import uvicorn

from app.config import settings


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        # Async psycopg can't run on Windows' default ProactorEventLoop.
        loop="asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto",
        # X-Forwarded-For is trusted only from uvicorn's default 127.0.0.1 (override with the
        # FORWARDED_ALLOW_IPS env var once a real proxy sits in front). Trusting "*" would let
        # any client spoof its IP and bypass the per-IP rate limits.
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
