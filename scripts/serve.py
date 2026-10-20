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
        # Trust X-Forwarded-* from a reverse proxy so client IPs (rate limiting) are real.
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
