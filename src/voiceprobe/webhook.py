"""HTTP receiver for Telnyx Voice API events."""

import logging
from typing import Any

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceProbe Webhook")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight health response."""
    return {"status": "ok"}


@app.post("/webhooks/telnyx")
async def receive_telnyx_webhook(request: Request) -> dict[str, bool]:
    """Acknowledge Telnyx events during initial integration."""
    payload: dict[str, Any] = await request.json()

    event_type = payload.get("data", {}).get("event_type", "unknown")
    logger.info("Received Telnyx event: %s", event_type)

    return {"received": True}
