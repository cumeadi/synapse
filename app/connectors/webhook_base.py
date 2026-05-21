"""
Synapse — Webhook Connector Base.

Provides the abstract interface that all connector implementations must satisfy,
plus shared HMAC-SHA256 signature verification and DB idempotency helpers.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from abc import ABC, abstractmethod

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import WebhookEvent

logger = logging.getLogger("synapse.connectors.webhook")

# Set SYNAPSE_WEBHOOK_SKIP_VERIFY=true to disable signature checks in development.
_SKIP_VERIFY = os.getenv("SYNAPSE_WEBHOOK_SKIP_VERIFY", "false").lower() == "true"

# Supported connector identifiers
SUPPORTED_CONNECTORS: list[str] = ["slack", "jira", "github"]


# ────────────────────────────────────────────────────────────────────
# HMAC helpers
# ────────────────────────────────────────────────────────────────────

def verify_hmac_sha256(secret: str, body: bytes, provided_sig: str) -> bool:
    """
    Constant-time HMAC-SHA256 comparison.

    Args:
        secret: The shared webhook secret (from env var).
        body: The raw request body bytes.
        provided_sig: The signature string from the request header.
                      May include a prefix like 'sha256=' which is stripped.

    Returns:
        True if the signature is valid.
    """
    if _SKIP_VERIFY:
        logger.warning("SYNAPSE_WEBHOOK_SKIP_VERIFY is enabled — skipping signature check.")
        return True

    # Strip common prefixes (sha256=, v0=sha256=)
    clean_sig = provided_sig
    for prefix in ("sha256=", "v0=sha256="):
        if clean_sig.startswith(prefix):
            clean_sig = clean_sig[len(prefix):]
            break

    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, clean_sig)


# ────────────────────────────────────────────────────────────────────
# Idempotency helper
# ────────────────────────────────────────────────────────────────────

async def is_duplicate(
    db: AsyncSession,
    namespace_id,
    connector_type: str,
    event_id: str,
) -> bool:
    """
    Check whether this event has already been processed.

    Uses the unique constraint on (namespace_id, connector_type, event_id).

    Args:
        db: Active async database session.
        namespace_id: UUID of the target namespace.
        connector_type: Connector identifier string.
        event_id: The source system's stable unique event identifier.

    Returns:
        True if a WebhookEvent record already exists for this event.
    """
    stmt = select(WebhookEvent).where(
        WebhookEvent.namespace_id == namespace_id,
        WebhookEvent.connector_type == connector_type,
        WebhookEvent.event_id == event_id,
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None


# ────────────────────────────────────────────────────────────────────
# Abstract Connector Base
# ────────────────────────────────────────────────────────────────────

class WebhookConnector(ABC):
    """
    Abstract base class for all Synapse webhook connectors.

    Each connector encapsulates:
    - Signature verification for its platform
    - A stable unique event ID extractor
    - An event type extractor
    - A normalizer that converts the raw payload to a plain-text
      description suitable for LLM entity/relationship extraction
    """

    #: Override in subclasses — used for routing and logging
    connector_type: str = ""

    @abstractmethod
    def verify_signature(self, body: bytes, headers: dict[str, str]) -> bool:
        """
        Verify the HMAC signature on this request.

        Args:
            body: Raw request body bytes.
            headers: Lowercased request headers dict.

        Returns:
            True if the signature is valid (or skip-verify is enabled).
        """

    @abstractmethod
    def extract_event_id(self, payload: dict, headers: dict[str, str]) -> str:
        """
        Extract a stable, unique identifier for this event.

        This is used for idempotency deduplication.

        Args:
            payload: Parsed JSON payload.
            headers: Lowercased request headers dict.

        Returns:
            A string that uniquely identifies this event in the source system.
        """

    @abstractmethod
    def extract_event_type(self, payload: dict, headers: dict[str, str]) -> str:
        """
        Extract the event type label.

        Args:
            payload: Parsed JSON payload.
            headers: Lowercased request headers dict.

        Returns:
            A short string like 'message', 'issue_created', 'push'.
        """

    @abstractmethod
    def normalize(self, payload: dict, headers: dict[str, str]) -> str | None:
        """
        Convert the raw payload to a plain-text description for LLM extraction.

        Args:
            payload: Parsed JSON payload.
            headers: Lowercased request headers dict.

        Returns:
            A plain-text string, or None if this event type should be skipped.
        """
