"""
Synapse — Slack Connector.

Handles Slack Events API webhooks:
  - URL verification challenge (responded to inline)
  - HMAC-SHA256 signature verification via X-Slack-Signature
  - Normalises 'message' events to plain text for LLM extraction
  - Skips bot messages, edits, thread metadata noise

Environment variables:
  SYNAPSE_SLACK_SIGNING_SECRET  — Slack app signing secret (required)
  SYNAPSE_SLACK_VISIBILITY      — ABAC label for ingested data (default: internal)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time

from app.connectors.webhook_base import WebhookConnector, verify_hmac_sha256

logger = logging.getLogger("synapse.connectors.slack")

_SIGNING_SECRET = os.getenv("SYNAPSE_SLACK_SIGNING_SECRET", "")
_VISIBILITY = os.getenv("SYNAPSE_SLACK_VISIBILITY", "internal")

# Slack signatures expire after 5 minutes to prevent replay attacks
_SLACK_MAX_AGE_SECONDS = 300


class SlackConnector(WebhookConnector):
    """Connector for Slack Events API webhooks."""

    connector_type = "slack"

    def verify_signature(self, body: bytes, headers: dict[str, str]) -> bool:
        """
        Slack signs requests using v0=HMAC-SHA256(signing_secret, 'v0:{timestamp}:{body}').

        Validates both the signature and that the timestamp is within 5 minutes
        to prevent replay attacks.
        """
        if not _SIGNING_SECRET:
            logger.warning("SYNAPSE_SLACK_SIGNING_SECRET not set — rejecting request.")
            return False

        timestamp = headers.get("x-slack-request-timestamp", "")
        provided_sig = headers.get("x-slack-signature", "")

        # Reject if timestamp is stale
        try:
            if abs(time.time() - int(timestamp)) > _SLACK_MAX_AGE_SECONDS:
                logger.warning("Slack webhook timestamp too old — possible replay attack.")
                return False
        except (ValueError, TypeError):
            return False

        # Slack's base string format: v0:{timestamp}:{body}
        base_string = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
        expected = "v0=" + hmac.new(
            _SIGNING_SECRET.encode("utf-8"), base_string, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, provided_sig)

    def extract_event_id(self, payload: dict, headers: dict[str, str]) -> str:
        """Use event_ts + channel as a stable composite ID."""
        event = payload.get("event", {})
        ts = event.get("event_ts") or event.get("ts", "")
        channel = event.get("channel", "")
        return f"{channel}:{ts}"

    def extract_event_type(self, payload: dict, headers: dict[str, str]) -> str:
        """Return the Slack event type string."""
        return payload.get("event", {}).get("type", "unknown")

    def normalize(self, payload: dict, headers: dict[str, str]) -> str | None:
        """
        Convert a Slack event to a plain-text description.

        Returns None for event types that carry no useful knowledge
        (reactions, message edits, bot messages, etc.).
        """
        event = payload.get("event", {})
        event_type = event.get("type", "")

        # Only process plain messages
        if event_type != "message":
            logger.debug(f"Slack: skipping event type '{event_type}'")
            return None

        # Skip bot messages and automated posts
        if event.get("bot_id") or event.get("subtype") in (
            "bot_message", "message_changed", "message_deleted",
            "channel_join", "channel_leave",
        ):
            logger.debug("Slack: skipping bot/edit/join event")
            return None

        text = (event.get("text") or "").strip()
        if not text:
            return None

        user = event.get("user", "unknown_user")
        channel = event.get("channel", "unknown_channel")

        return f"In Slack channel #{channel}, user {user} said: {text}"


def is_url_verification(payload: dict) -> bool:
    """Return True if this is Slack's URL verification challenge."""
    return payload.get("type") == "url_verification"


def get_challenge(payload: dict) -> str:
    """Return the challenge value for URL verification response."""
    return payload.get("challenge", "")
