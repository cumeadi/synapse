"""
Synapse — Jira Connector.

Handles Jira webhook events:
  - HMAC-SHA256 signature verification via X-Hub-Signature-256
  - Normalises issue created/updated and comment events to plain text
  - Skips system-generated events with no actionable content

Environment variables:
  SYNAPSE_JIRA_SECRET      — Jira webhook secret (required)
  SYNAPSE_JIRA_VISIBILITY  — ABAC label for ingested data (default: internal)
"""

from __future__ import annotations

import logging
import os

from app.connectors.webhook_base import WebhookConnector, verify_hmac_sha256

logger = logging.getLogger("synapse.connectors.jira")

_SECRET = os.getenv("SYNAPSE_JIRA_SECRET", "")
_VISIBILITY = os.getenv("SYNAPSE_JIRA_VISIBILITY", "internal")

# Jira event types we care about — anything else is skipped
_HANDLED_EVENTS = {
    "jira:issue_created",
    "jira:issue_updated",
    "comment_created",
    "comment_updated",
}


class JiraConnector(WebhookConnector):
    """Connector for Jira webhook events."""

    connector_type = "jira"

    def verify_signature(self, body: bytes, headers: dict[str, str]) -> bool:
        """Verify Jira's X-Hub-Signature-256 header."""
        if not _SECRET:
            logger.warning("SYNAPSE_JIRA_SECRET not set — rejecting request.")
            return False

        provided_sig = headers.get("x-hub-signature-256", "")
        return verify_hmac_sha256(_SECRET, body, provided_sig)

    def extract_event_id(self, payload: dict, headers: dict[str, str]) -> str:
        """
        Build a stable composite ID from event type + issue ID + timestamp.
        Jira doesn't provide a single delivery ID like GitHub, so we compose one.
        """
        event_type = payload.get("webhookEvent", "unknown")
        issue_id = payload.get("issue", {}).get("id", "")
        timestamp = str(payload.get("timestamp", ""))
        comment_id = payload.get("comment", {}).get("id", "")
        return f"{event_type}:{issue_id}:{comment_id}:{timestamp}"

    def extract_event_type(self, payload: dict, headers: dict[str, str]) -> str:
        """Return the Jira webhookEvent string."""
        return payload.get("webhookEvent", "unknown")

    def normalize(self, payload: dict, headers: dict[str, str]) -> str | None:
        """
        Convert a Jira event to a plain-text description for LLM extraction.

        Formats:
          Issue: "{user} {created|updated} Jira issue {key}: '{summary}'
                  (type={type}, status={status}, assignee={assignee})"
          Comment: "{user} commented on {key}: '{text}'"
        """
        event_type = payload.get("webhookEvent", "")

        if event_type not in _HANDLED_EVENTS:
            logger.debug(f"Jira: skipping unhandled event type '{event_type}'")
            return None

        user = (
            payload.get("user", {}).get("displayName")
            or payload.get("user", {}).get("name")
            or "unknown user"
        )

        # ── Issue events ─────────────────────────────────────────────
        if event_type in ("jira:issue_created", "jira:issue_updated"):
            issue = payload.get("issue", {})
            fields = issue.get("fields", {})
            key = issue.get("key", "UNKNOWN")
            summary = (fields.get("summary") or "").strip()
            issue_type = (
                fields.get("issuetype", {}).get("name") or "issue"
            )
            status = (
                fields.get("status", {}).get("name") or "unknown"
            )
            assignee_obj = fields.get("assignee") or {}
            assignee = (
                assignee_obj.get("displayName")
                or assignee_obj.get("name")
                or "unassigned"
            )

            action = "created" if event_type == "jira:issue_created" else "updated"
            text = (
                f"{user} {action} Jira issue {key}: '{summary}' "
                f"(type={issue_type}, status={status}, assignee={assignee})"
            )
            return text

        # ── Comment events ────────────────────────────────────────────
        if event_type in ("comment_created", "comment_updated"):
            issue = payload.get("issue", {})
            key = issue.get("key", "UNKNOWN")
            comment_body = (
                payload.get("comment", {}).get("body") or ""
            ).strip()
            # Truncate very long comments for LLM efficiency
            if len(comment_body) > 500:
                comment_body = comment_body[:500] + "…"

            return f"{user} commented on Jira issue {key}: '{comment_body}'"

        return None
