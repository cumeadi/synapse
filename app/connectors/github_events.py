"""
Synapse — GitHub Events Connector.

Handles passive GitHub webhook events (distinct from the existing
github.py repo importer which does a full architectural import).

Supported events:
  - push     — commit activity on branches
  - pull_request — PR lifecycle (opened, merged, closed, review requested)
  - issues   — issue lifecycle (opened, closed, assigned, commented)

HMAC-SHA256 signature verification via X-Hub-Signature-256 using
the X-GitHub-Delivery header as the stable idempotency key.

Environment variables:
  SYNAPSE_GITHUB_WEBHOOK_SECRET  — GitHub webhook secret (required)
  SYNAPSE_GITHUB_VISIBILITY      — ABAC label for ingested data (default: internal)
"""

from __future__ import annotations

import logging
import os

from app.connectors.webhook_base import WebhookConnector, verify_hmac_sha256

logger = logging.getLogger("synapse.connectors.github_events")

_SECRET = os.getenv("SYNAPSE_GITHUB_WEBHOOK_SECRET", "")
_VISIBILITY = os.getenv("SYNAPSE_GITHUB_VISIBILITY", "internal")

# GitHub event types we actively normalize; others are silently skipped
_HANDLED_EVENTS = {"push", "pull_request", "issues"}


class GitHubEventsConnector(WebhookConnector):
    """Connector for passive GitHub webhook events."""

    connector_type = "github"

    def verify_signature(self, body: bytes, headers: dict[str, str]) -> bool:
        """Verify GitHub's X-Hub-Signature-256 header."""
        if not _SECRET:
            logger.warning("SYNAPSE_GITHUB_WEBHOOK_SECRET not set — rejecting request.")
            return False

        provided_sig = headers.get("x-hub-signature-256", "")
        return verify_hmac_sha256(_SECRET, body, provided_sig)

    def extract_event_id(self, payload: dict, headers: dict[str, str]) -> str:
        """Use GitHub's X-GitHub-Delivery UUID as the stable event ID."""
        return headers.get("x-github-delivery", "") or str(payload.get("delivery", ""))

    def extract_event_type(self, payload: dict, headers: dict[str, str]) -> str:
        """Return the X-GitHub-Event header value."""
        return headers.get("x-github-event", "unknown")

    def normalize(self, payload: dict, headers: dict[str, str]) -> str | None:
        """
        Convert a GitHub event payload to a plain-text description.

        Returns None for unhandled or low-value event types.
        """
        event_type = headers.get("x-github-event", "")

        if event_type not in _HANDLED_EVENTS:
            logger.debug(f"GitHub: skipping unhandled event '{event_type}'")
            return None

        repo = payload.get("repository", {}).get("full_name", "unknown/repo")

        # ── Push events ──────────────────────────────────────────────
        if event_type == "push":
            return self._normalize_push(payload, repo)

        # ── Pull request events ──────────────────────────────────────
        if event_type == "pull_request":
            return self._normalize_pull_request(payload, repo)

        # ── Issue events ─────────────────────────────────────────────
        if event_type == "issues":
            return self._normalize_issue(payload, repo)

        return None

    # ── Private normalizers ──────────────────────────────────────────

    def _normalize_push(self, payload: dict, repo: str) -> str | None:
        """Format: '{pusher} pushed {n} commit(s) to {repo}/{branch}: '{last_msg}'"."""
        pusher = payload.get("pusher", {}).get("name", "unknown")
        commits = payload.get("commits", [])
        ref = payload.get("ref", "")

        # Only care about branch pushes (not tags)
        if not ref.startswith("refs/heads/"):
            return None
        branch = ref.removeprefix("refs/heads/")

        if not commits:
            return None

        n = len(commits)
        last_msg = (commits[-1].get("message") or "").split("\n")[0].strip()
        if len(last_msg) > 200:
            last_msg = last_msg[:200] + "…"

        return (
            f"{pusher} pushed {n} commit{'s' if n != 1 else ''} "
            f"to {repo}/{branch}: '{last_msg}'"
        )

    def _normalize_pull_request(self, payload: dict, repo: str) -> str | None:
        """Format: '{user} {action} PR #{n}: '{title}' in {repo} ({head}→{base})'."""
        action = payload.get("action", "")
        # Focus on meaningful lifecycle events only
        if action not in ("opened", "closed", "reopened", "review_requested",
                          "ready_for_review", "converted_to_draft"):
            return None

        pr = payload.get("pull_request", {})
        user = (
            pr.get("user", {}).get("login")
            or payload.get("sender", {}).get("login", "unknown")
        )
        number = pr.get("number", "?")
        title = (pr.get("title") or "").strip()
        if len(title) > 200:
            title = title[:200] + "…"
        head = pr.get("head", {}).get("ref", "?")
        base = pr.get("base", {}).get("ref", "?")
        merged = pr.get("merged", False)

        # Use "merged" instead of "closed" when appropriate
        display_action = "merged" if (action == "closed" and merged) else action

        return (
            f"{user} {display_action} pull request #{number}: "
            f"'{title}' in {repo} ({head} → {base})"
        )

    def _normalize_issue(self, payload: dict, repo: str) -> str | None:
        """Format: '{user} {action} issue #{n}: '{title}' in {repo}'."""
        action = payload.get("action", "")
        if action not in ("opened", "closed", "reopened", "assigned", "labeled"):
            return None

        issue = payload.get("issue", {})
        user = payload.get("sender", {}).get("login", "unknown")
        number = issue.get("number", "?")
        title = (issue.get("title") or "").strip()
        if len(title) > 200:
            title = title[:200] + "…"

        extra = ""
        if action == "assigned":
            assignee = (issue.get("assignee") or {}).get("login", "someone")
            extra = f" to {assignee}"
        elif action == "labeled":
            label = payload.get("label", {}).get("name", "")
            extra = f" with label '{label}'"

        return f"{user} {action} issue #{number}{extra}: '{title}' in {repo}"
