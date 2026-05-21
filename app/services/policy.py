"""
Synapse — Phase 3 ABAC Policy Evaluation.

Defines the clearance-ceiling label hierarchy and provides pure functions
for evaluating whether a caller's role permits access to a specific visibility label.

Label Hierarchy (ascending restriction):
    public < internal < confidential < restricted

A role at level N grants access to all labels at level <= N.
These functions are pure (no DB calls) and fully testable offline.
"""

from __future__ import annotations

from typing import Sequence

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

# Ordered from least to most restricted. Position in this list is the clearance level.
LABEL_ORDER: list[str] = ["public", "internal", "confidential", "restricted"]

# Default role granted to callers with no API key (dev mode / anonymous MCP)
DEFAULT_ROLE_DEV = "internal"

# Default role for MCP callers that don't supply a role parameter
DEFAULT_ROLE_MCP = "public"


# ────────────────────────────────────────────────────────────────────
# Core Policy Functions
# ────────────────────────────────────────────────────────────────────

def clearance_level(label: str) -> int:
    """
    Return the numeric clearance level for a label.

    Higher number = more restricted. Unknown labels default to 0 (public).

    Args:
        label: A visibility or role label string.

    Returns:
        Integer clearance level (0 = public, 3 = restricted).
    """
    try:
        return LABEL_ORDER.index(label.lower())
    except ValueError:
        return 0  # Treat unknown labels as public


def can_access(role: str, visibility_label: str) -> bool:
    """
    Determine if a caller role has clearance to read a given visibility label.

    Implements a clearance-ceiling model: a role grants access to all labels
    at or below its own clearance level.

    Args:
        role: The caller's ABAC role (e.g. "internal").
        visibility_label: The label on the resource being accessed (e.g. "confidential").

    Returns:
        True if the caller may access the resource, False otherwise.

    Examples:
        >>> can_access("internal", "public")    # True — public is below internal
        >>> can_access("internal", "internal")  # True — exact match
        >>> can_access("internal", "confidential")  # False — above ceiling
        >>> can_access("restricted", "restricted")  # True — ceiling reached
    """
    return clearance_level(role) >= clearance_level(visibility_label)


def permitted_labels(role: str) -> list[str]:
    """
    Return all visibility labels a given role is permitted to see.

    Used to build SQL IN clauses for database-level filtering.

    Args:
        role: The caller's ABAC role.

    Returns:
        List of permitted visibility label strings.

    Example:
        >>> permitted_labels("internal")
        ["public", "internal"]
    """
    ceiling = clearance_level(role)
    return LABEL_ORDER[: ceiling + 1]


def validate_label(label: str) -> str:
    """
    Validate and normalise a visibility label string.

    Raises ValueError if label is not in LABEL_ORDER.

    Args:
        label: Raw label string from user input.

    Returns:
        Normalised (lowercased) label string.

    Raises:
        ValueError: If the label is not a valid ABAC label.
    """
    normalised = label.lower()
    if normalised not in LABEL_ORDER:
        raise ValueError(
            f"Invalid visibility label '{label}'. "
            f"Must be one of: {', '.join(LABEL_ORDER)}"
        )
    return normalised


def resolve_caller_role(api_key_obj, master_key_used: bool = False) -> str:
    """
    Resolve the effective ABAC role for a request.

    Args:
        api_key_obj: The ApiKey ORM object from verify_api_key, or None.
        master_key_used: True if the caller authenticated with the master key.

    Returns:
        Role string to use for this request.
    """
    if master_key_used or api_key_obj is None:
        # Master key and dev-mode callers get full internal access.
        # In dev mode (no auth configured) we default to internal so
        # existing data without explicit labels stays fully visible.
        return DEFAULT_ROLE_DEV

    return getattr(api_key_obj, "role", DEFAULT_ROLE_DEV)
