"""
Synapse — Phase 4 Knowledge Confidence Scoring Engine.

Pure functions for computing confidence scores, detecting contradictions,
and propagating confidence updates through the knowledge graph.

Confidence model:
    raw_confidence  = 1 - exp(-k * source_diversity_count)
    temporal_factor = exp(-decay_rate * days_since_reinforcement)
    confidence      = raw_confidence * temporal_factor * contradiction_multiplier

Where k=0.5 gives:
    1 source  → ~39%
    3 sources → ~78%
    6 sources → ~95%

Entity confidence is the mean of its top-3 outgoing relationship confidences,
falling back to 0.5 if no relationships exist.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from app.models import Entity, Relationship

logger = logging.getLogger("synapse.services.confidence")

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

#: Saturation constant — controls how quickly confidence rises with more sources.
#: At k=0.5: 1 source≈39%, 3 sources≈78%, 6 sources≈95%
SATURATION_K: float = 0.5

#: Confidence penalty multiplier when a contradiction is detected.
CONTRADICTION_MULTIPLIER: float = 0.7

#: Minimum possible confidence score (floor).
CONFIDENCE_FLOOR: float = 0.01

#: Default confidence for brand-new observations with no history.
CONFIDENCE_DEFAULT: float = 0.5


# ────────────────────────────────────────────────────────────────────
# Pure computation functions (no DB, fully testable offline)
# ────────────────────────────────────────────────────────────────────

def compute_relationship_confidence(
    source_diversity_count: int,
    weight_decay_rate: float,
    days_since_reinforcement: float,
    has_contradiction: bool,
    k: float = SATURATION_K,
) -> float:
    """
    Compute a confidence score for a relationship using exponential saturation
    with temporal decay and optional contradiction penalty.

    Args:
        source_diversity_count: Number of distinct sources that observed this fact.
        weight_decay_rate: The relationship's decay rate (from Phase 2).
        days_since_reinforcement: Days elapsed since last reinforcement.
        has_contradiction: Whether a contradicting relationship exists.
        k: Saturation constant (default 0.5).

    Returns:
        Float in [CONFIDENCE_FLOOR, 1.0].
    """
    # Exponential saturation: saturates toward 1.0 as sources increase
    raw = 1.0 - math.exp(-k * max(source_diversity_count, 0))

    # Temporal decay: reuse Phase 2 decay mechanics
    temporal = math.exp(-weight_decay_rate * max(days_since_reinforcement, 0.0))

    score = raw * temporal

    # Contradiction penalty
    if has_contradiction:
        score *= CONTRADICTION_MULTIPLIER

    return max(CONFIDENCE_FLOOR, min(1.0, score))


def compute_entity_confidence_from_scores(relationship_confidences: list[float]) -> float:
    """
    Compute entity confidence from its relationship confidence scores.

    Uses the mean of the top-3 outgoing relationship confidences to avoid
    a single high-confidence edge inflating a poorly-evidenced entity.

    Args:
        relationship_confidences: List of confidence scores from outgoing relationships.

    Returns:
        Float in [CONFIDENCE_FLOOR, 1.0], or CONFIDENCE_DEFAULT if no relationships.
    """
    if not relationship_confidences:
        return CONFIDENCE_DEFAULT

    top = sorted(relationship_confidences, reverse=True)[:3]
    return max(CONFIDENCE_FLOOR, min(1.0, sum(top) / len(top)))


# ────────────────────────────────────────────────────────────────────
# DB-backed helpers (called after upsert operations)
# ────────────────────────────────────────────────────────────────────

async def detect_and_flag_contradictions(
    db: AsyncSession,
    source_entity_id,
    target_entity_id,
) -> bool:
    """
    Detect if ≥2 different relation_types exist between source and target.
    If yes, mark all such relationships has_contradiction=True.

    This is conservative: same-pair, different-type = contradiction signal.

    Args:
        db: Active async session.
        source_entity_id: UUID of the source entity.
        target_entity_id: UUID of the target entity.

    Returns:
        True if a contradiction was detected (and flagged).
    """
    from app.models import Relationship

    stmt = select(Relationship).where(
        and_(
            Relationship.source_entity_id == source_entity_id,
            Relationship.target_entity_id == target_entity_id,
        )
    )
    result = await db.execute(stmt)
    rels = result.scalars().all()

    relation_types = {r.relation_type for r in rels}
    has_contradiction = len(relation_types) >= 2

    for rel in rels:
        if rel.epistemic_state == "REFLEX" and has_contradiction:
            rel.epistemic_state = "FACT"
            rel.confidence = CONFIDENCE_FLOOR
            rel.has_contradiction = True
            logger.info(f"Reflex relationship {rel.id} contradicted! Reverting to FACT with confidence floor.")
        elif rel.has_contradiction != has_contradiction:
            rel.has_contradiction = has_contradiction
            await update_relationship_confidence(db, rel)

    return has_contradiction


async def update_relationship_confidence(
    db: AsyncSession,
    relationship,
) -> None:
    """
    Recompute and persist the confidence score for a relationship.

    Called immediately after any upsert or contradiction flag change.

    Args:
        db: Active async session.
        relationship: The Relationship ORM object to update.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    last = relationship.last_reinforced_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    days = (now - last).total_seconds() / 86400.0

    relationship.confidence = compute_relationship_confidence(
        source_diversity_count=relationship.source_diversity_count,
        weight_decay_rate=relationship.weight_decay_rate,
        days_since_reinforcement=days,
        has_contradiction=relationship.has_contradiction,
    )


async def update_entity_confidence(
    db: AsyncSession,
    entity_id,
) -> None:
    """
    Recompute and persist the confidence score for an entity.

    Pulls the current confidence of its outgoing relationships, uses
    the top-3 mean, and persists back to the entity row.

    Args:
        db: Active async session.
        entity_id: UUID of the entity to update.
    """
    from app.models import Entity, Relationship

    rels_stmt = select(Relationship.confidence).where(
        Relationship.source_entity_id == entity_id
    ).order_by(Relationship.confidence.desc()).limit(3)

    result = await db.execute(rels_stmt)
    scores = [row[0] for row in result.all()]

    new_confidence = compute_entity_confidence_from_scores(scores)

    entity = await db.get(Entity, entity_id)
    if entity:
        entity.confidence = new_confidence
