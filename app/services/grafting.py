"""
Synapse — Auto-Grafting Engine.

When a new KnowledgeSource is imported, this engine connects the new
domain entities to the user's existing personal graph by identifying
logical bridge relationships via LLM analysis.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import litellm
from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Entity, KnowledgeSource, Relationship
from app.prompts import GRAFTING_SYSTEM_PROMPT
from app.schemas import ExtractedRelationship, GraftingExtraction

logger = logging.getLogger("synapse.services.grafting")

LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")


async def run_grafting_cycle(
    namespace_id: uuid.UUID,
    new_source_id: uuid.UUID,
) -> None:
    """
    Background task: analyze the user's personal entities and the newly
    imported domain entities, then create bridge relationships between them.
    """
    from app.database import async_session_factory

    async with async_session_factory() as db:
        try:
            # Update source status to grafting
            source = await db.get(KnowledgeSource, new_source_id)
            if source:
                source.status = "grafting"
                await db.flush()

            # ── Step 1: Fetch user's core personal entities ───────
            personal_entities = await _fetch_personal_entities(db, namespace_id)
            if not personal_entities:
                logger.info("No personal entities found — skipping grafting.")
                if source:
                    source.status = "ready"
                    await db.commit()
                return

            # ── Step 2: Fetch top entities from new source ────────
            domain_entities = await _fetch_source_entities(
                db, namespace_id, new_source_id
            )
            if not domain_entities:
                logger.info("No domain entities found — skipping grafting.")
                if source:
                    source.status = "ready"
                    await db.commit()
                return

            logger.info(
                f"Grafting: {len(personal_entities)} personal entities "
                f"× {len(domain_entities)} domain entities"
            )

            # ── Step 3: Ask LLM to find bridge connections ────────
            bridges = await _identify_bridges(
                personal_entities, domain_entities, source.name if source else "Unknown"
            )
            logger.info(f"LLM identified {len(bridges)} bridge relationships.")

            # ── Step 4: Persist bridge relationships ──────────────
            # Build a combined entity lookup
            all_entity_names = {}
            for e in personal_entities + domain_entities:
                all_entity_names[e.name] = e

            created_count = 0
            for bridge in bridges:
                result = await _create_bridge_relationship(
                    db, all_entity_names, bridge, new_source_id
                )
                if result:
                    created_count += 1

            # Update source status
            if source:
                source.status = "ready"
                source.last_synced_at = datetime.now(timezone.utc)

            await db.commit()
            logger.info(
                f"Grafting complete: {created_count} bridge relationships created."
            )

        except Exception as e:
            await db.rollback()
            # Try to keep source status accurate
            try:
                async with async_session_factory() as db2:
                    source = await db2.get(KnowledgeSource, new_source_id)
                    if source and source.status == "grafting":
                        source.status = "ready"  # grafting failed but ingestion was fine
                        await db2.commit()
            except Exception:
                pass
            logger.error(f"Grafting failed: {e}", exc_info=True)


async def _fetch_personal_entities(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    limit: int = 30,
) -> list[Entity]:
    """Fetch the user's personal entities (source_id is NULL)."""
    stmt = (
        select(Entity)
        .where(
            and_(
                Entity.namespace_id == namespace_id,
                Entity.source_id.is_(None),
            )
        )
        .order_by(Entity.name)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _fetch_source_entities(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    source_id: uuid.UUID,
    limit: int = 30,
) -> list[Entity]:
    """
    Fetch the highest-connected entities from a KnowledgeSource,
    ranked by total relationship count.
    """
    # Count relationships per entity (as source or target)
    outgoing = (
        select(
            Relationship.source_entity_id.label("entity_id"),
            func.count().label("cnt"),
        )
        .group_by(Relationship.source_entity_id)
        .subquery()
    )
    incoming = (
        select(
            Relationship.target_entity_id.label("entity_id"),
            func.count().label("cnt"),
        )
        .group_by(Relationship.target_entity_id)
        .subquery()
    )

    stmt = (
        select(Entity)
        .where(
            and_(
                Entity.namespace_id == namespace_id,
                Entity.source_id == source_id,
            )
        )
        .outerjoin(outgoing, Entity.id == outgoing.c.entity_id)
        .outerjoin(incoming, Entity.id == incoming.c.entity_id)
        .order_by(
            (func.coalesce(outgoing.c.cnt, 0) + func.coalesce(incoming.c.cnt, 0)).desc()
        )
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


def _format_entity_list(entities: list[Entity]) -> str:
    """Format entities into a readable list for the LLM."""
    lines = []
    for e in entities:
        lines.append(f"  - {e.name} ({e.entity_type})")
    return "\n".join(lines)


async def _identify_bridges(
    personal_entities: list[Entity],
    domain_entities: list[Entity],
    source_name: str,
) -> list[ExtractedRelationship]:
    """
    Use LLM to identify logical connections between personal
    and domain entities.
    """
    personal_list = _format_entity_list(personal_entities)
    domain_list = _format_entity_list(domain_entities)

    user_message = (
        f"## User's Personal Knowledge Graph Entities:\n{personal_list}\n\n"
        f"## Newly Imported Domain Entities (from '{source_name}'):\n{domain_list}\n\n"
        f"Identify logical bridge connections between these two sets. "
        f"Only create connections that are factually reasonable."
    )

    try:
        response = await litellm.acompletion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": GRAFTING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=GraftingExtraction,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        extraction = GraftingExtraction.model_validate_json(raw)
        return extraction.relationships
    except Exception as e:
        logger.warning(f"Grafting LLM call failed: {e}")
        return []


async def _create_bridge_relationship(
    db: AsyncSession,
    entity_map: dict[str, Entity],
    bridge: ExtractedRelationship,
    source_id: uuid.UUID,
) -> Optional[Relationship]:
    """Create a bridge relationship if both entities exist and it's not a duplicate."""
    source_ent = entity_map.get(bridge.source)
    target_ent = entity_map.get(bridge.target)

    if not source_ent or not target_ent:
        logger.debug(
            f"Skipping bridge {bridge.source} -> {bridge.target}: entity not found"
        )
        return None

    # Check for existing
    stmt = select(Relationship).where(
        and_(
            Relationship.source_entity_id == source_ent.id,
            Relationship.target_entity_id == target_ent.id,
            Relationship.relation_type == bridge.relation,
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.weight = existing.weight + 0.5  # Lighter weight for inferred bridges
        return existing

    rel = Relationship(
        source_entity_id=source_ent.id,
        target_entity_id=target_ent.id,
        relation_type=bridge.relation,
        weight=0.5,  # Inferred connections start with lower weight
        source_id=source_id,
    )
    db.add(rel)
    await db.flush()
    return rel
