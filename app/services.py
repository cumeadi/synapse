"""
Synapse — Core service layer.

Contains all business logic for memory ingestion, hybrid search,
and graph neighborhood traversal.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import uuid
from typing import Any, Optional

import litellm
from sqlalchemy import and_, select, text, union_all
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Entity, Memory, Namespace, Relationship
from app.prompts import EXTRACTION_SYSTEM_PROMPT
from app.schemas import (
    ExtractedEntity,
    ExtractedRelationship,
    MemoryExtraction,
)
from app.services.policy import permitted_labels

logger = logging.getLogger("synapse.services")

LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ────────────────────────────────────────────────────────────────────
# Memory Ingestion (Background Task)
# ────────────────────────────────────────────────────────────────────
async def ingest_memory(
    namespace_id: uuid.UUID,
    memory_id: uuid.UUID,
    content: str,
    metadata: dict[str, Any],
) -> None:
    """
    Background task: generates embedding, stores memory, extracts
    entities/relationships via LLM, and persists them to the graph.
    """
    from app.database import async_session_factory

    async with async_session_factory() as db:
        try:
            # ── Step 1: Generate embedding ────────────────────────
            embedding = await _generate_embedding(content)

            # ── Step 2: Save Memory record ────────────────────────
            memory = Memory(
                id=memory_id,
                namespace_id=namespace_id,
                content=content,
                metadata_=metadata,
                embedding=embedding,
            )
            db.add(memory)
            await db.flush()
            logger.info(f"Memory {memory_id} saved with embedding.")

            # ── Step 3: Extract entities & relationships ──────────
            # Try Zero-LLM heuristic extraction first
            from app.services.heuristics import run_heuristics
            extraction = run_heuristics(content)
            
            if extraction:
                logger.info("Heuristic extraction succeeded (Zero-LLM).")
            else:
                logger.info("Heuristics failed, falling back to LLM extraction.")
                extraction = await _extract_knowledge(content)
                
            logger.info(
                f"Extracted {len(extraction.entities)} entities, "
                f"{len(extraction.relationships)} relationships."
            )

            # ── Step 4: Upsert entities (deduplicate) ─────────────
            entity_map: dict[str, Entity] = {}
            for ext_entity in extraction.entities:
                entity = await _upsert_entity(
                    db, namespace_id, ext_entity, memory_id
                )
                entity_map[ext_entity.name] = entity

            # ── Step 5: Insert relationships ──────────────────────
            for ext_rel in extraction.relationships:
                await _upsert_relationship(db, entity_map, ext_rel, memory_id)

            await db.commit()
            logger.info(f"Ingestion complete for memory {memory_id}.")

        except Exception as e:
            await db.rollback()
            logger.error(f"Ingestion failed for memory {memory_id}: {e}", exc_info=True)
            raise


async def _generate_embedding(text_content: str) -> list[float]:
    """Generate a 1536-d embedding via litellm."""
    response = await litellm.aembedding(
        model=EMBEDDING_MODEL,
        input=[text_content],
    )
    return response.data[0]["embedding"]


async def _extract_knowledge(content: str) -> MemoryExtraction:
    """Call LLM with structured output to extract entities & relationships."""
    try:
        response = await litellm.acompletion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format=MemoryExtraction,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        return MemoryExtraction.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"LLM extraction failed, returning empty: {e}")
        return MemoryExtraction(entities=[], relationships=[])


async def _upsert_entity(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    ext_entity: ExtractedEntity,
    context_id: uuid.UUID | None = None,
) -> Entity:
    """Insert entity if it doesn't exist, otherwise return existing."""
    from app.services.confidence import update_entity_confidence

    stmt = select(Entity).where(
        and_(
            Entity.namespace_id == namespace_id,
            Entity.name == ext_entity.name,
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.observation_count += 1
        await db.flush()
        await update_entity_confidence(db, existing.id)
        return existing

    entity = Entity(
        namespace_id=namespace_id,
        name=ext_entity.name,
        entity_type=ext_entity.entity_type,
        epistemic_state=ext_entity.epistemic_state,
        visibility_label="public",  # Phase 3 ABAC default
        observation_count=1,
        confidence=0.5,
    )
    db.add(entity)
    await db.flush()
    return entity


async def _upsert_relationship(
    db: AsyncSession,
    entity_map: dict[str, Entity],
    ext_rel: ExtractedRelationship,
    context_id: uuid.UUID | None = None,
) -> Optional[Relationship]:
    """Insert relationship if both entities exist and it's not a duplicate."""
    from app.services.confidence import (
        detect_and_flag_contradictions,
        update_relationship_confidence,
    )

    source = entity_map.get(ext_rel.source)
    target = entity_map.get(ext_rel.target)

    if not source or not target:
        logger.warning(
            f"Skipping relationship {ext_rel.source} -> {ext_rel.target}: "
            "entity not found in extraction."
        )
        return None

    # Check for existing relationship
    stmt = select(Relationship).where(
        and_(
            Relationship.source_entity_id == source.id,
            Relationship.target_entity_id == target.id,
            Relationship.relation_type == ext_rel.relation,
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        # Reinforce weight on repeated observations
        existing.weight = existing.weight + 1.0
        existing.last_reinforced_at = datetime.now(timezone.utc)

        # Phase 4: Diversity tracking
        # We use last_context_id on Relationship to detect new sources/memories.
        if context_id and existing.last_context_id != context_id:
            existing.source_diversity_count += 1
            existing.last_context_id = context_id
            
        await db.flush()
        
        await update_relationship_confidence(db, existing)
        await db.flush()
        return existing

    rel = Relationship(
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation_type=ext_rel.relation,
        epistemic_state=ext_rel.epistemic_state,
        weight=1.0,
        last_context_id=context_id,
        source_diversity_count=1,
        confidence=0.5,
        has_contradiction=False,
    )
    db.add(rel)
    await db.flush()
    
    await update_relationship_confidence(db, rel)
    await db.flush()
    return rel


# ────────────────────────────────────────────────────────────────────
# Hybrid Search (Semantic + Metadata)
# ────────────────────────────────────────────────────────────────────
async def hybrid_search(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    query: str,
    metadata_filter: Optional[dict[str, Any]] = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Semantic search using pgvector cosine distance with optional
    JSONB metadata containment filter.
    """
    query_embedding = await _generate_embedding(query)

    # Cosine similarity = 1 - cosine_distance
    score_expr = (1 - Memory.embedding.cosine_distance(query_embedding)).label("score")

    stmt = (
        select(Memory, score_expr)
        .where(Memory.namespace_id == namespace_id)
        .order_by(Memory.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )

    # Apply metadata filter if provided
    if metadata_filter:
        stmt = stmt.where(Memory.metadata_.contains(metadata_filter))

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "id": memory.id,
            "content": memory.content,
            "metadata": memory.metadata_,
            "score": round(float(score), 4),
            "created_at": memory.created_at,
        }
        for memory, score in rows
    ]


# ────────────────────────────────────────────────────────────────────
# Graph Search (Recursive Neighborhood Traversal)
# ────────────────────────────────────────────────────────────────────
async def graph_search(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    entity_name: str,
    depth: int = 1,
    threshold: float = 0.2,
    role: str = "internal",
    min_confidence: float = 0.0,
) -> dict:
    """
    Given an entity name, return its neighborhood up to `depth` hops
    using a recursive CTE for efficient graph traversal.
    """
    # Find the center entity
    center_stmt = select(Entity).where(
        and_(
            Entity.namespace_id == namespace_id,
            Entity.name == entity_name,
        )
    )
    result = await db.execute(center_stmt)
    center = result.scalar_one_or_none()

    if not center:
        return None

    if depth < 1:
        depth = 1
    if depth > 3:
        depth = 3

    # Resolve permitted visibility labels for the caller's role
    visible_labels = permitted_labels(role)

    # Recursive CTE to traverse graph edges up to `depth` hops
    # We only traverse relationships that have not decayed below the threshold
    # AND whose visibility label is within the caller's clearance
    cte_query = text("""
    WITH RECURSIVE traversal(entity_id, path_depth) AS (
        -- Anchor member: start at center entity (no label check — caller already resolved it)
        SELECT CAST(:center_id AS UUID) AS entity_id, 0 AS path_depth

        UNION

        -- Recursive member: traverse only permitted, non-decayed edges
        SELECT DISTINCT
            CASE
                WHEN r.source_entity_id = t.entity_id THEN r.target_entity_id
                ELSE r.source_entity_id
            END AS entity_id,
            t.path_depth + 1 AS path_depth
        FROM traversal t
        JOIN relationships r ON (r.source_entity_id = t.entity_id OR r.target_entity_id = t.entity_id)
        JOIN entities e ON e.id = CASE
            WHEN r.source_entity_id = t.entity_id THEN r.target_entity_id
            ELSE r.source_entity_id
        END
        WHERE t.path_depth < :max_depth
          AND (r.weight * EXP(-r.weight_decay_rate * (EXTRACT(EPOCH FROM (NOW() - r.last_reinforced_at)) / 86400.0))) >= :threshold
          AND r.visibility_label = ANY(:visible_labels)
          AND e.visibility_label = ANY(:visible_labels)
          AND r.confidence >= :min_confidence
          AND e.confidence >= :min_confidence
    )
    SELECT DISTINCT entity_id FROM traversal;
    """)

    result = await db.execute(
        cte_query,
        {
            "center_id": str(center.id),
            "max_depth": depth,
            "threshold": threshold,
            "visible_labels": visible_labels,
            "min_confidence": min_confidence,
        }
    )
    collected_entity_ids = {row[0] for row in result.all()}

    if not collected_entity_ids:
        collected_entity_ids = {center.id}

    # Fetch all entities in the neighborhood — filtered by visibility and confidence
    entities_stmt = select(Entity).where(
        and_(
            Entity.id.in_(collected_entity_ids),
            Entity.visibility_label.in_(visible_labels),
            Entity.confidence >= min_confidence,
        )
    )
    result = await db.execute(entities_stmt)
    entities = result.scalars().all()

    # Fetch relationships between collected entities — filtered by visibility + decay + confidence
    rels_stmt = (
        select(Relationship)
        .options(
            selectinload(Relationship.source_entity),
            selectinload(Relationship.target_entity),
        )
        .where(
            and_(
                Relationship.source_entity_id.in_(collected_entity_ids),
                Relationship.target_entity_id.in_(collected_entity_ids),
                Relationship.visibility_label.in_(visible_labels),
                Relationship.confidence >= min_confidence,
                (Relationship.weight * text(
                    "EXP(-relationships.weight_decay_rate * "
                    "(EXTRACT(EPOCH FROM (NOW() - relationships.last_reinforced_at)) / 86400.0))"
                )) >= threshold
            )
        )
    )
    result = await db.execute(rels_stmt)
    relationships = result.scalars().all()

    return {
        "center": center,
        "depth": depth,
        "entities": entities,
        "relationships": relationships,
    }
