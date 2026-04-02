"""
Synapse — Core service layer.

Contains all business logic for memory ingestion, hybrid search,
and graph neighborhood traversal.
"""

from __future__ import annotations

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

            # ── Step 3: Extract entities & relationships via LLM ──
            extraction = await _extract_knowledge(content)
            logger.info(
                f"Extracted {len(extraction.entities)} entities, "
                f"{len(extraction.relationships)} relationships."
            )

            # ── Step 4: Upsert entities (deduplicate) ─────────────
            entity_map: dict[str, Entity] = {}
            for ext_entity in extraction.entities:
                entity = await _upsert_entity(
                    db, namespace_id, ext_entity
                )
                entity_map[ext_entity.name] = entity

            # ── Step 5: Insert relationships ──────────────────────
            for ext_rel in extraction.relationships:
                await _upsert_relationship(db, entity_map, ext_rel)

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
) -> Entity:
    """Insert entity if it doesn't exist, otherwise return existing."""
    stmt = select(Entity).where(
        and_(
            Entity.namespace_id == namespace_id,
            Entity.name == ext_entity.name,
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        return existing

    entity = Entity(
        namespace_id=namespace_id,
        name=ext_entity.name,
        entity_type=ext_entity.entity_type,
    )
    db.add(entity)
    await db.flush()
    return entity


async def _upsert_relationship(
    db: AsyncSession,
    entity_map: dict[str, Entity],
    ext_rel: ExtractedRelationship,
) -> Optional[Relationship]:
    """Insert relationship if both entities exist and it's not a duplicate."""
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
        return existing

    rel = Relationship(
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation_type=ext_rel.relation,
        weight=1.0,
    )
    db.add(rel)
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

    # Recursive CTE to traverse graph edges up to `depth` hops
    # We collect all entity IDs reachable within the depth
    collected_entity_ids = {center.id}
    current_frontier = {center.id}

    for _ in range(depth):
        if not current_frontier:
            break

        # Find all entities connected to the current frontier
        outgoing = select(Relationship.target_entity_id).where(
            Relationship.source_entity_id.in_(current_frontier)
        )
        incoming = select(Relationship.source_entity_id).where(
            Relationship.target_entity_id.in_(current_frontier)
        )

        combined = union_all(outgoing, incoming)
        result = await db.execute(combined)
        neighbor_ids = {row[0] for row in result.all()}

        new_ids = neighbor_ids - collected_entity_ids
        collected_entity_ids |= new_ids
        current_frontier = new_ids

    # Fetch all entities in the neighborhood
    entities_stmt = select(Entity).where(Entity.id.in_(collected_entity_ids))
    result = await db.execute(entities_stmt)
    entities = result.scalars().all()

    # Fetch all relationships between collected entities
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
