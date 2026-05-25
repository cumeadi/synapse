"""
Synapse Studio Service.

Handles backend logic for the visualization UI, including
formatting the graph for vis-network and edge tracing.
"""

import uuid
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional, Dict, Any

from app.models import Entity, Relationship, Namespace, Memory
from app.services.core import _generate_embedding
from app.services.policy import permitted_labels

async def get_full_graph(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    role: str = "internal",
    min_confidence: float = 0.0,
) -> dict[str, list]:
    """Retrieve the entire graph formatted for VisNetwork, filtered by caller's ABAC role."""
    visible_labels = permitted_labels(role)

    # Fetch all entities the caller is permitted to see
    ent_stmt = select(Entity).where(
        and_(
            Entity.namespace_id == namespace_id,
            Entity.visibility_label.in_(visible_labels),
            Entity.confidence >= min_confidence,
        )
    )
    ent_result = await db.execute(ent_stmt)
    entities = ent_result.scalars().all()

    # Fetch all permitted relationships where the source entity belongs to this namespace
    rel_stmt = (
        select(Relationship)
        .join(Entity, Relationship.source_entity_id == Entity.id)
        .where(
            and_(
                Entity.namespace_id == namespace_id,
                Relationship.visibility_label.in_(visible_labels),
                Relationship.confidence >= min_confidence,
            )
        )
    )
    rel_result = await db.execute(rel_stmt)
    relationships = rel_result.scalars().all()

    nodes = []
    edges = []

    for ent in entities:
        nodes.append({
            "id": str(ent.id),
            "label": ent.name,
            "group": ent.entity_type,
            "confidence": ent.confidence,
            "epistemic_state": ent.epistemic_state,
        })

    for rel in relationships:
        edges.append({
            "id": str(rel.id),
            "from": str(rel.source_entity_id),
            "to": str(rel.target_entity_id),
            "label": rel.relation_type,
            "weight": rel.decayed_weight,
            "confidence": rel.confidence,
            "has_contradiction": rel.has_contradiction,
            "epistemic_state": rel.epistemic_state,
            "trigger_condition": rel.trigger_condition,
            "executable_payload": rel.executable_payload,
            "status": rel.status,
        })

    return {"nodes": nodes, "edges": edges}

async def delete_entity(db: AsyncSession, namespace_id: uuid.UUID, entity_id: uuid.UUID) -> bool:
    """Deletes an entity. Relationships cascade automatically due to DB foreign keys."""
    stmt = select(Entity).where(and_(Entity.id == entity_id, Entity.namespace_id == namespace_id))
    result = await db.execute(stmt)
    entity = result.scalar_one_or_none()
    
    if not entity:
        return False
        
    await db.delete(entity)
    await db.commit()
    return True

async def delete_relationship(db: AsyncSession, namespace_id: uuid.UUID, relationship_id: uuid.UUID) -> bool:
    """Deletes a relationship."""
    # Ensure it belongs to the namespace
    stmt = (
        select(Relationship)
        .join(Entity, Relationship.source_entity_id == Entity.id)
        .where(and_(Relationship.id == relationship_id, Entity.namespace_id == namespace_id))
    )
    result = await db.execute(stmt)
    rel = result.scalar_one_or_none()
    
    if not rel:
        return False
        
    await db.delete(rel)
    await db.commit()
    return True

async def trace_relationship(db: AsyncSession, namespace_id: uuid.UUID, relationship_id: uuid.UUID) -> dict[str, Any]:
    """
    Since we don't store memory_id directly on Relationship, we try to trace it by:
    1. Getting the relationship and its source/target entities.
    2. Embedding the relationship context.
    3. Finding the most semantically similar Memory in the namespace that contains the entity names.
    """
    stmt = (
        select(Relationship)
        .options(
            selectinload(Relationship.source_entity),
            selectinload(Relationship.target_entity)
        )
        .join(Entity, Relationship.source_entity_id == Entity.id)
        .where(and_(Relationship.id == relationship_id, Entity.namespace_id == namespace_id))
    )
    result = await db.execute(stmt)
    rel = result.scalar_one_or_none()
    
    if not rel:
        return {"relationship_id": relationship_id, "matched_memory_id": None, "content": None, "score": 0.0, "explanation": "Relationship not found."}
        
    source_name = rel.source_entity.name
    target_name = rel.target_entity.name
    rel_type = rel.relation_type
    
    # Build a query targeting this connection
    query = f"{source_name} {rel_type} {target_name}"
    query_embedding = await _generate_embedding(query)
    
    # We want memories that mention BOTH entities if possible, or at least one, and are semantically close
    # Use the same cosine distance logic as hybrid_search
    score_expr = (1 - Memory.embedding.cosine_distance(query_embedding)).label("score")
    
    # Add a strict ILIKE or text search requirement for at least the source or target to reduce false positives
    mem_stmt = (
        select(Memory, score_expr)
        .where(
            and_(
                Memory.namespace_id == namespace_id,
                # Simple containment check for the names (case-insensitive approximation)
                Memory.content.ilike(f"%{source_name}%")
            )
        )
        .order_by(Memory.embedding.cosine_distance(query_embedding))
        .limit(1)
    )
    
    mem_result = await db.execute(mem_stmt)
    row = mem_result.first()
    
    if row:
        memory, score = row
        return {
            "relationship_id": relationship_id,
            "matched_memory_id": memory.id,
            "content": memory.content,
            "score": round(float(score), 4),
            "explanation": f"Found via semantic match ({rel_type}) and text inclusion of source entity."
        }
        
    # Fallback if both exact match fails: just semantic match
    mem_stmt_fallback = (
        select(Memory, score_expr)
        .where(Memory.namespace_id == namespace_id)
        .order_by(Memory.embedding.cosine_distance(query_embedding))
        .limit(1)
    )
    fb_result = await db.execute(mem_stmt_fallback)
    row_fb = fb_result.first()
    
    if row_fb:
         memory, score = row_fb
         return {
            "relationship_id": relationship_id,
            "matched_memory_id": memory.id,
            "content": memory.content,
            "score": round(float(score), 4),
            "explanation": f"Best semantic match found, though entity names might not be perfectly verbatim in text."
        }
         
    return {"relationship_id": relationship_id, "matched_memory_id": None, "content": None, "score": 0.0, "explanation": "No matching memories found in semantic index."}


async def update_reflex_status(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    relationship_id: uuid.UUID,
    status: str,
) -> bool:
    """Updates the status of a reflex relationship (PROPOSED, ACTIVE, PAUSED)."""
    stmt = (
        select(Relationship)
        .join(Entity, Relationship.source_entity_id == Entity.id)
        .where(
            and_(
                Relationship.id == relationship_id,
                Entity.namespace_id == namespace_id,
            )
        )
    )
    result = await db.execute(stmt)
    rel = result.scalar_one_or_none()
    
    if not rel:
        return False
        
    rel.status = status
    await db.commit()
    return True
