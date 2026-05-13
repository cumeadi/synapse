"""
Synapse — Sleep Cycle / Memory Consolidation Engine.

Runs periodically to prune and optimize the knowledge graph by merging
synonymous entities and resolving temporal contradictions.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import litellm
from sqlalchemy import and_, select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Entity, Namespace, Relationship, Memory
from app.prompts import DISAMBIGUATION_SYSTEM_PROMPT, CONTRADICTION_SYSTEM_PROMPT
from app.schemas import DisambiguationResult, ContradictionResult

logger = logging.getLogger("synapse.services.sleep")

LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

async def run_sleep_cycle(namespace_id: uuid.UUID) -> None:
    """
    Main orchestrator for the sleep cycle.
    Runs Entity Disambiguation followed by Contradiction Pruning.
    """
    from app.database import async_session_factory

    logger.info(f"Starting sleep cycle for namespace: {namespace_id}")
    
    async with async_session_factory() as db:
        try:
            await _run_entity_disambiguation(db, namespace_id)
            await _run_contradiction_pruning(db, namespace_id)
            await db.commit()
            logger.info(f"Completed sleep cycle for namespace: {namespace_id}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Sleep cycle failed for namespace {namespace_id}: {e}", exc_info=True)


async def _run_entity_disambiguation(db: AsyncSession, namespace_id: uuid.UUID) -> None:
    """Find synonymous entities, merge their relationships, and delete duplicates."""
    logger.info("Running entity disambiguation...")
    
    # 1. Fetch all entities in the namespace
    stmt = select(Entity).where(Entity.namespace_id == namespace_id).order_by(Entity.name)
    result = await db.execute(stmt)
    entities = list(result.scalars().all())
    
    if len(entities) < 2:
        return
        
    entity_map = {e.name: e for e in entities}
    entity_names = list(entity_map.keys())
    
    # Format list for LLM
    entity_list_str = "\n".join([f"- {name}" for name in entity_names])
    
    try:
        response = await litellm.acompletion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": DISAMBIGUATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Entities to analyze:\n{entity_list_str}"},
            ],
            response_format=DisambiguationResult,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        disambiguation = DisambiguationResult.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"Disambiguation LLM call failed: {e}")
        return
        
    merged_count = 0
    for group in disambiguation.synonym_groups:
        canonical_name = group.canonical_name
        aliases = group.aliases
        
        # We need the canonical entity to exist in our map
        canonical_ent = entity_map.get(canonical_name)
        if not canonical_ent:
            # If LLM hallucinated a canonical name, pick the first valid alias
            valid_aliases = [a for a in aliases if a in entity_map]
            if not valid_aliases:
                continue
            canonical_name = valid_aliases[0]
            canonical_ent = entity_map[canonical_name]
            aliases = [a for a in aliases if a != canonical_name]
            
        for alias in aliases:
            alias_ent = entity_map.get(alias)
            if not alias_ent or alias_ent.id == canonical_ent.id:
                continue
                
            # Migrate outgoing relationships
            alias_outgoing = await db.execute(select(Relationship).where(Relationship.source_entity_id == alias_ent.id))
            for rel in alias_outgoing.scalars():
                existing = await db.execute(select(Relationship).where(and_(
                    Relationship.source_entity_id == canonical_ent.id, 
                    Relationship.target_entity_id == rel.target_entity_id, 
                    Relationship.relation_type == rel.relation_type
                )))
                if existing.scalar_one_or_none():
                    await db.execute(delete(Relationship).where(Relationship.id == rel.id))
                else:
                    rel.source_entity_id = canonical_ent.id
            
            # Migrate incoming relationships
            alias_incoming = await db.execute(select(Relationship).where(Relationship.target_entity_id == alias_ent.id))
            for rel in alias_incoming.scalars():
                existing = await db.execute(select(Relationship).where(and_(
                    Relationship.source_entity_id == rel.source_entity_id, 
                    Relationship.target_entity_id == canonical_ent.id, 
                    Relationship.relation_type == rel.relation_type
                )))
                if existing.scalar_one_or_none():
                    await db.execute(delete(Relationship).where(Relationship.id == rel.id))
                else:
                    rel.target_entity_id = canonical_ent.id
            
            await db.flush()
            
            # Delete the alias entity
            await db.execute(delete(Entity).where(Entity.id == alias_ent.id))
            merged_count += 1
            logger.info(f"Merged entity '{alias}' into '{canonical_name}'")
            
    # Clean up any potential duplicate relationships created by the merge
    # (e.g. if canonical and alias both had a 'USES' relationship to the same target)
    # A simple approach for this prototype is to let the unique constraint fail if we try 
    # to flush without handling it, but SQLAlchemy's update doesn't trigger python-side validation.
    # To be safe, we just let the database handle it. Since we didn't use ORM update, it might throw a UniqueViolation.
    # If we want to be bulletproof, we would do a more careful merge, but for this MVP we'll catch exceptions.
    
    logger.info(f"Entity disambiguation complete. Merged {merged_count} entities.")


async def _run_contradiction_pruning(db: AsyncSession, namespace_id: uuid.UUID) -> None:
    """Find contradictory relationships and prune the outdated ones."""
    logger.info("Running contradiction pruning...")
    
    # 1. Fetch relationships (limit to 200 for context window safety in MVP)
    stmt = (
        select(Relationship)
        .join(Entity, Relationship.source_entity_id == Entity.id)
        .where(Entity.namespace_id == namespace_id)
        .options(
            selectinload(Relationship.source_entity),
            selectinload(Relationship.target_entity)
        )
        .limit(200)
    )
    result = await db.execute(stmt)
    relationships = list(result.scalars().all())
    
    if not relationships:
        return
        
    # Format relationships for LLM
    rel_lines = []
    for r in relationships:
        rel_lines.append(
            f"ID: {r.id} | {r.source_entity.name} -> {r.relation_type} -> {r.target_entity.name} (Weight: {r.weight})"
        )
    rel_list_str = "\n".join(rel_lines)
    
    try:
        response = await litellm.acompletion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Relationships to analyze:\n{rel_list_str}"},
            ],
            response_format=ContradictionResult,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        contradictions = ContradictionResult.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"Contradiction LLM call failed: {e}")
        return
        
    pruned_count = 0
    for resolution in contradictions.resolutions:
        try:
            rel_id = uuid.UUID(resolution.relationship_id)
            # Delete the contradictory relationship
            result = await db.execute(
                delete(Relationship).where(Relationship.id == rel_id)
            )
            if result.rowcount > 0:
                pruned_count += 1
                logger.info(f"Pruned contradiction {rel_id}: {resolution.reason}")
        except ValueError:
            logger.warning(f"Invalid UUID returned by LLM: {resolution.relationship_id}")
            
    logger.info(f"Contradiction pruning complete. Pruned {pruned_count} relationships.")
