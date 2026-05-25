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
from app.prompts import DISAMBIGUATION_SYSTEM_PROMPT, CONTRADICTION_SYSTEM_PROMPT, PROCEDURAL_COMPRESSION_PROMPT
from app.schemas import DisambiguationResult, ContradictionResult

logger = logging.getLogger("synapse.services.sleep")

LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

async def run_sleep_cycle(namespace_id: uuid.UUID) -> None:
    """
    Main orchestrator for the sleep cycle.
    Runs Entity Disambiguation followed by Contradiction Pruning, and then Reflex Consolidation.
    """
    from app.database import async_session_factory

    logger.info(f"Starting sleep cycle for namespace: {namespace_id}")
    
    async with async_session_factory() as db:
        try:
            await _run_entity_disambiguation(db, namespace_id)
            await _run_contradiction_pruning(db, namespace_id)
            await consolidate_reflexes(db, namespace_id)
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


async def consolidate_reflexes(db: AsyncSession, namespace_id: uuid.UUID) -> None:
    """
    Look for repetitive workflows / multi-hop reasoning chains in recent AuditLogs and Memories,
    and compress them into REFLEX edges.
    """
    logger.info(f"Consolidating reflexes for namespace: {namespace_id}")
    
    from app.models import AuditLog, Memory, Entity, Relationship
    from app.services.core import _upsert_entity
    from app.schemas import ExtractedEntity
    from app.services.confidence import update_relationship_confidence
    from sqlalchemy import desc
    from pydantic import BaseModel, Field
    import json
    from datetime import datetime, timezone
    
    # 1. Fetch recent AuditLogs
    logs_stmt = select(AuditLog).where(AuditLog.namespace_id == namespace_id).order_by(desc(AuditLog.created_at)).limit(100)
    logs_res = await db.execute(logs_stmt)
    logs = logs_res.scalars().all()
    
    # 2. Fetch recent Memories
    mems_stmt = select(Memory).where(Memory.namespace_id == namespace_id).order_by(desc(Memory.created_at)).limit(30)
    mems_res = await db.execute(mems_stmt)
    mems = mems_res.scalars().all()
    
    if not logs and not mems:
        logger.info("No audit logs or memories found. Skipping reflex consolidation.")
        return
        
    # Serialize logs and memories for LLM
    log_texts = []
    for l in logs:
        log_texts.append(f"[{l.created_at.isoformat()}] Action: {l.action} | Entity: {l.entity_name} | Role: {l.role_used}")
    
    mem_texts = []
    for m in mems:
        mem_texts.append(f"[{m.created_at.isoformat()}] Memory: {m.content} (Metadata: {json.dumps(m.metadata_)})")
        
    context = "RECENT AUDIT LOGS:\n" + "\n".join(log_texts[:50]) + "\n\nRECENT MEMORIES:\n" + "\n".join(mem_texts[:20])
    
    # Define LLM Response schema for structured output
    class ProposedReflex(BaseModel):
        trigger_condition: dict[str, Any] = Field(description="JSON pattern that activates this reflex, e.g., {'query': 'pr review', 'repo': 'frontend'}")
        executable_payload: str = Field(description="The standing order or collapsed prompt for the agent to execute immediately without reasoning.")
        source_entity: str = Field(default="User", description="Source entity name, usually 'User'")
        target_entity: str = Field(default="Cerebellum", description="Target entity name, usually 'Cerebellum'")
        relation_type: str = Field(default="REFLEX", description="Relationship/verb type, usually 'REFLEX'")
        reasoning: str = Field(description="Explanation of why this reflex is proposed based on the repetitive pattern detected.")
        
    class ReflexConsolidationResult(BaseModel):
        reflexes: list[ProposedReflex] = Field(default_factory=list)
        
    try:
        response = await litellm.acompletion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PROCEDURAL_COMPRESSION_PROMPT},
                {"role": "user", "content": f"Analyze this context for procedural memory consolidation:\n\n{context}"},
            ],
            response_format=ReflexConsolidationResult,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        result = ReflexConsolidationResult.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"Reflex consolidation LLM call failed: {e}")
        return
        
    consolidated_count = 0
    for reflex in result.reflexes:
        logger.info(f"LLM proposed reflex: {reflex.reasoning}")
        try:
            # Upsert source and target entities
            source_ext = ExtractedEntity(name=reflex.source_entity, entity_type="System", epistemic_state="REFLEX")
            source = await _upsert_entity(db, namespace_id, source_ext)
            
            target_ext = ExtractedEntity(name=reflex.target_entity, entity_type="System", epistemic_state="REFLEX")
            target = await _upsert_entity(db, namespace_id, target_ext)
            
            # Create or update relationship
            stmt = select(Relationship).where(
                and_(
                    Relationship.source_entity_id == source.id,
                    Relationship.target_entity_id == target.id,
                    Relationship.relation_type == reflex.relation_type,
                )
            )
            rel_res = await db.execute(stmt)
            existing = rel_res.scalar_one_or_none()
            
            if existing:
                existing.trigger_condition = reflex.trigger_condition
                existing.executable_payload = reflex.executable_payload
                existing.epistemic_state = "REFLEX"
                if not existing.status:
                    existing.status = "PROPOSED"
                existing.last_reinforced_at = datetime.now(timezone.utc)
                await db.flush()
                await update_relationship_confidence(db, existing)
            else:
                rel = Relationship(
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    relation_type=reflex.relation_type,
                    epistemic_state="REFLEX",
                    weight=1.0,
                    source_diversity_count=1,
                    confidence=1.0,
                    has_contradiction=False,
                    trigger_condition=reflex.trigger_condition,
                    executable_payload=reflex.executable_payload,
                    status="PROPOSED",
                )
                db.add(rel)
                await db.flush()
                await update_relationship_confidence(db, rel)
                
            consolidated_count += 1
            logger.info(f"Consolidated reflex: {reflex.source_entity} -[{reflex.relation_type}]→ {reflex.target_entity}")
        except Exception as ex:
            logger.error(f"Failed to save consolidated reflex: {ex}", exc_info=True)
            
    logger.info(f"Reflex consolidation complete. Created/updated {consolidated_count} reflexes.")

