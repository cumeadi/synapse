import asyncio
import logging
from datetime import datetime, timezone
from sqlalchemy import select, text, update, func, and_
from app.database import async_session_factory
from app.models import Relationship
from app.services.confidence import detect_and_flag_contradictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synapse.dream_worker")

async def run_dream_cycle():
    """
    The Synapse Dream Cycle (Asynchronous Background Worker).
    Runs offline to perform maintenance tasks without blocking ingestion:
    1. Apply permanent weight decay
    2. Detect and flag contradictions
    3. (Optional) Entity disambiguation / deduplication
    """
    logger.info("Starting Synapse Dream Cycle...")
    
    async with async_session_factory() as db:
        # ── 1. Apply Permanent Weight Decay ──────────────────────────────
        # Mathematically, we can permanently shrink the weight and reset 
        # last_reinforced_at to NOW(). This prevents numbers from underflowing
        # over years and optimizes read paths if we want to drop dynamic decay.
        logger.info("Applying exponential decay to all relationships...")
        decay_stmt = text("""
            UPDATE relationships
            SET 
                weight = weight * EXP(-weight_decay_rate * (EXTRACT(EPOCH FROM (NOW() - last_reinforced_at)) / 86400.0)),
                last_reinforced_at = NOW()
            WHERE weight_decay_rate > 0
        """)
        await db.execute(decay_stmt)
        await db.commit()
        logger.info("Weight decay applied.")

        # ── 2. Contradiction Detection ───────────────────────────────────
        # Find all entity pairs that have multiple relationships.
        # This means they are candidates for contradictions (e.g. LIKES and DISLIKES).
        logger.info("Detecting contradictions...")
        pairs_stmt = (
            select(
                Relationship.source_entity_id,
                Relationship.target_entity_id
            )
            .group_by(Relationship.source_entity_id, Relationship.target_entity_id)
            .having(func.count(Relationship.id) > 1)
        )
        
        result = await db.execute(pairs_stmt)
        pairs = result.all()
        
        contradiction_count = 0
        for source_id, target_id in pairs:
            has_contra = await detect_and_flag_contradictions(db, source_id, target_id)
            if has_contra:
                contradiction_count += 1
                
        await db.commit()
        logger.info(f"Contradiction detection complete. Flagged {contradiction_count} pairs with contradictions.")
        
        # ── 3. Entity Disambiguation (Stub) ──────────────────────────────
        # In the future, this is where we query for entities with similar embeddings
        # or names (e.g., 'React' and 'ReactJS') and use the LLM to merge them.
        logger.info("Entity disambiguation (skipped - pending implementation).")

    logger.info("Synapse Dream Cycle completed successfully.")

if __name__ == "__main__":
    asyncio.run(run_dream_cycle())
