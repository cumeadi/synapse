import asyncio
import uuid
import math
from datetime import datetime, timezone

from sqlalchemy import text
from app.database import engine, async_session_factory, init_db
from app.models import Namespace, Entity, Relationship
from app.services.confidence import compute_relationship_confidence
from app.services.core import _upsert_entity, _upsert_relationship, graph_search
from app.schemas import ExtractedEntity, ExtractedRelationship

async def run_tests():
    # 1. Test pure functions (Formula correctness)
    print("1. Testing pure confidence functions...")
    conf_1 = compute_relationship_confidence(1, 0.01, 0.0, False)
    conf_3 = compute_relationship_confidence(3, 0.01, 0.0, False)
    conf_6 = compute_relationship_confidence(6, 0.01, 0.0, False)
    assert 0.38 < conf_1 < 0.40, f"Expected ~0.39, got {conf_1}"
    assert 0.77 < conf_3 < 0.78, f"Expected ~0.777, got {conf_3}"
    assert 0.94 < conf_6 < 0.96, f"Expected ~0.95, got {conf_6}"
    
    # Contradiction penalty
    conf_3_contra = compute_relationship_confidence(3, 0.01, 0.0, True)
    assert abs(conf_3_contra - (conf_3 * 0.7)) < 0.001, "Contradiction penalty failed"
    print("   ✅ Formula logic is mathematically correct.")

    # 2. Database Integration Setup
    await init_db()
    ns_id = uuid.uuid4()
    async with async_session_factory() as db:
        db.add(Namespace(id=ns_id, name="Test NS"))
        await db.commit()
    
    try:
        # Test Source Diversity Tracking
        print("2. Testing source diversity and contradiction detection...")
        async with async_session_factory() as db:
            ext_e1 = ExtractedEntity(name="Alice", entity_type="Person")
            ext_e2 = ExtractedEntity(name="Payments", entity_type="Service")
            
            e1 = await _upsert_entity(db, ns_id, ext_e1, context_id=uuid.uuid4())
            e2 = await _upsert_entity(db, ns_id, ext_e2, context_id=uuid.uuid4())
            
            entity_map = {"Alice": e1, "Payments": e2}
            ext_r = ExtractedRelationship(source="Alice", target="Payments", relation="MAINTAINS")
            
            # 1st source
            src1 = uuid.uuid4()
            rel = await _upsert_relationship(db, entity_map, ext_r, context_id=src1)
            assert rel.source_diversity_count == 1
            assert rel.has_contradiction == False
            
            # Same source again (diversity should not increase)
            rel = await _upsert_relationship(db, entity_map, ext_r, context_id=src1)
            assert rel.source_diversity_count == 1
            assert rel.weight == 2.0
            
            # 2nd source (diversity should increase)
            src2 = uuid.uuid4()
            rel = await _upsert_relationship(db, entity_map, ext_r, context_id=src2)
            assert rel.source_diversity_count == 2
            assert rel.weight == 3.0
            assert rel.confidence > 0.6  # approx 1 - exp(-0.5 * 2) = 0.63
            print("   ✅ Source diversity tracks correctly.")
            
            # Contradiction detection
            ext_r_contra = ExtractedRelationship(source="Alice", target="Payments", relation="BROKE")
            src3 = uuid.uuid4()
            rel2 = await _upsert_relationship(db, entity_map, ext_r_contra, context_id=src3)
            await db.commit()
            
            # Run the dream cycle to detect contradictions and apply decay
            from app.dream_worker import run_dream_cycle
            await run_dream_cycle()
            
            # Both should now be flagged as contradictory and confidence drops
            async with async_session_factory() as db2:
                from sqlalchemy import select
                stmt = select(Relationship).where(Relationship.source_entity_id == e1.id)
                res = await db2.execute(stmt)
                rels = res.scalars().all()
                assert len(rels) == 2
                assert all(r.has_contradiction == True for r in rels)
                
                # Because of decay + contradiction, confidence is much lower
                assert rels[0].confidence < 0.63
                print("   ✅ Contradiction correctly flagged and penalized (Offline Dream Cycle).")
            
                e1 = await db2.get(Entity, e1.id)
                assert e1.confidence > 0.01  # should be populated
                print("   ✅ Entity confidence correctly propagates.")

        print("3. Testing Graph Search filtering by confidence...")
        async with async_session_factory() as db:
            # e1 confidence is ~0.44. e2 has no outgoing rels, so confidence is 0.5.
            # If we query with min_confidence 0.7, nothing should return.
            res = await graph_search(db, ns_id, "Alice", min_confidence=0.7)
            assert len(res["entities"]) == 0 # everything filtered out
            assert len(res["relationships"]) == 0
            
            # With min_confidence 0.0, both entities and both relations should return
            res = await graph_search(db, ns_id, "Alice", min_confidence=0.0)
            assert len(res["entities"]) == 2
            assert len(res["relationships"]) == 2
            
            print("   ✅ Graph search filtering works correctly.")
            
    finally:
        async with async_session_factory() as db:
            await db.execute(text("DELETE FROM namespaces WHERE id = :id"), {"id": str(ns_id)})
            await db.commit()
    
    print("\n🎉 PHASE 4 CONFIDENCE VERIFICATION PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
