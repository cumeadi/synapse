import asyncio
import uuid
import math
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, text
from app.database import init_db, async_session_factory, engine
from app.models import Namespace, Entity, Relationship, Memory
from app.services.core import graph_search

async def test_temporal_decay():
    print("=== Testing Phase 2: Temporal Edge Mechanics (Knowledge Half-Life) ===")
    
    # 1. Initialize Database and run startup migrations
    print("\n1. Initializing DB and applying Phase 2 startup migrations...")
    await init_db()
    
    ns_name = f"temporal_test_{uuid.uuid4().hex[:8]}"
    print(f"Created temporary test namespace: '{ns_name}'")
    
    async with async_session_factory() as db:
        # Create Namespace
        namespace = Namespace(name=ns_name)
        db.add(namespace)
        await db.commit()
        namespace_id = namespace.id
        print(f"Namespace ID: {namespace_id}")

    # 2. Manually insert the test entities and memory
    print("\n2. Manually inserting test entities and relationship into database...")
    async with async_session_factory() as db:
        alice = Entity(namespace_id=namespace_id, name="Alice", entity_type="Person")
        tennis = Entity(namespace_id=namespace_id, name="Tennis", entity_type="Sport")
        db.add(alice)
        db.add(tennis)
        await db.flush()
        
        rel = Relationship(
            source_entity_id=alice.id,
            target_entity_id=tennis.id,
            relation_type="PLAYS",
            weight=1.0,
            weight_decay_rate=0.01  # 1% decay per day
        )
        db.add(rel)
        await db.commit()
        
        rel_id = rel.id
        alice_id = alice.id
        tennis_id = tennis.id
        print(f"Inserted: Alice({alice_id}) —[PLAYS]→ Tennis({tennis_id})")

    # 3. Check created relationship details
    print("\n3. Verifying relationship temporal metadata in DB...")
    async with async_session_factory() as db:
        rel = await db.get(Relationship, rel_id)
        print(f"Relationship found: ID={rel.id}")
        print(f"  Base weight: {rel.weight}")
        print(f"  valid_from: {rel.valid_from} (UTC)")
        print(f"  last_reinforced_at: {rel.last_reinforced_at} (UTC)")
        print(f"  weight_decay_rate: {rel.weight_decay_rate}")
        
        assert rel.valid_from is not None, "valid_from is null!"
        assert rel.last_reinforced_at is not None, "last_reinforced_at is null!"
        assert abs((rel.last_reinforced_at - rel.valid_from).total_seconds()) < 5, "Timestamps differ unexpectedly!"

    # 4. Simulate reinforcement
    print("\n4. Reinforcing relationship...")
    async with async_session_factory() as db:
        # Simulate that time has passed first
        rel = await db.get(Relationship, rel_id)
        rel.last_reinforced_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await db.flush()
        
        # Reinforcement logic (replicating _upsert_relationship in app/services/core.py)
        rel.weight = rel.weight + 1.0
        rel.last_reinforced_at = datetime.now(timezone.utc)
        await db.commit()
        
    async with async_session_factory() as db:
        rel = await db.get(Relationship, rel_id)
        print(f"After reinforcement:")
        print(f"  Base weight: {rel.weight}")
        print(f"  last_reinforced_at: {rel.last_reinforced_at} (UTC)")
        
        assert rel.weight == 2.0
        time_diff = (datetime.now(timezone.utc) - rel.last_reinforced_at.replace(tzinfo=timezone.utc)).total_seconds()
        print(f"  Time diff from NOW: {time_diff:.2f} seconds")
        assert time_diff < 5, "last_reinforced_at was not updated to current time!"

    # 5. Simulate 30-day temporal decay (Decay, but above threshold)
    print("\n5. Simulating 30 days of temporal decay (Decayed weight should be ~1.48)...")
    async with async_session_factory() as db:
        # Manually alter last_reinforced_at to 30 days ago
        await db.execute(
            text("UPDATE relationships SET last_reinforced_at = NOW() - INTERVAL '30 days' WHERE id = :id"),
            {"id": rel_id}
        )
        await db.commit()
        
    # Query graph search with depth=1, default threshold=0.2
    # Decayed weight: 2.0 * exp(-0.01 * 30) = 1.48
    search_res = await graph_search(db=async_session_factory(), namespace_id=namespace_id, entity_name="Alice", depth=1)
    
    assert search_res is not None, "Entity 'Alice' not found!"
    print(f"Graph Search results after 30 days decay:")
    print(f"  Reachable entities: {[e.name for e in search_res['entities']]}")
    print(f"  Reachable relationships count: {len(search_res['relationships'])}")
    assert len(search_res['relationships']) > 0, "Pruned unexpectedly at 30 days!"
    
    returned_rel = search_res['relationships'][0]
    print(f"  Returned weight (in response): {returned_rel.decayed_weight:.4f}")
    assert abs(returned_rel.decayed_weight - 2.0 * math.exp(-0.01 * 30)) < 0.05, "Decayed weight calculation incorrect!"

    # 6. Simulate 300-day temporal decay (Below threshold, should be pruned!)
    print("\n6. Simulating 300 days of temporal decay (Decayed weight should be ~0.10, threshold=0.20)...")
    async with async_session_factory() as db:
        # Manually alter last_reinforced_at to 300 days ago
        await db.execute(
            text("UPDATE relationships SET last_reinforced_at = NOW() - INTERVAL '300 days' WHERE id = :id"),
            {"id": rel_id}
        )
        await db.commit()
        
    # Query graph search with default threshold=0.2
    search_res_pruned = await graph_search(db=async_session_factory(), namespace_id=namespace_id, entity_name="Alice", depth=1, threshold=0.2)
    print(f"Graph Search results after 300 days decay (threshold=0.2):")
    print(f"  Reachable entities: {[e.name for e in search_res_pruned['entities']]}")
    print(f"  Reachable relationships count: {len(search_res_pruned['relationships'])}")
    # Relationship plays tennis should be pruned from the traversal, leaving only the center node!
    assert len(search_res_pruned['relationships']) == 0, "Relationship was not pruned correctly below threshold!"

    # 7. Reinforce a fully decayed/pruned relationship and recover it
    print("\n7. Reinforcing the fully decayed relationship back to active status...")
    async with async_session_factory() as db:
        rel = await db.get(Relationship, rel_id)
        rel.weight = rel.weight + 1.0
        rel.last_reinforced_at = datetime.now(timezone.utc)
        await db.commit()
        
    # Query graph search again
    search_res_active = await graph_search(db=async_session_factory(), namespace_id=namespace_id, entity_name="Alice", depth=1, threshold=0.2)
    print(f"Graph Search results after reinforcing:")
    print(f"  Reachable entities: {[e.name for e in search_res_active['entities']]}")
    print(f"  Reachable relationships count: {len(search_res_active['relationships'])}")
    assert len(search_res_active['relationships']) > 0, "Relationship failed to recover after reinforcement!"
    print(f"  Recovered weight: {search_res_active['relationships'][0].decayed_weight:.4f}")
    assert abs(search_res_active['relationships'][0].decayed_weight - 3.0) < 0.001, "Recovered weight does not equal base weight!"
    
    # Clean up test namespace
    print("\n8. Cleaning up test namespace...")
    async with async_session_factory() as db:
        await db.execute(text("DELETE FROM namespaces WHERE id = :id"), {"id": namespace_id})
        await db.commit()
        
    print("\n🎉 PHASE 2 TEMPORAL EDGE MECHANICS VERIFICATION PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_temporal_decay())
