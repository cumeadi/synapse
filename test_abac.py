"""
Synapse Phase 3 — ABAC Integration Test Suite.

Verifies offline (no LLM calls) that:
1. Policy pure functions work correctly (label hierarchy, clearance ceilings)
2. Startup migrations run without error on the live DB
3. visibility_label filtering works in graph_search (entity hidden from role below ceiling)
4. Graph traversal is pruned when a path passes through a restricted entity
5. Audit log entries are persisted correctly
6. Zero-downtime: existing entities (default 'public') remain visible to all callers
"""

import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import select, text

from app.database import init_db, async_session_factory
from app.models import Namespace, Entity, Relationship, AuditLog
from app.services.core import graph_search
from app.services.policy import (
    clearance_level,
    can_access,
    permitted_labels,
    validate_label,
    LABEL_ORDER,
)


# ─────────────────────────────────────────────────────────────────────
# Section 1: Pure policy function tests (no DB)
# ─────────────────────────────────────────────────────────────────────

def test_policy_functions():
    print("\n1. Testing pure policy functions...")

    # clearance_level
    assert clearance_level("public") == 0
    assert clearance_level("internal") == 1
    assert clearance_level("confidential") == 2
    assert clearance_level("restricted") == 3
    assert clearance_level("unknown_label") == 0  # defaults to public

    # can_access — clearance ceiling
    assert can_access("public", "public") is True
    assert can_access("public", "internal") is False
    assert can_access("internal", "public") is True
    assert can_access("internal", "internal") is True
    assert can_access("internal", "confidential") is False
    assert can_access("confidential", "restricted") is False
    assert can_access("restricted", "restricted") is True
    assert can_access("restricted", "public") is True

    # permitted_labels
    assert permitted_labels("public") == ["public"]
    assert permitted_labels("internal") == ["public", "internal"]
    assert permitted_labels("confidential") == ["public", "internal", "confidential"]
    assert permitted_labels("restricted") == LABEL_ORDER

    # validate_label
    assert validate_label("Public") == "public"   # normalises case
    assert validate_label("RESTRICTED") == "restricted"
    try:
        validate_label("top_secret")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("   ✅ All policy function assertions passed.")


# ─────────────────────────────────────────────────────────────────────
# Section 2: DB integration tests
# ─────────────────────────────────────────────────────────────────────

async def test_abac_integration():
    print("\n2. Initializing DB and applying Phase 3 migrations...")
    await init_db()

    ns_name = f"abac_test_{uuid.uuid4().hex[:8]}"

    async with async_session_factory() as db:
        namespace = Namespace(name=ns_name)
        db.add(namespace)
        await db.commit()
        namespace_id = namespace.id
    print(f"   Created namespace: '{ns_name}' ({namespace_id})")

    # ── Insert test graph ─────────────────────────────────────────────
    # Alice (public) --[KNOWS]--> Bob (internal) --[WORKS_WITH]--> Carol (restricted)
    print("\n3. Inserting test entities with mixed visibility labels...")
    async with async_session_factory() as db:
        alice = Entity(namespace_id=namespace_id, name="Alice", entity_type="Person", visibility_label="public")
        bob   = Entity(namespace_id=namespace_id, name="Bob",   entity_type="Person", visibility_label="internal")
        carol = Entity(namespace_id=namespace_id, name="Carol", entity_type="Person", visibility_label="restricted")
        db.add_all([alice, bob, carol])
        await db.flush()

        rel_ab = Relationship(
            source_entity_id=alice.id,
            target_entity_id=bob.id,
            relation_type="KNOWS",
            weight=1.0,
            visibility_label="internal",
        )
        rel_bc = Relationship(
            source_entity_id=bob.id,
            target_entity_id=carol.id,
            relation_type="WORKS_WITH",
            weight=1.0,
            visibility_label="restricted",
        )
        db.add_all([rel_ab, rel_bc])
        await db.commit()

        alice_id = alice.id
        bob_id = bob.id
        carol_id = carol.id

    print(f"   Alice({alice_id}) public")
    print(f"   Bob({bob_id}) internal")
    print(f"   Carol({carol_id}) restricted")
    print(f"   Alice --[KNOWS (internal)]--> Bob --[WORKS_WITH (restricted)]--> Carol")

    # ── Test 1: public role caller ─────────────────────────────────────
    print("\n4. Testing graph search with role='public' (should see Alice only, no edges)...")
    result_public = await graph_search(
        db=async_session_factory(),
        namespace_id=namespace_id,
        entity_name="Alice",
        depth=2,
        role="public",
    )
    assert result_public is not None, "Alice should be found!"
    entity_names_public = {e.name for e in result_public["entities"]}
    print(f"   Visible entities: {entity_names_public}")
    print(f"   Visible relationships: {len(result_public['relationships'])}")
    # Alice is public, but the Alice->Bob edge is 'internal' — public caller can't traverse it
    assert "Bob" not in entity_names_public, "Bob (internal) should NOT be visible to public role!"
    assert "Carol" not in entity_names_public, "Carol (restricted) should NOT be visible to public role!"
    assert len(result_public["relationships"]) == 0, "Public caller should see no edges!"
    print("   ✅ Public role correctly blocked from internal/restricted content.")

    # ── Test 2: internal role caller ────────────────────────────────────
    print("\n5. Testing graph search with role='internal' (should see Alice + Bob, not Carol)...")
    result_internal = await graph_search(
        db=async_session_factory(),
        namespace_id=namespace_id,
        entity_name="Alice",
        depth=2,
        role="internal",
    )
    entity_names_internal = {e.name for e in result_internal["entities"]}
    print(f"   Visible entities: {entity_names_internal}")
    print(f"   Visible relationships: {len(result_internal['relationships'])}")
    assert "Alice" in entity_names_internal
    assert "Bob" in entity_names_internal, "Bob (internal) should be visible to internal role!"
    assert "Carol" not in entity_names_internal, "Carol (restricted) should NOT be visible to internal role!"
    assert len(result_internal["relationships"]) == 1, "Should see exactly 1 edge (Alice->Bob, internal)!"
    print("   ✅ Internal role correctly sees internal content but not restricted.")

    # ── Test 3: restricted role caller ──────────────────────────────────
    print("\n6. Testing graph search with role='restricted' (should see all entities)...")
    result_restricted = await graph_search(
        db=async_session_factory(),
        namespace_id=namespace_id,
        entity_name="Alice",
        depth=2,
        role="restricted",
    )
    entity_names_restricted = {e.name for e in result_restricted["entities"]}
    print(f"   Visible entities: {entity_names_restricted}")
    print(f"   Visible relationships: {len(result_restricted['relationships'])}")
    assert "Alice" in entity_names_restricted
    assert "Bob" in entity_names_restricted
    assert "Carol" in entity_names_restricted, "Carol (restricted) should be visible to restricted role!"
    assert len(result_restricted["relationships"]) == 2, "Should see both edges!"
    print("   ✅ Restricted role correctly sees all content.")

    # ── Test 4: Zero-downtime backward compat ───────────────────────────
    print("\n7. Testing backward compatibility — default 'public' visibility entities...")
    async with async_session_factory() as db:
        # Insert entity with no explicit label (should default to 'public')
        legacy = Entity(namespace_id=namespace_id, name="LegacyEntity", entity_type="System")
        db.add(legacy)
        await db.commit()
        legacy_id = legacy.id

    result_legacy = await graph_search(
        db=async_session_factory(),
        namespace_id=namespace_id,
        entity_name="LegacyEntity",
        depth=1,
        role="public",
    )
    assert result_legacy is not None, "Legacy entity should be found!"
    assert result_legacy["center"].visibility_label == "public"
    print("   ✅ Default-label entity visible to public role — backward compatible.")

    # ── Test 5: Audit log persistence ───────────────────────────────────
    print("\n8. Testing audit log persistence...")
    async with async_session_factory() as db:
        audit_entry = AuditLog(
            namespace_id=namespace_id,
            api_key_id=None,
            action="graph_search",
            entity_name="Alice",
            result_count=2,
            role_used="internal",
        )
        db.add(audit_entry)
        await db.commit()
        audit_id = audit_entry.id

    async with async_session_factory() as db:
        entry = await db.get(AuditLog, audit_id)
        assert entry is not None
        assert entry.action == "graph_search"
        assert entry.role_used == "internal"
        assert entry.entity_name == "Alice"
        assert entry.result_count == 2
        assert entry.created_at is not None
    print(f"   Audit entry persisted: ID={audit_id}")
    print(f"   action={entry.action}, role={entry.role_used}, count={entry.result_count}")
    print("   ✅ Audit log entry persisted and retrieved correctly.")

    # ── Cleanup ──────────────────────────────────────────────────────────
    print("\n9. Cleaning up test namespace...")
    async with async_session_factory() as db:
        await db.execute(text("DELETE FROM namespaces WHERE id = :id"), {"id": namespace_id})
        await db.commit()


async def main():
    print("=== Testing Phase 3: Attribute-Based Access Control (ABAC) ===")

    # Run pure function tests first (no DB needed)
    test_policy_functions()

    # Run DB integration tests
    await test_abac_integration()

    print("\n🎉 PHASE 3 ABAC VERIFICATION PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(main())
