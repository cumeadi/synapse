import asyncio
import uuid
import json
from unittest.mock import AsyncMock, patch
from sqlalchemy import text, select, and_
from datetime import datetime, timezone

from app.database import async_session_factory, init_db
from app.models import Namespace, Entity, Relationship, Memory, AuditLog
from app.services.core import _upsert_entity, _upsert_relationship, hybrid_search, graph_search
from app.schemas import ExtractedEntity, ExtractedRelationship
from app.services.sleep import run_sleep_cycle, consolidate_reflexes
from app.mcp_server import store_reflex, report_reflex_failure

async def run_tests():
    print("Initializing Database...")
    await init_db()
    
    # Globally monkey-patch embedding generation for robust offline testing
    import app.services.core
    app.services.core._generate_embedding = AsyncMock(return_value=[0.0] * 1536)
    
    ns_id = uuid.uuid4()
    ns_name = f"test_ns_{ns_id.hex[:6]}"
    
    # Create the test namespace
    async with async_session_factory() as db:
        db.add(Namespace(id=ns_id, name=ns_name))
        await db.commit()
        
    try:
        # ────────────────────────────────────────────────────────────
        # 1. Test store_reflex MCP Tool logic (Governance: ACTIVE by default)
        # ────────────────────────────────────────────────────────────
        print("1. Testing store_reflex MCP tool (ACTIVE by default)...")
        trigger = {"event_type": "pull_request", "repo": "frontend"}
        payload = "Format files in {{repo}} and run prettier for {{query}}."
        
        result_str = await store_reflex(
            namespace_name=ns_name,
            trigger_condition=trigger,
            executable_payload=payload,
            source_entity="User",
            target_entity="Cerebellum",
            relation_type="REFLEX"
        )
        assert "Reflex stored successfully" in result_str, f"Expected success message, got {result_str}"
        print("   ✅ store_reflex tool registered successfully.")
        
        # Verify the database contains the reflex relationship
        async with async_session_factory() as db:
            stmt = select(Relationship).join(Entity, Relationship.source_entity_id == Entity.id).where(
                and_(
                    Entity.namespace_id == ns_id,
                    Relationship.epistemic_state == "REFLEX"
                )
            )
            res = await db.execute(stmt)
            rels = res.scalars().all()
            assert len(rels) == 1
            reflex_rel = rels[0]
            assert reflex_rel.trigger_condition == trigger
            assert reflex_rel.executable_payload == payload
            assert reflex_rel.status == "ACTIVE", f"Expected default status ACTIVE, got {reflex_rel.status}"
            reflex_rel_id = reflex_rel.id
            print("   ✅ Reflex edge persisted with correct columns, status='ACTIVE', and epistemic_state='REFLEX'.")
            
        # ────────────────────────────────────────────────────────────
        # 2. Test Parameterization (Dynamic Substitution) & Audit Logging
        # ────────────────────────────────────────────────────────────
        print("2. Testing parameter template substitution and audit logging...")
        async with async_session_factory() as db:
            # Check with a query matching the metadata trigger
            search_results = await hybrid_search(
                db=db,
                namespace_id=ns_id,
                query="PR review pipeline",
                metadata_filter={"event_type": "pull_request", "repo": "frontend"}
            )
            assert len(search_results) == 1
            item = search_results[0]
            assert item["metadata"]["is_reflex"] is True
            assert "[STANDING ORDER]" in item["content"]
            assert "Action: Format files in frontend and run prettier for PR review pipeline." in item["content"]
            assert '"repo": "frontend"' in item["content"]
            assert '"event_type": "pull_request"' in item["content"]
            assert item["score"] == 1.0
            print("   ✅ Templates with double brackets (e.g. {{repo}} and {{query}}) rendered perfectly.")
            
            # Assert AuditLog entry was written
            stmt = select(AuditLog).where(
                and_(
                    AuditLog.namespace_id == ns_id,
                    AuditLog.action == "reflex_triggered"
                )
            )
            res = await db.execute(stmt)
            audit_logs = res.scalars().all()
            assert len(audit_logs) >= 1
            print("   ✅ AuditLog 'reflex_triggered' successfully logged.")
            
        # ────────────────────────────────────────────────────────────
        # 3. Test graph_search Interception
        # ────────────────────────────────────────────────────────────
        print("3. Testing graph_search interception...")
        async with async_session_factory() as db:
            res = await graph_search(
                db=db,
                namespace_id=ns_id,
                entity_name="User"
            )
            assert len(res["relationships"]) == 1
            assert res["relationships"][0].epistemic_state == "REFLEX"
            assert res["relationships"][0].executable_payload == payload
            print("   ✅ graph_search correctly intercepted and returned the reflex edge.")

        # ────────────────────────────────────────────────────────────
        # 4. Test Offline Reflex Consolidation (Sleep Cycle PROPOSED default)
        # ────────────────────────────────────────────────────────────
        print("4. Testing offline sleep cycle consolidation (Defaults to PROPOSED)...")
        # Populate mock AuditLogs and Memories
        async with async_session_factory() as db:
            # Clear previous reflex to avoid mixing tests
            await db.execute(text("DELETE FROM relationships WHERE id = :id"), {"id": reflex_rel_id})
            await db.commit()
            
            db.add(AuditLog(namespace_id=ns_id, action="graph_search", entity_name="Format", role_used="internal"))
            db.add(AuditLog(namespace_id=ns_id, action="graph_search", entity_name="Lint", role_used="internal"))
            
            db.add(Memory(namespace_id=ns_id, content="User pushed code to frontend repository. Formatted files with prettier.", metadata_={"repo": "frontend"}))
            db.add(Memory(namespace_id=ns_id, content="Agent executed lint checking. Committed changes.", metadata_={"repo": "frontend"}))
            await db.commit()
            
        # Mock litellm to return a proposed reflex consolidation
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content=json.dumps({
                        "reflexes": [
                            {
                                "trigger_condition": {"query": "format code", "repo": "frontend"},
                                "executable_payload": "Perform Prettier autoformat in {{repo}}.",
                                "source_entity": "User",
                                "target_entity": "Cerebellum",
                                "relation_type": "REFLEX",
                                "reasoning": "Recurring push triggers Prettier formatting sequence."
                            }
                        ]
                    })
                )
            )
        ]
        
        with patch("litellm.acompletion", return_value=mock_response):
            async with async_session_factory() as db:
                await consolidate_reflexes(db, ns_id)
                await db.commit()
                
        # Verify the consolidated reflex is created in the db and defaults to PROPOSED
        async with async_session_factory() as db:
            stmt = select(Relationship).join(Entity, Relationship.source_entity_id == Entity.id).where(
                and_(
                    Entity.namespace_id == ns_id,
                    Relationship.epistemic_state == "REFLEX"
                )
            )
            res = await db.execute(stmt)
            rels = res.scalars().all()
            assert len(rels) == 1
            consolidated_reflex = rels[0]
            assert consolidated_reflex.trigger_condition == {"query": "format code", "repo": "frontend"}
            assert consolidated_reflex.status == "PROPOSED", f"Expected sleep-consolidated status PROPOSED, got {consolidated_reflex.status}"
            consolidated_id = consolidated_reflex.id
            print("   ✅ Sleep cycle consolidated reflexes autonomously and defaulted status to PROPOSED.")
            
        # ────────────────────────────────────────────────────────────
        # 5. Test Shadow Reflex Behavior (PROPOSED status)
        # ────────────────────────────────────────────────────────────
        print("5. Testing Shadow Reflex behavior (PROPOSED: no interception, logs shadow)...")
        async with async_session_factory() as db:
            # Query matching the trigger condition
            search_results = await hybrid_search(
                db=db,
                namespace_id=ns_id,
                query="format code",
                metadata_filter={"repo": "frontend"}
            )
            # Since status is PROPOSED, it is a shadow reflex. It should not intercept,
            # returning 0 or standard matches instead of the standing order.
            for item in search_results:
                assert item["metadata"].get("is_reflex") is not True, "Should not return reflex standing order under PROPOSED shadow state!"
            print("   ✅ Shadow reflex successfully avoided search interception.")
            
            # Assert reflex_shadow_triggered AuditLog entry was written
            stmt = select(AuditLog).where(
                and_(
                    AuditLog.namespace_id == ns_id,
                    AuditLog.action == "reflex_shadow_triggered"
                )
            )
            res = await db.execute(stmt)
            shadow_logs = res.scalars().all()
            assert len(shadow_logs) >= 1
            print("   ✅ AuditLog 'reflex_shadow_triggered' successfully logged.")
            
        # ────────────────────────────────────────────────────────────
        # 6. Test Promoting Reflex Status to ACTIVE
        # ────────────────────────────────────────────────────────────
        print("6. Testing reflex activation promotion...")
        async with async_session_factory() as db:
            # Promote the status to ACTIVE
            from app.services.studio import update_reflex_status
            success = await update_reflex_status(db, ns_id, consolidated_id, "ACTIVE")
            assert success is True, "Expected update_reflex_status to succeed"
            
        async with async_session_factory() as db:
            # Call hybrid_search again on the activated reflex
            search_results = await hybrid_search(
                db=db,
                namespace_id=ns_id,
                query="format code",
                metadata_filter={"repo": "frontend"}
            )
            assert len(search_results) == 1
            item = search_results[0]
            assert "[STANDING ORDER]" in item["content"]
            assert "Action: Perform Prettier autoformat in frontend." in item["content"]
            assert '"repo": "frontend"' in item["content"]
            assert '"query": "format code"' in item["content"]
            print("   ✅ Promoted reflex successfully intercepted search and rendered variables.")

        # ────────────────────────────────────────────────────────────
        # 7. Test Explicit Failure Tool (report_reflex_failure)
        # ────────────────────────────────────────────────────────────
        print("7. Testing explicit failure reporting (report_reflex_failure)...")
        result_str = await report_reflex_failure(
            namespace_name=ns_name,
            reflex_relationship_id=str(consolidated_id)
        )
        assert "Reflex failure reported successfully" in result_str, f"Expected success message, got {result_str}"
        print("   ✅ report_reflex_failure returned successfully.")
        
        async with async_session_factory() as db:
            # Verify the relationship is paused, penalized, and reverted to FACT
            stmt = select(Relationship).where(Relationship.id == consolidated_id)
            res = await db.execute(stmt)
            penalized_rel = res.scalar_one_or_none()
            
            assert penalized_rel is not None
            assert penalized_rel.epistemic_state == "FACT", f"Expected epistemic_state reverted to FACT, got {penalized_rel.epistemic_state}"
            assert penalized_rel.status == "PAUSED", f"Expected status PAUSED, got {penalized_rel.status}"
            assert penalized_rel.confidence == 0.01, f"Expected confidence floor 0.01, got {penalized_rel.confidence}"
            print("   ✅ Reflex successfully PAUSED, confidence floor set to 0.01, reverted to standard FACT.")
            
            # Assert reflex_failed AuditLog entry was written
            stmt = select(AuditLog).where(
                and_(
                    AuditLog.namespace_id == ns_id,
                    AuditLog.action == "reflex_failed"
                )
            )
            res = await db.execute(stmt)
            fail_logs = res.scalars().all()
            assert len(fail_logs) >= 1
            print("   ✅ AuditLog 'reflex_failed' successfully logged.")
            
        # ────────────────────────────────────────────────────────────
        # 8. Test Reflex Unlearning / Contradictions (Online)
        # ────────────────────────────────────────────────────────────
        print("8. Testing instant online contradiction unlearning...")
        # Create a new active reflex to contradict
        result_str = await store_reflex(
            namespace_name=ns_name,
            trigger_condition={"repo": "frontend"},
            executable_payload="Format code.",
            source_entity="User",
            target_entity="Cerebellum",
            relation_type="REFLEX"
        )
        
        async with async_session_factory() as db:
            stmt = select(Relationship).join(Entity, Relationship.source_entity_id == Entity.id).where(
                and_(
                    Entity.namespace_id == ns_id,
                    Relationship.epistemic_state == "REFLEX"
                )
            )
            res = await db.execute(stmt)
            active_reflex = res.scalars().first()
            active_id = active_reflex.id
            
            # Explicitly query entities to avoid lazy-loading MissingGreenlet errors
            stmt_user = select(Entity).where(and_(Entity.namespace_id == ns_id, Entity.name == "User"))
            res_user = await db.execute(stmt_user)
            e_user = res_user.scalar_one()

            stmt_cer = select(Entity).where(and_(Entity.namespace_id == ns_id, Entity.name == "Cerebellum"))
            res_cer = await db.execute(stmt_cer)
            e_cerebellum = res_cer.scalar_one()
            
            entity_map = {"User": e_user, "Cerebellum": e_cerebellum}
            
            ext_r = ExtractedRelationship(source="User", target="Cerebellum", relation="USES")
            # Ingesting the contradicting standard relationship
            await _upsert_relationship(db, entity_map, ext_r)
            await db.commit()
            
        async with async_session_factory() as db:
            # Fetch the reflex edge to see if it was reverted
            stmt = select(Relationship).where(Relationship.id == active_id)
            res = await db.execute(stmt)
            reverted_rel = res.scalar_one_or_none()
            
            assert reverted_rel is not None
            assert reverted_rel.epistemic_state == "FACT", f"Expected epistemic_state reverted to FACT, got {reverted_rel.epistemic_state}"
            assert reverted_rel.confidence == 0.01, f"Expected confidence penalized to 0.01, got {reverted_rel.confidence}"
            assert reverted_rel.has_contradiction is True
            print("   ✅ Contradicted active reflex successfully unlearned and reverted instantly.")
            
    finally:
        # Clean up database
        async with async_session_factory() as db:
            await db.execute(text("DELETE FROM namespaces WHERE id = :id"), {"id": str(ns_id)})
            await db.commit()
            print("Cleanup completed.")
            
    print("\n🎉 CEREBELLUM ENGINE VERIFICATION PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
