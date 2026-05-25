"""
Synapse — MCP (Model Context Protocol) Server.

Exposes Synapse capabilities as MCP tools so AI agents can
store and retrieve memories via the standard MCP interface.

Run standalone: python -m app.mcp_server
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from app.database import async_session_factory, init_db
from app.models import KnowledgeSource, Namespace
from app.services import graph_search, hybrid_search, ingest_memory
from app.connectors.github import ingest_github_repo
from app.services.grafting import run_grafting_cycle
from app.services.policy import DEFAULT_ROLE_MCP, validate_label

from sqlalchemy import select

logger = logging.getLogger("synapse.mcp")
logging.basicConfig(level=logging.INFO)

# ────────────────────────────────────────────────────────────────────
# MCP Server Instance
# ────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "Synapse",
)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
async def _resolve_namespace(namespace_name: str) -> uuid.UUID | None:
    """Look up a namespace by name, return its ID or None."""
    async with async_session_factory() as db:
        stmt = select(Namespace).where(Namespace.name == namespace_name)
        result = await db.execute(stmt)
        ns = result.scalar_one_or_none()
        return ns.id if ns else None


# ────────────────────────────────────────────────────────────────────
# Tool: store_memory
# ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def store_memory(
    namespace_name: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """
    Store a new memory in Synapse. The content will be embedded for
    semantic search and analyzed by an LLM to extract entities and
    relationships into the knowledge graph.

    Args:
        namespace_name: Name of the namespace to store the memory in.
        content: The text content to memorize and extract knowledge from.
        metadata: Optional key-value metadata for filtering later.

    Returns:
        Confirmation message with the memory ID.
    """
    ns_id = await _resolve_namespace(namespace_name)
    if not ns_id:
        return f"Error: Namespace '{namespace_name}' not found. Create it first."

    memory_id = uuid.uuid4()

    try:
        await ingest_memory(
            namespace_id=ns_id,
            memory_id=memory_id,
            content=content,
            metadata=metadata or {},
        )
        return (
            f"Memory stored successfully.\n"
            f"  ID: {memory_id}\n"
            f"  Namespace: {namespace_name}\n"
            f"  Status: Embedding generated and knowledge graph updated."
        )
    except Exception as e:
        return f"Error storing memory: {e}"


# ────────────────────────────────────────────────────────────────────
# Tool: search_semantic
# ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def search_semantic(
    namespace_name: str,
    query: str,
    metadata_filter: Optional[dict[str, Any]] = None,
    top_k: int = 5,
    role: str = DEFAULT_ROLE_MCP,
) -> str:
    """
    Search memories by semantic similarity. Uses vector cosine
    similarity with optional metadata filtering.

    Args:
        namespace_name: Name of the namespace to search in.
        query: Natural language search query.
        metadata_filter: Optional JSONB containment filter (e.g., {"source": "chat"}).
        top_k: Number of results to return (default 5).
        role: ABAC role for visibility filtering (public|internal|confidential|restricted).
              Defaults to 'public' for unauthenticated MCP callers.

    Returns:
        Formatted search results with content and similarity scores.
    """
    ns_id = await _resolve_namespace(namespace_name)
    if not ns_id:
        return f"Error: Namespace '{namespace_name}' not found."

    async with async_session_factory() as db:
        results = await hybrid_search(
            db=db,
            namespace_id=ns_id,
            query=query,
            metadata_filter=metadata_filter,
            top_k=top_k,
        )

    if not results:
        return f"No memories found matching query: '{query}'"

    output_lines = [f"Found {len(results)} memories matching '{query}':\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(
            f"  {i}. [Score: {r['score']:.4f}] {r['content'][:200]}"
        )
        if r["metadata"]:
            output_lines.append(f"     Metadata: {r['metadata']}")
    return "\n".join(output_lines)


# ────────────────────────────────────────────────────────────────────
# Tool: search_relational
# ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def search_relational(
    namespace_name: str,
    entity_name: str,
    depth: int = 1,
    role: str = DEFAULT_ROLE_MCP,
    min_confidence: float = 0.0,
) -> str:
    """
    Query the knowledge graph for an entity's neighborhood.
    Returns connected entities and their relationships.

    Args:
        namespace_name: Name of the namespace to search in.
        entity_name: Name of the entity to look up (e.g., 'User', 'Python').
        depth: How many hops to traverse (1-3, default 1).
        role: ABAC role for visibility filtering (public|internal|confidential|restricted).
              Defaults to 'public' for unauthenticated MCP callers.
        min_confidence: Minimum confidence score for entities/relationships (0.0 to 1.0).

    Returns:
        Formatted graph neighborhood with entities and relationships.
    """
    ns_id = await _resolve_namespace(namespace_name)
    if not ns_id:
        return f"Error: Namespace '{namespace_name}' not found."

    async with async_session_factory() as db:
        result = await graph_search(
            db=db,
            namespace_id=ns_id,
            entity_name=entity_name,
            depth=min(max(depth, 1), 3),
            role=role,
            min_confidence=min_confidence,
        )

    if result is None:
        return f"Entity '{entity_name}' not found in namespace '{namespace_name}'."

    center = result["center"]
    entities = result["entities"]
    relationships = result["relationships"]

    output_lines = [
        f"Graph neighborhood for '{center.name}' ({center.entity_type}) — depth {result['depth']}:",
        f"\nEntities ({len(entities)}):",
    ]
    for e in entities:
        marker = " ★" if e.id == center.id else ""
        output_lines.append(f"  • {e.name} ({e.entity_type}){marker}")

    output_lines.append(f"\nRelationships ({len(relationships)}):")
    for r in relationships:
        output_lines.append(
            f"  {r.source_entity.name} —[{r.relation_type} (w={r.weight})]→ {r.target_entity.name}"
        )

    return "\n".join(output_lines)


# ────────────────────────────────────────────────────────────────────
# Tool: store_reflex
# ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def store_reflex(
    namespace_name: str,
    trigger_condition: dict[str, Any],
    executable_payload: str,
    source_entity: str = "User",
    target_entity: str = "Cerebellum",
    relation_type: str = "REFLEX",
) -> str:
    """
    Explicitly save a recurring workflow / procedural standing order as a reflex.
    When a future query matches the trigger condition, Synapse will intercept
    the search and return this standing order directly.

    Args:
        namespace_name: Name of the namespace to store the reflex in.
        trigger_condition: A dictionary specifying context patterns (e.g. {"query": "pr review", "repo": "frontend"}).
        executable_payload: The collapsed prompt or standing order the agent should execute.
        source_entity: Name of the source entity (default "User").
        target_entity: Name of the target entity (default "Cerebellum").
        relation_type: Relationship type (default "REFLEX").

    Returns:
        Confirmation message with the reflex relationship ID.
    """
    from datetime import datetime, timezone
    from app.services.core import _upsert_entity
    from app.schemas import ExtractedEntity
    from app.models import Relationship
    from app.services.confidence import update_relationship_confidence
    from sqlalchemy import and_

    ns_id = await _resolve_namespace(namespace_name)
    if not ns_id:
        return f"Error: Namespace '{namespace_name}' not found. Create it first."

    try:
        async with async_session_factory() as db:
            # 1. Resolve or create source entity
            source_ext = ExtractedEntity(name=source_entity, entity_type="System", epistemic_state="REFLEX")
            source = await _upsert_entity(db, ns_id, source_ext)

            # 2. Resolve or create target entity
            target_ext = ExtractedEntity(name=target_entity, entity_type="System", epistemic_state="REFLEX")
            target = await _upsert_entity(db, ns_id, target_ext)

            # 3. Create or update the relationship as a reflex
            stmt = select(Relationship).where(
                and_(
                    Relationship.source_entity_id == source.id,
                    Relationship.target_entity_id == target.id,
                    Relationship.relation_type == relation_type,
                )
            )
            res = await db.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.trigger_condition = trigger_condition
                existing.executable_payload = executable_payload
                existing.epistemic_state = "REFLEX"
                existing.status = "ACTIVE"
                existing.last_reinforced_at = datetime.now(timezone.utc)
                await db.flush()
                await update_relationship_confidence(db, existing)
                rel_id = existing.id
            else:
                rel = Relationship(
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    relation_type=relation_type,
                    epistemic_state="REFLEX",
                    weight=1.0,
                    source_diversity_count=1,
                    confidence=1.0,
                    has_contradiction=False,
                    trigger_condition=trigger_condition,
                    executable_payload=executable_payload,
                    status="ACTIVE",
                )
                db.add(rel)
                await db.flush()
                await update_relationship_confidence(db, rel)
                rel_id = rel.id

            await db.commit()
            return (
                f"Reflex stored successfully.\n"
                f"  ID: {rel_id}\n"
                f"  Namespace: {namespace_name}\n"
                f"  Trigger: {trigger_condition}\n"
                f"  Payload: {executable_payload[:200]}..."
            )
    except Exception as e:
        return f"Error storing reflex: {e}"


# ────────────────────────────────────────────────────────────────────
# Tool: report_reflex_failure
# ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def report_reflex_failure(
    namespace_name: str,
    reflex_relationship_id: str,
) -> str:
    """
    Explicitly report that a reflex standing order failed during execution.
    This will instantly freeze (PAUSE) the reflex, set its confidence to
    the minimum floor (0.01), and revert it to a standard declarative FACT,
    forcing the agent to fall back to standard reasoning on the very next attempt.

    Args:
        namespace_name: Name of the namespace the reflex belongs to.
        reflex_relationship_id: The UUID string of the reflex relationship that failed.

    Returns:
        Confirmation message detailing the penalty and fallback status.
    """
    from app.models import Relationship, Entity, AuditLog
    from app.services.confidence import CONFIDENCE_FLOOR
    from sqlalchemy import and_

    ns_id = await _resolve_namespace(namespace_name)
    if not ns_id:
        return f"Error: Namespace '{namespace_name}' not found."

    try:
        rel_uuid = uuid.UUID(reflex_relationship_id)
    except ValueError:
        return f"Error: Invalid reflex relationship UUID: '{reflex_relationship_id}'."

    try:
        async with async_session_factory() as db:
            # Query the relationship, ensuring it belongs to the namespace
            stmt = (
                select(Relationship)
                .join(Entity, Relationship.source_entity_id == Entity.id)
                .where(
                    and_(
                        Relationship.id == rel_uuid,
                        Entity.namespace_id == ns_id,
                    )
                )
            )
            res = await db.execute(stmt)
            rel = res.scalar_one_or_none()

            if not rel:
                return f"Error: Reflex relationship '{reflex_relationship_id}' not found in namespace '{namespace_name}'."

            # Revert reflex to a paused standard fact with confidence floor
            rel.epistemic_state = "FACT"
            rel.status = "PAUSED"
            rel.confidence = CONFIDENCE_FLOOR
            await db.flush()

            # Log reflex failure event to AuditLog
            entry = AuditLog(
                namespace_id=ns_id,
                action="reflex_failed",
                entity_name=f"Reflex: {rel_uuid}",
                result_count=1,
                role_used="internal"
            )
            db.add(entry)
            await db.flush()

            await db.commit()
            return (
                f"Reflex failure reported successfully.\n"
                f"  Reflex ID: {reflex_relationship_id}\n"
                f"  Status: PAUSED\n"
                f"  Confidence: 0.01 (Penalized to floor)\n"
                f"  Epistemic State: Reverted to FACT\n"
                f"  Fallback: Standard reasoning has been restored immediately."
            )
    except Exception as e:
        return f"Error reporting reflex failure: {e}"


# ────────────────────────────────────────────────────────────────────
# Tool: learn_repository
# ────────────────────────────────────────────────────────────────────
@mcp.tool()
async def learn_repository(
    namespace_name: str,
    repo_url: str,
) -> str:
    """
    Import a GitHub repository's architecture into the knowledge graph.
    Fetches README and key files, extracts entities and relationships,
    then auto-grafts connections to the user's existing knowledge.

    Use this when you need to understand a library, framework, or project
    that isn't yet in your knowledge graph.

    Args:
        namespace_name: Name of the namespace to import into.
        repo_url: GitHub repository URL (e.g., 'https://github.com/tiangolo/fastapi').

    Returns:
        Status message about the import and grafting process.
    """
    ns_id = await _resolve_namespace(namespace_name)
    if not ns_id:
        return f"Error: Namespace '{namespace_name}' not found. Create it first."

    try:
        # Create KnowledgeSource record
        async with async_session_factory() as db:
            source = KnowledgeSource(
                namespace_id=ns_id,
                name=repo_url,
                source_type="github_repo",
                status="pending",
            )
            db.add(source)
            await db.commit()
            source_id = source.id

        # Run ingestion
        await ingest_github_repo(
            namespace_id=ns_id,
            source_id=source_id,
            repo_url=repo_url,
        )

        # Run auto-grafting
        await run_grafting_cycle(
            namespace_id=ns_id,
            new_source_id=source_id,
        )

        # Get final status
        async with async_session_factory() as db:
            source = await db.get(KnowledgeSource, source_id)
            status = source.status if source else "unknown"

        return (
            f"Repository learned successfully.\n"
            f"  URL: {repo_url}\n"
            f"  Source ID: {source_id}\n"
            f"  Status: {status}\n"
            f"  The repository's architecture has been extracted into your "
            f"knowledge graph and auto-grafted to your existing knowledge."
        )
    except Exception as e:
        return f"Error learning repository: {e}"


# ────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────
async def _startup():
    """Initialize the database before MCP server starts."""
    await init_db()
    logger.info("Synapse MCP server — database initialized.")


def main():
    """Run the MCP server with stdio transport."""
    asyncio.get_event_loop().run_until_complete(_startup())
    mcp.run()


if __name__ == "__main__":
    main()
