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
    description="Cognitive memory engine — store, search, and traverse agent memories.",
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
