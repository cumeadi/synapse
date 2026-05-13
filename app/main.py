"""
Synapse — FastAPI application with REST endpoints and auth middleware.

Provides endpoints for namespace management, memory ingestion,
hybrid search, and knowledge graph traversal.
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Query,
    Request,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, init_db
from app.models import ApiKey, Entity, KnowledgeSource, Namespace
from app.schemas import (
    GitHubSourceCreate,
    GraphEntityResponse,
    GraphNeighborhoodResponse,
    GraphRelationshipResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    KnowledgeSourceResponse,
    MemoryCreate,
    MemoryResponse,
    NamespaceCreate,
    NamespaceResponse,
    PackSourceCreate,
    SearchResultItem,
    StudioGraphResponse,
    EdgeTraceResponse,
    NamespaceListResponse
)
from app.services import graph_search, hybrid_search, ingest_memory, run_sleep_cycle
from app.connectors.github import ingest_github_repo
from app.services.grafting import run_grafting_cycle
from app.services.studio import get_full_graph, delete_entity, delete_relationship, trace_relationship

load_dotenv()

logger = logging.getLogger("synapse.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

MASTER_KEY = os.getenv("SYNAPSE_MASTER_KEY", "")


# ────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Synapse starting up — initializing database...")
    await init_db()
    logger.info("Database ready.")
    yield
    logger.info("Synapse shutting down.")


# ────────────────────────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Synapse",
    description="Cognitive Memory Engine — semantic & relational memory for AI agents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the Studio UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# ────────────────────────────────────────────────────────────────────
# Routes: Studio UI
# ────────────────────────────────────────────────────────────────────
@app.get("/studio")
async def serve_studio():
    """Serve the Synapse Studio HTML."""
    return FileResponse("static/index.html")


# ────────────────────────────────────────────────────────────────────
# Auth Dependency
# ────────────────────────────────────────────────────────────────────
async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> Optional[ApiKey]:
    """
    Verify the X-API-Key header. Allows:
    - Master key (full access)
    - Namespace-scoped API key
    - No key required if SYNAPSE_MASTER_KEY is empty (dev mode)
    """
    if not MASTER_KEY:
        # Dev mode — no auth required
        return None

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    if x_api_key == MASTER_KEY:
        return None  # Master key — full access

    # Check database for namespace-scoped key
    stmt = select(ApiKey).where(ApiKey.key == x_api_key, ApiKey.is_active == True)
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid or inactive API key.")

    return api_key


# ────────────────────────────────────────────────────────────────────
# Routes: Namespaces
# ────────────────────────────────────────────────────────────────────
@app.post("/namespaces/", response_model=NamespaceResponse, status_code=201)
async def create_namespace(
    body: NamespaceCreate,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Create a new namespace and generate an API key for it."""
    # Check if namespace already exists
    existing = await db.execute(
        select(Namespace).where(Namespace.name == body.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Namespace '{body.name}' already exists.")

    namespace = Namespace(name=body.name)
    db.add(namespace)
    await db.flush()

    # Generate API key for this namespace
    raw_key = f"syn_{secrets.token_urlsafe(32)}"
    api_key = ApiKey(
        key=raw_key,
        namespace_id=namespace.id,
        description=f"Auto-generated key for namespace '{body.name}'",
    )
    db.add(api_key)
    await db.flush()

    logger.info(f"Created namespace '{body.name}' with id {namespace.id}")

    return NamespaceResponse(
        id=namespace.id,
        name=namespace.name,
        created_at=namespace.created_at,
        api_key=raw_key,
    )

@app.get("/namespaces/", response_model=list[NamespaceListResponse])
async def list_namespaces(
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """List all namespaces (used by Studio dropdown)."""
    result = await db.execute(select(Namespace).order_by(Namespace.name))
    namespaces = result.scalars().all()
    return [NamespaceListResponse.model_validate(ns) for ns in namespaces]


# ────────────────────────────────────────────────────────────────────
# Routes: Memory Ingestion
# ────────────────────────────────────────────────────────────────────
@app.post(
    "/namespaces/{namespace_id}/memories/",
    response_model=MemoryResponse,
    status_code=202,
)
async def create_memory(
    namespace_id: uuid.UUID,
    body: MemoryCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """
    Accept a memory for ingestion. Returns 202 immediately and
    processes embedding generation + knowledge extraction in the background.
    """
    # Verify namespace exists
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    memory_id = uuid.uuid4()

    # Schedule background ingestion
    background_tasks.add_task(
        ingest_memory,
        namespace_id=namespace_id,
        memory_id=memory_id,
        content=body.content,
        metadata=body.metadata,
    )

    return MemoryResponse(
        id=memory_id,
        status="accepted",
        message="Memory ingestion is processing in the background.",
    )

# ────────────────────────────────────────────────────────────────────
# Routes: Sleep Cycle (Consolidation)
# ────────────────────────────────────────────────────────────────────
@app.post(
    "/namespaces/{namespace_id}/sleep",
    status_code=202,
)
async def trigger_sleep_cycle(
    namespace_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """
    Manually trigger the Sleep Cycle (entity disambiguation and contradiction pruning).
    Runs asynchronously in the background.
    """
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    background_tasks.add_task(run_sleep_cycle, namespace_id=namespace_id)

    return {"status": "accepted", "message": "Sleep cycle scheduled in the background."}



# ────────────────────────────────────────────────────────────────────
# Routes: Hybrid Search
# ────────────────────────────────────────────────────────────────────
@app.post(
    "/namespaces/{namespace_id}/search/hybrid",
    response_model=HybridSearchResponse,
)
async def search_hybrid(
    namespace_id: uuid.UUID,
    body: HybridSearchRequest,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Semantic vector search with optional metadata filtering."""
    # Verify namespace exists
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    results = await hybrid_search(
        db=db,
        namespace_id=namespace_id,
        query=body.query,
        metadata_filter=body.metadata_filter,
        top_k=body.top_k,
    )

    return HybridSearchResponse(
        query=body.query,
        results=[SearchResultItem(**r) for r in results],
        total=len(results),
    )


# ────────────────────────────────────────────────────────────────────
# Routes: Graph Search
# ────────────────────────────────────────────────────────────────────
@app.get(
    "/namespaces/{namespace_id}/search/graph",
    response_model=GraphNeighborhoodResponse,
)
async def search_graph(
    namespace_id: uuid.UUID,
    entity_name: str = Query(..., description="Name of the entity to query"),
    depth: int = Query(default=1, ge=1, le=3, description="Traversal depth (1-3 hops)"),
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Return the graph neighborhood of an entity up to N hops deep."""
    # Verify namespace exists
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    result = await graph_search(
        db=db,
        namespace_id=namespace_id,
        entity_name=entity_name,
        depth=depth,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_name}' not found in this namespace.",
        )

    return GraphNeighborhoodResponse(
        center=GraphEntityResponse.model_validate(result["center"]),
        depth=result["depth"],
        entities=[GraphEntityResponse.model_validate(e) for e in result["entities"]],
        relationships=[
            GraphRelationshipResponse(
                id=r.id,
                source=GraphEntityResponse.model_validate(r.source_entity),
                target=GraphEntityResponse.model_validate(r.target_entity),
                relation_type=r.relation_type,
                weight=r.weight,
            )
            for r in result["relationships"]
        ],
    )


# ────────────────────────────────────────────────────────────────────
# Routes: Knowledge Sources — GitHub Import
# ────────────────────────────────────────────────────────────────────
@app.post(
    "/namespaces/{namespace_id}/sources/github",
    response_model=KnowledgeSourceResponse,
    status_code=202,
)
async def import_github_source(
    namespace_id: uuid.UUID,
    body: GitHubSourceCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """
    Import a GitHub repository as a knowledge source.
    Returns 202 immediately and processes ingestion + auto-grafting in the background.
    """
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    # Create KnowledgeSource record
    source = KnowledgeSource(
        namespace_id=namespace_id,
        name=body.repo_url,
        source_type="github_repo",
        status="pending",
    )
    db.add(source)
    await db.flush()

    logger.info(f"Queued GitHub import: {body.repo_url} -> source {source.id}")

    # Schedule background: ingest then graft
    async def _ingest_and_graft():
        await ingest_github_repo(
            namespace_id=namespace_id,
            source_id=source.id,
            repo_url=body.repo_url,
        )
        await run_grafting_cycle(
            namespace_id=namespace_id,
            new_source_id=source.id,
        )

    background_tasks.add_task(_ingest_and_graft)

    return KnowledgeSourceResponse(
        id=source.id,
        name=source.name,
        source_type=source.source_type,
        status="pending",
        message="GitHub repository import and auto-grafting queued.",
    )


# ────────────────────────────────────────────────────────────────────
# Routes: Knowledge Sources — Static Pack Import
# ────────────────────────────────────────────────────────────────────
@app.post(
    "/namespaces/{namespace_id}/sources/pack",
    response_model=KnowledgeSourceResponse,
    status_code=202,
)
async def import_pack_source(
    namespace_id: uuid.UUID,
    body: PackSourceCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """
    Import a pre-compiled knowledge pack (entities + relationships).
    Returns 202 immediately and triggers auto-grafting in the background.
    """
    from app.connectors.github import (
        _upsert_entity_with_source,
        _upsert_relationship_with_source,
    )

    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    # Create KnowledgeSource record
    source = KnowledgeSource(
        namespace_id=namespace_id,
        name=body.name,
        source_type="static_pack",
        status="ingesting",
    )
    db.add(source)
    await db.flush()

    # Persist entities and relationships synchronously (they're already extracted)
    entity_map = {}
    for ext_ent in body.entities:
        entity = await _upsert_entity_with_source(
            db, namespace_id, ext_ent, source.id
        )
        entity_map[ext_ent.name] = entity

    for ext_rel in body.relationships:
        await _upsert_relationship_with_source(
            db, entity_map, ext_rel, source.id
        )

    source.status = "ready"
    await db.flush()

    logger.info(f"Pack '{body.name}' imported: {len(body.entities)} entities, {len(body.relationships)} relationships")

    # Schedule grafting as background task
    background_tasks.add_task(
        run_grafting_cycle,
        namespace_id=namespace_id,
        new_source_id=source.id,
    )

    return KnowledgeSourceResponse(
        id=source.id,
        name=source.name,
        source_type=source.source_type,
        status="ready",
        message="Knowledge pack imported. Auto-grafting queued.",
    )


# ────────────────────────────────────────────────────────────────────
# Routes: Studio Graph APIs
# ────────────────────────────────────────────────────────────────────
@app.get("/namespaces/{namespace_id}/graph", response_model=StudioGraphResponse)
async def get_namespace_graph(
    namespace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Get the full graph for a namespace formatted for VisNetwork."""
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")
    
    graph_data = await get_full_graph(db, namespace_id)
    return StudioGraphResponse(**graph_data)

@app.delete("/namespaces/{namespace_id}/entities/{entity_id}", status_code=204)
async def delete_graph_entity(
    namespace_id: uuid.UUID,
    entity_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Delete an entity and cascade its relationships."""
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")
        
    success = await delete_entity(db, namespace_id, entity_id)
    if not success:
        raise HTTPException(status_code=404, detail="Entity not found in this namespace.")
    return None

@app.delete("/namespaces/{namespace_id}/relationships/{relationship_id}", status_code=204)
async def delete_graph_relationship(
    namespace_id: uuid.UUID,
    relationship_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Delete a specific relationship between entities."""
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")
        
    success = await delete_relationship(db, namespace_id, relationship_id)
    if not success:
        raise HTTPException(status_code=404, detail="Relationship not found in this namespace.")
    return None
    
@app.get("/namespaces/{namespace_id}/relationships/{relationship_id}/trace", response_model=EdgeTraceResponse)
async def trace_graph_relationship(
    namespace_id: uuid.UUID,
    relationship_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Trace a relationship back to the best-matching source memory via semantic search."""
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")
        
    trace_data = await trace_relationship(db, namespace_id, relationship_id)
    return EdgeTraceResponse(**trace_data)


# ────────────────────────────────────────────────────────────────────
# Health Check
# ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "synapse"}
