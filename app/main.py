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
from app.models import ApiKey, AuditLog, Entity, KnowledgeSource, Namespace
from app.schemas import (
    AuditLogListResponse,
    AuditLogResponse,
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
    NamespaceListResponse,
    WebhookEventResponse,
    ReflexStatusUpdate,
)
from app.services import graph_search, hybrid_search, ingest_memory, run_sleep_cycle
from app.connectors.github import ingest_github_repo
from app.services.grafting import run_grafting_cycle
from app.services.studio import get_full_graph, delete_entity, delete_relationship, trace_relationship
from app.services.policy import resolve_caller_role, validate_label
from app.connectors.webhook_base import SUPPORTED_CONNECTORS, is_duplicate
from app.connectors.slack import SlackConnector, is_url_verification, get_challenge
from app.connectors.jira import JiraConnector
from app.connectors.github_events import GitHubEventsConnector

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


def _get_role(auth: Optional[ApiKey], x_api_key: Optional[str] = None) -> str:
    """
    Derive the effective ABAC role from the resolved auth object.
    Master key callers and dev-mode callers are granted 'internal' (safe default).
    """
    master_key_used = (x_api_key == MASTER_KEY) if (MASTER_KEY and x_api_key) else False
    return resolve_caller_role(auth, master_key_used=master_key_used)


async def _write_audit_log(
    db: AsyncSession,
    namespace_id,
    action: str,
    role: str,
    api_key_id=None,
    entity_name: Optional[str] = None,
    result_count: int = 0,
) -> None:
    """Persist an immutable audit log entry."""
    entry = AuditLog(
        namespace_id=namespace_id,
        api_key_id=api_key_id,
        action=action,
        entity_name=entity_name,
        result_count=result_count,
        role_used=role,
    )
    db.add(entry)
    # Flush within the existing session — committed by get_db() on success


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
    # Validate role if provided
    role = body.role or "internal"
    try:
        validate_label(role)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Check if namespace already exists
    existing = await db.execute(
        select(Namespace).where(Namespace.name == body.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Namespace '{body.name}' already exists.")

    namespace = Namespace(name=body.name)
    db.add(namespace)
    await db.flush()

    # Generate API key for this namespace with the requested role
    raw_key = f"syn_{secrets.token_urlsafe(32)}"
    api_key = ApiKey(
        key=raw_key,
        namespace_id=namespace.id,
        description=f"Auto-generated key for namespace '{body.name}'",
        role=role,
    )
    db.add(api_key)
    await db.flush()

    logger.info(f"Created namespace '{body.name}' with id {namespace.id} (role={role})")

    return NamespaceResponse(
        id=namespace.id,
        name=namespace.name,
        created_at=namespace.created_at,
        api_key=raw_key,
        role=role,
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
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score for edges/nodes"),
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    """Return the graph neighborhood of an entity up to N hops deep."""
    # Verify namespace exists
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    role = _get_role(_auth, x_api_key)

    result = await graph_search(
        db=db,
        namespace_id=namespace_id,
        entity_name=entity_name,
        depth=depth,
        role=role,
        min_confidence=min_confidence,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_name}' not found in this namespace.",
        )

    # Audit log
    await _write_audit_log(
        db=db,
        namespace_id=namespace_id,
        action="graph_search",
        role=role,
        api_key_id=_auth.id if _auth else None,
        entity_name=entity_name,
        result_count=len(result["entities"]),
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
                visibility_label=r.visibility_label,
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
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score for edges/nodes"),
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    """Get the full graph for a namespace formatted for VisNetwork."""
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    role = _get_role(_auth, x_api_key)
    graph_data = await get_full_graph(db, namespace_id, role=role, min_confidence=min_confidence)

    await _write_audit_log(
        db=db,
        namespace_id=namespace_id,
        action="studio_view",
        role=role,
        api_key_id=_auth.id if _auth else None,
        result_count=len(graph_data["nodes"]),
    )

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


@app.patch("/namespaces/{namespace_id}/relationships/{relationship_id}/status")
async def update_reflex_status_route(
    namespace_id: uuid.UUID,
    relationship_id: uuid.UUID,
    body: ReflexStatusUpdate,
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
):
    """Update the governance/HITL status of a reflex relationship (PROPOSED|ACTIVE|PAUSED)."""
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")
        
    status_upper = body.status.upper()
    if status_upper not in ("PROPOSED", "ACTIVE", "PAUSED"):
        raise HTTPException(status_code=400, detail="Invalid status. Must be PROPOSED, ACTIVE, or PAUSED.")
        
    from app.services.studio import update_reflex_status
    success = await update_reflex_status(db, namespace_id, relationship_id, status_upper)
    if not success:
        raise HTTPException(status_code=404, detail="Reflex relationship not found in this namespace.")
        
    return {"status": status_upper, "message": "Reflex status updated successfully."}


# ────────────────────────────────────────────────────────────────────
# Routes: Audit Log (Phase 3 ABAC)
# ────────────────────────────────────────────────────────────────────
@app.get(
    "/namespaces/{namespace_id}/audit",
    response_model=AuditLogListResponse,
)
async def get_audit_log(
    namespace_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=500, description="Max entries to return"),
    db: AsyncSession = Depends(get_db),
    _auth: Optional[ApiKey] = Depends(verify_api_key),
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Retrieve recent audit log entries for this namespace.
    Only accessible with the master key (full-access callers).
    """
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    # Audit endpoint is master-key only
    if MASTER_KEY and x_api_key != MASTER_KEY:
        raise HTTPException(
            status_code=403,
            detail="Audit log access requires the master key.",
        )

    from sqlalchemy import desc
    stmt = (
        select(AuditLog)
        .where(AuditLog.namespace_id == namespace_id)
        .order_by(desc(AuditLog.created_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    entries = result.scalars().all()

    return AuditLogListResponse(
        entries=[AuditLogResponse.model_validate(e) for e in entries],
        total=len(entries),
    )


# ────────────────────────────────────────────────────────────────────
# Routes: Webhook Ingestion (Phase 1 Ambient Ingestion)
# ────────────────────────────────────────────────────────────────────

# Registry of active connector instances
_CONNECTORS: dict[str, object] = {
    "slack": SlackConnector(),
    "jira": JiraConnector(),
    "github": GitHubEventsConnector(),
}


@app.post(
    "/webhooks/{namespace_id}/{connector_type}",
    response_model=WebhookEventResponse,
)
async def receive_webhook(
    namespace_id: uuid.UUID,
    connector_type: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Generic webhook receiver for ambient enterprise event ingestion.

    Accepts payloads from configured connectors (slack, jira, github).
    Each request is:
      1. Signature-verified via HMAC-SHA256
      2. Checked for idempotency (duplicate events silently skipped)
      3. Normalized to plain text
      4. Dispatched as a background LLM extraction task

    Returns immediately with 200 — processing is non-blocking.
    """
    # ── Validate connector type ──────────────────────────────────────
    if connector_type not in SUPPORTED_CONNECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported connector '{connector_type}'. "
                   f"Supported: {', '.join(SUPPORTED_CONNECTORS)}",
        )

    # ── Validate namespace ───────────────────────────────────────────
    ns = await db.get(Namespace, namespace_id)
    if not ns:
        raise HTTPException(status_code=404, detail="Namespace not found.")

    # ── Read raw body bytes (needed for HMAC before JSON parsing) ────
    body = await request.body()
    # Lowercase all header names for consistent access in connectors
    headers = {k.lower(): v for k, v in request.headers.items()}

    # ── Get connector ────────────────────────────────────────────────
    connector = _CONNECTORS[connector_type]

    # ── Parse JSON payload ───────────────────────────────────────────
    import json as _json
    try:
        payload = _json.loads(body)
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # ── Slack URL verification challenge (must respond before auth) ──
    if connector_type == "slack" and is_url_verification(payload):
        challenge = get_challenge(payload)
        return WebhookEventResponse(
            status="challenge",
            connector_type=connector_type,
            event_id="url_verification",
            event_type="url_verification",
            message="Slack URL verification challenge accepted.",
        )

    # ── Signature verification ───────────────────────────────────────
    if not connector.verify_signature(body, headers):
        raise HTTPException(
            status_code=403,
            detail="Webhook signature verification failed.",
        )

    # ── Extract stable event identifiers ────────────────────────────
    event_id = connector.extract_event_id(payload, headers)
    event_type = connector.extract_event_type(payload, headers)

    if not event_id:
        raise HTTPException(
            status_code=400,
            detail="Could not extract event ID from payload.",
        )

    # ── Idempotency check ────────────────────────────────────────────
    if await is_duplicate(db, namespace_id, connector_type, event_id):
        logger.info(f"Webhook: duplicate event skipped ({connector_type}/{event_id})")
        return WebhookEventResponse(
            status="skipped",
            connector_type=connector_type,
            event_id=event_id,
            event_type=event_type,
            message="Event already processed (idempotent skip).",
        )

    # ── Normalize payload to plain text ──────────────────────────────
    normalized_text = connector.normalize(payload, headers)

    # ── Persist WebhookEvent record ──────────────────────────────────
    from app.models import WebhookEvent
    webhook_record = WebhookEvent(
        namespace_id=namespace_id,
        connector_type=connector_type,
        event_id=event_id,
        event_type=event_type,
        status="skipped" if normalized_text is None else "pending",
        normalized_text=normalized_text,
    )
    db.add(webhook_record)
    # session committed by get_db() after this handler returns

    # ── Dispatch background extraction (only if normalized) ──────────
    if normalized_text:
        logger.info(
            f"Webhook accepted: {connector_type}/{event_type} "
            f"→ dispatching extraction for namespace {namespace_id}"
        )
        background_tasks.add_task(
            ingest_memory,
            namespace_id=namespace_id,
            memory_id=uuid.uuid4(),
            content=normalized_text,
            metadata={
                "source": f"webhook:{connector_type}",
                "event_type": event_type,
                "event_id": event_id,
            },
        )
    else:
        logger.debug(f"Webhook: no normalization for {connector_type}/{event_type} — skipped.")

    return WebhookEventResponse(
        status="accepted" if normalized_text else "skipped",
        connector_type=connector_type,
        event_id=event_id,
        event_type=event_type,
        message=(
            "Event accepted and queued for knowledge extraction."
            if normalized_text else
            "Event received but skipped (no actionable content)."
        ),
    )


# ────────────────────────────────────────────────────────────────────
# Health Check
# ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "synapse"}
