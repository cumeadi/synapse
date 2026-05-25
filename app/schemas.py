"""
Synapse — Pydantic schemas for extraction and API request/response.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# LLM Extraction Schemas (used with litellm structured outputs)
# ────────────────────────────────────────────────────────────────────
class ExtractedEntity(BaseModel):
    """A single entity extracted from user text."""
    name: str = Field(
        ..., description="Normalized entity name (e.g., 'User', 'Python', 'AWS')"
    )
    entity_type: str = Field(
        ..., description="Category (e.g., 'Person', 'Technology', 'Organization')"
    )
    epistemic_state: str = Field(
        default="FACT", description="'FACT' for objective reality, 'TAKE' for subjective opinions"
    )


class ExtractedRelationship(BaseModel):
    """A directed relationship between two extracted entities."""
    source: str = Field(
        ..., description="Exact match to an entity name defined above"
    )
    target: str = Field(
        ..., description="Exact match to an entity name defined above"
    )
    relation: str = Field(
        ..., description="UPPER_SNAKE_CASE relationship (e.g., 'USES', 'DISLIKES')"
    )
    epistemic_state: str = Field(
        default="FACT", description="'FACT' for objective reality, 'TAKE' for subjective opinions"
    )


class MemoryExtraction(BaseModel):
    """Complete extraction result from a single user message."""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# API Request Schemas
# ────────────────────────────────────────────────────────────────────
class NamespaceCreate(BaseModel):
    """Request body for creating a new namespace."""
    name: str = Field(
        ..., min_length=1, max_length=255,
        description="Unique name for the namespace",
        examples=["agent-alpha", "user-chikau"],
    )
    role: Optional[str] = Field(
        default="internal",
        description="ABAC role for the auto-generated API key (public|internal|confidential|restricted)",
        examples=["internal"],
    )


class MemoryCreate(BaseModel):
    """Request body for ingesting a new memory."""
    content: str = Field(
        ..., min_length=1,
        description="The text content to store and extract knowledge from",
        examples=["I love Python and work at Google. I dislike Java."],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for filtering",
        examples=[{"source": "chat", "session_id": "abc123"}],
    )


class HybridSearchRequest(BaseModel):
    """Request body for hybrid (semantic + metadata) search."""
    query: str = Field(
        ..., min_length=1,
        description="Natural language search query",
        examples=["What programming languages does the user like?"],
    )
    metadata_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSONB containment filter (e.g., {'source': 'chat'})",
    )
    top_k: int = Field(
        default=10, ge=1, le=100,
        description="Number of results to return",
    )


class ReflexStatusUpdate(BaseModel):
    """Request body for updating a reflex status."""
    status: str = Field(
        ...,
        description="The new status of the reflex (PROPOSED|ACTIVE|PAUSED)",
        examples=["ACTIVE"],
    )


# ────────────────────────────────────────────────────────────────────
# API Response Schemas
# ────────────────────────────────────────────────────────────────────
class NamespaceResponse(BaseModel):
    """Response for a namespace."""
    id: uuid.UUID
    name: str
    created_at: datetime
    api_key: Optional[str] = Field(
        default=None,
        description="API key (only returned on creation)",
    )
    role: Optional[str] = Field(
        default=None,
        description="ABAC role granted to the auto-generated API key",
    )

    model_config = {"from_attributes": True}


class MemoryResponse(BaseModel):
    """Response confirming memory ingestion was accepted."""
    id: uuid.UUID
    status: str = "accepted"
    message: str = "Memory ingestion is processing in the background."


class SearchResultItem(BaseModel):
    """A single search result with similarity score."""
    id: uuid.UUID
    content: str
    metadata: dict[str, Any]
    score: float = Field(description="Cosine similarity score (0-1, higher is better)")
    created_at: datetime

    model_config = {"from_attributes": True}


class HybridSearchResponse(BaseModel):
    """Response for hybrid search."""
    query: str
    results: List[SearchResultItem]
    total: int


class GraphEntityResponse(BaseModel):
    """An entity in the graph response."""
    id: uuid.UUID
    name: str
    entity_type: str
    visibility_label: str = "public"
    observation_count: int = 1
    confidence: float = 0.5
    epistemic_state: str = "FACT"

    model_config = {"from_attributes": True}


class GraphRelationshipResponse(BaseModel):
    """A relationship in the graph response."""
    id: uuid.UUID
    source: GraphEntityResponse
    target: GraphEntityResponse
    relation_type: str
    weight: float
    visibility_label: str = "public"
    source_diversity_count: int = 1
    confidence: float = 0.5
    has_contradiction: bool = False
    epistemic_state: str = "FACT"
    trigger_condition: Optional[dict[str, Any]] = None
    executable_payload: Optional[str] = None
    status: Optional[str] = None

    model_config = {"from_attributes": True}


class GraphNeighborhoodResponse(BaseModel):
    """Full neighborhood response for a graph query."""
    center: GraphEntityResponse
    depth: int
    entities: List[GraphEntityResponse]
    relationships: List[GraphRelationshipResponse]


# ────────────────────────────────────────────────────────────────────
# Grafting Extraction Schema (used with litellm structured outputs)
# ────────────────────────────────────────────────────────────────────
class GraftingExtraction(BaseModel):
    """Bridge relationships identified by the grafting engine."""
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Sleep Cycle / Consolidation Schemas (used with litellm structured outputs)
# ────────────────────────────────────────────────────────────────────
class SynonymGroup(BaseModel):
    """A group of entity names that all refer to the exact same concept."""
    canonical_name: str = Field(description="The preferred name for the merged entity")
    aliases: List[str] = Field(description="List of other names that should be merged into the canonical name")

class DisambiguationResult(BaseModel):
    """Result of the entity disambiguation process."""
    synonym_groups: List[SynonymGroup] = Field(default_factory=list)

class ContradictionResolution(BaseModel):
    """Action to resolve a contradiction."""
    relationship_id: str = Field(description="The UUID of the relationship to prune/invalidate")
    reason: str = Field(description="Explanation of why this relationship is invalid based on context")

class ContradictionResult(BaseModel):
    """Result of the contradiction pruning process."""
    resolutions: List[ContradictionResolution] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Knowledge Source API Schemas
# ────────────────────────────────────────────────────────────────────
class GitHubSourceCreate(BaseModel):
    """Request body for importing a GitHub repository."""
    repo_url: str = Field(
        ..., min_length=1,
        description="GitHub repository URL (e.g., 'https://github.com/tiangolo/fastapi')",
        examples=["https://github.com/tiangolo/fastapi"],
    )


class PackSourceCreate(BaseModel):
    """Request body for importing a pre-compiled knowledge pack."""
    name: str = Field(
        ..., min_length=1, max_length=512,
        description="Name for this knowledge pack",
        examples=["FastAPI Docs"],
    )
    entities: List[ExtractedEntity] = Field(
        ..., min_length=1,
        description="List of entities to import",
    )
    relationships: List[ExtractedRelationship] = Field(
        default_factory=list,
        description="List of relationships between the entities",
    )


class KnowledgeSourceResponse(BaseModel):
    """Response for a knowledge source operation."""
    id: uuid.UUID
    name: str
    source_type: str
    status: str
    message: str = "Processing in the background."

    model_config = {"from_attributes": True}


# ────────────────────────────────────────────────────────────────────
# Synapse Studio Schemas (VisNetwork formats)
# ────────────────────────────────────────────────────────────────────
class StudioNodeResponse(BaseModel):
    id: str  # VisNetwork expects string IDs
    label: str
    group: str
    confidence: Optional[float] = 0.5
    epistemic_state: Optional[str] = "FACT"
    
class StudioEdgeResponse(BaseModel):
    id: str
    to: str
    from_: str = Field(alias="from")
    label: str
    weight: float
    confidence: Optional[float] = 0.5
    has_contradiction: Optional[bool] = False
    epistemic_state: Optional[str] = "FACT"
    trigger_condition: Optional[dict[str, Any]] = None
    executable_payload: Optional[str] = None
    status: Optional[str] = None

class StudioGraphResponse(BaseModel):
    nodes: List[StudioNodeResponse]
    edges: List[StudioEdgeResponse]

class EdgeTraceResponse(BaseModel):
    relationship_id: uuid.UUID
    matched_memory_id: Optional[uuid.UUID]
    content: Optional[str]
    score: float
    explanation: str

class NamespaceListResponse(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ────────────────────────────────────────────────────────────────────
# Audit Log Schemas (Phase 3 ABAC)
# ────────────────────────────────────────────────────────────────────
class AuditLogResponse(BaseModel):
    """A single audit log entry."""
    id: uuid.UUID
    namespace_id: uuid.UUID
    api_key_id: Optional[uuid.UUID]
    action: str
    entity_name: Optional[str]
    result_count: int
    role_used: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogListResponse(BaseModel):
    """Paginated list of audit log entries."""
    entries: List[AuditLogResponse]
    total: int


# ────────────────────────────────────────────────────────────────────
# Webhook Ingestion Schemas (Phase 1 Ambient Ingestion)
# ────────────────────────────────────────────────────────────────────
class WebhookEventResponse(BaseModel):
    """Response confirming a webhook event was accepted or skipped."""
    status: str  # "accepted" | "skipped" | "challenge"
    connector_type: str
    event_id: str
    event_type: str
    message: str

