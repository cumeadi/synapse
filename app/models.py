"""
Synapse — SQLAlchemy ORM Models.

Defines the full schema: Namespace, Memory (vector store), Entity & Relationship
(knowledge graph), KnowledgeSource (imported graph tracking), and ApiKey (authentication).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all Synapse models."""
    pass


# ────────────────────────────────────────────────────────────────────
# Namespace — sandbox for a specific agent / user
# ────────────────────────────────────────────────────────────────────
class Namespace(Base):
    __tablename__ = "namespaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    memories = relationship("Memory", back_populates="namespace", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="namespace", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="namespace", cascade="all, delete-orphan")
    knowledge_sources = relationship("KnowledgeSource", back_populates="namespace", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Namespace(name={self.name!r})>"


# ────────────────────────────────────────────────────────────────────
# Memory — semantic vector store
# ────────────────────────────────────────────────────────────────────
class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    embedding = Column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    namespace = relationship("Namespace", back_populates="memories")

    def __repr__(self) -> str:
        return f"<Memory(id={self.id}, content={self.content[:40]!r}...)>"


# HNSW index for fast cosine similarity search
Index(
    "ix_memories_embedding_hnsw",
    Memory.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# ────────────────────────────────────────────────────────────────────
# KnowledgeSource — tracks imported graph sources
# ────────────────────────────────────────────────────────────────────
class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # "static_pack", "github_repo"
    status: Mapped[str] = mapped_column(
        String(64), default="pending", nullable=False
    )  # "pending", "ingesting", "grafting", "ready", "failed"
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    namespace = relationship("Namespace", back_populates="knowledge_sources")
    entities = relationship("Entity", back_populates="source", cascade="all, delete-orphan")
    relationships = relationship(
        "Relationship", back_populates="source", cascade="all, delete-orphan",
        foreign_keys="Relationship.source_id",
    )

    def __repr__(self) -> str:
        return f"<KnowledgeSource(name={self.name!r}, type={self.source_type!r}, status={self.status!r})>"


# ────────────────────────────────────────────────────────────────────
# Entity — knowledge graph node
# ────────────────────────────────────────────────────────────────────
class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("namespace_id", "name", name="uq_entity_namespace_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False)
    visibility_label: Mapped[str] = mapped_column(
        String(32), default="public", nullable=False
    )  # ABAC label: public | internal | confidential | restricted
    observation_count: Mapped[int] = mapped_column(
        default=1, nullable=False
    )  # How many times any source has mentioned this entity
    confidence: Mapped[float] = mapped_column(
        Float, default=0.5, nullable=False
    )  # Computed confidence score [0, 1] — updated on every corroboration
    epistemic_state: Mapped[str] = mapped_column(
        String(16), default="FACT", nullable=False
    )  # 'FACT' for objective, 'TAKE' for subjective
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_sources.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
    )

    # Relationships
    namespace = relationship("Namespace", back_populates="entities")
    source = relationship("KnowledgeSource", back_populates="entities")
    outgoing_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    incoming_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Entity(name={self.name!r}, type={self.entity_type!r})>"


# ────────────────────────────────────────────────────────────────────
# Relationship — knowledge graph edge
# ────────────────────────────────────────────────────────────────────
class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint(
            "source_entity_id", "target_entity_id", "relation_type",
            name="uq_relationship_source_target_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    visibility_label: Mapped[str] = mapped_column(
        String(32), default="public", nullable=False
    )  # ABAC label: public | internal | confidential | restricted
    source_diversity_count: Mapped[int] = mapped_column(
        default=1, nullable=False
    )  # Distinct ingestion sources that have observed this relationship
    confidence: Mapped[float] = mapped_column(
        Float, default=0.5, nullable=False
    )  # Computed confidence score [0, 1]
    has_contradiction: Mapped[bool] = mapped_column(
        default=False, nullable=False
    )  # True if a conflicting relation_type exists between the same entity pair
    last_context_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, default=None
    )  # Used for diversity tracking (can be memory_id or source_id)
    epistemic_state: Mapped[str] = mapped_column(
        String(16), default="FACT", nullable=False
    )  # 'FACT' for objective, 'TAKE' for subjective
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_reinforced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    weight_decay_rate: Mapped[float] = mapped_column(
        Float, default=0.01, nullable=False
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_sources.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
    )

    @property
    def decayed_weight(self) -> float:
        """Dynamically compute the edge weight based on its last reinforcement time."""
        import math
        now = datetime.now(timezone.utc)
        last_reinforced = self.last_reinforced_at
        if last_reinforced.tzinfo is None:
            last_reinforced = last_reinforced.replace(tzinfo=timezone.utc)
        days_since = (now - last_reinforced).total_seconds() / 86400.0
        decayed = self.weight * math.exp(-self.weight_decay_rate * days_since)
        return max(decayed, 0.0)

    # Relationships
    source_entity = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="outgoing_relationships"
    )
    target_entity = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="incoming_relationships"
    )
    source = relationship(
        "KnowledgeSource", back_populates="relationships",
        foreign_keys=[source_id],
    )

    def __repr__(self) -> str:
        return f"<Relationship({self.relation_type})>"


# ────────────────────────────────────────────────────────────────────
# ApiKey — authentication for namespace access
# ────────────────────────────────────────────────────────────────────
class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False
    )
    description: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), default="internal", nullable=False
    )  # ABAC role: public | internal | confidential | restricted
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    namespace = relationship("Namespace", back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<ApiKey(namespace={self.namespace_id}, role={self.role}, active={self.is_active})>"


# ────────────────────────────────────────────────────────────────────
# AuditLog — immutable access trail for ABAC compliance
# ────────────────────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, default=None
    )  # Null for master-key or dev-mode callers
    action: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # e.g. "graph_search", "hybrid_search", "studio_view"
    entity_name: Mapped[str | None] = mapped_column(
        String(255), nullable=True, default=None
    )
    result_count: Mapped[int] = mapped_column(default=0, nullable=False)
    role_used: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<AuditLog(action={self.action!r}, role={self.role_used!r}, ns={self.namespace_id})>"


# ────────────────────────────────────────────────────────────────────
# WebhookEvent — idempotency store for ambient ingestion events
# ────────────────────────────────────────────────────────────────────
class WebhookEvent(Base):
    __tablename__ = "webhook_events"
    __table_args__ = (
        UniqueConstraint(
            "namespace_id", "connector_type", "event_id",
            name="uq_webhook_event_namespace_connector_event",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    connector_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "slack" | "jira" | "github"
    event_id: Mapped[str] = mapped_column(
        String(256), nullable=False
    )  # Source system's stable unique identifier
    event_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # e.g. "message", "issue_created", "push"
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False
    )  # "pending" | "processed" | "skipped" | "failed"
    normalized_text: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )  # The plain-text description passed to the LLM extractor
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return (
            f"<WebhookEvent(connector={self.connector_type!r}, "
            f"type={self.event_type!r}, status={self.status!r})>"
        )
