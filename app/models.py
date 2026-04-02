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
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_sources.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
    )

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
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    namespace = relationship("Namespace", back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<ApiKey(namespace={self.namespace_id}, active={self.is_active})>"
