"""
Synapse — Database connection and session management.

Provides async SQLAlchemy engine, session factory, and initialization
for PostgreSQL with the pgvector extension.
"""

import os
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text
from dotenv import load_dotenv
import logging

logger = logging.getLogger("synapse.database")

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://synapse:synapse_secret@localhost:5433/synapse",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """FastAPI dependency — yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create pgvector extension and all tables on startup."""
    from app.models import Base  # noqa: F811 — deferred to avoid circular imports

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        # Phase 2: Dynamic migration for relationship temporal fields
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "last_reinforced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "weight_decay_rate DOUBLE PRECISION DEFAULT 0.01"
        ))

        # Phase 3: ABAC — visibility labels on graph nodes/edges, role on API keys
        await conn.execute(text(
            "ALTER TABLE entities ADD COLUMN IF NOT EXISTS "
            "visibility_label VARCHAR(32) DEFAULT 'public' NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "visibility_label VARCHAR(32) DEFAULT 'public' NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS "
            "role VARCHAR(32) DEFAULT 'internal' NOT NULL"
        ))

        # Phase 4: Knowledge Confidence Scoring
        await conn.execute(text(
            "ALTER TABLE entities ADD COLUMN IF NOT EXISTS "
            "observation_count INTEGER DEFAULT 1 NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE entities ADD COLUMN IF NOT EXISTS "
            "confidence DOUBLE PRECISION DEFAULT 0.5 NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "source_diversity_count INTEGER DEFAULT 1 NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "confidence DOUBLE PRECISION DEFAULT 0.5 NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "has_contradiction BOOLEAN DEFAULT FALSE NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "last_context_id UUID"
        ))
        await conn.execute(text(
            "ALTER TABLE entities ADD COLUMN IF NOT EXISTS "
            "epistemic_state VARCHAR(16) DEFAULT 'FACT' NOT NULL"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "epistemic_state VARCHAR(16) DEFAULT 'FACT' NOT NULL"
        ))

        # Phase 5: Cerebellum Engine — Reflex Edges
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "trigger_condition JSONB"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "executable_payload TEXT"
        ))
        await conn.execute(text(
            "ALTER TABLE relationships ADD COLUMN IF NOT EXISTS "
            "status VARCHAR(32) DEFAULT NULL"
        ))

    logger.info("Database initialized successfully.")
