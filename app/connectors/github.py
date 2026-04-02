"""
Synapse — GitHub Repository Connector.

Fetches README and architecture files from a GitHub repository,
chunks the content, and runs it through the extraction pipeline
to build a domain knowledge graph tagged with a KnowledgeSource.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Optional
from datetime import datetime, timezone

import httpx
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Entity, KnowledgeSource, Relationship
from app.prompts import EXTRACTION_SYSTEM_PROMPT
from app.schemas import ExtractedEntity, ExtractedRelationship, MemoryExtraction
from app.services import _extract_knowledge, _upsert_entity, _upsert_relationship

logger = logging.getLogger("synapse.connectors.github")

# Files to look for in a repo (in priority order)
ARCHITECTURE_FILES = [
    "README.md",
    "readme.md",
    "ARCHITECTURE.md",
    "docs/README.md",
    "docs/architecture.md",
    "CONTRIBUTING.md",
    "docs/OVERVIEW.md",
    "pyproject.toml",
    "package.json",
]

# Max chunk size in characters (~1500 tokens)
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400


def _parse_github_url(repo_url: str) -> tuple[str, str]:
    """
    Parse a GitHub URL into (owner, repo).
    Supports: https://github.com/owner/repo, github.com/owner/repo, owner/repo
    """
    # Strip trailing slashes and .git
    url = repo_url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # Try to extract owner/repo
    match = re.search(r"(?:github\.com/)?([^/]+)/([^/]+)$", url)
    if not match:
        raise ValueError(f"Cannot parse GitHub URL: {repo_url}")

    return match.group(1), match.group(2)


def _chunk_markdown(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split markdown text into overlapping chunks, preferring to break
    at heading boundaries (## or ###).
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    # Split on markdown headings
    sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)

    current_chunk = ""
    for section in sections:
        if len(current_chunk) + len(section) <= chunk_size:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If a single section is too large, split by paragraphs
            if len(section) > chunk_size:
                paragraphs = section.split("\n\n")
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
            else:
                current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:chunk_size]]


async def _fetch_file_content(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    path: str,
) -> Optional[str]:
    """Fetch raw file content from GitHub's raw content URL."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code == 200:
            return response.text
        return None
    except httpx.HTTPError:
        return None


async def fetch_repo_content(repo_url: str) -> list[dict[str, str]]:
    """
    Fetch architecture-relevant files from a GitHub repo.
    Returns list of {"filename": ..., "content": ...} dicts.
    """
    owner, repo = _parse_github_url(repo_url)
    documents: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for filepath in ARCHITECTURE_FILES:
            content = await _fetch_file_content(client, owner, repo, filepath)
            if content and len(content.strip()) > 50:
                documents.append({
                    "filename": filepath,
                    "content": content,
                })
                logger.info(f"Fetched {filepath} from {owner}/{repo} ({len(content)} chars)")

    if not documents:
        # Try fetching repo description via API as fallback
        async with httpx.AsyncClient(timeout=15.0) as client:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = await client.get(api_url)
            if resp.status_code == 200:
                data = resp.json()
                desc = data.get("description", "")
                topics = data.get("topics", [])
                lang = data.get("language", "")
                fallback = (
                    f"# {data.get('full_name', repo)}\n\n"
                    f"{desc}\n\n"
                    f"Primary language: {lang}\n"
                    f"Topics: {', '.join(topics)}\n"
                )
                documents.append({"filename": "api_metadata", "content": fallback})

    return documents


async def ingest_github_repo(
    namespace_id: uuid.UUID,
    source_id: uuid.UUID,
    repo_url: str,
) -> None:
    """
    Background task: fetch repo content, chunk it, extract entities
    and relationships, and save them tagged with the source_id.
    """
    from app.database import async_session_factory

    async with async_session_factory() as db:
        try:
            # Update source status
            source = await db.get(KnowledgeSource, source_id)
            if source:
                source.status = "ingesting"
                await db.flush()

            # ── Step 1: Fetch repo content ────────────────────────
            documents = await fetch_repo_content(repo_url)
            if not documents:
                logger.warning(f"No content found for {repo_url}")
                if source:
                    source.status = "failed"
                    await db.commit()
                return

            logger.info(
                f"Fetched {len(documents)} files from {repo_url}. "
                f"Chunking and extracting..."
            )

            # ── Step 2: Chunk and extract from each document ──────
            all_entities: dict[str, ExtractedEntity] = {}
            all_relationships: list[ExtractedRelationship] = []

            for doc in documents:
                chunks = _chunk_markdown(doc["content"])
                logger.info(
                    f"  {doc['filename']}: {len(chunks)} chunks"
                )

                for chunk in chunks:
                    extraction = await _extract_knowledge(
                        f"[Source: {doc['filename']}]\n\n{chunk}"
                    )

                    for ent in extraction.entities:
                        # Deduplicate across chunks
                        all_entities[ent.name] = ent

                    all_relationships.extend(extraction.relationships)

            logger.info(
                f"Extraction complete: {len(all_entities)} entities, "
                f"{len(all_relationships)} relationships"
            )

            # ── Step 3: Persist entities with source_id ───────────
            entity_map: dict[str, Entity] = {}
            for ext_entity in all_entities.values():
                entity = await _upsert_entity_with_source(
                    db, namespace_id, ext_entity, source_id
                )
                entity_map[ext_entity.name] = entity

            # ── Step 4: Persist relationships with source_id ──────
            for ext_rel in all_relationships:
                await _upsert_relationship_with_source(
                    db, entity_map, ext_rel, source_id
                )

            # Update source status and sync time
            if source:
                source.status = "ready"
                source.last_synced_at = datetime.now(timezone.utc)

            await db.commit()
            logger.info(f"GitHub ingestion complete for {repo_url}")

        except Exception as e:
            await db.rollback()
            # Try to mark source as failed
            try:
                async with async_session_factory() as db2:
                    source = await db2.get(KnowledgeSource, source_id)
                    if source:
                        source.status = "failed"
                        await db2.commit()
            except Exception:
                pass
            logger.error(f"GitHub ingestion failed for {repo_url}: {e}", exc_info=True)
            raise


async def _upsert_entity_with_source(
    db: AsyncSession,
    namespace_id: uuid.UUID,
    ext_entity: ExtractedEntity,
    source_id: uuid.UUID,
) -> Entity:
    """Insert entity with source_id if it doesn't exist, otherwise return existing."""
    stmt = select(Entity).where(
        and_(
            Entity.namespace_id == namespace_id,
            Entity.name == ext_entity.name,
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        return existing

    entity = Entity(
        namespace_id=namespace_id,
        name=ext_entity.name,
        entity_type=ext_entity.entity_type,
        source_id=source_id,
    )
    db.add(entity)
    await db.flush()
    return entity


async def _upsert_relationship_with_source(
    db: AsyncSession,
    entity_map: dict[str, Entity],
    ext_rel: ExtractedRelationship,
    source_id: uuid.UUID,
) -> Optional[Relationship]:
    """Insert relationship with source_id if not duplicate."""
    source_ent = entity_map.get(ext_rel.source)
    target_ent = entity_map.get(ext_rel.target)

    if not source_ent or not target_ent:
        return None

    stmt = select(Relationship).where(
        and_(
            Relationship.source_entity_id == source_ent.id,
            Relationship.target_entity_id == target_ent.id,
            Relationship.relation_type == ext_rel.relation,
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.weight = existing.weight + 1.0
        return existing

    rel = Relationship(
        source_entity_id=source_ent.id,
        target_entity_id=target_ent.id,
        relation_type=ext_rel.relation,
        weight=1.0,
        source_id=source_id,
    )
    db.add(rel)
    await db.flush()
    return rel
