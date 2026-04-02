# API & SDK Reference

Synapse is built API-first. Every action you can perform in Synapse Studio can be triggered programmatically via our REST API. For convenience in Python environments, we also provide the `SynapseClient` SDK.

---

## ⚡ Default Base URL
If you are running the backend locally:
```text
http://localhost:8000
```

*Note: All endpoints handling memory modification or graph traversal require a valid API key passed in the `X-API-Key` header.*

---

## 🔐 Authentication

Synapse supports per-namespace API keys as well as a global master key. 
Pass your key in the header of all requests (except the health check).

```http
X-API-Key: syn_abc123...
```

---

## 🏗️ Namespaces

Namespaces are isolated memory sandboxes. You can use separate namespaces for different agents, users, or projects.

### Create a Namespace

**Endpoint:** `POST /namespaces/`

Creates a new namespace and returns an access key scoped strictly to this namespace.

**Request Body:**
```json
{
  "name": "agent-alpha"
}
```

**Response:** `201 Created`
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "agent-alpha",
  "created_at": "2026-04-02T10:00:00Z",
  "api_key": "syn_MOZbn... (Only returned once!)"
}
```

### List Namespaces

**Endpoint:** `GET /namespaces/`

Returns all namespaces. *(Requires Master Key)*

---

## 🧠 Memory Ingestion

### Store a Memory

**Endpoint:** `POST /namespaces/{id}/memories/`

Ingests raw text. In the background, Synapse handles chunking (if necessary), vector embedding, and structured entity/relationship extraction via the configured LLM.

**Request Body:**
```json
{
  "content": "I am migrating the backend to FastAPI. The user dislikes Java.",
  "metadata": {"source": "slack", "user_id": "u_987"}
}
```

**Response:** `202 Accepted`
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174001",
  "status": "accepted",
  "message": "Memory ingestion is processing in the background."
}
```

---

## 🔍 Retrieval & Search

### Hybrid Search (Vector + Metadata)

**Endpoint:** `POST /namespaces/{id}/search/hybrid`

Performs a semantic cosine-similarity search across vector embeddings, optionally pre-filtered by JSONB metadata constraints.

**Request Body:**
```json
{
  "query": "What backend framework are they migrating to?",
  "metadata_filter": {"source": "slack"},
  "top_k": 5
}
```

**Response:** `200 OK`
```json
{
  "query": "What backend framework...",
  "results": [
    {
      "id": "...",
      "content": "I am migrating the backend to FastAPI. The user dislikes Java.",
      "metadata": {"source": "slack", "user_id": "u_987"},
      "score": 0.8921,
      "created_at": "2026-04-02T10:05:00Z"
    }
  ],
  "total": 1
}
```

### Relational Graph Traversal

**Endpoint:** `GET /namespaces/{id}/search/graph`

Queries the knowledge graph for a specific entity and returns its neighborhood up to `N` hops deep.

**Query Parameters:**
*   `entity_name` (required): The precise name of the entity.
*   `depth` (optional): Hops to traverse. Defaults to 1. Max 3.

**Response:** `200 OK`
(Returns arrays of `entities` and `relationships` detailing the graph structure around the center node).

---

## 📦 Synapse Nodes (Knowledge Sources)

### Ingest GitHub Repository

**Endpoint:** `POST /namespaces/{id}/sources/github`

Pulls a repository's overarching architecture (`README.md`, `pyproject.toml`, `package.json`, etc.), processes the texts into domain entities, and uses the Auto-Grafting engine to bridge connections to your agent's existing personal memory graph.

**Request Body:**
```json
{
  "repo_url": "https://github.com/tiangolo/fastapi"
}
```

### Ingest Static Pack

**Endpoint:** `POST /namespaces/{id}/sources/pack`

Identical to GitHub ingestion, but allows you to bypass the external fetch and directly feed an array of pre-defined graph nodes and edges into a `KnowledgeSource` bucket.

---

## 💻 Python SDK Quick Reference

If you are using the `synapse-core` Python package (as advertised in our quickstart), here is a brief overview:

```python
from synapse import SynapseClient

client = SynapseClient(db_url="...")

# Ingestion
await client.ingest(namespace="dev_agent", text="...")

# Hybrid Search 
results = await client.search(namespace="dev_agent", query="...")

# Graph Traversal
neighborhood = await client.graph_traversal(namespace="dev_agent", entity="FastAPI", depth=2)
```
