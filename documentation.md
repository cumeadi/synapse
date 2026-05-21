# Synapse Documentation

Welcome to the official documentation for **Synapse**, the open-source cognitive memory engine for AI agents. 

Synapse moves beyond simple vector search by combining **hybrid vector search** with an **automated knowledge graph**. When your agent ingests information, Synapse doesn't just store text—it reasons about it, extracting entities and mapping their relationships to build a long-term, relational memory.

---

## 1. Core Concepts

### Vectors vs. Graphs
Most AI memory systems use **Vector Databases** (like storing raw chat logs and finding similar sentences via cosine distance). This is great for fuzzy matching but terrible for relational logic (e.g., "What did the developer who wrote the auth service say about my bug?").

Synapse uses a **Hybrid Approach**:
1. **Semantic Search:** We embed every memory into a 1536-dimensional vector using `pgvector` for fast semantic retrieval.
2. **Relational Graph:** We pass the text through an LLM to extract entities (Nodes) and relationships (Edges). 

If a user says *"I hate Java,"* Synapse stores the raw text vector, but explicitly maps: `[User] -> [DISLIKES] -> [Java]`.

### Namespaces
A **Namespace** is an isolated sandbox. You should create a separate namespace for every distinct agent, user, or project. Graphs and vectors are strictly partitioned by namespace.

### The Sleep Cycle
Agents shouldn't hoard data forever. Synapse features an automated **Sleep Cycle** that you can trigger to optimize the graph:
- **Entity Disambiguation:** Merges duplicate or synonymous entities (e.g., "AWS" and "Amazon Web Services").
- **Contradiction Pruning:** Resolves logical inconsistencies (e.g., if an agent learns a user likes Python, but previously learned they hated it, the older relationship is pruned).

---

## 2. Getting Started

### Starting the Engine
Synapse requires a PostgreSQL database with the `pgvector` extension. The easiest way to run it is via Docker:

```bash
# Clone the repository
git clone https://github.com/cumeadi/synapse.git
cd synapse

# Spin up Postgres + pgvector
docker-compose up -d

# Install dependencies and run the API
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 3. Python SDK (`synapse-core`)

The Synapse Python SDK allows you to interact with the engine using clean, asynchronous methods.

### Initialization
```python
import asyncio
from synapse import SynapseClient

async def main():
    # Connect to the local Synapse engine
    client = SynapseClient(db_url="postgresql://user:pass@localhost:5432/synapse")
```

### Ingestion
Feed raw text to the engine. It will automatically generate embeddings and extract graph nodes in the background.
```python
    # Returns 202 Accepted instantly; processing happens in the background.
    await client.ingest(
        namespace="dev_agent",
        text="I'm migrating our backend to FastAPI. Do not use Pydantic v1."
    )
```

### Hybrid Search
Retrieve context using both semantic similarity and metadata filtering.
```python
    context = await client.search(
        namespace="dev_agent", 
        query="What framework are we using?",
        top_k=5
    )
    # Returns matched memories and their graph paths
```

### Graph Traversal
Traverse the relational graph starting from a specific entity node.
```python
    graph = await client.graph_traversal(
        namespace="dev_agent", 
        entity="FastAPI", 
        depth=2 # Number of hops to traverse
    )
```

### Triggering the Sleep Cycle
Run background maintenance to merge synonyms and resolve contradictions.
```python
    await client.sleep(namespace="dev_agent")
```

---

## 4. Synapse Studio

Synapse Studio is a local visualizer that lets you see exactly what your agent is thinking. Stop interacting with your AI's memory via a blind CLI!

1. Start the API server (`uvicorn app.main:app`).
2. Navigate to `http://localhost:8000/studio` in your browser.
3. Select your namespace from the dropdown.
4. **Interact:** Click on nodes, view connection weights, and manually delete hallucinated entities or relationships directly from the UI.

---

## 5. Model Context Protocol (MCP) Support

Synapse ships with native Model Context Protocol (MCP) support. This means you can plug your Synapse memory engine directly into AI IDEs like **Claude Desktop**, **Cursor**, or **Windsurf**.

The MCP server is located at `app/mcp_server.py`. Once configured in your IDE, your AI assistant can automatically query Synapse for context without you needing to paste code snippets.

---

## 6. Architecture & Privacy

Synapse is designed to be fully local and private. 
- **No Vendor Lock-in:** You own your data. It lives in your Postgres instance.
- **Bring Your Own LLM:** Synapse uses `litellm` under the hood. By changing your `.env` file, you can swap out OpenAI/Anthropic for local models via Ollama. 

```env
# .env example for local Ollama usage
LLM_MODEL=ollama/llama3
EMBEDDING_MODEL=ollama/nomic-embed-text
```
