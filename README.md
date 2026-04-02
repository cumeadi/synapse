# 🧠 Synapse 

**The open-source cognitive memory engine for AI agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Semantic search is just `Ctrl+F` with math. When you dump raw chat logs into a standard vector database, your agent doesn't actually *learn*—it just hoards text. 

Synapse is different. It doesn't just store data; it reasons about it. By combining **hybrid vector search** with an **automated knowledge graph** and **background memory consolidation**, Synapse gives your agents an actual brain, not just a bigger filing cabinet.

![Synapse Studio Demo](./docs/assets/studio-demo.gif)
*(Above: Synapse Studio mapping an agent's knowledge graph in real-time).*

---

## 🔥 Why Synapse?

* **Relational Graph Memory:** Synapse extracts entities and relationships from raw text during ingestion. If a user says "I hate Java," it explicitly maps `[User] -> [DISLIKES] -> [Java]`. No more fuzzy cosine-distance guessing.
* **The "Sleep" Cycle:** Background workers periodically synthesize sprawling conversational history into high-value "core principles," archiving the noise.
* **Self-Healing Updates:** When new information contradicts old logs, Synapse automatically prunes outdated nodes and rewires relationships.
* **Native MCP Server:** Out-of-the-box Model Context Protocol support. Plug it directly into Claude Desktop, Cursor, or Windsurf in seconds.
* **Synapse Studio:** A built-in, local visualizer. Stop interacting with your AI's memory via a blind CLI. See the graph, click the nodes, and prune hallucinations manually.

---

## ⚡ 60-Second Quickstart

**1. Spin up the Database (Postgres + pgvector)**
```bash
docker-compose up -d
```

**2. Install Synapse**
```bash
pip install synapse-core
```

**3. Ingest & Retrieve (Python)**
```python
import asyncio
from synapse import SynapseClient

async def main():
    # Connect to the local engine
    client = SynapseClient(db_url="postgresql://user:pass@localhost:5432/synapse")
    
    # 1. Ingest (Automatically generates vectors AND graph nodes)
    await client.ingest(
        namespace="dev_agent",
        text="I'm migrating our backend to FastAPI. Do not use Pydantic v1."
    )
    
    # 2. Retrieve Hybrid Context
    context = await client.search(
        namespace="dev_agent", 
        query="What framework are we using?"
    )
    print(context) 
    # Returns the vector match + the exact graph path: [User] -> [MIGRATING_TO] -> [FastAPI]

asyncio.run(main())
```

**4. Launch Synapse Studio**
See what your agent just learned:
```bash
synapse studio --port 8000
```
Go to `http://localhost:8000/studio` to view the living graph.

---

## 📚 Documentation
Ready to build something complex? Check out the full docs:

- [Core Concepts: Vectors vs. Graphs](./docs/core-concepts.md)
- [API & SDK Reference](./docs/api-reference.md)
- [Synapse Nodes: Importing GitHub Repos](./docs/synapse-nodes.md)
- [Connecting MCP to Cursor & Claude](./docs/mcp-setup.md)

---

## 🛡️ Architecture & Privacy
Synapse runs locally by default. No vendor lock-in. No piping your proprietary architecture through a hosted cloud service. Bring your own embedding model via `litellm` (OpenAI, Anthropic, or local Ollama).

---

## 🤝 Contributing
PRs are welcome. See `CONTRIBUTING.md` for our local development setup and testing guidelines.
