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
* **The "Cerebellum" Engine:** Collapses recurring agent workflows into zero-token, low-latency **Procedural Reflexes** (Standing Orders) governed by safe Human-in-the-Loop (HITL) states.
* **Self-Healing Updates:** When new information contradicts old logs, Synapse automatically prunes outdated nodes and rewires relationships.
* **Native MCP Server:** Out-of-the-box Model Context Protocol support. Plug it directly into Claude Desktop, Cursor, or Windsurf in seconds.
* **Synapse Studio:** A built-in, local visualizer. Stop interacting with your AI's memory via a blind CLI. See the graph, click the nodes, and prune hallucinations manually.

---

## ⚡ The "Cerebellum" Engine (Procedural Memory)

Traditional memory models only map *what* an agent knows (`[User] -> (LIKES) -> [Python]`). However, agents still must spend expensive LLM tokens retrieving these facts, reasoning over them, and choosing actions on every single turn.

The **Cerebellum Engine** solves this by expanding Synapse's background sleep cycle to monitor recurring agent workflows, identify repetitive multi-hop reasoning chains, and collapse them into **Procedural Reflexes** (deterministic standing orders).

### ⚙️ How it Works:
1. **Procedural Compression:** Background workers analyze chronological history and compile repetitive workflows into `PROPOSED` reflexes (e.g. *"When a PR is pushed to frontend, format the code"*).
2. **Shadow Reflexes (HITL Governance):** Newly consolidated reflexes default to `PROPOSED`. They act as a shadow layer: logging telemetry (`reflex_shadow_triggered`) without intercepting queries. Developers review and promote them to `ACTIVE` via the Synapse Studio UI or REST API.
3. **Parameter Template Substitution:** Payloads accept variables (e.g., `Run tests in {{repo}} for {{query}}`), dynamically rendering substituting values from active metadata filters.
4. **Self-Healing & Reversion:** If a reflex fails, calling `report_reflex_failure` instantly freezes (PAUSES) the reflex, drops confidence to `0.01` (floor), and reverts it to a standard declarative `FACT` so standard agent reasoning takes back control instantly.

### 🛡️ Studio Kill Switch
Reflexes are fully inspectable. Synapse Studio features a dedicated **Cerebellum Card** inside the relationship inspector, displaying status badges and one-click buttons to **Activate ⚡️**, **Pause ⏸️**, or **Delete 🗑️** standing orders instantly.

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

    # 3. Trigger the Sleep Cycle (Memory Consolidation)
    # Merges duplicate entities and prunes contradictions in the background
    await client.sleep(namespace="dev_agent")

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
