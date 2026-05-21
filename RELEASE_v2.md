# Synapse Enterprise v2.0.0 Release Notes

Welcome to Synapse Enterprise v2.0.0! This release marks the official graduation of Synapse from a simple active memory layer to a fully-fledged, passive, ambient cognitive infrastructure capable of surviving high-noise, highly-regulated enterprise environments.

This release encompasses the completion of all 5 Phases of the Enterprise Evolution roadmap, introducing features designed to drastically improve ingestion bandwidth, control token costs, strictly govern data visibility, and prevent LLM hallucinations.

## 🚀 Key Features

### 1. Ambient Data Ingestion (The Subconscious)
Synapse now listens to your organization rather than waiting to be told.
- **Webhook Connectors:** Plugs directly into Slack, Jira, and GitHub.
- **Idempotency:** Re-delivered webhook events are silently dropped, preventing duplicate graph bloat.
- **Background Processing:** Events are processed entirely asynchronously.

### 2. Temporal Edge Mechanics (Knowledge Half-Life)
Not all knowledge is permanent. 
- **Decay Engine:** Graph edge weights decay exponentially over time (`weight = weight * e^(-decay * time)`).
- **Corroboration Reinforcement:** When multiple sources observe the same fact, the `last_reinforced_at` timestamp is bumped and the weight recovers, forming a robust forgetting curve.

### 3. Attribute-Based Access Control (ABAC)
Enterprise knowledge is strictly governed.
- **Data Governance:** Every entity and relationship is tagged with a `visibility_label` (`public`, `internal`, `confidential`, `restricted`).
- **Caller Filtering:** Graph queries and MCP context retrieval dynamically filter nodes and edges in SQL based on the caller's authorized role.

### 4. Knowledge Confidence Scoring
Synapse evaluates the truthfulness of facts by measuring corroboration.
- **Source Diversity:** Facts observed by multiple independent sources accrue confidence asymptotically towards 1.0. 
- **Contradiction Penalties:** If contradictory facts are logged (e.g., LIKES vs DISLIKES), the system flags them with `has_contradiction = True` and penalizes their confidence by 30%.
- **Noise Filtering:** Retrieve only highly corroborated facts using the new `min_confidence` parameter in the APIs.

### 5. GBrain Mechanics (Takes vs. Facts & Heuristics)
To further eliminate hallucinations and drastically reduce costs:
- **Epistemic Humility:** The LLM now categorizes subjective opinions as `TAKE` and objective events as `FACT`, allowing consumers to weigh the objectivity of data.
- **Zero-LLM Extraction (Heuristics):** Highly structured payloads (Jira issue creation, GitHub PR merges) are now intercepted and parsed via deterministic Regex, bypassing the expensive LLM.
- **The "Dream Cycle":** Heavy maintenance (permanent decay application, contradiction sweeps) has been moved out of the hot path into an asynchronous, offline worker (`dream_worker.py`).

## 🛠 Upgrading to v2.0.0

1. **Database Migrations:** The schema has evolved significantly. When upgrading, the `app/database.py` initialization will automatically run `ALTER TABLE` to append the new columns (`has_contradiction`, `confidence`, `epistemic_state`, `last_context_id`, etc.).
2. **Cron Jobs:** To benefit from the Dream Cycle, configure your infrastructure (e.g., Kubernetes CronJob or Celery) to execute `app/dream_worker.py` on a nightly schedule.
