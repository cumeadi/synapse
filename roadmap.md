# Synapse Enterprise Roadmap

This document outlines the strategic vision for Synapse Enterprise as we move towards v3 and beyond. Having successfully delivered the foundational Graph and Ambient Ingestion mechanics in v1 and v2, our next objective is to maximize stickiness, commercial value, and enterprise readiness.

## Core Pillars of the v3 Evolution

Our goal is to make Synapse the indispensable "System of Record for Human Context" for Fortune 500 organizations. To achieve this, we will execute across four key pillars.

---

### Phase 6: The "System of Record" Integrations
*The value of a knowledge graph explodes when we overlay structural organizational data on top of organic behavioral data.*

- **HRIS Integration (Workday / Gusto):** Automatically map the corporate org chart (Manager -> Direct Report) into the graph via `REPORTS_TO` edges. This immediately provides the graph with the context of *who* has the authority to make certain decisions.
- **CRM Exhaust (Salesforce / HubSpot):** Ingest Account and Opportunity activity. This enables Synapse to connect engineering efforts to revenue (e.g., understanding when a specific Jira ticket is blocking a high-value Salesforce opportunity).

### Phase 7: Proactive Alerting & "Cognitive Triggers"
*An enterprise product must proactively deliver value, rather than waiting to be queried.*

- **Complex Event Processing:** Implement a rules engine that monitors the graph in real-time. If Synapse detects a dangerous contradiction (e.g., "DevOps reports the deployment is stable" vs "SRE Slack channel reports production errors"), it proactively triggers an alert to the incident response team.
- **At-Risk Expertise Tracking:** Monitor the graph for central, load-bearing nodes (employees who are the sole experts on critical legacy systems). If HRIS signals that an employee with high graph centrality is offboarding, automatically flag their owned repositories and projects for risk mitigation.

### Phase 8: Enterprise Trust & Compliance
*To clear procurement and InfoSec reviews at enterprise scale, we must guarantee data sovereignty, security, and auditability.*

- **SAML / SSO Integration:** Connect the Synapse Studio UI and APIs to Okta / Entra ID. This allows us to tie our existing Attribute-Based Access Control (ABAC) policies directly to Okta security groups.
- **Role-Based Audit Logging:** Automatically log an immutable audit trail whenever a highly confidential or restricted node is accessed by an LLM query.
- **Automated Data Redaction (DLP):** Introduce a pre-processing pipeline on webhook ingestion that scrubs PII (Credit Cards, SSNs, personal phone numbers) before the text ever reaches the LLM extraction phase.

### Phase 9: The "Knowledge Portal" Dashboard
*The end-user of Synapse should not just be AI agents, but also engineering leaders and product managers.*

- **Executive Graph Visualizer:** Upgrade the Synapse Studio UI into a tailored dashboard for non-technical managers.
- **Query Templates:** Provide one-click answers to high-value executive questions:
  - *"Who are our top experts in Kubernetes?"*
  - *"What projects are currently bottlenecked by the Core Platform team?"*
  - *"Which new hires are ramping up the fastest based on organic contributions?"*
