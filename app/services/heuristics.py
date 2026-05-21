import re
from typing import Optional

from app.schemas import MemoryExtraction, ExtractedEntity, ExtractedRelationship

def run_heuristics(content: str) -> Optional[MemoryExtraction]:
    """
    Zero-LLM extraction pipeline for structured event payloads.
    Matches normalized webhook strings and extracts deterministic facts.
    Returns a MemoryExtraction if a heuristic matches, else None.
    """
    # ── 1. Jira: Issue Created ───────────────────────────────────────
    # Format: "Alice Chen created Jira issue ENG-137: 'Implement OAuth2 token refresh' (type=Story, status=To Do, assignee=Bob Kim)"
    m_jira_created = re.match(r"^(.+?) created Jira issue ([A-Z0-9\-]+):.*assignee=(.+?)\)", content)
    if m_jira_created:
        creator = m_jira_created.group(1).strip()
        issue = m_jira_created.group(2).strip()
        assignee = m_jira_created.group(3).strip()
        
        entities = [
            ExtractedEntity(name=creator, entity_type="Person", epistemic_state="FACT"),
            ExtractedEntity(name=issue, entity_type="Ticket", epistemic_state="FACT"),
        ]
        relationships = [
            ExtractedRelationship(source=creator, target=issue, relation="CREATED", epistemic_state="FACT"),
        ]
        if assignee and assignee.lower() != "unassigned":
            entities.append(ExtractedEntity(name=assignee, entity_type="Person", epistemic_state="FACT"))
            relationships.append(ExtractedRelationship(source=assignee, target=issue, relation="ASSIGNED_TO", epistemic_state="FACT"))
            
        return MemoryExtraction(entities=entities, relationships=relationships)

    # ── 2. Jira: Issue Comment ───────────────────────────────────────
    # Format: "Carol Davis commented on Jira issue ENG-137: 'I'll pick this up in the next sprint.'"
    m_jira_comment = re.match(r"^(.+?) commented on Jira issue ([A-Z0-9\-]+):", content)
    if m_jira_comment:
        commenter = m_jira_comment.group(1).strip()
        issue = m_jira_comment.group(2).strip()
        
        return MemoryExtraction(
            entities=[
                ExtractedEntity(name=commenter, entity_type="Person", epistemic_state="FACT"),
                ExtractedEntity(name=issue, entity_type="Ticket", epistemic_state="FACT"),
            ],
            relationships=[
                ExtractedRelationship(source=commenter, target=issue, relation="COMMENTED_ON", epistemic_state="FACT"),
            ]
        )

    # ── 3. GitHub: Pull Request Opened/Merged ─────────────────────────
    # Format: "bob opened pull request #42 in acme/payments: 'Refactor...'"
    # Format: "bob merged pull request #42 in acme/payments: 'Refactor...'"
    m_gh_pr = re.match(r"^(.+?) (opened|merged) pull request #(\d+) in ([a-zA-Z0-9_\-\/]+):", content)
    if m_gh_pr:
        user = m_gh_pr.group(1).strip()
        action = m_gh_pr.group(2).strip().upper()
        pr_number = m_gh_pr.group(3).strip()
        repo = m_gh_pr.group(4).strip()
        
        pr_name = f"{repo}#{pr_number}"
        
        return MemoryExtraction(
            entities=[
                ExtractedEntity(name=user, entity_type="Person", epistemic_state="FACT"),
                ExtractedEntity(name=pr_name, entity_type="PullRequest", epistemic_state="FACT"),
                ExtractedEntity(name=repo, entity_type="Repository", epistemic_state="FACT"),
            ],
            relationships=[
                ExtractedRelationship(source=user, target=pr_name, relation=action, epistemic_state="FACT"),
                ExtractedRelationship(source=pr_name, target=repo, relation="BELONGS_TO", epistemic_state="FACT"),
            ]
        )

    # ── 4. GitHub: Push ──────────────────────────────────────────────
    # Format: "alice pushed 2 commits to acme/payments/main: 'Fix...'"
    m_gh_push = re.match(r"^(.+?) pushed \d+ commits? to ([a-zA-Z0-9_\-\/]+):", content)
    if m_gh_push:
        user = m_gh_push.group(1).strip()
        branch_path = m_gh_push.group(2).strip() # e.g. acme/payments/main
        
        # Split branch from repo if possible
        parts = branch_path.rsplit("/", 1)
        if len(parts) == 2:
            repo, branch = parts
        else:
            repo = branch_path
            branch = "unknown"
            
        return MemoryExtraction(
            entities=[
                ExtractedEntity(name=user, entity_type="Person", epistemic_state="FACT"),
                ExtractedEntity(name=repo, entity_type="Repository", epistemic_state="FACT"),
            ],
            relationships=[
                ExtractedRelationship(source=user, target=repo, relation="COMMITTED_TO", epistemic_state="FACT"),
            ]
        )

    # ── 5. GitHub: Issue Opened ──────────────────────────────────────
    # Format: "carol opened issue #99 in acme/payments: 'Payment gateway...'"
    m_gh_issue = re.match(r"^(.+?) opened issue #(\d+) in ([a-zA-Z0-9_\-\/]+):", content)
    if m_gh_issue:
        user = m_gh_issue.group(1).strip()
        issue_number = m_gh_issue.group(2).strip()
        repo = m_gh_issue.group(3).strip()
        
        issue_name = f"{repo}#{issue_number}"
        
        return MemoryExtraction(
            entities=[
                ExtractedEntity(name=user, entity_type="Person", epistemic_state="FACT"),
                ExtractedEntity(name=issue_name, entity_type="Issue", epistemic_state="FACT"),
                ExtractedEntity(name=repo, entity_type="Repository", epistemic_state="FACT"),
            ],
            relationships=[
                ExtractedRelationship(source=user, target=issue_name, relation="OPENED", epistemic_state="FACT"),
                ExtractedRelationship(source=issue_name, target=repo, relation="BELONGS_TO", epistemic_state="FACT"),
            ]
        )

    return None
