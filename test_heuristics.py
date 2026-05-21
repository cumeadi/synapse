from app.services.heuristics import run_heuristics

def test_jira_issue_created():
    content = "Alice Chen created Jira issue ENG-137: 'Implement OAuth2 token refresh' (type=Story, status=To Do, assignee=Bob Kim)"
    ext = run_heuristics(content)
    assert ext is not None
    assert len(ext.entities) == 3
    names = {e.name for e in ext.entities}
    assert "Alice Chen" in names
    assert "ENG-137" in names
    assert "Bob Kim" in names
    
    assert len(ext.relationships) == 2
    relations = {r.relation for r in ext.relationships}
    assert "CREATED" in relations
    assert "ASSIGNED_TO" in relations
    
    # Check epistemic state
    assert all(e.epistemic_state == "FACT" for e in ext.entities)

def test_jira_issue_comment():
    content = "Carol Davis commented on Jira issue ENG-137: 'I'll pick this up in the next sprint.'"
    ext = run_heuristics(content)
    assert ext is not None
    assert len(ext.entities) == 2
    assert ext.relationships[0].relation == "COMMENTED_ON"

def test_github_pr_opened():
    content = "bob opened pull request #42 in acme/payments: 'Refactor payment service authentication' (feature/auth-refactor → main)"
    ext = run_heuristics(content)
    assert ext is not None
    names = {e.name for e in ext.entities}
    assert "acme/payments#42" in names
    assert "bob" in names
    assert "acme/payments" in names
    
    rels = {(r.source, r.target, r.relation) for r in ext.relationships}
    assert ("bob", "acme/payments#42", "OPENED") in rels
    assert ("acme/payments#42", "acme/payments", "BELONGS_TO") in rels

def test_github_pr_merged():
    content = "bob merged pull request #42 in acme/payments: 'Refactor payment service authentication'"
    ext = run_heuristics(content)
    assert ext is not None
    rels = {(r.source, r.target, r.relation) for r in ext.relationships}
    assert ("bob", "acme/payments#42", "MERGED") in rels

def test_github_push():
    content = "alice pushed 2 commits to acme/payments/main: 'Fix null pointer in token refresh'"
    ext = run_heuristics(content)
    assert ext is not None
    names = {e.name for e in ext.entities}
    assert "alice" in names
    assert "acme/payments" in names
    assert "acme/payments/main" not in names # Should extract repo

    rels = {(r.source, r.target, r.relation) for r in ext.relationships}
    assert ("alice", "acme/payments", "COMMITTED_TO") in rels

def test_github_issue_opened():
    content = "carol opened issue #99 in acme/payments: 'Payment gateway timeout in production'"
    ext = run_heuristics(content)
    assert ext is not None
    names = {e.name for e in ext.entities}
    assert "acme/payments#99" in names
    assert "carol" in names
    
    rels = {(r.source, r.target, r.relation) for r in ext.relationships}
    assert ("carol", "acme/payments#99", "OPENED") in rels
    
def test_no_match():
    content = "Just a regular chat message that should go to the LLM."
    ext = run_heuristics(content)
    assert ext is None

if __name__ == "__main__":
    print("Testing Heuristics Extraction...")
    test_jira_issue_created()
    test_jira_issue_comment()
    test_github_pr_opened()
    test_github_pr_merged()
    test_github_push()
    test_github_issue_opened()
    test_no_match()
    print("✅ All heuristic extraction tests passed!")
