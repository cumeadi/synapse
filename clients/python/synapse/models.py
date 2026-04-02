from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MemoryResponse(BaseModel):
    """Response returned when storing a memory."""
    id: str
    status: str
    message: str

class SearchResultItem(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    created_at: str

class SearchHybridResponse(BaseModel):
    """Response from a hybrid search."""
    query: str
    results: List[SearchResultItem]
    total: int

class GraphEntity(BaseModel):
    id: str
    name: str
    entity_type: str

class GraphRelationship(BaseModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    weight: float

class GraphTraversalResponse(BaseModel):
    """Response containing a graph neighborhood."""
    center: Optional[GraphEntity] = None
    depth: int
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]

class SourceImportResponse(BaseModel):
    """Response from importing a knowledge source (e.g. GitHub)."""
    id: str
    name: str
    source_type: str
    status: str
    message: str
