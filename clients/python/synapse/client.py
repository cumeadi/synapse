import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional

from .exceptions import APIError, AuthenticationError, NamespaceNotFoundError
from .models import (
    MemoryResponse, 
    SearchHybridResponse, 
    GraphTraversalResponse, 
    SourceImportResponse
)

# Exception types that we should safely retry on (e.g., connect timeouts, 5xx server errors, rate limits)
class RetryableAPIError(APIError):
    pass

def _raise_for_status(response: httpx.Response):
    """Raise parsed API errors."""
    if response.is_success:
        return
        
    status = response.status_code
    text = response.text
    
    if status in (401, 403):
        raise AuthenticationError(f"HTTP {status}: Invalid or missing API key.")
    elif status == 404:
        raise NamespaceNotFoundError(f"HTTP 404: {text}")
    elif status in (429, 500, 502, 503, 504):
        raise RetryableAPIError(f"HTTP {status}: {text}", status_code=status)
    else:
        raise APIError(f"HTTP {status}: {text}", status_code=status)


class SynapseClient:
    """Bulletproof async client for interacting with the Synapse Cognitive Memory Engine API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
            
        # Configure robust AsyncClient with explicit timeouts
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url, 
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )
        self._namespace_cache: Dict[str, str] = {}
        
    async def __aenter__(self):
        """Enable async with context managers."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure connection closure on exit."""
        await self.close()
        
    @retry(
        retry=(retry_if_exception_type(httpx.RequestError) | retry_if_exception_type(RetryableAPIError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def _resolve_namespace(self, namespace_name: str) -> str:
        """Resolves a human-readable namespace string to its UUID. Creates it if missing."""
        if namespace_name in self._namespace_cache:
            return self._namespace_cache[namespace_name]
            
        try:
            response = await self.http_client.get("/namespaces/")
            _raise_for_status(response)
            
            namespaces = response.json()
            for ns in namespaces:
                if ns["name"] == namespace_name:
                    self._namespace_cache[namespace_name] = ns["id"]
                    return ns["id"]
                    
            create_resp = await self.http_client.post("/namespaces/", json={"name": namespace_name})
            _raise_for_status(create_resp)
            
            new_ns = create_resp.json()
            self._namespace_cache[namespace_name] = new_ns["id"]
            return new_ns["id"]
            
        except httpx.RequestError as e:
            raise APIError(f"Network error resolving namespace: {str(e)}") from e
            
    @retry(
        retry=(retry_if_exception_type(httpx.RequestError) | retry_if_exception_type(RetryableAPIError)),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def ingest(self, namespace: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryResponse:
        """Ingest raw text into the memory vector store and auto-extract graph entities."""
        ns_id = await self._resolve_namespace(namespace)
        
        payload = {"content": text}
        if metadata:
            payload["metadata"] = metadata
            
        try:
            res = await self.http_client.post(f"/namespaces/{ns_id}/memories/", json=payload)
            _raise_for_status(res)
            return MemoryResponse(**res.json())
        except httpx.RequestError as e:
            raise APIError(f"Network error during ingestion: {str(e)}") from e
        
    @retry(
        retry=(retry_if_exception_type(httpx.RequestError) | retry_if_exception_type(RetryableAPIError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def search(self, namespace: str, query: str, metadata_filter: Optional[Dict[str, Any]] = None, top_k: int = 5) -> SearchHybridResponse:
        """Perform a hybrid semantic search across the namespace."""
        ns_id = await self._resolve_namespace(namespace)
        
        payload = {"query": query, "top_k": top_k}
        if metadata_filter:
            payload["metadata_filter"] = metadata_filter
            
        try:
            res = await self.http_client.post(f"/namespaces/{ns_id}/search/hybrid", json=payload)
            _raise_for_status(res)
            return SearchHybridResponse(**res.json())
        except httpx.RequestError as e:
            raise APIError(f"Network error during search: {str(e)}") from e
        
    @retry(
        retry=(retry_if_exception_type(httpx.RequestError) | retry_if_exception_type(RetryableAPIError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def graph_traversal(self, namespace: str, entity: str, depth: int = 1) -> GraphTraversalResponse:
        """Traverse the relational knowledge graph starting from a specific entity node."""
        ns_id = await self._resolve_namespace(namespace)
        
        params = {"entity_name": entity, "depth": depth}
        try:
            res = await self.http_client.get(f"/namespaces/{ns_id}/search/graph", params=params)
            _raise_for_status(res)
            return GraphTraversalResponse(**res.json())
        except httpx.RequestError as e:
            raise APIError(f"Network error during graph traversal: {str(e)}") from e
        
    @retry(
        retry=(retry_if_exception_type(httpx.RequestError) | retry_if_exception_type(RetryableAPIError)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def learn_repository(self, namespace: str, repo_url: str) -> SourceImportResponse:
        """Instruct Synapse to fetch, chunk, and ingest a GitHub repository's architecture."""
        ns_id = await self._resolve_namespace(namespace)
        
        payload = {"repo_url": repo_url}
        try:
            # We bump explicit request timeout here as scraping GH takes longer
            res = await self.http_client.post(f"/namespaces/{ns_id}/sources/github", json=payload, timeout=45.0)
            _raise_for_status(res)
            return SourceImportResponse(**res.json())
        except httpx.RequestError as e:
            raise APIError(f"Network error during repo learning: {str(e)}") from e
        
    @retry(
        retry=(retry_if_exception_type(httpx.RequestError) | retry_if_exception_type(RetryableAPIError)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def sleep(self, namespace: str) -> Dict[str, Any]:
        """Trigger the background memory consolidation (Sleep Cycle) for a namespace."""
        ns_id = await self._resolve_namespace(namespace)
        try:
            res = await self.http_client.post(f"/namespaces/{ns_id}/sleep")
            _raise_for_status(res)
            return res.json()
        except httpx.RequestError as e:
            raise APIError(f"Network error during sleep cycle trigger: {str(e)}") from e
            
        
    async def close(self):
        """Close the underlying HTTPX client. Prefer using `async with` context manager."""
        if not self.http_client.is_closed:
            await self.http_client.aclose()
