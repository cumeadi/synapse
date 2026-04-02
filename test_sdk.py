import asyncio
import os
from dotenv import load_dotenv

# Try to import from the local source for testing
import sys
sys.path.insert(0, "./clients/python")
from synapse import SynapseClient
from synapse.exceptions import SynapseError

load_dotenv()
API_KEY = os.getenv("SYNAPSE_MASTER_KEY")

async def main():
    print("Initializing SynapseClient...")
    namespace = "sdk_test_agent"
    
    async with SynapseClient(base_url="http://localhost:8000", api_key=API_KEY) as client:
        print(f"\n1. Ingesting memory into namespace '{namespace}'...")
        try:
            # response is now a MemoryResponse object
            response = await client.ingest(
                namespace=namespace,
                text="The team is standardizing on FastAPI for the backend. Do not use Pydantic v1.",
                metadata={"source": "test_script"}
            )
            print("Ingestion queued:", response.message) # dot notation
        except SynapseError as e:
            print("Failed to ingest:", e)
            return
            
        print("\n2. Waiting 3 seconds for backend background tasks to finish extracting graph...")
        await asyncio.sleep(3)
        
        print("\n3. Retrieving Hybrid Context...")
        try:
            # context is now a SearchHybridResponse object
            context = await client.search(
                namespace=namespace, 
                query="What framework are we standardizing on?"
            )
            print("\nSearch Results:")
            for res in context.results:
                print(f" - {res.content} (Score: {res.score})")
        except SynapseError as e:
            print("Failed to search:", e)
            
        print("\n4. Graph Traversal (Checking entities)...")
        try:
            # graph is now a GraphTraversalResponse object
            graph = await client.graph_traversal(namespace=namespace, entity="FastAPI", depth=1)
            print("\nGraph Data:")
            print(f"Nodes: {len(graph.entities)}, Edges: {len(graph.relationships)}")
            if graph.center:
                print(f"Center Node: {graph.center.name}")
        except SynapseError as e:
            print("Failed graph traversal:", e)
        
    print("\nChecking the Python code context is completed.")

if __name__ == "__main__":
    asyncio.run(main())
