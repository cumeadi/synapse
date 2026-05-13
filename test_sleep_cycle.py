import asyncio
import uuid
import time
from synapse import SynapseClient

async def main():
    # Use a unique namespace for testing
    ns = f"test_sleep_{uuid.uuid4().hex[:8]}"
    print(f"Using namespace: {ns}")
    
    async with SynapseClient() as client:
        # 1. Ingest synonymous entities
        print("Ingesting memories with synonymous entities...")
        await client.ingest(ns, "I am deploying my application on AWS.")
        await client.ingest(ns, "Amazon Web Services is a cloud provider I use.")
        
        # 2. Ingest contradictory facts
        print("Ingesting contradictory memories...")
        await client.ingest(ns, "I really hate programming in Python.")
        await client.ingest(ns, "Actually, Python is great, I love it now.")
        
        # Give background ingestion a moment to finish extracting entities
        print("Waiting for ingestion background tasks to finish (10s)...")
        time.sleep(10)
        
        # 3. Trigger Sleep Cycle
        print("Triggering Sleep Cycle...")
        res = await client.sleep(ns)
        print("Sleep cycle response:", res)
        
        # Wait for the background sleep cycle to finish
        print("Waiting for sleep cycle background tasks to finish (15s)...")
        time.sleep(15)
        
        # 4. Verify results via graph traversal
        print("Verifying graph...")
        # Since AWS and Amazon Web Services are synonyms, we can traverse from AWS
        try:
            graph = await client.graph_traversal(ns, "AWS", depth=2)
            print("\nGraph around 'AWS':")
            for ent in graph.entities:
                print(f"  Node: {ent.name} ({ent.entity_type})")
            for rel in graph.relationships:
                print(f"  Edge: {rel.source.name} -> {rel.relation_type} -> {rel.target.name}")
        except Exception as e:
            print(f"Could not traverse from AWS: {e}")
            
        print("\nVerification complete!")
        print("Check Synapse Studio to visually confirm 'AWS' and 'Amazon Web Services' merged, and contradictory Python relationships resolved.")

if __name__ == "__main__":
    asyncio.run(main())
