# run_query_dsl.py
from parser import parse
from core.query.executor import QueryExecutor, LMDBMetaStore, LocalGraphStoreFallback
import os

# Option A: use real LMDB + Qdrant (if installed and running)
use_real = os.environ.get("USE_REAL", "0") == "1"

if use_real:
    lmdb_store = LMDBMetaStore(path="data/lmdb_meta")
    # If Qdrant is running locally:
    try:
        from qdrant_client import QdrantClient
        q_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        collection = os.environ.get("QDRANT_COLLECTION", "default_collection")
    except Exception:
        q_client = None
        collection = None
else:
    lmdb_store = LMDBMetaStore()  # in-memory fallback when lmdb not installed
    q_client = None
    collection = None

# Simple embedder placeholder - Member A will replace this with real embedder
def dummy_embed(text: str):
    # naive embedding: map chars to small vector (demo only)
    return [float(ord(c) % 10) / 10.0 for c in text[:32]]

# simple graph seed
graph = LocalGraphStoreFallback()
graph.seed({"A":["B"], "B":["C"], "C":[]})

executor = QueryExecutor(
    qdrant_client=q_client,
    qdrant_collection=collection,
    lmdb_store=lmdb_store,
    embedder=dummy_embed,
    graph=graph
)

EXAMPLES = [
    'FIND NODES WHERE type="book" LIMIT 5',
    'GET NODE node_id="n1"',
    'GET VECTOR node_id="n2"',
    'SEARCH VECTOR EMBED("hello world") K 3',
    'FIND PATH FROM node_id="A" TO node_id="C" MAXHOPS 5'
]

if __name__ == "__main__":
    for q in EXAMPLES:
        print("\nQUERY:", q)
        ast = parse(q)
        print("AST:", ast)
        print("RESULT:", executor.execute(ast))
