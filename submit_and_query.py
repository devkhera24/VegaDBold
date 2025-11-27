

import sys
import time
import json
import uuid
from typing import Dict, Any

# local imports (make sure adapters & core exist)
from core.query.embedder_adapter import embed as embed_fn
from core.query.qdrant_adapter import QdrantAdapter
from core.query.executor import LMDBMetaStore, QueryExecutor
from core.query.graph_adapter import graph as adapter_graph

def read_multiline_input(prompt="Paste your text:\n"):
    print(prompt)
    print("=" * 60)
    print("OPTIONS TO FINISH:")
    print("  1. Press Enter on an empty line")
    print("  2. Type 'END' on a line by itself")
    print("  3. Press Ctrl+D (Linux/Mac) or Ctrl+Z then Enter (Windows)")
    print("=" * 60)
    print()
    
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            print("\n[EOF detected - processing input...]")
            break
        
        # Check for END marker
        if line.strip().upper() == "END":
            print("\n[END marker detected - processing input...]")
            break
            
        # Check for empty line
        if line.strip() == "":
            if lines:  # Only break if we have some content already
                print("\n[Empty line detected - processing input...]")
                break
            else:
                continue  # Skip leading empty lines
        
        lines.append(line)
    
    result = "\n".join(lines).strip()
    
    if not result:
        print("\n[WARNING: No text collected]")
    else:
        print(f"\n[Collected {len(lines)} lines, {len(result)} characters]")
    
    return result

def ingest_text(text: str, lmdb: LMDBMetaStore, qdrant_adapter: QdrantAdapter, collection: str = "demo", meta_extra: Dict[str,Any]=None):
    ts = int(time.time() * 1000)
    doc_id = f"doc_{ts}_{uuid.uuid4().hex[:6]}"
    meta = {"id": doc_id, "text": text, "created_at": ts}
    if meta_extra:
        meta.update(meta_extra)

    # compute embedding
    vec = None
    try:
        vec = embed_fn(text)
        if vec is None:
            print("Warning: embedder returned None", file=sys.stderr)
    except Exception as e:
        print(f"Warning: embedder failed - {e}", file=sys.stderr)

    # Store vector in meta for fallback search
    if vec:
        meta["vector"] = vec

    # write to LMDB
    try:
        lmdb.put(doc_id, meta)
        print(f"✓ Stored in LMDB: {doc_id}")
    except Exception as e:
        print(f"Warning: LMDB put failed - {e}", file=sys.stderr)

    # upsert to qdrant via adapter if vector present
    if vec:
        try:
            qdrant_adapter.upsert(collection_name=collection, points=[{"id": doc_id, "vector": vec, "payload": meta}])
            print(f"✓ Stored in Qdrant: {collection}")
        except Exception as e:
            print(f"Warning: qdrant upsert failed - {e}", file=sys.stderr)
    else:
        print("⚠ Skipped Qdrant (no vector)")

    return doc_id, vec

def lmdb_scan_all(lmdb: LMDBMetaStore):
    """
    Safely scan all items from LMDB store.
    Supports both scan() method and direct access patterns.
    """
    # Try different access patterns
    if hasattr(lmdb, 'scan'):
        # If scan method exists
        try:
            return list(lmdb.scan())
        except Exception as e:
            print(f"Warning: scan() failed - {e}", file=sys.stderr)
    
    # Try accessing internal env/db
    if hasattr(lmdb, 'env') and lmdb.env:
        try:
            import lmdb as lmdb_lib
            results = []
            with lmdb.env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    k = key.decode('utf-8')
                    # Try to parse value
                    try:
                        from core.query.executor import loads_bytes
                        v = loads_bytes(value)
                    except:
                        try:
                            v = json.loads(value.decode('utf-8'))
                        except:
                            continue
                    results.append((k, v))
            return results
        except Exception as e:
            print(f"Warning: LMDB cursor scan failed - {e}", file=sys.stderr)
    
    # If in-memory fallback
    if hasattr(lmdb, '_mem'):
        return list(lmdb._mem.items())
    
    return []

def search_vector(vec, q_adapter: QdrantAdapter, lmdb: LMDBMetaStore, collection: str = "demo", k: int = 5):
    """
    Search for similar vectors. First tries Qdrant, then falls back to LMDB-based search.
    """
    # Try qdrant adapter search first
    hits = []
    if vec and q_adapter:
        try:
            hits = q_adapter.search(collection_name=collection, query_vector=vec, limit=k)
            if hits:
                print(f"✓ Found {len(hits)} results from Qdrant")
        except Exception as e:
            print(f"Warning: qdrant search error - {e}", file=sys.stderr)
            hits = []

    # If adapter returned hits, enrich with LMDB meta
    results = []
    if hits:
        for h in hits:
            pid = str(h.get("id"))
            score = h.get("score")
            meta = lmdb.get(pid)
            results.append({"id": pid, "score": score, "meta": meta})
        return results

    # Fallback: naive LMDB search (dot product with stored vectors)
    if vec:
        print("⚠ Falling back to LMDB-based search...")
        candidates = []
        
        try:
            all_items = lmdb_scan_all(lmdb)
            
            for k0, md in all_items:
                v = md.get("vector")
                if not v:
                    continue
                try:
                    # Calculate dot product similarity
                    score = sum(x*y for x,y in zip(v, vec))
                    candidates.append({"id": k0, "score": score, "meta": md})
                except Exception:
                    continue
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            results = candidates[:k]
            
            if results:
                print(f"✓ Found {len(results)} results from LMDB fallback")
            else:
                print("⚠ No results found in LMDB fallback")
                
            return results
            
        except Exception as e:
            print(f"Error during LMDB fallback search: {e}", file=sys.stderr)
            return []
    
    print("⚠ No vector available for search")
    return []

def main():
    print("\n" + "=" * 60)
    print("SUBMIT AND QUERY - Document Indexing & Search")
    print("=" * 60 + "\n")
    
    # quick checks & init stores / adapters
    try:
        lmdb = LMDBMetaStore(path="data/lmdb_meta")
        print("✓ LMDB initialized")
    except Exception as e:
        print(f"✗ LMDB initialization failed: {e}")
        sys.exit(1)
    
    q_adapter = None
    try:
        q_adapter = QdrantAdapter(collection="demo")
        print("✓ Qdrant adapter initialized")
    except Exception as e:
        print(f"⚠ Qdrant adapter warning: {e}")
    
    k = 5

    text = read_multiline_input()
    if not text:
        print("\n✗ No text provided — exiting.")
        sys.exit(0)

    print("\n" + "-" * 60)
    print("INDEXING DOCUMENT")
    print("-" * 60)

    doc_id, vec = ingest_text(text, lmdb, q_adapter, collection="demo")
    print(f"\n✓ Document ID: {doc_id}")

    print("\n" + "-" * 60)
    print(f"RUNNING VECTOR SEARCH (top-k = {k})")
    print("-" * 60)
    
    if vec:
        print("Using vector similarity search...")
    else:
        print("No vector available; attempting naive LMDB-based matching...")

    results = search_vector(vec, q_adapter, lmdb, collection="demo", k=k)

    out = {
        "query_doc_id": doc_id,
        "query_text_snippet": text[:200],
        "num_results": len(results),
        "results": results
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n✓ Done!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)