#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import os
import json
from typing import Any, Dict

from core.query.parser import parse
from core.query.executor import QueryExecutor, LMDBMetaStore, LocalGraphStoreFallback

app = FastAPI(title="VectorGraph Query API")

# try multiple export names to stay compatible with different auto_schema versions
try:
    # preferred names
    from auto_schema import (
        infer_schema_from_doc,
        register_schema,
        get_schema_by_name,
        read_reindex_queue,
        enqueue_reindex,
        lmdb_env,
    )
    # normalize a common name used elsewhere
    get_schema = get_schema_by_name
    _read_reindex_queue = read_reindex_queue
except Exception:
    # try older/alternate names
    try:
        from auto_schema import infer_schema_from_doc, register_schema, get_schema, lmdb_env, read_reindex_queue as _read_reindex_queue, enqueue_reindex
    except Exception:
        infer_schema_from_doc = None
        register_schema = None
        get_schema = None
        lmdb_env = None
        _read_reindex_queue = None
        enqueue_reindex = None

# wire default stores (use env to control real vs mock)
USE_REAL = os.environ.get("USE_REAL", "0") == "1"

if USE_REAL:
    lmdb_store = LMDBMetaStore(path=os.environ.get("LMDB_PATH", "data/lmdb_meta"))
    # qdrant client may be auto-initialized by executor if available
else:
    lmdb_store = LMDBMetaStore()  # in-memory fallback

graph = LocalGraphStoreFallback()
# seed can be done elsewhere or via a separate admin endpoint (simple seed for demo)
graph.seed({"A": ["B"], "B": ["C"], "C": []})

executor = QueryExecutor(lmdb_store=lmdb_store, graph=graph)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def run_query(req: QueryRequest):
    try:
        ast = parse(req.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"parse error: {e}")

    try:
        res = executor.execute(ast)
        return {"ast": ast, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execution error: {e}")


# Simple in-lmdb reindex queue helpers (uses lmdb_env from auto_schema)
def _read_reindex_queue_fallback() -> Dict[str, Any]:
    try:
        import msgpack, lmdb
        if lmdb_env is None:
            return {}
        db = lmdb_env.open_db(b"reindex_db")
        out = {}
        with lmdb_env.begin(db=db, write=False) as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                out[k.decode()] = msgpack.unpackb(v, raw=False)
        return out
    except Exception:
        return {}


@app.get("/schema/{name}")
async def schema_inspect(name: str):
    """Return schema and meta stored in LMDB (via auto_schema)."""
    if get_schema is None:
        raise HTTPException(status_code=501, detail="auto_schema not available on this runtime")
    s = get_schema(name)
    if s is None:
        raise HTTPException(status_code=404, detail="schema not found")
    return s


@app.post("/schema/register")
async def schema_register(payload: Dict[str, Any] = Body(...)):
    """
    Register an inferred schema for a doc JSON.
    Body: a full document JSON (like the doc you would ingest, with 'chunks', 'meta', etc.)
    Returns: registered schema meta
    """
    if infer_schema_from_doc is None or register_schema is None:
        raise HTTPException(status_code=501, detail="auto_schema not available on this runtime")
    try:
        jschema, meta = infer_schema_from_doc(payload)
        name = payload.get("source") or payload.get("title") or f"auto_{meta.get('fingerprint', '') if isinstance(meta, dict) else getattr(meta, 'fingerprint', '')}"
        out = register_schema(name, jschema, meta)
        return {"registered": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"schema register failed: {e}")


@app.get("/schema/reindex_queue")
async def schema_reindex_queue():
    """Show items queued for reindex (simple LMDB queue read)."""
    if _read_reindex_queue is None and _read_reindex_queue_fallback is None:
        raise HTTPException(status_code=501, detail="reindex queue not available")
    q = {}
    if _read_reindex_queue:
        try:
            q = _read_reindex_queue()
        except Exception:
            q = _read_reindex_queue_fallback()
    else:
        q = _read_reindex_queue_fallback()
    return {"queue_size": len(q), "items": q}


@app.post("/schema/reindex_one/{doc_id}")
async def schema_reindex_one(doc_id: str):
    """
    Trigger a single doc reindex. Implementation here is intentionally tiny:
      - push a lightweight queue entry in LMDB reindex_db (via auto_schema.enqueue_reindex or fallback)
    In your real runtime, wire this to your reindex worker.
    """
    if enqueue_reindex is None and lmdb_env is None:
        raise HTTPException(status_code=501, detail="auto_schema reindex enqueue not available on this runtime")
    try:
        if enqueue_reindex:
            enqueue_reindex(doc_id, reason="api_request")
        else:
            # fallback: write directly into LMDB reindex_db
            import msgpack, lmdb
            db = lmdb_env.open_db(b"reindex_db")
            item = {"doc_id": doc_id, "requested_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
            with lmdb_env.begin(write=True, db=db) as txn:
                txn.put(doc_id.encode("utf-8"), msgpack.packb(item, use_bin_type=True))
        return {"queued": {"doc_id": doc_id, "requested_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"queueing failed: {e}")
