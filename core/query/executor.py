# core/query/executor.py
"""
Schema-aware Query Executor for Vector+Graph system.

This executor is intentionally small and conservative:
- It expects the parser to produce a compact AST (examples below).
- It will attempt to fetch a schema via 'schema_fp' in a Qdrant point payload
  and cast query literal values according to the schema before applying filters.

Example ASTs this executor understands:
  {"type":"find_similar", "k":5, "text":"how to install qdrant", "filters":[ {"field":"meta.published_at","op":">","value":"2025-01-01"} ] }
  {"type":"graph_traverse", "chunk_id":"<uuid>", "depth":2 }

If your parser's AST differs, adapt the small `execute()` switch.
"""

import os
from typing import Any, Dict, List, Optional
import json
import logging

# optional qdrant client (if you want to do vector search)
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

# LMDB meta store placeholder (simple in-memory fallback if not provided)
try:
    import lmdb, msgpack
except Exception:
    lmdb = None

# import schema adapter helpers (best-effort; fall back to no-ops)
try:
    from core.query.executor_schema_adapter import (
        resolve_schema_for_payload,
        get_field_type_from_schema,
        cast_value_to_type,
    )
except Exception:
    # no-op fallbacks
    def resolve_schema_for_payload(payload): return None
    def get_field_type_from_schema(schema_stored, field_path): return None
    def cast_value_to_type(value, target_type): return value

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LMDBMetaStore:
    """
    Very small LMDB-backed or in-memory meta store used by the executor for storing
    lightweight metadata about docs/chunks. This mirrors the idea of your LMDB meta layer.
    """

    def __init__(self, path: Optional[str] = None):
        self._mem = {}
        self.path = path
        if path and lmdb:
            os.makedirs(path, exist_ok=True)
            self.env = lmdb.Environment(path=path, map_size=1_000_000_000, max_dbs=4)
        else:
            self.env = None

    def put(self, key: str, value: Dict[str, Any]):
        if self.env:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode("utf-8"), msgpack.packb(value, use_bin_type=True))
        else:
            self._mem[key] = value

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.env:
            with self.env.begin(write=False) as txn:
                raw = txn.get(key.encode("utf-8"))
                if not raw:
                    return None
                return msgpack.unpackb(raw, raw=False)
        return self._mem.get(key)


class LocalGraphStoreFallback:
    """
    Tiny graph store for demo use. Stores adjacency list in-memory.
    """

    def __init__(self):
        self.adj = {}

    def seed(self, mapping: Dict[str, List[str]]):
        self.adj.update(mapping)

    def traverse(self, start_node: str, max_depth: int = 1) -> List[str]:
        # BFS up to max_depth, return list of node ids visited (including start)
        out = []
        seen = set()
        queue = [(start_node, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            out.append(node)
            if depth < max_depth:
                for nbr in self.adj.get(node, []):
                    queue.append((nbr, depth + 1))
        return out


class QueryExecutor:
    def __init__(self, lmdb_store: Optional[LMDBMetaStore] = None, graph: Optional[LocalGraphStoreFallback] = None, qdrant_client: Optional[Any] = None):
        self.lmdb_store = lmdb_store or LMDBMetaStore()
        self.graph = graph or LocalGraphStoreFallback()
        # optionally accept an external Qdrant client or initialize from env
        if qdrant_client:
            self.qdrant = qdrant_client
        else:
            qdrant_url = os.environ.get("QDRANT_HOST")
            qdrant_port = os.environ.get("QDRANT_PORT")
            if QdrantClient and qdrant_url:
                try:
                    self.qdrant = QdrantClient(host=qdrant_url, port=int(qdrant_port) if qdrant_port else None)
                except Exception as e:
                    logger.warning("Failed to init QdrantClient: %s", e)
                    self.qdrant = None
            else:
                self.qdrant = None

    # -------------------------
    # public executor surface
    # -------------------------
    def execute(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch execution based on the AST 'type' field.

        This function intentionally supports a small set of operations.
        Extend / adapt to your full AST shape as needed.
        """
        if not isinstance(ast, dict):
            raise ValueError("AST must be a dict-shaped node")

        t = ast.get("type")
        if t == "find_similar":
            return self._exec_find_similar(ast)
        if t == "graph_traverse":
            return self._exec_graph_traverse(ast)
        raise ValueError(f"unsupported ast type: {t}")

    # -------------------------
    # find_similar implementation
    # -------------------------
    def _exec_find_similar(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected AST fields:
          - text: the query string
          - k: number of neighbors
          - filters: optional list of filter objects: {field, op, value}
        """
        text = ast.get("text")
        k = int(ast.get("k", 5))
        filters = ast.get("filters", []) or []

        if not text:
            raise ValueError("find_similar requires 'text'")

        if not self.qdrant:
            # fallback behavior: if Qdrant not available, try to return a placeholder
            logger.info("Qdrant client not configured; returning empty result set")
            return {"results": [], "meta": {"source": "fallback"}}

        # encode locally? executor won't run embed model — expect qdrant collection configured to accept text search or external querying
        # We'll call qdrant.search with a vector if the caller provided one in AST (rare). For hackathon, use a text->vector helper if available.
        query_vector = ast.get("vector")  # optional precomputed vector
        if query_vector is None:
            # Not ideal — try to use qdrant's text search if collection supports it.
            # If user's stack handles embeddings before query, they can pass vector in AST.
            raise RuntimeError("Executor requires a precomputed 'vector' in AST or qdrant client needs server-side text2vec (not assumed)")

        # perform vector search
        try:
            search_res = self.qdrant.query_points(collection_name=os.environ.get("QDRANT_COLLECTION", "docs_chunks"), query=query_vector, limit=k).points
        except Exception as e:
            logger.error("qdrant.query_points failed: %s", e)
            return {"results": [], "error": str(e)}

        # post-process results: apply filters with schema-aware casting
        final = []
        for p in search_res:
            pid = getattr(p, "id", None) or p.get("id")
            payload = getattr(p, "payload", None) or p.get("payload", {}) or {}
            score = getattr(p, "score", None) or p.get("score", None)

            # attempt schema-aware filter casting
            schema_stored = resolve_schema_for_payload(payload)
            if not self._payload_passes_filters(payload, filters, schema_stored):
                continue

            final.append({"id": pid, "score": score, "payload": payload})

        return {"results": final}

    # -------------------------
    # graph_traverse implementation
    # -------------------------
    def _exec_graph_traverse(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        chunk_id = ast.get("chunk_id")
        depth = int(ast.get("depth", 1))
        if not chunk_id:
            raise ValueError("graph_traverse requires 'chunk_id'")
        visited = self.graph.traverse(chunk_id, max_depth=depth)
        return {"nodes": visited}

    # -------------------------
    # filter evaluation helpers
    # -------------------------
    def _payload_passes_filters(self, payload: Dict[str, Any], filters: List[Dict[str, Any]], schema_stored: Optional[Dict[str, Any]]) -> bool:
        """
        Evaluate list of filter dicts against a payload. Filters are of form:
          {"field":"meta.published_at", "op":">", "value":"2025-01-01"}

        This function will:
          - for each filter, try to get expected type from schema_stored (via get_field_type_from_schema)
          - cast filter's literal value using cast_value_to_type
          - perform the comparison
        """
        if not filters:
            return True

        for f in filters:
            field = f.get("field")
            op = f.get("op")
            raw_val = f.get("value")

            # get payload value (support dot notation)
            pval = self._get_payload_value(payload, field)

            # resolve expected type (from schema if available)
            expected_type = None
            if schema_stored:
                expected_type = get_field_type_from_schema(schema_stored, field)

            # cast the raw_val to expected type if possible
            if expected_type:
                casted_val = cast_value_to_type(raw_val, expected_type)
            else:
                casted_val = raw_val

            # do the comparison (safe)
            if not self._compare_values(pval, op, casted_val):
                return False
        return True

    @staticmethod
    def _get_payload_value(payload: Dict[str, Any], field_path: str):
        """
        Supports simple dot-access like "meta.published_at" or "text" or "doc_id".
        For nested arrays (chunks.*) we do not attempt deep matching here.
        """
        if not field_path:
            return None
        parts = field_path.split(".")
        cur = payload
        for p in parts:
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                # can't traverse further
                return None
        return cur

    @staticmethod
    def _compare_values(left, op: str, right) -> bool:
        """
        Narrow, safe comparison implementation. Handles None gracefully.
        """
        try:
            if op == "==" or op == "=":
                return left == right
            if op == "!=":
                return left != right
            # order comparisons: if types mismatch but both are strings that look like numbers, try numeric compare
            if left is None or right is None:
                return False
            # numeric compare
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
            # contains (for arrays/strings)
            if op == "in":
                # right in left? or left in right? We treat: left contains right
                if isinstance(left, (list, tuple, set)):
                    return right in left
                if isinstance(left, str):
                    return str(right) in left
                return False
            # fallback: equality
            return left == right
        except Exception:
            # conservative: if comparison fails, deny
            return False


# small convenience factory used in your api.py wiring
def make_executor(lmdb_store=None, graph=None):
    # try to init a Qdrant client from env if available
    qdrant_client = None
    if QdrantClient and os.environ.get("QDRANT_HOST"):
        try:
            qdrant_client = QdrantClient(host=os.environ.get("QDRANT_HOST"), port=int(os.environ.get("QDRANT_PORT", "6333")))
        except Exception as e:
            logger.warning("Failed to init Qdrant client in factory: %s", e)
            qdrant_client = None
    return QueryExecutor(lmdb_store=lmdb_store, graph=graph, qdrant_client=qdrant_client)