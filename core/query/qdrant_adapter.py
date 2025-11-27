from typing import Any, List, Dict, Optional
try:
    from core.qdrant_client import QdrantDB as _QdrantDB
    _HAS_QDRANTDB = True
except Exception:
    _QDRANTDB = None
    _HAS_QDRANTDB = False

class QdrantAdapter:
    def __init__(self, qdrantdb: Optional[Any] = None, collection: str = "default_collection"):
        if qdrantdb is None and _HAS_QDRANTDB:
            self.client = _QdrantDB(collection=collection)
        else:
            self.client = qdrantdb
        self.collection = collection

    def upsert(self, collection_name: str, points: List[Dict[str, Any]]):
        if not self.client:
            return
        for p in points:
            vid = p.get("id")
            vec = p.get("vector")
            payload = p.get("payload") or {}
            try:
                vid_int = int(vid)
            except Exception:
                vid_int = vid
            try:
                self.client.upsert_vector(vid_int, vec, payload)
            except Exception:
                pass

    def search(self, collection_name: str, query_vector: List[float], limit: int = 10, query_filter: Optional[Any] = None):
        if not self.client:
            return []
        try:
            hits = self.client.search_vectors(query_vector, top_k=limit)
        except Exception:
            return []
        out = []
        for h in hits:
            vid = h.get("vector_id", h.get("id", None))
            score = h.get("score", None)
            out.append({"id": str(vid), "score": float(score) if score is not None else None})
        return out

    def get_point(self, collection_name: str, point_id: Any):
        # Member B's QdrantDB doesn't expose get_point; return None for fallback logic
        return None