from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class QdrantDB:
    def __init__(
        self,
        collection: str = "default_collection",
        vector_size: int = 384,
    ):
        # In-memory Qdrant (portable, no Docker)
        self.client = QdrantClient(path=":memory:")

        self.collection = collection
        self.vector_size = vector_size

        # Create collection if not exists
        try:
            self.client.get_collection(collection_name=self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert_vector(self, vector_id: int, vector: List[float], payload: Dict[str, Any]) -> None:
        point = PointStruct(id=vector_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection, points=[point])

    def search_vectors(self, vector: List[float], top_k: int = 5) -> List[Dict[str, float]]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k
        )
        return [{"vector_id": h.id, "score": float(h.score)} for h in hits]

    def delete_vector(self, vector_id: int) -> None:
        """
        Delete a vector from the collection using its ID.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector={"points": [vector_id]}
        )