from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class QdrantDB:
    def __init__(
        self,
        collection: str = "default_collection",
        vector_size: int = 384,
        path: str = "data/qdrant_db",
        host: str = None,
        port: int = None,
    ):
        # Initialize Qdrant Client
        # If host is provided, use server mode. Otherwise use local persistent mode.
        if host:
            self.client = QdrantClient(host=host, port=port)
        else:
            # Local persistent storage
            self.client = QdrantClient(path=path)

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
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k
        )
        return [{"vector_id": h.id, "score": float(h.score)} for h in response.points]

    def delete_vector(self, vector_id: int) -> None:
        """
        Delete a vector from the collection using its ID.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=[vector_id]
        )