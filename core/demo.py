from core.embeddings import Embeddings
from core.qdrant_client import QdrantDB

def main():
    emb = Embeddings()
    db = QdrantDB(collection="demo_collection")

    text = "This is a demo text"
    v = emb.embed(text)

    db.upsert_vector(vector_id=1, vector=v, payload={"text": text})
    res = db.search_vectors(v, top_k=1)

    print(res)

if __name__ == "__main__":
    main()