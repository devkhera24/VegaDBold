import time
from core.qdrant_client import QdrantDB

def test_upsert_and_search_local():
    db = QdrantDB(collection="test_collection_for_unit_tests", vector_size=384)

    vec = [0.01] * 384
    db.upsert_vector(vector_id=123, vector=vec, payload={"text": "test"})

    time.sleep(0.3)

    results = db.search_vectors(vec, top_k=1)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["vector_id"] == 123
    assert isinstance(results[0]["score"], float)

def test_delete_vector():
    db = QdrantDB(collection="delete_test_collection")

    vec = [0.05] * 384
    db.upsert_vector(99, vec, {"text": "delete me"})

    # Ensure it exists
    results_before = db.search_vectors(vec, top_k=1)
    assert results_before[0]["vector_id"] == 99

    # Delete
    db.delete_vector(99)

    # Search again – should not return the deleted vector
    results_after = db.search_vectors(vec, top_k=1)

    assert len(results_after) == 0 or results_after[0]["vector_id"] != 99
