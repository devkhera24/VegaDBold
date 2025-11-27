# update_chunk.py
from pipeline_ingest import embed_texts, qdrant, QDRANT_COLLECTION, neo_driver

def update_chunk_text(chunk_id: str, new_text: str):
    emb = embed_texts([new_text])[0]
    # upsert vector
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[
        {"id": chunk_id, "vector": emb, "payload": {"text": new_text}}
    ])
    # update Neo4j
    with neo_driver.session() as s:
        s.write_transaction(lambda tx: tx.run("MATCH (c:Chunk {chunk_id:$id}) SET c.text=$text", id=chunk_id, text=new_text))
    return True
