# pipeline_ingest.py
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import spacy
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

from neo4j import GraphDatabase

# =====================
# CONFIG
# =====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "docs_chunks"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =====================
# LOAD MODELS
# =====================
nlp = spacy.load("en_core_web_sm", disable=["parser"])
embed_model = SentenceTransformer(EMBED_MODEL)

# =====================
# INIT QDRANT + NEO4J
# =====================
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ===============================
# SAFE QDRANT INIT (non-blocking)
# ===============================
QDRANT_AVAILABLE = True
try:
    # Check if Qdrant responds
    qdrant.get_collections()

    # Create collection only if missing
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=embed_model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            ),
        )
except Exception as e:
    print(f"[QDRANT WARNING] Qdrant is NOT reachable — skipping vector storage. ({e})")
    QDRANT_AVAILABLE = False


def store_to_qdrant(points):
    if not QDRANT_AVAILABLE:
        print("[QDRANT] Skipping upsert because Qdrant is not available.")
        return
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)


neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# =====================
# HELPER FUNCTIONS
# =====================
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict]:
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + chunk_size, L)
        chunk_txt = text[i:end]
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_txt,
            "start_char": i,
            "end_char": end
        })
        i = end - overlap
        if i < 0:
            i = 0
    return chunks


def extract_entities(text: str):
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
    ]


def embed_texts(texts: List[str]):
    return embed_model.encode(texts, convert_to_numpy=True).tolist()


def store_to_qdrant(points):
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)


def store_to_neo4j(doc_json: dict):
    with neo_driver.session() as session:
        session.write_transaction(_neo4j_tx, doc_json)


def _neo4j_tx(tx, doc_json):
    # create document node
    tx.run(
        """
        MERGE (d:Document {doc_id:$doc_id})
        SET d.title=$title, d.source=$source, d.meta=$meta, d.created_at=$created_at
        """,
        doc_id=doc_json["doc_id"],
        title=doc_json.get("title"),
        source=doc_json.get("source"),
        meta=str(doc_json.get("meta", {})),
        created_at=doc_json.get("created_at"),
    )

    # chunk nodes
    for c in doc_json["chunks"]:
        tx.run(
            """
            MERGE (c:Chunk {chunk_id:$cid})
            SET c.text=$text, c.start_char=$start, c.end_char=$end, 
                c.entities=$entities, c.keywords=$keywords
            MERGE (d:Document {doc_id:$doc_id})-[:HAS_CHUNK]->(c)
            """,
            cid=c["chunk_id"],
            text=c["text"],
            start=c["start_char"],
            end=c["end_char"],
            entities=str(c.get("entities", [])),
            keywords=str(c.get("keywords", [])),
            doc_id=doc_json["doc_id"],
        )


# ============================
# AUTO-SCHEMA IMPORT
# ============================
try:
    from auto_schema import infer_schema_from_doc, register_schema
except Exception:
    infer_schema_from_doc = None
    register_schema = None


# =====================
# MAIN INGEST FUNCTION
# =====================
def ingest_document_text(source: str, text: str, title: str = None, meta: dict = None):
    doc_id = str(uuid.uuid4())
    meta = meta or {}

    # --------------------
    # 1) CHUNKING
    # --------------------
    chunks = chunk_text(text)
    chunk_texts = [c["text"] for c in chunks]

    # --------------------
    # 2) ENTITY EXTRACTION
    # --------------------
    for c in chunks:
        c["entities"] = extract_entities(c["text"])
        c["keywords"] = []     # optional: your keyword extractor later
        c["relations"] = []    # optional: graph relations auto or manual

    # --------------------
    # 3) EMBEDDINGS
    # --------------------
    embeddings = embed_texts(chunk_texts)

    # --------------------
    # 4) BUILD DOC JSON
    # --------------------
    doc_json = {
        "doc_id": doc_id,
        "source": source,
        "title": title or source,
        "meta": meta,
        "chunks": [{k: v for k, v in c.items() if k != "embedding"} for c in chunks],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    # --------------------
    # 5) AUTO SCHEMA STEP
    # --------------------
    schema_fp = None
    if infer_schema_from_doc and register_schema:
        jschema, metadata = infer_schema_from_doc(doc_json)
        schema_name = doc_json.get("source") or doc_json.get("title") or f"auto_{metadata.fingerprint}"
        registered = register_schema(schema_name, jschema, metadata)
        schema_fp = registered["meta"]["fingerprint"]
        print(f"[SCHEMA] Registered {schema_name} -> fp={schema_fp}")

    # --------------------
    # 6) BUILD QDRANT POINTS (include schema_fp!)
    # --------------------
    qdrant_points = []
    for c, emb in zip(chunks, embeddings):
        payload = {
            "doc_id": doc_id,
            "text": c["text"],
        }
        if schema_fp:
            payload["schema_fp"] = schema_fp

        qdrant_points.append(
            PointStruct(
                id=c["chunk_id"],
                vector=emb,
                payload=payload,
            )
        )

    # --------------------
    # 7) UPSERT + GRAPH SAVE
    # --------------------
    store_to_qdrant(qdrant_points)
    store_to_neo4j(doc_json)

    return doc_id


# =====================
# CLI USAGE
# =====================
if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    txt = Path(p).read_text()
    doc_id = ingest_document_text(source=p, text=txt, title=Path(p).name)
    print("Ingested doc:", doc_id)
