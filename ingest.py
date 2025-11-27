#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import Callable, Dict, Any, Optional

# try optional libs
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

try:
    import lmdb
    LMDB_AVAILABLE = True
except Exception:
    LMDB_AVAILABLE = False

# fast json
try:
    import orjson as _orjson
    def dumps_bytes(o): return _orjson.dumps(o)
    def loads_bytes(b): return _orjson.loads(b)
except Exception:
    def dumps_bytes(o): return json.dumps(o).encode("utf-8")
    def loads_bytes(b): return json.loads(b.decode("utf-8"))

# Basic LMDB wrapper (same shape as executor's LMDBMetaStore)
class LMDBWriter:
    def __init__(self, path="data/lmdb_meta", map_size=(1<<30)):
        if LMDB_AVAILABLE:
            self.env = lmdb.open(path, map_size=map_size, subdir=True, create=True, lock=True)
        else:
            self.env = None
            self._mem = {}

    def put(self, key: str, value: Dict[str,Any]):
        if self.env:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode("utf-8"), dumps_bytes(value))
        else:
            self._mem[key] = value

# Qdrant helper (simple)
class QdrantWriter:
    def __init__(self, url="http://localhost:6333", collection="default_collection"):
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(url=url)
            self.collection = collection
        else:
            self.client = None
            self.collection = collection

    def ensure_collection(self, vector_size=128, distance="Cosine"):
        if not self.client:
            return
        try:
            self.client.get_collection(self.collection)
        except Exception:
            # create simple collection
            self.client.recreate_collection(self.collection, vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE))

    def upsert_point(self, point_id: str, vector: Optional[list], payload: Dict[str,Any]):
        if not self.client:
            return
        vec = vector if vector is not None else []
        self.client.upsert(
            collection_name=self.collection,
            points=[qmodels.PointStruct(id=point_id, vector=vec, payload=payload)]
        )

# ingest pipeline
def ingest_jsonl(
    path: str,
    lmdb_writer: LMDBWriter,
    qdrant_writer: QdrantWriter,
    embedder: Optional[Callable[[str], list]] = None,
    text_field: str = "text",
    id_field: str = "id",
    vector_field: str = "vector",
    metadata_field: Optional[str] = None
):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            node_id = str(rec.get(id_field) or rec.get("id"))
            meta = rec.get(metadata_field) if metadata_field and metadata_field in rec else rec.copy()
            # ensure id on meta
            meta["id"] = node_id

            # vector: prefer precomputed; else embed text
            vec = rec.get(vector_field)
            if vec is None and embedder and rec.get(text_field):
                vec = embedder(rec.get(text_field))

            # store metadata in LMDB
            lmdb_writer.put(node_id, meta)

            # upsert to qdrant if vector exists
            if vec is not None:
                qdrant_writer.upsert_point(node_id, vec, meta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, help="input JSONL file")
    parser.add_argument("--lmdb-path", default="data/lmdb_meta", help="lmdb path")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL","http://localhost:6333"))
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION","default_collection"))
    parser.add_argument("--vector-size", type=int, default=128)
    args = parser.parse_args()

    lmdbw = LMDBWriter(path=args.lmdb_path)
    qw = QdrantWriter(url=args.qdrant_url, collection=args.collection)
    qw.ensure_collection(vector_size=args.vector_size)

    # *** NOTE: Provide your embedder here if you need embeddings from text ***
    # Example: from my_embed_module import embed; embedder=embed
    embedder = None

    ingest_jsonl(args.input, lmdbw, qw, embedder=embedder)

if __name__ == "__main__":
    main()
