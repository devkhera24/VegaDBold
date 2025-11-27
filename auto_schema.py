"""
auto_schema.py
Auto-schema inference + LMDB registry for Vector+Graph pipeline.

Provides:
 - infer_schema_from_doc(doc) -> (json_schema, meta)
 - register_schema(name, json_schema, meta, allow_evolution=True) -> stored dict
 - get_schema_by_name(name) -> stored dict | None
 - get_schema_by_fp(fp) -> stored dict | None
 - reindex queue helpers: enqueue_reindex(doc_id), read_reindex_queue()
 - CLI for quick inspect / register
"""
import re
import json
import hashlib
from typing import Any, Dict, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime
import msgpack
import lmdb
import os

# ---------- CONFIG ----------
LMDB_INDEX_PATH = os.environ.get("LMDB_PATH", "data/lmdb_meta")
LMDB_MAP_SIZE = int(os.environ.get("LMDB_MAP_SIZE", 1_000_000_000))
# DB names
SCHEMA_DB = b"schema_db"
REINDEX_DB = b"reindex_db"
# ----------------------------

os.makedirs(LMDB_INDEX_PATH, exist_ok=True)
lmdb_env = lmdb.Environment(path=LMDB_INDEX_PATH, map_size=LMDB_MAP_SIZE, max_dbs=8)
schema_db = lmdb_env.open_db(SCHEMA_DB)
reindex_db = lmdb_env.open_db(REINDEX_DB)


# ---------- helpers ----------
def _type_of_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, list):
        if len(v) == 0:
            return "array"
        inner = set(_type_of_value(x) for x in v)
        if len(inner) == 1:
            return f"array<{list(inner)[0]}>"
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"


def _normalize_field_name(s: str) -> str:
    s = re.sub(r"\W+", "_", str(s)).strip("_")
    return s[:64] or "field"


def schema_hash(schema: dict) -> str:
    j = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:16]


@dataclass
class SchemaMeta:
    name: str
    version: int
    created_at: str
    fingerprint: str
    sample_count: int
    heuristics: Dict[str, Any]


# ---------- core functions ----------
def infer_schema_from_doc(doc: Dict, max_chunk_samples: int = 100) -> Tuple[dict, SchemaMeta]:
    """
    Inspect a doc JSON (with 'chunks') and return an approximate JSON Schema and meta.
    doc: {doc_id, source, title, meta, chunks: [ {chunk stuff} ]}
    """
    # top-level
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {},
        "required": []
    }
    heuristics = {}

    # sample top-level fields (except chunks heavy blob)
    for k, v in doc.items():
        if k == "chunks":
            continue
        t = _type_of_value(v)
        json_schema["properties"][k] = {"type": t}

    # inspect chunks
    chunks = doc.get("chunks", [])[:max_chunk_samples]
    chunk_props = {}
    chunk_counts = Counter()
    for c in chunks:
        for ck, cv in c.items():
            if ck == "embedding":
                continue
            t = _type_of_value(cv)
            chunk_props.setdefault(ck, Counter())[t] += 1
            chunk_counts[ck] += 1

    # convert chunk_props to schema props (pick most common type)
    chunk_schema_props = {}
    for ck, cnt in chunk_props.items():
        most_common_type = cnt.most_common(1)[0][0]
        chunk_schema_props[ck] = {"type": most_common_type}

    json_schema["properties"]["chunks"] = {"type": "array", "items": {"type": "object", "properties": chunk_schema_props}}
    heuristics["chunk_counts"] = dict(chunk_counts)
    fp = schema_hash(json_schema)
    meta = SchemaMeta(name="inferred", version=1, created_at=datetime.utcnow().isoformat() + "Z", fingerprint=fp, sample_count=len(chunks), heuristics=heuristics)
    return json_schema, meta


def _merge_json_schemas(a: dict, b: dict) -> dict:
    """
    Simple union merge for properties.
    On conflict, produce anyOf between existing and incoming. Conservative.
    """
    res = dict(a)
    a_props = a.get("properties", {})
    b_props = b.get("properties", {})
    merged_props = dict(a_props)
    for k, v in b_props.items():
        if k not in merged_props:
            merged_props[k] = v
            continue
        # same?
        if merged_props[k] == v:
            continue
        # conflict -> anyOf
        merged_props[k] = {"anyOf": [merged_props[k], v]}
    res["properties"] = merged_props
    # union required
    res["required"] = list(set(a.get("required", [])) | set(b.get("required", [])))
    return res


def register_schema(schema_name: str, json_schema: dict, meta: SchemaMeta, allow_evolution: bool = True) -> Dict[str, Any]:
    """
    Register schema into LMDB. If exists and differs, merge/evolve (if allowed).
    Returns stored dict {schema:..., meta:...}
    """
    key = f"schema:{schema_name}".encode("utf-8")
    with lmdb_env.begin(write=True, db=schema_db) as txn:
        raw = txn.get(key)
        if raw:
            stored = msgpack.unpackb(raw, raw=False)
            stored_schema = stored["schema"]
            stored_meta = stored["meta"]
            if stored_meta.get("fingerprint") == meta.fingerprint:
                return stored
            if not allow_evolution:
                raise RuntimeError("Schema conflict and evolution disabled")
            merged = _merge_json_schemas(stored_schema, json_schema)
            new_fp = schema_hash(merged)
            new_version = stored_meta.get("version", 1) + 1
            new_meta = {
                "name": schema_name,
                "version": new_version,
                "created_at": meta.created_at,
                "fingerprint": new_fp,
                "sample_count": stored_meta.get("sample_count", 0) + meta.sample_count,
                "heuristics": {**stored_meta.get("heuristics", {}), **meta.heuristics}
            }
            out = {"schema": merged, "meta": new_meta}
            txn.put(key, msgpack.packb(out, use_bin_type=True))
            # also create fp -> name mapping for quick lookup
            txn.put(f"fp:{new_fp}".encode("utf-8"), key)
            return out
        else:
            out = {"schema": json_schema, "meta": asdict(meta)}
            txn.put(key, msgpack.packb(out, use_bin_type=True))
            txn.put(f"fp:{meta.fingerprint}".encode("utf-8"), key)
            return out


def get_schema_by_name(schema_name: str) -> Dict[str, Any]:
    key = f"schema:{schema_name}".encode("utf-8")
    with lmdb_env.begin(write=False, db=schema_db) as txn:
        raw = txn.get(key)
        if not raw:
            return None
        return msgpack.unpackb(raw, raw=False)


def get_schema_by_fp(fp: str) -> Dict[str, Any]:
    with lmdb_env.begin(write=False, db=schema_db) as txn:
        map_key = txn.get(f"fp:{fp}".encode("utf-8"))
        if not map_key:
            return None
        raw = txn.get(map_key)
        if not raw:
            return None
        return msgpack.unpackb(raw, raw=False)


# ---------- reindex queue ----------
def enqueue_reindex(doc_id: str, reason: str = "manual") -> None:
    item = {"doc_id": doc_id, "reason": reason, "requested_at": datetime.utcnow().isoformat() + "Z"}
    with lmdb_env.begin(write=True, db=reindex_db) as txn:
        txn.put(doc_id.encode("utf-8"), msgpack.packb(item, use_bin_type=True))


def read_reindex_queue() -> Dict[str, Any]:
    out = {}
    with lmdb_env.begin(write=False, db=reindex_db) as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            out[k.decode("utf-8")] = msgpack.unpackb(v, raw=False)
    return out


def pop_reindex_item(doc_id: str) -> None:
    with lmdb_env.begin(write=True, db=reindex_db) as txn:
        txn.delete(doc_id.encode("utf-8"))


# ---------- CLI helpers ----------
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="auto_schema utility")
    parser.add_argument("--inspect", help="inspect schema by name", type=str)
    parser.add_argument("--inspect-fp", help="inspect schema by fingerprint", type=str)
    parser.add_argument("--enqueue", help="enqueue doc_id for reindex", type=str)
    args = parser.parse_args()
    if args.inspect:
        s = get_schema_by_name(args.inspect)
        print(json.dumps(s, indent=2))
        sys.exit(0)
    if args.inspect_fp:
        s = get_schema_by_fp(args.inspect_fp)
        print(json.dumps(s, indent=2))
        sys.exit(0)
    if args.enqueue:
        enqueue_reindex(args.enqueue, reason="cli")
        print("enqueued", args.enqueue)
        sys.exit(0)
    print("nothing to do. try --inspect name or --inspect-fp fp")
