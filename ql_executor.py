# ql_executor.py
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pipeline_ingest import qdrant, QDRANT_COLLECTION, embed_model
import re

model = embed_model  # reuse

def q_find_similar(query:str, top_k:int=5, filter_payload:dict=None):
    vec = model.encode([query])[0].tolist()
    filter_payload = filter_payload or {}
    # simple search: no filter example
    resp = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=top_k)
    # return id + payload + score
    return [{"id": p.id, "score": p.score, "payload": p.payload} for p in resp]

def q_graph_traverse(chunk_id:str, depth:int=1, rel_type:str=None):
    from neo4j import GraphDatabase
    # reuse neo_driver from pipeline_ingest
    from pipeline_ingest import neo_driver
    q = """
    MATCH (start:Chunk {chunk_id:$cid})
    CALL apoc.path.subgraphNodes(start, {maxLevel:$depth, relationshipFilter:$relFilter}) YIELD node
    RETURN node.chunk_id as chunk_id, labels(node) as labels, node.text as text
    """
    rel_filter = rel_type or ""
    with neo_driver.session() as session:
        res = session.run(q, cid=chunk_id, depth=depth, relFilter=rel_filter)
        return [r.data() for r in res]

def execute_query(ql:str):
    # very tiny parser
    ql = ql.strip()
    if m := re.match(r'FIND TOP (\d+) SIMILAR "(.*)"', ql):
        k = int(m.group(1)); txt = m.group(2)
        return q_find_similar(txt, top_k=k)
    if m := re.match(r'GRAPH TRAVERSE FROM (\S+) DEPTH (\d+)', ql):
        cid = m.group(1); depth = int(m.group(2))
        return q_graph_traverse(cid, depth)
    raise ValueError("unsupported query")
