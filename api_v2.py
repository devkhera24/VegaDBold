#!/usr/bin/env python3
"""
api_v2.py - Complete CRUD + Hybrid Search API
Implements all problem statement requirements
"""
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import json
import uuid
from datetime import datetime

# Core imports
from core.query.parser import parse
from core.query.executor import QueryExecutor, LMDBMetaStore
from core.embeddings import Embeddings
from core.qdrant_client import QdrantDB

# Pipeline imports for full ingestion
try:
    from pipeline_ingest import (
        chunk_text, extract_entities, embed_texts,
        store_to_qdrant, store_to_neo4j, QDRANT_COLLECTION
    )
    PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

app = FastAPI(
    title="Vector+Graph Hybrid Database API",
    description="Production-grade CRUD + Hybrid Search for AI Retrieval",
    version="2.0"
)

# ==================== INITIALIZATION ====================
embedder = Embeddings()
lmdb_store = LMDBMetaStore(path=os.environ.get("LMDB_PATH", "data/lmdb_meta"))

try:
    qdrant_db = QdrantDB(
        collection=os.environ.get("QDRANT_COLLECTION", "docs_chunks"),
        vector_size=384,
        path=os.environ.get("QDRANT_PATH", "data/qdrant_db")
    )
    QDRANT_AVAILABLE = True
except Exception as e:
    print(f"Qdrant initialization warning: {e}")
    QDRANT_AVAILABLE = False
    qdrant_db = None

# Neo4j connection (optional)
try:
    from neo4j import GraphDatabase
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    NEO4J_AVAILABLE = True
except Exception as e:
    print(f"Neo4j not available: {e}")
    NEO4J_AVAILABLE = False
    neo_driver = None

# ==================== PYDANTIC MODELS ====================
class NodeCreate(BaseModel):
    text: str = Field(..., description="Text content of the node")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    node_type: Optional[str] = Field(default="document", description="Type of node")
    title: Optional[str] = Field(default=None, description="Title of the document")

class NodeUpdate(BaseModel):
    text: Optional[str] = Field(default=None, description="Updated text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")

class EdgeCreate(BaseModel):
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relation_type: str = Field(..., description="Type of relationship")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Edge properties")

class VectorSearchRequest(BaseModel):
    query_text: str = Field(..., description="Text query for semantic search")
    top_k: int = Field(default=5, description="Number of results")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

class GraphTraversalRequest(BaseModel):
    start_node_id: str = Field(..., description="Starting node ID")
    depth: int = Field(default=1, description="Traversal depth")
    relation_filter: Optional[str] = Field(default=None, description="Filter by relation type")

class HybridSearchRequest(BaseModel):
    query_text: str = Field(..., description="Text query")
    top_k: int = Field(default=5, description="Number of results")
    vector_weight: float = Field(default=0.7, description="Weight for vector similarity (0-1)")
    graph_weight: float = Field(default=0.3, description="Weight for graph proximity (0-1)")
    include_neighbors: bool = Field(default=True, description="Include neighboring nodes")

class CustomQueryRequest(BaseModel):
    query: str = Field(..., description="Custom query in HX-QL format")

# ==================== HELPER FUNCTIONS ====================
def structure_node_response(node_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw metadata to structured node response"""
    return {
        "node_id": node_id,
        "type": metadata.get("type", "document"),
        "title": metadata.get("title", ""),
        "text_snippet": metadata.get("text", "")[:200] + "..." if len(metadata.get("text", "")) > 200 else metadata.get("text", ""),
        "entities": metadata.get("entities", []),
        "keywords": metadata.get("keywords", []),
        "created_at": metadata.get("created_at"),
        "metadata": {k: v for k, v in metadata.items() if k not in ["text", "vector", "id"]}
    }

def extract_structured_entities(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract structured entities from text"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
        doc = nlp(text)
        
        entities_by_type = {}
        for ent in doc.ents:
            entity_dict = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            if ent.label_ not in entities_by_type:
                entities_by_type[ent.label_] = []
            entities_by_type[ent.label_].append(entity_dict)
        
        return entities_by_type
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return {}

# ==================== HEALTH & STATUS ====================
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "services": {
            "lmdb": "available",
            "qdrant": "available" if QDRANT_AVAILABLE else "unavailable",
            "neo4j": "available" if NEO4J_AVAILABLE else "unavailable",
            "embedder": "available"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "total_nodes": 0,
        "total_vectors": 0,
        "storage_backend": {
            "lmdb": "active",
            "qdrant": QDRANT_AVAILABLE,
            "neo4j": NEO4J_AVAILABLE
        }
    }
    
    # Count LMDB entries
    try:
        count = 0
        if hasattr(lmdb_store, '_mem'):
            count = len(lmdb_store._mem)
        elif hasattr(lmdb_store, 'env') and lmdb_store.env:
            with lmdb_store.env.begin(write=False) as txn:
                count = txn.stat()['entries']
        stats["total_nodes"] = count
    except Exception:
        pass
    
    return stats

# ==================== NODE CRUD ====================
@app.post("/api/v1/nodes", response_model=Dict[str, Any])
async def create_node(node: NodeCreate):
    """
    CREATE: Add a new node with text, embeddings, and metadata
    Returns structured JSON with node details
    """
    try:
        # Generate unique node ID
        node_id = f"node_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Extract entities
        entities = extract_structured_entities(node.text)
        
        # Generate embedding
        embedding = None
        if QDRANT_AVAILABLE and qdrant_db:
            try:
                embedding = embedder.embed(node.text)
            except Exception as e:
                print(f"Embedding generation failed: {e}")
        
        # Build structured metadata
        metadata = {
            "id": node_id,
            "type": node.node_type,
            "title": node.title or f"Document {node_id}",
            "text": node.text,
            "entities": entities,
            "keywords": list(entities.keys()) if entities else [],
            "created_at": timestamp,
            "updated_at": timestamp,
            **(node.metadata or {})
        }
        
        if embedding:
            metadata["vector"] = embedding
        
        # Store in LMDB
        lmdb_store.put(node_id, metadata)
        
        # Store in Qdrant if available
        if QDRANT_AVAILABLE and qdrant_db and embedding:
            try:
                qdrant_db.upsert_vector(
                    vector_id=node_id,
                    vector=embedding,
                    payload={"node_id": node_id, "text": node.text, **metadata}
                )
            except Exception as e:
                print(f"Qdrant upsert failed: {e}")
        
        # Return structured response
        return {
            "status": "created",
            "node": structure_node_response(node_id, metadata),
            "storage": {
                "lmdb": True,
                "qdrant": QDRANT_AVAILABLE and embedding is not None,
                "neo4j": False  # Can be enabled if needed
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node creation failed: {str(e)}")

@app.get("/api/v1/nodes/{node_id}", response_model=Dict[str, Any])
async def get_node(node_id: str):
    """
    READ: Get node by ID with structured output
    """
    try:
        metadata = lmdb_store.get(node_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Get relationships from Neo4j if available
        relationships = []
        if NEO4J_AVAILABLE and neo_driver:
            try:
                with neo_driver.session() as session:
                    result = session.run(
                        """
                        MATCH (n {node_id: $nid})-[r]-(m)
                        RETURN type(r) as rel_type, m.node_id as connected_id, 
                               labels(m) as labels
                        LIMIT 10
                        """,
                        nid=node_id
                    )
                    relationships = [dict(record) for record in result]
            except Exception as e:
                print(f"Neo4j relationship query failed: {e}")
        
        response = structure_node_response(node_id, metadata)
        response["relationships"] = relationships
        response["neighbor_count"] = len(relationships)
        
        return {
            "status": "success",
            "node": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve node: {str(e)}")

@app.put("/api/v1/nodes/{node_id}", response_model=Dict[str, Any])
async def update_node(node_id: str, update: NodeUpdate):
    """
    UPDATE: Modify existing node
    """
    try:
        # Get existing metadata
        metadata = lmdb_store.get(node_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Update fields
        if update.text:
            metadata["text"] = update.text
            # Regenerate embedding
            if QDRANT_AVAILABLE and qdrant_db:
                try:
                    new_embedding = embedder.embed(update.text)
                    metadata["vector"] = new_embedding
                    qdrant_db.upsert_vector(
                        vector_id=node_id,
                        vector=new_embedding,
                        payload={"node_id": node_id, "text": update.text}
                    )
                except Exception as e:
                    print(f"Embedding update failed: {e}")
            
            # Re-extract entities
            metadata["entities"] = extract_structured_entities(update.text)
        
        if update.metadata:
            metadata.update(update.metadata)
        
        metadata["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Save updated metadata
        lmdb_store.put(node_id, metadata)
        
        return {
            "status": "updated",
            "node": structure_node_response(node_id, metadata)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@app.delete("/api/v1/nodes/{node_id}")
async def delete_node(node_id: str):
    """
    DELETE: Remove node and all associated data
    """
    try:
        # Check if node exists
        metadata = lmdb_store.get(node_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Delete from Qdrant
        if QDRANT_AVAILABLE and qdrant_db:
            try:
                qdrant_db.delete_vector(node_id)
            except Exception as e:
                print(f"Qdrant deletion warning: {e}")
        
        # Delete from LMDB
        if hasattr(lmdb_store, 'env') and lmdb_store.env:
            with lmdb_store.env.begin(write=True) as txn:
                txn.delete(node_id.encode('utf-8'))
        elif hasattr(lmdb_store, '_mem'):
            lmdb_store._mem.pop(node_id, None)
        
        # Delete from Neo4j if available
        if NEO4J_AVAILABLE and neo_driver:
            try:
                with neo_driver.session() as session:
                    session.run(
                        "MATCH (n {node_id: $nid}) DETACH DELETE n",
                        nid=node_id
                    )
            except Exception as e:
                print(f"Neo4j deletion warning: {e}")
        
        return {
            "status": "deleted",
            "node_id": node_id,
            "deleted_from": {
                "lmdb": True,
                "qdrant": QDRANT_AVAILABLE,
                "neo4j": NEO4J_AVAILABLE
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

# ==================== EDGE/RELATIONSHIP CRUD ====================
@app.post("/api/v1/edges", response_model=Dict[str, Any])
async def create_edge(edge: EdgeCreate):
    """
    CREATE: Create relationship between nodes
    """
    try:
        # Verify both nodes exist
        source_meta = lmdb_store.get(edge.source_id)
        target_meta = lmdb_store.get(edge.target_id)
        
        if not source_meta or not target_meta:
            raise HTTPException(status_code=404, detail="Source or target node not found")
        
        edge_id = f"edge_{uuid.uuid4().hex[:12]}"
        
        # Store edge in Neo4j if available
        if NEO4J_AVAILABLE and neo_driver:
            try:
                with neo_driver.session() as session:
                    session.run(
                        f"""
                        MATCH (a {{node_id: $src}}), (b {{node_id: $tgt}})
                        CREATE (a)-[r:{edge.relation_type} {{edge_id: $eid}}]->(b)
                        SET r += $props
                        """,
                        src=edge.source_id,
                        tgt=edge.target_id,
                        eid=edge_id,
                        props=edge.properties or {}
                    )
            except Exception as e:
                print(f"Neo4j edge creation failed: {e}")
        
        return {
            "status": "created",
            "edge": {
                "edge_id": edge_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "properties": edge.properties
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edge creation failed: {str(e)}")

@app.get("/api/v1/edges/{edge_id}")
async def get_edge(edge_id: str):
    """
    READ: Get edge details
    """
    if not NEO4J_AVAILABLE or not neo_driver:
        raise HTTPException(status_code=501, detail="Neo4j not available")
    
    try:
        with neo_driver.session() as session:
            result = session.run(
                """
                MATCH (a)-[r {edge_id: $eid}]->(b)
                RETURN r, a.node_id as source, b.node_id as target, type(r) as rel_type
                """,
                eid=edge_id
            )
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail="Edge not found")
            
            return {
                "status": "success",
                "edge": {
                    "edge_id": edge_id,
                    "source_id": record["source"],
                    "target_id": record["target"],
                    "relation_type": record["rel_type"],
                    "properties": dict(record["r"])
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve edge: {str(e)}")

# ==================== SEARCH OPERATIONS ====================
@app.post("/api/v1/search/vector", response_model=Dict[str, Any])
async def vector_search(request: VectorSearchRequest):
    """
    Vector-only semantic search
    """
    try:
        # Generate query embedding
        query_vector = embedder.embed(request.query_text)
        
        # Search Qdrant
        if not QDRANT_AVAILABLE or not qdrant_db:
            raise HTTPException(status_code=503, detail="Qdrant not available")
        
        hits = qdrant_db.search_vectors(query_vector, top_k=request.top_k)
        
        # Enrich results with metadata
        results = []
        for hit in hits:
            node_id = str(hit.get("vector_id", hit.get("id")))
            metadata = lmdb_store.get(node_id) or {}
            
            results.append({
                "node_id": node_id,
                "score": hit.get("score"),
                "relevance": "high" if hit.get("score", 0) > 0.8 else "medium" if hit.get("score", 0) > 0.6 else "low",
                "content": structure_node_response(node_id, metadata)
            })
        
        return {
            "status": "success",
            "query": request.query_text,
            "search_type": "vector_only",
            "result_count": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.post("/api/v1/search/graph", response_model=Dict[str, Any])
async def graph_traversal(request: GraphTraversalRequest):
    """
    Graph-only traversal search
    """
    if not NEO4J_AVAILABLE or not neo_driver:
        raise HTTPException(status_code=501, detail="Neo4j not available for graph traversal")
    
    try:
        with neo_driver.session() as session:
            rel_filter = f":{request.relation_filter}" if request.relation_filter else ""
            
            result = session.run(
                f"""
                MATCH path = (start {{node_id: $start_id}})-[{rel_filter}*1..{request.depth}]-(connected)
                RETURN DISTINCT connected.node_id as node_id, 
                       length(path) as distance,
                       [rel in relationships(path) | type(rel)] as path_relations
                LIMIT 50
                """,
                start_id=request.start_node_id
            )
            
            nodes = []
            for record in result:
                node_id = record["node_id"]
                metadata = lmdb_store.get(node_id) or {}
                nodes.append({
                    "node_id": node_id,
                    "distance": record["distance"],
                    "path_relations": record["path_relations"],
                    "content": structure_node_response(node_id, metadata)
                })
            
            return {
                "status": "success",
                "start_node_id": request.start_node_id,
                "search_type": "graph_traversal",
                "depth": request.depth,
                "result_count": len(nodes),
                "results": nodes
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph traversal failed: {str(e)}")

@app.post("/api/v1/search/hybrid", response_model=Dict[str, Any])
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search combining vector similarity and graph structure
    This demonstrates improvement over single-mode retrieval
    """
    try:
        # Step 1: Vector search
        query_vector = embedder.embed(request.query_text)
        vector_hits = qdrant_db.search_vectors(query_vector, top_k=request.top_k * 2) if QDRANT_AVAILABLE and qdrant_db else []
        
        # Step 2: For each vector hit, get graph neighbors if requested
        combined_results = {}
        
        for hit in vector_hits:
            node_id = str(hit.get("vector_id", hit.get("id")))
            vector_score = hit.get("score", 0)
            
            # Get metadata
            metadata = lmdb_store.get(node_id) or {}
            
            # Initialize with vector score
            combined_results[node_id] = {
                "node_id": node_id,
                "vector_score": vector_score,
                "graph_score": 0,
                "hybrid_score": 0,
                "content": structure_node_response(node_id, metadata),
                "neighbors": []
            }
            
            # Add graph neighbors if Neo4j available
            if request.include_neighbors and NEO4J_AVAILABLE and neo_driver:
                try:
                    with neo_driver.session() as session:
                        result = session.run(
                            """
                            MATCH (n {node_id: $nid})-[r]-(neighbor)
                            RETURN neighbor.node_id as neighbor_id, type(r) as rel_type
                            LIMIT 5
                            """,
                            nid=node_id
                        )
                        
                        neighbors = [dict(record) for record in result]
                        combined_results[node_id]["neighbors"] = neighbors
                        combined_results[node_id]["graph_score"] = len(neighbors) * 0.1  # Simple graph score
                except Exception as e:
                    print(f"Neighbor fetch failed for {node_id}: {e}")
        
        # Step 3: Calculate hybrid scores
        for node_id, data in combined_results.items():
            data["hybrid_score"] = (
                data["vector_score"] * request.vector_weight +
                data["graph_score"] * request.graph_weight
            )
        
        # Step 4: Sort by hybrid score and return top-k
        ranked_results = sorted(
            combined_results.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:request.top_k]
        
        return {
            "status": "success",
            "query": request.query_text,
            "search_type": "hybrid",
            "weights": {
                "vector": request.vector_weight,
                "graph": request.graph_weight
            },
            "result_count": len(ranked_results),
            "results": ranked_results,
            "improvement_note": "Hybrid search combines semantic similarity with graph structure for better relevance"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

# ==================== CUSTOM QUERY LANGUAGE ====================
@app.post("/api/v1/query/custom", response_model=Dict[str, Any])
async def custom_query(request: CustomQueryRequest):
    """
    Execute custom query using HX-QL (your custom query language)
    Examples:
    - FIND NODES WHERE type="document" LIMIT 5
    - GET NODE node_id="node_abc123"
    - SEARCH VECTOR EMBED("machine learning") K 3
    - FIND PATH FROM node_id="A" TO node_id="B" MAXHOPS 5
    """
    try:
        # Parse query
        ast = parse(request.query)
        
        # Execute using your executor
        executor = QueryExecutor(lmdb_store=lmdb_store, graph=None)
        result = executor.execute(ast)
        
        # Structure the result
        structured_result = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "id" in item:
                    node_id = item["id"]
                    metadata = lmdb_store.get(node_id) or item
                    structured_result.append(structure_node_response(node_id, metadata))
                else:
                    structured_result.append(item)
        else:
            structured_result = result
        
        return {
            "status": "success",
            "query": request.query,
            "ast": ast,
            "result_count": len(structured_result) if isinstance(structured_result, list) else 1,
            "results": structured_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)