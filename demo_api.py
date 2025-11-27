#!/usr/bin/env python3
"""
demo_api.py - Comprehensive API demonstration
Shows all CRUD operations and search capabilities
"""
import requests
import json
import time

API_BASE = "http://localhost:8000/api/v1"

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(result):
    print(json.dumps(result, indent=2))

def main():
    print_section("VECTOR+GRAPH DATABASE API DEMO")
    
    # 1. CREATE NODES
    print_section("1. CREATE - Adding Nodes")
    
    nodes = []
    sample_texts = [
        {
            "text": "Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
            "title": "ML Introduction",
            "type": "concept"
        },
        {
            "text": "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "title": "Neural Networks",
            "type": "concept"
        },
        {
            "text": "Deep learning uses multiple layers in neural networks to progressively extract higher level features.",
            "title": "Deep Learning",
            "type": "concept"
        }
    ]
    
    for doc in sample_texts:
        response = requests.post(
            f"{API_BASE}/nodes",
            json={
                "text": doc["text"],
                "title": doc["title"],
                "node_type": doc["type"]
            }
        )
        result = response.json()
        nodes.append(result["node"]["node_id"])
        print(f"\n✓ Created: {result['node']['title']} (ID: {result['node']['node_id']})")
    
    time.sleep(0.5)
    
    # 2. READ NODES
    print_section("2. READ - Retrieving Node Details")
    
    response = requests.get(f"{API_BASE}/nodes/{nodes[0]}")
    print_result(response.json())
    
    # 3. CREATE RELATIONSHIPS
    print_section("3. CREATE - Adding Relationships")
    
    if len(nodes) >= 2:
        response = requests.post(
            f"{API_BASE}/edges",
            json={
                "source_id": nodes[0],
                "target_id": nodes[1],
                "relation_type": "RELATES_TO",
                "properties": {"strength": "high"}
            }
        )
        print_result(response.json())
    
    # 4. VECTOR SEARCH
    print_section("4. VECTOR SEARCH - Semantic Similarity")
    
    response = requests.post(
        f"{API_BASE}/search/vector",
        json={
            "query_text": "What is artificial intelligence and deep learning?",
            "top_k": 3
        }
    )
    print_result(response.json())
    
    # 5. HYBRID SEARCH
    print_section("5. HYBRID SEARCH - Vector + Graph")
    
    response = requests.post(
        f"{API_BASE}/search/hybrid",
        json={
            "query_text": "neural networks and AI",
            "top_k": 3,
            "vector_weight": 0.7,
            "graph_weight": 0.3,
            "include_neighbors": True
        }
    )
    print_result(response.json())
    
    # 6. CUSTOM QUERY
    print_section("6. CUSTOM QUERY - Using HX-QL")
    
    response = requests.post(
        f"{API_BASE}/query/custom",
        json={
            "query": f'GET NODE node_id="{nodes[0]}"'
        }
    )
    print_result(response.json())
    
    # 7. UPDATE NODE
    print_section("7. UPDATE - Modifying Node")
    
    response = requests.put(
        f"{API_BASE}/nodes/{nodes[0]}",
        json={
            "text": "Machine learning is a powerful subset of AI focusing on pattern recognition and data analysis.",
            "metadata": {"updated": True}
        }
    )
    print_result(response.json())
    
    # 8. STATS
    print_section("8. SYSTEM STATISTICS")
    
    response = requests.get(f"{API_BASE}/stats")
    print_result(response.json())
    
    # 9. DELETE NODE
    print_section("9. DELETE - Removing Node")
    
    if len(nodes) >= 3:
        response = requests.delete(f"{API_BASE}/nodes/{nodes[2]}")
        print_result(response.json())
    
    print_section("DEMO COMPLETED")
    print("\n✓ All API operations demonstrated successfully!")
    print(f"\n📝 Remaining nodes: {nodes[:2]}")
    print("\nAPI Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()