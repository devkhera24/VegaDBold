#!/usr/bin/env python3
"""
submit_and_query.py - Interactive document submission with structured JSON output
"""
import sys
import json
import requests
from typing import Dict, Any

API_BASE = "http://localhost:8000/api/v1"

def read_multiline_input(prompt="Paste your text:\n"):
    print(prompt)
    print("=" * 60)
    print("OPTIONS TO FINISH:")
    print("  1. Press Enter on an empty line")
    print("  2. Type 'END' on a line by itself")
    print("  3. Press Ctrl+D (Linux/Mac) or Ctrl+Z+Enter (Windows)")
    print("=" * 60)
    print()
    
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            print("\n[EOF detected - processing input...]")
            break
        
        if line.strip().upper() == "END":
            print("\n[END marker detected - processing input...]")
            break
            
        if line.strip() == "":
            if lines:
                print("\n[Empty line detected - processing input...]")
                break
            else:
                continue
        
        lines.append(line)
    
    result = "\n".join(lines).strip()
    
    if not result:
        print("\n[WARNING: No text collected]")
    else:
        print(f"\n[Collected {len(lines)} lines, {len(result)} characters]")
    
    return result

def main():
    print("\n" + "=" * 60)
    print("STRUCTURED DOCUMENT SUBMISSION & SEARCH")
    print("=" * 60 + "\n")
    
    # Get text input
    text = read_multiline_input()
    if not text:
        print("\n✗ No text provided — exiting.")
        sys.exit(0)

    print("\n" + "-" * 60)
    print("STEP 1: CREATING NODE (INGESTION)")
    print("-" * 60)
    
    # Create node via API
    try:
        response = requests.post(
            f"{API_BASE}/nodes",
            json={
                "text": text,
                "metadata": {"source": "cli_submission"},
                "node_type": "document",
                "title": "CLI Document"
            }
        )
        response.raise_for_status()
        create_result = response.json()
        
        print("\n✓ NODE CREATED:")
        print(json.dumps(create_result, indent=2))
        
        node_id = create_result["node"]["node_id"]
        
    except Exception as e:
        print(f"\n✗ Node creation failed: {e}")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("STEP 2: VECTOR SEARCH (Top-5 Similar)")
    print("-" * 60)
    
    # Perform vector search
    try:
        response = requests.post(
            f"{API_BASE}/search/vector",
            json={
                "query_text": text[:200],  # Use first 200 chars as query
                "top_k": 5
            }
        )
        response.raise_for_status()
        search_result = response.json()
        
        print("\n✓ VECTOR SEARCH RESULTS:")
        print(json.dumps(search_result, indent=2))
        
    except Exception as e:
        print(f"\n⚠ Vector search failed: {e}")

    print("\n" + "-" * 60)
    print("STEP 3: HYBRID SEARCH (Vector + Graph)")
    print("-" * 60)
    
    # Perform hybrid search
    try:
        response = requests.post(
            f"{API_BASE}/search/hybrid",
            json={
                "query_text": text[:200],
                "top_k": 5,
                "vector_weight": 0.7,
                "graph_weight": 0.3,
                "include_neighbors": True
            }
        )
        response.raise_for_status()
        hybrid_result = response.json()
        
        print("\n✓ HYBRID SEARCH RESULTS:")
        print(json.dumps(hybrid_result, indent=2))
        
    except Exception as e:
        print(f"\n⚠ Hybrid search failed: {e}")

    print("\n" + "=" * 60)
    print("✓ ALL OPERATIONS COMPLETED")
    print("=" * 60)
    print(f"\nYour node ID: {node_id}")
    print("\nYou can now:")
    print(f"  - Retrieve it: GET {API_BASE}/nodes/{node_id}")
    print(f"  - Update it: PUT {API_BASE}/nodes/{node_id}")
    print(f"  - Delete it: DELETE {API_BASE}/nodes/{node_id}")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user (Ctrl+C)")
        sys.exit(1)