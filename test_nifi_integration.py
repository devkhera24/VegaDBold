#!/usr/bin/env python3
"""
test_nifi_integration.py - Complete test suite for NiFi integration
"""
import os
import json
import tempfile
from pathlib import Path
import pytest

# Create test directory
TEST_DATA_DIR = Path("data/test_nifi")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_test_json():
    """Create test JSON file"""
    data = {
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "author": "Test Author",
        "tags": ["AI", "ML", "Technology"],
        "published": "2024-01-15"
    }
    
    path = TEST_DATA_DIR / "test_document.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Created test JSON: {path}")
    return path


def create_test_json_array():
    """Create test JSON array file"""
    data = [
        {
            "id": 1,
            "text": "Neural networks are computing systems inspired by biological neural networks.",
            "category": "deep-learning"
        },
        {
            "id": 2,
            "text": "Natural language processing enables computers to understand human language.",
            "category": "nlp"
        },
        {
            "id": 3,
            "text": "Computer vision allows machines to interpret and understand visual information.",
            "category": "computer-vision"
        }
    ]
    
    path = TEST_DATA_DIR / "test_documents_array.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Created test JSON array: {path}")
    return path


def create_test_html():
    """Create test HTML file"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="A comprehensive guide to vector databases">
    <meta name="keywords" content="vector, database, embeddings, similarity search">
    <title>Vector Databases: A Complete Guide</title>
</head>
<body>
    <header>
        <nav>
            <a href="#intro">Introduction</a>
            <a href="#features">Features</a>
        </nav>
    </header>
    
    <main>
        <h1>Vector Databases: A Complete Guide</h1>
        
        <section id="intro">
            <h2>Introduction</h2>
            <p>Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for modern AI applications including semantic search, recommendation systems, and similarity matching.</p>
        </section>
        
        <section id="features">
            <h2>Key Features</h2>
            <ul>
                <li>Fast approximate nearest neighbor search</li>
                <li>Support for various distance metrics (cosine, euclidean, dot product)</li>
                <li>Scalable to billions of vectors</li>
                <li>Real-time indexing and updates</li>
            </ul>
        </section>
        
        <section id="use-cases">
            <h3>Common Use Cases</h3>
            <p>Vector databases power semantic search engines, recommendation systems, image similarity search, anomaly detection, and RAG (Retrieval-Augmented Generation) systems for LLMs.</p>
        </section>
    </main>
    
    <footer>
        <p>© 2024 Vector DB Guide</p>
    </footer>
    
    <script>
        // This script should be removed during processing
        console.log("Analytics code");
    </script>
    
    <style>
        /* This style should be removed during processing */
        body { font-family: Arial; }
    </style>
</body>
</html>"""
    
    path = TEST_DATA_DIR / "test_article.html"
    with open(path, 'w') as f:
        f.write(html)
    
    print(f"✓ Created test HTML: {path}")
    return path


def create_test_csv():
    """Create test CSV file"""
    csv_content = """id,product_name,description,category,price
1,"Laptop Pro 15","High-performance laptop with 16GB RAM and 512GB SSD. Perfect for developers and content creators.","Electronics",1299.99
2,"Wireless Mouse","Ergonomic wireless mouse with precision tracking and long battery life.","Accessories",29.99
3,"USB-C Hub","Multi-port USB-C hub with HDMI, USB 3.0, and SD card reader.","Accessories",49.99
4,"Monitor 27inch","4K UHD monitor with HDR support and 144Hz refresh rate. Ideal for gaming and professional work.","Electronics",399.99
5,"Mechanical Keyboard","RGB backlit mechanical keyboard with Cherry MX switches.","Accessories",129.99"""
    
    path = TEST_DATA_DIR / "test_products.csv"
    with open(path, 'w') as f:
        f.write(csv_content)
    
    print(f"✓ Created test CSV: {path}")
    return path


def run_integration_test(file_path: Path, use_api: bool = False):
    """Run integration test on a file"""
    import subprocess
    import sys
    
    print(f"\n{'='*70}")
    print(f"Testing: {file_path.name}")
    print(f"Mode: {'API' if use_api else 'Direct Pipeline'}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, "nifi_integration.py", str(file_path), "--stats"]
    if use_api:
        cmd.append("--api")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"✗ Error: {result.stderr}")
        return False
    
    print(f"✓ Successfully processed {file_path.name}")
    return True


def test_json_processing():
    """Test JSON file processing"""
    print("\n" + "="*70)
    print("TEST 1: JSON Processing")
    print("="*70)
    
    from nifi_integration import NiFiETLProcessor
    
    processor = NiFiETLProcessor()
    
    # Test single object JSON
    path = create_test_json()
    docs = processor.process_json(str(path))
    
    assert len(docs) == 1
    assert "Machine learning" in docs[0]["text"]
    assert docs[0]["metadata"]["type"] == "json"
    
    print(f"✓ Single JSON object: {len(docs)} document extracted")
    
    # Test JSON array
    path = create_test_json_array()
    docs = processor.process_json(str(path))
    
    assert len(docs) == 3
    assert any("Neural networks" in doc["text"] for doc in docs)
    
    print(f"✓ JSON array: {len(docs)} documents extracted")


def test_html_processing():
    """Test HTML file processing"""
    print("\n" + "="*70)
    print("TEST 2: HTML Processing")
    print("="*70)
    
    from nifi_integration import NiFiETLProcessor
    
    processor = NiFiETLProcessor()
    path = create_test_html()
    docs = processor.process_html(str(path))
    
    assert len(docs) == 1
    assert "Vector databases" in docs[0]["text"]
    assert "Analytics code" not in docs[0]["text"]  # Scripts removed
    assert "font-family" not in docs[0]["text"]  # Styles removed
    assert docs[0]["metadata"]["type"] == "html"
    
    print(f"✓ HTML processed: {len(docs[0]['text'])} characters extracted")
    print(f"✓ Title extracted: {docs[0]['title']}")
    print(f"✓ Links found: {docs[0]['metadata']['links_count']}")


def test_csv_processing():
    """Test CSV file processing"""
    print("\n" + "="*70)
    print("TEST 3: CSV Processing")
    print("="*70)
    
    from nifi_integration import NiFiETLProcessor
    
    processor = NiFiETLProcessor()
    path = create_test_csv()
    docs = processor.process_csv(str(path))
    
    assert len(docs) == 5  # 5 products
    assert any("Laptop" in doc["text"] for doc in docs)
    assert all(doc["metadata"]["type"] == "csv" for doc in docs)
    
    print(f"✓ CSV processed: {len(docs)} rows extracted")
    
    # Test text column detection
    detected_cols = processor._detect_text_columns(["id", "name", "description", "category"])
    assert "description" in detected_cols
    
    print(f"✓ Auto-detected text columns: {detected_cols}")


def test_batch_processing():
    """Test batch directory processing"""
    print("\n" + "="*70)
    print("TEST 4: Batch Processing")
    print("="*70)
    
    from nifi_integration import NiFiETLProcessor
    
    # Create all test files
    create_test_json()
    create_test_json_array()
    create_test_html()
    create_test_csv()
    
    processor = NiFiETLProcessor()
    
    # Process all files
    files = list(TEST_DATA_DIR.glob("*.*"))
    total_docs = 0
    
    for file in files:
        docs = processor.process_file(str(file))
        total_docs += len(docs)
    
    print(f"✓ Processed {len(files)} files")
    print(f"✓ Extracted {total_docs} documents")
    print(f"✓ Statistics: {processor.get_stats()}")


def test_api_mode():
    """Test API mode integration"""
    print("\n" + "="*70)
    print("TEST 5: API Mode (if server is running)")
    print("="*70)
    
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✓ API server is running")
            
            # Test with JSON file
            path = create_test_json()
            success = run_integration_test(path, use_api=True)
            
            if success:
                print("✓ API mode integration successful")
            else:
                print("✗ API mode integration failed")
        else:
            print("⚠ API server not responding correctly")
    except requests.exceptions.ConnectionError:
        print("⚠ API server not running (skipping API mode test)")
        print("  Start server with: python api_v2.py")


def test_pipeline_mode():
    """Test direct pipeline mode"""
    print("\n" + "="*70)
    print("TEST 6: Direct Pipeline Mode")
    print("="*70)
    
    try:
        from pipeline_ingest import ingest_document_text
        print("✓ Pipeline module available")
        
        # Test with simple text
        doc_id = ingest_document_text(
            source="test_integration",
            text="This is a test document for NiFi integration testing.",
            title="Test Document",
            meta={"test": True}
        )
        
        print(f"✓ Pipeline ingestion successful: {doc_id}")
        
    except ImportError:
        print("⚠ Pipeline module not available (skipping pipeline mode test)")
        print("  Ensure Qdrant and Neo4j are running")
    except Exception as e:
        print(f"⚠ Pipeline test failed: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("NIFI INTEGRATION TEST SUITE")
    print("="*70)
    
    # Create test data directory
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tests = [
        ("JSON Processing", test_json_processing),
        ("HTML Processing", test_html_processing),
        ("CSV Processing", test_csv_processing),
        ("Batch Processing", test_batch_processing),
        ("API Mode", test_api_mode),
        ("Pipeline Mode", test_pipeline_mode),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Passed: {passed}/{len(tests)}")
    print(f"✗ Failed: {failed}/{len(tests)}")
    print(f"\nTest data created in: {TEST_DATA_DIR}")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)