#!/bin/bash
# quickstart_nifi.sh - Quick setup for NiFi integration

set -e

echo "=================================================="
echo "NiFi ETL Integration - Quick Start"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/nifi_input
mkdir -p data/nifi_processed
mkdir -p data/lmdb_meta
mkdir -p data/test_nifi

echo -e "${GREEN}✓ Directories created${NC}"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q beautifulsoup4 lxml requests 2>/dev/null || true

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create sample files
echo ""
echo "Creating sample test files..."

# Sample JSON
cat > data/nifi_input/sample.json << 'EOF'
{
  "title": "Introduction to Vector Databases",
  "content": "Vector databases are specialized systems designed to store and query high-dimensional vectors efficiently. They enable semantic search, similarity matching, and AI-powered applications.",
  "author": "Tech Writer",
  "category": "Database Technology",
  "tags": ["vectors", "databases", "AI", "search"]
}
EOF

# Sample HTML
cat > data/nifi_input/sample.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Graph Databases Explained</title>
    <meta name="description" content="Understanding graph databases and their use cases">
</head>
<body>
    <h1>Graph Databases Explained</h1>
    <p>Graph databases use graph structures with nodes, edges, and properties to represent and store data. They excel at managing highly connected data and complex relationships.</p>
    <h2>Key Benefits</h2>
    <ul>
        <li>Efficient traversal of relationships</li>
        <li>Flexible schema</li>
        <li>Natural representation of connected data</li>
    </ul>
</body>
</html>
EOF

# Sample CSV
cat > data/nifi_input/sample.csv << 'EOF'
id,title,description,category
1,"Machine Learning Basics","Introduction to ML algorithms and concepts","AI"
2,"Deep Learning Guide","Comprehensive guide to neural networks","AI"
3,"NLP Fundamentals","Natural language processing essentials","AI"
EOF

echo -e "${GREEN}✓ Sample files created in data/nifi_input/${NC}"

# Check if services are running
echo ""
echo "Checking services..."

API_RUNNING=false
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API server is running${NC}"
    API_RUNNING=true
else
    echo -e "${YELLOW}⚠ API server is not running${NC}"
    echo "  Start with: python api_v2.py"
fi

QDRANT_RUNNING=false
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant is running${NC}"
    QDRANT_RUNNING=true
else
    echo -e "${YELLOW}⚠ Qdrant is not running${NC}"
fi

# Test processing
echo ""
echo "=================================================="
echo "Testing NiFi Integration"
echo "=================================================="

if [ "$API_RUNNING" = true ]; then
    echo ""
    echo "Test 1: Processing JSON (API mode)..."
    python3 nifi_integration.py data/nifi_input/sample.json --api
    
    echo ""
    echo "Test 2: Processing HTML (API mode)..."
    python3 nifi_integration.py data/nifi_input/sample.html --api
    
    echo ""
    echo "Test 3: Processing CSV (API mode)..."
    python3 nifi_integration.py data/nifi_input/sample.csv --api
    
    echo ""
    echo -e "${GREEN}✓ All tests completed successfully!${NC}"
    
elif [ "$QDRANT_RUNNING" = true ]; then
    echo ""
    echo "Using direct pipeline mode..."
    python3 nifi_integration.py data/nifi_input/sample.json
    echo -e "${GREEN}✓ Processing completed${NC}"
    
else
    echo ""
    echo -e "${YELLOW}Testing file processing only (no ingestion)...${NC}"
    python3 test_nifi_integration.py
fi

# Summary
echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "📁 Sample files created in: data/nifi_input/"
echo "📁 Processed files will go to: data/nifi_processed/"
echo ""
echo "Quick Commands:"
echo "  • Process single file:    python nifi_integration.py data/nifi_input/sample.json"
echo "  • Process directory:      python nifi_integration.py data/nifi_input/"
echo "  • Use API mode:           python nifi_integration.py [file] --api"
echo "  • Run tests:              python test_nifi_integration.py"
echo ""

if [ "$API_RUNNING" = false ]; then
    echo -e "${YELLOW}To enable full integration:${NC}"
    echo "  1. Start API: python api_v2.py"
    echo "  2. Or start services: docker-compose -f docker-compose.nifi.yml up -d"
    echo ""
fi

echo "For NiFi setup, see: nifi_setup_guide.md"
echo "=================================================="