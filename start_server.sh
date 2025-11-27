#!/bin/bash
# start_server.sh - Start the API server

echo "Starting Vector+Graph Database API Server..."
echo "============================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q fastapi uvicorn pydantic sentence-transformers qdrant-client lmdb msgpack spacy neo4j

# Download spaCy model
python -m spacy download en_core_web_sm 2>/dev/null || true

# Start server
echo ""
echo "✓ Server starting on http://localhost:8000"
echo "✓ API Documentation: http://localhost:8000/docs"
echo "✓ Press Ctrl+C to stop"
echo ""

python api_v2.py
```

## 5. UPDATE: `.gitignore` - Add new entries

Add these lines to your existing `.gitignore`:
```
# API related
api_v2.py~
*.log

# Test data
test_data/
demo_output/