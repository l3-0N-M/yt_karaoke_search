#!/bin/bash

echo "🎵 Discogs API Integration - Setup & Test"
echo "========================================"

# Check if aiohttp is installed
echo "Checking dependencies..."
python3 -c "import aiohttp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing aiohttp..."
    pip install aiohttp
fi

echo "✅ Dependencies ready"

# Check for Discogs token
if [ -z "$DISCOGS_TOKEN" ]; then
    echo ""
    echo "⚠️  DISCOGS_TOKEN not found!"
    echo ""
    echo "To get your Discogs token:"
    echo "1. Go to: https://www.discogs.com/settings/developers" 
    echo "2. Generate a Personal Access Token"
    echo "3. Run: export DISCOGS_TOKEN=your_token_here"
    echo "4. Run this script again"
    echo ""
    exit 1
fi

echo "✅ Discogs token found"

# Run the prototype test
echo ""
echo "Running Discogs API prototype test..."
echo "====================================="
python3 discogs_prototype.py

echo ""
echo "🎯 Next steps:"
echo "1. Review the test results above"
echo "2. If success rate > 80%, proceed with Phase 1"
echo "3. If issues found, investigate API limits or token permissions"