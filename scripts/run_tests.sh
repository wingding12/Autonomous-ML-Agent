#!/bin/bash

# Autonomous ML Agent Test Runner Script
# This script runs the test suite for the project

set -e

echo "🧪 Running Autonomous ML Agent Tests..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install test dependencies if not already installed
echo "📥 Installing test dependencies..."
pip install -r requirements.txt

# Run tests with coverage
echo "🚀 Running tests with coverage..."
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

echo "✅ Tests completed!"
echo ""
echo "Coverage report generated in htmlcov/index.html"
echo "Open the file in your browser to view detailed coverage information"
