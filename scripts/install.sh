#!/bin/bash

# Autonomous ML Agent Installation Script
# This script installs and sets up the Autonomous ML Agent

set -e

echo "🚀 Installing Autonomous ML Agent..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p cache/models
mkdir -p cache/experiments
mkdir -p logs
mkdir -p data

# Copy environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment configuration..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your API keys"
else
    echo "✅ Environment file already exists"
fi

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh

echo "🎉 Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Start the API: uvicorn src.main:app --reload"
echo "4. Start the UI: streamlit run src/ui/streamlit_app.py"
echo ""
echo "For more information, see the README.md file"
