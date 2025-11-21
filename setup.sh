#!/bin/bash

echo "=========================================="
echo "Universal Media Enhancer - Setup Script"
echo "=========================================="
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p src/image_processing
mkdir -p src/signal_processing
mkdir -p src/metrics
mkdir -p output
mkdir -p data/sample_images
mkdir -p data/sample_audio

# Create __init__.py files
echo "Creating Python package files..."
touch src/__init__.py
touch src/image_processing/__init__.py
touch src/signal_processing/__init__.py
touch src/metrics/__init__.py

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your DSP/DIP toolkit modules to src/"
echo "2. Add sample data to data/ folder (optional)"
echo "3. Run: python main.py"
echo ""
