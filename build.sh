#!/bin/bash
set -e

echo "========================================"
echo "Starting Build Process"
echo "========================================"

# Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies with verbose output
echo "Installing Python dependencies..."
pip install --no-cache-dir --verbose -r requirements.txt

echo "========================================"
echo "Build Complete!"
echo "========================================"
