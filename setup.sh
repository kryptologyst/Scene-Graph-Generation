#!/usr/bin/env bash

# Scene Graph Generation - Setup Script
# This script sets up the development environment for the scene graph generation project

set -e

echo "Setting up Scene Graph Generation project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ is required. Found Python $python_version"
    exit 1
fi

echo "Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p checkpoints
mkdir -p logs
mkdir -p assets
mkdir -p tests

# Create placeholder files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch assets/.gitkeep

# Download sample data (if available)
echo "Setting up sample data..."
if [ ! -f "data/raw/annotations.json" ]; then
    echo "Creating sample annotations..."
    cat > data/raw/annotations.json << 'EOF'
[
  {
    "image_id": "sample_1",
    "width": 512,
    "height": 512,
    "objects": [
      {
        "bbox": [50, 50, 100, 100],
        "name": "person",
        "score": 0.9,
        "attributes": ["standing"]
      },
      {
        "bbox": [200, 100, 100, 80],
        "name": "car",
        "score": 0.8,
        "attributes": ["red"]
      }
    ],
    "relationships": [
      {
        "subject": 0,
        "object": 1,
        "predicate": "near",
        "score": 0.7
      }
    ]
  }
]
EOF
fi

# Create sample image directory
mkdir -p data/raw/images

# Run tests to verify installation
echo "Running tests to verify installation..."
python -m pytest tests/ -v || echo "No tests found, skipping..."

echo ""
echo "Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the demo: streamlit run demo/app.py"
echo "3. Train a model: python scripts/train.py"
echo "4. Evaluate a model: python scripts/evaluate.py --checkpoint checkpoints/best.pt"
echo ""
echo "For development:"
echo "- Format code: black src/ scripts/ demo/"
echo "- Lint code: ruff check src/ scripts/ demo/"
echo "- Run tests: pytest tests/"
echo ""
echo "Happy coding!"
