#!/bin/bash
# Setup script for FOSS4G 2025 GeoFM Demo development environment

echo "🚀 Setting up FOSS4G 2025 GeoFM Demo development environment..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
elif [[ -f ".venv/pyvenv.cfg" ]]; then
    echo "🔍 Found local .venv, activating..."
    source .venv/bin/activate
    echo "✅ Activated virtual environment: $(which python)"
else
    echo "⚠️  No virtual environment detected. Creating with uv..."
    uv venv
    source .venv/bin/activate
    echo "✅ Created and activated virtual environment"
fi

# Install development dependencies using uv (faster than pip)
echo "📦 Installing development dependencies with uv..."
if command -v uv &> /dev/null; then
    echo "🔧 Installing project in development mode with dev dependencies..."
    uv pip install -e ".[dev]"
else
    echo "⚠️  uv not found, falling back to pip..."
    pip install -e ".[dev]"
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Setup nbstripout for notebooks
echo "📓 Setting up notebook cleaning with nbstripout..."
nbstripout --install --attributes .gitattributes

echo "✅ Setup completed!"
echo "📋 Pre-commit hooks are now installed and will run automatically on commits"
echo "🧹 Notebooks will be automatically cleaned (outputs and metadata removed)"
echo "⚡ Using Ruff for fast Python formatting and linting"
echo ""
echo "To manually run pre-commit on all files:"
echo "  pre-commit run --all-files"
echo ""
echo "To manually format code with Ruff:"
echo "  ruff format ."
echo "  ruff check . --fix"
echo ""
echo "To skip pre-commit for a specific commit:"
echo "  git commit --no-verify -m 'your message'"
echo ""
echo "To activate this environment in the future:"
echo "  source .venv/bin/activate"
