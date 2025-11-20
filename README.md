# FOSS4G 2025 GeoFM Demo - Loading via odc-stac into TerraTorch with TerraMind

This repository demonstrates how to load geospatial data using **odc-stac** (Open Data Cube STAC) and integrate it with **TerraTorch** for geospatial foundation model workflows, featuring **IBM's TerraMind** foundation model. This demo is designed for the FOSS4G 2025 conference to showcase modern cloud-native geospatial data processing and state-of-the-art AI pipelines.

✨ **NEW**: Now includes working **TerraMind v1** integration with the latest TerraTorch from GitHub!

## Overview

The project showcases the integration between:
- **STAC (SpatioTemporal Asset Catalog)**: A specification for describing geospatial assets
- **odc-stac**: Python library for loading STAC items into xarray Datasets
- **TerraTorch**: IBM's toolkit for fine-tuning Geospatial Foundation Models (GFMs)
- **TerraMind v1**: IBM's state-of-the-art geospatial foundation model for generating 768-dimensional embeddings from 16x16 Sentinel-2 RGB patches

## Key Features

- 🌍 Load multi-temporal satellite imagery from STAC catalogs (Element84 Earth Search, Microsoft Planetary Computer)
- 📊 Convert STAC data to xarray Datasets for efficient processing with odc-stac
- 🤖 **TerraMind v1 Integration**: IBM's geospatial foundation model with 768-dimensional embeddings
- 🎯 **16x16 Patch Processing**: Optimized for TerraMind's designed input format
- 🧠 Smart model fallback: TerraMind → Clay → Prithvi → ResNet18 for maximum compatibility
- 📡 Support for major satellite data sources (Sentinel-2, Landsat, MODIS, etc.)
- ⚡ Cloud-native, scalable geospatial ML pipelines with configurable chunking
- 📈 Interactive visualization with PCA/t-SNE analysis of embedding spaces
- 🔧 Latest TerraTorch from GitHub with full TerraMind model registry

## Quick Start

### Installation

**Prerequisites**: Python 3.11+

```bash
# Clone the repository
git clone <repository-url>
cd foss4g_2025_geofm

# Option 1: Quick setup with latest TerraTorch (recommended)
pip install "git+https://github.com/terrastackai/terratorch.git"
pip install -e .

# Option 2: Development installation with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install "git+https://github.com/terrastackai/terratorch.git"
uv pip install -e ".[dev]"

# Option 3: Complete development setup
./setup-dev.sh
```

**Important**: TerraMind models require the latest TerraTorch from GitHub (not PyPI) for full model registry support.

### Development Setup

For contributors and developers:

```bash
# Quick setup with pre-commit hooks and notebook cleaning
./setup-dev.sh

# This will:
# - Create/activate a virtual environment with uv
# - Install all dependencies from pyproject.toml
# - Set up pre-commit hooks for code formatting
# - Configure automatic notebook cleaning
```

### Optional Dependencies

```bash
# Install additional visualization tools
uv pip install -e ".[vis]"

# Install additional ML tools
uv pip install -e ".[ml]"

# Install everything for full development
uv pip install -e ".[dev,vis,ml]"
```

### Basic Usage

```python
import pystac_client
import odc.stac

# Connect to a STAC catalog
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

# Search for Sentinel-2 data
search = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-06-30",
    bbox=[-74.2, 40.6, -73.7, 41.0],  # New York area
    limit=5
)

# Load into xarray Dataset
dataset = odc.stac.load(
    search.items(),
    bands=["red", "green", "blue", "nir"],
    resolution=100,  # 100m resolution
    chunks={"time": 1, "x": 1024, "y": 1024}
)

# Basic visualization
dataset.red.isel(time=0).plot()
```

### TerraTorch Integration

```python
import terratorch
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

# Configure data module for TerraTorch
datamodule = GenericNonGeoSegmentationDataModule(
    # Configuration will depend on your specific use case
    # See examples in the notebooks directory
)

# Example model creation (configuration-dependent)
model = terratorch.models.create_model({
    # Model configuration
})
```

### TerraMind Embedding Generation

```python
# Generate embeddings with IBM TerraMind v1 foundation model
from terratorch.registry import BACKBONE_REGISTRY
import torch
import numpy as np

# Load TerraMind v1 Base model (768D embeddings)
model = BACKBONE_REGISTRY.build(
    'terratorch_terramind_v1_base',
    modalities=['S2RGB'],  # 16x16 RGB patches
    pretrained=True
)
model.eval()

# Create RGB composite from STAC data
rgb_composite = dataset.red.isel(time=-1), dataset.green.isel(time=-1), dataset.blue.isel(time=-1)
rgb_array = np.stack(rgb_composite, axis=-1) / 10000.0  # Convert to [0,1]

# Extract 16x16 patches (TerraMind's designed input)
patches = extract_16x16_patches(rgb_array)
patches_tensor = prepare_terramind_input(patches)  # Normalize for model

# Generate embeddings
with torch.no_grad():
    embeddings = model({'S2RGB': patches_tensor})
    embeddings = embeddings[-1].squeeze(1)  # Use last layer, remove sequence dim

print(f"Generated {embeddings.shape[0]} TerraMind embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")
# Generated 8908 TerraMind embeddings
# Embedding dimension: 768
```

## Examples

The repository includes several example notebooks and scripts:

- Basic STAC data loading with odc-stac
- Multi-temporal analysis workflows
- TerraTorch model training examples
- TerraMind embedding generation
- Visualization and analysis notebooks

## Supported Data Sources

- **Element84 Earth Search**: AWS Open Data STAC catalog (primary)
- **Microsoft Planetary Computer**: Global satellite imagery catalog
- **Custom STAC APIs**: Any STAC-compliant data source

## Dependencies

All dependencies are managed in `pyproject.toml` for consistency and reproducibility:

**Core Dependencies:**
- `odc-stac>=0.3.0`: STAC data loading into xarray
- `terratorch @ git+https://github.com/terrastackai/terratorch.git`: Latest TerraTorch with TerraMind support
- `xarray>=2022.12.0`: Multi-dimensional labeled arrays
- `pystac-client>=0.7.0`: STAC catalog interaction
- `rasterio>=1.3.0`: Geospatial raster data I/O
- `torch>=2.0.0`: Deep learning framework
- `huggingface_hub>=0.16.0`: For downloading TerraMind pretrained weights
- `holoviews>=1.17.0`: Interactive visualization

**Development Dependencies:**
- `ruff>=0.14.5`: Fast Python linter and formatter
- `pre-commit>=3.0.0`: Git hooks for code quality
- `pytest>=7.0.0`: Testing framework

**Installation:**
```bash
# Install latest TerraTorch first (required for TerraMind)
pip install "git+https://github.com/terrastackai/terratorch.git"

# Production dependencies
pip install -e .

# With development tools
uv pip install -e ".[dev]"

# With optional visualization/ML tools
uv pip install -e ".[dev,vis,ml]"
```

See `pyproject.toml` for the complete dependency specification.

## Performance Tips

- Use appropriate chunking for large datasets
- Leverage Dask for distributed processing
- Consider data locality when working with cloud data
- Use lazy evaluation to minimize memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Resources

- [odc-stac Documentation](https://github.com/opendatacube/odc-stac)
- [TerraTorch Documentation](https://github.com/ibm/terratorch)
- [STAC Specification](https://stacspec.org/)
- [Element84 Earth Search](https://github.com/element84/earth-search)
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## FOSS4G 2025

This demo is part of the FOSS4G 2025 conference presentation on "Modern Geospatial ML Workflows with STAC and Foundation Models". The presentation demonstrates how open standards like STAC can be integrated with cutting-edge machine learning frameworks for scalable geospatial analysis.
