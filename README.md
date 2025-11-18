# FOSS4G 2025 GeoFM Demo - Loading via odc-stac into TerraTorch with TerraMind

This repository demonstrates how to load geospatial data using **odc-stac** (Open Data Cube STAC) and integrate it with **TerraTorch** for geospatial foundation model workflows, including **TerraMind** embedding generation. This demo is designed for the FOSS4G 2025 conference to showcase modern cloud-native geospatial data processing and AI pipelines.

## Overview

The project showcases the integration between:
- **STAC (SpatioTemporal Asset Catalog)**: A specification for describing geospatial assets
- **odc-stac**: Python library for loading STAC items into xarray Datasets
- **TerraTorch**: IBM's toolkit for fine-tuning Geospatial Foundation Models (GFMs)
- **TerraMind**: Geospatial foundation model for generating rich embeddings from satellite imagery

## Key Features

- 🌍 Load multi-temporal satellite imagery from STAC catalogs
- 📊 Convert STAC data to xarray Datasets for efficient processing
- 🤖 Integrate with TerraTorch for machine learning workflows
- 🧠 Generate 768-dimensional embeddings using TerraMind foundation model
- 📡 Support for major satellite data sources (Sentinel-2, Landsat, etc.)
- ⚡ Cloud-native, scalable geospatial ML pipelines
- 📈 Comprehensive visualization and analysis tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd foss4g_2025_geofm

# Install dependencies
pip install -e .
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
# Generate embeddings with TerraMind foundation model
from main import load_terramind_model, generate_terramind_embeddings, create_rgb_composite

# Load TerraMind model
model = load_terramind_model()

# Create RGB composite from STAC data
rgb_composite = create_rgb_composite(dataset, time_index=-1)
rgb_array = rgb_composite.values

# Generate 768-dimensional embeddings
embeddings, metadata = generate_terramind_embeddings(
    rgb_array, 
    model, 
    patch_size=16, 
    batch_size=32
)

print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
# Generated 1024 embeddings of dimension 768
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

Core dependencies:
- `odc-stac>=0.3.0`: STAC data loading
- `terratorch>=0.1.0`: Geospatial foundation models
- `xarray>=2022.12.0`: Multi-dimensional arrays
- `pystac-client>=0.7.0`: STAC catalog interaction
- `rasterio>=1.3.0`: Geospatial data I/O
- `pytorch>=2.0.0`: Deep learning framework

See `pyproject.toml` for complete dependency list.

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