# Co-pilot Instructions: FOSS4G 2025 GeoFM Demo - Loading via odc-stac into TerraTorch with TerraMind

## Project Overview
This codebase demonstrates how to load geospatial data via odc-stac (Open Data Cube STAC) into TerraTorch for geospatial foundation model fine-tuning and analysis, including TerraMind embedding generation. The project serves as a practical example for the FOSS4G 2025 conference, showcasing the integration between STAC (SpatioTemporal Asset Catalog) data loading, modern geospatial machine learning workflows, and state-of-the-art foundation model embeddings.

## Core Technologies
- **odc-stac**: Loads STAC items into xarray Datasets for efficient geospatial data handling
- **TerraTorch**: IBM's Python toolkit for fine-tuning Geospatial Foundation Models (GFMs)
- **TerraMind**: IBM's geospatial foundation model for generating rich 768-dimensional embeddings
- **xarray**: Multi-dimensional labeled arrays and datasets for scientific computing
- **STAC**: SpatioTemporal Asset Catalog specification for geospatial asset metadata

## Project Structure
```
foss4g_2025_geofm/
├── main.py                          # Main demonstration script
├── config.yaml                      # Configuration file with sample areas and parameters
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project configuration
├── README.md                        # Project documentation
├── .copilot_instructions           # These instructions
├── notebooks/
│   └── odc_stac_terratorch_demo.ipynb # Complete Jupyter notebook demo
├── examples/
│   ├── basic_stac_loading.py       # Simple STAC loading example
│   ├── multitemporal_analysis.py   # Multi-temporal data analysis
│   └── terramind_embeddings.py     # TerraMind embedding generation demo
└── outputs/                        # Generated visualizations and results
```

## Key Components and Architecture

### 1. Data Loading Pipeline (odc-stac)
The primary data loading mechanism uses odc-stac to:
- Query STAC catalogs for geospatial data
- Load STAC items into xarray Datasets
- Handle multi-temporal and multi-spectral data efficiently
- Support various geospatial data sources (Sentinel-2, Landsat, etc.)

**Core Pattern:**
```python
import pystac_client
import odc.stac

# Connect to STAC catalog
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

# Search for data
query = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-06-30",
    bbox=[-74.0, 40.7, -73.9, 40.8]
)

# Load into xarray Dataset
dataset = odc.stac.load(
    query.items(),
    bands=["red", "green", "blue", "nir"],
    resolution=100,
    chunks={"time": 1, "x": 512, "y": 512}
)
```

### 2. TerraTorch Integration
TerraTorch provides the machine learning framework for:
- Processing xarray Datasets for model training
- Fine-tuning geospatial foundation models
- Handling various geospatial ML tasks (segmentation, classification, regression)
- Supporting multi-temporal data analysis

**Example Integration Pattern:**
```python
import terratorch
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

# Convert odc-stac data to TerraTorch format
class ODCSTACDataset:
    def __init__(self, dataset: xr.Dataset, tile_size: int = 256):
        self.dataset = dataset
        self.tile_size = tile_size
        # Implementation details in notebook
```

### 3. TerraMind Integration
TerraMind provides geospatial foundation model capabilities for:
- Generating 768-dimensional embeddings from satellite imagery patches
- Processing 16x16 RGB patches using Vision Transformer architecture
- Supporting Sentinel-2 RGB data specifically
- Enabling similarity analysis and feature extraction from geospatial data

**TerraMind Integration Pattern:**
```python
from terratorch.models.backbones import BACKBONE_REGISTRY

# Load TerraMind model
model = BACKBONE_REGISTRY.build(
    "terramind_v1_base",
    modalities=["S2RGB"],
    pretrained=True
)

# Generate embeddings
embeddings = model({"S2RGB": normalized_patches})
# Returns: tensor of shape (num_patches, 768)
```

### 4. Supported Data Sources
- **Element84 Earth Search**: AWS Open Data STAC catalog (primary)
- **Microsoft Planetary Computer**: Global satellite imagery catalog
- **Sentinel-2**: Multi-spectral optical imagery
- **Landsat**: Long-term Earth observation data
- **MODIS**: Moderate Resolution Imaging Spectroradiometer data
- **Custom STAC catalogs**: Any STAC-compliant data source

## Quick Start Guide

### Installation
```bash
# Clone and install
git clone <repository-url>
cd foss4g_2025_geofm
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Run the Demo
```bash
# Run the main demonstration script
python main.py

# Or explore the interactive notebook
jupyter notebook notebooks/odc_stac_terratorch_demo.ipynb

# Try the simple examples
python examples/basic_stac_loading.py
python examples/multitemporal_analysis.py
```

## Development Guidelines

### When working with odc-stac:
1. **Always specify resolution and CRS** when loading data to ensure consistency
2. **Use appropriate chunking** for large datasets to optimize memory usage
3. **Handle coordinate reference systems** properly when combining multiple data sources
4. **Consider temporal alignment** when working with multi-temporal data
5. **Use band aliases** for consistent naming across different sensors
6. **Apply cloud filtering** using STAC query parameters: `{"eo:cloud_cover": {"lt": 20}}`
7. **Use groupby="solar_day"** to handle overlapping Sentinel-2 scenes

### When integrating with TerraTorch:
1. **Ensure data format compatibility** between xarray and TerraTorch expectations
2. **Handle missing data and nodata values** appropriately
3. **Configure proper data transformations** for model input requirements
4. **Use appropriate data modules** based on the ML task (classification, segmentation, etc.)
5. **Consider GPU memory constraints** when processing large geospatial datasets
6. **Normalize reflectance values** (Sentinel-2: divide by 10000, clip to 0-1)
7. **Implement proper tile extraction** for training data preparation

### Configuration Best Practices:
1. **Use YAML/JSON configuration files** for reproducible experiments
2. **Separate data loading and model configuration** for modularity
3. **Include metadata and provenance** information in configurations
4. **Support multiple environments** (local, cloud, HPC)
5. **Configure appropriate tile sizes** based on model requirements
6. **Set reasonable chunk sizes** for memory management

## Common Workflows

### 1. Exploratory Data Analysis
```python
# Load and explore STAC data
catalog = pystac_client.Client.open(stac_url)
items = catalog.search(**search_params)
dataset = odc.stac.load(items, bands=bands, resolution=resolution)

# Basic exploration
print(dataset)
dataset.red.isel(time=0).plot()  # Quick visualization
```

### 2. Multi-temporal Analysis
```python
# Calculate NDVI time series
ndvi = (dataset.nir - dataset.red) / (dataset.nir + dataset.red)
ndvi_mean = ndvi.mean(dim=["x", "y"])

# Plot temporal profile
import matplotlib.pyplot as plt
plt.plot(ndvi.time, ndvi_mean)
plt.title("NDVI Time Series")
```

### 3. Model Training Pipeline
```python
# Load training data via odc-stac
train_data = odc.stac.load(train_items, **load_params)

# Configure TerraTorch data module
datamodule = GenericNonGeoSegmentationDataModule(
    dataset_config=dataset_config,
    transform=transforms
)

# Train model
model = terratorch.models.create_model(model_config)
trainer = pl.Trainer(**trainer_config)
trainer.fit(model, datamodule)
```

## Error Handling and Debugging

### Common Issues:
1. **CRS mismatches**: Always check and align coordinate reference systems
2. **Memory issues**: Use appropriate chunking and lazy loading
3. **Missing bands**: Verify band availability before loading
4. **Authentication**: Ensure proper credentials for private STAC catalogs
5. **Network timeouts**: Implement retry logic for large downloads
6. **Empty search results**: Verify bbox, datetime, and cloud cover parameters

### Debugging Tips:
1. **Use .compute() sparingly** with dask arrays to avoid memory issues
2. **Check data shapes and types** before feeding to models
3. **Validate STAC item properties** before batch processing
4. **Monitor memory usage** during large data loading operations
5. **Test with small areas first** before scaling to larger regions
6. **Use logging** to track data loading progress

## Example Configurations

### Sample Area Configurations (from config.yaml):
```yaml
sample_areas:
  san_francisco:
    bbox: [-122.5, 37.4, -121.8, 38.0]
    description: "San Francisco Bay Area"

  manhattan:
    bbox: [-74.0, 40.7, -73.9, 40.8]
    description: "Manhattan, New York"
```

### Data Loading Parameters:
```yaml
data_loading:
  default_resolution: 60  # meters
  chunk_size:
    time: 1
    x: 512
    y: 512
  bands:
    optical: ["red", "green", "blue", "nir"]
```

## Performance Optimization

### For large datasets:
1. **Use Dask for distributed processing** when working with multiple scenes
2. **Implement proper caching** for frequently accessed data
3. **Consider data locality** when running on cloud platforms
4. **Use efficient file formats** (COG, Zarr) when possible
5. **Parallelize data loading** across multiple workers
6. **Start with lower resolution** (60m) for prototyping

### Memory management:
1. **Load data in chunks** rather than entire datasets
2. **Use lazy evaluation** with xarray and dask
3. **Clear unused variables** explicitly in long-running processes
4. **Monitor GPU memory** usage during model training
5. **Use appropriate tile sizes** (128-512 pixels typically)

## Testing and Validation

### Data validation:
1. **Verify spatial and temporal coverage** of loaded data
2. **Check data quality and completeness**
3. **Validate coordinate reference systems**
4. **Test with different data sources and sensors**

### Model validation:
1. **Use cross-validation** for robust model evaluation
2. **Test on held-out geographic regions**
3. **Validate temporal generalization**
4. **Compare with baseline models**

## Key Files and Their Purposes

- **`main.py`**: Complete demonstration script showing the full workflow
- **`notebooks/odc_stac_terratorch_demo.ipynb`**: Interactive Jupyter notebook with detailed explanations
- **`examples/basic_stac_loading.py`**: Minimal example for beginners
- **`examples/multitemporal_analysis.py`**: Multi-temporal data analysis example
- **`config.yaml`**: Configuration file with sample areas and parameters
- **`requirements.txt`**: Dependencies list

## Contribution Guidelines
When contributing to this demo:
1. **Follow PEP 8** style guidelines
2. **Include comprehensive docstrings** for all functions
3. **Add unit tests** for new functionality
4. **Update documentation** as needed
5. **Test with multiple data sources** to ensure generalizability
6. **Include example configurations** for new features

## Resources and Documentation
- [odc-stac Documentation](https://github.com/opendatacube/odc-stac)
- [TerraTorch Documentation](https://github.com/ibm/terratorch)
- [STAC Specification](https://stacspec.org/)
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- [Earth Search STAC API](https://github.com/element84/earth-search)

This project serves as a practical example for the geospatial community on leveraging modern cloud-native geospatial data formats with state-of-the-art machine learning frameworks for the FOSS4G 2025 conference.
