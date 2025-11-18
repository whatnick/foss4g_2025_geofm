#!/usr/bin/env python3
"""
FOSS4G 2025 Demo: Loading geospatial data via odc-stac into TerraTorch

This script demonstrates how to:
1. Connect to a STAC catalog
2. Search for satellite imagery
3. Load data using odc-stac into xarray Datasets
4. Prepare data for TerraTorch workflows
5. Visualize the loaded data

Example usage:
    python main.py
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import pystac_client
    import odc.stac
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from typing import Tuple
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install dependencies with: pip install -e .")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_stac_data(
    catalog_url: str = "https://earth-search.aws.element84.com/v1",
    collection: str = "sentinel-2-l2a",
    bbox: list = [174.5, -37.0, 175.0, -36.7],  # Auckland, New Zealand
    datetime: str = "2023-06-01/2023-06-30",
    bands: list = ["red", "green", "blue", "nir"],
    resolution: int = 100,
    limit: int = 5,
) -> xr.Dataset:
    """
    Load satellite imagery from a STAC catalog using odc-stac.

    Args:
        catalog_url: STAC catalog endpoint URL
        collection: STAC collection name
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        datetime: Date range in ISO format
        bands: List of bands to load
        resolution: Output resolution in meters
        limit: Maximum number of items to load

    Returns:
        xarray Dataset with loaded satellite imagery
    """
    logger.info(f"Connecting to STAC catalog: {catalog_url}")

    try:
        # Connect to STAC catalog
        catalog = pystac_client.Client.open(catalog_url)
        logger.info("Successfully connected to catalog")

        # Search for data
        logger.info(f"Searching for {collection} data...")
        search = catalog.search(
            collections=[collection], bbox=bbox, datetime=datetime, limit=limit
        )

        # Get items from search
        items = list(search.items())
        logger.info(f"Found {len(items)} items")

        if not items:
            raise ValueError("No items found for the given search criteria")

        # Load data using odc-stac
        logger.info(f"Loading data with bands: {bands}")
        dataset = odc.stac.load(
            items,
            bands=bands,
            resolution=resolution,
            chunks={"time": 1, "x": 512, "y": 512},
            # Chunking for memory efficiency, or matching TerraTorch expectations
        )

        logger.info(f"Successfully loaded dataset with shape: {dataset.dims}")
        return dataset

    except Exception as e:
        logger.error(f"Error loading STAC data: {e}")
        raise


def visualize_dataset(dataset: xr.Dataset, output_dir: Path = Path("outputs")) -> None:
    """
    Create basic visualizations of the loaded dataset.

    Args:
        dataset: xarray Dataset to visualize
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)

    logger.info("Creating visualizations...")

    # Plot the first time step of RGB composite
    if all(band in dataset for band in ["red", "green", "blue"]):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create RGB composite (normalize for display)
        rgb_data = np.stack(
            [
                dataset.red.isel(time=0).values,
                dataset.green.isel(time=0).values,
                dataset.blue.isel(time=0).values,
            ],
            axis=-1,
        )

        # Simple linear stretch for visualization
        rgb_data = np.clip(rgb_data / 3000, 0, 1)  # Adjust scaling as needed

        ax.imshow(
            rgb_data,
            extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()],
        )
        ax.set_title("RGB Composite (First Time Step)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(output_dir / "rgb_composite.png", dpi=300, bbox_inches="tight")
        logger.info(f"RGB composite saved to {output_dir / 'rgb_composite.png'}")

        plt.close()

    # Plot time series for a sample pixel
    if "time" in dataset.dims and len(dataset.time) > 1:
        center_x = len(dataset.x) // 2
        center_y = len(dataset.y) // 2

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, band in enumerate(["red", "green", "blue", "nir"][:4]):
            if band in dataset:
                time_series = dataset[band].isel(x=center_x, y=center_y)
                axes[i].plot(time_series.time, time_series.values, "o-")
                axes[i].set_title(f"{band.upper()} Band Time Series")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Reflectance")
                axes[i].grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / "time_series.png", dpi=300, bbox_inches="tight")
        logger.info(f"Time series plot saved to {output_dir / 'time_series.png'}")

        plt.close()


def prepare_for_terratorch(dataset: xr.Dataset) -> Dict[str, Any]:
    """
    Prepare the loaded dataset for use with TerraTorch.

    Args:
        dataset: xarray Dataset from odc-stac

    Returns:
        Dictionary with prepared data and metadata
    """
    logger.info("Preparing data for TerraTorch...")

    # Basic data preparation
    prepared_data = {
        "dataset": dataset,
        "bands": list(dataset.data_vars.keys()),
        "spatial_dims": {"x": len(dataset.x), "y": len(dataset.y)},
        "temporal_dim": len(dataset.time) if "time" in dataset.dims else 1,
        "crs": str(dataset.spatial_ref.attrs.get("crs_wkt", "Unknown")),
        "resolution": float(dataset.x[1] - dataset.x[0]),  # Assuming regular grid
        "bbox": [
            float(dataset.x.min()),
            float(dataset.y.min()),
            float(dataset.x.max()),
            float(dataset.y.max()),
        ],
    }

    logger.info("Prepared data summary:")
    logger.info(f"  - Bands: {prepared_data['bands']}")
    logger.info(f"  - Spatial dimensions: {prepared_data['spatial_dims']}")
    logger.info(f"  - Temporal dimension: {prepared_data['temporal_dim']}")
    logger.info(f"  - CRS: {prepared_data['crs']}")

    return prepared_data


# TerraMind embedding generation functions
def rgb_smooth_quantiles(rgb_array: np.ndarray) -> np.ndarray:
    """
    Apply smooth quantile normalization to RGB array.

    Args:
        rgb_array: RGB array of shape (..., 3)

    Returns:
        Normalized RGB array
    """
    # Calculate smooth quantiles (2% and 98%)
    lower_quantile = np.percentile(rgb_array, 2, axis=(0, 1), keepdims=True)
    upper_quantile = np.percentile(rgb_array, 98, axis=(0, 1), keepdims=True)

    # Clip and normalize to [0, 1]
    clipped = np.clip(rgb_array, lower_quantile, upper_quantile)
    normalized = (clipped - lower_quantile) / (upper_quantile - lower_quantile + 1e-8)

    return normalized


def prepare_terramind_patches(rgb_data: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Prepare RGB data for TerraMind inference by extracting patches.

    Args:
        rgb_data: RGB data array of shape (H, W, 3)
        patch_size: Size of square patches (default 16x16)

    Returns:
        Array of patches of shape (num_patches, patch_size, patch_size, 3)
    """
    height, width, channels = rgb_data.shape

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Crop to fit exact patches
    cropped_h = num_patches_h * patch_size
    cropped_w = num_patches_w * patch_size
    cropped_data = rgb_data[:cropped_h, :cropped_w, :]

    # Reshape into patches
    patches = (
        cropped_data.reshape(
            num_patches_h, patch_size, num_patches_w, patch_size, channels
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, patch_size, patch_size, channels)
    )

    return patches


def normalize_terramind_input(patches: np.ndarray) -> torch.Tensor:
    """
    Normalize patches for TerraMind input.
    TerraMind expects inputs normalized with ImageNet statistics.

    Args:
        patches: Array of patches of shape (num_patches, 16, 16, 3)

    Returns:
        Normalized tensor of shape (num_patches, 3, 16, 16)
    """
    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to tensor and normalize
    patches_tensor = torch.from_numpy(patches).float()

    # Transpose to (batch, channels, height, width)
    patches_tensor = patches_tensor.permute(0, 3, 1, 2)

    # Normalize
    for i in range(3):
        patches_tensor[:, i] = (patches_tensor[:, i] - mean[i]) / std[i]

    return patches_tensor


def load_terramind_model():
    """
    Load the TerraMind model for embedding generation.

    Returns:
        Loaded TerraMind model or None if loading fails
    """
    try:
        from terratorch.models.backbones import BACKBONE_REGISTRY

        logger.info("Loading TerraMind model...")
        model = BACKBONE_REGISTRY.build(
            "terramind_v1_base", modalities=["S2RGB"], pretrained=True
        )
        model.eval()

        # Check device availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if device.type == "cuda":
            model = model.to(device)
            logger.info("Model moved to GPU")

        return model

    except ImportError as e:
        logger.warning(f"Could not load TerraMind model: {e}")
        logger.warning("Please ensure terratorch is installed: pip install terratorch")
        return None
    except Exception as e:
        logger.error(f"Error loading TerraMind model: {e}")
        return None


def generate_terramind_embeddings(
    rgb_data: np.ndarray, model, patch_size: int = 16, batch_size: int = 32
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Generate TerraMind embeddings for RGB satellite imagery.

    Args:
        rgb_data: RGB data array of shape (H, W, 3), values in [0, 1]
        model: Loaded TerraMind model
        patch_size: Size of patches for processing
        batch_size: Batch size for inference

    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    if model is None:
        logger.warning("Model not available. Cannot generate embeddings.")
        return None, {"error": "Model not available"}

    device = next(model.parameters()).device

    # Step 1: Apply smooth quantile normalization
    logger.info("Applying smooth quantile normalization...")
    normalized_rgb = rgb_smooth_quantiles(rgb_data)

    # Step 2: Extract patches
    logger.info(f"Extracting {patch_size}x{patch_size} patches...")
    patches = prepare_terramind_patches(normalized_rgb, patch_size)
    logger.info(f"Extracted {len(patches)} patches")

    if len(patches) == 0:
        logger.warning("No patches extracted. Check input data size.")
        return None, {"error": "No patches extracted"}

    # Step 3: Normalize for model input
    logger.info("Normalizing patches for model input...")
    patches_tensor = normalize_terramind_input(patches)

    # Step 4: Generate embeddings in batches
    logger.info("Generating embeddings...")
    embeddings_list = []

    try:
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i : i + batch_size].to(device)

                # Generate embeddings
                batch_embeddings = model({"S2RGB": batch})

                # Move back to CPU and store
                embeddings_list.append(batch_embeddings.cpu().numpy())

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(
                        f"Processed {i + len(batch)}/{len(patches_tensor)} patches"
                    )

        # Combine all embeddings
        embeddings = np.vstack(embeddings_list)

        # Create metadata
        metadata = {
            "num_patches": len(patches),
            "patch_size": patch_size,
            "embedding_dim": embeddings.shape[1],
            "original_shape": rgb_data.shape,
            "patches_shape": patches.shape,
            "device_used": str(device),
        }

        logger.info(
            f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
        )
        return embeddings, metadata

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        return None, {"error": str(e)}


def create_rgb_composite(dataset: xr.Dataset, time_index: int = -1) -> xr.DataArray:
    """
    Create an RGB composite from the dataset.

    Args:
        dataset: xarray Dataset with red, green, blue bands
        time_index: Time index to use (-1 for most recent)

    Returns:
        RGB composite as xarray DataArray
    """
    # Select time slice
    if "time" in dataset.dims:
        ds = dataset.isel(time=time_index)
    else:
        ds = dataset

    # Create RGB stack
    rgb = xr.concat([ds.red, ds.green, ds.blue], dim="band")
    rgb = rgb.transpose("y", "x", "band")

    # Normalize to 0-1 range (Sentinel-2 values are typically 0-10000)
    rgb = rgb / 10000.0
    rgb = rgb.clip(0, 1)

    return rgb


def main():
    """Main demonstration function."""
    logger.info(
        "Starting FOSS4G 2025 GeoFM Demo: odc-stac to TerraTorch with TerraMind"
    )

    try:
        # Load satellite data from STAC
        dataset = load_stac_data(
            bbox=[174.6, -36.95, 174.85, -36.75],  # Central Auckland area
            datetime="2023-12-01/2023-12-31",  # Summer in Southern Hemisphere
            bands=["red", "green", "blue", "nir"],
            resolution=100,
            limit=3,  # Limit to 3 scenes for demo
        )

        # Print dataset information
        logger.info("\nDataset Information:")
        logger.info(f"Dimensions: {dict(dataset.dims)}")
        logger.info(f"Coordinates: {list(dataset.coords.keys())}")
        logger.info(f"Data variables: {list(dataset.data_vars.keys())}")

        # Create visualizations
        visualize_dataset(dataset)

        # Prepare for TerraTorch
        prepared_data = prepare_for_terratorch(dataset)

        # Generate TerraMind embeddings
        logger.info("\n" + "=" * 50)
        logger.info("TERRAMIND EMBEDDING GENERATION")
        logger.info("=" * 50)

        # Load TerraMind model
        model = load_terramind_model()

        if model is not None:
            # Create RGB composite from most recent image
            rgb_composite = create_rgb_composite(dataset, time_index=-1)
            rgb_array = rgb_composite.values

            # Check for valid data
            if not np.isnan(rgb_array).all():
                logger.info(f"RGB data shape: {rgb_array.shape}")
                logger.info(
                    f"RGB data range: [{np.nanmin(rgb_array):.3f}, {np.nanmax(rgb_array):.3f}]"
                )

                # Generate embeddings
                embeddings, metadata = generate_terramind_embeddings(
                    rgb_array,
                    model,
                    patch_size=16,
                    batch_size=16,  # Conservative batch size
                )

                if embeddings is not None:
                    logger.info("\nEmbedding Generation Results:")
                    logger.info(f"Generated {len(embeddings)} embeddings")
                    logger.info(f"Embedding shape: {embeddings.shape}")
                    logger.info("Embedding statistics:")
                    logger.info(f"  Mean: {np.mean(embeddings):.4f}")
                    logger.info(f"  Std: {np.std(embeddings):.4f}")
                    logger.info(f"  Min: {np.min(embeddings):.4f}")
                    logger.info(f"  Max: {np.max(embeddings):.4f}")

                    # Add embeddings to prepared data
                    prepared_data["embeddings"] = embeddings
                    prepared_data["embedding_metadata"] = metadata
                else:
                    logger.warning("Failed to generate embeddings")
            else:
                logger.warning("No valid RGB data found for embedding generation")
        else:
            logger.warning(
                "TerraMind model not available, skipping embedding generation"
            )

        logger.info("\nDemo completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Check the 'outputs' directory for visualizations")
        logger.info("2. Use the prepared_data for TerraTorch model training")
        logger.info("3. Explore the Jupyter notebooks for advanced examples")
        if "embeddings" in prepared_data:
            logger.info("4. Use the generated embeddings for similarity analysis")

        return prepared_data

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
