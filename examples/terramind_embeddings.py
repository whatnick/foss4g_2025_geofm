#!/usr/bin/env python3
"""
TerraMind Embedding Generation Example

This script demonstrates how to generate embeddings from satellite imagery
using TerraMind (IBM's geospatial foundation model) integrated with odc-stac.

Example usage:
    python examples/terramind_embeddings.py
"""

import logging
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import main functions
sys.path.append(str(Path(__file__).parent.parent))

try:
    from main import (
        load_stac_data,
        create_rgb_composite,
        load_terramind_model,
        generate_terramind_embeddings,
    )
except ImportError as e:
    print(f"Error importing from main: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_embeddings(embeddings: np.ndarray, metadata: dict) -> dict:
    """
    Analyze generated embeddings and return summary statistics.

    Args:
        embeddings: Generated embeddings array
        metadata: Metadata dictionary from embedding generation

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "shape": embeddings.shape,
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
        "metadata": metadata,
    }

    # Calculate per-dimension statistics
    dim_means = np.mean(embeddings, axis=0)
    dim_stds = np.std(embeddings, axis=0)

    analysis["dimension_stats"] = {
        "most_variable_dims": np.argsort(dim_stds)[-5:].tolist(),
        "least_variable_dims": np.argsort(dim_stds)[:5].tolist(),
        "mean_activation_dims": np.argsort(np.abs(dim_means))[-5:].tolist(),
    }

    # Calculate patch similarity if we have multiple patches
    if len(embeddings) > 1:
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(embeddings[:10])  # First 10 patches
            analysis["avg_cosine_similarity"] = float(np.mean(similarity_matrix))
        except ImportError:
            logger.warning("scikit-learn not available for similarity analysis")

    return analysis


def run_terramind_demo(
    bbox: list = [174.6, -36.95, 174.85, -36.75],  # Auckland
    datetime_range: str = "2023-12-01/2023-12-31",
    patch_size: int = 16,
    batch_size: int = 8,
):
    """
    Run the complete TerraMind embedding demo.

    Args:
        bbox: Bounding box for data loading
        datetime_range: Date range for STAC search
        patch_size: Size of patches for TerraMind processing
        batch_size: Batch size for inference
    """
    logger.info("Starting TerraMind Embedding Generation Demo")
    logger.info(f"Area: {bbox}")
    logger.info(f"Date range: {datetime_range}")

    try:
        # Step 1: Load satellite data
        logger.info("Loading satellite data via STAC...")
        dataset = load_stac_data(
            bbox=bbox,
            datetime=datetime_range,
            bands=["red", "green", "blue", "nir"],
            resolution=100,
            limit=2,  # Just 2 scenes for this demo
        )

        logger.info(f"Loaded dataset with shape: {dict(dataset.dims)}")

        # Step 2: Load TerraMind model
        logger.info("Loading TerraMind model...")
        model = load_terramind_model()

        if model is None:
            logger.error("Could not load TerraMind model. Exiting.")
            return None

        # Step 3: Create RGB composite
        logger.info("Creating RGB composite...")
        rgb_composite = create_rgb_composite(dataset, time_index=-1)
        rgb_array = rgb_composite.values

        # Validate data
        if np.isnan(rgb_array).all():
            logger.error("No valid RGB data found. Check data loading.")
            return None

        logger.info(f"RGB composite shape: {rgb_array.shape}")
        logger.info(
            f"RGB value range: [{np.nanmin(rgb_array):.3f}, {np.nanmax(rgb_array):.3f}]"
        )

        # Step 4: Generate embeddings
        logger.info("Generating TerraMind embeddings...")
        embeddings, metadata = generate_terramind_embeddings(
            rgb_array, model, patch_size=patch_size, batch_size=batch_size
        )

        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return None

        # Step 5: Analyze embeddings
        logger.info("Analyzing embeddings...")
        analysis = analyze_embeddings(embeddings, metadata)

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("TERRAMIND EMBEDDING RESULTS")
        logger.info("=" * 60)

        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimensionality: {embeddings.shape[1]}")
        logger.info("Statistics:")
        logger.info(f"  Mean: {analysis['mean']:.4f}")
        logger.info(f"  Std:  {analysis['std']:.4f}")
        logger.info(f"  Min:  {analysis['min']:.4f}")
        logger.info(f"  Max:  {analysis['max']:.4f}")

        if "avg_cosine_similarity" in analysis:
            logger.info(
                f"  Avg cosine similarity: {analysis['avg_cosine_similarity']:.4f}"
            )

        logger.info(
            f"Most variable dimensions: {analysis['dimension_stats']['most_variable_dims']}"
        )
        logger.info(
            f"Highest activation dimensions: {analysis['dimension_stats']['mean_activation_dims']}"
        )

        # Save results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        embedding_file = output_dir / "terramind_embeddings.npy"
        np.save(embedding_file, embeddings)
        logger.info(f"Saved embeddings to: {embedding_file}")

        metadata_file = output_dir / "embedding_metadata.txt"
        with open(metadata_file, "w") as f:
            f.write("TerraMind Embedding Generation Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {len(embeddings)} embeddings\n")
            f.write(f"Dimensionality: {embeddings.shape[1]}\n")
            f.write(f"Patch size: {patch_size}x{patch_size}\n")
            f.write(f"Original image shape: {rgb_array.shape}\n")
            f.write("Statistics:\n")
            f.write(f"  Mean: {analysis['mean']:.4f}\n")
            f.write(f"  Std:  {analysis['std']:.4f}\n")
            f.write(f"  Min:  {analysis['min']:.4f}\n")
            f.write(f"  Max:  {analysis['max']:.4f}\n")

        logger.info(f"Saved metadata to: {metadata_file}")

        return {
            "embeddings": embeddings,
            "analysis": analysis,
            "dataset": dataset,
            "rgb_composite": rgb_composite,
        }

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def main():
    """Main function for the TerraMind embedding demo."""
    # Example 1: Auckland, New Zealand
    logger.info("Example 1: Auckland, New Zealand")
    result_auckland = run_terramind_demo(
        bbox=[174.6, -36.95, 174.85, -36.75],
        datetime_range="2023-12-01/2023-12-31",  # Southern hemisphere summer
    )

    # Example 2: San Francisco, USA (if Auckland worked)
    if result_auckland is not None:
        logger.info("\nExample 2: San Francisco, USA")
        _result_sf = run_terramind_demo(
            bbox=[-122.5, 37.7, -122.3, 37.8],
            datetime_range="2023-06-01/2023-06-30",  # Northern hemisphere summer
        )

    logger.info("\nTerraMind embedding demo completed!")
    logger.info("Check the 'outputs' directory for saved embeddings and metadata.")


if __name__ == "__main__":
    main()
