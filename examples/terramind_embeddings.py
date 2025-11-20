#!/usr/bin/env python3
"""
TerraMind Embedding Generation Example

This script demonstrates how to generate embeddings from satellite imagery
using TerraMind v1 (IBM's geospatial foundation model) integrated with odc-stac.

✨ Now includes working TerraMind v1 Base model integration!

Example usage:
    python examples/terramind_embeddings.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import main functions
sys.path.append(str(Path(__file__).parent.parent))

try:
    from terratorch.registry import BACKBONE_REGISTRY

    from main import create_rgb_composite, load_stac_data
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure TerraTorch is installed:")
    print("  pip install 'git+https://github.com/terrastackai/terratorch.git'")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_terramind_model():
    """
    Load TerraMind v1 model with the latest TerraTorch integration.

    Returns:
        Loaded TerraMind model ready for inference
    """
    try:
        logger.info("Loading TerraMind v1 Base model...")

        # Load TerraMind with working configuration
        model = BACKBONE_REGISTRY.build(
            "terratorch_terramind_v1_base",
            modalities=["S2RGB"],  # 16x16 RGB patches
            pretrained=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        logger.info(f"✅ TerraMind model loaded successfully on {device}")
        logger.info("🎯 Model: TerraMind v1 Base (768D embeddings)")
        logger.info("🔧 Input: 16x16 RGB patches from Sentinel-2")

        return model

    except Exception as e:
        logger.error(f"Failed to load TerraMind model: {e}")
        logger.error("Make sure you have the latest TerraTorch installed:")
        logger.error(
            "  pip install 'git+https://github.com/terrastackai/terratorch.git'"
        )
        return None


def prepare_terramind_patches(rgb_array, patch_size=16):
    """
    Extract 16x16 patches optimized for TerraMind.

    Args:
        rgb_array: RGB image array [H, W, 3]
        patch_size: Patch size (16 for TerraMind)

    Returns:
        patches: Array of patches [N, 16, 16, 3]
        coordinates: Patch coordinates for spatial reference
    """
    height, width, channels = rgb_array.shape
    patches = []
    coordinates = []

    # Extract non-overlapping 16x16 patches
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = rgb_array[y : y + patch_size, x : x + patch_size, :]

            # Skip patches with too many NaN values
            if np.isnan(patch).sum() / patch.size < 0.1:
                patches.append(patch)
                coordinates.append((y, x))

    return np.array(patches), np.array(coordinates)


def generate_terramind_embeddings(rgb_array, model, patch_size=16, batch_size=32):
    """
    Generate TerraMind embeddings from RGB satellite imagery.

    Args:
        rgb_array: RGB image array [H, W, 3] in [0, 1] range
        model: Loaded TerraMind model
        patch_size: Patch size (16 for TerraMind)
        batch_size: Batch size for processing

    Returns:
        embeddings: Generated embeddings [N, 768]
        metadata: Processing metadata
    """
    logger.info(f"Preparing {patch_size}x{patch_size} patches for TerraMind...")

    # Extract patches
    patches, coordinates = prepare_terramind_patches(rgb_array, patch_size)

    if len(patches) == 0:
        logger.error("No valid patches extracted")
        return None, None

    logger.info(f"Extracted {len(patches)} patches")

    # Prepare for TerraMind (expects [0-255] range like Sentinel-2 RGB)
    patches_tensor = torch.from_numpy(patches).float()
    patches_tensor = patches_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Convert to [0-255] range and apply ImageNet normalization
    patches_tensor = patches_tensor * 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    patches_tensor = (patches_tensor / 255.0 - mean) / std

    # Generate embeddings
    device = next(model.parameters()).device
    embeddings_list = []

    logger.info("Generating TerraMind embeddings...")

    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i : i + batch_size].to(device)

            # TerraMind expects dictionary input
            outputs = model({"S2RGB": batch})
            batch_embeddings = outputs[-1].squeeze(1)  # Last layer, remove seq dim

            embeddings_list.append(batch_embeddings.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(patches_tensor)} patches")

    embeddings = np.vstack(embeddings_list)

    metadata = {
        "model": "terratorch_terramind_v1_base",
        "embedding_dim": embeddings.shape[1],
        "num_patches": len(patches),
        "patch_size": patch_size,
        "patch_coordinates": coordinates.tolist(),
        "original_shape": rgb_array.shape,
    }

    logger.info(
        f"✅ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
    )

    return embeddings, metadata


def analyze_embeddings(embeddings: np.ndarray, metadata: dict) -> dict:
    """
    Analyze TerraMind embeddings and return comprehensive statistics.

    Args:
        embeddings: Generated TerraMind embeddings array [N, 768]
        metadata: Metadata dictionary from embedding generation

    Returns:
        Dictionary with analysis results including PCA and t-SNE
    """
    logger.info("Analyzing TerraMind embeddings...")

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
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    analysis["dimension_stats"] = {
        "most_variable_dims": np.argsort(dim_stds)[-10:].tolist(),
        "least_variable_dims": np.argsort(dim_stds)[:10].tolist(),
        "highest_activation_dims": np.argsort(np.abs(dim_means))[-10:].tolist(),
        "mean_norm": float(np.mean(embedding_norms)),
        "std_norm": float(np.std(embedding_norms)),
    }

    # Calculate patch similarity
    if len(embeddings) > 1:
        n_sample = min(100, len(embeddings))  # Sample for efficiency
        sample_indices = np.random.choice(len(embeddings), n_sample, replace=False)
        similarity_matrix = cosine_similarity(embeddings[sample_indices])

        # Remove diagonal (self-similarity)
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, 0)

        analysis["similarity_stats"] = {
            "avg_cosine_similarity": float(np.mean(similarity_matrix[mask])),
            "std_cosine_similarity": float(np.std(similarity_matrix[mask])),
            "min_similarity": float(np.min(similarity_matrix[mask])),
            "max_similarity": float(np.max(similarity_matrix[mask])),
        }

    # Dimensionality reduction for visualization
    if len(embeddings) > 10:
        logger.info("Computing PCA and t-SNE for visualization...")

        # PCA
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        embeddings_pca = pca.fit_transform(embeddings)

        analysis["pca_stats"] = {
            "explained_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)[
                :10
            ].tolist(),
        }

        # t-SNE (on PCA-reduced data for efficiency)
        if len(embeddings) > 30:
            n_components = min(10, embeddings_pca.shape[1])
            embeddings_reduced = embeddings_pca[:, :n_components]

            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(embeddings) // 4),
            )
            embeddings_tsne = tsne.fit_transform(embeddings_reduced)

            analysis["tsne_embeddings"] = embeddings_tsne

    return analysis


def run_terramind_demo(
    bbox: list = None,  # Auckland
    datetime_range: str = "2023-12-01/2023-12-31",
    patch_size: int = 16,
    batch_size: int = 16,
):
    """
    Run complete TerraMind v1 embedding generation demo.

    Args:
        bbox: Bounding box coordinates [minx, miny, maxx, maxy]
        datetime_range: ISO date range string
        patch_size: Size of image patches (16 for TerraMind)
        batch_size: Batch size for inference

    Returns:
        Dictionary with results and embeddings
    """
    if bbox is None:
        bbox = [174.6, -36.95, 174.85, -36.75]  # Auckland
    logger.info("🚀 Starting TerraMind v1 Embedding Generation Demo")
    logger.info(f"📍 Area: {bbox}")
    logger.info(f"📅 Date range: {datetime_range}")
    logger.info(f"🔲 Patch size: {patch_size}x{patch_size}")

    try:
        # Step 1: Load TerraMind model
        logger.info("🤖 Loading TerraMind v1 model...")
        model = load_terramind_model()

        if model is None:
            logger.error("❌ Could not load TerraMind model. Exiting.")
            return None

        # Step 2: Load satellite data
        logger.info("🛰️ Loading satellite data via STAC...")
        dataset = load_stac_data(
            bbox=bbox,
            datetime=datetime_range,
            bands=["red", "green", "blue", "nir"],
            resolution=60,  # Better for 16x16 patches
            limit=2,  # Just 2 scenes for this demo
        )

        logger.info(f"📊 Loaded dataset with shape: {dict(dataset.dims)}")

        # Step 3: Create RGB composite
        logger.info("🎨 Creating RGB composite...")
        rgb_composite = create_rgb_composite(dataset, time_index=-1)
        rgb_array = rgb_composite.values

        # Validate data
        if np.isnan(rgb_array).all():
            logger.error("❌ No valid RGB data found. Check data loading.")
            return None

        logger.info(f"📸 RGB composite shape: {rgb_array.shape}")
        logger.info(
            f"📈 RGB value range: [{np.nanmin(rgb_array):.3f}, {np.nanmax(rgb_array):.3f}]"
        )

        # Step 4: Generate TerraMind embeddings
        logger.info("🧠 Generating TerraMind v1 embeddings...")
        embeddings, metadata = generate_terramind_embeddings(
            rgb_array, model, patch_size=patch_size, batch_size=batch_size
        )

        if embeddings is None:
            logger.error("❌ Failed to generate embeddings")
            return None

        # Step 5: Analyze embeddings
        logger.info("📊 Analyzing TerraMind embeddings...")
        analysis = analyze_embeddings(embeddings, metadata)

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 TERRAMIND v1 EMBEDDING RESULTS")
        logger.info("=" * 60)

        logger.info(f"📊 Generated {len(embeddings)} embeddings")
        logger.info(f"🧠 Embedding dimensionality: {embeddings.shape[1]}")
        logger.info(f"🔲 From {patch_size}x{patch_size} RGB patches")
        logger.info("📈 Statistics:")
        logger.info(f"  Mean: {analysis['mean']:.4f}")
        logger.info(f"  Std:  {analysis['std']:.4f}")
        logger.info(f"  Min:  {analysis['min']:.4f}")
        logger.info(f"  Max:  {analysis['max']:.4f}")
        logger.info(f"  Avg norm: {analysis['dimension_stats']['mean_norm']:.4f}")

        if "similarity_stats" in analysis:
            logger.info(
                f"🔗 Cosine similarity: {analysis['similarity_stats']['avg_cosine_similarity']:.4f} ± {analysis['similarity_stats']['std_cosine_similarity']:.4f}"
            )

        if "pca_stats" in analysis:
            logger.info(
                f"📉 PCA: {analysis['pca_stats']['explained_variance_ratio'][0]:.3f} variance in 1st component"
            )

        # Save results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        embedding_file = output_dir / "terramind_v1_embeddings.npy"
        np.save(embedding_file, embeddings)
        logger.info(f"💾 Saved embeddings to: {embedding_file}")

        if "tsne_embeddings" in analysis:
            tsne_file = output_dir / "terramind_v1_tsne.npy"
            np.save(tsne_file, analysis["tsne_embeddings"])
            logger.info(f"💾 Saved t-SNE coordinates to: {tsne_file}")

        # Save comprehensive metadata
        metadata_file = output_dir / "terramind_v1_metadata.txt"
        with open(metadata_file, "w") as f:
            f.write("TerraMind v1 Embedding Generation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {metadata['model']}\n")
            f.write(f"Generated: {len(embeddings)} embeddings\n")
            f.write(f"Dimensionality: {embeddings.shape[1]}\n")
            f.write(f"Patch size: {patch_size}x{patch_size}\n")
            f.write(f"Original image shape: {rgb_array.shape}\n")
            f.write(f"Area: {bbox}\n")
            f.write(f"Date range: {datetime_range}\n\n")
            f.write("Statistics:\n")
            f.write(f"  Mean: {analysis['mean']:.4f}\n")
            f.write(f"  Std:  {analysis['std']:.4f}\n")
            f.write(f"  Min:  {analysis['min']:.4f}\n")
            f.write(f"  Max:  {analysis['max']:.4f}\n")
            if "similarity_stats" in analysis:
                f.write(
                    f"  Avg cosine similarity: {analysis['similarity_stats']['avg_cosine_similarity']:.4f}\n"
                )

        logger.info(f"📄 Saved metadata to: {metadata_file}")

        return {
            "embeddings": embeddings,
            "analysis": analysis,
            "dataset": dataset,
            "rgb_composite": rgb_composite,
        }

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


def main():
    """Main function for the TerraMind v1 embedding demo."""
    logger.info("🌍 TerraMind v1 Embedding Generation Demo")
    logger.info("🚀 FOSS4G 2025 - Geospatial Foundation Models")
    logger.info("=" * 60)

    # Example 1: Auckland, New Zealand (Southern Hemisphere Summer)
    logger.info("📍 Example 1: Auckland, New Zealand")
    result_auckland = run_terramind_demo(
        bbox=[174.6, -36.95, 174.85, -36.75],
        datetime_range="2023-12-01/2023-12-31",  # Southern hemisphere summer
        patch_size=16,  # TerraMind's designed input
        batch_size=16,
    )

    # Example 2: San Francisco, USA (if Auckland worked)
    if result_auckland is not None:
        logger.info("\n📍 Example 2: San Francisco, USA")
        _result_sf = run_terramind_demo(
            bbox=[-122.5, 37.7, -122.3, 37.8],
            datetime_range="2023-06-01/2023-06-30",  # Northern hemisphere summer
            patch_size=16,
            batch_size=16,
        )

    logger.info("\\n🎯 TerraMind v1 embedding demo completed!")
    logger.info("📁 Check the 'outputs' directory for:")
    logger.info("   - terramind_v1_embeddings.npy: 768D embeddings")
    logger.info("   - terramind_v1_tsne.npy: t-SNE coordinates for visualization")
    logger.info("   - terramind_v1_metadata.txt: Complete analysis results")
    logger.info("\\n✨ Ready for FOSS4G 2025 presentation!")


if __name__ == "__main__":
    main()
