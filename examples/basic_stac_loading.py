"""
Example: Basic STAC data loading with odc-stac

This script demonstrates the simplest possible workflow for loading
satellite imagery from a STAC catalog using odc-stac.
"""

import pystac_client
import odc.stac
import matplotlib.pyplot as plt
import numpy as np


def basic_stac_example():
    """Basic example of loading Sentinel-2 data via STAC."""

    print("🛰️  Basic STAC Data Loading Example")
    print("=" * 40)

    # Connect to catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    # Search for data
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[-122.4, 37.7, -122.3, 37.8],  # Small area in San Francisco
        datetime="2023-06-01/2023-06-30",
        limit=1,  # Just one scene for simplicity
    )

    # Get items
    items = list(search.items())
    print(f"Found {len(items)} items")

    if not items:
        print("No items found! Try adjusting search parameters.")
        return

    # Load data
    dataset = odc.stac.load(
        items,
        bands=["red", "green", "blue"],
        resolution=100,  # 100m resolution for quick loading
    )

    print(f"Loaded dataset with shape: {dataset.dims}")

    # Create simple RGB visualization
    rgb = np.stack(
        [
            dataset.red.isel(time=0).values,
            dataset.green.isel(time=0).values,
            dataset.blue.isel(time=0).values,
        ],
        axis=-1,
    )

    # Simple scaling for visualization
    rgb_scaled = np.clip(rgb / 3000, 0, 1)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_scaled)
    plt.title("Sentinel-2 RGB Composite")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("✅ Basic example completed!")


if __name__ == "__main__":
    basic_stac_example()
