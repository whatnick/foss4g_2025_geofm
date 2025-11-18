"""
Example: Multi-temporal analysis with odc-stac

This script demonstrates loading and analyzing multi-temporal satellite data
for change detection and time series analysis.
"""

import pystac_client
import odc.stac
import matplotlib.pyplot as plt
import pandas as pd


def multitemporal_analysis():
    """Example of multi-temporal analysis workflow."""

    print("📊 Multi-temporal Analysis Example")
    print("=" * 40)

    # Connect to catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    # Search for data across multiple months
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[-121.5, 38.5, -121.3, 38.7],  # Agricultural area in California
        datetime="2023-03-01/2023-09-30",  # Growing season
        query={"eo:cloud_cover": {"lt": 15}},  # Low cloud cover
    )

    items = list(search.items())
    print(f"Found {len(items)} items across the growing season")

    if len(items) < 3:
        print("Not enough clear scenes found. Try a different area or time range.")
        return

    # Load multi-temporal data
    dataset = odc.stac.load(
        items[:10],  # Limit to first 10 scenes
        bands=["red", "nir", "green"],
        resolution=100,
        groupby="solar_day",
    )

    print(f"Loaded dataset: {dataset.dims}")

    # Calculate NDVI time series
    ndvi = (dataset.nir - dataset.red) / (dataset.nir + dataset.red)

    # Get spatial mean NDVI for each time step
    ndvi_mean = ndvi.mean(dim=["x", "y"])

    # Create time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot NDVI time series
    dates = pd.to_datetime(ndvi_mean.time.values)
    ax1.plot(dates, ndvi_mean.values, "go-", linewidth=2, markersize=6)
    ax1.set_title("NDVI Time Series (Spatial Average)")
    ax1.set_ylabel("NDVI")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Show NDVI change between first and last image
    ndvi_change = ndvi.isel(time=-1) - ndvi.isel(time=0)

    im = ax2.imshow(
        ndvi_change.values,
        cmap="RdYlGn",
        vmin=-0.5,
        vmax=0.5,
        extent=[dataset.x.min(), dataset.x.max(), dataset.y.min(), dataset.y.max()],
    )
    ax2.set_title("NDVI Change (Last - First Image)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # Add colorbar
    plt.colorbar(im, ax=ax2, label="NDVI Change")

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\\nTemporal Statistics:")
    print(f"Time range: {dates.min()} to {dates.max()}")
    print(f"Number of observations: {len(dates)}")
    print(f"Mean NDVI across all times: {ndvi_mean.mean().values:.3f}")
    print(f"NDVI standard deviation: {ndvi_mean.std().values:.3f}")

    # Identify peak growing season
    peak_idx = ndvi_mean.argmax().values
    peak_date = dates[peak_idx]
    peak_ndvi = ndvi_mean.max().values
    print(f"Peak growing season: {peak_date} (NDVI: {peak_ndvi:.3f})")

    print("✅ Multi-temporal analysis completed!")

    return dataset, ndvi


if __name__ == "__main__":
    multitemporal_analysis()
