"""Quick test for the Werner-PSO magnetic inversion workflow.

This test creates a small synthetic magnetic raster, saves it as a GeoTIFF,
runs the workflow from src/appendix_a_werner_pso.py, and writes a CSV summary
to outputs/quick_test_results.csv.
"""

from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from appendix_a_werner_pso import run_workflow


def create_synthetic_raster(path: Path, nrows: int = 120, ncols: int = 120) -> None:
    y, x = np.mgrid[0:nrows, 0:ncols]

    anomaly_1 = 900 * np.exp(
        -(((x - 45) ** 2) / (2 * 9 ** 2) + ((y - 55) ** 2) / (2 * 12 ** 2))
    )
    anomaly_2 = 450 * np.exp(
        -(((x - 82) ** 2) / (2 * 7 ** 2) + ((y - 75) ** 2) / (2 * 8 ** 2))
    )
    background = 0.5 * x + 0.2 * y

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 10, size=(nrows, ncols))

    data = (anomaly_1 + anomaly_2 + background + noise).astype("float32")
    transform = from_origin(0, nrows, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def main() -> None:
    examples_dir = REPO_ROOT / "examples"
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    raster_path = examples_dir / "synthetic_magnetic_anomaly.tif"
    results_path = outputs_dir / "quick_test_results.csv"

    create_synthetic_raster(raster_path)
    results = run_workflow(str(raster_path))
    results.to_csv(results_path, index=False)

    print("Quick test completed successfully.")
    print(f"Synthetic raster: {raster_path}")
    print(f"Results CSV: {results_path}")
    print(results)


if __name__ == "__main__":
    main()
