import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import differential_evolution

# ===========================================
# Load Raster
# ===========================================
def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    return data, transform

# ===========================================
# Peak Detection
# ===========================================
def detect_peaks(grid, percentile=95, window=15):
    smooth = gaussian_filter(grid, sigma=1.5)
    threshold = np.nanpercentile(smooth, percentile)
    local_max = smooth == maximum_filter(smooth, size=window)
    mask = (smooth >= threshold) & local_max
    rows, cols = np.where(mask)
    return rows, cols

# ===========================================
# Profile Extraction
# ===========================================
def extract_profile(grid, row, col, half_width=200):
    r0 = max(0, row - half_width)
    r1 = min(grid.shape[0], row + half_width)
    profile = grid[r0:r1, col]
    x = np.arange(len(profile))
    return x, profile

# ===========================================
# Simplified Werner Initialization
# ===========================================
def werner_init(x, y):
    idx = np.nanargmax(np.abs(y))
    x0 = x[idx]
    peak = np.abs(y[idx])
    half = peak * 0.5
    left = idx
    while left > 0 and np.abs(y[left]) > half:
        left -= 1
    right = idx
    while right < len(y)-1 and np.abs(y[right]) > half:
        right += 1
    width = abs(x[right] - x[left])
    z0 = max(width / 4.0, 1.0)
    return x0, z0

# ===========================================
# Forward Model
# ===========================================
def forward_model(x, params):
    x0, z0, A = params
    return A * z0 / ((x - x0)**2 + z0**2)

# ===========================================
# Objective Function
# ===========================================
def objective(params, x, y):
    pred = forward_model(x, params)
    return np.sqrt(np.nanmean((y - pred)**2))

# ===========================================
# Global Optimization (PSO-like)
# ===========================================
def invert_profile(x, y, x0_init, z0_init):
    bounds = [
        (x0_init - 50, x0_init + 50),
        (1.0, z0_init * 5),
        (-10000, 10000)
    ]
    result = differential_evolution(objective, bounds=bounds, args=(x, y), maxiter=100, seed=42)
    x0, z0, A = result.x
    rmse = result.fun
    return x0, z0, A, rmse

# ===========================================
# Main Workflow
# ===========================================
def run_workflow(raster_path):
    grid, transform = load_raster(raster_path)
    rows, cols = detect_peaks(grid)

    results = []

    for i, (r, c) in enumerate(zip(rows[:5], cols[:5])):
        x, profile = extract_profile(grid, r, c)
        if np.all(np.isnan(profile)):
            continue
        x0_init, z0_init = werner_init(x, profile)
        x0, z0, A, rmse = invert_profile(x, profile, x0_init, z0_init)
        results.append({
            "target": i+1,
            "x0": x0,
            "z0": z0,
            "amplitude": A,
            "rmse": rmse
        })

    return pd.DataFrame(results)

# ===========================================
# Run Example
# ===========================================
if __name__ == "__main__":
    raster_file = "your_raster.tif"
    results = run_workflow(raster_file)
    print(results)

# The full implementation used in this study is available from the authors upon request.
