"""FDK projection data loading, normalization, and downsampling utilities.

Shared across FDK_reduce_memory, FDK_from_scratch, FDK_astra, FDK_custom_filter,
and Iteratives pipelines. Uses parallel TIFF loading for I/O-bound speed on
network-mounted or WSL filesystem paths.
"""

import gc
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import numpy as np
import tifffile as tiff

logger = logging.getLogger(__name__)

PROGRESS_LOG_INTERVAL_SEC = 5.0
# I/O-bound: more workers hide per-call latency, especially over the WSL filesystem bridge.
MAX_LOAD_WORKERS = 12


# ─── Private helpers ──────────────────────────────────────────────────────────

def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_images(tiff_folder: str) -> np.ndarray:
    """Load all TIFF projections from a folder into a (H, W, N) array.

    Files are sorted by the leading integer in their filename.
    Loading is parallelised with ThreadPoolExecutor to hide I/O latency.

    Args:
        tiff_folder: Absolute path to the folder containing TIFF files.

    Returns:
        projections: Array of shape (H, W, N), dtype matches the source files.

    Raises:
        FileNotFoundError: If tiff_folder does not exist.
        ValueError: If no TIFF files are found.
    """
    if not os.path.isdir(tiff_folder):
        raise FileNotFoundError(f"Projection folder not found: {tiff_folder!r}")

    logger.info("Loading images from: %s", tiff_folder)
    start_time = time.perf_counter()

    def _extract_number(filename: str) -> int:
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    file_list = [f for f in os.listdir(tiff_folder)
                 if f.lower().endswith(('.tif', '.tiff'))]
    file_list.sort(key=_extract_number)
    num_files = len(file_list)

    if num_files == 0:
        raise ValueError(f"No TIFF files found in: {tiff_folder!r}")

    logger.info("Found %d TIFF files | first: %s | last: %s",
                num_files, file_list[0], file_list[-1])

    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape
    logger.info("Image shape: (%d, %d) | dtype: %s", height, width, first_img.dtype)

    # Pre-allocate output array; dtype taken from the first image
    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)

    completed = 0
    last_log_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_LOAD_WORKERS) as pool:
        futures = {
            pool.submit(tiff.imread, os.path.join(tiff_folder, f)): i
            for i, f in enumerate(file_list)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            projections[:, :, i] = fut.result()
            completed += 1
            now = time.perf_counter()
            if completed < num_files and (now - last_log_time) >= PROGRESS_LOG_INTERVAL_SEC:
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                logger.info("Loaded %d/%d (%.1f%%) | %.2f img/s",
                            completed, num_files, 100.0 * completed / num_files, rate)
                last_log_time = now

    total_elapsed = time.perf_counter() - start_time
    logger.info("Loaded %d images in %s (%.2f img/s)",
                num_files, _format_elapsed(total_elapsed),
                num_files / total_elapsed if total_elapsed > 0 else 0.0)
    return projections


def generate_collapsed_sinogram(projections: np.ndarray) -> np.ndarray:
    """Sum projections along the height axis to produce a collapsed sinogram.

    Args:
        projections: Projection stack, shape (H, W, n_angles).

    Returns:
        sino_sum: Collapsed sinogram, shape (W, n_angles), dtype float64.
    """
    logger.info("Generating collapsed sinogram from shape %s", projections.shape)
    return np.sum(projections, axis=0)


# ─── I0 selection ─────────────────────────────────────────────────────────────

def selecionar_roi_I0(sino_raw: np.ndarray) -> tuple | None:
    """Interactive ROI selection for background (I0) normalization.

    Displays the collapsed sinogram and lets the user draw a rectangle over
    an open-beam (air) region. Returns the selected ROI coordinates.

    Args:
        sino_raw: Collapsed sinogram, shape (n_detector_px, n_angles).

    Returns:
        roi_coords: (det_start, det_end, ang_start, ang_end) pixel indices, or
            None if the user closes without confirming.
    """
    roi_background = [None]
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.20)

    ax.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("Draw ROI on BACKGROUND (air) for I0 normalization | Then click Confirm",
                 fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("θ (degree)")

    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, 361, 30)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(d)}°" for d in ticks_deg])

    rect_bg = None

    def on_select_callback(eclick, erelease) -> None:
        nonlocal rect_bg
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        ang_start, ang_end = sorted([x1, x2])
        det_start, det_end = sorted([y1, y2])
        roi_background[0] = (det_start, det_end, ang_start, ang_end)
        # Draw green rectangle for the background ROI
        if rect_bg:
            rect_bg.remove()
        patch = plt.Rectangle(
            (ang_start, det_start), ang_end - ang_start, det_end - det_start,
            fill=False, edgecolor='green', linewidth=2, linestyle='--'
        )
        ax.add_patch(patch)
        ax.set_title("I0 ROI selected (green) | Click Confirm", fontsize=12)
        fig.canvas.draw()

    RectangleSelector(
        ax, on_select_callback, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels', interactive=True,
        props=dict(facecolor='green', edgecolor='green', alpha=0.2, fill=True)
    )

    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_button, 'Confirm')
    btn_confirm.on_clicked(lambda _: plt.close(fig))
    plt.show(block=True)

    return roi_background[0]


def get_I0_from_roi(
    sino_raw: np.ndarray,
    roi_background: tuple,
    height: int,
) -> float:
    """Compute the mean I0 intensity from the selected background ROI.

    Divides by height because sino_raw is a sum-projection (collapsed along H),
    so each pixel represents the sum over `height` detector rows.

    Args:
        sino_raw: Collapsed sinogram, shape (n_detector_px, n_angles).
        roi_background: (det_start, det_end, ang_start, ang_end) from selecionar_roi_I0.
        height: Number of detector rows summed into the sinogram (projections.shape[0]).

    Returns:
        mean_I0: Average per-row open-beam intensity (float).
    """
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]
    mean_I0 = float(np.mean(roi_crop)) / height
    logger.info("I0 from ROI: %.2f (ROI mean=%.2f, height=%d)", mean_I0,
                mean_I0 * height, height)
    return mean_I0


# ─── Normalization ────────────────────────────────────────────────────────────

_I0_CLIP_RATIO_MAX = 1.2   # I/I0 values above this are physically impossible
_I0_CLIP_RATIO_MIN = 1e-6  # Minimum ratio before log to avoid log(0)


def normalize_projections(
    projections_raw: np.ndarray,
    I0_override: float | None = None,
) -> np.ndarray:
    """Apply Beer-Lambert normalization: attenuation = -log(I / I0).

    If I0_override is None, falls back to the 99th percentile of the raw
    projections and logs a warning.

    Args:
        projections_raw: Raw detector counts, shape (H, W, n_angles), float32.
        I0_override: Known open-beam intensity from a calibrated ROI. Strongly
            recommended over the fallback.

    Returns:
        projections_norm: Attenuation values, same shape, float32.

    Raises:
        ValueError: If the resolved I0 is <= 0.
    """
    print("--> Normalizing projections...")
    print(f"    Input shape: {projections_raw.shape}, "
          f"size: {projections_raw.nbytes / 1e6:.1f} MB")

    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 99))
        logger.warning(
            "No I0_override provided — using 99th-percentile fallback: %.2f. "
            "Provide I0_override from a calibrated open-beam ROI.", I0
        )
    else:
        I0 = float(I0_override)

    if I0 <= 0:
        raise ValueError(
            f"I0 must be positive, got {I0:.4f}. Check ROI selection or raw data."
        )

    projections_norm = projections_raw.astype(np.float32, copy=True)
    projections_norm /= (I0 + 1e-6)

    total_pixels = projections_norm.size
    n_above = np.count_nonzero(projections_norm > _I0_CLIP_RATIO_MAX)
    n_below = np.count_nonzero(projections_norm < 0)
    logger.debug(
        "Clipping report — above %.1f: %.4f%% | below 0: %.4f%%",
        _I0_CLIP_RATIO_MAX,
        100.0 * n_above / total_pixels,
        100.0 * n_below / total_pixels,
    )

    np.clip(projections_norm, _I0_CLIP_RATIO_MIN, _I0_CLIP_RATIO_MAX, out=projections_norm)
    np.log(projections_norm, out=projections_norm)
    projections_norm *= -1.0
    projections_norm[projections_norm < 0] = 0  # Remove sub-zero log artefacts

    if not np.isfinite(projections_norm).all():
        n_bad = int(np.sum(~np.isfinite(projections_norm)))
        raise RuntimeError(
            f"Normalization produced {n_bad} non-finite values. "
            "Check I0 value and raw projection data."
        )

    return projections_norm


# ─── Downsampling ─────────────────────────────────────────────────────────────

def downsample_block_mean_pad(proj: np.ndarray, f: int) -> np.ndarray:
    """Downsample projections by block-mean averaging with edge padding.

    Each (f × f) block of pixels is replaced by its mean. If H or W is not
    evenly divisible by f, edge-replication padding is applied first.

    Args:
        proj: Projection stack, shape (H, W, n_angles).
        f: Downsampling factor. f=1 returns the input unchanged.

    Returns:
        downsampled: Array of shape (H//f, W//f, n_angles), float32.
    """
    if f == 1:
        return proj

    Height, Width, Angles = proj.shape
    pad_h = (-Height) % f
    pad_w = (-Width) % f

    proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge') if (pad_h or pad_w) else proj

    Hc, Wc = proj_p.shape[0], proj_p.shape[1]
    blocks = proj_p.reshape(Hc // f, f, Wc // f, f, Angles)
    return blocks.mean(axis=(1, 3)).astype(np.float32)


# ─── Volume info ──────────────────────────────────────────────────────────────

def print_volume_info(volume: np.ndarray, geo) -> None:
    """Print a summary of the reconstructed volume and its geometry.

    Args:
        volume: Reconstructed volume, shape (nz, ny, nx).
        geo: Geometry object with dVoxel, nVoxel, sVoxel attributes.
    """
    print("RECONSTRUCTED VOLUME INFORMATION")
    print(f"  Shape:  {volume.shape}  |  dtype: {volume.dtype}")
    print(f"  Voxel size (mm): [{geo.dVoxel[0]:.3f}, {geo.dVoxel[1]:.3f}, {geo.dVoxel[2]:.3f}]")
    print(f"  Num voxels:      {geo.nVoxel}")
    print(f"  Physical size:   [{geo.sVoxel[0]:.3f}, {geo.sVoxel[1]:.3f}, {geo.sVoxel[2]:.3f}] mm")
