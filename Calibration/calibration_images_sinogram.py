"""
IO utilities for calibration process.
"""

import os
import re
import logging

import numpy as np
import tifffile as tiff


logger = logging.getLogger(__name__)


def _extract_number(filename):
    """Extract first numeric token for deterministic file sorting."""
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    return 0


def _validate_projection_stack(projections, source_folder):
    """Validate projection array shape and numerical integrity."""
    # Guard: calibration expects a 3D stack with shape (H, W, A).
    if projections.ndim != 3:
        raise ValueError(f"Expected projection stack with 3 dimensions, got shape {projections.shape}")

    # Guard: empty stacks should fail early with a clear message.
    if projections.size == 0 or projections.shape[2] == 0:
        raise ValueError(f"No projection frames were loaded from folder: {source_folder}")

    # Guard: reject NaN/Inf before downstream segmentation math.
    if not np.isfinite(projections).all():
        raise ValueError(f"Loaded projections contain NaN or Inf values: {source_folder}")


def load_images(tiff_folder: str) -> np.ndarray:
    """Load all TIFF projections from a folder into a (H, W, N) array.

    Args:
        tiff_folder: Absolute path to the folder containing TIFF files.

    Returns:
        projections: Array of shape (H, W, N), dtype matches the source files.

    Raises:
        FileNotFoundError: If tiff_folder does not exist.
        ValueError: If no TIFF files are found, images have inconsistent shapes,
            or projections contain non-finite values.
    """
    logger.info("Loading images from: %s", tiff_folder)

    if not os.path.isdir(tiff_folder):
        raise FileNotFoundError(f"Projection folder does not exist: {tiff_folder}")

    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith((".tif", ".tiff"))]
    file_list.sort(key=_extract_number)
    num_files = len(file_list)
    logger.info("Number of TIFF files found: %d", num_files)

    if not file_list:
        raise ValueError(f"No TIFF files were found in folder: {tiff_folder}")

    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    # Guard: calibration stack expects grayscale 2D projection frames.
    if first_img.ndim != 2:
        raise ValueError(
            "Expected 2D TIFF projections, "
            f"got shape {first_img.shape} for file '{file_list[0]}'"
        )

    height, width = first_img.shape
    logger.info("Projection dimensions: %d (H) x %d (W) | %d projections", height, width, num_files)

    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)
    for i, f in enumerate(file_list):
        current_img = tiff.imread(os.path.join(tiff_folder, f))

        # Guard: reject mixed image dimensions inside the same projection set.
        if current_img.shape != (height, width):
            raise ValueError(
                "Inconsistent TIFF shape detected "
                f"for file '{f}'. Expected {(height, width)}, got {current_img.shape}."
            )

        projections[:, :, i] = current_img

    _validate_projection_stack(projections, tiff_folder)

    logger.info("Images loaded successfully")
    return projections


def generate_collapsed_sinogram(projections: np.ndarray) -> np.ndarray:
    """Sum projections along the height axis to produce a collapsed sinogram.

    Args:
        projections: Projection stack, shape (H, W, n_angles).

    Returns:
        sino_sum: Collapsed sinogram, shape (W, n_angles), dtype float32.

    Raises:
        ValueError: If projections is not 3D or the result contains non-finite values.
    """
    logger.info("Creating collapsed sinogram (sum projection)")

    # Guard: sinogram generation assumes projection shape (H, W, A).
    if projections.ndim != 3:
        raise ValueError(f"Cannot generate sinogram from shape {projections.shape}; expected 3D array")

    # Use float32 accumulation to reduce overflow risk from integer TIFF inputs.
    projections_float = projections.astype(np.float32, copy=False)
    sino_sum = np.sum(projections_float, axis=0, dtype=np.float32)

    # Guard: reject non-finite collapsed sinogram values before ROI/segmentation.
    if not np.isfinite(sino_sum).all():
        raise ValueError("Collapsed sinogram contains NaN or Inf values")

    return sino_sum
