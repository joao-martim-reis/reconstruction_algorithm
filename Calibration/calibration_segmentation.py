"""
Segmentation and shift computation for calibration.
"""

import logging

import numpy as np
from scipy.ndimage import label, binary_opening, find_objects


MIN_COMPONENT_PIXELS = 200
_BACKGROUND_SIGMA_FACTOR = 2.0
logger = logging.getLogger(__name__)


def _validate_roi_bounds(sino_raw: np.ndarray, roi_coords: tuple, roi_name: str) -> None:
    """Validate ROI coordinates against sinogram bounds."""
    rows, cols = sino_raw.shape
    r_start, r_end, c_start, c_end = roi_coords

    # Guard: ROI bounds must be ordered and inside image limits.
    if not (0 <= r_start < r_end <= rows and 0 <= c_start < c_end <= cols):
        raise ValueError(
            f"Invalid {roi_name} ROI {roi_coords} for sinogram shape {sino_raw.shape}. "
            "Expected (r_start < r_end, c_start < c_end) within bounds."
        )


def calculate_shift_segmentation(
    sino_raw: np.ndarray,
    roi_background: tuple,
    roi_object: tuple,
) -> tuple:
    """Compute detector shift using ROI-driven segmentation.

    Args:
        sino_raw: Collapsed sinogram array with shape (n_detector_px, n_angles).
        roi_background: Background ROI as (r_start, r_end, c_start, c_end).
        roi_object: Object ROI as (r_start, r_end, c_start, c_end).

    Returns:
        Tuple containing:
            - shift_val (float): detector shift in pixels.
            - mask_final (np.ndarray): final binary object mask.
            - object_center (float): detected object center in detector pixels.
            - det_min (int): minimum detector index of object.
            - det_max (int): maximum detector index of object.
            - background_mean (float): mean background intensity from ROI.

    Raises:
        ValueError: If inputs are invalid or segmentation cannot find valid object pixels.
    """
    logger.info("Processing segmentation mask")

    # Guard: segmentation expects finite 2D sinogram input.
    if sino_raw is None or sino_raw.ndim != 2:
        raise ValueError("sino_raw must be a 2D array for segmentation")
    if not np.isfinite(sino_raw).all():
        raise ValueError("sino_raw contains NaN or Inf values")

    _validate_roi_bounds(sino_raw, roi_background, "background")
    _validate_roi_bounds(sino_raw, roi_object, "object")

    # Use background ROI for threshold stats
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]

    # Guard: ROI crops must contain valid finite values.
    if roi_crop.size == 0 or not np.isfinite(roi_crop).all():
        raise ValueError("Background ROI is empty or contains non-finite values")

    background_mean = np.mean(roi_crop)
    std_background = np.std(roi_crop)
    threshold = _BACKGROUND_SIGMA_FACTOR * std_background

    # Determine object contrast direction
    r_obj_start, r_obj_end, c_obj_start, c_obj_end = roi_object
    roi_object_crop = sino_raw[r_obj_start:r_obj_end, c_obj_start:c_obj_end]

    # Guard: object ROI must be valid for reliable contrast-direction detection.
    if roi_object_crop.size == 0 or not np.isfinite(roi_object_crop).all():
        raise ValueError("Object ROI is empty or contains non-finite values")

    mean_object = np.mean(roi_object_crop)

    logger.info("Background mean: %.2f | Object mean: %.2f", background_mean, mean_object)

    if mean_object < background_mean:
        logger.info("Object is DARKER than background")
        mask_initial = (background_mean - sino_raw) > threshold
    else:
        logger.info("Object is BRIGHTER than background")
        mask_initial = (sino_raw - background_mean) > threshold

    mask_cleaned = binary_opening(mask_initial, structure=np.ones((3, 3)))
    labeled_array, num_features = label(mask_cleaned)
    logger.info("Detected %d connected components", num_features)

    # Guard: explicit failure keeps calibration deterministic when segmentation is ambiguous.
    if num_features == 0:
        raise ValueError("No connected components were detected in segmentation mask")

    r_start, r_end, c_start, c_end = roi_object
    roi_mask = np.zeros_like(labeled_array, dtype=bool)
    roi_mask[r_start:r_end, c_start:c_end] = True

    best_label = None
    best_overlap = 0

    for i in range(1, num_features + 1):
        component = labeled_array == i

        n_pixels = np.sum(component)
        if n_pixels < MIN_COMPONENT_PIXELS:
            continue
        overlap = np.sum(component & roi_mask)

        if overlap > best_overlap:
            best_overlap = overlap
            best_label = i

    # Guard: reject fallback-to-largest-component to avoid silent wrong-center estimates.
    if best_label is None:
        raise ValueError(
            "No valid segmented component overlapped the object ROI. "
            "Adjust ROI selection or segmentation threshold conditions."
        )

    mask_final = labeled_array == best_label
    logger.info("Selected component %d with %d pixels inside ROI", best_label, best_overlap)

    obj_slices = find_objects(mask_final.astype(int))
    if obj_slices and obj_slices[0] is not None:
        det_min = obj_slices[0][0].start
        det_max = obj_slices[0][0].stop - 1
    else:
        coords = np.where(mask_final)
        # Guard: avoid silent crashes when no object survives segmentation.
        if coords[0].size == 0:
            raise ValueError("Segmentation produced an empty object mask; cannot compute detector shift")
        det_min = np.min(coords[0])
        det_max = np.max(coords[0])

    object_center = (det_min + det_max) / 2.0
    geo_center = sino_raw.shape[0] / 2.0

    shift_val = geo_center - object_center

    logger.info("Shift report | Object Center: %.2f | Geo Center: %.2f | Shift: %.2f px", object_center, geo_center, shift_val)

    return shift_val, mask_final, object_center, det_min, det_max, background_mean
