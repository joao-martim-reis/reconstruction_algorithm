"""
Visualization utilities for calibration.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)

_ANGLE_TICK_STEP_DEG = 30


def show_results(
    sino_raw: np.ndarray,
    mask: np.ndarray | None,
    shift_val: float,
    object_center: float,
    d_min: int,
    d_max: int,
    auto_mode: bool,
) -> None:
    """Display a three-panel calibration result figure.

    Shows: (1) raw sinogram with geometric centre, (2) segmentation mask with
    object and geometric centre lines, (3) calibration result with computed shift.

    Args:
        sino_raw: Collapsed sinogram, shape (n_detector_px, n_angles).
        mask: Binary segmentation mask, same shape as sino_raw, or None.
        shift_val: Computed detector shift in pixels.
        object_center: Detected object centre in detector pixels.
        d_min: Minimum detector index of the segmented object.
        d_max: Maximum detector index of the segmented object.
        auto_mode: True when automated shift was computed; False for manual mode.
    """
    logger.info("Showing calibration visualization")
    geo_center = sino_raw.shape[0] / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    title = "Automated Shift" if auto_mode else "Manual (Already Centered)"
    plt.suptitle(f"CT Calibration: {title}", fontsize=16)

    # Set x-axis ticks every 30 degrees
    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, 361, _ANGLE_TICK_STEP_DEG)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]

    # PLOT 1: RAW
    ax1 = axes[0]
    ax1.imshow(sino_raw, cmap="gray", aspect="auto", origin="upper")
    ax1.set_title("1. RAW Sinogram")
    ax1.axhline(y=geo_center, color="red", linestyle="--", label="Geo Center")
    ax1.legend()
    ax1.set_ylabel("Detector (px)")
    ax1.set_xlabel("θ (degree)")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # PLOT 2: MASK WITH LIMITS
    ax2 = axes[1]
    if auto_mode and mask is not None:
        ax2.imshow(mask, cmap="binary", aspect="auto", origin="upper")
        ax2.set_title(f"2. Mask (Min={d_min}, Max={d_max})")

        ax2.axhline(y=d_min, color="orange", linestyle=":", linewidth=2, label=f"Min={d_min}")
        ax2.axhline(y=d_max, color="orange", linestyle=":", linewidth=2, label=f"Max={d_max}")
        ax2.axhline(
            y=object_center,
            color="lime",
            linestyle="-",
            linewidth=2,
            label=f"Object={object_center:.1f}",
        )
        ax2.axhline(
            y=geo_center,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Geo={geo_center:.1f}",
        )
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Mask not used", ha="center", va="center")
        ax2.set_title("2. Mask Ignored")
    ax2.set_ylabel("Detector (px)")
    ax2.set_xlabel("θ (degree)")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    # PLOT 3: RESULT
    ax3 = axes[2]
    ax3.imshow(sino_raw, cmap="gray", aspect="auto", origin="upper")
    ax3.set_title(f"3. Calibration Result\nShift: {shift_val:.2f} px")
    ax3.axhline(y=geo_center, color="red", linestyle="--", label="Geo Center")
    if auto_mode:
        ax3.axhline(y=object_center, color="lime", linestyle="-", label="Object Center")
    ax3.set_ylabel("Detector (px)")
    ax3.set_xlabel("θ (degree)")
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.legend()

    plt.tight_layout()
    plt.show()
