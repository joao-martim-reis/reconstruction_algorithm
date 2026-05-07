"""
Reporting utilities for calibration.
"""

import json
import os
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


def _validate_results_lengths(distances, shifts, bg_means):
    """Validate that per-position result vectors are aligned."""
    # Guard: avoid silent CSV/JSON truncation from mismatched list lengths.
    if not (len(distances) == len(shifts) == len(bg_means)):
        raise ValueError(
            "Result length mismatch: "
            f"distances={len(distances)}, shifts={len(shifts)}, bg_means={len(bg_means)}"
        )


def compute_statistics(
    shifts: list,
    bg_means: list,
) -> tuple[float, float, float | None]:
    """Compute summary statistics from a list of calibration shifts.

    Args:
        shifts: Per-position detector shift values in pixels.
        bg_means: Per-position background mean intensities.

    Returns:
        Tuple of (avg_shift, std_shift, avg_bg_mean). avg_bg_mean is None when
        bg_means is empty.
    """
    avg_shift = np.mean(shifts) if len(shifts) > 0 else 0.0
    std_shift = np.std(shifts) if len(shifts) > 0 else 0.0
    avg_bg_mean = float(np.mean(bg_means)) if len(bg_means) > 0 else None
    return float(avg_shift), float(std_shift), avg_bg_mean


def save_results(
    parent_folder: str,
    distances: list,
    shifts: list,
    bg_means: list,
    avg_shift: float,
    std_shift: float,
    avg_bg_mean: float | None,
    calibration_mode: str = "multi_distance",
) -> tuple[str | None, str | None]:
    """Save calibration results to timestamped JSON and CSV files.

    Args:
        parent_folder: Directory where output files are written.
        distances: Per-position distances in centimeters.
        shifts: Per-position detector shift values in pixels.
        bg_means: Per-position background mean intensities.
        avg_shift: Mean shift across all positions in pixels.
        std_shift: Standard deviation of shifts in pixels.
        avg_bg_mean: Mean background intensity, or None if unavailable.
        calibration_mode: One of "multi_distance" or "single_folder".

    Returns:
        Tuple of (json_path, csv_path) — either may be None if the write failed.
    """
    _validate_results_lengths(distances, shifts, bg_means)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "calibration_mode": calibration_mode,
        "single_folder_mode": calibration_mode == "single_folder",
        "distances_cm": distances,
        "shifts_px": shifts,
        "average_shift_px": float(avg_shift),
        "std_shift_px": float(std_shift),
        "background_means": bg_means,
        "average_background_mean": avg_bg_mean,
        "n_positions": len(shifts),
        "timestamp": datetime.now().isoformat(),
    }

    json_path = None
    csv_path = None

    try:
        json_path = os.path.join(parent_folder, f"calibration_results_{timestamp_str}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("JSON saved: %s", json_path)
    except Exception as e:
        logger.warning("Failed to save JSON: %s", e)

    try:
        csv_path = os.path.join(parent_folder, f"calibration_results_{timestamp_str}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Distance_cm,Shift_px,Background_mean\n")
            for dist, shift, bg in zip(distances, shifts, bg_means):
                f.write(f"{dist:.2f},{shift:.4f},{bg:.4f}\n")
            f.write("\n# Summary\n")
            f.write(f"# Calibration mode,{calibration_mode}\n")
            f.write(f"# Average Shift (px),{avg_shift:.4f}\n")
            f.write(f"# Std Deviation (px),{std_shift:.4f}\n")
            f.write(f"# Number of positions,{len(shifts)}\n")
        logger.info("CSV saved: %s", csv_path)
    except Exception as e:
        logger.warning("Failed to save CSV: %s", e)

    return json_path, csv_path


def plot_results(
    distances: list,
    shifts: list,
    avg_shift: float,
    std_shift: float,
) -> None:
    """Plot calibration shift vs. position with mean ± std reference lines.

    Args:
        distances: Per-position distances in centimeters.
        shifts: Per-position detector shift values in pixels.
        avg_shift: Mean shift in pixels (drawn as dashed reference line).
        std_shift: Standard deviation in pixels (drawn as dotted bounds).
    """
    if len(shifts) == 0:
        logger.warning("No shifts available for summary plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(
        distances,
        shifts,
        "o-",
        markersize=10,
        linewidth=2,
        color="dodgerblue",
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    plt.axhline(avg_shift, linestyle="--", color="red", linewidth=2, label=f"Average = {avg_shift:.2f} px")
    plt.axhline(avg_shift + std_shift, linestyle=":", color="gray", linewidth=1)
    plt.axhline(avg_shift - std_shift, linestyle=":", color="gray", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Distance from Center [cm]", fontsize=12, fontweight="bold")
    plt.ylabel("Detector Shift [pixels]", fontsize=12, fontweight="bold")
    plt.title("Multi-Position Calibration", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def print_final_report(
    distances: list,
    shifts: list,
    avg_shift: float,
    std_shift: float,
    calibration_mode: str = "multi_distance",
) -> None:
    """Print a human-readable calibration summary to stdout.

    Args:
        distances: Per-position distances in centimeters.
        shifts: Per-position detector shift values in pixels.
        avg_shift: Mean shift in pixels.
        std_shift: Standard deviation of shifts in pixels.
        calibration_mode: One of "multi_distance" or "single_folder".
    """
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print(f"Mode: {calibration_mode}")
    print(f"Processed folders: {len(shifts)}")

    # Guard: provide explicit summary when no valid folders produced shifts.
    if len(shifts) == 0:
        print("No valid shifts available for final report.")
        return

    print("\nIndividual shifts:")
    for i, (dist, shift) in enumerate(zip(distances, shifts)):
        if calibration_mode == "single_folder":
            print(f"  {i + 1}. Single-folder input (0.00 cm) -> Shift {shift:+.2f} px")
        else:
            print(f"  {i + 1}. Distance {dist:+.2f} cm -> Shift {shift:+.2f} px")
    print("\nFinal Result:")
    print(f"  Average Shift: {avg_shift:.2f} pixels")
    print(f"  Std Deviation: {std_shift:.2f} pixels")
