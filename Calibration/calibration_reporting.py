"""
Reporting utilities for calibration.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def compute_statistics(shifts, bg_means):
    avg_shift = np.mean(shifts) if len(shifts) > 0 else 0.0
    std_shift = np.std(shifts) if len(shifts) > 0 else 0.0
    avg_bg_mean = float(np.mean(bg_means)) if len(bg_means) > 0 else None
    return float(avg_shift), float(std_shift), avg_bg_mean


def save_results(parent_folder, distances, shifts, bg_means, avg_shift, std_shift, avg_bg_mean):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
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
        print(f"✓ JSON saved: {json_path}")
    except Exception as e:
        print(f"Warning: failed to save JSON: {e}")

    try:
        csv_path = os.path.join(parent_folder, f"calibration_results_{timestamp_str}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Distance_cm,Shift_px,Background_mean\n")
            for dist, shift, bg in zip(distances, shifts, bg_means):
                f.write(f"{dist:.2f},{shift:.4f},{bg:.4f}\n")
            f.write("\n# Summary\n")
            f.write(f"# Average Shift (px),{avg_shift:.4f}\n")
            f.write(f"# Std Deviation (px),{std_shift:.4f}\n")
            f.write(f"# Number of positions,{len(shifts)}\n")
        print(f"✓ CSV saved: {csv_path}")
    except Exception as e:
        print(f"Warning: failed to save CSV: {e}")

    return json_path, csv_path


def plot_results(distances, shifts, avg_shift, std_shift):
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


def print_final_report(distances, shifts, avg_shift, std_shift):
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print(f"Processed folders: {len(shifts)}")
    print("\nIndividual shifts:")
    for i, (dist, shift) in enumerate(zip(distances, shifts)):
        print(f"  {i + 1}. Distance {dist:+.2f} cm → Shift {shift:+.2f} px")
    print("\nFinal Result:")
    print(f"  Average Shift: {avg_shift:.2f} pixels")
    print(f"  Std Deviation: {std_shift:.2f} pixels")
