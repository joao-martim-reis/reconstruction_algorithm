"""
Visualization utilities for calibration.
"""

import matplotlib.pyplot as plt
import numpy as np


def show_results(sino_raw, mask, shift_val, centro_obj, d_min, d_max, modo_auto):
    print("--> Showing result...")
    centro_geo = sino_raw.shape[0] / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    titulo = "Automated Shift" if modo_auto else "Manual (Already Centered)"
    plt.suptitle(f"CT Calibration: {titulo}", fontsize=16)

    # Set x-axis ticks every 30 degrees
    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, 361, 30)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]

    # PLOT 1: RAW
    ax1 = axes[0]
    ax1.imshow(sino_raw, cmap="gray", aspect="auto", origin="upper")
    ax1.set_title("1. RAW Sinogram")
    ax1.axhline(y=centro_geo, color="red", linestyle="--", label="Geo Center")
    ax1.legend()
    ax1.set_ylabel("Detector (px)")
    ax1.set_xlabel("θ (degree)")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # PLOT 2: MASK WITH LIMITS
    ax2 = axes[1]
    if modo_auto and mask is not None:
        ax2.imshow(mask, cmap="binary", aspect="auto", origin="upper")
        ax2.set_title(f"2. Mask (Min={d_min}, Max={d_max})")

        ax2.axhline(y=d_min, color="orange", linestyle=":", linewidth=2, label=f"Min={d_min}")
        ax2.axhline(y=d_max, color="orange", linestyle=":", linewidth=2, label=f"Max={d_max}")
        ax2.axhline(
            y=centro_obj,
            color="lime",
            linestyle="-",
            linewidth=2,
            label=f"Object={centro_obj:.1f}",
        )
        ax2.axhline(
            y=centro_geo,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Geo={centro_geo:.1f}",
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
    ax3.axhline(y=centro_geo, color="red", linestyle="--", label="Geo Center")
    if modo_auto:
        ax3.axhline(y=centro_obj, color="lime", linestyle="-", label="Object Center")
    ax3.set_ylabel("Detector (px)")
    ax3.set_xlabel("θ (degree)")
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.legend()

    plt.tight_layout()
    plt.show()
