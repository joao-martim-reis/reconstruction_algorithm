"""
ROI selection tools for calibration.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, RadioButtons


logger = logging.getLogger(__name__)

_ANGLE_TICK_STEP_DEG = 30


def selecionar_roi_interativamente(sino_raw: np.ndarray) -> tuple[tuple, tuple, bool]:
    """Interactively select background and object ROIs on a collapsed sinogram.

    Displays a matplotlib window where the user draws two rectangular ROIs:
    first the background (green), then the object (orange). A radio button
    allows the user to skip shift calculation if the detector is already centred.

    Args:
        sino_raw: Collapsed sinogram, shape (n_detector_px, n_angles).

    Returns:
        Tuple of (roi_background, roi_object, calculate_shift), where each ROI is
        (det_start, det_end, ang_start, ang_end) and calculate_shift is True when
        shift computation was requested.
    """
    logger.info("ROI selection and operation mode")

    roi_background = [None]
    roi_object = [None]
    calculate_options = [True]
    current_mode = ["background"]

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(sino_raw, cmap="gray", aspect="auto", origin="upper")
    ax.set_title(
        "STEP 1: Draw Background ROI (green) | STEP 2: Draw Object ROI (orange) | STEP 3: Confirm",
        fontsize=11,
    )
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("θ (degree)")

    # Set x-axis ticks every 30 degrees
    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, 361, _ANGLE_TICK_STEP_DEG)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    rect_bg = None
    rect_obj = None
    selector = [None]  # Keep selector alive to prevent garbage collection

    def on_select_callback(eclick, erelease):
        nonlocal rect_bg, rect_obj
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        ang_start, ang_end = sorted([x1, x2])
        det_start, det_end = sorted([y1, y2])
        roi_coords = (det_start, det_end, ang_start, ang_end)

        if current_mode[0] == "background":
            roi_background[0] = roi_coords
            logger.info(
                "Background ROI selected: Det[%d:%d], Ang[%d:%d]",
                det_start,
                det_end,
                ang_start,
                ang_end,
            )
            if rect_bg:
                rect_bg.remove()
            rect_bg = plt.Rectangle(
                (ang_start, det_start),
                ang_end - ang_start,
                det_end - det_start,
                fill=False,
                edgecolor="green",
                linewidth=3,
                linestyle="--",
            )
            ax.add_patch(rect_bg)
            current_mode[0] = "object"
            ax.set_title("STEP 2: Draw Object ROI (orange) | STEP 3: Confirm", fontsize=11)
        else:
            roi_object[0] = roi_coords
            logger.info(
                "Object ROI selected: Det[%d:%d], Ang[%d:%d]",
                det_start,
                det_end,
                ang_start,
                ang_end,
            )
            if rect_obj:
                rect_obj.remove()
            rect_obj = plt.Rectangle(
                (ang_start, det_start),
                ang_end - ang_start,
                det_end - det_start,
                fill=False,
                edgecolor="orange",
                linewidth=3,
                linestyle="-",
            )
            ax.add_patch(rect_obj)
            ax.set_title("DONE: Background (green) + Object (orange) | Click Confirm", fontsize=11)

        fig.canvas.draw_idle()

    selector[0] = RectangleSelector(
        ax,
        on_select_callback,
        useblit=False,  # Changed to False for better compatibility
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,  # Changed to False to avoid conflicts with drawn rectangles
        props=dict(facecolor="cyan", edgecolor="cyan", alpha=0.3, fill=True, linewidth=2),
    )

    ax_radio = plt.axes([0.05, 0.05, 0.25, 0.15], facecolor="#e4e4e4")
    radio = RadioButtons(ax_radio, ("Calculate Shift", "Already Centered (0 px)"))

    def change_mode(label):
        calculate_options[0] = label == "Calculate Shift"
        logger.info("Mode changed to: %s", label)

    radio.on_clicked(change_mode)

    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_button, "Confirm")

    def close_plot(event):
        plt.close(fig)

    btn_confirm.on_clicked(close_plot)
    plt.show(block=True)

    # Fallbacks
    if roi_background[0] is None:
        logger.warning("No background ROI selected. Using top-left fallback ROI")
        roi_background[0] = (0, 50, 0, 50)

    if roi_object[0] is None:
        logger.warning("No object ROI selected. Using full-image fallback ROI")
        height, width = sino_raw.shape
        roi_object[0] = (0, height, 0, width)

    return roi_background[0], roi_object[0], calculate_options[0]
