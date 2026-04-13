import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.widgets import Button, RectangleSelector


def load_images(tiff_folder):
    """
    Memory-optimized loader entry point (name aligned with `FBP/data_processing_2D.py`).
    Returns sorted TIFF file paths instead of loading all images into RAM.
    """
    if not os.path.exists(tiff_folder):
        raise FileNotFoundError(f"Folder does not exist: {tiff_folder}")

    def extract_number(filename):
        match = re.search(r"\d+", filename)
        if match:
            return int(match.group())
        return 0

    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith((".tif", ".tiff"))]
    file_list.sort(key=extract_number)

    if not file_list:
        raise FileNotFoundError(f"No TIFF files found in folder: {tiff_folder}")

    return [os.path.join(tiff_folder, f) for f in file_list]


def list_tiff_files(tiff_folder):
    """
    Backward-compatible alias for previous naming.
    """
    return load_images(tiff_folder)


def validate_projection_set(file_paths):
    first_img = tiff.imread(file_paths[0])
    if first_img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale projections. Got shape {first_img.shape}")

    height, width = first_img.shape
    n_proj = len(file_paths)

    sample_idx = sorted(set([0, n_proj // 2, n_proj - 1]))
    for idx in sample_idx:
        img = tiff.imread(file_paths[idx])
        if img.shape != (height, width):
            raise ValueError(
                f"Projection shape mismatch at index {idx}: got {img.shape}, expected {(height, width)}"
            )

    return height, width, n_proj


def load_projection_batch(file_paths, start_idx, end_idx, dtype=np.float32, row_range=None, col_range=None):
    """
    Load a batch of projection files and optionally return only a row/column subregion.

    Parameters
    - file_paths: list of file paths
    - start_idx, end_idx: slice indices into file_paths (end exclusive)
    - dtype: output dtype
    - row_range: tuple (row_start, row_end) in full-image coordinates or None
    - col_range: tuple (col_start, col_end) in full-image coordinates or None

    Returns array of shape (rows, cols, n_files)
    """
    batch_files = file_paths[start_idx:end_idx]
    first = tiff.imread(batch_files[0])
    h, w = first.shape

    r0, r1 = (0, h) if row_range is None else (int(row_range[0]), int(row_range[1]))
    c0, c1 = (0, w) if col_range is None else (int(col_range[0]), int(col_range[1]))

    out_h = r1 - r0
    out_w = c1 - c0
    out = np.empty((out_h, out_w, len(batch_files)), dtype=dtype)

    # read first and slice
    tmp = first[r0:r1, c0:c1].astype(dtype, copy=False)
    out[:, :, 0] = tmp

    for i, fp in enumerate(batch_files[1:], start=1):
        img = tiff.imread(fp)
        out[:, :, i] = img[r0:r1, c0:c1].astype(dtype, copy=False)

    return out


def build_collapsed_sinogram_from_crop(file_paths, crop_params, angle_batch_size=32):
    n_proj = len(file_paths)

    test_img = tiff.imread(file_paths[0])
    full_h, full_w = test_img.shape

    if crop_params is None:
        row_start, row_end = 0, full_h
        col_start, col_end = 0, full_w
    else:
        row_start = crop_params["row_start"]
        row_end = crop_params["row_end"]
        col_start = crop_params["col_start"]
        col_end = crop_params["col_end"]

    width_crop = col_end - col_start
    collapsed = np.empty((width_crop, n_proj), dtype=np.float32)

    for a0 in range(0, n_proj, angle_batch_size):
        a1 = min(a0 + angle_batch_size, n_proj)
        for ang, fp in enumerate(file_paths[a0:a1], start=a0):
            proj = tiff.imread(fp).astype(np.float32, copy=False)
            cropped = proj[row_start:row_end, col_start:col_end]
            collapsed[:, ang] = np.sum(cropped, axis=0)

    return collapsed


def extract_sinogram_raw(file_paths, crop_params=None, angle_batch_size=32):
    """
    Memory-optimized raw sinogram extractor (name aligned with `FBP/data_processing_2D.py`).
    Here, extraction is performed from file paths in batches, optionally with crop.
    """
    return build_collapsed_sinogram_from_crop(
        file_paths=file_paths,
        crop_params=crop_params,
        angle_batch_size=angle_batch_size,
    )


def selecionar_roi_I0(sino_raw):
    print("--> Select I0 ROI for normalization...")

    roi_background = [None]
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.20)

    ax.imshow(sino_raw, cmap="gray", aspect="auto", origin="upper")
    ax.set_title("Draw ROI on BACKGROUND (air) for I0 normalization | Then click Confirm", fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("Theta (degree)")

    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, 361, 30)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    rect_bg = None

    def on_select_callback(eclick, erelease):
        nonlocal rect_bg
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        ang_start, ang_end = sorted([x1, x2])
        det_start, det_end = sorted([y1, y2])
        roi_coords = (det_start, det_end, ang_start, ang_end)

        roi_background[0] = roi_coords

        if rect_bg:
            rect_bg.remove()
        rect_bg = plt.Rectangle(
            (ang_start, det_start),
            ang_end - ang_start,
            det_end - det_start,
            fill=False,
            edgecolor="green",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(rect_bg)
        ax.set_title("I0 ROI selected (green) | Click Confirm", fontsize=12)

        fig.canvas.draw()

    rect_selector = RectangleSelector(
        ax,
        on_select_callback,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(facecolor="green", edgecolor="green", alpha=0.2, fill=True),
    )

    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_button, "Confirm")

    def close_plot(_event):
        plt.close(fig)

    btn_confirm.on_clicked(close_plot)
    plt.show(block=True)

    return roi_background[0]


def get_I0_from_roi(collapsed_sino, roi_background, cropped_height=None):
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = collapsed_sino[r_start:r_end, c_start:c_end]
    mean_I0 = float(np.mean(roi_crop))
    if cropped_height is not None:
        mean_I0 /= float(cropped_height)
    print(f"--> I0 value: {mean_I0:.4f}")
    return mean_I0


def normalize_sinogram(sinogram_raw, I0_val=None):
    if I0_val is None:
        I0_val = float(np.percentile(sinogram_raw, 99))
        print(f"    WARNING: No I0 provided. Using 99th percentile fallback: {I0_val:.2f}")
    else:
        I0_val = float(I0_val)

    if I0_val <= 0:
        raise ValueError(f"I0 must be positive. Got {I0_val}")

    ratio = sinogram_raw.astype(np.float32, copy=True)
    ratio /= (I0_val + 1e-9)
    np.clip(ratio, 1e-6, 1.2, out=ratio)
    np.log(ratio, out=ratio)
    ratio *= -1.0
    ratio[ratio < 0] = 0

    return ratio
