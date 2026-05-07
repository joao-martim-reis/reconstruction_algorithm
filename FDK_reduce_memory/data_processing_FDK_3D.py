import logging
import os #os is a module that provides a way of using operating system dependent functionality
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
# ThreadPoolExecutor: runs multiple tasks concurrently using a pool of threads.
# as_completed: returns futures in the order they finish (not submission order),
#               so the main thread can process each result as soon as it is ready.

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import numpy as np
import tifffile as tiff

logger = logging.getLogger(__name__)
PROGRESS_LOG_INTERVAL_SEC = 5.0
# I/O-bound: more workers hide per-call latency, especially over the WSL filesystem bridge.
MAX_LOAD_WORKERS = 12


def _format_elapsed(seconds):
    return f"{seconds:.2f}s"

def load_images(tiff_folder):
    logger.info("Loading images from: %s", tiff_folder)
    start_time = time.perf_counter()
    
    if not os.path.exists(tiff_folder): #os.path.exists checks if a path exists
        logger.error("Folder does not exist: %s", tiff_folder)
        return None

    def extract_number(filename):
        match = re.search(r'\d+', filename) #re.search finds the first occurrence of the pattern
        if match: # If a match is found, return the integer value
            return int(match.group())
        return 0

    # Accept common TIFF extensions ('.tif' and '.tiff') in a case-insensitive way
    list_start = time.perf_counter()
    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith(('.tif', '.tiff'))] #os.listdir lists files in a directory
    file_list.sort(key=extract_number)
    num_files = len(file_list)
    list_elapsed = time.perf_counter() - list_start
    logger.info("Found %d TIFF files in %s", num_files, _format_elapsed(list_elapsed))

    if num_files > 0:
        logger.info("First file: %s", file_list[0])
        logger.info("Last file: %s", file_list[-1])

    if num_files == 0:
        logger.error("No TIFF files found in the folder.")

    
    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape 
    logger.info(
        "First image shape: (%d, %d) | dtype: %s | projections: %d",
        height,
        width,
        str(first_img.dtype),
        num_files,
    )
  
    # Pre-allocate the full 3D array (H x W x N) in one shot to avoid repeated memory reallocation.
    # dtype is taken from the first image so uint16 and float32 files are both handled correctly.
    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)

    completed = 0
    last_log_time = time.perf_counter()

    # --- Parallel loading with ThreadPoolExecutor ---
    # BEFORE: a plain for-loop called tiff.imread() one file at a time. Each call had to
    #         wait for the previous one to finish before starting — pure sequential I/O.
    #         Over the WSL filesystem bridge (\\wsl.localhost\...) this was ~0.03 img/s
    #         because every read crosses a virtual-network layer with high per-call overhead.
    #
    # AFTER:  up to MAX_LOAD_WORKERS threads issue imread() calls concurrently. While one
    #         thread is waiting for the OS to return bytes, another thread's read is already
    #         in flight. This hides per-call latency and keeps the I/O pipeline full.
    #         Reading is I/O-bound (not CPU-bound), so many threads help without fighting
    #         over the CPU.
    with ThreadPoolExecutor(max_workers=MAX_LOAD_WORKERS) as pool:

        # Submit all file reads at once. pool.submit() is non-blocking: it schedules the
        # work and immediately returns a Future object. The dict maps each Future → its
        # original index i, so we know where to store the result in `projections`.
        futures = {
            pool.submit(tiff.imread, os.path.join(tiff_folder, f)): i
            for i, f in enumerate(file_list)
        }

        # as_completed() yields each Future the moment its thread finishes reading.
        # The main thread is the only one writing to `projections`, so there is no
        # race condition: each i is unique, meaning every write goes to a different
        # memory region (projections[:, :, i] is a distinct slice for each i).
        for fut in as_completed(futures):
            i = futures[fut]                   # recover the original file index
            projections[:, :, i] = fut.result()  # store the decoded image into the right slot
            completed += 1
            now = time.perf_counter()
            if completed < num_files and (now - last_log_time) >= PROGRESS_LOG_INTERVAL_SEC:
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Loaded %d/%d images (%.1f%%) | %.2f img/s | elapsed %s",
                    completed,
                    num_files,
                    100.0 * completed / num_files,
                    rate,
                    _format_elapsed(elapsed),
                )
                last_log_time = now
    
    total_elapsed = time.perf_counter() - start_time
    avg_rate = num_files / total_elapsed if total_elapsed > 0 else 0.0
    logger.info(
        "Images loaded successfully: %d in %s (avg %.2f img/s)",
        num_files,
        _format_elapsed(total_elapsed),
        avg_rate,
    )
    return projections


def generate_collapsed_sinogram(projections):
    print("--> Creating collapsed sinogram (sum projection)...")
    sino_sum = np.sum(projections, axis=0) #axis = 0, means we are summing along the first dimension (height), resulting in a 2D array of shape (width, num_projections) which is the collapsed sinogram.
    return sino_sum


def selecionar_roi_I0(sino_raw): 
    """
    Simplified interface to select only the background ROI (I0) for normalization.
    The shift is obtained from prior calibration.
    """
    print("--> Select I0 ROI for normalization...")
    
    roi_background = [None]
    fig, ax = plt.subplots(figsize=(14, 8)) 
    plt.subplots_adjust(bottom=0.20)

    ax.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("Draw ROI on BACKGROUND (air) for I0 normalization | Then click Confirm", fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("θ (degree)")

    # Set x-axis ticks every 30 degrees
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
        
        # Desenhar retângulo verde para background
        if rect_bg:
            rect_bg.remove()
        rect_bg = plt.Rectangle((ang_start, det_start), ang_end-ang_start, det_end-det_start,
                                 fill=False, edgecolor='green', linewidth=2, linestyle='--')
        ax.add_patch(rect_bg)
        ax.set_title("I0 ROI selected (green) | Click Confirm", fontsize=12)
        
        fig.canvas.draw()

    rect_selector = RectangleSelector(
        ax, on_select_callback, useblit=True, button=[1], 
        minspanx=5, minspany=5, spancoords='pixels', interactive=True,
        props=dict(facecolor='green', edgecolor='green', alpha=0.2, fill=True)
    )

    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_button, 'Confirm') 

    def close_plot(event):
        plt.close(fig)

    btn_confirm.on_clicked(close_plot)
    plt.show(block=True)
        
    return roi_background[0]


def get_I0_from_roi(sino_raw, roi_background, height):
    """
    Calculate the mean I0 value from the background ROI.
    """
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]
    mean_I0 = (np.mean(roi_crop)) / height # Normalize by height to get average per pixel as the sinogram is a sum projection 
    print(f"--> I0 value: {mean_I0:.2f}")
    return mean_I0