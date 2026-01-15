import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import re

def load_images(tiff_folder):
    print(f"--> [1/5] Loading images from: {tiff_folder}")
    
    if not os.path.exists(tiff_folder):
        print(f"Error: Folder does not exist.")
        return None

    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return 0

    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith('.tif')]
    file_list.sort(key=extract_number)
    num_files = len(file_list)
    print(f"Number of files found: {num_files}") 

    if not file_list:
        print('Error: The folder is empty')
        return None


    print("First 5 files (check order):") 
    for f in file_list[:5]: print(f" - {f}")
    print("Last 5 files (check order):") 
    for f in file_list[-5:]: print(f" - {f}")


    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape 
    print(f"Dimensions: {height} (H) x {width} (W) | {num_files} projections.")
    try:
        print(f"    First image dtype: {first_img.dtype}, min={first_img.min()}, max={first_img.max()}")
    except Exception:
        print(f"    First image dtype: {first_img.dtype}")
    

    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)
    for i, f in enumerate(file_list):
        projections[:, :, i] = tiff.imread(os.path.join(tiff_folder, f))
    
    try:
        print(f"    Projections array dtype: {projections.dtype}, min={projections.min()}, max={projections.max()}")
    except Exception:
        print(f"    Projections array dtype: {projections.dtype}")

    print("--> Images Loaded Successfully")
    return projections


def generate_collapsed_sinogram(projections):
    print("--> [2/5] Creating collapsed sinogram (sum projection)...")
    sino_sum = np.sum(projections, axis=0)
    return sino_sum


def selecionar_roi_I0(sino_raw): 
    """
    Interface simplificada para selecionar apenas a ROI de background (I0) para normalização.
    O shift é obtido da calibração prévia.
    """
    print("--> [3/5] Select I0 ROI for normalization...")
    
    roi_background = [None]

    fig, ax = plt.subplots(figsize=(14, 8)) 
    plt.subplots_adjust(bottom=0.20)

    ax.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("Draw ROI on BACKGROUND (air) for I0 normalization | Then click Confirm", fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("θ (degree)")

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
        print(f"I0 ROI: Det[{det_start}:{det_end}], Ang[{ang_start}:{ang_end}]")
        
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

    if roi_background[0] is None:
        print("Warning: No I0 ROI selected. Using top-left corner.")
        roi_background[0] = (0, 50, 0, 50)
        
    return roi_background[0]


def get_I0_from_roi(sino_raw, roi_background):
    """
    Calcula o valor médio de I0 a partir da ROI de background.
    """
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]
    mean_I0 = np.mean(roi_crop)
    print(f"--> I0 (background mean): {mean_I0:.2f}")
    return mean_I0


def show_results(sino_raw, sino_final_norm, shift_val):
    """
    Visualização simplificada: sinograma raw e normalizado.
    """
    print("--> [5/5] Showing final result...")
    centro_geo = sino_raw.shape[0] / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle(f"Processing Complete | Calibrated Shift: {shift_val:.2f} px", fontsize=14)

    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, 361, 30)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]

    # PLOT 1: RAW
    ax1 = axes[0]
    ax1.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax1.set_title("1. RAW Sinogram")
    ax1.axhline(y=centro_geo, color='red', linestyle='--', label='Geo Center')
    ax1.legend()
    ax1.set_ylabel("Detector (px)")
    ax1.set_xlabel("θ (degree)")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # PLOT 2: NORMALIZED
    ax2 = axes[1]
    ax2.imshow(sino_final_norm, cmap='gray', aspect='auto', origin='upper')
    ax2.set_title("2. Normalized Sinogram")
    ax2.axhline(y=centro_geo, color='red', linestyle='--', label='Geo Center')
    ax2.set_ylabel("Detector (px)")
    ax2.set_xlabel("θ (degree)")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


