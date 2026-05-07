import numpy as np 
import tifffile as tiff 
import os # for directory operations
import matplotlib.pyplot as plt 
from matplotlib.widgets import RectangleSelector, Button
import re 

def load_images(tiff_folder):
    print(f"Loading images from folder: {tiff_folder}")

    if not os.path.isdir(tiff_folder):
        raise FileNotFoundError(f"Projection folder not found: {tiff_folder!r}")

    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    def extract_number(filename): # Extract numeric part for sorting to overcome some ordering issues 
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return 0

    file_list.sort(key=extract_number)
    print(f"Number of files found: {len(file_list)}") 

    print("First 5 files (check order):") 
    for f in file_list[:5]: 
        print(f" - {f}")
    print("Last 5 files (check order):") 
    for f in file_list[-5:]: 
        print(f" - {f}")
    
    raw_data = [tiff.imread(os.path.join(tiff_folder, f)) for f in file_list]
    information = raw_data[0].shape
    projections_stack = np.stack(raw_data, axis=-1)
    dimension_stack = projections_stack.shape
    print(f"Loaded stack data shape: {dimension_stack} (Height, Width, Projections)") 

    return projections_stack


def extract_sinogram_raw(projections_stack, linha_escolhida):
    """
    Extract a specific line and stack into a sinogram.
    """
    # .T: (W, A) → (A, W); [:, ::-1]: flip detector axis to match TIGRE convention
    sino_raw = projections_stack[linha_escolhida, :, :].astype(np.float32).T[:, ::-1]
    return sino_raw

def selecionar_roi_I0(sino_raw): 
    """
    Interface to select a ROI for Io normalization.
    """
    print("--> Select I0 ROI for normalization...")
    
    roi_background = [None]
    fig, ax = plt.subplots(figsize=(14, 8)) 
    plt.subplots_adjust(bottom=0.20)

    ax.imshow(sino_raw.T, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("Draw ROI on BACKGROUND (air) for I0 normalization | Then click Confirm", fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("Angle (idx)")

    n_proj = sino_raw.shape[0]
    ax.set_xticks([0, n_proj-1])
    ax.set_xticklabels(['0°', '360°'])

    rect_bg = None

    def on_select_callback(eclick, erelease):
        nonlocal rect_bg
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        ang_start, ang_end = sorted([x1, x2])
        det_start, det_end = sorted([y1, y2])
        roi_coords = (ang_start, ang_end, det_start, det_end)

        roi_background[0] = roi_coords
        
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


def get_I0_from_roi(sino_raw, roi_background):
    """
    Find the mean value in the selected ROI for IO normalization.
    """
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]
    mean_I0 = np.mean(roi_crop)
    print(f"--> Calculated I0 from ROI: {mean_I0:.2f}")
    return mean_I0


def normalize_sinogram(sinogram_raw, I0_val=None):
    """
    Normalization: -ln(I / I0)
    """
    if I0_val is None:
        I0_val = float(np.percentile(sinogram_raw, 99))
        print(f"    WARNING: No I0_override provided. Using 99th percentile fallback: {I0_val:.2f}")
        print("    It is strongly recommended to provide I0_override from a calibrated ROI.")
    else:
        I0_val = float(I0_val)

    if I0_val <= 0:
        raise ValueError(f"I0 value is {I0_val:.4f} - must be positive. Check your ROI selection or raw data.")

    median_projection = float(np.percentile(sinogram_raw, 50))
    if I0_val < median_projection:
        print(f"    WARNING: I0 ({I0_val:.2f}) is below the median projection value ({median_projection:.2f}).")
        print("    This likely means I0 is too low and will produce incorrect attenuation values.")

    ratio = sinogram_raw.astype(np.float32, copy=True)
    ratio /= (I0_val + 1e-9)

    total_pixels = ratio.size
    clipped_above_count = np.count_nonzero(ratio > 1.2)
    clipped_below_count = np.count_nonzero(ratio < 0)
    clipped_above_pct = (clipped_above_count / total_pixels) * 100.0
    clipped_below_pct = (clipped_below_count / total_pixels) * 100.0
    print(f"    Clipping report (I/I0 ratio):")
    print(f"      - Pixels > 1.2: {clipped_above_pct:.4f}% ({clipped_above_count}/{total_pixels})")
    print(f"      - Pixels < 0:   {clipped_below_pct:.4f}% ({clipped_below_count}/{total_pixels})")

    np.clip(ratio, 1e-6, 1.2, out=ratio)
    
    sino_norm = -np.log(ratio)
    sino_norm[sino_norm < 0] = 0
    
    return sino_norm


def show_results(sino_raw, sino_final_norm, shift_val, linha_escolhida):
    """
    Visualization: sinograms before and after normalization and shift correction.
    """
    print("--> Showing final result...")

    centro_geo = sino_raw.shape[1] / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_proj = sino_raw.shape[0]
    ticks_deg = np.arange(0, 361, 30)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]

    ax1 = axes[0]
    ax1.imshow(sino_raw.T, cmap='gray', aspect='auto', origin='upper')
    ax1.set_title(f"1. RAW Sinogram - Slice {linha_escolhida}")
    ax1.axhline(y=centro_geo, color='red', linestyle='--', label='Geo Center')
    ax1.legend()
    ax1.set_ylabel("Detector (px)")
    ax1.set_xlabel("θ (degree)")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    ax2 = axes[1]
    ax2.imshow(sino_final_norm.T, cmap='gray', aspect='auto', origin='upper')
    ax2.set_title(f"2. Normalized Sinogram - Shift: {shift_val:.2f} px")
    ax2.axhline(y=centro_geo, color='red', linestyle='--', label='Geo Center')
    ax2.set_ylabel("Detector (px)")
    ax2.set_xlabel("θ (degree)")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()