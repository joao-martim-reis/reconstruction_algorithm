import numpy as np
import tifffile as tiff
import os #os is a module that provides a way of using operating system dependent functionality
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import re

def load_images(tiff_folder, projection_fraction=1.0):
    print(f"--> Loading images from: {tiff_folder}")
    
    if not os.path.exists(tiff_folder): #os.path.exists checks if a path exists
        print(f"Error: Folder does not exist: {tiff_folder}")
        return None

    def extract_number(filename):
        match = re.search(r'\d+', filename) #re.search finds the first occurrence of the pattern
        if match: # If a match is found, return the integer value
            return int(match.group())
        return 0

    # Accept common TIFF extensions ('.tif' and '.tiff') in a case-insensitive way
    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith(('.tif', '.tiff'))] #os.listdir lists files in a directory
    file_list.sort(key=extract_number)
    if projection_fraction <= 0 or projection_fraction > 1:
        raise ValueError(f"projection_fraction must be in (0, 1], got {projection_fraction}")

    original_count = len(file_list)
    selected_count = max(1, int(np.floor(original_count * projection_fraction)))
    file_list = file_list[:selected_count]

    if projection_fraction < 1.0:
        print(f"Using first {projection_fraction*100:.1f}% of projections: {selected_count}/{original_count}")

    num_files = len(file_list)
    print(f"Number of files found: {num_files}") 

    print("First 5 files (check order):") 
    for f in file_list[:5]: print(f" - {f}")
    print("Last 5 files (check order):") 
    for f in file_list[-5:]: print(f" - {f}")

    if num_files == 0:
        print("Error: No TIFF files found in the folder.")

    
    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape 
    print(f"Dimensions: {height} (H) x {width} (W) | {num_files} projections.")
  
    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)
    for i, f in enumerate(file_list): #i is the index, f is the filename
        projections[:, :, i] = tiff.imread(os.path.join(tiff_folder, f))
    
    print("--> Images Loaded Successfully")
    return projections


def generate_collapsed_sinogram(projections):
    print("--> Creating collapsed sinogram (sum projection)...")
    sino_sum = np.sum(projections, axis=0)
    return sino_sum


def selecionar_roi_I0(sino_raw, angle_span_deg=360): 
    """
    Interface simplificada para selecionar apenas a ROI de background (I0) para normalização.
    O shift é obtido da calibração prévia.
    """
    print("--> Select I0 ROI for normalization...")
    
    roi_background = [None]
    fig, ax = plt.subplots(figsize=(14, 8)) 
    plt.subplots_adjust(bottom=0.20)

    ax.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("Draw ROI on BACKGROUND (air) for I0 normalization | Then click Confirm", fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("θ (degree)")

    # Set x-axis ticks every 30 degrees for the configured angular span
    n_proj = sino_raw.shape[1]
    ticks_deg = np.arange(0, int(np.ceil(angle_span_deg)) + 1, 30)
    tick_positions = [int(d * n_proj / float(angle_span_deg)) for d in ticks_deg]
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