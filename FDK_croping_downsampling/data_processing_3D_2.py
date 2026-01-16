import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import re

def load_images(tiff_folder):
    print(f"--> [1/6] Loading images from: {tiff_folder}")
    
    if not os.path.exists(tiff_folder):
        print(f"Error: Folder does not exist.")
        return None

    # function to extract numbers from filenames for sorting 
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return 0

    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith('.tif')]
    file_list.sort(key=extract_number)
    num_files = len(file_list)
    print(f"Number of files found: {num_files}") 


    print("First 5 files (check order):") 
    for f in file_list[:5]: print(f" - {f}")
    print("Last 5 files (check order):") 
    for f in file_list[-5:]: print(f" - {f}")


    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape 
    print(f"Dimensions: {height} (H) x {width} (W) | {num_files} projections.")
    

    projections = np.zeros((height, width, num_files), dtype=first_img.dtype) #dtype information is need for memory optimization
    for i, f in enumerate(file_list):
        projections[:, :, i] = tiff.imread(os.path.join(tiff_folder, f))
    
    print("--> Images Loaded Successfully")
    return projections


def select_symmetric_roi_crop(projections):
    """
    Interactive symmetric ROI selection for detector cropping.
    
    User workflow:
    1. Click a pixel to set horizontal center (auto-symmetric)
    2. Enter crop width (pixels from center in each direction)
    3. Select two horizontal lines for vertical cropping (top and bottom)
    
    Returns:
        roi_bounds: tuple (row_start, row_end, col_start, col_end) or None if cancelled
    """
    print("--> [2/6] Select ROI for detector cropping (symmetric horizontal + manual vertical)...")
    
    # Show first projection
    first_proj = projections[:, :, 0]
    H, W = first_proj.shape
    center_col = W // 2  # Original detector center
    
    roi_bounds = [None]
    h_center = [center_col]  # Horizontal center for symmetric crop
    crop_width = [200]  # Default crop width (pixels from center)
    v_lines = [None, None]  # Top and bottom lines
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)
    
    im = ax.imshow(first_proj, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("1) Click pixel for horizontal center | 2) Adjust width | 3) Click top & bottom lines | 4) Confirm", 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Horizontal (px)")
    ax.set_ylabel("Vertical (px)")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Show original center line
    center_line = ax.axvline(x=center_col, color='yellow', linestyle=':', linewidth=1.5, 
                             label=f'Original Center ({center_col})')
    
    # ROI preview lines
    left_line = ax.axvline(x=center_col - crop_width[0], color='lime', linestyle='--', linewidth=2, 
                          label='Symmetric Crop')
    right_line = ax.axvline(x=center_col + crop_width[0], color='lime', linestyle='--', linewidth=2)
    top_line = ax.axhline(y=0, color='red', linestyle='--', linewidth=2, visible=False, label='Vertical Crop')
    bottom_line = ax.axhline(y=H-1, color='red', linestyle='--', linewidth=2, visible=False)
    
    ax.legend(loc='upper right')
    
    # Text boxes for info
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update_info():
        col_start = h_center[0] - crop_width[0]
        col_end = h_center[0] + crop_width[0]
        new_width = col_end - col_start
        
        row_start = int(v_lines[0]) if v_lines[0] is not None else 0
        row_end = int(v_lines[1]) if v_lines[1] is not None else H
        new_height = row_end - row_start
        
        reduction = (new_width * new_height) / (W * H)
        
        info_text.set_text(
            f"H-Center: {h_center[0]} px\n"
            f"Crop Width: ±{crop_width[0]} px\n"
            f"Horizontal: [{col_start}, {col_end}] = {new_width} px\n"
            f"Vertical: [{row_start}, {row_end}] = {new_height} px\n"
            f"Memory: {reduction*100:.1f}% of original"
        )
    
    def update_lines():
        col_start = h_center[0] - crop_width[0]
        col_end = h_center[0] + crop_width[0]
        left_line.set_xdata([col_start, col_start])
        right_line.set_xdata([col_end, col_end])
        center_line.set_xdata([h_center[0], h_center[0]])
        center_line.set_label(f'Selected Center ({h_center[0]})')
        
        if v_lines[0] is not None:
            top_line.set_ydata([v_lines[0], v_lines[0]])
            top_line.set_visible(True)
        if v_lines[1] is not None:
            bottom_line.set_ydata([v_lines[1], v_lines[1]])
            bottom_line.set_visible(True)
        
        ax.legend(loc='upper right')
        update_info()
        fig.canvas.draw()
    
    # Click handler for center selection and vertical lines
    click_count = [0]
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        # First click: set horizontal center
        if click_count[0] == 0:
            h_center[0] = int(event.xdata)
            click_count[0] = 1
            ax.set_title("Good! Now adjust width with slider, then click TOP line", 
                        fontsize=11, fontweight='bold')
            update_lines()
        
        # Second click: top line
        elif click_count[0] == 1:
            v_lines[0] = int(event.ydata)
            click_count[0] = 2
            ax.set_title("Great! Now click BOTTOM line, then Confirm", 
                        fontsize=11, fontweight='bold')
            update_lines()
        
        # Third click: bottom line
        elif click_count[0] == 2:
            v_lines[1] = int(event.ydata)
            click_count[0] = 3
            ax.set_title("Perfect! Click Confirm to apply ROI", 
                        fontsize=11, fontweight='bold', color='green')
            update_lines()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Width slider
    from matplotlib.widgets import Slider
    ax_slider = plt.axes([0.25, 0.12, 0.5, 0.03])
    slider = Slider(ax_slider, 'Crop Width (±px)', 50, W//2, valinit=crop_width[0], valstep=10)
    
    def update_width(val):
        crop_width[0] = int(val)
        update_lines()
    
    slider.on_changed(update_width)
    
    # Buttons
    ax_confirm = plt.axes([0.7, 0.05, 0.1, 0.05])
    ax_cancel = plt.axes([0.55, 0.05, 0.1, 0.05])
    btn_confirm = Button(ax_confirm, 'Confirm')
    btn_cancel = Button(ax_cancel, 'Cancel')
    
    def confirm(event):
        if click_count[0] >= 3:  # All steps completed
            col_start = h_center[0] - crop_width[0]
            col_end = h_center[0] + crop_width[0]
            row_start = int(min(v_lines[0], v_lines[1]))
            row_end = int(max(v_lines[0], v_lines[1]))
            
            # Clamp to valid range
            col_start = max(0, col_start)
            col_end = min(W, col_end)
            row_start = max(0, row_start)
            row_end = min(H, row_end)
            
            roi_bounds[0] = (row_start, row_end, col_start, col_end)
            plt.close(fig)
        else:
            print("Please complete all steps: 1) Click center, 2) Adjust width, 3) Click top & bottom lines")
    
    def cancel(event):
        roi_bounds[0] = None
        plt.close(fig)
    
    btn_confirm.on_clicked(confirm)
    btn_cancel.on_clicked(cancel)
    
    update_info()
    plt.show(block=True)
    
    return roi_bounds[0]


def apply_roi_crop(projections, roi_bounds):
    """
    Apply ROI cropping to all projections.
    
    Args:
        projections: array (H, W, A)
        roi_bounds: tuple (row_start, row_end, col_start, col_end)
    
    Returns:
        cropped_projections: array (H_new, W_new, A)
        crop_info: dict with original center offset for geometry adjustment
    """
    row_start, row_end, col_start, col_end = roi_bounds
    
    print(f"--> [3/6] Applying ROI crop...")
    print(f"    Original shape: {projections.shape}")
    print(f"    ROI bounds: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")
    
    cropped = projections[row_start:row_end, col_start:col_end, :]
    
    # Calculate offset from original center for geometry correction
    H_orig, W_orig, A = projections.shape
    H_new, W_new, _ = cropped.shape
    
    original_center_col = W_orig / 2.0
    new_center_col = (col_start + col_end) / 2.0
    horizontal_offset_px = new_center_col - original_center_col
    
    original_center_row = H_orig / 2.0
    new_center_row = (row_start + row_end) / 2.0
    vertical_offset_px = new_center_row - original_center_row
    
    crop_info = {
        'original_shape': (H_orig, W_orig),
        'cropped_shape': (H_new, W_new),
        'horizontal_offset_px': horizontal_offset_px,
        'vertical_offset_px': vertical_offset_px,
        'roi_bounds': roi_bounds
    }
    
    print(f"    Cropped shape: {cropped.shape}")
    print(f"    Horizontal offset from center: {horizontal_offset_px:.2f} px")
    print(f"    Vertical offset from center: {vertical_offset_px:.2f} px")
    print(f"    Memory reduction: {100*(1 - cropped.size/projections.size):.1f}%")
    
    return cropped, crop_info


def generate_collapsed_sinogram(projections):
    print("--> Creating collapsed sinogram (sum projection)...")
    sino_sum = np.sum(projections, axis=0)
    return sino_sum


def selecionar_roi_I0(sino_raw): 
    """
    Interface simplificada para selecionar apenas a ROI de background (I0) para normalização.
    O shift é obtido da calibração prévia.
    """
    print("--> [4/6] Select I0 ROI for normalization...")
    
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


def get_I0_from_roi(sino_raw, roi_background):
    """
    Calcula o valor médio de I0 a partir da ROI de background.
    """
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]
    mean_I0 = np.mean(roi_crop)
    return mean_I0



