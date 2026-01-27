"""
================================================================================
CONTRAST-TO-NOISE RATIO (CNR) - MICRO-CT QUALITY CONTROL
================================================================================

PHANTOM USED: Water Phantom (with air region visible outside)

METRIC CALCULATED:
------------------
CNR (Contrast-to-Noise Ratio): Measures the ability to distinguish between 
two different materials/tissues based on their grey value difference relative 
to the noise level.

FORMULA:
--------
    CNR = |μ_water - μ_air| / √(σ_water² + σ_air²)

Where:
- μ_water: Mean grey value in the water ROI
- μ_air: Mean grey value in the air ROI  
- σ_water: Standard deviation (noise) in the water ROI
- σ_air: Standard deviation (noise) in the air ROI

WHY THIS METRIC?
----------------
- CNR indicates the detectability of low-contrast structures
- Higher CNR = better ability to distinguish between materials
- Important for diagnostic imaging where subtle differences must be detected
- Reference: Mambrini et al. (2022), Scientific Reports

MEASUREMENT PROCEDURE:
----------------------
1. Place one ROI in the center of the water phantom (water region)
2. Place one ROI outside the phantom (air region) - USER SELECTS BY CLICKING
3. Both ROIs are cylindrical (3D) spanning multiple slices for better statistics
4. Analysis is performed on GREY SCALE values (no HU conversion)

================================================================================
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from openpyxl import Workbook
from datetime import datetime
import os


# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_roi_circular(stack, center_x, center_y, radius, start_slice, num_slices):
    """
    Extract all pixel values within a cylindrical ROI.
    
    Similar to the uniformity script, this extracts a cylindrical volume
    for better statistical reliability in CNR calculations.
    
    Parameters:
    -----------
    stack : ndarray
        3D image stack (slices, height, width)
    center_x, center_y : int
        Center coordinates of the circular ROI in pixels
    radius : int
        Radius of the circular ROI in pixels
    start_slice : int
        First slice to include
    num_slices : int
        Number of slices to include (height of cylinder)
    
    Returns:
    --------
    values : ndarray
        1D array of all pixel values within the cylindrical ROI
    
    Notes:
    ------
    - Uses circular mask: (x-cx)² + (y-cy)² ≤ r²
    - Handles boundary cases: crops if ROI extends beyond image edges
    """
    
    # Define bounding box (rectangular region containing the circle)
    y_min = max(0, center_y - radius)
    y_max = min(stack.shape[1], center_y + radius + 1)
    x_min = max(0, center_x - radius)
    x_max = min(stack.shape[2], center_x + radius + 1)
    
    # Create circular mask using distance formula
    y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    
    # Extract pixel values across all slices
    values = []
    for i in range(start_slice, start_slice + num_slices):
        roi_region = stack[i, y_min:y_max, x_min:x_max]
        values.extend(roi_region[mask])
    
    return np.array(values)


def calculate_roi_size_mm(radius_px, height_slices, pixel_mm):
    """
    Calculate physical ROI dimensions in mm³.
    
    Parameters:
    -----------
    radius_px : int
        ROI radius in pixels
    height_slices : int
        Number of slices (height of cylinder)
    pixel_mm : float
        Pixel size in mm (assumes isotropic voxels)
    
    Returns:
    --------
    diameter_mm, height_mm, volume_mm3 : tuple of floats
        Physical dimensions in millimeters
    """
    
    diameter_mm = 2 * radius_px * pixel_mm
    height_mm = height_slices * pixel_mm
    volume_mm3 = np.pi * (radius_px * pixel_mm)**2 * height_mm
    
    return diameter_mm, height_mm, volume_mm3


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main(tif_path, output_folder, pixel_size_mm, roi_radius, slices_above_below):
    """
    Main function for CNR analysis.
    
    Parameters:
    -----------
    tif_path : str
        Path to input TIFF stack
    output_folder : str
        Path to output folder for results
    pixel_size_mm : float
        Pixel size in mm (for physical dimension conversion)
    roi_radius : int
        Radius of circular ROIs in pixels
    slices_above_below : int
        Number of slices above/below center for volumetric ROI
    """
    
    # =========================================================================
    # LOAD IMAGE STACK
    # =========================================================================
    
    print("Loading image stack...")
    stack = tifffile.imread(tif_path)
    num_slices, height, width = stack.shape
    
    # Calculate image center - water ROI will be placed here
    center_slice = num_slices // 2
    center_y, center_x = height // 2, width // 2
    print(f"Stack loaded: {num_slices} slices, {height}×{width} pixels")
    print(f"Image center: ({center_x}, {center_y}), slice {center_slice}")
    
    # =========================================================================
    # CALCULATE SLICE RANGE FOR VOLUMETRIC ROI
    # =========================================================================
    
    # Use multiple slices for better statistical reliability
    start_slice = max(0, center_slice - slices_above_below)
    end_slice = min(num_slices, center_slice + slices_above_below + 1)
    actual_height = end_slice - start_slice
    
    # =========================================================================
    # CALCULATE AND DISPLAY ROI PHYSICAL SIZE
    # =========================================================================
    
    diameter_mm, height_mm, volume_mm3 = calculate_roi_size_mm(roi_radius, actual_height, pixel_size_mm)
    print(f"\nROI Configuration:")
    print(f"  Size: {diameter_mm:.2f} × {diameter_mm:.2f} × {height_mm:.2f} mm³ (D × D × H)")
    print(f"  Volume: {volume_mm3:.2f} mm³")
    print(f"  Slices: {start_slice} to {end_slice-1} (center: {center_slice}, total: {actual_height})")
    
    # =========================================================================
    # EXTRACT WATER ROI (CENTER OF PHANTOM)
    # =========================================================================
    
    print("\n--- Extracting WATER ROI (center of phantom) ---")
    # Water ROI is automatically placed at the center of the image
    # This assumes the phantom is properly centered in the field of view
    water_values = extract_roi_circular(stack, center_x, center_y, roi_radius, start_slice, actual_height)
    water_mean = np.mean(water_values)
    water_std = np.std(water_values)
    print(f"Water ROI: mean = {water_mean:.2f}, std = {water_std:.2f}, n = {len(water_values)} pixels")
    
    # =========================================================================
    # INTERACTIVE AIR ROI SELECTION
    # =========================================================================
    
    print("\n--- Select AIR ROI (click outside the phantom) ---")
    print("Instructions:")
    print("  1. A window will open showing the central slice")
    print("  2. Click on a location OUTSIDE the phantom (in the air region)")
    print("  3. A circular ROI will be drawn at that location")
    print("  4. Close the window to continue")
    
    # Create a figure for interactive ROI selection
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(stack[center_slice], cmap='gray')
    
    # Draw the water ROI (already placed at center)
    water_circle = plt.Circle((center_x, center_y), roi_radius, fill=False, color='cyan', linewidth=2)
    ax.add_patch(water_circle)
    ax.text(center_x, center_y - roi_radius - 10, 'WATER', color='cyan', ha='center', fontweight='bold', fontsize=12)
    
    ax.set_title(f'Click to select AIR ROI location\n(Slice {center_slice})', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Variable to store the clicked air ROI position
    air_x, air_y = None, None
    air_circle_patch = None
    
    # Event handler for mouse click
    def onclick(event):
        """
        Mouse click event handler - captures the user's click position
        and draws the air ROI at that location.
        """
        nonlocal air_x, air_y, air_circle_patch
        
        # Only process clicks inside the axes
        if event.inaxes != ax:
            return
        
        # Get click coordinates (rounded to nearest pixel)
        air_x, air_y = int(round(event.xdata)), int(round(event.ydata))
        
        # Remove previous air ROI circle if it exists
        if air_circle_patch is not None:
            air_circle_patch.remove()
        
        # Draw new air ROI circle at clicked position
        air_circle_patch = plt.Circle((air_x, air_y), roi_radius, fill=False, color='red', linewidth=2)
        ax.add_patch(air_circle_patch)
        ax.text(air_x, air_y - roi_radius - 10, 'AIR', color='red', ha='center', fontweight='bold', fontsize=12)
        
        # Redraw the figure to show the new ROI
        fig.canvas.draw()
        print(f"  Air ROI positioned at: ({air_x}, {air_y})")
    
    # Connect the click event to the handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    # =========================================================================
    # VALIDATE AIR ROI SELECTION
    # =========================================================================
    
    # Check if user actually clicked to select an air ROI
    if air_x is None or air_y is None:
        print("\nERROR: No air ROI was selected! Please run the script again and click on the air region.")
        return
    
    print(f"\nAir ROI selected at: ({air_x}, {air_y})")
    
    # =========================================================================
    # EXTRACT AIR ROI VALUES
    # =========================================================================
    
    print("\n--- Extracting AIR ROI ---")
    # Extract pixel values from the user-selected air region
    air_values = extract_roi_circular(stack, air_x, air_y, roi_radius, start_slice, actual_height)
    air_mean = np.mean(air_values)
    air_std = np.std(air_values)
    print(f"Air ROI: mean = {air_mean:.2f}, std = {air_std:.2f}, n = {len(air_values)} pixels")
    
    # =========================================================================
    # CALCULATE CNR
    # =========================================================================
    
    print("\n--- Calculating CNR ---")
    
    # CNR Formula: |μ_water - μ_air| / √(σ_water² + σ_air²)
    # 
    # The numerator represents the signal difference (contrast)
    # The denominator represents the combined noise from both regions
    # 
    # Interpretation:
    # - CNR > 5: Excellent contrast, structures easily distinguishable
    # - CNR 3-5: Good contrast, structures clearly visible
    # - CNR 1-3: Moderate contrast, structures visible with some difficulty
    # - CNR < 1: Poor contrast, structures barely distinguishable from noise
    
    signal_diff = abs(water_mean - air_mean)                              # Contrast (signal difference)
    combined_noise = np.sqrt(water_std**2 + air_std**2)                  # Combined noise from both ROIs
    cnr = signal_diff / combined_noise                                    # Contrast-to-Noise Ratio
    
    print(f"  Signal difference: {signal_diff:.2f} grey values")
    print(f"  Combined noise: {combined_noise:.2f} grey values")
    print(f"  CNR: {cnr:.2f}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    print("\nCreating visualization...")
    os.makedirs(output_folder, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------------------------------------------
    # LEFT PANEL: Central slice with both ROIs displayed
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.imshow(stack[center_slice], cmap='gray')
    
    # Draw water ROI (cyan)
    water_circle = plt.Circle((center_x, center_y), roi_radius, fill=False, color='cyan', linewidth=2)
    ax.add_patch(water_circle)
    ax.text(center_x, center_y - roi_radius - 5, 'WATER', color='cyan', ha='center', fontsize=10, fontweight='bold')
    
    # Draw air ROI (red)
    air_circle = plt.Circle((air_x, air_y), roi_radius, fill=False, color='red', linewidth=2)
    ax.add_patch(air_circle)
    ax.text(air_x, air_y - roi_radius - 5, 'AIR', color='red', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_title(f'ROI Positions (Slice {center_slice})\nROI Size: {diameter_mm:.2f} × {diameter_mm:.2f} × {height_mm:.2f} mm³')
    ax.axis('off')
    
    # -------------------------------------------------------------------------
    # RIGHT PANEL: Histogram comparison showing grey value distributions
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    # Plot histograms for both water and air ROIs
    # The separation between these distributions indicates the contrast
    ax.hist(water_values, bins=50, alpha=0.6, color='cyan', label=f'Water (μ={water_mean:.1f}, σ={water_std:.1f})', edgecolor='black')
    ax.hist(air_values, bins=50, alpha=0.6, color='red', label=f'Air (μ={air_mean:.1f}, σ={air_std:.1f})', edgecolor='black')
    
    # Draw vertical lines at the means
    ax.axvline(water_mean, color='cyan', linestyle='--', linewidth=2, label=f'Water mean')
    ax.axvline(air_mean, color='red', linestyle='--', linewidth=2, label=f'Air mean')
    
    ax.set_xlabel('Grey Value')
    ax.set_ylabel('Frequency (number of pixels)')
    ax.set_title(f'Grey Value Distributions\nCNR = {cnr:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = os.path.join(output_folder, f"cnr_{timestamp}.png")
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {figure_path}")
    
    # =========================================================================
    # EXPORT RESULTS TO EXCEL
    # =========================================================================
    
    print("\nExporting results to Excel...")
    
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "CNR Results"
    
    # -------------------------------------------------------------------------
    # HEADER INFORMATION
    # -------------------------------------------------------------------------
    sheet['A1'] = 'MICRO-CT QUALITY CONTROL - CNR MEASUREMENT'
    sheet['A2'] = f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    sheet['A3'] = f'Input File: {os.path.basename(tif_path)}'
    sheet['A4'] = f'Pixel Size: {pixel_size_mm} mm'
    sheet['A5'] = f'ROI Size: {diameter_mm:.2f} × {diameter_mm:.2f} × {height_mm:.2f} mm³'
    sheet['A6'] = f'ROI Volume: {volume_mm3:.2f} mm³'
    
    # -------------------------------------------------------------------------
    # CNR RESULT
    # -------------------------------------------------------------------------
    sheet['A8'] = 'METRIC'
    sheet['B8'] = 'VALUE'
    sheet['C8'] = 'INTERPRETATION'
    
    sheet['A9'] = 'CNR'
    sheet['B9'] = f'{cnr:.2f}'
    if cnr > 5:
        interpretation = 'Excellent - structures easily distinguishable'
    elif cnr > 3:
        interpretation = 'Good - structures clearly visible'
    elif cnr > 1:
        interpretation = 'Moderate - structures visible with some difficulty'
    else:
        interpretation = 'Poor - structures barely distinguishable'
    sheet['C9'] = interpretation
    
    # -------------------------------------------------------------------------
    # DETAILED ROI STATISTICS
    # -------------------------------------------------------------------------
    sheet['A11'] = 'DETAILED ROI STATISTICS'
    sheet['A12'] = 'ROI'
    sheet['B12'] = 'Position X (px)'
    sheet['C12'] = 'Position Y (px)'
    sheet['D12'] = 'Mean Grey Value'
    sheet['E12'] = 'Std Dev'
    sheet['F12'] = 'Number of Pixels'
    
    sheet['A13'] = 'Water'
    sheet['B13'] = center_x
    sheet['C13'] = center_y
    sheet['D13'] = f'{water_mean:.2f}'
    sheet['E13'] = f'{water_std:.2f}'
    sheet['F13'] = len(water_values)
    
    sheet['A14'] = 'Air'
    sheet['B14'] = air_x
    sheet['C14'] = air_y
    sheet['D14'] = f'{air_mean:.2f}'
    sheet['E14'] = f'{air_std:.2f}'
    sheet['F14'] = len(air_values)
    
    # -------------------------------------------------------------------------
    # CALCULATION DETAILS
    # -------------------------------------------------------------------------
    sheet['A16'] = 'CALCULATION DETAILS'
    sheet['A17'] = 'Signal Difference'
    sheet['B17'] = f'{signal_diff:.2f}'
    sheet['C17'] = '|μ_water - μ_air|'
    
    sheet['A18'] = 'Combined Noise'
    sheet['B18'] = f'{combined_noise:.2f}'
    sheet['C18'] = '√(σ_water² + σ_air²)'
    
    sheet['A19'] = 'CNR'
    sheet['B19'] = f'{cnr:.2f}'
    sheet['C19'] = 'Signal Difference / Combined Noise'
    
    # -------------------------------------------------------------------------
    # SAVE EXCEL FILE
    # -------------------------------------------------------------------------
    excel_path = os.path.join(output_folder, f"cnr_{timestamp}.xlsx")
    workbook.save(excel_path)
    print(f"  Excel saved: {excel_path}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"CNR: {cnr:.2f} ({interpretation})")


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - MODIFY THESE PARAMETERS
    # =========================================================================
    
    # Input/Output paths
    tif_path = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift\export_volumes_Fantoma_agua_26_jan_15h0\Substack (70-110).tif"
    output_folder = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Pasta_Resultados\FDK\Excel_metricas"
    
    # Pixel size in mm (0.05 mm = 50 micrometers)
    # Critical for converting pixel measurements to physical dimensions
    pixel_size_mm = 0.05
    
    # ROI settings (in pixels)
    roi_radius = 25                  # Radius of circular ROIs - must be small enough to fit entirely in water and air regions
    slices_above_below = 20          # Number of slices above/below center - more slices = better statistics
    
    # Run analysis
    main(tif_path, output_folder, pixel_size_mm, roi_radius, slices_above_below)
