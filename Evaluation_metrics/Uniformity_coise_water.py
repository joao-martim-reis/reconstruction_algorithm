"""
================================================================================
UNIFORMITY, NOISE AND SNR METRICS - MICRO-CT QUALITY CONTROL
================================================================================

PHANTOM USED: Water Phantom (uniform material)

METRICS CALCULATED:
-------------------
1. UNIFORMITY: Measures spatial homogeneity across the image field
   - Formula: mean(|μ_central - μ_peripheral|) for the 4 peripheral ROIs
   - Lower values = better uniformity
   - Reference: Mambrini et al. (2022), Scientific Reports

2. NOISE: Measures random variations in pixel values (standard deviation)
   - Formula: σ = std(pixel values in ROI)
   - Average noise = mean of std across all 5 ROIs
   - Lower values = less noise

3. SNR (Signal-to-Noise Ratio): Ratio of signal strength to noise level
   - Formula: SNR = μ / σ (mean / standard deviation)
   - Higher values = better image quality
   - Reference: Standard QC protocols

WHY THESE METRICS?
------------------
- Uniformity detects beam hardening, ring artifacts, and reconstruction errors
- Noise affects detectability of low-contrast structures
- SNR determines the reliability of quantitative measurements

ROI PLACEMENT (as per Mambrini et al.):
---------------------------------------
- 1 central ROI + 4 peripheral ROIs (top, bottom, left, right)
- Cylindrical ROIs spanning multiple slices for better statistics
- Analysis on GREY SCALE values (no HU conversion)

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

def extract_roi_values(stack, center_x, center_y, radius, start_slice, num_slices):
    """
    Extract all pixel values within a cylindrical ROI.
    
    The ROI is circular in the XY plane and extends across multiple Z slices,
    forming a cylindrical volume. This provides better statistics than a single slice.
    
    Parameters:
    -----------
    stack : ndarray
        3D image stack (slices, height, width)
    center_x, center_y : int
        Center coordinates of the circular ROI in pixels
    radius : int
        Radius of the circular ROI in pixels
    start_slice : int
        First slice to include in the ROI
    num_slices : int
        Number of slices to include (height of cylinder)
    
    Returns:
    --------
    values : ndarray
        1D array of all pixel values within the cylindrical ROI
    
    Notes:
    ------
    - Uses circular mask: (x-center_x)² + (y-center_y)² ≤ radius²
    - Handles boundary cases: crops ROI if it extends beyond image edges
    - Returns flattened array for easy statistical analysis
    """
    
    # Define bounding box for the ROI (rectangular region containing the circle)
    # We need to ensure we don't go outside the image boundaries
    y_min = max(0, center_y - radius)
    y_max = min(stack.shape[1], center_y + radius + 1)
    x_min = max(0, center_x - radius)
    x_max = min(stack.shape[2], center_x + radius + 1)
    
    # Create meshgrid for circular mask calculation
    # This creates a grid of (x,y) coordinates within the bounding box
    y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
    
    # Create circular mask using distance formula: (x-cx)² + (y-cy)² ≤ r²
    # Pixels inside the circle = True, outside = False
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    
    # Extract all pixel values within the cylindrical ROI
    # We iterate through each slice and collect all pixels where mask is True
    values = []
    for i in range(start_slice, start_slice + num_slices):
        # Extract the ROI region from this slice
        roi_region = stack[i, y_min:y_max, x_min:x_max]
        # Apply circular mask and collect valid pixels
        values.extend(roi_region[mask])
    
    return np.array(values)


def calculate_roi_size_mm(radius_px, height_slices, pixel_mm):
    """
    Calculate physical ROI dimensions in mm.
    
    Converts pixel-based measurements to physical dimensions (mm³).
    
    Parameters:
    -----------
    radius_px : int
        ROI radius in pixels
    height_slices : int
        Number of slices (height of ROI cylinder)
    pixel_mm : float
        Pixel size in mm (assumes isotropic voxels)
    
    Returns:
    --------
    diameter_mm : float
        Diameter in mm
    height_mm : float
        Height in mm
    volume_mm3 : float
        Volume in mm³ (π × r² × h)
    """
    
    diameter_mm = 2 * radius_px * pixel_mm
    height_mm = height_slices * pixel_mm
    # Volume of cylinder: π × r² × h
    volume_mm3 = np.pi * (radius_px * pixel_mm)**2 * height_mm
    
    return diameter_mm, height_mm, volume_mm3


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main(tif_path, output_folder, pixel_size_mm, roi_radius, distance_from_center, slices_above_below):
    """
    Main function for uniformity, noise and SNR analysis.
    
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
    distance_from_center : int
        Distance of peripheral ROIs from center in pixels
    slices_above_below : int
        Number of slices above/below center for volumetric ROI
    """
    
    # =========================================================================
    # LOAD IMAGE STACK
    # =========================================================================
    
    print("Loading image stack...")
    stack = tifffile.imread(tif_path)
    num_slices, height, width = stack.shape
    
    # Calculate image center - this assumes the phantom is centered in the FOV
    center_slice = num_slices // 2
    center_y, center_x = height // 2, width // 2
    print(f"Stack loaded: {num_slices} slices, {height}×{width} pixels")
    print(f"Image center: ({center_x}, {center_y}), slice {center_slice}")

    # =========================================================================
    # DEFINE ROI POSITIONS
    # =========================================================================
    
    # Define ROI positions: 1 central + 4 peripheral (at 12, 3, 6, 9 o'clock positions)
    # This arrangement follows the standard QC protocol from Mambrini et al. (2022)
    roi_positions = {
        'central': (center_x, center_y),                                    # Center of the phantom
        'top': (center_x, center_y - distance_from_center),                # 12 o'clock position
        'bottom': (center_x, center_y + distance_from_center),             # 6 o'clock position
        'left': (center_x - distance_from_center, center_y),               # 9 o'clock position
        'right': (center_x + distance_from_center, center_y)               # 3 o'clock position
    }

    # =========================================================================
    # CALCULATE SLICE RANGE FOR VOLUMETRIC ROI
    # =========================================================================
    
    # We use a volumetric (3D) ROI by extracting data from multiple slices
    # This improves statistical reliability compared to single-slice analysis
    start_slice = max(0, center_slice - slices_above_below)                # Ensure we don't go below slice 0
    end_slice = min(num_slices, center_slice + slices_above_below + 1)     # Ensure we don't exceed total slices
    actual_height = end_slice - start_slice                                 # Actual number of slices used

    # =========================================================================
    # CALCULATE AND DISPLAY ROI PHYSICAL SIZE
    # =========================================================================
    
    diameter_mm, height_mm, volume_mm3 = calculate_roi_size_mm(roi_radius, actual_height, pixel_size_mm)
    print(f"\nROI Configuration:")
    print(f"  Size: {diameter_mm:.2f} × {diameter_mm:.2f} × {height_mm:.2f} mm³ (D × D × H)")
    print(f"  Volume: {volume_mm3:.2f} mm³")
    print(f"  Slices: {start_slice} to {end_slice-1} (center: {center_slice}, total: {actual_height})")

    # =========================================================================
    # EXTRACT ROI VALUES AND CALCULATE STATISTICS
    # =========================================================================
    
    # For each ROI position, extract all pixel values within the cylindrical volume
    # and calculate mean (signal) and standard deviation (noise)
    roi_data = {}
    print("\nExtracting ROI values...")
    for name, (x, y) in roi_positions.items():
        # Extract all pixel values within the circular ROI across all specified slices
        values = extract_roi_values(stack, x, y, roi_radius, start_slice, actual_height)
        
        # Store statistics for this ROI
        roi_data[name] = {
            'mean': np.mean(values),      # Average grey value (signal)
            'std': np.std(values),        # Standard deviation (noise)
            'values': values              # Keep raw values for potential further analysis
        }
        print(f"  {name}: mean={roi_data[name]['mean']:.2f}, std={roi_data[name]['std']:.2f}, n={len(values)} pixels")

    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    
    print("\nCalculating metrics...")
    
    # 1. UNIFORMITY: Measures spatial homogeneity across the image
    #    Formula: mean(|μ_central - μ_peripheral|) for the 4 peripheral ROIs
    #    Interpretation: Lower uniformity value = better spatial homogeneity
    #    
    #    A perfectly uniform phantom should have the same grey value everywhere.
    #    Deviations indicate artifacts (beam hardening, ring artifacts, etc.)
    uniformity_diffs = []
    for peripheral in ['top', 'bottom', 'left', 'right']:
        diff = abs(roi_data['central']['mean'] - roi_data[peripheral]['mean'])
        uniformity_diffs.append(diff)
    uniformity = np.mean(uniformity_diffs)
    print(f"  Uniformity: {uniformity:.4f} grey values")

    # 2. NOISE: Measures random variations in pixel values
    #    Formula: Average standard deviation across all 5 ROIs
    #    Interpretation: Lower noise = better image quality
    #    
    #    In a uniform region, all pixel-to-pixel variations are due to noise.
    #    We average across all ROIs to get a robust estimate of system noise.
    noise = np.mean([roi_data[name]['std'] for name in roi_positions.keys()])
    print(f"  Noise (avg std): {noise:.4f} grey values")

    # 3. SNR: Signal-to-Noise Ratio from central ROI
    #    Formula: SNR = μ / σ (mean divided by standard deviation)
    #    Interpretation: Higher SNR = better image quality
    #    
    #    SNR indicates how many times the signal exceeds the noise level.
    #    Higher SNR means more reliable quantitative measurements.
    water_value_mean = roi_data['central']['mean']
    water_value_std = roi_data['central']['std']
    snr = water_value_mean / water_value_std
    print(f"  SNR: {snr:.2f}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    print("\nCreating visualization...")
    os.makedirs(output_folder, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------------------------------------------------------------------------
    # LEFT PANEL: Central slice with ROI positions overlaid
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.imshow(stack[center_slice], cmap='gray')
    
    # Draw each ROI with a different color for visibility
    colors = {'central': 'cyan', 'top': 'red', 'bottom': 'green', 'left': 'yellow', 'right': 'magenta'}
    for name, (x, y) in roi_positions.items():
        # Draw circular ROI boundary
        circle = plt.Circle((x, y), roi_radius, fill=False, color=colors[name], linewidth=2)
        ax.add_patch(circle)
        # Add label above the ROI
        ax.text(x, y - roi_radius - 5, name, color=colors[name], ha='center', fontsize=9, fontweight='bold')
    
    ax.set_title(f'ROI Positions (Slice {center_slice})\nROI Size: {diameter_mm:.2f} × {diameter_mm:.2f} × {height_mm:.2f} mm³')
    ax.axis('off')

    # -------------------------------------------------------------------------
    # RIGHT PANEL: Line profiles through center (visual check for uniformity)
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    # Extract horizontal and vertical line profiles through the phantom center
    # These help visualize spatial variations and artifacts
    profile_horizontal = stack[center_slice, center_y, :]
    profile_vertical = stack[center_slice, :, center_x]
    
    ax.plot(profile_horizontal, label='Horizontal', alpha=0.7, linewidth=1.5)
    ax.plot(profile_vertical, label='Vertical', alpha=0.7, linewidth=1.5)
    
    # Show expected water value as reference
    ax.axhline(water_value_mean, color='cyan', linestyle='--', linewidth=2, 
               label=f'Water mean = {water_value_mean:.1f}')
    
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Grey Value')
    ax.set_title('Line Profiles Through Center\n(Visual check for uniformity and artifacts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = os.path.join(output_folder, f"uniformity_noise_snr_{timestamp}.png")
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {figure_path}")

    # =========================================================================
    # EXPORT RESULTS TO EXCEL
    # =========================================================================
    
    print("\nExporting results to Excel...")
    
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Results"

    # -------------------------------------------------------------------------
    # HEADER INFORMATION
    # -------------------------------------------------------------------------
    sheet['A1'] = 'MICRO-CT QUALITY CONTROL - WATER PHANTOM'
    sheet['A2'] = f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    sheet['A3'] = f'Input File: {os.path.basename(tif_path)}'
    sheet['A4'] = f'Pixel Size: {pixel_size_mm} mm'
    sheet['A5'] = f'ROI Size: {diameter_mm:.2f} × {diameter_mm:.2f} × {height_mm:.2f} mm³'
    sheet['A6'] = f'ROI Volume: {volume_mm3:.2f} mm³'

    # -------------------------------------------------------------------------
    # METRIC RESULTS
    # -------------------------------------------------------------------------
    sheet['A8'] = 'METRIC'
    sheet['B8'] = 'VALUE'
    sheet['C8'] = 'UNITS'
    sheet['D8'] = 'INTERPRETATION'

    sheet['A9'] = 'Uniformity'
    sheet['B9'] = f'{uniformity:.4f}'
    sheet['C9'] = 'grey values'
    sheet['D9'] = 'Lower is better (less spatial variation)'

    sheet['A10'] = 'Noise'
    sheet['B10'] = f'{noise:.4f}'
    sheet['C10'] = 'grey values (std)'
    sheet['D10'] = 'Lower is better (less random variation)'

    sheet['A11'] = 'SNR'
    sheet['B11'] = f'{snr:.2f}'
    sheet['C11'] = 'dimensionless'
    sheet['D11'] = 'Higher is better (signal exceeds noise)'

    # -------------------------------------------------------------------------
    # DETAILED ROI STATISTICS
    # -------------------------------------------------------------------------
    sheet['A13'] = 'DETAILED ROI STATISTICS'
    sheet['A14'] = 'ROI Name'
    sheet['B14'] = 'Position X (px)'
    sheet['C14'] = 'Position Y (px)'
    sheet['D14'] = 'Mean Grey Value'
    sheet['E14'] = 'Std Dev'
    sheet['F14'] = 'Number of Pixels'

    row = 15
    for name, (x, y) in roi_positions.items():
        sheet[f'A{row}'] = name
        sheet[f'B{row}'] = x
        sheet[f'C{row}'] = y
        sheet[f'D{row}'] = f'{roi_data[name]["mean"]:.2f}'
        sheet[f'E{row}'] = f'{roi_data[name]["std"]:.2f}'
        sheet[f'F{row}'] = len(roi_data[name]['values'])
        row += 1

    # -------------------------------------------------------------------------
    # SAVE EXCEL FILE
    # -------------------------------------------------------------------------
    excel_path = os.path.join(output_folder, f"uniformity_noise_snr_{timestamp}.xlsx")
    workbook.save(excel_path)
    print(f"  Excel saved: {excel_path}")

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Uniformity: {uniformity:.4f} grey values")
    print(f"Noise: {noise:.4f} grey values")
    print(f"SNR: {snr:.2f}")


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - MODIFY THESE PARAMETERS
    # =========================================================================
    
    # Input/Output paths
    tif_path = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift\export_volumes_Fantoma_agua_26_jan_15h0\Substack (70-110).tif"
    output_folder = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Pasta_Resultados\FDK\Excel_metricas"
    
    # Pixel size in mm (0.05 mm = 50 micrometers)
    # This is critical for converting pixel measurements to physical dimensions
    pixel_size_mm = 0.05
    
    # ROI settings (in pixels)
    roi_radius = 40                  # Radius of circular ROIs - larger ROI = more pixels = better statistics
    distance_from_center = 100       # Distance of peripheral ROIs from image center - should be far enough to detect spatial variations
    slices_above_below = 10          # Number of slices above/below center for volumetric ROI - more slices = better statistics
    
    # Run analysis
    main(tif_path, output_folder, pixel_size_mm, roi_radius, distance_from_center, slices_above_below)


