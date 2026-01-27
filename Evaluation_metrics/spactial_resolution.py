"""
================================================================================
SPATIAL RESOLUTION (MTF) - MICRO-CT QUALITY CONTROL
================================================================================

PHANTOM USED: Bar Pattern Phantom (line-pair phantom)

METRIC CALCULATED:
------------------
MTF (Modulation Transfer Function): Measures the spatial resolution capability
of the imaging system - how well it can distinguish between closely spaced 
features (line pairs).

FORMULA (from bar pattern):
---------------------------
    MTF (%) = [(CT_high - CT_low) / (CT_high + CT_low)] × 100

Where:
- CT_high: Mean grey value of bright bars (high attenuation)
- CT_low: Mean grey value of dark bars (low attenuation)

INTERPRETATION:
---------------
- MTF = 100%: Perfect contrast preservation (ideal system)
- MTF = 0%: No contrast (features indistinguishable)
- Higher MTF = better spatial resolution

The MTF value obtained from a specific line-pair frequency indicates how well
the system can resolve structures at that spatial frequency. Typically, the
spatial resolution limit is defined where MTF drops to 10% or 50%.

WHY THIS METRIC?
----------------
- Spatial resolution determines the smallest detectable structures
- Important for detecting small lesions or fine anatomical details
- Affected by detector pixel size, focal spot size, reconstruction filter
- Reference: Mambrini et al. (2022), Scientific Reports; AAPM Task Group 233

MEASUREMENT PROCEDURE:
----------------------
1. Select a slice through the bar pattern phantom
2. Draw a line profile perpendicular to the line pairs
3. Detect peaks (bright bars) and valleys (dark bars) automatically
4. Calculate MTF from the contrast between peaks and valleys

================================================================================
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
from openpyxl import Workbook
from datetime import datetime
import os


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main(tif_path, output_folder, pixel_size_mm, bar_width_um, peak_distance):
    """
    Main function for MTF (spatial resolution) analysis.
    
    Parameters:
    -----------
    tif_path : str
        Path to input TIFF stack
    output_folder : str
        Path to output folder for results
    pixel_size_mm : float
        Pixel size in mm (for physical dimension conversion)
    bar_width_um : int
        Width of bars being analyzed in micrometers
    peak_distance : int
        Minimum distance between peaks in pixels (for peak detection)
    """
    
    # =========================================================================
    # LOAD IMAGE STACK
    # =========================================================================
    
    print("Loading image stack...")
    stack = tifffile.imread(tif_path)
    
    # Handle both single image and stack formats
    if stack.ndim == 2:
        # Single image - add a "slice" dimension for consistency
        stack = stack[np.newaxis, :, :]
    
    num_slices, height, width = stack.shape
    print(f"Stack loaded: {num_slices} slices, {height}×{width} pixels")
    
    # =========================================================================
    # INTERACTIVE SLICE SELECTION
    # =========================================================================
    
    print("\n--- Select the analysis slice ---")
    print("Instructions:")
    print("  1. Use the slider to navigate through slices")
    print("  2. Find a slice that shows clear bar pattern (good contrast)")
    print("  3. Close the window once you've found the right slice")
    
    # Variables to store the selected slice index
    selected_slice = [0]  # Use list so it can be modified in nested function
    
    # Create interactive figure with slider
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)  # Make room for slider
    
    # Display the first slice initially
    im = ax.imshow(stack[0], cmap='gray')
    ax.set_title(f'Slice 0 / {num_slices-1}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Create slider widget for slice selection
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices-1, valinit=0, valstep=1)
    
    # Slider update function - called when user moves the slider
    def update_slice(val):
        """Update the displayed slice when slider moves."""
        slice_idx = int(slider.val)
        selected_slice[0] = slice_idx
        im.set_data(stack[slice_idx])
        ax.set_title(f'Slice {slice_idx} / {num_slices-1}', fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update_slice)
    
    plt.show()
    
    # Get the final selected slice
    analysis_slice = selected_slice[0]
    print(f"\nAnalyzing slice {analysis_slice}")
    img = stack[analysis_slice]
    
    # =========================================================================
    # INTERACTIVE LINE PROFILE SELECTION
    # =========================================================================
    
    print("\n--- Draw line profile perpendicular to bars ---")
    print("Instructions:")
    print("  1. Click TWO points to define a line across the bar pattern")
    print("  2. The line should be PERPENDICULAR to the bars")
    print("  3. Make it long enough to cross multiple bars")
    print("  4. Close the window after clicking both points")
    
    # Variables to store the two clicked points
    points = []
    
    # Create figure for line selection
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title('Click 2 points to define line profile\n(perpendicular to bars)', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Event handler for mouse clicks
    def onclick(event):
        """
        Captures user's clicks to define a line profile.
        Two clicks are needed: start point and end point.
        """
        # Only process clicks inside the axes
        if event.inaxes != ax:
            return
        
        # Store the clicked point (x, y coordinates)
        points.append((event.xdata, event.ydata))
        
        # Plot the clicked point as a red circle
        ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
        
        # If this is the second point, draw the line connecting them
        if len(points) == 2:
            x_coords = [points[0][0], points[1][0]]
            y_coords = [points[0][1], points[1][1]]
            ax.plot(x_coords, y_coords, 'r-', linewidth=2)
            ax.set_title('Line profile defined! Close window to continue.', 
                        fontsize=14, fontweight='bold', color='green')
        
        # Redraw the figure
        fig.canvas.draw()
        print(f"  Point {len(points)}: ({event.xdata:.1f}, {event.ydata:.1f})")
    
    # Connect click event to handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    # =========================================================================
    # VALIDATE LINE SELECTION
    # =========================================================================
    
    # Check if user clicked two points
    if len(points) != 2:
        print("\nERROR: You must click exactly 2 points to define the line. Please run the script again.")
        return
    
    # Extract start and end points of the line
    x0, y0 = points[0]
    x1, y1 = points[1]
    print(f"\nLine profile: ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f})")
    
    # =========================================================================
    # EXTRACT LINE PROFILE VALUES
    # =========================================================================
    
    print("\n--- Extracting line profile ---")
    
    # Calculate line length in pixels
    line_length = int(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
    
    # Create interpolated points along the line
    # We sample at regular intervals to get a smooth profile
    x_coords = np.linspace(x0, x1, line_length)
    y_coords = np.linspace(y0, y1, line_length)
    
    # Extract grey values along the line using bilinear interpolation
    # This gives us the "profile" - how grey values change along the line
    from scipy.ndimage import map_coordinates
    profile = map_coordinates(img, [y_coords, x_coords], order=1)
    
    # Convert pixel positions to physical distance (mm)
    distance_mm = np.arange(line_length) * pixel_size_mm
    
    print(f"  Profile length: {line_length} pixels ({line_length * pixel_size_mm:.2f} mm)")
    print(f"  Profile values: min={profile.min():.1f}, max={profile.max():.1f}, mean={profile.mean():.1f}")
    
    # =========================================================================
    # DETECT PEAKS (BRIGHT BARS) AND VALLEYS (DARK BARS)
    # =========================================================================
    
    print("\n--- Detecting peaks and valleys ---")
    
    # Find peaks (local maxima) - these correspond to the bright bars
    # The 'distance' parameter ensures we don't detect noise as peaks
    peaks, _ = find_peaks(profile, distance=peak_distance)
    
    # Find valleys (local minima) by inverting the profile
    # Valleys correspond to the dark bars (gaps between bright bars)
    valleys, _ = find_peaks(-profile, distance=peak_distance)
    
    print(f"  Detected {len(peaks)} peaks (bright bars)")
    print(f"  Detected {len(valleys)} valleys (dark bars)")
    
    # Validate detection - we need at least one peak and one valley
    if len(peaks) == 0 or len(valleys) == 0:
        print("\nERROR: Could not detect peaks and valleys!")
        print("Suggestions:")
        print("  - Try a different slice with clearer bars")
        print("  - Draw a longer line crossing more bars")
        print("  - Adjust peak_distance parameter in the configuration")
        return
    
    # =========================================================================
    # CALCULATE MTF AND LINE-PAIR RESOLUTION
    # =========================================================================
    
    print("\n--- Calculating MTF and Spatial Resolution ---")
    
    # Calculate mean grey values of peaks (bright bars) and valleys (dark bars)
    ct_high = np.mean(profile[peaks])
    ct_low = np.mean(profile[valleys])
    
    # MTF Formula: [(high - low) / (high + low)] × 100
    # 
    # This formula measures contrast preservation:
    # - Numerator: contrast (difference between bright and dark)
    # - Denominator: average signal level
    # - Result: percentage of contrast preserved by the imaging system
    # 
    # Interpretation:
    # - MTF = 100%: Perfect system, full contrast preserved
    # - MTF = 50%: Half the contrast is preserved
    # - MTF = 10%: Typical resolution limit threshold
    # - MTF = 0%: No contrast, bars indistinguishable
    
    mtf = ((ct_high - ct_low) / (ct_high + ct_low)) * 100
    
    # Line-Pair Resolution Calculation
    # 
    # Spatial resolution in micro-CT is typically expressed as line-pairs per mm (lp/mm)
    # This represents the highest spatial frequency that the system can resolve.
    # 
    # Formula: lp/mm = 1000 / (2 × bar_width_µm)
    # 
    # Where:
    # - bar_width_µm: width of one bar in micrometers
    # - Factor of 2: one line-pair = one bright bar + one dark bar (one cycle)
    # - Factor of 1000: converts µm to mm
    # 
    # Example: 25 µm bars → 1000/(2×25) = 20 lp/mm
    # 
    # The MTF at this frequency tells you how well the system preserves contrast
    # at that spatial frequency. If MTF < 10%, the bars are at the resolution limit.
    
    line_pair_frequency = 1000 / (2 * bar_width_um)  # lp/mm
    
    print(f"  Bar width analyzed: {bar_width_um} µm")
    print(f"  Line-pair frequency: {line_pair_frequency:.2f} lp/mm")
    print(f"  Mean peak value (bright bars): {ct_high:.2f} grey values")
    print(f"  Mean valley value (dark bars): {ct_low:.2f} grey values")
    print(f"  Contrast: {ct_high - ct_low:.2f} grey values")
    print(f"  MTF at {line_pair_frequency:.2f} lp/mm: {mtf:.2f}%")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    print("\nCreating visualization...")
    os.makedirs(output_folder, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # -------------------------------------------------------------------------
    # TOP PANEL: Image with line profile overlaid
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.imshow(img, cmap='gray')
    
    # Draw the line profile (red line)
    ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2, label='Profile line')
    
    # Mark start and end points
    ax.plot(x0, y0, 'go', markersize=10, label='Start')
    ax.plot(x1, y1, 'mo', markersize=10, label='End')
    
    ax.set_title(f'Bar Pattern (Slice {analysis_slice}) with Line Profile', fontsize=12, fontweight='bold')
    ax.legend()
    ax.axis('off')
    
    # -------------------------------------------------------------------------
    # BOTTOM PANEL: Line profile with detected peaks and valleys
    # -------------------------------------------------------------------------
    ax = axes[1]
    
    # Plot the profile (grey values along the line)
    ax.plot(distance_mm, profile, 'b-', linewidth=1.5, label='Profile', alpha=0.7)
    
    # Mark detected peaks (bright bars) with red triangles
    ax.plot(distance_mm[peaks], profile[peaks], 'rv', markersize=10, 
            label=f'Peaks (bright bars, n={len(peaks)})')
    
    # Mark detected valleys (dark bars) with green triangles
    ax.plot(distance_mm[valleys], profile[valleys], 'g^', markersize=10, 
            label=f'Valleys (dark bars, n={len(valleys)})')
    
    # Draw horizontal lines showing mean peak and valley levels
    ax.axhline(ct_high, color='red', linestyle='--', linewidth=2, 
               label=f'Mean peak = {ct_high:.1f}', alpha=0.7)
    ax.axhline(ct_low, color='green', linestyle='--', linewidth=2, 
               label=f'Mean valley = {ct_low:.1f}', alpha=0.7)
    
    ax.set_xlabel('Distance (mm)', fontsize=11)
    ax.set_ylabel('Grey Value', fontsize=11)
    ax.set_title(f'Line Profile Analysis\nBar Width: {bar_width_um} µm | Frequency: {line_pair_frequency:.2f} lp/mm | MTF = {mtf:.2f}%', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = os.path.join(output_folder, f"mtf_{timestamp}.png")
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {figure_path}")
    
    # =========================================================================
    # EXPORT RESULTS TO EXCEL
    # =========================================================================
    
    print("\nExporting results to Excel...")
    
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "MTF Results"
    
    # -------------------------------------------------------------------------
    # HEADER INFORMATION
    # -------------------------------------------------------------------------
    sheet['A1'] = 'MICRO-CT QUALITY CONTROL - SPATIAL RESOLUTION (MTF)'
    sheet['A2'] = f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    sheet['A3'] = f'Input File: {os.path.basename(tif_path)}'
    sheet['A4'] = f'Slice Analyzed: {analysis_slice}'
    sheet['A5'] = f'Pixel Size: {pixel_size_mm} mm'
    sheet['A6'] = f'Profile Length: {line_length} pixels ({line_length * pixel_size_mm:.2f} mm)'
    sheet['A7'] = f'Bar Width: {bar_width_um} µm'
    sheet['A8'] = f'Line-Pair Frequency: {line_pair_frequency:.2f} lp/mm'
    
    # -------------------------------------------------------------------------
    # MTF AND RESOLUTION RESULTS
    # -------------------------------------------------------------------------
    sheet['A10'] = 'METRIC'
    sheet['B10'] = 'VALUE'
    sheet['C10'] = 'INTERPRETATION'
    
    sheet['A11'] = 'Line-Pair Frequency'
    sheet['B11'] = f'{line_pair_frequency:.2f} lp/mm'
    sheet['C11'] = f'Analyzing {bar_width_um} µm bars'
    
    sheet['A12'] = 'MTF'
    sheet['B12'] = f'{mtf:.2f}%'
    if mtf > 80:
        interpretation = 'Excellent - very high contrast preservation at this frequency'
    elif mtf > 50:
        interpretation = 'Good - adequate contrast preservation'
    elif mtf > 10:
        interpretation = 'Moderate - approaching resolution limit'
    else:
        interpretation = 'Poor - at or beyond resolution limit'
    sheet['C12'] = interpretation
    
    # -------------------------------------------------------------------------
    # RESOLUTION INTERPRETATION GUIDE
    # -------------------------------------------------------------------------
    sheet['A14'] = 'RESOLUTION INTERPRETATION GUIDE'
    sheet['A15'] = 'MTF > 50%: Excellent resolution at this frequency, bars clearly distinguishable'
    sheet['A16'] = 'MTF 10-50%: Acceptable resolution, bars visible but with reduced contrast'
    sheet['A17'] = 'MTF < 10%: Resolution limit reached, bars barely distinguishable from noise'
    sheet['A18'] = ''
    sheet['A19'] = 'Your QRM phantom has multiple bar widths you can test:'
    sheet['A20'] = '  Fine: 5, 10, 15, 20, 25, 30 µm (100, 50, 33.3, 25, 20, 16.7 lp/mm)'
    sheet['A21'] = '  Coarse: 5, 10, 25, 50, 100, 150 µm (100, 50, 20, 10, 5, 3.3 lp/mm)'
    sheet['A22'] = 'Test different bar widths to find where MTF drops below 10% = your resolution limit'
    
    # -------------------------------------------------------------------------
    # DETAILED MEASUREMENTS
    # -------------------------------------------------------------------------
    sheet['A24'] = 'DETAILED MEASUREMENTS'
    sheet['A25'] = 'Parameter'
    sheet['B25'] = 'Value'
    sheet['C25'] = 'Description'
    
    sheet['A26'] = 'Bar Width'
    sheet['B26'] = f'{bar_width_um} µm'
    sheet['C26'] = 'Width of analyzed bars'
    
    sheet['A27'] = 'Line-Pair Frequency'
    sheet['B27'] = f'{line_pair_frequency:.2f} lp/mm'
    sheet['C27'] = '1000 / (2 × bar_width_µm)'
    
    sheet['A28'] = 'Mean Peak Value (CT_high)'
    sheet['B28'] = f'{ct_high:.2f}'
    sheet['C28'] = 'Average grey value of bright bars'
    
    sheet['A29'] = 'Mean Valley Value (CT_low)'
    sheet['B29'] = f'{ct_low:.2f}'
    sheet['C29'] = 'Average grey value of dark bars'
    
    sheet['A30'] = 'Contrast'
    sheet['B30'] = f'{ct_high - ct_low:.2f}'
    sheet['C30'] = 'CT_high - CT_low'
    
    sheet['A31'] = 'Number of Peaks'
    sheet['B31'] = len(peaks)
    sheet['C31'] = 'Number of bright bars detected'
    
    sheet['A32'] = 'Number of Valleys'
    sheet['B32'] = len(valleys)
    sheet['C32'] = 'Number of dark bars detected'
    
    # -------------------------------------------------------------------------
    # CALCULATION FORMULA
    # -------------------------------------------------------------------------
    sheet['A34'] = 'MTF CALCULATION'
    sheet['A35'] = 'Formula: MTF = [(CT_high - CT_low) / (CT_high + CT_low)] × 100'
    sheet['A36'] = f'MTF = [({ct_high:.2f} - {ct_low:.2f}) / ({ct_high:.2f} + {ct_low:.2f})] × 100'
    sheet['A37'] = f'MTF = {mtf:.2f}%'
    
    # -------------------------------------------------------------------------
    # SAVE EXCEL FILE
    # -------------------------------------------------------------------------
    excel_path = os.path.join(output_folder, f"mtf_{timestamp}.xlsx")
    workbook.save(excel_path)
    print(f"  Excel saved: {excel_path}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Bar Width: {bar_width_um} µm")
    print(f"Line-Pair Frequency: {line_pair_frequency:.2f} lp/mm")
    print(f"MTF: {mtf:.2f}% ({interpretation})")
    print(f"\nTip: Test different bar widths to map your full MTF curve and find your resolution limit!")


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - MODIFY THESE PARAMETERS
    # =========================================================================
    
    # Input/Output paths
    tif_path = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift\export_volumes_Bar_pattern_v3_26_jan_18h23\318.tif"
    output_folder = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Pasta_Resultados\FDK\Excel_metricas"
    
    # Pixel size in mm (0.05 mm = 50 micrometers)
    # Used to convert pixel coordinates to physical distance (mm) in plots
    pixel_size_mm = 0.05
    
    # Bar pattern phantom specifications
    # IMPORTANT: Set this to the line width (in micrometers) of the bars you're analyzing
    # Your QRM phantom has these options:
    #   - Fine resolution: 5, 10, 15, 20, 25, 30 µm (Blocks B, C)
    #   - Coarse resolution: 5, 10, 25, 50, 100, 150 µm (Blocks A, E)
    # 
    # If you can't see certain bars, it indicates your resolution limit!
    # Start with larger bar widths (e.g., 50 µm) and work down to find your limit.
    bar_width_um = 25  # Line width in micrometers (µm)
    
    # Peak detection parameter
    # Minimum distance between adjacent peaks in pixels - depends on bar spacing
    # If bars are closely spaced, reduce this value; if widely spaced, increase it
    peak_distance = 10
    
    # Run analysis
    main(tif_path, output_folder, pixel_size_mm, bar_width_um, peak_distance)
