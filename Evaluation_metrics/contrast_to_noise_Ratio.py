import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from openpyxl import Workbook
from datetime import datetime
import os

# ===== CONFIGURATION =====
tif_path = "C:\\Users\\joaomartimreis\\Desktop\\Substack (70-130).tif"  # Path to TIF file
output_folder = "C:\\Users\\joaomartimreis\\Desktop\\Joao_CT\\Pasta_Resultados\\FDK\\Excel_metricas"
pixel_size_mm = 0.0

# ROI settings
roi_radius = 25
roi_height = 25

# ===== LOAD IMAGES =====
print("Loading image stack...")
stack = tifffile.imread(tif_path)
if stack.ndim == 2:
    stack = stack[np.newaxis, :, :]

num_slices, height, width = stack.shape
center_slice = num_slices // 2
center_y, center_x = height // 2, width // 2

print(f"Stack loaded: {num_slices} slices, {height}x{width} pixels")

# ===== SELECT WATER ROI (CENTER) =====
def extract_roi_circular(stack, x, y, radius, start_slice, num_slices):
    values = []
    y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x_coords**2 + y_coords**2 <= radius**2
    
    for i in range(start_slice, min(start_slice + num_slices, stack.shape[0])):
        y1, y2 = max(0, y-radius), min(stack.shape[1], y+radius+1)
        x1, x2 = max(0, x-radius), min(stack.shape[2], x+radius+1)
        roi_slice = stack[i, y1:y2, x1:x2]
        values.extend(roi_slice[mask[:roi_slice.shape[0], :roi_slice.shape[1]]].flatten())
    
    return np.array(values)

start_slice = max(0, center_slice - roi_height // 2)
end_slice = min(num_slices, center_slice + roi_height // 2 + 1)
actual_height = end_slice - start_slice

# Water ROI (center) - volume centered at center_slice
water_values = extract_roi_circular(stack, center_x, center_y, roi_radius, start_slice, actual_height)
water_mean = np.mean(water_values)
water_std = np.std(water_values)

print(f"ROI volume: slices {start_slice} to {end_slice-1} (center: {center_slice})")

# ===== SELECT AIR ROI (MANUAL) =====
print("\n" + "="*60)
print("AIR ROI SELECTION")
print("="*60)
print("View the image and choose the position of the air ROI.")
print("The ROI should be outside the phantom (dark area).")

# Show image to help
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(stack[center_slice], cmap='gray')
circle_water = plt.Circle((center_x, center_y), roi_radius, fill=False, color='cyan', linewidth=2, label='Water ROI')
ax.add_patch(circle_water)
ax.set_title('Choose AIR ROI position\n(click near edge, outside phantom)')
ax.legend()

# Capture user click
air_position = []
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        air_position.append((int(event.xdata), int(event.ydata)))
        circle_air = plt.Circle((event.xdata, event.ydata), roi_radius, fill=False, color='red', linewidth=2)
        ax.add_patch(circle_air)
        ax.text(event.xdata, event.ydata, 'AIR', color='red', ha='center', fontsize=10)
        plt.draw()
        print(f"Air ROI selected at: x={int(event.xdata)}, y={int(event.ydata)}")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(air_position) == 0:
    print("WARNING: No position selected. Using default position (top-left corner)")
    air_x, air_y = roi_radius + 10, roi_radius + 10
else:
    air_x, air_y = air_position[0]

# Extract air ROI values - volume centered at center_slice
air_values = extract_roi_circular(stack, air_x, air_y, roi_radius, start_slice, actual_height)
air_mean = np.mean(air_values)
air_std = np.std(air_values)

# ===== CALCULATE CNR (Eq 4.5) =====
cnr = abs(water_mean - air_mean) / np.sqrt(water_std**2 + air_std**2)

# ===== CREATE GRAPH =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Show ROIs
ax = axes[0]
ax.imshow(stack[center_slice], cmap='gray')
circle_water = plt.Circle((center_x, center_y), roi_radius, fill=False, color='cyan', linewidth=2)
circle_air = plt.Circle((air_x, air_y), roi_radius, fill=False, color='red', linewidth=2)
ax.add_patch(circle_water)
ax.add_patch(circle_air)
ax.text(center_x, center_y, 'WATER', color='cyan', ha='center', fontsize=10)
ax.text(air_x, air_y, 'AIR', color='red', ha='center', fontsize=10)
ax.set_title('ROIs for CNR')
ax.axis('off')

# Histograms
ax = axes[1]
ax.hist(water_values, bins=50, alpha=0.6, label='Water', color='cyan')
ax.hist(air_values, bins=50, alpha=0.6, label='Air', color='red')
ax.axvline(water_mean, color='cyan', linestyle='--', linewidth=2)
ax.axvline(air_mean, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Intensity')
ax.set_ylabel('Frequency')
ax.set_title(f'Distributions - CNR = {cnr:.2f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
graph_path = os.path.join(output_folder, f'CNR_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
print(f"\nGraph saved: {graph_path}")
plt.close()

# ===== EXPORT TO EXCEL =====
wb = Workbook()
ws = wb.active
ws.title = "CNR"

ws['A1'] = "MICRO-CT SYSTEM PERFORMANCE - CONTRAST-TO-NOISE RATIO"
ws['A2'] = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
ws['A3'] = f"File: {os.path.basename(tif_path)}"
ws['A4'] = f"Pixel size: {pixel_size_mm if pixel_size_mm > 0 else 'N/A'} mm"

ws['A6'] = "RESULT"
ws['A7'] = "CNR (Contrast-to-Noise Ratio)"
ws['B7'] = cnr

ws['A9'] = "DETAILS"
ws['A10'] = "ROI"
ws['B10'] = "Mean"
ws['C10'] = "Std Dev"
ws['A11'] = "Water"
ws['B11'] = water_mean
ws['C11'] = water_std
ws['A12'] = "Air"
ws['B12'] = air_mean
ws['C12'] = air_std

ws['A14'] = "PARAMETERS"
ws['A15'] = "ROI radius (pixels)"
ws['B15'] = roi_radius
ws['A16'] = "ROI height (slices)"
ws['B16'] = roi_height
ws['A17'] = "Water position (x, y)"
ws['B17'] = f"({center_x}, {center_y})"
ws['A18'] = "Air position (x, y)"
ws['B18'] = f"({air_x}, {air_y})"
ws['A19'] = "Center slice"
ws['B19'] = center_slice
ws['A20'] = "Slice range"
ws['B20'] = f"{start_slice} to {end_slice-1}"

excel_path = os.path.join(output_folder, f'CNR_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
wb.save(excel_path)
print(f"Results saved: {excel_path}")

# ===== PRINT RESULTS =====
print("\n" + "="*50)
print("RESULTS - CNR")
print("="*50)
print(f"CNR: {cnr:.4f}")
print(f"Water Mean: {water_mean:.4f} ± {water_std:.4f}")
print(f"Air Mean: {air_mean:.4f} ± {air_std:.4f}")
print("="*50)