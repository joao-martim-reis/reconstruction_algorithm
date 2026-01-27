import numpy as np
import tifffile
import matplotlib.pyplot as plt
from openpyxl import Workbook
from datetime import datetime
import os

# ===== CONFIGURATION =====
tif_path = "C:\\Users\\joaomartimreis\\Desktop\\Joao_CT\\Volumes_reconstrucao\\reconstructed_volumes_Nift\\export_volumes_Fantoma_agua_26_jan_15h0\\Substack (70-110).tif"  # Path to TIF file
output_folder = "C:\\Users\\joaomartimreis\\Desktop\\Joao_CT\\Pasta_Resultados\\FDK\\Excel_metricas"      # Folder to save Excel and graphs
pixel_size_mm = 0.0  # Set 0 to use pixels, or enter size (e.g., 0.05 for 50µm)

# ROI settings (in pixels)
roi_radius = 40  # ROI circular radius in pixels
roi_height = 25   # number of slices for 3D analysis
distance_from_center = 100  # distance of peripheral ROIs from center

# ===== LOAD IMAGES =====
print("Loading image stack...")
stack = tifffile.imread(tif_path)
if stack.ndim == 2:
    stack = stack[np.newaxis, :, :]  # Convert to 3D if needed

num_slices, height, width = stack.shape
center_slice = num_slices // 2
center_y, center_x = height // 2, width // 2

print(f"Stack loaded: {num_slices} slices, {height}x{width} pixels")

# ===== DEFINE ROIs =====
# Central and peripheral ROIs
roi_positions = {
    'central': (center_x, center_y),
    'top': (center_x, center_y - distance_from_center),
    'bottom': (center_x, center_y + distance_from_center),
    'left': (center_x - distance_from_center, center_y),
    'right': (center_x + distance_from_center, center_y)
}

# ===== EXTRACT ROI VALUES =====
def extract_roi_values(stack, x, y, radius, start_slice, num_slices):
    """Extract values from a circular ROI across multiple slices"""
    values = []
    y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x_coords**2 + y_coords**2 <= radius**2
    
    for i in range(start_slice, min(start_slice + num_slices, stack.shape[0])):
        y1, y2 = max(0, y-radius), min(stack.shape[1], y+radius+1)
        x1, x2 = max(0, x-radius), min(stack.shape[2], x+radius+1)
        roi_slice = stack[i, y1:y2, x1:x2]
        values.extend(roi_slice[mask[:roi_slice.shape[0], :roi_slice.shape[1]]].flatten())
    
    return np.array(values)

# Extract values - ROI volume centered at center_slice
start_slice = max(0, center_slice - roi_height // 2)
end_slice = min(num_slices, center_slice + roi_height // 2 + 1)
actual_height = end_slice - start_slice

roi_data = {}

for name, (x, y) in roi_positions.items():
    values = extract_roi_values(stack, x, y, roi_radius, start_slice, actual_height)
    roi_data[name] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'values': values
    }

print(f"ROI volume: slices {start_slice} to {end_slice-1} (center: {center_slice})")

# ===== CALCULATE METRICS =====
# Uniformity (Eq 4.4)
uniformity_diffs = []
for peripheral in ['top', 'bottom', 'left', 'right']:
    diff = abs(roi_data['central']['mean'] - roi_data[peripheral]['mean'])
    uniformity_diffs.append(diff)
uniformity = np.mean(uniformity_diffs)

# Noise
noise = np.mean([roi_data[name]['std'] for name in roi_positions.keys()])

# Water evaluation (Eq 4.3)
water_value = roi_data['central']['mean']

# ===== CREATE GRAPH =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Show central slice with ROIs
ax = axes[0]
ax.imshow(stack[center_slice], cmap='gray')
for name, (x, y) in roi_positions.items():
    circle = plt.Circle((x, y), roi_radius, fill=False, color='red', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, name, color='yellow', ha='center', fontsize=8)
ax.set_title('ROI Positions')
ax.axis('off')

# Line profile
ax = axes[1]
profile = stack[center_slice, center_y, :]
ax.plot(profile)
ax.set_xlabel('Pixel X')
ax.set_ylabel('Intensity')
ax.set_title('Line Profile (horizontal)')
ax.grid(True)

plt.tight_layout()
graph_path = os.path.join(output_folder, f'uniformity_noise_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
print(f"Graph saved: {graph_path}")
plt.close()

# ===== EXPORT TO EXCEL =====
wb = Workbook()
ws = wb.active
ws.title = "Uniformity_Noise"

# Header
ws['A1'] = "MICRO-CT SYSTEM PERFORMANCE - UNIFORMITY & NOISE"
ws['A2'] = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
ws['A3'] = f"File: {os.path.basename(tif_path)}"
ws['A4'] = f"Pixel size: {pixel_size_mm if pixel_size_mm > 0 else 'N/A'} mm"

# Results
ws['A6'] = "RESULTS"
ws['A7'] = "Uniformity"
ws['B7'] = uniformity
ws['A8'] = "Noise (avg std)"
ws['B8'] = noise
ws['A9'] = "Water Value (central)"
ws['B9'] = water_value

# ROI details
ws['A11'] = "ROI"
ws['B11'] = "Mean"
ws['C11'] = "Std Dev"
row = 12
for name in ['central', 'top', 'bottom', 'left', 'right']:
    ws[f'A{row}'] = name
    ws[f'B{row}'] = roi_data[name]['mean']
    ws[f'C{row}'] = roi_data[name]['std']
    row += 1

# Parameters
ws['A' + str(row+1)] = "PARAMETERS"
ws['A' + str(row+2)] = "ROI radius (pixels)"
ws['B' + str(row+2)] = roi_radius
ws['A' + str(row+3)] = "ROI height (slices)"
ws['B' + str(row+3)] = roi_height
ws['A' + str(row+4)] = "Distance from center (pixels)"
ws['B' + str(row+4)] = distance_from_center

excel_path = os.path.join(output_folder, f'uniformity_noise_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
wb.save(excel_path)
print(f"\nResults saved: {excel_path}")

# ===== PRINT RESULTS =====
print("\n" + "="*50)
print("RESULTS - UNIFORMITY & NOISE")
print("="*50)
print(f"Uniformity: {uniformity:.4f}")
print(f"Noise (avg std): {noise:.4f}")
print(f"Water Value: {water_value:.4f}")
print("="*50)