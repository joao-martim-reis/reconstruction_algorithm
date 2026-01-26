import numpy as np
import tifffile
import matplotlib.pyplot as plt
from openpyxl import Workbook
from datetime import datetime
import os

# ===== CONFIGURATION =====
tif_path = "path/to/your/stack.tif"
output_folder = "path/to/output"
pixel_size_mm = 0.0  # set pixel size in mm if known

# ===== LOAD IMAGES =====
print("Loading image stack...")
stack = tifffile.imread(tif_path)
if stack.ndim == 2:
    stack = stack[np.newaxis, :, :]

num_slices, height, width = stack.shape
print(f"Stack loaded: {num_slices} slices, {height}x{width} pixels")

# ===== LINE SELECTION FOR MTF =====
print("\n" + "="*60)
print("LINE SELECTION FOR MTF")
print("="*60)
print("Click TWICE to define a line through the bars (line-pairs)")
print("First click defines start, second click defines end.")

center_slice = num_slices // 2

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(stack[center_slice], cmap='gray')
ax.set_title('Select LINE through line-pairs\n(2 clicks: start and end)')

line_points = []
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        line_points.append((int(event.xdata), int(event.ydata)))
        ax.plot(event.xdata, event.ydata, 'ro', markersize=10)
        
        if len(line_points) == 2:
            ax.plot([line_points[0][0], line_points[1][0]], 
                   [line_points[0][1], line_points[1][1]], 'r-', linewidth=2)
            plt.draw()
            print(f"Line defined: {line_points[0]} -> {line_points[1]}")
        else:
            plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(line_points) < 2:
    print("ERROR: Need to select 2 points. Using default horizontal line.")
    line_points = [(width//4, height//2), (3*width//4, height//2)]

# ===== EXTRACT LINE PROFILE =====
x0, y0 = line_points[0]
x1, y1 = line_points[1]

# Create line of points between the two selected points
num_points = int(np.sqrt((x1-x0)**2 + (y1-y0)**2))
x_coords = np.linspace(x0, x1, num_points).astype(int)
y_coords = np.linspace(y0, y1, num_points).astype(int)

# Extract values along the line
profile = stack[center_slice, y_coords, x_coords]

# ===== CALCULATE MTF (Eq 4.6) =====
print("\n" + "="*60)
print("MTF CALCULATION")
print("="*60)
print("Detecting peaks automatically from line profile...")

# Automatically detect peaks
from scipy.signal import find_peaks

peaks_high, _ = find_peaks(profile, distance=10)
peaks_low, _ = find_peaks(-profile, distance=10)

if len(peaks_high) > 0 and len(peaks_low) > 0:
    CT_high = np.mean(profile[peaks_high][:5])  # mean of top 5 highest peaks
    CT_low = np.mean(profile[peaks_low][:5])
    print(f"Automatically detected:")
    print(f"CT_high (bright bars): {CT_high:.2f}")
    print(f"CT_low (dark bars): {CT_low:.2f}")
else:
    CT_high = np.max(profile)
    CT_low = np.min(profile)
    print(f"Using extreme values:")
    print(f"CT_high: {CT_high:.2f}")
    print(f"CT_low: {CT_low:.2f}")

# Calculate MTF (Eq 4.6)
mtf_percent = ((CT_high - CT_low) / (CT_high + CT_low)) * 100

print(f"\nMTF: {mtf_percent:.2f}%")

# ===== CREATE GRAPHS =====
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Image with line
ax = axes[0]
ax.imshow(stack[center_slice], cmap='gray')
ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2, label='Line Profile')
ax.plot(x0, y0, 'ro', markersize=10)
ax.plot(x1, y1, 'ro', markersize=10)
ax.set_title('Line Profile Position')
ax.legend()
ax.axis('off')

# Line profile
ax = axes[1]
ax.plot(profile, 'b-', linewidth=1.5)
if len(peaks_high) > 0:
    ax.plot(peaks_high, profile[peaks_high], 'r^', markersize=8, label='Bright bars')
if len(peaks_low) > 0:
    ax.plot(peaks_low, profile[peaks_low], 'gv', markersize=8, label='Dark bars')
ax.axhline(CT_high, color='red', linestyle='--', alpha=0.5, label=f'CT_high = {CT_high:.1f}')
ax.axhline(CT_low, color='green', linestyle='--', alpha=0.5, label=f'CT_low = {CT_low:.1f}')
ax.set_xlabel('Position along line (pixels)')
ax.set_ylabel('Intensity')
ax.set_title(f'Line Profile - MTF = {mtf_percent:.2f}%')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
graph_path = os.path.join(output_folder, f'MTF_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
print(f"\nGraph saved: {graph_path}")
plt.close()

# ===== EXPORT TO EXCEL =====
wb = Workbook()
ws = wb.active
ws.title = "MTF"

ws['A1'] = "MICRO-CT SYSTEM PERFORMANCE - SPATIAL RESOLUTION (MTF)"
ws['A2'] = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
ws['A3'] = f"File: {os.path.basename(tif_path)}"
ws['A4'] = f"Pixel size: {pixel_size_mm if pixel_size_mm > 0 else 'N/A'} mm"

ws['A6'] = "RESULT"
ws['A7'] = "MTF (%)"
ws['B7'] = mtf_percent

ws['A9'] = "DETAILS"
ws['A10'] = "CT_high (bright bars)"
ws['B10'] = CT_high
ws['A11'] = "CT_low (dark bars)"
ws['B11'] = CT_low

ws['A13'] = "PARAMETERS"
ws['A14'] = "Line start (x, y)"
ws['B14'] = f"({x0}, {y0})"
ws['A15'] = "Line end (x, y)"
ws['B15'] = f"({x1}, {y1})"
ws['A16'] = "Profile length (pixels)"
ws['B16'] = num_points
ws['A17'] = "Analyzed slice"
ws['B17'] = center_slice

# Add profile data
ws['A19'] = "LINE PROFILE DATA"
ws['A20'] = "Pixel"
ws['B20'] = "Intensity"
for i, val in enumerate(profile[:100], start=21):  # first 100 points
    ws[f'A{i}'] = i-21
    ws[f'B{i}'] = float(val)

excel_path = os.path.join(output_folder, f'MTF_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
wb.save(excel_path)
print(f"Results saved: {excel_path}")

# ===== PRINT RESULTS =====
print("\n" + "="*50)
print("RESULTS - SPATIAL RESOLUTION (MTF)")
print("="*50)
print(f"MTF: {mtf_percent:.2f}%")
print(f"CT_high: {CT_high:.4f}")
print(f"CT_low: {CT_low:.4f}")
print("="*50)