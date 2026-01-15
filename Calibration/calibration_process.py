"""
CT Calibration Process - Detector Center Alignment

This module performs systematic calibration to determine the misalignment 
between the rotation center and detector center in CT imaging systems.

Author: João Martim Reis
Date: December 2025
"""

import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, RadioButtons
from scipy.ndimage import label, binary_opening, find_objects
import re
import json
from datetime import datetime


def load_images(tiff_folder):
    print(f"--> Loading images from: {tiff_folder}")
    
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

    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape 
    print(f"Dimensions: {height} (H) x {width} (W) | {num_files} projections.")

    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)
    for i, f in enumerate(file_list):
        projections[:, :, i] = tiff.imread(os.path.join(tiff_folder, f))
    
    print("--> Images Loaded Successfully")
    return projections


def generate_collapsed_sinogram(projections):
    print("--> Creating collapsed sinogram (sum projection)...")
    sino_sum = np.sum(projections, axis=0)
    return sino_sum


def selecionar_roi_interativamente(sino_raw): 
    print("--> ROI selection and operation mode...")
    
    roi_background = [None]
    roi_object = [None]
    calculate_options = [True]
    current_mode = ['background']

    fig, ax = plt.subplots(figsize=(14, 8)) 
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("STEP 1: Draw Background ROI (green) | STEP 2: Draw Object ROI (orange) | STEP 3: Confirm", fontsize=11)
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
    rect_obj = None

    def on_select_callback(eclick, erelease):
        nonlocal rect_bg, rect_obj
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        ang_start, ang_end = sorted([x1, x2])
        det_start, det_end = sorted([y1, y2])
        roi_coords = (det_start, det_end, ang_start, ang_end)

        if current_mode[0] == 'background':
            roi_background[0] = roi_coords
            print(f"Background ROI: Det[{det_start}:{det_end}], Ang[{ang_start}:{ang_end}]")
            if rect_bg:
                rect_bg.remove()
            rect_bg = plt.Rectangle((ang_start, det_start), ang_end-ang_start, det_end-det_start,
                                     fill=False, edgecolor='green', linewidth=2, linestyle='--')
            ax.add_patch(rect_bg)
            current_mode[0] = 'object'
            ax.set_title("STEP 2: Draw Object ROI (orange) | STEP 3: Confirm", fontsize=11)
        else:
            roi_object[0] = roi_coords
            print(f"Object ROI: Det[{det_start}:{det_end}], Ang[{ang_start}:{ang_end}]")
            if rect_obj:
                rect_obj.remove()
            rect_obj = plt.Rectangle((ang_start, det_start), ang_end-ang_start, det_end-det_start,
                                      fill=False, edgecolor='orange', linewidth=2, linestyle='-')
            ax.add_patch(rect_obj)
            ax.set_title("DONE: Background (green) + Object (orange) | Click Confirm", fontsize=11)
        
        fig.canvas.draw()

    rect_selector = RectangleSelector(
        ax, on_select_callback, useblit=True, button=[1], 
        minspanx=5, minspany=5, spancoords='pixels', interactive=True,
        props=dict(facecolor='cyan', edgecolor='cyan', alpha=0.2, fill=True)
    )

    ax_radio = plt.axes([0.05, 0.05, 0.25, 0.15], facecolor='#e4e4e4')
    radio = RadioButtons(ax_radio, ('Calculate Shift', 'Already Centered (0 px)'))

    def change_mode(label):
        calculate_options[0] = (label == 'Calculate Shift')
        print(f"Mode changed to: {label}")
    
    radio.on_clicked(change_mode)

    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_button, 'Confirm') 

    def close_plot(event):
        plt.close(fig)

    btn_confirm.on_clicked(close_plot)
    plt.show(block=True)

    # Fallbacks
    if roi_background[0] is None:
        print("Warning: No Background ROI. Using top-left corner.")
        roi_background[0] = (0, 50, 0, 50)
    
    if roi_object[0] is None:
        print("Warning: No Object ROI. Using full image.")
        H, W = sino_raw.shape
        roi_object[0] = (0, H, 0, W)
        
    return roi_background[0], roi_object[0], calculate_options[0]


def calculate_shift_segmentation(sino_raw, roi_background, roi_object):
    print("--> Processing mask...")

    # Usar ROI de background para calcular estatísticas do fundo
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]
    
    # Estatísticas do fundo (para threshold)
    mean_fundo = np.mean(roi_crop)
    std_fundo = np.std(roi_crop)
    threshold = 2 * std_fundo 
    
    # Determinar se objeto é escuro ou claro comparando ROIs
    r_obj_start, r_obj_end, c_obj_start, c_obj_end = roi_object
    roi_object_crop = sino_raw[r_obj_start:r_obj_end, c_obj_start:c_obj_end]
    mean_object = np.mean(roi_object_crop)
    
    print(f"  Background mean: {mean_fundo:.2f} | Object mean: {mean_object:.2f}")
    
    # Threshold direcional: deteta apenas objeto (escuro OU claro)
    if mean_object < mean_fundo:
        print("  Object is DARKER than background")
        mascara_inicial = (mean_fundo - sino_raw) > threshold
    else:
        print("  Object is BRIGHTER than background")
        mascara_inicial = (sino_raw - mean_fundo) > threshold

    mascara_limpa = binary_opening(mascara_inicial, structure=np.ones((3,3)))
    labeled_array, num_features = label(mascara_limpa)
    print(f"Detected {num_features} connected components.")

    if num_features > 0:
        # Usar a ROI de objeto como "hint" do utilizador sobre onde está o sinograma
        r_start, r_end, c_start, c_end = roi_object
        roi_mask = np.zeros_like(labeled_array, dtype=bool)
        roi_mask[r_start:r_end, c_start:c_end] = True
        
        # Calcular sobreposição de cada componente com a ROI do utilizador
        best_label = None
        best_overlap = 0
        
        for i in range(1, num_features + 1):
            componente = (labeled_array == i)
            
            n_pixels = np.sum(componente)
            if n_pixels < 200:
                continue
            overlap = np.sum(componente & roi_mask)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = i
        
        if best_label is not None:
            mascara_final = (labeled_array == best_label)
            print(f"Selected component {best_label} with {best_overlap} pixels inside ROI.")
        else:
            sizes = np.bincount(labeled_array.ravel())[1:]
            if len(sizes) > 0:
                best = np.argmax(sizes) + 1
                mascara_final = (labeled_array == best)
                print(f"Warning: No component overlaps ROI. Using largest component.")
            else:
                mascara_final = mascara_limpa
    else:
        print("Warning: No object detected.")
        mascara_final = mascara_inicial

    # Encontrar coordenadas do objeto
    obj_slices = find_objects(mascara_final.astype(int))
    if obj_slices and obj_slices[0] is not None:
        det_min = obj_slices[0][0].start
        det_max = obj_slices[0][0].stop - 1
    else:
        coords = np.where(mascara_final)
        det_min = np.min(coords[0])
        det_max = np.max(coords[0])

    centro_objeto = (det_min + det_max) / 2.0
    centro_geo = sino_raw.shape[0] / 2.0 

    shift_val = centro_geo - centro_objeto

    print(f"--- Shift Report ---")
    print(f"Object Center: {centro_objeto:.2f} | Geo Center: {centro_geo:.2f}")
    print(f"Calculated shift: {shift_val:.2f} pixels")

    return shift_val, mascara_final, centro_objeto, det_min, det_max, mean_fundo


def show_results(sino_raw, mask, shift_val, centro_obj, d_min, d_max, modo_auto):
    print("--> Showing result...")
    centro_geo = sino_raw.shape[0] / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    titulo = "Automated Shift" if modo_auto else "Manual (Already Centered)"
    plt.suptitle(f"CT Calibration: {titulo}", fontsize=16)

    # Set x-axis ticks every 30 degrees
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

    # PLOT 2: MASK WITH LIMITS
    ax2 = axes[1]
    if modo_auto and mask is not None:
        ax2.imshow(mask, cmap='binary', aspect='auto', origin='upper')
        ax2.set_title(f"2. Mask (Min={d_min}, Max={d_max})")
        
        ax2.axhline(y=d_min, color='orange', linestyle=':', linewidth=2, label=f'Min={d_min}')
        ax2.axhline(y=d_max, color='orange', linestyle=':', linewidth=2, label=f'Max={d_max}')
        ax2.axhline(y=centro_obj, color='lime', linestyle='-', linewidth=2, label=f'Object={centro_obj:.1f}')
        ax2.axhline(y=centro_geo, color='red', linestyle='--', linewidth=2, label=f'Geo={centro_geo:.1f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Mask not used", ha='center', va='center')
        ax2.set_title("2. Mask Ignored")
    ax2.set_ylabel("Detector (px)")
    ax2.set_xlabel("θ (degree)")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    # PLOT 3: RESULT
    ax3 = axes[2]
    ax3.imshow(sino_raw, cmap='gray', aspect='auto', origin='upper')
    ax3.set_title(f"3. Calibration Result\nShift: {shift_val:.2f} px")
    ax3.axhline(y=centro_geo, color='red', linestyle='--', label='Geo Center')
    if modo_auto:
        ax3.axhline(y=centro_obj, color='lime', linestyle='-', label='Object Center')
    ax3.set_ylabel("Detector (px)")
    ax3.set_xlabel("θ (degree)")
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def main(parent_folder=None):
    """
    Main calibration workflow - processes multiple folders from parent directory.
    """
    print("\n" + "="*60)
    print("CT DETECTOR CENTER CALIBRATION")
    print("="*60 + "\n")

    # Get all subfolders
    subfolders = [f for f in os.listdir(parent_folder) 
                  if os.path.isdir(os.path.join(parent_folder, f))]
    
    print(f"\nFound {len(subfolders)} subfolders:")
    for folder in subfolders:
        print(f"  - {folder}")
    
    # Extract distances from folder names and sort
    folders_with_distances = []
    for folder in subfolders:
        try:
            distance = float(folder.replace(',', '.'))
            folders_with_distances.append((distance, folder))
        except:
            print(f"WARNING: Cannot extract distance from folder name '{folder}'. Skipping.")
    
    if len(folders_with_distances) == 0:
        print("ERROR: No valid distance folders found.")
        return
    
    folders_with_distances.sort(key=lambda x: x[0])
    
    print(f"\nProcessing {len(folders_with_distances)} folders in order:")
    for dist, folder in folders_with_distances:
        print(f"  {dist:.2f} cm → {folder}")
    
    # Process each folder
    distances = []
    shifts = []
    bg_means = []
    
    for i, (distance, folder) in enumerate(folders_with_distances):
        folder_path = os.path.join(parent_folder, folder)
        
        print(f"\n{'='*60}")
        print(f"FOLDER {i+1}/{len(folders_with_distances)}: {folder}")
        print(f"Distance: {distance:.2f} cm")
        print(f"{'='*60}")
        
        # 1. Load projections
        projections = load_images(folder_path)
        if projections is None:
            print(f"→ Failed to load projections")
            continue
        
        # 2. Create collapsed sinogram
        sino_raw = generate_collapsed_sinogram(projections)
        
        # 3. ROI selection
        roi_bg, roi_obj, calculate_mode = selecionar_roi_interativamente(sino_raw)
        
        if not calculate_mode:
            print("Skipping - 'Already Centered' mode selected.")
            continue
        
        # 4. Calculate shift
        shift, mask, obj_center, det_min, det_max, bg_mean = calculate_shift_segmentation(
            sino_raw, roi_bg, roi_obj
        )

        # 5. Show results for this folder
        show_results(sino_raw, mask, shift, obj_center, det_min, det_max, calculate_mode)

        distances.append(distance)
        shifts.append(shift)
        bg_means.append(float(bg_mean))
        print(f"→ Shift: {shift:.2f} pixels")
    
    # Calculate statistics
    avg_shift = np.mean(shifts)
    std_shift = np.std(shifts)
    avg_bg_mean = float(np.mean(bg_means)) if len(bg_means) > 0 else None

    # Save results
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    results = {
        'distances_cm': distances,
        'shifts_px': shifts,
        'average_shift_px': float(avg_shift),
        'std_shift_px': float(std_shift),
        'background_means': bg_means,
        'average_background_mean': avg_bg_mean,
        'n_positions': len(shifts),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        json_path = os.path.join(parent_folder, f"calibration_results_{timestamp_str}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ JSON saved: {json_path}")
    except Exception as e:
        print(f"Warning: failed to save JSON: {e}")
    
    # Save CSV for Excel/plotting
    try:
        csv_path = os.path.join(parent_folder, f"calibration_results_{timestamp_str}.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Distance_cm,Shift_px,Background_mean\n")
            for dist, shift, bg in zip(distances, shifts, bg_means):
                f.write(f"{dist:.2f},{shift:.4f},{bg:.4f}\n")
            f.write(f"\n# Summary\n")
            f.write(f"# Average Shift (px),{avg_shift:.4f}\n")
            f.write(f"# Std Deviation (px),{std_shift:.4f}\n")
            f.write(f"# Number of positions,{len(shifts)}\n")
        print(f"✓ CSV saved: {csv_path}")
    except Exception as e:
        print(f"Warning: failed to save CSV: {e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(distances, shifts, 'o-', markersize=10, linewidth=2, 
             color='dodgerblue', markeredgecolor='black', markeredgewidth=1.5)
    plt.axhline(avg_shift, linestyle='--', color='red', linewidth=2, 
                label=f'Average = {avg_shift:.2f} px')
    plt.axhline(avg_shift + std_shift, linestyle=':', color='gray', linewidth=1)
    plt.axhline(avg_shift - std_shift, linestyle=':', color='gray', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Distance from Center [cm]', fontsize=12, fontweight='bold')
    plt.ylabel('Detector Shift [pixels]', fontsize=12, fontweight='bold')
    plt.title('Multi-Position Calibration', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Final report
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print(f"Processed folders: {len(shifts)}")
    print(f"\nIndividual shifts:")
    for i, (dist, shift) in enumerate(zip(distances, shifts)):
        print(f"  {i+1}. Distance {dist:+.2f} cm → Shift {shift:+.2f} px")
    print(f"\nFinal Result:")
    print(f"  Average Shift: {avg_shift:.2f} pixels")
    print(f"  Std Deviation: {std_shift:.2f} pixels")


if __name__ == "__main__":
    main(parent_folder=r"C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\calibracao")
