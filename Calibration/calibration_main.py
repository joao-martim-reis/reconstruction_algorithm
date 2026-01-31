"""
CT Calibration Process - Detector Center Alignment

This module orchestrates the calibration workflow and delegates
IO, ROI selection, segmentation, visualization, and reporting
to separate modules for better organization.
"""

import os
import sys

# Ensure local `Calibration` modules import when running this script directly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from calibration_images_sinogram import load_images, generate_collapsed_sinogram
from calibration_roi import selecionar_roi_interativamente
from calibration_segmentation import calculate_shift_segmentation
from calibration_visualization import show_results
from calibration_reporting import compute_statistics, save_results, plot_results, print_final_report


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
    avg_shift, std_shift, avg_bg_mean = compute_statistics(shifts, bg_means)

    # Save results
    save_results(parent_folder, distances, shifts, bg_means, avg_shift, std_shift, avg_bg_mean)

    # Plot results
    plot_results(distances, shifts, avg_shift, std_shift)

    # Final report
    print_final_report(distances, shifts, avg_shift, std_shift)


if __name__ == "__main__":
    #main(parent_folder=r"C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\calibracao")
    main(parent_folder=r"C:\Users\joaomartimreis\Desktop\Joao_CT\test1")
