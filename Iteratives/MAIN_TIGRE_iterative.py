import numpy as np
import os
import sys
import gc
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu
import matplotlib.pyplot as plt
import napari
import nibabel as nib
from datetime import datetime


from geometry_reconstruction_Voxel_size import setup_geometry
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_FDK_3D import load_images, generate_collapsed_sinogram, selecionar_roi_I0, get_I0_from_roi
from export_volumes import export_volume_to_nii, export_volume_HU
from iterative_parameters import get_algorithm_config

def print_volume_info(volume, geo=None):
    """
    Prints detailed information about the reconstructed volume.
    """
    print("RECONSTRUCTED VOLUME INFORMATION")
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}")
    print(f"\nGeometric information:")
    print(f"  - Voxel size: [{geo.dVoxel[0]:.3f}, {geo.dVoxel[1]:.3f}, {geo.dVoxel[2]:.3f}] mm")
    print(f"  - Number of voxels: {geo.nVoxel}")
    print(f"  - Physical dimensions: [{geo.sVoxel[0]:.3f}, {geo.sVoxel[1]:.3f}, {geo.sVoxel[2]:.3f}] mm")



def normalize_projections(projections_raw, I0_override=None):
    """
    Normalizes projections using -log(I/I0).
    This function receives cropped projections for better memory efficiency.
    """
    print("--> Normalizing cropped projections...")
    print(f"    Input shape: {projections_raw.shape}, Memory size: {projections_raw.nbytes / 1e6:.1f} MB")

    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 99))
        print(f"    WARNING: No I0_override provided. Using 99th percentile fallback: {I0:.2f}")
        print("    It is strongly recommended to provide I0_override from a calibrated ROI.")
    else:
        I0 = float(I0_override)

    if I0 <= 0:
        raise ValueError(f"I0 value is {I0:.4f} - must be positive. Check your ROI selection or raw data.")

    median_projection = float(np.percentile(projections_raw, 50))
    if I0 < median_projection:
        print(f"    WARNING: I0 ({I0:.2f}) is below the median projection value ({median_projection:.2f}).")
        print("    This likely means I0 is too low and will produce incorrect attenuation values.")

    projections_norm = projections_raw.astype(np.float32, copy=True)
    projections_norm /= (I0 + 1e-6)  # Avoid division by zero

    total_pixels = projections_norm.size
    clipped_above_count = np.count_nonzero(projections_norm > 1.2)
    clipped_below_count = np.count_nonzero(projections_norm < 0)
    clipped_above_pct = (clipped_above_count / total_pixels) * 100.0
    clipped_below_pct = (clipped_below_count / total_pixels) * 100.0
    print(f"    Clipping report (I/I0 ratio):")
    print(f"      - Pixels > 1.2: {clipped_above_pct:.4f}% ({clipped_above_count}/{total_pixels})")
    print(f"      - Pixels < 0:   {clipped_below_pct:.4f}% ({clipped_below_count}/{total_pixels})")

    np.clip(projections_norm, 1e-6, 1.2, out=projections_norm)  # Avoid log(0) and extreme values
    np.log(projections_norm, out=projections_norm)
    projections_norm *= -1.0  # Beer-Lambert law: -log(I/I0)
    projections_norm[projections_norm < 0] = 0  # Remove negative values (artifacts)
    print(f"    ✓ Normalization complete")

    return projections_norm


def downsample_block_mean_pad(proj, f):
    """
    This function reduces the spatial dimensions of projection images by computing
    the mean value of non-overlapping blocks of pixels. Each block has dimensions
    (f × f), and the resulting downsampled image has dimensions (H//f × W//f).

    Edge padding is applied when the original dimensions are not evenly divisible
    by the downsampling factor f. The padding uses edge replication mode, meaning
    the last row/column of pixels are duplicated to fill the required padding space,
    ensuring that all pixels can be grouped into complete f×f blocks without loss.

    The downsampling process:
    1. Pads the height and width dimensions to make them divisible by f
    2. Reshapes the padded array to isolate f×f blocks
    3. Computes the mean value across each block, replacing f×f pixels with 1
    4. Preserves all angle dimensions without modification
    """
    Height, Width, Angles = proj.shape  # Height, Width, Angles
    
    # Calculate padding needed to make dimensions divisible by f
    pad_h = (-Height) % f # verify if Height is divisible by f; if not, calculate required padding
    pad_w = (-Width) % f # verify if Width is divisible by f; if not, calculate required padding
    
    # Apply edge padding if necessary to ensure clean division
    if pad_h or pad_w:
        proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        proj_p = proj
    
    # Reshape to separate blocks and compute mean across block elements
    Hc = int(proj_p.shape[0])  # Padded height in pixels
    Wc = int(proj_p.shape[1])  # Padded width in pixels

    downsampling_h = Hc // f # Calculate new height after downsampling
    downsampling_w = Wc // f # Calculate new width after downsampling

    # Step 1: reshape into blocks of shape
    # Each element blocks[i, :, j, :, k] contains the f×f pixel block for output pixel (i, j) at angle k.
    blocks = proj_p.reshape(downsampling_h, f, downsampling_w, f, Angles) 

    # Step 2: compute the mean across the two block axes (f, f) -> axes 1 and 3
    # This averages each f×f block into a single pixel, producing shape (downsampling_h, downsampling_w, A).
    downsampled = blocks.mean(axis=(1, 3))

    # Now `downsampled` has shape (H//f, W//f, A) and contains the block-wise averaged projections.
    return downsampled




def main(tiff_folder, configurations, output_folder=None):
    """
    Main ITERATIVE reconstruction pipeline (CORRECTED VERSION)
    This version uses VOXEL SIZE as the primary input parameter instead of PIXEL SIZE.
    This makes it easier to compare with commercial micro-CT systems and published papers.

    PHASE 1: Data Loading & Preprocessing
    PHASE 2: Spatial Optimization (Memory Reduction)
    PHASE 3: Geometry & Reconstruction
    PHASE 4: Post-Processing & Export
    """
    
    
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    
    projections_raw = load_images(tiff_folder)  # Step 1.1: Load raw TIFF projection images
    sino_raw = generate_collapsed_sinogram(projections_raw)  # Step 1.2: Generate collapsed sinogram for I0 reference selection
    roi_background = selecionar_roi_I0(sino_raw)  # Step 1.3: User selects background ROI and calculates mean I0 value
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    del sino_raw # Clean up: free sinogram memory
    gc.collect()
    
    
    print("\nPHASE 2: SPATIAL OPTIMIZATION (MEMORY REDUCTION)")

    crop_params = select_crop_region(projections_raw[:, :, 0])  # Step 2.1: Interactively define crop region on first raw projection
    projections_cropped_raw = apply_crop_to_projections(projections_raw, crop_params)  # Step 2.2: Apply crop to all raw projections
    del projections_raw
    gc.collect()

    projections_cropped = normalize_projections(projections_cropped_raw, I0_override=mean_I0)  # Step 2.3: Normalize cropped projections
    del projections_cropped_raw
    gc.collect()
    
    f = configurations['downsample']  # Step 2.4: Optional downsampling (applied AFTER crop for maximum efficiency) 
    if f > 1:
        print(f"Downsampling by factor {f}x...")
        print(f"This will reduce maximum achievable resolution")
        projections_final = downsample_block_mean_pad(projections_cropped, f).astype(np.float32) #convert to float32 to save memory
        print(f"Final shape: {projections_final.shape}")
        del projections_cropped
        gc.collect()
    elif f < 1:
        print(f"Error: Downsampling factor must be >=1 or equal to 1")
        return None
    else:
        projections_final = projections_cropped
    
    

    print("PHASE 3: GEOMETRY SETUP & RECONSTRUCTION (VOXEL-SIZE-FIRST)")
    
    calibrated_shift_px = configurations['calibrated_shift_px']  # Step 3.1: Calculate detector shift (adjusted for downsampling factor)
    shift_val = calibrated_shift_px / f
    print(f" Detector shift calculated: {shift_val:.3f} pixels")
    
    effective_voxel_size = configurations['voxel_size'] * f  # Step 3.2: Adjust voxel size for downsampling
    print(f" Voxel size adjustment: {configurations['voxel_size']:.2f} μm × {f} = {effective_voxel_size:.2f} μm")
    
    geo, angles = setup_geometry(  # Step 3.3: Setup TIGRE geometry with ADJUSTED voxel size
        projections_final.shape, 
        effective_voxel_size,  # ← Already adjusted for downsampling
        configurations['DSD'], 
        configurations['DSO'], 
        shift_val, 
        configurations['total_angle'], 
        shift_sign=configurations['shift_sign'],
        downsample_factor=configurations['downsample'],  # ← For Nyquist limit calculation
        crop_params=crop_params  # ← Adjusts detector offset for cropped region
    )

    input_data = np.transpose(projections_final, (2, 0, 1)).copy()  # Step 3.4: Prepare data for TIGRE (transpose to TIGRE format: angles × height × width)
    del projections_final
    gc.collect()


    # Step 3.5: Load algorithm configuration
    algorithm_name = configurations['algorithm']
    algo_config = get_algorithm_config(algorithm_name)
    
    print(f"ITERATIVE ALGORITHM: {algorithm_name}")
    print(f"Category: {algo_config['category']}")
    print(f"Iterations: {algo_config['iterations']}")

    if algo_config['params']:
        print(f"Parameters: {algo_config['params']}")

    print(f"\nRunning {algorithm_name} algorithm...")
    algo_function = getattr(algs, algo_config['function_name'])

    algo_kwargs = {
        'niter': algo_config['iterations'],
        **algo_config['params']  # Unpack any additional parameters
    }
    
  
    volume = algo_function(input_data, geo, angles, **algo_kwargs)
    print_volume_info(volume, geo)

    
    print("PHASE 4: POST-PROCESSING & EXPORT")
    
    print(f"Opening Napari viewer...")
    
    # Voxel scale for napari
    voxel_scale = (geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]) if volume.ndim == 3 else (geo.dVoxel[1], geo.dVoxel[2])
    viewer = napari.Viewer()  # Step 4.1: NAPARI visualization
    viewer.add_image(volume, scale=voxel_scale, name="CT Volume")
    napari.run()
    

    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()  # Step 4.2: Optional NIfTI export
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print(f"Volume exported to NIfTI format")
    else:
        print(f"NIfTI export skipped")
    

    if export_choice == 'y' and nii_filepath is not None:  # Step 4.3: Optional Hounsfield Unit (HU) conversion
        hu_choice = input("\nDo you want to also export in Hounsfield Units (HU)? (y/n): ").strip().lower()
        if hu_choice == 'y':
            print("Please provide the gray scale values measured from the reconstruction:")
            water_val = float(input("Enter gray scale value for WATER: "))
            air_val = float(input("Enter gray scale value for AIR: "))
            export_volume_HU(nii_filepath, volume, water_val, air_val)
            print(f"Volume converted to Hounsfield Units and exported")
        else:
            print(f"HU conversion skipped")


    print("RECONSTRUCTION PIPELINE COMPLETED")

    return volume


if __name__ == "__main__":


    CONFIG = {
        'voxel_size': 25,  # μm - CHOOSE YOUR DESIRED RESOLUTION HERE
        'calibrated_shift_px': 0, #5.12 / 24.5 / 19.5 (bar pattern nivel 2) / 21.5 (Fantoma agua) 
        'total_angle': 2 * np.pi,
        'shift_sign': 1,  # Try -1 if reconstruction looks wrong
        'DSD': 463,
        'DSO': 244,
        #'DSD': 457,  # Distance Source to Detector (mm)
        #'DSO': 211,  # Distance Source to Object (mm)
        'downsample': 1, # NOTE: This affects maximum achievable resolution!
        'algorithm': 'MLEM',  # ← CHANGE THIS to select algorithm
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift',
        'filtered_volumes_folder': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\Filtered_volumes.Nift'
    }
    
    folder = r'D:\tentativa_Lara_tiff'  # Path to your TIFF projections folder
    
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))