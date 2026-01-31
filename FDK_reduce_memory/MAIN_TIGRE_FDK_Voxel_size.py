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
from matplotlib.widgets import Slider, Button
from datetime import datetime


from geometry_reconstruction_Voxel_size import setup_geometry
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_FDK_3D import load_images, generate_collapsed_sinogram, selecionar_roi_I0, get_I0_from_roi
from export_volumes import export_volume_to_nii, export_volume_HU
from napari_filters import interactive_filter_viewer


def print_volume_info(volume, geo=None):
    """
    Prints detailed information about the reconstructed volume.
    """
    # It is common for reconstructed CT volumes to contain mall negative values in background regions. These negatives can
    # arise from numerical effects of the reconstruction filter (e.g. Ram-Lak ringing), slight mis-centering/shift errors, noise, or
    # algorithmic artifacts. 
    # Small negative values are usually not a sign of catastrophic failure; however, if you require strictly
    # non-negative data its possible to clip.(`volume[volume<0]=0`),but clipping may hide underlying issues that are better fixed
    # (I0 calibration, center of rotation, filter choice, etc.).

    print("\n" + "="*60)
    print("RECONSTRUCTED VOLUME INFORMATION")
    print("="*60)
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}")
    print("="*60 + "\n") 
    print(f"\nGeometric information:")
    print(f"  - Voxel size: [{geo.dVoxel[0]:.3f}, {geo.dVoxel[1]:.3f}, {geo.dVoxel[2]:.3f}] mm")
    print(f"  - Number of voxels: {geo.nVoxel}")
    print(f"  - Physical dimensions: [{geo.sVoxel[0]:.3f}, {geo.sVoxel[1]:.3f}, {geo.sVoxel[2]:.3f}] mm")
    print("="*60 + "\n")



def normalize_projections(projections_raw, I0_override=None):
    """
    Normalizes projections using -log(I/I0).
    
    This function receives ALREADY CROPPED projections, drastically reducing the number of mathematical operations.
    """

    print("--> Normalizing cropped projections...")
    print(f"    Input shape: {projections_raw.shape}, Memory size: {projections_raw.nbytes / 1e6:.1f} MB")

    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 1))
    else:
        I0 = float(I0_override)

    # Convert to float32 for calculations (lighter than float64)
    projections_raw = projections_raw.astype(np.float32)
    ratio = projections_raw / (I0 + 1e-6)# Avoid division by zero
    ratio = np.clip(ratio, 1e-6, 1.2)# np clip to avoid log(0) and extreme values
    
    # Beer-Lambert law: -log(I/I0)
    projections_norm = -np.log(ratio)
    
    # Remove negative values (artifacts)
    projections_norm[projections_norm < 0] = 0
    
    print(f"    ✓ Normalization complete")

    # Debug prints (commented): inspect dtype and range after normalization
    # print('DEBUG projections_raw:', projections_raw.dtype, projections_raw.min(), projections_raw.max())
    # print('DEBUG projections_norm:', projections_norm.dtype, projections_norm.min(), projections_norm.max())
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
    # This averages each f×f block into a single pixel, producing shape
    # (downsampling_h, downsampling_w, A).
    downsampled = blocks.mean(axis=(1, 3))

    # Now `downsampled` has shape (H//f, W//f, A) and contains the block-wise averaged projections.
    
    return downsampled




def main(tiff_folder, configurations, output_folder=None):
    """
    Main FDK reconstruction pipeline 
    This version uses VOXEL SIZE as the primary input parameter instead of PIXEL SIZE.
    This makes it easier to compare with commercial micro-CT systems and published papers.

    PHASE 1: Data Loading & Preprocessing
        - Load raw projections from TIFF files
        - Generate collapsed sinogram for I0 calibration
        - Select background ROI and calculate I0 reference value
    
    PHASE 2: Spatial Optimization (Memory Reduction)
        - Define crop region on first raw projection
        - Apply crop to all raw projections (reduces memory footprint)
        - Normalize cropped projections using Beer-Lambert law
        - Optional: Downsample for faster reconstruction
    
    PHASE 3: Geometry & Reconstruction
        - Setup TIGRE geometry with VOXEL SIZE as primary input
        - Calculate required pixel size from desired voxel size
        - Execute FDK algorithm with filtered backprojection
    
    PHASE 4: Post-Processing & Export
        - Visualize volume in Napari
        - Optional: Export to NIfTI format
        - Optional: Convert to Hounsfield Units (HU)
    
    """
    
    
    # PHASE 1: DATA LOADING & PREPROCESSING
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    
    # Step 1.1: Load raw TIFF projection images
    projections_raw = load_images(tiff_folder)

    # Step 1.2: Generate collapsed sinogram for I0 reference selection
    sino_raw = generate_collapsed_sinogram(projections_raw)
    
    # Step 1.3: User selects background ROI and calculates mean I0 value
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    
    # Clean up: free sinogram memory
    del sino_raw
    gc.collect()
    
    
    # PHASE 2: SPATIAL OPTIMIZATION (MEMORY REDUCTION)
    print("PHASE 2: SPATIAL OPTIMIZATION")
    
    # Step 2.1: Define crop region on first raw projection
    first_proj_raw = projections_raw[:, :, 0]
    crop_params = select_crop_region(first_proj_raw)
    
    # Step 2.2: Apply crop to ALL raw projections
    projections_cropped_raw = apply_crop_to_projections(projections_raw, crop_params)
    del projections_raw
    gc.collect()
    
    # Step 2.3: Normalize cropped projections using Beer-Lambert law: -log(I/I0)
    projections_norm = normalize_projections(projections_cropped_raw, I0_override=mean_I0)
    del projections_cropped_raw
    gc.collect()
    
    # Step 2.4: Optional downsampling (applied AFTER crop for maximum efficiency)
    # NOTE: Downsampling affects achievable resolution 
    f = configurations['downsample'] 
    
    if f > 1:
        print(f"Downsampling by factor {f}x...")
        print(f"      NOTE: This will REDUCE maximum achievable resolution")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32) #convert to float32 to save memory
        print(f"      Final shape: {projections_final.shape}")
        del projections_norm
        gc.collect()
    elif f < 1:
        print(f"Error: Downsampling factor must be >=1 or equal to 1")
        return None
    else:
        projections_final = projections_norm
    
    

    print("PHASE 3: GEOMETRY SETUP & RECONSTRUCTION (VOXEL-SIZE-FIRST)")
    
    # Step 3.1: Calculate detector shift (adjusted for downsampling factor)
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / f
    print(f" Detector shift calculated: {shift_val:.3f} pixels")
    
    # Step 3.2: Adjust voxel size for downsampling
    # CRITICAL: Downsampling increases effective voxel size
    # Example: voxel_size = 20 μm, downsample = 4 → effective_voxel_size = 80 μm
    effective_voxel_size = configurations['voxel_size'] * f
    print(f" Voxel size adjustment: {configurations['voxel_size']:.2f} μm × {f} = {effective_voxel_size:.2f} μm")
    
    # Step 3.3: Setup TIGRE geometry with ADJUSTED voxel size
    # The voxel_size entering setup_geometry is already adjusted for downsampling
    geo, angles = setup_geometry(
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
    print(f"      Detector size: {geo.nDetector}")
    print(f"      Voxel size: {geo.dVoxel}")

    # Step 3.3: Prepare data for TIGRE (transpose to TIGRE format: angles × height × width)
    print(f"Preparing data for TIGRE...")
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    del projections_final
    gc.collect()
    # Debug prints inspect input to TIGRE
    # print('DEBUG input_data dtype/min/max:', input_data.dtype, input_data.min(), input_data.max())


    # Step 3.4: EXECUTE FDK reconstruction algorithm
    print(f"Running FDK algorithm with '{configurations['filter_type']}' filter...")
    volume = algs.fdk(input_data, geo, angles, filter=configurations['filter_type'])
    print_volume_info(volume, geo)

    # Debug prints: inspect reconstructed volume range
    # print('DEBUG volume dtype/min/max:', volume.dtype, volume.min(), volume.max())
    
    
    # PHASE 4: POST-PROCESSING & EXPORT
    print("PHASE 4: POST-PROCESSING & EXPORT")
    
    # Step 4.1: NAPARI visualization
    print(f"Opening Napari viewer...")
    
    # Voxel scale for napari
    voxel_scale = (geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]) if volume.ndim == 3 else (geo.dVoxel[1], geo.dVoxel[2])
    
    # Interactive filtering (commented out for now)
    # filter_choice = input("\nDo you want to use INTERACTIVE filtering? (y/n): ").strip().lower()
    # if filter_choice == 'y':
    #     filtered_folder = configurations.get('filtered_volumes_folder')
    #     viewer, final_filter_params = interactive_filter_viewer(volume, name="CT Volume", scale=voxel_scale, nii_filepath=None, filtered_output_folder=filtered_folder)
    #     napari.run()
    #     from napari_filters import print_applied_filters_summary
    #     print_applied_filters_summary(final_filter_params)
    # else:
    
    # View reconstructed 3D volume (FDK always produces 3D output)
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=voxel_scale, name="CT Volume")
    napari.run()
    

    # Step 4.2: Optional NIfTI export
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print(f"Volume exported to NIfTI format")
    else:
        print(f"NIfTI export skipped")
    

    # Step 4.3: Optional Hounsfield Unit (HU) conversion
    if export_choice == 'y' and nii_filepath is not None:
        hu_choice = input("\nDo you want to also export in Hounsfield Units (HU)? (y/n): ").strip().lower()
        if hu_choice == 'y':
            print(f"HU CONVERSION SETUP")
            print("      Please provide the gray scale values measured from the reconstruction:")
            water_val = float(input("      Enter gray scale value for WATER: "))
            air_val = float(input("      Enter gray scale value for AIR: "))
            
            export_volume_HU(nii_filepath, volume, water_val, air_val)
            print(f"      Volume converted to Hounsfield Units and exported")
        else:
            print(f"[4.3] HU conversion skipped")


    print("\n" + "="*70)
    print("RECONSTRUCTION PIPELINE COMPLETED")
    print("="*70 + "\n")
    
    return volume


if __name__ == "__main__":

    CONFIG = {

        'voxel_size': 26,  # μm - CHOOSE YOUR DESIRED RESOLUTION HERE

        #'calibrated_shift_px': 5.12,
        'calibrated_shift_px': 32,
        #'calibrated_shift_px': (5.12),

        # Geometry parameters 
        'total_angle': 2 * np.pi,
        'shift_sign': +1,
        'DSD': 457,  # Distance Source to Detector (mm)
        'DSO': 211,  # Distance Source to Object (mm)

        # Downsampling reduces resolution but speeds up reconstruction
        # NOTE: This affects maximum achievable resolution!
        'downsample': 1,
        
        # Reconstruction filter
        'filter_type': 'ram_lak',  # Options: 'ram_lak', 'shepp_logan', 'cosine', 'hamming', 'hann'
        
        # Output folders
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift',
        'filtered_volumes_folder': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\Filtered_volumes.Nift'
    }
    

    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\marta_caixa_SiPM'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Mouse_PC'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Laranja'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Haste_perfeita\Try_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\PEIXE\PEIXE'


    # Análise de resultados
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\test' #bar pattern
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Fantoma_agua' #agua
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+DOD_246\PMMA'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+DOD_246\PMMA+haste'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Fantoma_espuma'

    
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
