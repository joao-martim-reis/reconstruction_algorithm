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


# VOXEL-SIZE-FIRST APPROACH: Import from voxel-size geometry module
from geometry_reconstruction_voxel_size import setup_geometry

from data_processing_3D_2 import (
    load_images, 
    generate_collapsed_sinogram,
    selecionar_roi_I0,
    get_I0_from_roi,
)

from export_volumes import export_volume_to_nii, export_volume_HU

# Import HU conversion function
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from HU_conversion import HU_conversion


def print_volume_info(volume, geo=None):
    """
    Prints detailed information about the reconstructed volume.
    """
    # It is common for reconstructed CT volumes to contain small negative values in background regions. These negatives can
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
    print("--> Normalizing projections...")

    # I0 estimation (Air) — can be overridden with mean_fundo from ROI
    if I0_override is None:
        # fallback: estimate I0 from dark percentile of the projections
        I0 = float(np.percentile(projections_raw, 1)) # if there is no override, use 1st percentile as I0
    else:
        I0 = float(I0_override)


    projections_raw = projections_raw.astype(np.float32) # Ensure float for log calculation
    ratio = projections_raw / (I0 + 1e-6) # Avoid division by zero
    ratio = np.clip(ratio, 1e-6, 1.2)  # Prevent log(0) and extreme values
    
    projections_norm = -np.log(ratio)
    projections_norm[projections_norm < 0] = 0
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
    Main reconstruction pipeline.
    
    *** VOXEL-SIZE-FIRST VERSION ***
    This version uses VOXEL SIZE as the primary input parameter instead of PIXEL SIZE.
    This makes it easier to compare with commercial micro-CT systems and published papers.
    
    Key difference: CONFIG uses 'voxel_size' (in μm) instead of 'pixel_size' (in mm)
    """
    # 1. Load
    projections = load_images(tiff_folder)
    
    # Safety check
    if projections is None:
        raise ValueError(f"Failed to load projections from folder: {tiff_folder}")

    # 2. Downsample (anti-aliased block-average to reduce memory and computation)
    # You control the downsampling factor f via configurations['downsample']
    # Higher f = lower resolution but faster & less memory. Adjust based on your needs.
    # NOTE: Downsampling affects achievable resolution - see geometry_reconstruction_voxel_size.py
    if configurations['downsample'] > 1:
        f = configurations['downsample']  # downsampling factor from config
        print(f"Downsampling by factor {f}x...")
        print(f"      NOTE: This will REDUCE maximum achievable resolution")
        print(f"      Effective pixel size increases from 50 μm to {50*f} μm")
        # Use block-average instead of stride sampling to avoid aliasing artifacts
        projections = downsample_block_mean_pad(projections, f).astype(np.float32)
    else:
        f = 1

    # 3. Get calibrated shift (adjusted for downsampling)
    # The calibrated shift was measured at original resolution (downsample=1)
    # We divide by the current downsampling factor to get the correct pixel shift
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / configurations['downsample']
    print(f"--> Using calibrated shift: {calibrated_shift_px:.2f} px (original) -> {shift_val:.2f} px (after downsample {configurations['downsample']}x)")
    
    # 3.5. Adjust voxel size for downsampling
    # CRITICAL: Downsampling increases effective voxel size
    # This matches pixel-size-first approach: pixel_size = configurations['pixel_size'] * f
    effective_voxel_size = configurations['voxel_size'] * f
    print(f"--> Voxel size adjustment: {configurations['voxel_size']:.2f} μm × {f} = {effective_voxel_size:.2f} μm")

    # 4. Select I0 ROI for normalization
    sino_raw = generate_collapsed_sinogram(projections)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections.shape[0])

    # 5. Normalize projections
    projections_norm = normalize_projections(projections, I0_override=mean_I0)
    
    del projections #del is used to free memory
    gc.collect()

    # 6. Geometry and Reconstruction
    # Setup geometry with ADJUSTED voxel size (already accounts for downsampling)
    # The voxel_size entering setup_geometry is already adjusted for downsampling
    # Inside: required_pixel_size = voxel_size × magnification (simple!)
    geo, angles = setup_geometry(
        projections_norm.shape, 
        effective_voxel_size,  # ← Already adjusted for downsampling
        configurations['DSD'], 
        configurations['DSO'], 
        shift_val, 
        configurations['total_angle'], 
        shift_sign=configurations['shift_sign'],
        downsample_factor=configurations['downsample']  # ← For Nyquist limit calculation
    )

    print("--> Preparing data for TIGRE...")
    input_data = np.transpose(projections_norm, (2, 0, 1)).copy() # TIGRE expects (Angles, DetectorV, DetectorU)
    
    print("--> Running FDK...")
    volume = algs.fdk(input_data, geo, angles, filter=configurations['filter_type'])  # FDK reconstruction
    print_volume_info(volume, geo)
    
    # PHASE 4: POST-PROCESSING & EXPORT
    print("PHASE 4: POST-PROCESSING & EXPORT")
    
    # Step 4.1: NAPARI visualization
    print("--> Opening Napari viewer...")
    
    # Voxel scale for napari
    voxel_scale = (geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2])
    
    # Interactive filtering (commented out for now)
    # filter_choice = input("\nDo you want to use INTERACTIVE filtering? (y/n): ").strip().lower()
    # if filter_choice == 'y':
    #     from napari_filters import interactive_filter_viewer
    #     viewer = interactive_filter_viewer(volume, name="CT Volume", scale=voxel_scale, nii_filepath=None)
    #     napari.run()
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
            print("\n" + "="*60)
            print("  HU CONVERSION SETUP")
            print("="*60)
            print("Please provide the gray scale values measured from the reconstruction:")
            water_val = float(input("  Enter gray scale value for WATER: "))
            air_val = float(input("  Enter gray scale value for AIR: "))
            
            # Export HU volume (reuses original .nii header)
            export_volume_HU(nii_filepath, volume, water_val, air_val)
    
    return volume

if __name__ == "__main__":
    # ========== VOXEL-SIZE-FIRST CONFIGURATION ==========
    # KEY CHANGE: Use 'voxel_size' in μm instead of 'pixel_size' in mm
    # This makes it easier to compare with commercial micro-CT systems
    
    CONFIG = {
        # PRIMARY INPUT: Desired voxel size in micrometers (μm)
        # This is what commercial systems and papers report
        # Examples:
        #   - 56.1 μm: Nyquist-optimal for setup (DSD=925, DSO=893, 50μm detector)
        #   - 30 μm: Higher resolution (oversampling - smooth but larger files)
        #   - 100 μm: Fast preview (undersampling - faster but less detail)
        'voxel_size': 56.1,  # μm - CHOOSE YOUR DESIRED RESOLUTION HERE
        
        # Geometry parameters (same as original)
        'DSD': 925,  # Distance Source to Detector (mm)
        'DSO': (925-32),  # Distance Source to Object (mm)
        
        # Downsampling reduces resolution but speeds up reconstruction
        # NOTE: This affects maximum achievable resolution!
        # downsample=1: Full detector resolution (50 μm pixels)
        # downsample=2: Half resolution (100 μm effective pixels)
        # downsample=4: Quarter resolution (200 μm effective pixels)
        'downsample': 2,
        
        # Acquisition parameters
        'total_angle': 2 * np.pi,
        'calibrated_shift_px': 5.12,  # From calibration (in original pixels, downsample=1)
        'shift_sign': 1,
        
        # Reconstruction filter
        'filter_type': 'hann',
        

        # Output folder
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Image_reconstruction\reconstructed_volumes_Nift'
    }
    

    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\marta_caixa_SiPM'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Mouse_PC'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Laranja'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Haste_perfeita\Try_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\PEIXE\PEIXE'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\Suporte_micro_ct_I3N'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\peixe_joao'

    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
