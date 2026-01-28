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
from rotation_alignment import apply_rotation_to_projections

# Import HU conversion function
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from HU_conversion import HU_conversion


def print_volume_info(volume, geo=None):

    """
    Prints detailed information about the reconstructed volume.
    """
    print("\n" + "="*60)
    print("RECONSTRUCTED VOLUME INFORMATION")
    print("="*60)
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}")
    print("="*60 + "\n") 
    print(f"\nGeometric information:")
    print(f"  - Voxel size: {geo.dVoxel} mm")
    print(f"  - Number of voxels: {geo.nVoxel}")
    print(f"  - Physical dimensions: {geo.sVoxel} mm")
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
    """Downsample projections by factor f using block-average with edge padding.
    
    This anti-aliased downsampling averages f×f pixel blocks instead of picking one pixel.
    Reduces aliasing artifacts and preserves signal better than stride sampling.

    • Detector pixel size increases by factor f: new_pixel_size = original_pixel_size × f
    • Reconstructed voxel size also scales by f (via Nyquist: voxel_size = pixel_size / magnification)
    • Spatial resolution in reconstructed volume DECREASES by factor f
    • This is a trade-off: lower resolution for reduced memory (~f² reduction) and faster reconstruction
    
    IMPORTANT: Downsampling increases effective pixel size by factor f
    This REDUCES the maximum achievable resolution (Nyquist limit increases)
    """

    H, W, A = proj.shape
    # compute padding so H and W become divisible by f
    pad_h = (-H) % f # amount of padding (pixels) needed in height
    pad_w = (-W) % f # amount of padding (pixels) needed in width
    if pad_h or pad_w:
        proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge') #np.pad is used to pad arrays which means adding values to the edges of an array
    else:
        proj_p = proj
    
    Hc, Wc = proj_p.shape[:2]
    # reshape to blocks and average over the block axes (anti-aliasing)
    return proj_p.reshape(Hc//f, f, Wc//f, f, A).mean(axis=(1, 3))


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

    # 2. Apply rotation correction (BEFORE downsampling)
    rotation_angle = configurations.get('rotation_angle', 0.0)
    if rotation_angle != 0.0:
        print(f"Applying manual rotation correction of {rotation_angle}° to projections...")
        projections = apply_rotation_to_projections(projections, rotation_angle, order=3)
    else:
        print("No rotation applied (rotation_angle = 0)")

    # 3. Downsample (anti-aliased block-average to reduce memory and computation)
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
    
    # Ask if you want to export to .nii format
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
    
    # Open Napari to visualize the volume 
    print("--> Opening Napari...")
    print(f"--> Volume shape: {volume.shape}, ndim={getattr(volume, 'ndim', 'unknown')}")
    viewer = napari.Viewer()


    ndim = volume.ndim 
    if ndim == 2:
        viewer.add_image(volume, scale=(geo.dVoxel[1], geo.dVoxel[2]))
    elif ndim == 3:
        viewer.add_image(volume, scale=(geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]))
    else:
        # Fallback: add without explicit scale (napari will guess)
        viewer.add_image(volume)

    napari.run()
    
    
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
        
        # Rotation correction (positive=counterclockwise, negative=clockwise, 0=no rotation)
        'rotation_angle': 0.0,
        
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
