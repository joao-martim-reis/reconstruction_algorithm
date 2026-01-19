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

from geometry_reconstruction import setup_geometry
from crop_functions import select_crop_region, apply_crop_to_projections

from data_processing_crop import (
    load_images, 
    generate_collapsed_sinogram,
    selecionar_roi_I0,
    get_I0_from_roi,
)

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


def export_volume_to_nii(volume, geo, source_folder, base_output=None):
    """
    Exports the reconstructed volume to NIfTI format (.nii).
    Creates a unique subfolder for each reconstruction inside base_output.
    """
    if base_output is None:
        base_output = "reconstructed_volumes"
    
    dataset_name = os.path.basename(os.path.normpath(source_folder))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    now = datetime.now()
    month_abbr = now.strftime("%b").lower()
    timestamp = f"{now.day}_{month_abbr}_{now.hour}h{now.minute}"
    
    folder_name = f"{script_name}_{dataset_name}_{timestamp}"
    output_folder = os.path.join(base_output, folder_name)
    
    os.makedirs(output_folder, exist_ok=True)
    
    filename = f"{dataset_name}.nii"
    filepath = os.path.join(output_folder, filename)
    
    volume_export = volume.astype(np.float32)
    volume_export = np.transpose(volume_export, (2, 1, 0))
    
    affine = np.eye(4)
    affine[0, 0] = geo.dVoxel[2]
    affine[1, 1] = geo.dVoxel[1]
    affine[2, 2] = geo.dVoxel[0]
    
    affine[0, 3] = -(volume_export.shape[0] * geo.dVoxel[2]) / 2.0
    affine[1, 3] = -(volume_export.shape[1] * geo.dVoxel[1]) / 2.0
    affine[2, 3] = -(volume_export.shape[2] * geo.dVoxel[0]) / 2.0
    
    nii_img = nib.Nifti1Image(volume_export, affine)
    nii_img.header.set_xyzt_units('mm', 'sec')
    nii_img.header['descrip'] = f'FDK Reconstruction - {dataset_name}'
    
    print(f"    Saving NIfTI file...")
    nib.save(nii_img, filepath)
    print(f"    ✓ File saved successfully!")
    
    metadata_file = os.path.join(output_folder, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Volume Reconstruction Metadata\n")
        f.write(f"Volume Information (exported to NIfTI):\n")
        f.write(f"  Exported shape (X, Y, Z): {volume_export.shape}\n")
        f.write(f"  Original shape (TIGRE): {volume.shape} (Z, Y, X)\n")
        f.write(f"  Data type: {volume_export.dtype}\n")
        f.write(f"  Total size (bytes): {volume_export.nbytes}\n\n")
        f.write(f"  Geometry Information:\n")
        f.write(f"  Voxel size (mm): {geo.dVoxel}\n")
        f.write(f"  Number of voxels: {geo.nVoxel}\n")
        f.write(f"  Physical dimensions (mm): {geo.sVoxel}\n")
        f.write(f"  DSD: {geo.DSD} mm\n")
        f.write(f"  DSO: {geo.DSO} mm\n\n")
        f.write(f"  Python: {sys.version.split()[0]} ({sys.platform})\n")
        f.write(f"  TIGRE version: {getattr(tigre, '__version__', 'unknown')}\n")

    return filepath


def export_volume_HU(original_nii_path, volume, water_val, air_val):
    """
    Converts the volume to HU and saves with _HU suffix.
    Reuses the header/affine from the original .nii file.
    """
    volume_HU = HU_conversion(volume, water_val, air_val)
    original_nii = nib.load(original_nii_path)
    
    volume_HU_export = volume_HU.astype(np.float32)
    volume_HU_export = np.transpose(volume_HU_export, (2, 1, 0))
    
    nii_HU = nib.Nifti1Image(volume_HU_export, original_nii.affine, original_nii.header)
    nii_HU.header['descrip'] = original_nii.header['descrip'].decode() + ' (HU)'
    
    filepath_HU = original_nii_path.replace('.nii', '_HU.nii')
    nib.save(nii_HU, filepath_HU)
    
    filename_HU = os.path.basename(filepath_HU)
    print(f"    ✓ Volume in Hounsfield Units saved: {filename_HU}")
    
    return filepath_HU


def normalize_projections(projections_raw, I0_override=None):
    """
    Normalizes projections using -log(I/I0).
    
    This function receives ALREADY CROPPED projections, drastically reducing the number of mathematical operations.
    """

    print("--> Normalizing cropped projections...")
    print(f"    Input shape: {projections_raw.shape}")
    print(f"    Memory size: {projections_raw.nbytes / 1e6:.1f} MB")

    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 1))
    else:
        I0 = float(I0_override)

    # Convert to float32 for calculations (lighter than float64)
    projections_raw = projections_raw.astype(np.float32)
    ratio = projections_raw / (I0 + 1e-6)# Avoid division by zero
    ratio = np.clip(ratio, 1e-6, 1.2)# Prevent log(0) and extreme values
    
    # Beer-Lambert law: -log(I/I0)
    projections_norm = -np.log(ratio)
    
    # Remove negative values (artifacts)
    projections_norm[projections_norm < 0] = 0
    
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
    H, W, A = proj.shape # Height, Width, Angles
    
    # Calculate padding needed to make dimensions divisible by f
    pad_h = (-H) % f
    pad_w = (-W) % f
    
    # Apply edge padding if necessary to ensure clean division
    if pad_h or pad_w:
        proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        proj_p = proj
    
    # Reshape to separate blocks and compute mean across block elements
    Hc, Wc = proj_p.shape[:2]
    return proj_p.reshape(Hc//f, f, Wc//f, f, A).mean(axis=(1, 3))


def main(tiff_folder, configurations, output_folder=None):
    """
    Main reconstruction pipeline with memory-efficient cropping.
    
    OPTIMIZED PIPELINE:
    1. Load images (RAW)
    2. Generate collapsed sinogram (RAW - for I0 selection)
    3. Select I0 ROI (on collapsed sinogram)
    4. SELECT CROP REGION (on first RAW projection)
    5. Apply crop to all RAW projections → HUGE memory reduction
    6. Normalize ONLY the cropped projections → much faster!
    7. Downsample if needed (optional, after crop)
    8. Setup geometry (adjusted for crop)
    9. Reconstruct with FDK

    """
    
    # 1. Load RAW images
    projections_raw = load_images(tiff_folder)
    
    print(f"\n--> RAW projections loaded: {projections_raw.shape}")
    print(f"    Memory: {projections_raw.nbytes / 1e6:.1f} MB")

    # 2. Collapsed sinogram for I0 selection (using RAW data)
    sino_raw = generate_collapsed_sinogram(projections_raw)
    
    # 3. Select I0 ROI
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background)
    
    del sino_raw  # Free memory
    gc.collect() # Garbage collection
    
    # 4. SELECT CROP on first RAW projection
    first_proj_raw = projections_raw[:, :, 0]
    crop_params = select_crop_region(first_proj_raw)
    
    # 5. Apply crop to all RAW projections (BEFORE normalization!)
    projections_cropped_raw = apply_crop_to_projections(projections_raw, crop_params)
    
    del projections_raw  # Free original data
    gc.collect() # Garbage collection
    
    # 6. Normalize ONLY the cropped projections (much faster!)
    projections_norm = normalize_projections(projections_cropped_raw, I0_override=mean_I0)
    
    del projections_cropped_raw  # Free cropped raw data
    gc.collect() # Garbage collection
    
    # 7. Downsample if needed (applied AFTER crop for max efficiency)
    if configurations['downsample'] > 1:
        f = configurations['downsample']
        print(f"--> Downsampling by {f}x...")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32)
        pixel_size = configurations['pixel_size'] * f
        print(f"    Final shape: {projections_final.shape}")
        
        del projections_norm
        gc.collect() # Garbage collection
    else:
        projections_final = projections_norm
        pixel_size = configurations['pixel_size']
    
    # 8. Adjust shift for downsampling
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / configurations['downsample']
    
    # 9. Setup geometry (CRITICAL: pass crop_params for correct offset)
    geo, angles = setup_geometry(
        projections_final.shape, 
        pixel_size, 
        configurations['DSD'], 
        configurations['DSO'], 
        shift_val, 
        configurations['total_angle'], 
        shift_sign=configurations['shift_sign'], 
        voxel_ratio=configurations['voxel_ratio'],
        crop_params=crop_params  # ← CRITICAL for correct detector offset
    )

    # 10. Reconstruction
    print("--> Preparing data for TIGRE...")
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    
    del projections_final
    gc.collect() # Garbage collection
    
    print("--> Running FDK...")
    volume = algs.fdk(input_data, geo, angles, filter=configurations['filter_type'])
    print_volume_info(volume, geo)
    
    # Export options
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
    
    # Napari visualization
    print("--> Opening Napari...")
    viewer = napari.Viewer()
    
    ndim = volume.ndim 
    if ndim == 2:
        viewer.add_image(volume, scale=(geo.dVoxel[1], geo.dVoxel[2]))
    elif ndim == 3:
        viewer.add_image(volume, scale=(geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]))
    else:
        viewer.add_image(volume)

    napari.run()
    
    # HU conversion
    if export_choice == 'y' and nii_filepath is not None:
        hu_choice = input("\nDo you want to also export in Hounsfield Units (HU)? (y/n): ").strip().lower()
        if hu_choice == 'y':
            print("\n" + "="*60)
            print("  HU CONVERSION SETUP")
            print("="*60)
            print("Please provide the gray scale values measured from the reconstruction:")
            water_val = float(input("  Enter gray scale value for WATER: "))
            air_val = float(input("  Enter gray scale value for AIR: "))
            
            export_volume_HU(nii_filepath, volume, water_val, air_val)
    
    return volume


if __name__ == "__main__":
    CONFIG = {
        'pixel_size': 0.05,
        'DSD': 925,
        'DSO': (925-32),
        'downsample': 4,
        'total_angle': 2 * np.pi,
        'calibrated_shift_px': 5.12,
        'shift_sign': 1,           
        'filter_type': 'hann',
        'voxel_ratio': 2,
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Image_reconstruction\reconstructed_volumes_Nift'
    }
    
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\marta_caixa_SiPM'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Mouse_PC'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Laranja'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Haste_perfeita\Try_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\PEIXE\PEIXE'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\Suporte_micro_ct_I3N'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\peixe_joao'
    
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))