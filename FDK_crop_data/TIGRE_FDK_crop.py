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

from data_processing_crop import (
    load_images, 
    generate_collapsed_sinogram,
    selecionar_roi_I0,
    get_I0_from_roi,
    select_crop_region,
    apply_crop_to_projections
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
    Normaliza as projeções usando -log(I/I0).
    
    OTIMIZAÇÃO: Esta função agora recebe projeções JÁ CROPPED,
    reduzindo drasticamente o número de operações matemáticas.
    """
    print("--> Normalizing cropped projections...")
    print(f"    Input shape: {projections_raw.shape}")
    print(f"    Memory size: {projections_raw.nbytes / 1e6:.1f} MB")

    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 1))
    else:
        I0 = float(I0_override)

    # Convert to float32 for calculations (mais leve que float64)
    projections_raw = projections_raw.astype(np.float32)
    
    # Avoid division by zero
    ratio = projections_raw / (I0 + 1e-6)
    
    # Prevent log(0) and extreme values
    ratio = np.clip(ratio, 1e-6, 1.2)
    
    # Beer-Lambert law: -log(I/I0)
    projections_norm = -np.log(ratio)
    
    # Remove negative values (artifacts)
    projections_norm[projections_norm < 0] = 0
    
    print(f"    ✓ Normalization complete")
    return projections_norm


def downsample_block_mean_pad(proj, f):
    """Downsample projections by factor f using block-average with edge padding."""
    H, W, A = proj.shape
    pad_h = (-H) % f
    pad_w = (-W) % f
    if pad_h or pad_w:
        proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        proj_p = proj
    
    Hc, Wc = proj_p.shape[:2]
    return proj_p.reshape(Hc//f, f, Wc//f, f, A).mean(axis=(1, 3))


def setup_geometry(img_shape, pixel_size, DSD, DSO, shift_pixels, total_angle, 
                   shift_sign, voxel_ratio=1.0, crop_params=None):
    """
    Setup TIGRE geometry with support for cropped projections.
    
    CRITICAL: When projections are cropped, the detector geometry must be adjusted:
    - nDetector: new height and width after crop
    - sDetector: physical size changes proportionally
    - offDetector: must account for the shift of the cropped region center
    
    Args:
        img_shape: (height, width, n_angles) - AFTER crop and downsample
        pixel_size: Detector pixel size in mm (after downsample)
        DSD: Distance Source to Detector in mm
        DSO: Distance Source to Object in mm
        shift_pixels: Detector shift in pixels (for center correction)
        total_angle: Total rotation angle in radians
        shift_sign: Sign of the shift (+1 or -1)
        voxel_ratio: Multiplier for voxel size
        crop_params: Dict with crop information (from select_crop_region)
    """
    
    print(f"--> Setting up geometry...")
    
    height, width, n_angles = img_shape
    geo = tigre.geometry(mode="cone")
    
    # Detector size after crop
    geo.nDetector = np.array([height, width])
    geo.dDetector = np.array([pixel_size, pixel_size])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    # Calculate voxel size
    magnification = DSD / DSO 
    voxel_size_base = pixel_size / magnification
    voxel_size = voxel_size_base * voxel_ratio
    
    print(f"    Magnification: {magnification:.4f}")
    print(f"    Base voxel size (Nyquist): {voxel_size_base:.6f} mm")
    print(f"    Final voxel size (ratio={voxel_ratio}): {voxel_size:.6f} mm")
    
    geo.dVoxel = np.array([voxel_size, voxel_size, voxel_size])
    
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size),
        int(geo.sDetector[1] / voxel_size),
        int(geo.sDetector[1] / voxel_size)
    ])
    
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = DSD
    geo.DSO = DSO
    
    # CRITICAL: Detector offset adjustment for cropped projections
    # The calibrated shift is relative to the ORIGINAL detector center
    # After crop, we need to account for:
    # 1. The horizontal shift from crop
    # 2. The original calibrated shift
    
    shift_mm = shift_pixels * pixel_size
    
    if crop_params is not None:
        # Calculate how much the detector center moved due to crop
        original_center = crop_params['original_center_col']
        new_center = (crop_params['col_end'] + crop_params['col_start']) / 2.0
        crop_shift_pixels = new_center - original_center
        crop_shift_mm = crop_shift_pixels * pixel_size
        
        # Total offset = calibrated shift + crop shift
        total_shift_mm = shift_mm + crop_shift_mm * shift_sign
        
        print(f"    Calibrated shift: {shift_pixels:.2f} px = {shift_mm:.4f} mm")
        print(f"    Crop shift: {crop_shift_pixels:.2f} px = {crop_shift_mm:.4f} mm")
        print(f"    Total detector offset: {total_shift_mm:.4f} mm")
        
        geo.offDetector = np.array([0.0, total_shift_mm])
    else:
        # No crop, just use calibrated shift
        print(f"    Detector shift: {shift_pixels:.2f} px = {shift_mm:.4f} mm")
        geo.offDetector = np.array([0.0, shift_mm * shift_sign])
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    
    print(f"    Detector size: {geo.nDetector} px = {geo.sDetector} mm")
    print(f"    Volume size: {geo.nVoxel} voxels = {geo.sVoxel} mm")
    
    return geo, angles


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
    
    WHY THIS ORDER IS BETTER:
    - Crop BEFORE normalize → fewer pixels to process in log/division operations
    - Original script was normalizing ALL pixels then cropping → waste of CPU
    - This order can be 2-5x faster depending on crop size
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
    gc.collect()
    
    # 4. SELECT CROP on first RAW projection
    first_proj_raw = projections_raw[:, :, 0]
    crop_params = select_crop_region(first_proj_raw)
    
    # 5. Apply crop to all RAW projections (BEFORE normalization!)
    projections_cropped_raw = apply_crop_to_projections(projections_raw, crop_params)
    
    del projections_raw  # Free original data
    gc.collect()
    
    # 6. Normalize ONLY the cropped projections (much faster!)
    projections_norm = normalize_projections(projections_cropped_raw, I0_override=mean_I0)
    
    del projections_cropped_raw  # Free cropped raw data
    gc.collect()
    
    # 7. Downsample if needed (applied AFTER crop for max efficiency)
    if configurations['downsample'] > 1:
        f = configurations['downsample']
        print(f"--> Downsampling by {f}x...")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32)
        pixel_size = configurations['pixel_size'] * f
        print(f"    Final shape: {projections_final.shape}")
        
        del projections_norm
        gc.collect()
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
    gc.collect()
    
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
        'downsample': 2,
        'total_angle': 2 * np.pi,
        'calibrated_shift_px': 5.12,
        'shift_sign': 1,           
        'filter_type': 'hann',
        'voxel_ratio': 2,
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Image_reconstruction\reconstructed_volumes_Nift'
    }
    
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))