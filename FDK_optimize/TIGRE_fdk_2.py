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



from data_processing_3D_2 import (
    load_images, 
    generate_collapsed_sinogram,
    selecionar_roi_I0,
    get_I0_from_roi,
    show_results,
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
    
    Args:
        volume: Reconstructed volume array
        geo: TIGRE geometry object
        source_folder: Path to source data folder
        base_output: Custom output folder path. If None, uses 'reconstructed_volumes' in project root
    
    IMPORTANT - VALUE PRESERVATION FOR ANALYSIS:
    ================================================
    • Attenuation coefficients (μ values) are stored in FLOAT32 format
    • NO scaling, normalization, or clipping is applied to the data
    • Raw FDK algorithm values are preserved EXACTLY as they are
    • Contrast/brightness adjustments in ImageJ, Napari, or 3D Slicer are ONLY 
      for visualization - the numeric values in the file NEVER change
    ================================================

    """
    # Use default folder if none specified
    if base_output is None:
        base_output = "reconstructed_volumes"
    
    # Create output folder structure (name to be saved)
    dataset_name = os.path.basename(os.path.normpath(source_folder))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create timestamp: day_month_hour/minute
    now = datetime.now()
    month_abbr = now.strftime("%b").lower()  # jan, feb, mar, ...
    timestamp = f"{now.day}_{month_abbr}_{now.hour}h{now.minute}"
    
    # Create folder name: script_dataset_timestamp
    folder_name = f"{script_name}_{dataset_name}_{timestamp}"
    output_folder = os.path.join(base_output, folder_name)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Nome dos arquivos
    filename = f"{dataset_name}.nii"
    filepath = os.path.join(output_folder, filename)
    

    volume_export = volume.astype(np.float32)  # Ensure float32 for attenuation coefficients
    
    # NIfTI padrão: (X, Y, Z) onde Z é o eixo superior-inferior
    # TIGRE retorna (Z, Y, X), então precisamos transpor para (X, Y, Z)
    volume_export = np.transpose(volume_export, (2, 1, 0))

    
    # Create affine matrix with correct spacing
    # NIfTI uses RAS orientation (Right-Anterior-Superior)
    affine = np.eye(4)
    affine[0, 0] = geo.dVoxel[2]  # X spacing (Left-Right)
    affine[1, 1] = geo.dVoxel[1]  # Y spacing (Posterior-Anterior)
    affine[2, 2] = geo.dVoxel[0]  # Z spacing (Inferior-Superior)
    
    # Centralizar o volume na origem
    affine[0, 3] = -(volume_export.shape[0] * geo.dVoxel[2]) / 2.0
    affine[1, 3] = -(volume_export.shape[1] * geo.dVoxel[1]) / 2.0
    affine[2, 3] = -(volume_export.shape[2] * geo.dVoxel[0]) / 2.0
    
    # Create NIfTI image com header completo
    nii_img = nib.Nifti1Image(volume_export, affine)
    
    # Configurar header para garantir interpretação correta
    nii_img.header.set_xyzt_units('mm', 'sec')
    nii_img.header['descrip'] = f'FDK Reconstruction - {dataset_name}'
    
    # Save file
    print(f"    Saving NIfTI file...")
    nib.save(nii_img, filepath)
    print(f"    ✓ File saved successfully!")
    
    # Save metadata in txt (expanded)
    metadata_file = os.path.join(output_folder, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Volume Reconstruction Metadata\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Source Dataset: {dataset_name}\n")
        f.write(f"Source Path: {source_folder}\n\n")
        f.write(f"Volume Information (exported to NIfTI):\n")
        f.write(f"  Exported shape (X, Y, Z): {volume_export.shape}\n")
        f.write(f"  Original shape (TIGRE): {volume.shape} (Z, Y, X)\n")
        f.write(f"  Data type: {volume_export.dtype}\n")
        f.write(f"  Min: {np.min(volume_export):.6f}\n")
        f.write(f"  Max: {np.max(volume_export):.6f}\n")
        f.write(f"  Mean: {np.mean(volume_export):.6f}\n")
        f.write(f"  Std: {np.std(volume_export):.6f}\n")
        f.write(f"  Total size (bytes): {volume_export.nbytes}\n\n")
        f.write(f"  Geometry Information:\n")
        f.write(f"  Voxel size (mm): {geo.dVoxel}\n")
        f.write(f"  Number of voxels: {geo.nVoxel}\n")
        f.write(f"  Physical dimensions (mm): {geo.sVoxel}\n")
        f.write(f"  DSD: {geo.DSD} mm\n")
        f.write(f"  DSO: {geo.DSO} mm\n\n")
        f.write(f"  Python: {sys.version.split()[0]} ({sys.platform})\n")
        f.write(f"  TIGRE version: {getattr(tigre, '__version__', 'unknown')}\n")
        # Attempt to record reconstruction parameters if available
        recon_info = globals().get('CONFIG') or globals().get('configurations') or None
        if recon_info:
            f.write("\nReconstruction Parameters:\n")
            try:
                for k, v in recon_info.items():
                    f.write(f"  {k}: {v}\n")
            except Exception:
                f.write(f"  (could not enumerate reconstruction parameters)\n")

    return filepath


def export_volume_HU(original_nii_path, volume, water_val, air_val):
    """
    Converts the volume to HU and saves with _HU suffix.
    Reuses the header/affine from the original .nii file.
    """
    
    # Convert volume to HU
    volume_HU = HU_conversion(volume, water_val, air_val)
    
    # Load original .nii to get header and affine
    original_nii = nib.load(original_nii_path)
    
    # Transpose HU volume to match NIfTI format
    volume_HU_export = volume_HU.astype(np.float32)
    volume_HU_export = np.transpose(volume_HU_export, (2, 1, 0))
    
    # Create new .nii with HU data but same header/affine
    nii_HU = nib.Nifti1Image(volume_HU_export, original_nii.affine, original_nii.header)
    nii_HU.header['descrip'] = original_nii.header['descrip'].decode() + ' (HU)'
    
    # Save with _HU suffix
    filepath_HU = original_nii_path.replace('.nii', '_HU.nii')
    nib.save(nii_HU, filepath_HU)
    
    filename_HU = os.path.basename(filepath_HU)
    print(f"    ✓ Volume in Hounsfield Units saved: {filename_HU}")
    
    return filepath_HU


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


def setup_geometry(img_shape, pixel_size, DSD, DSO, shift_pixels, total_angle, shift_sign, voxel_ratio=1.0):
    print(f"--> Setting up geometry with shift: {shift_pixels:.2f} px")
    
    height, width, n_angles = img_shape # img_shape = (Height, Width, Angles)
    geo = tigre.geometry(mode="cone") # Cone beam geometry
    
    geo.nDetector = np.array([height, width]) #ndetector is the number of pixels in detector 
    geo.dDetector = np.array([pixel_size, pixel_size]) #ddetector is the size of each pixel 
    geo.sDetector = geo.nDetector * geo.dDetector # Size of the detector 

    
    # Calculate base voxel size from Nyquist criterion
    magnification = DSD / DSO 
    voxel_size_base = pixel_size / magnification
    
    # Apply user-defined voxel ratio (allows oversampling or downsampling)
    voxel_size = voxel_size_base * voxel_ratio
    
    print(f"--> Magnification: {magnification:.4f}")
    print(f"--> Base voxel size (Nyquist): {voxel_size_base:.6f} mm")
    print(f"--> Final voxel size (ratio={voxel_ratio}): {voxel_size:.6f} mm")
    
    # geo.dVoxel = Size of EACH voxel in mm [dZ, dY, dX]
    # Individual voxel dimensions (isotropic = same size in all directions)
    geo.dVoxel = np.array([voxel_size, voxel_size, voxel_size])
    
    # geo.nVoxel = NUMBER of voxels in each dimension [nZ, nY, nX]
    # Calculate how many voxels fit in the detector field of view
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size),  # Z (height): detector height / voxel_size
        int(geo.sDetector[1] / voxel_size),  # Y (depth): detector width / voxel_size
        int(geo.sDetector[1] / voxel_size)   # X (width): same as Y to create cubic FOV in axial plane
    ])
    # Note: nVoxel[1] == nVoxel[2] creates a square/cubic volume in the XY plane (axial slices)
    # You can change nVoxel[2] to be different (e.g., int(geo.sDetector[1] * 0.8 / voxel_size))
    # to create a rectangular FOV and reduce memory usage
    #Aumentar nVoxel (mais voxels com mesmo dVoxel) → maior detalhe espacial, mais RAM/VRAM e tempo.
    
    # geo.sVoxel = TOTAL physical size of volume in mm [sZ, sY, sX]
    # Total volume dimensions = number of voxels × size of each voxel
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = DSD # Distance Source to Detector
    geo.DSO = DSO # Distance Source to Object
    
    # APPLYING SHIFT 
    shift_mm = shift_pixels * pixel_size
    geo.offDetector = np.array([0.0, shift_mm * shift_sign]) 
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    
    return geo, angles

def main(tiff_folder, configurations, output_folder=None):
    """
    Main reconstruction pipeline.
    
    Args:
        tiff_folder: Path to folder containing TIFF images
        configurations: Configuration dictionary
        output_folder: Custom output folder for .nii files (optional)
    """
    # 1. Load
    projections = load_images(tiff_folder)

    # 2. Downsample
    if configurations['downsample'] > 1:
        f = configurations['downsample'] # downsample factor, the configuartions is used to get the value of downsample
        projections = projections[::f, ::f, :]
        pixel_size = configurations['pixel_size'] * f
    else:
        pixel_size = configurations['pixel_size']

    # 3. Get calibrated shift (adjusted for downsample)
    # The calibrated shift was measured at original resolution (downsample=1)
    # We divide by the current downsample factor to get the correct pixel shift
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / configurations['downsample']
    print(f"--> Using calibrated shift: {calibrated_shift_px:.2f} px (original) -> {shift_val:.2f} px (after downsample {configurations['downsample']}x)")

    # 4. Select I0 ROI for normalization
    sino_raw = generate_collapsed_sinogram(projections)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background)

    # 5. Normalize projections
    projections_norm = normalize_projections(projections, I0_override=mean_I0)
    sino_norm_preview = generate_collapsed_sinogram(projections_norm)
    show_results(sino_raw, sino_norm_preview, shift_val)
    
    del projections, sino_raw, sino_norm_preview #del is used to free memory
    gc.collect()

    # 6. Geometry and Reconstruction
    geo, angles = setup_geometry(projections_norm.shape, pixel_size, configurations['DSD'], configurations['DSO'], shift_val, configurations['total_angle'], shift_sign=configurations['shift_sign'], voxel_ratio=configurations['voxel_ratio'])

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
    CONFIG = {
        'pixel_size': 0.05,
        'DSD': 925,
        'DSO': (925-32),
        'downsample': 4,
        'total_angle': 2 * np.pi,
        'calibrated_shift_px': 5.12,  # From calibration (in original pixels, downsample=1)
        'shift_sign': 1,           
        'filter_type': 'hann',
        
        # VOXEL SIZE CONTROL
        'voxel_ratio': 1.0,
        # voxel_ratio multiplies the Nyquist voxel size (pixel_size / magnification)
        # • voxel_ratio = 1.0: Optimal resolution matching detector pixels (RECOMMENDED)

        # • voxel_ratio < 1.0 (e.g., 0.5): SMALLER voxels, HIGHER resolution
        #     ✓ Smoother images, less pixelation
        #     ✓ May reduce aliasing artifacts
        #     ✗ Does NOT add real detail beyond detector limit
        #     ✗ Volume size increases by (1/ratio)³ → more RAM/VRAM needed
        #     ✗ Reconstruction time increases significantly

        # • voxel_ratio > 1.0 (e.g., 2.0): LARGER voxels, LOWER resolution
        #     ✓ Much faster reconstruction (volume reduced by ratio³)
        #     ✓ Lower memory usage
        #     ✓ Good for quick previews or testing
        #     ✗ Loss of spatial resolution and detail
        
        
        # OTIMIZAÇÕES
        'volume_roi': None,  # None = Volume completo | Ex: {'z_range': [50,150], 'xy_crop': 0.7}
        # 'volume_roi': {'z_range': None, 'xy_crop': 0.8}  # Exemplo: 80% do FOV central (acelera)

        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Image_reconstruction\reconstructed_volumes_Nift'
    }
    
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\marta_caixa_SiPM'
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Mouse_PC'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Laranja'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Haste_perfeita\Try_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\PEIXE\PEIXE'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\Suporte_micro_ct_I3N'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\peixe_joao'

    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))

