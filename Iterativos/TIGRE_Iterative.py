import numpy as np
import os
import sys
import gc
import time
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu
import matplotlib.pyplot as plt
import napari
import nibabel as nib
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HU_conversion import HU_conversion

from geometry_reconstruction import setup_geometry

from data_processing_3D_2 import (
    load_images, 
    generate_collapsed_sinogram,
    selecionar_roi_I0,
    get_I0_from_roi,
)


def print_volume_info(volume, geo=None):
    """Prints detailed information about the reconstructed volume."""
    print("\n" + "="*60)
    print("RECONSTRUCTED VOLUME INFORMATION")
    print("="*60)
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}")
    if geo is not None:
        print(f"\nGeometric information:")
        print(f"  - Voxel size: {geo.dVoxel} mm")
        print(f"  - Number of voxels: {geo.nVoxel}")
        print(f"  - Physical dimensions: {geo.sVoxel} mm")
    print("="*60 + "\n")


def export_volume_to_nii(volume, geo, source_folder, base_output=None):
    """
    Exports the reconstructed volume to NIfTI format (.nii).
    Creates a unique subfolder for each reconstruction inside base_output.

    IMPORTANT - VALUE PRESERVATION FOR ANALYSIS:
  
    • Attenuation coefficients (μ values) are stored in FLOAT32 format
    • NO scaling, normalization, or clipping is applied to the data
    • Raw algorithm values are preserved EXACTLY as they are
    • Contrast/brightness adjustments in ImageJ, Napari, or 3D Slicer are ONLY 
      for visualization - the numeric values in the file NEVER change
    
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
    nii_img.header['descrip'] = f'Iterative Reconstruction - {dataset_name}'
    
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
    """Converts the volume to HU and saves with _HU suffix."""
    volume_HU = HU_conversion(volume, water_val, air_val)
    original_nii = nib.load(original_nii_path)
    
    volume_HU_export = np.transpose(volume_HU.astype(np.float32), (2, 1, 0))
    
    nii_HU = nib.Nifti1Image(volume_HU_export, original_nii.affine, original_nii.header)
    nii_HU.header['descrip'] = original_nii.header['descrip'].decode() + ' (HU)'
    
    filepath_HU = original_nii_path.replace('.nii', '_HU.nii')
    nib.save(nii_HU, filepath_HU)
    print(f"    ✓ HU Volume saved: {os.path.basename(filepath_HU)}")
    return filepath_HU


def normalize_projections(projections_raw, I0_override=None):
    """
    Normalize projections using Beer-Lambert law: I = I0 * exp(-μt)
    Converting to attenuation: -ln(I/I0) = μt

    """
    print("--> Normalizing projections...")

    # I0 estimation (Air) — can be overridden with mean_fundo from ROI
    if I0_override is None:
        # fallback: estimate I0 from dark percentile of the projections
        I0 = float(np.percentile(projections_raw, 1))  # 1st percentile = dark regions (air)
    else:
        I0 = float(I0_override)

    projections_raw = projections_raw.astype(np.float32)  # Ensure float for log calculation
    ratio = projections_raw / (I0 + 1e-6)  # Avoid division by zero
    ratio = np.clip(ratio, 1e-6, 1.2)  # Prevent log(0) and extreme values
    
    projections_norm = -np.log(ratio)
    projections_norm[projections_norm < 0] = 0
    return projections_norm


def downsample_block_mean_pad(proj, f):
    """Downsample projections by factor f using block-average with edge padding.
    
    This anti-aliased downsampling averages f×f pixel blocks instead of picking one pixel.
    Reduces aliasing artifacts and preserves signal better than stride sampling.
    
    Args:
        proj: numpy array with shape (H, W, A) - projection images
        f: integer downsampling factor (e.g., 2, 4, 8)
        
    Returns:
        downsampled array with shape (ceil(H/f), ceil(W/f), A)
        
    IMPORTANT - RESOLUTION & VOXEL SIZE:
    =====================================
    • Detector pixel size increases by factor f: new_pixel_size = original_pixel_size × f
    • Reconstructed voxel size also scales by f (via Nyquist: voxel_size = pixel_size / magnification)
    • Spatial resolution in reconstructed volume DECREASES by factor f
    • This is a trade-off: lower resolution for reduced memory (~f² reduction) and faster reconstruction
    • You control the downsampling factor via configurations['downsample'] - adjust based on your needs:
        ◦ f=1: Full resolution (no downsampling) - highest detail, most memory
        ◦ f=2: Half resolution - good balance for most cases
        ◦ f=4: Quarter resolution - fast previews, less memory
        ◦ f=8+: Very coarse - quick tests only
    """
    H, W, A = proj.shape
    # compute padding so H and W become divisible by f
    pad_h = (-H) % f
    pad_w = (-W) % f
    if pad_h or pad_w:
        proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        proj_p = proj
    
    Hc, Wc = proj_p.shape[:2]
    # reshape to blocks and average over the block axes (anti-aliasing)
    return proj_p.reshape(Hc//f, f, Wc//f, f, A).mean(axis=(1, 3))




def show_intermediate_slices(volume, iteration, algorithm_name):
    """Display intermediate reconstruction results."""
    z, y, x = np.array(volume.shape) // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{algorithm_name} - Iteration {iteration}', fontweight='bold')
    
    axes[0].imshow(volume[z, :, :], cmap='gray'); axes[0].set_title('Axial')
    axes[1].imshow(volume[:, y, :], cmap='gray'); axes[1].set_title('Coronal')
    axes[2].imshow(volume[:, :, x], cmap='gray'); axes[2].set_title('Sagittal')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()



ITERATIVE_ALGORITHMS = {
    'SIRT': {
        'name': 'SIRT',
        'full_name': 'Simultaneous Iterative Reconstruction Technique',
        'function': algs.sirt,
        'description': 'Basic iterative method. Stable and robust but slower convergence.',
        'use_cases': 'General purpose, good for complete data, stable baseline',
        'parameters': [],  # No extra parameters beyond niter, init
        'can_interrupt': True,  # Can be interrupted without quality loss
    },
    
    'OS-SART': {
        'name': 'OS-SART',
        'full_name': 'Ordered Subsets SART',
        'function': algs.ossart,
        'description': 'Faster convergence than SIRT using ordered subsets.',
        'use_cases': 'Quick reconstructions, testing, when speed matters',
        'parameters': ['blocksize'],  # Requires blocksize parameter
        'can_interrupt': True,
    },
    
    'CGLS': {
        'name': 'CGLS',
        'full_name': 'Conjugate Gradient Least Squares',
        'function': algs.cgls,
        'description': 'Fast convergence with conjugate gradient optimization.',
        'use_cases': 'Good convergence rate, well-suited for complete projections',
        'parameters': [],
        'can_interrupt': False,  # Better to run continuously
    },
    
    'ASD-POCS': {
        'name': 'ASD-POCS',
        'full_name': 'Adaptive Steepest Descent POCS with TV',
        'function': algs.asd_pocs,
        'description': 'TV regularization for sparse/incomplete data. Excellent for low-dose.',
        'use_cases': 'Low-dose CT, sparse-view, limited-angle, noise reduction',
        'parameters': ['maxl2err', 'alpha'],  # 'ng' not supported in this TIGRE version
        'can_interrupt': False,  # TV algorithms need continuous execution
    },
    
}


def run_algorithm(algorithm_name, input_data, geo, angles, niter, init, config):
    """Execute iterative algorithm with its required parameters."""

    if algorithm_name not in ITERATIVE_ALGORITHMS:
        available = ', '.join(ITERATIVE_ALGORITHMS.keys())
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {available}")
    
    algorithm_iterative = ITERATIVE_ALGORITHMS[algorithm_name]
    kwargs = {p: config[p] for p in algorithm_iterative['parameters'] if p in config}
    
    return algorithm_iterative['function'](input_data, geo, angles, niter=niter, init=init, **kwargs)


def reconstruct_iterative(input_data, geo, angles, config):
    """
    Run iterative reconstruction with optional progress visualization.
    Similar to FDK but with iteration loop for iterative algorithms.
    """
    algorithm = config['algorithm'].upper()
    n_iter = config['n_iter']
    display_interval = config.get('display_interval', n_iter)
    
    if algorithm not in ITERATIVE_ALGORITHMS:
        available = ', '.join(ITERATIVE_ALGORITHMS.keys())
        raise ValueError(f"Algorithm '{algorithm}' not found. Available: {available}")
    
    algorithm_iterative = ITERATIVE_ALGORITHMS[algorithm]
    print(f"--> Algorithm: {algorithm_iterative['full_name']}")
    
    # Determine if we show intermediate results
    can_interrupt = algorithm_iterative['can_interrupt']
    step = display_interval if can_interrupt else n_iter
    
    
    print(f"--> Running {algorithm} ({n_iter} iterations)...")

    volumes_history = []
    volume = None
    start_time = time.time()

    # Reconstruction loop
    for batch_start in range(0, n_iter, step):
        batch_iter = min(step, n_iter - batch_start)
        volume = run_algorithm(algorithm, input_data, geo, angles, batch_iter, volume, config)
        
        if batch_start + batch_iter < n_iter:
            print(f"    Completed {batch_start + batch_iter}/{n_iter} iterations...")
        
        if can_interrupt or batch_start + batch_iter == n_iter:
            show_intermediate_slices(volume, batch_start + batch_iter, algorithm)
        
        if config.get('show_history_in_napari', False):
            volumes_history.append(volume.copy())
        gc.collect()

    print(f"    ✓ Reconstruction complete in {time.time() - start_time:.1f}s")
    return volume, volumes_history


def main(tiff_folder, configurations, output_folder=None):

    # 1. Load images
    projections = load_images(tiff_folder)

    # 2. Downsample (anti-aliased block-average to reduce memory and computation)
    # You control the downsampling factor f via configurations['downsample']
    # Higher f = lower resolution but faster & less memory. Adjust based on your needs.
    if configurations['downsample'] > 1:
        f = configurations['downsample']  # downsampling factor from config
        # Use block-average instead of stride sampling to avoid aliasing artifacts
        projections = downsample_block_mean_pad(projections, f).astype(np.float32)
        pixel_size = configurations['pixel_size'] * f  # effective pixel size increases by f
    else:
        pixel_size = configurations['pixel_size']

    # 3. Get calibrated shift (adjusted for downsampling)
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / configurations['downsample']
    print(f"--> Using calibrated shift: {calibrated_shift_px:.2f} px (original) -> {shift_val:.2f} px (after downsample {configurations['downsample']}x)")

    # 4. Select I0 ROI for normalization
    sino_raw = generate_collapsed_sinogram(projections)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections.shape[0])
    
    # 5. Normalize projections
    projections_norm = normalize_projections(projections, I0_override=mean_I0)
    
    # Memory cleanup: Free heavy variables before reconstruction
    del projections
    gc.collect()

    # 6. Setup Geometry
    geo, angles = setup_geometry(
        projections_norm.shape, pixel_size,
        configurations['DSD'], configurations['DSO'],
        shift_val, configurations['total_angle'],
        shift_sign=configurations['shift_sign'],
        voxel_ratio=configurations['voxel_ratio']
    )

    print("--> Preparing data for TIGRE...")
    input_data = np.transpose(projections_norm, (2, 0, 1)).astype(np.float32)
    del projections_norm
    gc.collect()
    
    # 7. Iterative Reconstruction
    volume, volumes_history = reconstruct_iterative(input_data, geo, angles, configurations)
    print_volume_info(volume, geo)
    
    # 8. Visualization (Napari)
    print("--> Opening Napari...")
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=(geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]), name='Final_Volume')
    
    if configurations.get('show_history_in_napari', False) and volumes_history:
        for idx, vol_hist in enumerate(volumes_history):
            iter_num = (idx + 1) * configurations.get('display_interval', configurations['n_iter'])
            viewer.add_image(vol_hist, scale=(geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2]), 
                           name=f'Iter_{iter_num}', visible=False)
    
    napari.run()
    
    # 9. Export
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
    

    # Optional: Export in Hounsfield Units
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

    return volume, volumes_history


if __name__ == "__main__":
    CONFIG = {
        # GEOMETRY PARAMETERS
        'pixel_size': 0.05,   
        'DSD': 925,           
        'DSO': (925-32),           
        'total_angle': 2 * np.pi,  
        'downsample': 4,      
        'calibrated_shift_px': 5.12,  # From calibration (in original pixels, downsample=1)
        'shift_sign': 1,
        
        # VOXEL SIZE CONTROL
        # voxel_ratio multiplies the Nyquist voxel size (pixel_size / magnification)
        # • voxel_ratio = 1.0: Optimal resolution matching detector pixels (RECOMMENDED)
        # • voxel_ratio < 1.0 (e.g., 0.5): SMALLER voxels, HIGHER resolution
        #     ✓ Smoother images, less pixelation
        #     ✗ Volume size increases by (1/ratio)³ → more RAM/VRAM needed
        #     ✗ Reconstruction time increases significantly
        # • voxel_ratio > 1.0 (e.g., 2.0): LARGER voxels, LOWER resolution
        #     ✓ Much faster reconstruction (volume reduced by ratio³)
        #     ✓ Lower memory usage
        #     ✗ Loss of spatial resolution and detail
        'voxel_ratio': 1.0,

        # ITERATIVE ALGORITHM SETTINGS
        'algorithm': 'OS-SART',     # Algorithm name
        'n_iter': 2,                # Total number of iterations
        'display_interval': 2,      # Show intermediate results every N iterations (set = n_iter for no intermediate display)

        # ALGORITHM-SPECIFIC PARAMETERS
        'blocksize': 20,        # OS-SART: number of projections per subset
        'maxl2err': 0.20,       # ASD-POCS, B-ASD-POCS-BETA, AWASD-POCS:based maximum L2 error tolerance
        'alpha': 0.002,         # ASD-POCS, B-ASD-POCS-BETA, AWASD-POCS: TV regularization weight
        'ng': 25,               # ASD-POCS: number of gradient descent iterations
        'tviter': 25,           # B-ASD-POCS-BETA, AWASD-POCS: TV minimization iterations
        'beta': 1.0,            # B-ASD-POCS-BETA: beta parameter
        'hyper': 0.01,          # FISTA: regularization parameter
        'lmbda': 10,            # IRN-TV-CGLS: lambda regularization parameter

        # VISUALIZATION & EXPORT
        'show_history_in_napari': False,  # Save and show intermediate volumes in Napari (uses more memory)
        
        # OUTPUT FOLDER (optional)
        # If None, saves to 'reconstructed_volumes' in project root
        # You can specify any absolute path
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Image_reconstruction\reconstructed_volumes_Nift'
    }
    
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\marta_caixa_SiPM'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Mouse_PC'
    
    vol, history = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
