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
from data_processing_FDK_3D import load_images, generate_collapsed_sinogram, selecionar_roi_I0, get_I0_from_roi
from export_volumes import export_volume_to_nii, export_volume_HU


# Import HU conversion function
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from HU_conversion import HU_conversion


# OS ALGORITHM PRESETS
# These presets provide default configurations for commonly used Ordered Subset (OS) iterative algorithms
OS_ALGORITHM_PRESETS = {
    'ossart': {
        'name': 'OSSART',
        'description': 'Ordered Subset Simultaneous Algebraic Reconstruction Technique',
        'function': algs.ossart,
        'default_params': {
            'niter': 50,           # Number of iterations
            'blocksize': 20,       # Number of projections per subset
            'lmbda': 1.0,          # Relaxation parameter
            'lmbda_red': 0.99,     # Reduction of lambda per iteration
        }
    },
    'ossart_tv': {
        'name': 'OSSART-TV',
        'description': 'OSSART with Total Variation regularization',
        'function': algs.ossart_tv,
        'default_params': {
            'niter': 50,
            'blocksize': 20,
            'lmbda': 1.0,
            'lmbda_red': 0.99,
            'alpha': 0.002,        # TV regularization weight
            'alpha_red': 0.95,     # Reduction of alpha per iteration
            'ng': 25,              # Number of iterations in TV minimization
        }
    },
    'sirt': {
        'name': 'SIRT',
        'description': 'Simultaneous Iterative Reconstruction Technique',
        'function': algs.sirt,
        'default_params': {
            'niter': 100,
        }
    },
    'cgls': {
        'name': 'CGLS',
        'description': 'Conjugate Gradient Least Squares',
        'function': algs.cgls,
        'default_params': {
            'niter': 50,
        }
    },
    'sart': {
        'name': 'SART',
        'description': 'Simultaneous Algebraic Reconstruction Technique',
        'function': algs.sart,
        'default_params': {
            'niter': 50,
            'lmbda': 1.0,
            'lmbda_red': 0.99,
        }
    },
    'os_asd_pocs': {
        'name': 'OS-ASD-POCS',
        'description': 'Ordered Subset Adaptive Steepest Descent POCS',
        'function': algs.os_asd_pocs,
        'default_params': {
            'niter': 50,
            'blocksize': 20,
            'alpha': 0.002,
            'alpha_red': 0.95,
            'ng': 20,
            'epsilon': 0.0,
        }
    },
    'os_awasd_pocs': {
        'name': 'OS-AwASD-POCS',
        'description': 'Ordered Subset Adaptive-weighted ASD POCS',
        'function': algs.os_awasd_pocs,
        'default_params': {
            'niter': 50,
            'blocksize': 20,
            'alpha': 0.002,
            'alpha_red': 0.95,
            'ng': 20,
            'delta': -0.005,
        }
    },
}


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
    print(f"  - Voxel size: [{geo.dVoxel[0]:.3f}, {geo.dVoxel[1]:.3f}, {geo.dVoxel[2]:.3f}] mm")
    print(f"  - Number of voxels: {geo.nVoxel}")
    print(f"  - Physical dimensions: [{geo.sVoxel[0]:.3f}, {geo.sVoxel[1]:.3f}, {geo.sVoxel[2]:.3f}] mm")
    print("="*60 + "\n")


def normalize_projections(projections_raw, I0_override=None):
    """
    Normalizes projections using -log(I/I0).
    """
    print("--> Normalizing projections...")
    print(f"    Input shape: {projections_raw.shape}")
    print(f"    Memory size: {projections_raw.nbytes / 1e6:.1f} MB")

    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 1))
    else:
        I0 = float(I0_override)

    # Convert to float32 for calculations
    projections_raw = projections_raw.astype(np.float32)
    ratio = projections_raw / (I0 + 1e-6)  # Avoid division by zero
    ratio = np.clip(ratio, 1e-6, 1.2)  # Prevent log(0) and extreme values
    
    # Beer-Lambert law: -log(I/I0)
    projections_norm = -np.log(ratio)
    
    # Remove negative values (artifacts)
    projections_norm[projections_norm < 0] = 0
    
    print(f"    ✓ Normalization complete")
    return projections_norm


def downsample_block_mean_pad(proj, f):
    """
    Downsample projections by factor f using block-average with edge padding.
    
    This anti-aliased downsampling averages f×f pixel blocks instead of picking one pixel.
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
    # reshape to blocks and average over the block axes
    return proj_p.reshape(Hc//f, f, Wc//f, f, A).mean(axis=(1, 3))


def print_available_algorithms():
    """Print available OS algorithms and their descriptions."""
    print("\n" + "="*70)
    print("AVAILABLE ITERATIVE ALGORITHMS")
    print("="*70)
    for key, preset in OS_ALGORITHM_PRESETS.items():
        print(f"\n{key}:")
        print(f"  Name: {preset['name']}")
        print(f"  Description: {preset['description']}")
        print(f"  Default parameters:")
        for param, value in preset['default_params'].items():
            print(f"    - {param}: {value}")
    print("="*70 + "\n")


def main(tiff_folder, configurations, output_folder=None):
    """
    Main iterative reconstruction pipeline with OS algorithm support.
    
    This script supports iterative reconstruction algorithms with Ordered Subsets (OS),
    which are generally slower than FDK but can produce better quality reconstructions,
    especially when dealing with:
    - Limited angle reconstructions
    - Sparse data
    - Noisy projections
    - Need for artifact reduction
    
    PHASES:
    1. Data Loading & Preprocessing
    2. Normalization (no cropping in this version for simplicity)
    3. Geometry & Iterative Reconstruction
    4. Post-Processing & Export
    """
    
    # Get algorithm configuration
    algorithm_key = configurations.get('algorithm', 'ossart')
    
    if algorithm_key not in OS_ALGORITHM_PRESETS:
        print(f"ERROR: Algorithm '{algorithm_key}' not found in presets!")
        print_available_algorithms()
        return None
    
    preset = OS_ALGORITHM_PRESETS[algorithm_key]
    algorithm_func = preset['function']
    
    # Merge default parameters with user overrides
    algorithm_params = preset['default_params'].copy()
    if 'algorithm_params' in configurations:
        algorithm_params.update(configurations['algorithm_params'])
    
    print("\n" + "="*70)
    print(f"ITERATIVE RECONSTRUCTION: {preset['name']}")
    print("="*70)
    print(f"Description: {preset['description']}")
    print(f"Parameters:")
    for param, value in algorithm_params.items():
        print(f"  - {param}: {value}")
    print("="*70 + "\n")
    
    
    # PHASE 1: DATA LOADING & PREPROCESSING
    print("\n" + "="*70)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*70)
    
    projections_raw = load_images(tiff_folder)
    
    # Generate collapsed sinogram for I0 reference selection
    sino_raw = generate_collapsed_sinogram(projections_raw)
    
    # User selects background ROI and calculates mean I0 value
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    
    del sino_raw
    gc.collect()
    
    
    # PHASE 2: NORMALIZATION & DOWNSAMPLING
    print("\n" + "="*70)
    print("PHASE 2: NORMALIZATION & DOWNSAMPLING")
    print("="*70)
    
    projections_norm = normalize_projections(projections_raw, I0_override=mean_I0)
    
    del projections_raw
    gc.collect()
    
    # Optional downsampling
    f = configurations.get('downsample', 1)
    
    if f > 1:
        print(f"Downsampling by factor {f}x...")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32)
        print(f"      Final shape: {projections_final.shape}")
        del projections_norm
        gc.collect()
    else:
        projections_final = projections_norm
    
    
    # PHASE 3: GEOMETRY SETUP & ITERATIVE RECONSTRUCTION
    print("\n" + "="*70)
    print(f"PHASE 3: GEOMETRY SETUP & {preset['name']} RECONSTRUCTION")
    print("="*70)
    
    # Calculate detector shift
    calibrated_shift_px = configurations.get('calibrated_shift_px', 0.0)
    shift_val = calibrated_shift_px / f
    print(f"Detector shift: {shift_val:.3f} pixels")
    
    # Adjust voxel size for downsampling
    effective_voxel_size = configurations['voxel_size'] * f
    print(f"Voxel size adjustment: {configurations['voxel_size']:.2f} μm × {f} = {effective_voxel_size:.2f} μm")
    
    # Setup TIGRE geometry
    geo, angles = setup_geometry(
        projections_final.shape, 
        effective_voxel_size,
        configurations['DSD'], 
        configurations['DSO'], 
        shift_val, 
        configurations['total_angle'], 
        shift_sign=configurations.get('shift_sign', 1),
        downsample_factor=f,
        crop_params=None  # No cropping in this version
    )
    
    print(f"Detector size: {geo.nDetector}")
    print(f"Voxel size: {geo.dVoxel}")
    
    # Prepare data for TIGRE
    print(f"Preparing data for TIGRE...")
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    
    del projections_final
    gc.collect()
    
    # EXECUTE ITERATIVE RECONSTRUCTION
    print(f"\nRunning {preset['name']} algorithm...")
    print(f"This may take several minutes depending on parameters...")
    
    volume = algorithm_func(input_data, geo, angles, **algorithm_params)
    print_volume_info(volume, geo)
    
    
    # PHASE 4: POST-PROCESSING & EXPORT
    print("\n" + "="*70)
    print("PHASE 4: POST-PROCESSING & EXPORT")
    print("="*70)
    
    # Optional NIfTI export
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print(f"Volume exported to NIfTI format")
    else:
        print(f"NIfTI export skipped")
    
    # Open Napari to visualize
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
    
    # Optional HU conversion
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
    # Print available algorithms
    print_available_algorithms()
    
    # Example configuration for OSSART
    CONFIG = {
        # Algorithm selection
        'algorithm': 'ossart',  # Choose from: ossart, ossart_tv, sirt, cgls, sart, os_asd_pocs, os_awasd_pocs
        
        # Algorithm parameters (optional - will use defaults if not specified)
        'algorithm_params': {
            'niter': 50,        # Number of iterations
            'blocksize': 20,    # Projections per subset (only for OS algorithms)
            'lmbda': 1.0,       # Relaxation parameter
            'lmbda_red': 0.99,  # Lambda reduction per iteration
        },
        
        # Geometry parameters
        'voxel_size': 22,       # μm - desired voxel resolution
        'DSD': 457,             # Distance Source to Detector (mm)
        'DSO': 211,             # Distance Source to Object (mm)
        'total_angle': 2 * np.pi,  # Full 360° rotation
        
        # Calibration
        'calibrated_shift_px': 5.12,  # Detector center offset (pixels)
        'shift_sign': 1,
        
        # Preprocessing
        'downsample': 2,        # Downsampling factor (1 = no downsampling)
        
        # Output
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Image_reconstruction\reconstructed_volumes_Nift'
    }
    
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
