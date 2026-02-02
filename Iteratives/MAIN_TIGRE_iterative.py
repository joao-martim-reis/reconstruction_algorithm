"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MAIN SCRIPT - ITERATIVE RECONSTRUCTION                  ║
║                                                                              ║
║  Consolidated script for all iterative algorithms (basic and TV-regularized) ║
║  Algorithm parameters are loaded automatically from iterative_parameters.py  ║
╚══════════════════════════════════════════════════════════════════════════════╝


1. Choose the algorithm in the "ALGORITHM SELECTION" section
2. Configure geometry and paths in the "CONFIGURATION" section
3. Select the dataset in the "SELECT YOUR DATASET" section
4. Run the script
"""

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
from napari_filters import interactive_filter_viewer

# Import iterative parameters
from iterative_parameters import (
    get_algorithm_config, 
    list_available_algorithms,
    print_algorithm_info
)

# Import HU conversion function
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from HU_conversion import HU_conversion



def print_volume_info(volume, geo=None):
    """Prints detailed information about the reconstructed volume."""
    print("\n" + "="*60)
    print("RECONSTRUCTED VOLUME INFORMATION")
    print("="*60)
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}")
    print("="*60 + "\n") 
    if geo:
        print(f"\nGeometric information:")
        print(f"  - Voxel size: [{geo.dVoxel[0]:.3f}, {geo.dVoxel[1]:.3f}, {geo.dVoxel[2]:.3f}] mm")
        print(f"  - Number of voxels: {geo.nVoxel}")
        print(f"  - Physical dimensions: [{geo.sVoxel[0]:.3f}, {geo.sVoxel[1]:.3f}, {geo.sVoxel[2]:.3f}] mm")
        print("="*60 + "\n")


def normalize_projections(projections_raw, I0_override=None):
    """Normalizes projections using -log(I/I0)."""
    print("--> Normalizing cropped projections...")


    if I0_override is None:
        I0 = float(np.percentile(projections_raw, 1))
    else:
        I0 = float(I0_override)

    projections_raw = projections_raw.astype(np.float32)
    ratio = projections_raw / (I0 + 1e-6)
    ratio = np.clip(ratio, 1e-6, 1.2)
    
    projections_norm = -np.log(ratio)
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



def run_reconstruction(input_data, geo, angles, algorithm_name, config):
  
    # Map algorithm names to TIGRE functions
    ALGORITHM_MAP = {
        # Basic algorithms
        'SIRT': algs.sirt,
        'CGLS': algs.cgls,
        'LSQR': algs.lsqr,
        'LSMR': algs.lsmr,
        'OSSART': algs.ossart,
        'SART': algs.sart,
        # TV-regularized algorithms
        'OSSART_TV': algs.ossart_tv,
        'SART_TV': algs.sart_tv,
        'ASD_POCS': algs.asd_pocs,
        'AWASD_POCS': algs.awasd_pocs,
    }
    
    if algorithm_name not in ALGORITHM_MAP:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}\n"
            f"Available: {list(ALGORITHM_MAP.keys())}"
        )
    
    alg_function = ALGORITHM_MAP[algorithm_name]
    iterations = config['iterations']
    category = config['category']
    
    # Print header
    if category == 'tv':
        print(f"RUNNING {algorithm_name.upper()} RECONSTRUCTION (TV-REGULARIZED)")
    else:
        print(f"RUNNING {algorithm_name.upper()} RECONSTRUCTION")
    print(f"  Iterations: {iterations}")

    # Prepare algorithm parameters
    algo_params = {}
    
    # Transforms parameters defined in the dictionary to function arguments
    if 'params' in config and config['params']:
        params = config['params']
        
        if 'blocksize' in params:
            algo_params['blocksize'] = params['blocksize']

        if 'tv_lambda' in params:
            algo_params['lmbda'] = params['tv_lambda']
        
        if 'tv_ng' in params:
            algo_params['ng'] = params['tv_ng']

        if 'asd_alpha' in params:
            algo_params['alpha'] = params['asd_alpha']
        
        if 'asd_epsilon' in params:
            algo_params['epsilon'] = params['asd_epsilon']
    
    # Run reconstruction
    print(f"\n  Starting reconstruction...")
    
    volume = alg_function(input_data, geo, angles, iterations, **algo_params)
    
    print(f"  ✓ Reconstruction complete!")
    return volume



def main(tiff_folder, configurations, output_folder=None):
    """
    Main unified iterative reconstruction pipeline.
    
    Supports ALL iterative algorithms:
    - Basic: SIRT, CGLS, LSQR, LSMR, OSSART, SART
    - TV: OSSART_TV, SART_TV, ASD_POCS, AWASD_POCS
    
    The algorithm and its parameters are automatically loaded from
    iterative_parameters.py based on the 'algorithm_config' in configurations.
    """
    
    # Get algorithm configuration
    algo_config = configurations['algorithm_config']
    algorithm_name = algo_config['algorithm_name']
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Iterations: {algo_config['iterations']}")
    if algo_config['params']:
        print(f"Parameters: {algo_config['params']}")

    
    

    print("PHASE 1: DATA LOADING & PREPROCESSING")
    
    projections_raw = load_images(tiff_folder)
    
    # Safety check
    if projections_raw is None:
        raise ValueError(f"Failed to load projections from folder: {tiff_folder}")
    
    sino_raw = generate_collapsed_sinogram(projections_raw)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    
    del sino_raw
    gc.collect()


    print("PHASE 2: SPATIAL OPTIMIZATION")

    
    first_proj_raw = projections_raw[:, :, 0]
    crop_params = select_crop_region(first_proj_raw)
    projections_cropped_raw = apply_crop_to_projections(projections_raw, crop_params)
    
    del projections_raw
    gc.collect()
    
    projections_norm = normalize_projections(projections_cropped_raw, I0_override=mean_I0)
    
    del projections_cropped_raw
    gc.collect()
    
    # Downsampling
    f = configurations['downsample']
    
    if f > 1:
        print(f"Downsampling by factor {f}x...")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32)
        print(f"      Final shape: {projections_final.shape}")
        del projections_norm
        gc.collect()
    else:
        projections_final = projections_norm
    
    

    print("PHASE 3: GEOMETRY SETUP & ITERATIVE RECONSTRUCTION")

    
    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / f
    print(f" Detector shift calculated: {shift_val:.3f} pixels")
    
    effective_voxel_size = configurations['voxel_size'] * f
    print(f" Voxel size adjustment: {configurations['voxel_size']:.2f} μm × {f} = {effective_voxel_size:.2f} μm")
    
    geo, angles = setup_geometry(
        projections_final.shape, 
        effective_voxel_size,
        configurations['DSD'], 
        configurations['DSO'], 
        shift_val, 
        configurations['total_angle'], 
        shift_sign=configurations['shift_sign'],
        downsample_factor=configurations['downsample'],
        crop_params=crop_params
    )
    

    print(f"GEOMETRY VALIDATION")
    print(f"Detector dimensions (nDetector): {geo.nDetector}")
    print(f"  - Type: {type(geo.nDetector)}")
    print(f"  - Shape: {geo.nDetector.shape}")
    print(f"Detector pixel size (dDetector): {geo.dDetector}")
    print(f"Voxel size (dVoxel): {geo.dVoxel}")
    

    print(f"DATA PREPARATION")
    print(f"Input shape: {projections_final.shape} (H, W, N_angles)")
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    print(f"After transpose: {input_data.shape} (N_angles, H, W)")
    print(f"Expected by TIGRE: ({len(angles)}, {geo.nDetector[0]}, {geo.nDetector[1]})")
    
    del projections_final
    gc.collect()
    
    
    volume = run_reconstruction(input_data, geo, angles, algorithm_name, algo_config)
    print_volume_info(volume, geo)
    
    

    print("\n" + "="*70)
    print("PHASE 4: POST-PROCESSING & EXPORT")
    print("="*70)
    
    # Step 4.1: NAPARI visualization
    print(f"Opening Napari viewer...")
    voxel_scale = (geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2])
    
    # Interactive filtering (commented out for now)
    # filter_choice = input("\nDo you want to use INTERACTIVE filtering? (y/n): ").strip().lower()
    # if filter_choice == 'y':
    #     filtered_folder = configurations.get('filtered_volumes_folder')
    #     viewer, final_filter_params = interactive_filter_viewer(
    #         volume, 
    #         name=f"{algorithm_name} Volume", 
    #         scale=voxel_scale, 
    #         nii_filepath=None, 
    #         filtered_output_folder=filtered_folder
    #     )
    #     napari.run()
    #     from napari_filters import print_applied_filters_summary
    #     print_applied_filters_summary(final_filter_params)
    # else:
    
    # View reconstructed 3D volume
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=voxel_scale, name=f"{algorithm_name} Volume")
    napari.run()
    
    # Step 4.2: Optional NIfTI export
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print(f"Volume exported to NIfTI format")
    else:
        print(f"NIfTI export skipped")
    
    
    # Step 4.3: Optional HU conversion
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
            print(f"HU conversion skipped")
    
    
    print("\n" + "="*70)
    print("RECONSTRUCTION PIPELINE COMPLETED")
    print("="*70 + "\n")
    
    return volume



if __name__ == "__main__":
    

    CONFIG = {
        # PRIMARY INPUT: Desired voxel size
        'voxel_size': 26,  # μm
        
        # Geometry parameters
        'DSD': 457,  # mm - Source to Detector Distance
        'DSO': 235,  # mm - Source to Object Distance
        
        # Downsampling
        'downsample': 1,  # Factor for downsampling (1 = no downsampling)
        
        # Acquisition parameters
        'total_angle': 2 * np.pi,  # Total rotation angle (radians)
        'calibrated_shift_px': 15.6,  # Detector shift (pixels)
        'shift_sign': 1,  # Sign of the shift (+1 or -1)
        
        # Output folders
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift',
        'filtered_volumes_folder': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\Filtered_volumes.Nift'
    }
    

    #Choose algorithm directly
    
    # BASIC algorithms (no TV):
    #algorithm_config = get_algorithm_config('SIRT')       # Classic, balanced
    #algorithm_config = get_algorithm_config('CGLS')       # Fast, good for details
    #algorithm_config = get_algorithm_config('LSQR')       # Numerically stable
    #algorithm_config = get_algorithm_config('LSMR')       # Improved over LSQR
    #algorithm_config = get_algorithm_config('OSSART')     # Very fast (preview)
    #algorithm_config = get_algorithm_config('SART')       # Alternative to SIRT
    
    
    # TV-regularized algorithms (reduce artifacts):
    #algorithm_config = get_algorithm_config('OSSART_TV')  # RECOMMENDED for metal
    algorithm_config = get_algorithm_config('SART_TV')    # More precise than OSSART_TV
    #algorithm_config = get_algorithm_config('ASD_POCS')   # Severe artifacts
    #algorithm_config = get_algorithm_config('AWASD_POCS') # Adaptive variant
    
    
    # Preset configurations removed — select algorithms via `get_algorithm_config()`
    
    
    # Determine algorithm name and add to CONFIG
    algorithm_name = algorithm_config.get('algorithm_name') or algorithm_config.get('algorithm')
    if algorithm_name is None:
        raise ValueError("Could not determine algorithm name from configuration")

    algorithm_config['algorithm_name'] = algorithm_name
    CONFIG['algorithm_config'] = algorithm_config
    
    
    # HELP FUNCTIONS (uncomment to view information)
    #list_available_algorithms()
    #list_presets()
    
    # Print detailed information about an algorithm:
    #print_algorithm_info('OSSART_TV')
    

    # SELECT DATASET
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\45kv+0.45mA\Fantoma_agua_destilada'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_v3' #bar pattern
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+DOD_246\PMMA+haste'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_centrado'
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\test3' 


    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
