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
    get_preset_config, 
    list_available_algorithms,
    list_presets,
    print_algorithm_info
)

# Import HU conversion function
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from HU_conversion import HU_conversion


# ═══════════════════════════════════════════════════════════════════════════
#                            UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

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
    """Downsample projections by factor f using block averaging."""
    H, W, A = proj.shape
    
    pad_h = (-H) % f
    pad_w = (-W) % f
    
    if pad_h or pad_w:
        proj_p = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        proj_p = proj
    
    Hc, Wc = proj_p.shape[:2]
    return proj_p.reshape(Hc//f, f, Wc//f, f, A).mean(axis=(1, 3))


# ═══════════════════════════════════════════════════════════════════════════
#                       RECONSTRUCTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_reconstruction(input_data, geo, angles, algorithm_name, config):
    """
    Args:
        input_data: Projection data (angles × height × width)
        geo: TIGRE geometry object
        angles: Array of projection angles
        algorithm_name: Name of algorithm (e.g., 'SIRT', 'OSSART_TV')
        config: Algorithm configuration dictionary from iterative_parameters.py
    
    Returns:
        Reconstructed volume
    """
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


# ═══════════════════════════════════════════════════════════════════════════
#                           MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

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

    
    
    # ===================================================================
    # PHASE 1: DATA LOADING & PREPROCESSING
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*70)
    
    projections_raw = load_images(tiff_folder)
    sino_raw = generate_collapsed_sinogram(projections_raw)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    
    del sino_raw
    gc.collect()
    
    
    # ===================================================================
    # PHASE 2: SPATIAL OPTIMIZATION (MEMORY REDUCTION)
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 2: SPATIAL OPTIMIZATION")
    print("="*70)
    
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
    
    
    # ===================================================================
    # PHASE 3: GEOMETRY SETUP & RECONSTRUCTION
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 3: GEOMETRY SETUP & ITERATIVE RECONSTRUCTION")
    print("="*70)
    
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
    
    print(f"Detector size: {geo.nDetector}")
    print(f"Voxel size: {geo.dVoxel}")
    
    # Prepare data for TIGRE
    print(f"Preparing data for TIGRE...")
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    
    del projections_final
    gc.collect()
    
    
    # ===================================================================
    # ITERATIVE RECONSTRUCTION
    # ===================================================================
    volume = run_reconstruction(input_data, geo, angles, algorithm_name, algo_config)
    
    print_volume_info(volume, geo)
    
    
    # ===================================================================
    # PHASE 4: POST-PROCESSING & EXPORT
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 4: POST-PROCESSING & EXPORT")
    print("="*70)
    
    # NIfTI export
    export_choice = input("\nDo you want to export the volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print(f"Volume exported to NIfTI format")
    else:
        print(f"NIfTI export skipped")
    
    
    # Napari visualization
    print(f"Opening Napari viewer...")
    voxel_scale = (geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2])
    
    filter_choice = input("\nDo you want to use INTERACTIVE filtering? (y/n): ").strip().lower()
    
    if filter_choice == 'y':
        filtered_folder = configurations.get('filtered_volumes_folder')
        viewer, final_filter_params = interactive_filter_viewer(
            volume, 
            name=f"{algorithm_name} Volume", 
            scale=voxel_scale, 
            nii_filepath=nii_filepath, 
            filtered_output_folder=filtered_folder
        )
        napari.run()
        
        from napari_filters import print_applied_filters_summary
        print_applied_filters_summary(final_filter_params)
    else:
        viewer = napari.Viewer()
        viewer.add_image(volume, scale=voxel_scale, name=f"{algorithm_name} Volume")
        napari.run()
    
    
    # HU conversion
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


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ===================================================================
    # CONFIGURATION
    # ===================================================================
    CONFIG = {
        # PRIMARY INPUT: Desired voxel size
        'voxel_size': 26,  # μm
        
        # Geometry parameters
        'DSD': 457,  # mm - Source to Detector Distance
        'DSO': 235,  # mm - Source to Object Distance
        
        # Downsampling
        'downsample': 2,  # Factor for downsampling (1 = no downsampling)
        
        # Acquisition parameters
        'total_angle': 2 * np.pi,  # Total rotation angle (radians)
        'calibrated_shift_px': 5.12,  # Detector shift (pixels)
        'shift_sign': 1,  # Sign of the shift (+1 or -1)
        
        # Output folders
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift',
        'filtered_volumes_folder': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\Filtered_volumes.Nift'
    }
    
    
    # ===================================================================
    # ALGORITHM SELECTION
    # ===================================================================
    # 
    # OPTION 1: Choose algorithm directly
    # ────────────────────────────────────────
    # Uncomment one of the lines below to choose the algorithm:
    
    # BASIC algorithms (no TV):
    #algorithm_config = get_algorithm_config('SIRT')       # Classic, balanced
    #algorithm_config = get_algorithm_config('CGLS')       # Fast, good for details
    #algorithm_config = get_algorithm_config('LSQR')       # Numerically stable
    #algorithm_config = get_algorithm_config('LSMR')       # Improved over LSQR
    #algorithm_config = get_algorithm_config('OSSART')     # Very fast (preview)
    #algorithm_config = get_algorithm_config('SART')       # Alternative to SIRT
    
    # TV-regularized algorithms (reduce artifacts):
    algorithm_config = get_algorithm_config('OSSART_TV')  # RECOMMENDED for metal
    #algorithm_config = get_algorithm_config('SART_TV')    # More precise than OSSART_TV
    #algorithm_config = get_algorithm_config('ASD_POCS')   # Severe artifacts
    #algorithm_config = get_algorithm_config('AWASD_POCS') # Adaptive variant
    
    
    # OPTION 2: Select a PRESET configuration
    # ──────────────────────────────────
    # Uncomment one of the lines below to use a predefined preset:
    
    #algorithm_config = get_preset_config('fantoma_pmma')      # Optimized for PMMA
    #algorithm_config = get_preset_config('fantoma_agua')      # Optimized for water
    #algorithm_config = get_preset_config('padrao_barras')     # Resolution test
    #algorithm_config = get_preset_config('metal_artifacts')   # Remove metal artifacts
    #algorithm_config = get_preset_config('ruido_alto')        # High-noise data
    #algorithm_config = get_preset_config('preview_rapido')    # Quick preview
    #algorithm_config = get_preset_config('maxima_qualidade')  # Maximum quality
    
    
    # Determine algorithm name and add to CONFIG
    from iterative_parameters import ALGORITHM_CONFIGS
    
    # Find algorithm name by matching the config
    algorithm_name = None
    
    # Check if it's a preset (has 'algorithm' key)
    if 'algorithm' in algorithm_config:
        algorithm_name = algorithm_config['algorithm']
    else:
        # It's a direct algorithm config - find matching name
        for name, cfg in ALGORITHM_CONFIGS.items():
            if (cfg['iterations'] == algorithm_config['iterations'] and 
                cfg['category'] == algorithm_config['category'] and 
                cfg['description'] == algorithm_config['description']):
                algorithm_name = name
                break
    
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
    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern'
    #folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+DOD_246\PMMA+haste'
    


    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
