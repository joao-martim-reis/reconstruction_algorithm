"""
Example configurations for using different OS iterative algorithms.

This file demonstrates various use cases and parameter configurations
for the iterative reconstruction algorithms available in MAIN_TIGRE_iterative.py
"""

import numpy as np


# ============================================================================
# EXAMPLE 1: Fast Preview with CGLS
# ============================================================================
# Use this for quick testing and parameter tuning
# CGLS converges faster than OSSART and is good for initial exploration

CONFIG_FAST_PREVIEW = {
    'algorithm': 'cgls',
    
    'algorithm_params': {
        'niter': 20,  # Few iterations for speed
    },
    
    # Geometry (example values - adjust to your scanner)
    'voxel_size': 50,       # μm - larger voxels for speed
    'DSD': 457,             # mm
    'DSO': 211,             # mm
    'total_angle': 2 * np.pi,
    
    # Calibration
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    # Heavy downsampling for maximum speed
    'downsample': 4,
    
    # Output
    'output_folder_NiFT': './output'
}


# ============================================================================
# EXAMPLE 2: Standard Quality OSSART
# ============================================================================
# Balanced approach for general-purpose reconstruction
# Good starting point for most applications

CONFIG_STANDARD_OSSART = {
    'algorithm': 'ossart',
    
    'algorithm_params': {
        'niter': 50,
        'blocksize': 20,
        'lmbda': 1.0,
        'lmbda_red': 0.99,
    },
    
    # Geometry
    'voxel_size': 22,       # μm
    'DSD': 457,
    'DSO': 211,
    'total_angle': 2 * np.pi,
    
    # Calibration
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    # Moderate downsampling
    'downsample': 2,
    
    'output_folder_NiFT': './output'
}


# ============================================================================
# EXAMPLE 3: High Quality with TV Regularization
# ============================================================================
# Use for noisy data where you want edge preservation
# TV (Total Variation) reduces noise while maintaining sharp edges

CONFIG_HIGH_QUALITY_TV = {
    'algorithm': 'ossart_tv',
    
    'algorithm_params': {
        'niter': 100,           # More iterations for convergence
        'blocksize': 20,
        'lmbda': 1.0,
        'lmbda_red': 0.99,
        'alpha': 0.003,         # TV regularization weight
        'alpha_red': 0.95,      # Gradually reduce TV weight
        'ng': 30,               # TV minimization steps
    },
    
    # Geometry - no downsampling for maximum quality
    'voxel_size': 22,
    'DSD': 457,
    'DSO': 211,
    'total_angle': 2 * np.pi,
    
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    'downsample': 1,  # No downsampling
    
    'output_folder_NiFT': './output'
}


# ============================================================================
# EXAMPLE 4: Limited Angle Reconstruction
# ============================================================================
# For incomplete angular coverage (e.g., 180° or less)
# Uses OS-ASD-POCS which is designed for limited angle problems

CONFIG_LIMITED_ANGLE = {
    'algorithm': 'os_asd_pocs',
    
    'algorithm_params': {
        'niter': 150,           # More iterations needed
        'blocksize': 15,        # Smaller subsets
        'alpha': 0.005,         # Stronger regularization
        'alpha_red': 0.95,
        'ng': 25,
        'epsilon': 0.0,
    },
    
    # Geometry with limited angle
    'voxel_size': 22,
    'DSD': 457,
    'DSO': 211,
    'total_angle': np.pi,  # Only 180° scan instead of 360°
    
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    'downsample': 2,
    
    'output_folder_NiFT': './output'
}


# ============================================================================
# EXAMPLE 5: Sparse View Reconstruction
# ============================================================================
# When you have very few projections (e.g., 100 instead of 720)
# Uses adaptive weighting for better handling of sparse data

CONFIG_SPARSE_VIEW = {
    'algorithm': 'os_awasd_pocs',
    
    'algorithm_params': {
        'niter': 200,           # Many iterations for convergence
        'blocksize': 10,        # Small subsets for sparse data
        'alpha': 0.008,         # Strong regularization
        'alpha_red': 0.95,
        'ng': 30,
        'delta': -0.005,        # Adaptive weighting parameter
    },
    
    # Geometry
    'voxel_size': 25,           # Slightly lower resolution acceptable
    'DSD': 457,
    'DSO': 211,
    'total_angle': 2 * np.pi,
    
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    'downsample': 2,
    
    'output_folder_NiFT': './output'
}


# ============================================================================
# EXAMPLE 6: Simple SIRT (No Subsets)
# ============================================================================
# Classic iterative algorithm without ordered subsets
# Slower convergence but very stable

CONFIG_SIMPLE_SIRT = {
    'algorithm': 'sirt',
    
    'algorithm_params': {
        'niter': 100,  # SIRT needs more iterations than OS methods
    },
    
    # Geometry
    'voxel_size': 22,
    'DSD': 457,
    'DSO': 211,
    'total_angle': 2 * np.pi,
    
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    'downsample': 2,
    
    'output_folder_NiFT': './output'
}


# ============================================================================
# EXAMPLE 7: Automated Batch Processing
# ============================================================================
# Use for scripted/automated workflows without user interaction
# All prompts are bypassed using config parameters

CONFIG_AUTOMATED = {
    'algorithm': 'cgls',
    
    'algorithm_params': {
        'niter': 30,
    },
    
    # Geometry
    'voxel_size': 25,
    'DSD': 457,
    'DSO': 211,
    'total_angle': 2 * np.pi,
    
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    'downsample': 2,
    
    # Automated output control (no user prompts)
    'auto_export': True,        # Automatically export without prompting
    'show_napari': False,       # Skip visualization for batch processing
    'hu_conversion': {          # Automatic HU conversion with preset values
        'water_value': 0.020,
        'air_value': -0.001
    },
    
    'output_folder_NiFT': './output'
}


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ITERATIVE RECONSTRUCTION - EXAMPLE CONFIGURATIONS")
    print("=" * 70)
    print("\nAvailable example configurations:")
    print("  1. CONFIG_FAST_PREVIEW       - Quick CGLS test (20 iterations)")
    print("  2. CONFIG_STANDARD_OSSART    - Balanced quality/speed")
    print("  3. CONFIG_HIGH_QUALITY_TV    - Best quality with TV regularization")
    print("  4. CONFIG_LIMITED_ANGLE      - For 180° or limited angle scans")
    print("  5. CONFIG_SPARSE_VIEW        - For very few projections")
    print("  6. CONFIG_SIMPLE_SIRT        - Classic SIRT without subsets")
    print("  7. CONFIG_AUTOMATED          - Batch processing without user prompts")
    print("\nTo use any configuration:")
    print("  1. Copy the desired CONFIG to your script")
    print("  2. Adjust geometry parameters (DSD, DSO) to match your scanner")
    print("  3. Set your data folder path")
    print("  4. Run: volume = main(folder, CONFIG)")
    print("\nExample:")
    print("  from MAIN_TIGRE_iterative import main")
    print("  folder = './my_projections'")
    print("  volume = main(folder, CONFIG_FAST_PREVIEW)")
    print("\nFor automated/batch processing:")
    print("  Use CONFIG_AUTOMATED as template")
    print("  Set 'auto_export': True, 'show_napari': False")
    print("  Optionally add 'hu_conversion' dict for automatic HU export")
    print("=" * 70)
