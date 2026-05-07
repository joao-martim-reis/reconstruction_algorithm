import gc
import logging
import os
import sys
import time

import numpy as np
import tigre
import tigre.algorithms as algs
from tigre.utilities import gpu
import matplotlib.pyplot as plt
import napari
import nibabel as nib
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from geometry_reconstruction_Voxel_size import setup_geometry
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_fdk import (
    load_images, generate_collapsed_sinogram, selecionar_roi_I0, get_I0_from_roi,
    normalize_projections, downsample_block_mean_pad, print_volume_info,
)
from export_volumes import export_volume_to_nii, export_volume_HU
from iterative_parameters import get_algorithm_config

logger = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def main(tiff_folder: str, configurations: dict, output_folder: str | None = None):
    """Main iterative reconstruction pipeline (voxel-size-first).

    Args:
        tiff_folder: Path to folder containing raw TIFF projections.
        configurations: Parameter dict. Required keys: voxel_size (μm),
            calibrated_shift_px, shift_sign, total_angle, DSD (mm), DSO (mm),
            downsample, algorithm.
        output_folder: Root directory for NIfTI export.

    Returns:
        volume: Reconstructed volume, shape (nz, ny, nx), float32.
    """
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    load_start = time.perf_counter()

    projections_raw = load_images(tiff_folder)
    sino_raw = generate_collapsed_sinogram(projections_raw)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0 = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    del sino_raw
    gc.collect()

    print("\nPHASE 2: SPATIAL OPTIMISATION (MEMORY REDUCTION)")

    crop_params = select_crop_region(projections_raw[:, :, 0])
    projections_cropped_raw = apply_crop_to_projections(projections_raw, crop_params)
    del projections_raw
    gc.collect()

    projections_cropped = normalize_projections(projections_cropped_raw, I0_override=mean_I0)
    del projections_cropped_raw
    gc.collect()

    f = configurations['downsample']
    if f > 1:
        print(f"Downsampling by factor {f}x — reduces maximum achievable resolution")
        projections_final = downsample_block_mean_pad(projections_cropped, f)
        print(f"Final shape: {projections_final.shape}")
        del projections_cropped
        gc.collect()
    elif f < 1:
        raise ValueError(f"Downsampling factor must be >= 1, got {f}")
    else:
        projections_final = projections_cropped

    logger.info("Load + preprocess completed in %s", _format_elapsed(time.perf_counter() - load_start))

    print("PHASE 3: GEOMETRY SETUP & RECONSTRUCTION (VOXEL-SIZE-FIRST)")

    calibrated_shift_px = configurations['calibrated_shift_px']
    shift_val = calibrated_shift_px / f
    print(f" Detector shift: {shift_val:.3f} px (calibrated {calibrated_shift_px} / factor {f})")

    effective_voxel_size = configurations['voxel_size'] * f
    print(f" Effective voxel size: {configurations['voxel_size']:.2f} μm × {f} = "
          f"{effective_voxel_size:.2f} μm")

    geo, angles = setup_geometry(
        projections_final.shape,
        effective_voxel_size,
        configurations['DSD'],
        configurations['DSO'],
        shift_val,
        configurations['total_angle'],
        shift_sign=configurations['shift_sign'],
        downsample_factor=configurations['downsample'],
        crop_params=crop_params,
    )

    # Transpose to TIGRE format: (n_angles, H, W)
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    del projections_final
    gc.collect()

    algorithm_name = configurations['algorithm']
    algo_config = get_algorithm_config(algorithm_name)

    print(f"ITERATIVE ALGORITHM: {algorithm_name}")
    print(f"  Category: {algo_config['category']}")
    print(f"  Iterations: {algo_config['iterations']}")
    if algo_config['params']:
        print(f"  Parameters: {algo_config['params']}")

    algo_function = getattr(algs, algo_config['function_name'])
    algo_kwargs = {'niter': algo_config['iterations'], **algo_config['params']}

    volume = algo_function(input_data, geo, angles, **algo_kwargs)
    print_volume_info(volume, geo)

    print("PHASE 4: POST-PROCESSING & EXPORT")
    print("Opening Napari viewer...")

    voxel_scale = tuple(geo.dVoxel)
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=voxel_scale, name="CT Volume")
    napari.run()

    export_choice = input("\nExport volume to .nii format? (y/n): ").strip().lower()
    nii_filepath = None
    if export_choice == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print("Volume exported to NIfTI format")
    else:
        print("NIfTI export skipped")

    if export_choice == 'y' and nii_filepath is not None:
        hu_choice = input("\nExport in Hounsfield Units (HU)? (y/n): ").strip().lower()
        if hu_choice == 'y':
            water_val = float(input("Gray-scale value for WATER: "))
            air_val = float(input("Gray-scale value for AIR: "))
            export_volume_HU(nii_filepath, volume, water_val, air_val)
            print("Volume converted to Hounsfield Units and exported")
        else:
            print("HU conversion skipped")

    print("RECONSTRUCTION PIPELINE COMPLETED")
    return volume


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    CONFIG = {
        'voxel_size': 25,           # μm
        'calibrated_shift_px': 0,
        # Alternative shifts tested: 5.12 / 24.5 / 19.5 / 21.5
        'total_angle': 2 * np.pi,
        'shift_sign': 1,
        'DSD': 463,                 # mm — distance source-to-detector
        'DSO': 244,                 # mm — distance source-to-origin
        # Previous geometry: DSD=457, DSO=211
        'downsample': 1,
        'algorithm': 'MLEM',        # Change this to select algorithm
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift',
        'filtered_volumes_folder': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\Filtered_volumes.Nift',
    }

    folder = r'D:\tentativa_Lara_tiff'
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
