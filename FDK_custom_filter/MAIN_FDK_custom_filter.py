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
from matplotlib.widgets import Slider, Button
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from geometry_reconstruction_Voxel_size import setup_geometry
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_fdk import (
    load_images, generate_collapsed_sinogram, selecionar_roi_I0, get_I0_from_roi,
    normalize_projections, downsample_block_mean_pad, print_volume_info,
)
from export_volumes import export_volume_to_nii, export_volume_HU
from custom_filter_fdk import (
    design_ct_filter,
    filt_len_for_geometry,
    reconstruct_fdk_custom_filter,
    plot_filter_frequency_response,
)

logger = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.2f}s"


def main(tiff_folder: str, configurations: dict, output_folder: str | None = None):
    """Main FDK reconstruction pipeline with support for custom frequency-domain filters.

    Phases:
        1. Data Loading & Preprocessing
        2. Spatial Optimisation (Memory Reduction)
        3. Geometry & Reconstruction  ← custom filter applied here
        4. Post-Processing & Export

    Custom filter configuration
    ---------------------------
    Set one of the following keys in configurations:

    Option A — provide a pre-designed filter array:
        configurations['custom_filter_array'] = H   # np.ndarray 1D (half-spectrum)

    Option B — provide a design callback (recommended):
        def my_filter(H, geo):
            H[int(0.7 * len(H)):] = 0   # modify H here
            return H
        configurations['custom_filter_fn'] = my_filter

    Option C — set 'cutoff_fraction' < 1.0 (quick shortcut):
        configurations['cutoff_fraction'] = 0.85
        No callback needed. Reduces high-frequency noise and ringing.

    If none of the above are set, falls back to standard algs.fdk() with 'filter_type'.

    Args:
        tiff_folder: Path to folder containing raw TIFF projections.
        configurations: Parameter dict. Required keys: voxel_size (μm),
            calibrated_shift_px, shift_sign, total_angle, DSD (mm), DSO (mm),
            downsample, filter_type. Optional: custom_filter_fn, custom_filter_array,
            cutoff_fraction, detector_tilt.
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
        detector_tilt=configurations.get('detector_tilt', 0),
    )

    # Transpose to TIGRE format: (n_angles, H, W)
    input_data = np.transpose(projections_final, (2, 0, 1)).copy()
    del projections_final
    gc.collect()

    # ─── RECONSTRUCTION ───────────────────────────────────────────────────────
    #
    # This pipeline differs from FDK_reduce_memory in that TIGRE's algs.fdk()
    # hardcodes bandwidth d=1 and cannot accept a custom filter array.
    # The workaround: pre-filter the projections here, then call TIGRE's Atb
    # directly for GPU backprojection.
    #
    # Custom filter is resolved in priority order:
    #   1. custom_filter_array  — use as-is (resample if length mismatch)
    #   2. custom_filter_fn     — callback(H_base, geo) returns modified H
    #   3. cutoff_fraction < 1  — simple bandwidth reduction (recommended starting point)
    #   4. No custom config     — falls back to algs.fdk() with filter_type string
    # ─────────────────────────────────────────────────────────────────────────

    custom_filter_fn    = configurations.get('custom_filter_fn')
    custom_filter_array = configurations.get('custom_filter_array')
    cutoff              = configurations.get('cutoff_fraction', 1.0)
    filter_name         = configurations.get('filter_type', 'ram_lak')

    use_custom_pipeline = (
        custom_filter_fn is not None
        or custom_filter_array is not None
        or cutoff != 1.0
    )

    if use_custom_pipeline:
        n = filt_len_for_geometry(geo)
        if custom_filter_array is not None:
            H = np.asarray(custom_filter_array, dtype=np.float32)
            label = 'pre-designed array'
        else:
            H = design_ct_filter(filter_name, n_samples=n, cutoff_fraction=cutoff)
            label = f'{filter_name}, cutoff={cutoff:.2f}'
            if custom_filter_fn is not None:
                H = custom_filter_fn(H, geo)
                label += ' + callback'
        print(f" Custom filter ({label}): length={len(H)}, nonzero={np.count_nonzero(H)}")
        plot_filter_frequency_response(H, geo, title=f'Filter — {label}')
        volume = reconstruct_fdk_custom_filter(input_data, geo, angles, H)
    else:
        # No custom configuration: use standard TIGRE FDK
        volume = algs.fdk(input_data, geo, angles, filter=filter_name)

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

    print("RECONSTRUCTION PIPELINE COMPLETED")
    return volume


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    CONFIG = {
        'voxel_size': 25,               # μm
        'calibrated_shift_px': 19.5,
        'shift_sign': 1,
        'total_angle': 2 * np.pi,
        'DSD': 457,                     # mm — distance source-to-detector
        'DSO': 224,                     # mm — distance source-to-origin
        'downsample': 1,
        'filter_type': 'hann',          # Options: 'ram_lak', 'shepp_logan', 'cosine', 'hamming', 'hann'
                                        # 'hann' is more aggressive — reduces streaking/ringing
        'cutoff_fraction': 0.60,        # Passband as fraction of Nyquist. 1.0 = TIGRE default.
                                        # 0.60 = stronger high-frequency suppression (less noise, lower resolution)
        'detector_tilt': 0,
        'output_folder_NiFT': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift',
        'filtered_volumes_folder': r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\Filtered_volumes.Nift',

        # ── CUSTOM FILTER CONFIGURATION ──────────────────────────────────────
        # Option B (recommended): callback executed inside main() once geometry
        # is available. Receives (H_base, geo) and returns modified H.
        #
        # Example 1 — band cutoff at 70% Nyquist with Hann window:
        #   def my_filter(H, geo):
        #       cutoff = int(0.7 * len(H))
        #       window = np.hanning(2 * (len(H) - cutoff))[:len(H) - cutoff]
        #       H[cutoff:] *= window[::-1]
        #       return H
        #
        # Example 2 — hard cutoff at 50% and manual notch:
        #   def my_filter(H, geo):
        #       H[int(0.5 * len(H)):] = 0    # cut above 50% Nyquist
        #       H[10:15] = 0                  # notch at low frequencies
        #       return H
        #
        # Place your function below and set 'custom_filter_fn':
        # ─────────────────────────────────────────────────────────────────────
        'custom_filter_fn': None,       # set to a callable(H, geo) -> np.ndarray
        'custom_filter_array': None,    # or set to a pre-designed np.ndarray 1D
    }

    # ── DEFINE YOUR FILTER HERE ───────────────────────────────────────────────
    #
    # Uncomment and modify one of the examples above.
    # H: float32 array, half-spectrum (index 0 = DC, index -1 = Nyquist)
    # geo: TIGRE geometry (use geo.dDetector[1] to convert indices to cycles/mm)
    #
    # def my_filter(H, geo):
    #     # Example: hard cutoff at 70% Nyquist
    #     H[int(0.7 * len(H)):] = 0
    #     return H
    #
    # CONFIG['custom_filter_fn'] = my_filter
    # ─────────────────────────────────────────────────────────────────────────

    folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_nivel_2'
    vol = main(folder, CONFIG, output_folder=CONFIG.get('output_folder_NiFT'))
