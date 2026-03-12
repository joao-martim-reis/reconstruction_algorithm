"""
recon_astra_fdk.py
==================
FDK cone-beam CT reconstruction using the ASTRA Toolbox.

Pipeline:
  Phase 1 – Data loading & I0 selection        ← SHARED with TIGRE / scratch
  Phase 2 – Crop, normalise, downsample         ← SHARED with TIGRE / scratch
  Phase 3 – ASTRA geometry + FDK_CUDA          ← *** ASTRA SPECIFIC ***
  Phase 4 – Napari viewer & NIfTI export        ← SHARED with TIGRE / scratch

Run in the ct_recon conda environment:
  conda activate ct_recon
  python recon_astra_fdk.py

Key difference vs TIGRE sinogram axis order:
  projections array shape:  (n_rows, n_cols, n_angles)
  ASTRA expects:            (n_rows, n_angles, n_cols)  → transpose (0, 2, 1)
  TIGRE expects:            (n_angles, n_rows, n_cols)  → transpose (2, 0, 1)
"""

import numpy as np
import os
import sys
import gc
import napari

# Ensure local modules are found regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ASTRA-specific helper modules (no TIGRE dependency)
# Original TIGRE versions are untouched for TIGRE-based scripts
from geometry_reconstruction_Voxel_size_ASTRA import setup_geometry
from export_volumes_ASTRA import export_volume_to_nii, export_volume_HU

# Shared modules (identical across all reconstruction scripts)
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_FDK_3D import (load_images, generate_collapsed_sinogram,
                                     selecionar_roi_I0, get_I0_from_roi)

# ============================================================
#  ASTRA IMPORT
# ============================================================
import astra
# ============================================================


# ---------------------------------------------------------------------------
# Shared helper functions  (identical in TIGRE / ASTRA / scratch scripts)
# ---------------------------------------------------------------------------

def normalize_projections(projections_raw, I0_override=None):
    print("--> Normalizing cropped projections...")
    print(f"    Input shape: {projections_raw.shape}, "
          f"Memory: {projections_raw.nbytes / 1e6:.1f} MB")
    I0 = float(I0_override) if I0_override is not None \
         else float(np.percentile(projections_raw, 1))
    projections_raw  = projections_raw.astype(np.float32)
    ratio            = np.clip(projections_raw / (I0 + 1e-6), 1e-6, 1.2)
    projections_norm = -np.log(ratio)
    projections_norm[projections_norm < 0] = 0
    print("    ✓ Normalization complete")
    return projections_norm


def downsample_block_mean_pad(proj, f):
    H, W, A   = proj.shape
    pad_h     = (-H) % f
    pad_w     = (-W) % f
    if pad_h or pad_w:
        proj  = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    dh, dw    = proj.shape[0] // f, proj.shape[1] // f
    return proj.reshape(dh, f, dw, f, A).mean(axis=(1, 3))


def print_volume_info(volume, geo=None):
    print("RECONSTRUCTED VOLUME INFORMATION")
    print(f"  Dtype:      {volume.dtype}")
    print(f"  Shape:      {volume.shape}  (Z, Y, X)")
    if geo is not None:
        print(f"  Voxel size: {geo.dVoxel[0]*1000:.2f} μm")
        print(f"  Physical:   {geo.sVoxel} mm")


# ============================================================
#  ASTRA-SPECIFIC: geometry builder
#  Compare this section with the equivalent in:
#    - recon_tigre_fdk.py   → tigre.geometry + tigre.FDK()
#    - recon_fdk_scratch.py → manual backprojection loop
# ============================================================

def build_astra_geometry(geo, angles):
    """
    Convert a CTGeometry object into ASTRA cone_vec format.

    ASTRA cone_vec — each of the 12 values per projection:
      [srcX, srcY, srcZ]   source position
      [dX,   dY,   dZ  ]   detector centre position
      [uX,   uY,   uZ  ]   horizontal detector axis (scaled by pixel pitch)
      [vX,   vY,   vZ  ]   vertical   detector axis (scaled by pixel pitch)

    Convention (same as TIGRE):
      X = right,  Y = into scanner at theta=0,  Z = up
      theta=0 → source at -Y, increases CCW viewed from above.
    """
    DSO   = float(geo.DSO)
    DOD   = float(geo.DSD - geo.DSO)
    du    = float(geo.dDetector[1])    # col pitch, mm
    dv    = float(geo.dDetector[0])    # row pitch, mm
    off_h = float(geo.offDetector[1])  # horizontal COR offset, mm
    off_v = float(geo.offDetector[0])  # vertical offset, mm
    tilt  = float(geo.rotDetector[2])  # in-plane detector tilt, rad

    vectors = np.zeros((len(angles), 12), dtype=np.float64)

    for i, theta in enumerate(angles):
        ct, st = np.cos(theta), np.sin(theta)

        # Source
        srcX, srcY, srcZ = DSO * st, -DSO * ct, 0.0

        # Detector centre + offsets
        dX = -DOD * st + off_h * ct
        dY =  DOD * ct + off_h * st
        dZ =  off_v

        # u-axis (horizontal, rotates with gantry)
        ux, uy, uz = ct * du, st * du, 0.0

        # Apply in-plane tilt via Rodrigues rotation if needed
        if abs(tilt) > 1e-12:
            nx, ny, nz   = -st, ct, 0.0
            c, s         = np.cos(tilt), np.sin(tilt)
            cx = ny*uz - nz*uy;  cy = nz*ux - nx*uz;  cz = nx*uy - ny*ux
            nd = nx*ux + ny*uy + nz*uz
            ux = ux*c + cx*s + nx*nd*(1-c)
            uy = uy*c + cy*s + ny*nd*(1-c)
            uz = uz*c + cz*s + nz*nd*(1-c)

        # v-axis (vertical, fixed; negative → row 0 at top)
        vx, vy, vz = 0.0, 0.0, -dv

        vectors[i] = [srcX, srcY, srcZ, dX, dY, dZ, ux, uy, uz, vx, vy, vz]

    n_rows, n_cols = int(geo.nDetector[0]), int(geo.nDetector[1])
    proj_geom = astra.create_proj_geom('cone_vec', n_rows, n_cols, vectors)

    nZ, nY, nX = [int(v) for v in geo.nVoxel]
    dZ, dY, dX = [float(v) for v in geo.dVoxel]
    vol_geom = astra.create_vol_geom(
        nY, nX, nZ,
        -nX*dX/2,  nX*dX/2,
        -nY*dY/2,  nY*dY/2,
        -nZ*dZ/2,  nZ*dZ/2,
    )

    return proj_geom, vol_geom


def run_astra_fdk(sinogram, proj_geom, vol_geom):
    """
    Run ASTRA FDK_CUDA and return the reconstructed volume.

    sinogram shape:  (n_rows, n_angles, n_cols)   ← ASTRA convention
    volume shape:    (nZ, nY, nX)                 ← same as TIGRE convention
    """
    sino_id = astra.data3d.create('-sino', proj_geom, sinogram)
    vol_id  = astra.data3d.create('-vol',  vol_geom,  0)

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId']     = sino_id
    cfg['ReconstructionDataId'] = vol_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    volume = astra.data3d.get(vol_id).astype(np.float32)

    # Free GPU memory immediately
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)

    return volume

# ============================================================
#  END OF ASTRA-SPECIFIC SECTION
# ============================================================


def print_geometry_summary(geo, angles):
    print("\n  GEOMETRY  (CTGeometry → ASTRA cone_vec)")
    print(f"  DSD            {geo.DSD:.1f} mm")
    print(f"  DSO            {geo.DSO:.1f} mm")
    print(f"  DOD            {geo.DSD - geo.DSO:.1f} mm")
    print(f"  nDetector      {geo.nDetector}  [rows, cols]")
    print(f"  dDetector      {geo.dDetector*1000} μm")
    print(f"  offDetector    {geo.offDetector} mm  [vertical, horizontal]")
    print(f"  nVoxel         {geo.nVoxel}  [Z, Y, X]")
    print(f"  dVoxel         {geo.dVoxel*1000} μm")
    print(f"  Angles         {len(angles)} × "
          f"[{np.degrees(angles[0]):.1f}° … {np.degrees(angles[-1]):.1f}°]")
    print(f"  Filter         Ram-Lak (fixed in ASTRA FDK_CUDA)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(tiff_folder, configurations, output_folder=None):

    # ── PHASE 1  –  DATA LOADING & I0 SELECTION  (shared) ──────────────────
    print("\n" + "="*60)
    print("PHASE 1: DATA LOADING & I0 SELECTION")
    print("="*60)

    projections_raw = load_images(tiff_folder)
    if projections_raw is None or projections_raw.size == 0:
        raise ValueError(f"No images loaded from: {tiff_folder}")

    sino_raw       = generate_collapsed_sinogram(projections_raw)
    roi_background = selecionar_roi_I0(sino_raw)
    mean_I0        = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    del sino_raw;  gc.collect()

    # ── PHASE 2  –  CROP | NORMALISE | DOWNSAMPLE  (shared) ────────────────
    print("\n" + "="*60)
    print("PHASE 2: CROP  |  NORMALISE  |  DOWNSAMPLE")
    print("="*60)

    crop_params         = select_crop_region(projections_raw[:, :, 0])
    projections_cropped = apply_crop_to_projections(projections_raw, crop_params)
    del projections_raw;  gc.collect()

    projections_norm    = normalize_projections(projections_cropped, I0_override=mean_I0)
    del projections_cropped;  gc.collect()

    f = configurations['downsample']
    if f > 1:
        print(f"--> Downsampling {f}x ...")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32)
        del projections_norm;  gc.collect()
    else:
        projections_final = projections_norm

    # ── PHASE 3  –  ASTRA GEOMETRY & FDK RECONSTRUCTION  (ASTRA specific) ──
    print("\n" + "="*60)
    print("PHASE 3: ASTRA GEOMETRY & FDK_CUDA RECONSTRUCTION")
    print("         *** THIS IS THE ASTRA-SPECIFIC PHASE ***")
    print("         Compare with Phase 3 in:")
    print("           recon_tigre_fdk.py   → tigre.FDK()")
    print("           recon_fdk_scratch.py → manual Ram-Lak + backproject")
    print("="*60)

    shift_val          = configurations['calibrated_shift_px'] / f
    effective_voxel_sz = configurations['voxel_size'] * f

    # Build geometry (ASTRA version uses CTGeometry, not tigre.geometry)
    geo, angles = setup_geometry(
        projections_final.shape,
        effective_voxel_sz,
        configurations['DSD'],
        configurations['DSO'],
        shift_val,
        configurations['total_angle'],
        shift_sign        = configurations['shift_sign'],
        downsample_factor = configurations['downsample'],
        crop_params       = crop_params,
        detector_tilt     = configurations.get('detector_tilt', 0),
    )

    print_geometry_summary(geo, angles)

    # Build ASTRA geometry objects
    proj_geom, vol_geom = build_astra_geometry(geo, angles)

    # Reorder sinogram axes for ASTRA
    # ---------------------------------------------------------------
    # Input shape:       (n_rows, n_cols,   n_angles)
    # ASTRA expects:     (n_rows, n_angles, n_cols  )  → transpose (0, 2, 1)
    # TIGRE expects:     (n_angles, n_rows, n_cols  )  → transpose (2, 0, 1)
    # ---------------------------------------------------------------
    sinogram = np.transpose(projections_final, (0, 2, 1)).copy().astype(np.float32)
    del projections_final;  gc.collect()

    print(f"\n--> Sinogram shape sent to ASTRA: {sinogram.shape}  (n_rows, n_angles, n_cols)")
    print("--> Running FDK_CUDA ...")

    # Run ASTRA FDK
    volume = run_astra_fdk(sinogram, proj_geom, vol_geom)
    del sinogram;  gc.collect()

    print("--> ✓ Reconstruction complete")
    print_volume_info(volume, geo)

    # ── PHASE 4  –  VISUALISATION & EXPORT  (shared) ───────────────────────
    print("\n" + "="*60)
    print("PHASE 4: VISUALISATION & EXPORT")
    print("="*60)

    voxel_scale = tuple(geo.dVoxel)
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=voxel_scale, name="ASTRA FDK – CT Volume")
    napari.run()

    if input("\nExport to .nii? (y/n): ").strip().lower() == 'y':
        nii_path = export_volume_to_nii(volume, geo, tiff_folder,
                                        base_output=output_folder)
        print(f"Saved: {nii_path}")

        if input("Convert to Hounsfield Units? (y/n): ").strip().lower() == 'y':
            water_val = float(input("Gray-scale value for WATER: "))
            air_val   = float(input("Gray-scale value for AIR:   "))
            export_volume_HU(nii_path, volume, water_val, air_val)

    print("\nRECONSTRUCTION COMPLETE")
    return volume


# ---------------------------------------------------------------------------
# Configuration – edit before running
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    CONFIG = {
        'voxel_size':          25,         # μm  – reconstruction voxel size
        'calibrated_shift_px': 19.5,       # px  – centre-of-rotation shift
        'shift_sign':          1,          # +1 or -1
        'total_angle':         2 * np.pi,  # rad – full 360° rotation
        'DSD':                 488,        # mm  – source to detector
        'DSO':                 255,        # mm  – source to object
        'downsample':          1,          # 1 = no downsampling
        'detector_tilt':       0,          # rad – in-plane tilt correction
        # NOTE: filter is Ram-Lak only in ASTRA FDK_CUDA, not user-selectable
    }

    FOLDER        = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_nivel_2'
    OUTPUT_FOLDER = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift'

    main(FOLDER, CONFIG, output_folder=OUTPUT_FOLDER)