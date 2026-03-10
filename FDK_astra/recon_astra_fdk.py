import numpy as np
import os
import sys
import gc
import matplotlib.pyplot as plt
import napari
from datetime import datetime


from FDK_astra.geometry_reconstruction_Voxel_size_ASTRA import setup_geometry
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_FDK_3D import load_images, generate_collapsed_sinogram, selecionar_roi_I0, get_I0_from_roi
from FDK_astra.export_volumes_ASTRA import export_volume_to_nii, export_volume_HU

# ---------------------------------------------------------------------------
# ASTRA dependency check
# ---------------------------------------------------------------------------
try:
    import astra
    if not astra.use_cuda():
        raise RuntimeError(
            "ASTRA is installed but CUDA is not available.\n"
            "ASTRA FDK_CUDA requires a CUDA GPU.  Ensure you installed the GPU build:\n"
            "  pip install astra-toolbox   (or the conda-forge CUDA variant)"
        )
except ImportError:
    raise ImportError(
        "ASTRA Toolbox is not installed.\n"
        "Install: pip install astra-toolbox\n"
        "See:     https://www.astra-toolbox.com/docs/install.html"
    )


# ---------------------------------------------------------------------------
# Helper functions (same as MAIN_TIGRE_FDK_Voxel_size.py)
# ---------------------------------------------------------------------------

def normalize_projections(projections_raw, I0_override=None):
    print("--> Normalizing cropped projections...")
    print(f"    Input shape: {projections_raw.shape}, Memory: {projections_raw.nbytes / 1e6:.1f} MB")
    I0 = float(I0_override) if I0_override is not None else float(np.percentile(projections_raw, 1))
    projections_raw = projections_raw.astype(np.float32)
    ratio = projections_raw / (I0 + 1e-6)
    ratio = np.clip(ratio, 1e-6, 1.2)
    projections_norm = -np.log(ratio)
    projections_norm[projections_norm < 0] = 0
    print(f"    ✓ Normalization complete")
    return projections_norm


def downsample_block_mean_pad(proj, f):
    Height, Width, Angles = proj.shape
    pad_h = (-Height) % f
    pad_w = (-Width)  % f
    if pad_h or pad_w:
        proj = np.pad(proj, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    dh, dw = proj.shape[0] // f, proj.shape[1] // f
    blocks = proj.reshape(dh, f, dw, f, Angles)
    return blocks.mean(axis=(1, 3))


def print_volume_info(volume, geo=None):
    print("RECONSTRUCTED VOLUME INFORMATION")
    print(f"Dtype: {volume.dtype}")
    print(f"Dimensions: {volume.shape}  (Z, Y, X)")
    if geo is not None:
        print(f"  Voxel size: {geo.dVoxel[0]*1000:.2f} μm")
        print(f"  Physical size: {geo.sVoxel} mm")


# ---------------------------------------------------------------------------
# ASTRA geometry builder
# ---------------------------------------------------------------------------

def build_astra_geometry(geo, angles):
    """
    Map a TIGRE geometry object to ASTRA cone_vec format.

    ASTRA cone_vec: each projection is described by 12 values
        [srcX, srcY, srcZ,  dX, dY, dZ,  uX, uY, uZ,  vX, vY, vZ]

    Coordinate system matches TIGRE:
        X right, Y into scanner at θ=0, Z up.
        Angles CCW from above (+Z), θ=0 → source at −Y.

    For angle θ:
        source centre  = ( DSO·sin θ,  −DSO·cos θ,  0 )
        detector centre= (−DOD·sin θ,   DOD·cos θ,  0 ) + offsets
        u  (horizontal)= ( cos θ·du,    sin θ·du,   0 )  ← rotates with gantry
        v  (vertical)  = ( 0,           0,           −dv ) ← row 0 at top

    Detector offset:
        offDetector[1] (horizontal, mm) rotates with the gantry  →  add along û
        offDetector[0] (vertical, mm) is fixed in Z              →  add along +Z

    In-plane detector tilt (rotDetector[2], rad) rotates u around the
    beam-direction axis (Rodrigues rotation).
    """
    DSO   = float(geo.DSO)
    DOD   = float(geo.DSD - geo.DSO)
    du    = float(geo.dDetector[1])          # col pixel pitch  mm
    dv    = float(geo.dDetector[0])          # row pixel pitch  mm
    off_h = float(geo.offDetector[1])        # horizontal offset mm
    off_v = float(geo.offDetector[0])        # vertical   offset mm
    tilt  = float(geo.rotDetector[2])        # in-plane tilt rad

    n_ang   = len(angles)
    vectors = np.zeros((n_ang, 12), dtype=np.float64)

    for i, theta in enumerate(angles):
        ct = np.cos(theta);  st = np.sin(theta)

        # Source position
        srcX, srcY, srcZ =  DSO * st,  -DSO * ct,  0.0

        # Detector centre (raw + horizontal + vertical offsets)
        dX = -DOD * st  +  off_h * ct
        dY =  DOD * ct  +  off_h * st
        dZ =  off_v

        # u-axis (horizontal, co-rotates with gantry), scaled by pixel pitch
        ux_base = ct * du
        uy_base = st * du
        uz_base = 0.0

        # Apply in-plane tilt around the detector normal n = R(θ)·[0,1,0]
        if abs(tilt) > 1e-12:
            nx, ny, nz = -st, ct, 0.0
            cos_t, sin_t = np.cos(tilt), np.sin(tilt)
            cross_x = ny * uz_base - nz * uy_base
            cross_y = nz * ux_base - nx * uz_base
            cross_z = nx * uy_base - ny * ux_base
            nd_u = nx * ux_base + ny * uy_base + nz * uz_base
            uX = ux_base * cos_t + cross_x * sin_t + nx * nd_u * (1 - cos_t)
            uY = uy_base * cos_t + cross_y * sin_t + ny * nd_u * (1 - cos_t)
            uZ = uz_base * cos_t + cross_z * sin_t + nz * nd_u * (1 - cos_t)
        else:
            uX, uY, uZ = ux_base, uy_base, uz_base

        # v-axis (vertical, fixed in lab frame)
        # Negative because row 0 is at the TOP of the detector → rows increase downward
        vX, vY, vZ = 0.0,  0.0,  -dv

        vectors[i] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

    # Projection geometry
    n_rows, n_cols = int(geo.nDetector[0]), int(geo.nDetector[1])
    proj_geom = astra.create_proj_geom('cone_vec', n_rows, n_cols, vectors)

    # Volume geometry  (ASTRA: rows=Y, cols=X, slices=Z)
    # Extents are symmetric around the origin
    nZ, nY, nX = [int(v) for v in geo.nVoxel]
    dZ, dY, dX = [float(v) for v in geo.dVoxel]
    half_x, half_y, half_z = nX * dX / 2.0, nY * dY / 2.0, nZ * dZ / 2.0

    # create_vol_geom(ny, nx, nz, min_x, max_x, min_y, max_y, min_z, max_z)
    vol_geom = astra.create_vol_geom(
        nY, nX, nZ,
        -half_x,  half_x,
        -half_y,  half_y,
        -half_z,  half_z,
    )

    return proj_geom, vol_geom


def print_geometry_mapping(geo, angles):
    """Print the TIGRE → ASTRA field-by-field translation for verification."""
    print("\n  GEOMETRY MAPPING  TIGRE → ASTRA cone_vec")
    print(f"  {'DSD':20s}  {geo.DSD:.3f} mm   (source-to-detector)")
    print(f"  {'DSO':20s}  {geo.DSO:.3f} mm   (source-to-object)")
    print(f"  {'DOD':20s}  {geo.DSD - geo.DSO:.3f} mm   (object-to-detector)")
    print(f"  {'nDetector':20s}  {geo.nDetector}  [rows, cols]")
    print(f"  {'dDetector':20s}  {geo.dDetector*1000} μm  [row_pitch, col_pitch]")
    print(f"  {'offDetector':20s}  {geo.offDetector} mm  [vertical, horizontal]")
    print(f"  {'rotDetector[2]':20s}  {geo.rotDetector[2]:.6f} rad  (in-plane tilt)")
    print(f"  {'nVoxel [Z,Y,X]':20s}  {geo.nVoxel}")
    print(f"  {'dVoxel [Z,Y,X]':20s}  {geo.dVoxel*1000} μm")
    print(f"  {'Angles':20s}  {len(angles)} projections  "
          f"[{angles[0]:.4f} … {angles[-1]:.4f}] rad  (CCW, same convention)")
    print()
    print("  NOTE: ASTRA FDK_CUDA uses Ram-Lak filter only.")
    print("        If filter_type != 'ram_lak', only TIGRE/scratch apply the")
    print("        requested kernel; ASTRA always uses Ram-Lak.")
    print("  NOTE: ASTRA FDK may differ in absolute intensity from TIGRE due to")
    print("        different filter normalisation constants.  Compare")
    print("        relative contrast first; apply a linear rescale if needed.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(tiff_folder, configurations, output_folder=None):
    """
    ASTRA FDK reconstruction pipeline.

    Phase 1 – Data loading & I0 selection       (identical to TIGRE)
    Phase 2 – Crop, normalise, downsample        (identical to TIGRE)
    Phase 3 – ASTRA geometry + FDK_CUDA
    Phase 4 – Napari viewer & optional NIfTI export
    """

    # ── PHASE 1 ─────────────────────────────────────────────────────────────
    print("\nPHASE 1: DATA LOADING & I0 SELECTION")

    projections_raw = load_images(tiff_folder)
    sino_raw        = generate_collapsed_sinogram(projections_raw)
    roi_background  = selecionar_roi_I0(sino_raw)
    mean_I0         = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    del sino_raw
    gc.collect()

    # ── PHASE 2 ─────────────────────────────────────────────────────────────
    print("\nPHASE 2: CROP  |  NORMALISE  |  DOWNSAMPLE")

    crop_params         = select_crop_region(projections_raw[:, :, 0])
    projections_cropped = apply_crop_to_projections(projections_raw, crop_params)
    del projections_raw
    gc.collect()

    projections_norm    = normalize_projections(projections_cropped, I0_override=mean_I0)
    del projections_cropped
    gc.collect()

    f = configurations['downsample']
    if f > 1:
        print(f"Downsampling {f}x …")
        projections_final = downsample_block_mean_pad(projections_norm, f).astype(np.float32)
        del projections_norm
        gc.collect()
    else:
        projections_final = projections_norm

    # ── PHASE 3 ─────────────────────────────────────────────────────────────
    print("\nPHASE 3: ASTRA GEOMETRY & FDK_CUDA RECONSTRUCTION")

    shift_val          = configurations['calibrated_shift_px'] / f
    effective_voxel_sz = configurations['voxel_size'] * f

    # Build TIGRE geo object – reused for shift/crop calculations and NIfTI export
    geo, angles = setup_geometry(
        projections_final.shape,
        effective_voxel_sz,
        configurations['DSD'],
        configurations['DSO'],
        shift_val,
        configurations['total_angle'],
        shift_sign       = configurations['shift_sign'],
        downsample_factor= configurations['downsample'],
        crop_params      = crop_params,
        detector_tilt    = configurations.get('detector_tilt', 0),
    )

    # Build ASTRA geometry from geo fields
    print_geometry_mapping(geo, angles)
    proj_geom, vol_geom = build_astra_geometry(geo, angles)

    # Sinogram: ASTRA expects (n_angles, n_rows, n_cols)  – same as TIGRE input
    sinogram = np.transpose(projections_final, (2, 0, 1)).copy().astype(np.float32)
    del projections_final
    gc.collect()

    # Create ASTRA GPU data objects
    sino_id = astra.data3d.create('-sino', proj_geom, sinogram)
    vol_id  = astra.data3d.create('-vol',  vol_geom,  0)

    # Configure and run FDK_CUDA
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId']    = sino_id
    cfg['ReconstructionDataId']= vol_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Retrieve volume – ASTRA returns (nZ, nY, nX), same axis order as TIGRE
    volume = astra.data3d.get(vol_id).astype(np.float32)

    # Clean up GPU objects immediately
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)

    print_volume_info(volume, geo)

    # ── PHASE 4 ─────────────────────────────────────────────────────────────
    print("\nPHASE 4: VISUALISATION & EXPORT")

    voxel_scale = (geo.dVoxel[0], geo.dVoxel[1], geo.dVoxel[2])
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=voxel_scale, name="ASTRA FDK – CT Volume")
    napari.run()

    nii_filepath = None
    if input("\nExport to .nii? (y/n): ").strip().lower() == 'y':
        nii_filepath = export_volume_to_nii(volume, geo, tiff_folder,
                                            base_output=output_folder)
        print(f"Saved: {nii_filepath}")

        if nii_filepath and input("Convert to Hounsfield Units? (y/n): ").strip().lower() == 'y':
            water_val = float(input("Gray-scale value for WATER: "))
            air_val   = float(input("Gray-scale value for AIR  : "))
            export_volume_HU(nii_filepath, volume, water_val, air_val)

    print("\nRECONSTRUCTION COMPLETE")
    return volume


# ---------------------------------------------------------------------------
# Configuration – edit before running
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    CONFIG = {
        'voxel_size':           25,           # μm
        'calibrated_shift_px':  40,
        'shift_sign':           1,
        'total_angle':          2 * np.pi,
        'DSD':                  488,           # mm  (457 + 31)
        'DSO':                  255,           # mm  (224 + 31)
        'downsample':           1,
        'filter_type':          'ram_lak',     # ASTRA FDK_CUDA uses Ram-Lak regardless
        'detector_tilt':        0,
    }

    FOLDER        = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Motor_grande'
    OUTPUT_FOLDER = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift'

    main(FOLDER, CONFIG, output_folder=OUTPUT_FOLDER)
