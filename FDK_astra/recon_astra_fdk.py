"""
recon_astra_fdk.py
==================
FDK cone-beam CT reconstruction using the ASTRA Toolbox.

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

import astra

# Shared helper functions  (identical in TIGRE / ASTRA / scratch scripts)

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
# ============================================================

def build_astra_geometry(geo, angles):
    """
    Convert a CTGeometry object into ASTRA cone_vec format.

    ASTRA cone_vec — each of the 12 values per projection:
      [srcX, srcY, srcZ]   source position
      [dX,   dY,   dZ  ]   detector centre position
      [uX,   uY,   uZ  ]   horizontal detector axis 
      [vX,   vY,   vZ  ]   vertical   detector axis 

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
        srcX, srcY, srcZ = DSO * st, -DSO * ct, 0.0 #this ,0.0 is because the source is assumed to be in the plane of rotation (Z=0) in this geometry setup. If the source had a vertical offset, it would be added to srcZ here.
        #the srcY is -DSO * ct because the source is located at a distance DSO from the center of rotation, and as the gantry rotates, the source moves in a circular path around the object. The negative sign ensures that the source moves in the correct direction (clockwise or counterclockwise) depending on the angle theta.

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
    proj_geom = astra.create_proj_geom('cone_vec', n_rows, n_cols, vectors) # is used to create the projection geometry for cone-beam CT reconstruction using the ASTRA Toolbox. 
    #The 'cone_vec' type indicates that the geometry is defined by a set of vectors, which specify the source and detector positions and orientations for each projection angle.

    nZ, nY, nX = [int(v) for v in geo.nVoxel]
    dZ, dY, dX = [float(v) for v in geo.dVoxel]
    vol_geom = astra.create_vol_geom( #vol_geom is used to create the volume geometry for the reconstructed image. It defines the size and spatial resolution of the volume that will be reconstructed from the projection data.
        nY, nX, nZ,
        -nX*dX/2,  nX*dX/2, # The volume is centered at the origin, so the minimum and maximum coordinates are set to -nX*dX/2 and nX*dX/2 for the X-axis, -nY*dY/2 and nY*dY/2 for the Y-axis, and -nZ*dZ/2 and nZ*dZ/2 for the Z-axis. This means that the reconstructed volume will span from -nX*dX/2 to nX*dX/2 in the X direction, from
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
    sino_id = astra.data3d.create('-sino', proj_geom, sinogram) #astra.data3d.create is used to create a 3D data object in the ASTRA Toolbox. The '-sino' argument specifies that the data object being created is a sinogram, which is a common format for storing projection data in computed tomography (CT). The proj_geom argument provides the geometry information for the projections, and the sinogram argument contains the actual projection data that will be used for reconstruction.
    vol_id  = astra.data3d.create('-vol',  vol_geom,  0) #vol_id is created as an empty volume (initialized to zeros) with the specified volume geometry. This volume will be filled with the reconstructed image after running the FDK algorithm. The vol_geom argument defines the size and spatial resolution of the reconstructed volume, while the 0 argument initializes all voxel values to zero before reconstruction.

    cfg = astra.astra_dict('FDK_CUDA') #astra.astra_dict is a function in the ASTRA Toolbox that creates a configuration dictionary for a specified reconstruction algorithm. In this case, 'FDK_CUDA' indicates that the configuration being created is for the FDK (Feldkamp-Davis-Kress) algorithm implemented on NVIDIA GPUs using CUDA. The resulting cfg dictionary will contain default parameters for the FDK_CUDA algorithm, which can then be modified as needed before running the reconstruction.
    cfg['ProjectionDataId']     = sino_id 
    cfg['ReconstructionDataId'] = vol_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id) #astra.algorithm.run is used to execute the specified reconstruction algorithm in the ASTRA Toolbox. The alg_id argument is the identifier of the algorithm that was created and configured earlier using astra.algorithm.create. When this function is called, it runs the FDK_CUDA reconstruction algorithm using the provided projection data (sino_id) and fills the reconstruction volume (vol_id) with the resulting image.

    volume = astra.data3d.get(vol_id).astype(np.float32) #astra.data3d.get is used to retrieve the reconstructed volume data from the ASTRA Toolbox after running the reconstruction algorithm. 
    #The vol_id argument specifies the identifier of the volume data object that was created earlier to hold the reconstructed image. The resulting volume is then converted to a NumPy array of type float32 for further processing or analysis.

    # Free GPU memory immediately
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)

    return volume


def main(tiff_folder, configurations, output_folder=None):


    print("PHASE 1: DATA LOADING & I0 SELECTION")
    projections_raw = load_images(tiff_folder)
    sino_raw       = generate_collapsed_sinogram(projections_raw)
    roi_background = selecionar_roi_I0(sino_raw)
    if roi_background is None:
        raise RuntimeError("I0 ROI not selected. Draw a ROI on the collapsed sinogram and click Confirm.")
    mean_I0        = get_I0_from_roi(sino_raw, roi_background, projections_raw.shape[0])
    del sino_raw;  gc.collect()


    print("PHASE 2: CROP  |  NORMALISE  |  DOWNSAMPLE")
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


    print("PHASE 3: ASTRA GEOMETRY & FDK_CUDA RECONSTRUCTION")
    print("ASTRA-SPECIFIC PHASE ")


    shift_val          = configurations['calibrated_shift_px'] / f
    effective_voxel_sz = configurations['voxel_size'] * f

    # Build geometry 
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


    print("PHASE 4: VISUALISATION & EXPORT")
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