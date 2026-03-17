"""
recon_fdk_scratch.py
====================
FDK cone-beam CT reconstruction from scratch.
Reference: Feldkamp, Davis & Kress (1984), JOSA A 1(6), 612-619.

Giovanni Di Domenico
DOI: http://dx.doi.org/10.3204/DESY-PROC-2014-05/35

Steps:
  A. Cosine pre-weighting     — FDK Eq. 3
  B. 1-D ramp filter (FFT)    — Kak & Slaney Ch. 3
  C. Weighted backprojection  — FDK Eq. 10:  weight = (DSO/U)²


"""

import gc
import time
import warnings
import numpy as np
import napari
from scipy.ndimage import map_coordinates

from geometry_reconstruction_Voxel_size import setup_geometry
from crop_projections import select_crop_region, apply_crop_to_projections
from data_processing_FDK_3D import (load_images, generate_collapsed_sinogram,
                                     selecionar_roi_I0, get_I0_from_roi)
from export_volumes import export_volume_to_nii, export_volume_HU

# ── GPU backend (optional CuPy)
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpndi
    _GPU = True
    _CUFFT = hasattr(cp, 'fft') and callable(getattr(cp.fft, 'fft', None))
except Exception:
    cp = None
    cpndi = None
    _GPU = False
    _CUFFT = False

def _xp(use_gpu=True):
    if use_gpu and _GPU:
        return cp
    return np

def _to_cpu(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ── Pre-processing helpers ────────────────────────────────────────────────────

def normalize_projections(projections_raw, I0_override=None):
    I0   = float(I0_override) if I0_override is not None else float(np.percentile(projections_raw, 1))
    proj = np.clip(projections_raw.astype(np.float32) / (I0 + 1e-6), 1e-6, 1.2)
    out  = np.maximum(-np.log(proj), 0)
    return out


def downsample_block_mean_pad(proj, f):
    H, W, A = proj.shape
    proj = np.pad(proj, ((0, (-H) % f), (0, (-W) % f), (0, 0)), mode='edge')
    return proj.reshape(proj.shape[0]//f, f, proj.shape[1]//f, f, A).mean(axis=(1, 3))




# ── Step A: Cosine pre-weighting  ─────────────────────────────
#
#   g1(u, v, β) = g(u, v, β) · R_d / sqrt(R_d² + u² + v²)
#
#   R_d = DSD  (source-to-detector distance)
#   u, v = lateral / axial detector coordinates in mm from the beam axis
#
#   The weight downscales off-axis rays by the cosine of the cone angle,
#   exactly compensating for the longer path length through the object.
def preweight(sinogram, geo, use_gpu=True):
    xp = _xp(use_gpu)

    n_rows, n_cols = int(geo.nDetector[0]), int(geo.nDetector[1])
    du, dv         = float(geo.dDetector[1]), float(geo.dDetector[0])
    off_h, off_v   = float(geo.offDetector[1]), float(geo.offDetector[0])
    DSD            = float(geo.DSD)


    # Physical detector coordinates (mm) centred on the beam axis.
    # `off_h` / `off_v` are detector offsets in millimetres (geo.offDetector)
    # computed from the calibrated pixel shift inside `setup_geometry`.
    # These offsets realign the detector coordinate system to the calibrated detector centre before weighting.

    u = (xp.arange(n_cols) - (n_cols - 1) / 2.0) * du + off_h
    v = ((n_rows - 1) / 2.0 - xp.arange(n_rows)) * dv + off_v  # row 0 = top → flip
    U2D, V2D = xp.meshgrid(u, v) # detector coordinates in mm, shape (n_rows, n_cols), meshgrid is used to create 2D arrays of u and v coordinates for the entire detector plane

    # Cosine weight: DSD / sqrt(DSD² + u² + v²)
    W = (DSD / xp.sqrt(DSD**2 + U2D**2 + V2D**2)).astype(xp.float32)
    return _to_cpu(xp.asarray(sinogram, dtype=xp.float32) * W[xp.newaxis])


# ── Step B: Ramp filter  ──────────────────────────────────────
#
#   g2(u, v, β) = g1(u, v, β) ⊗ h_ramp(u)
#
#   Convolution is along the u-axis (detector columns) only.
#   Implemented in the frequency domain via FFT: multiply by |f| (× window).
#   The `* du` factor is the Riemann-sum scaling of the continuous convolution.
#   Zero-padding to the next power-of-2 prevents circular-convolution artefacts.
#
#   Available kernels: 'ram_lak' | 'shepp_logan' | 'hann' | 'cosine'
def ramp_filter(weighted, geo, filter_name='ram_lak', use_gpu=True):
    fft_gpu = use_gpu and _GPU and _CUFFT
    xp      = cp if fft_gpu else np

    n_cols = weighted.shape[2]
    du     = float(geo.dDetector[1])        # detector pixel pitch in mm
    
    #padding - adding extra pixels/elements to the array to increase its size, typically to the next power of 2, to optimize the performance of the FFT and prevent circular convolution artefacts.
    n_pad  = int(2 ** np.ceil(np.log2(2 * n_cols)))  # next power-of-2 ≥ 2·n_cols


    freqs = xp.fft.fftfreq(n_pad, d=du)    # frequency axis matching the FFT output
    absf  = xp.abs(freqs)                  # |f| — the pure ramp



    H = (absf * du).astype(xp.float32)  # frequency-domain Ram-Lak filter
    sino_padded = xp.pad(xp.asarray(weighted, dtype=xp.float32), [(0,0), (0,0), (0, n_pad - n_cols)])
    out = xp.real(xp.fft.ifft(xp.fft.fft(sino_padded, axis=2) * H[xp.newaxis, xp.newaxis], axis=2))
    return _to_cpu(out[:, :, :n_cols].astype(xp.float32))  # discard padding


# ── Step C: Weighted backprojection  [Image Eq. 5] ──────────────────────────
#
#   f̂(x,y,z) = (1/2) ∫₀²π  (R_s / U)²  ·  g2(u_p, v_p, β)  dβ
#
#   R_s = DSO  (source-to-isocentre distance)
#
#   For each voxel (x, y, z) at projection angle θ:
#     U   = DSO + x·sinθ − y·cosθ        source-to-voxel-plane distance
#     u_p = DSD · (x·cosθ + y·sinθ) / U  projected detector column (mm)
#     v_p = DSD · z / U                  projected detector row (mm)
#
#   The (DSO/U)² weight corrects for the diverging cone beam (magnification).
#   The 1/2 · dβ factor discretises the continuous integral.
#
#   Note: the sign convention in U differs from some references only because
#   the rotation direction is defined differently (θ vs. β); the physics is identical.

def backproject(filtered, geo, angles, use_gpu=True):
    xp = _xp(use_gpu)
    gpu_mode = use_gpu and _GPU

    DSD, DSO       = float(geo.DSD), float(geo.DSO)
    nZ, nY, nX     = [int(v) for v in geo.nVoxel]
    dZ, dY, dX     = [float(v) for v in geo.dVoxel]
    n_rows, n_cols = int(geo.nDetector[0]), int(geo.nDetector[1])
    du, dv         = float(geo.dDetector[1]), float(geo.dDetector[0])
    off_h, off_v   = float(geo.offDetector[1]), float(geo.offDetector[0])
    c_cen, r_cen   = (n_cols - 1) / 2.0, (n_rows - 1) / 2.0

    # Per-angle Δθ via centred differences (handles non-uniform spacing)
    ang           = np.asarray(angles, dtype=np.float64)
    d_theta       = np.empty(len(ang))
    d_theta[1:-1] = (ang[2:] - ang[:-2]) / 2.0 # Central differences for interior angles
    d_theta[0]    = ang[1] - ang[0]
    d_theta[-1]   = ang[-1] - ang[-2]

    x = xp.asarray((np.arange(nX) - (nX-1)/2.0) * dX, dtype=xp.float32)
    y = xp.asarray((np.arange(nY) - (nY-1)/2.0) * dY, dtype=xp.float32)
    z = xp.asarray((np.arange(nZ) - (nZ-1)/2.0) * dZ, dtype=xp.float32)

    X2D, Y2D = xp.meshgrid(x, y, indexing='xy')
    volume   = xp.zeros((nZ, nY, nX), dtype=xp.float32)
    z_chunk  = 64 if gpu_mode else 16

    # Optionally transfer the entire filtered sinogram to GPU once to avoid
    # repeated PCIe transfers inside the angle loop.
    if gpu_mode:
        filtered_gpu = cp.asarray(filtered, dtype=cp.float32)

    t0 = time.perf_counter()
    n_ang = len(ang)
    progress_step = max(1, n_ang // 10)

    for i, theta in enumerate(ang):
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        # ── U: source-to-voxel-plane distance  (denominator in Eq. 5) ──────
        U2D  = DSO + X2D * sin_t - Y2D * cos_t
        U2D  = xp.where(xp.abs(U2D) < 1e-6, 1e-6, U2D)  # avoid /0 singularity

        # Precompute DSD/U once — shared by column projection, row projection,
        # and the (DSO/U)² weight.  Avoids three separate large-array divisions.
        dsd_U = (DSD / U2D).astype(xp.float32)           # DSD/U  (magnification factor)

        # Projected detector coordinates of each XY voxel (mm → pixel index).
        # `off_h` and `off_v` are millimetre offsets (geo.offDetector) derived
        # from the calibrated pixel shift. We subtract these mm offsets from
        # the projected physical coordinates and then divide by `du`/`dv`
        # (mm/px) to obtain pixel indices — this is how the calibrated
        # detector offset shifts the projection sampling on the pixel grid.
        col2D = (dsd_U * (X2D * cos_t + Y2D * sin_t) - off_h) / du + c_cen  # u_p → col
        w2D   = (DSO / U2D).astype(xp.float32) ** 2      # (DSO/U)²  — divergence weight

        aw   = float(d_theta[i]) / 2.0                   # (1/2)·dβ  — quadrature weight
        proj = filtered_gpu[i] if gpu_mode else filtered[i].astype(np.float32)

        for z0 in range(0, nZ, z_chunk):
            z1  = min(z0 + z_chunk, nZ)
            v_p = dsd_U[xp.newaxis] * z[z0:z1, xp.newaxis, xp.newaxis]  # v_p = DSD·z/U (mm)
            row = r_cen - (v_p - off_v) / dv                              # v_p → row index
            col = xp.broadcast_to(col2D, row.shape)

            if gpu_mode:
                vals = cpndi.map_coordinates(proj, xp.stack([row.ravel(), col.ravel()]),
                                             order=1, mode='constant', cval=0.0)
            else:
                vals = map_coordinates(proj.astype(np.float64),
                                       [row.ravel(), col.ravel()],
                                       order=1, mode='constant', cval=0.0)

            volume[z0:z1] += aw * vals.reshape(z1-z0, nY, nX).astype(xp.float32) * w2D[xp.newaxis]

        if (i+1) % progress_step == 0:
            el = time.perf_counter() - t0
            print(f"    {i+1:4d}/{n_ang}  {el:5.0f}s  ETA {el/(i+1)*(n_ang-i-1):5.0f}s")

    if gpu_mode:
        del filtered_gpu

    return _to_cpu(volume).astype(np.float32)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(tiff_folder, cfg, output_folder=None, use_gpu=True):

    print("\n── Phase 1: Load & I0 ──")
    proj_raw = load_images(tiff_folder)
    sino_raw = generate_collapsed_sinogram(proj_raw)
    mean_I0  = get_I0_from_roi(sino_raw, selecionar_roi_I0(sino_raw), proj_raw.shape[0])
    del sino_raw;  gc.collect()

    print("\n── Phase 2: Crop / Normalise / Downsample ──")
    crop_params  = select_crop_region(proj_raw[:, :, 0])
    proj_cropped = apply_crop_to_projections(proj_raw, crop_params)
    del proj_raw;  gc.collect()

    proj_norm = normalize_projections(proj_cropped, I0_override=mean_I0)
    del proj_cropped;  gc.collect()

    f = cfg['downsample']
    proj_final = downsample_block_mean_pad(proj_norm, f).astype(np.float32) if f > 1 else proj_norm
    if f > 1:
        del proj_norm;  gc.collect()

    print(f"\n── Phase 3: FDK  [filter: {cfg['filter_type']}] ──")



    geo, angles = setup_geometry(
        proj_final.shape,
        cfg['voxel_size'] * f,
        cfg['DSD'], cfg['DSO'],
        cfg['calibrated_shift_px'] / f,
        cfg['total_angle'],
        shift_sign        = cfg['shift_sign'],
        downsample_factor = cfg['downsample'],
        crop_params       = crop_params,
        detector_tilt     = cfg.get('detector_tilt', 0),
    )

    sinogram = np.transpose(proj_final, (2, 0, 1)).copy().astype(np.float32)
    del proj_final;  gc.collect()

    weighted = preweight(sinogram, geo, use_gpu);                            
    filtered = ramp_filter(weighted, geo, cfg['filter_type'], use_gpu);      
    volume   = backproject(filtered, geo, angles, use_gpu);                  

    print(f"\nVolume {volume.shape}  range [{volume.min():.4f}, {volume.max():.4f}]")

    print("\n── Phase 4: Viewer ──")
    viewer = napari.Viewer()
    viewer.add_image(volume, scale=tuple(geo.dVoxel), name=f"FDK [{cfg['filter_type']}]")
    napari.run()

    if input("\nExport to .nii? (y/n): ").strip().lower() == 'y':
        path = export_volume_to_nii(volume, geo, tiff_folder, base_output=output_folder)
        print(f"Saved: {path}")
        if input("Convert to HU? (y/n): ").strip().lower() == 'y':
            export_volume_HU(path, volume,
                             float(input("Water value: ")),
                             float(input("Air value  : ")))

    return volume


if __name__ == "__main__":

    CONFIG = {
        'voxel_size':          25,        # μm
        'calibrated_shift_px': 19.5,
        'shift_sign':          1,
        'total_angle':         2 * np.pi,
        'DSD':                 488,        # mm
        'DSO':                 255,        # mm
        'downsample':          1,
        'filter_type':         'ram_lak',  # ram_lak | shepp_logan | hann | cosine
        'detector_tilt':       0,
    }

    FOLDER = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_nivel_2'
    OUTPUT = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift'

    main(FOLDER, CONFIG, output_folder=OUTPUT, use_gpu=True)