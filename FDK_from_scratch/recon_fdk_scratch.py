"""
recon_fdk_scratch.py
====================
FDK cone-beam CT reconstruction from scratch.
Reference: Feldkamp, Davis & Kress (1984), JOSA A 1(6), 612-619.

Steps:
  A. Cosine pre-weighting     — FDK Eq. 3
  B. 1-D ramp filter (FFT)    — Kak & Slaney Ch. 3
  C. Weighted backprojection  — FDK Eq. 10:  weight = (DSO/U)²

GPU (CuPy) used when available, falls back to NumPy/CPU.
cuFFT is probed separately at startup — if unavailable (DLL missing),
Step B falls back to NumPy FFT while Step C continues on GPU.
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

# ── GPU backend ──────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpndi
    _GPU = True
except Exception:
    _GPU = False
    warnings.warn("CuPy not found — running entirely on CPU (NumPy).", RuntimeWarning)

_CUFFT = False
if _GPU:
    try:
        _t = cp.fft.fft(cp.zeros(8, dtype=cp.float32)); del _t
        _CUFFT = True
    except Exception as e:
        warnings.warn(
            f"cuFFT unavailable ({type(e).__name__}: {e}). "
            "Step B will use NumPy FFT; Step C still runs on GPU.",
            RuntimeWarning,
        )


def _xp(use_gpu):
    return cp if (use_gpu and _GPU) else np

def _to_cpu(arr):
    return cp.asnumpy(arr) if (_GPU and isinstance(arr, cp.ndarray)) else np.asarray(arr)


# ── Pre-processing helpers ────────────────────────────────────────────────────

def normalize_projections(projections_raw, I0_override=None):
    I0   = float(I0_override) if I0_override is not None else float(np.percentile(projections_raw, 1))
    proj = np.clip(projections_raw.astype(np.float32) / (I0 + 1e-6), 1e-6, 1.2)
    out  = np.maximum(-np.log(proj), 0)
    print(f"    I0 = {I0:.1f}  |  attenuation range [{out.min():.3f}, {out.max():.3f}]")
    return out


def downsample_block_mean_pad(proj, f):
    H, W, A = proj.shape
    proj = np.pad(proj, ((0, (-H) % f), (0, (-W) % f), (0, 0)), mode='edge')
    return proj.reshape(proj.shape[0]//f, f, proj.shape[1]//f, f, A).mean(axis=(1, 3))


# ── Step A: Cosine pre-weighting ──────────────────────────────────────────────

def preweight(sinogram, geo, use_gpu=True):
    """
    w(u,v) = DSD / sqrt(DSD² + u² + v²)    [FDK Eq. 3]
    u, v : detector coordinates (mm) from detector centre.
    v-axis flipped so positive v points upward (row 0 = top).
    """
    xp = _xp(use_gpu)

    n_rows, n_cols = int(geo.nDetector[0]), int(geo.nDetector[1])
    du, dv         = float(geo.dDetector[1]), float(geo.dDetector[0])
    off_h, off_v   = float(geo.offDetector[1]), float(geo.offDetector[0])
    DSD            = float(geo.DSD)

    u = (xp.arange(n_cols) - (n_cols - 1) / 2.0) * du + off_h
    v = ((n_rows - 1) / 2.0 - xp.arange(n_rows)) * dv + off_v
    U2D, V2D = xp.meshgrid(u, v)

    W = (DSD / xp.sqrt(DSD**2 + U2D**2 + V2D**2)).astype(xp.float32)
    return _to_cpu(xp.asarray(sinogram, dtype=xp.float32) * W[xp.newaxis])


# ── Step B: Ramp filter ───────────────────────────────────────────────────────

def ramp_filter(weighted, geo, filter_name='ram_lak', use_gpu=True):
    """
    1-D filter along the column axis (u-direction) of each projection.
    Zero-padding to next power-of-2 avoids circular convolution artefacts.
    H *= du — Riemann-sum scaling of the convolution integral.

    Kernels: 'ram_lak' | 'shepp_logan' | 'hann' | 'cosine'
    """
    fft_gpu = use_gpu and _GPU and _CUFFT
    xp      = cp if fft_gpu else np

    n_cols = weighted.shape[2]
    du     = float(geo.dDetector[1])
    fc     = 1.0 / (2.0 * du)
    n_pad  = int(2 ** np.ceil(np.log2(2 * n_cols)))

    freqs = xp.fft.fftfreq(n_pad, d=du)
    absf  = xp.abs(freqs)
    fn    = absf / fc

    kernels = {
        'ram_lak':     lambda: absf,
        'shepp_logan': lambda: absf * xp.sinc(fn / 2.0),
        'hann':        lambda: absf * xp.where(absf <= fc, 0.5 + 0.5 * xp.cos(np.pi * fn), 0.0),
        'cosine':      lambda: absf * xp.where(absf <= fc, xp.cos(0.5 * np.pi * fn), 0.0),
    }
    if filter_name not in kernels:
        raise ValueError(f"Unknown filter '{filter_name}'. Choose: {list(kernels)}")

    H    = (kernels[filter_name]() * du).astype(xp.float32)
    sino = xp.pad(xp.asarray(weighted, dtype=xp.float32), [(0,0), (0,0), (0, n_pad - n_cols)])
    out  = xp.real(xp.fft.ifft(xp.fft.fft(sino, axis=2) * H[xp.newaxis, xp.newaxis], axis=2))
    return _to_cpu(out[:, :, :n_cols].astype(xp.float32))


# ── Step C: Backprojection ────────────────────────────────────────────────────

def backproject(filtered, geo, angles, use_gpu=True):
    """
    FDK Eq. 10:
        f(x,y,z) = (1/2) ∫ [DSO/U]² · g_filtered(u_p, v_p, θ) dθ

    For voxel (x,y,z) at angle θ:
        U   = DSO + x·sin(θ) − y·cos(θ)    [source-to-voxel-plane distance]
        u_p = DSD · (x·cosθ + y·sinθ) / U  [detector column, mm]
        v_p = DSD · z / U                   [detector row, mm]
    """
    xp = _xp(use_gpu)

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
    d_theta[1:-1] = (ang[2:] - ang[:-2]) / 2.0
    d_theta[0]    = ang[1] - ang[0]
    d_theta[-1]   = ang[-1] - ang[-2]

    x = xp.asarray((np.arange(nX) - (nX-1)/2.0) * dX, dtype=xp.float32)
    y = xp.asarray((np.arange(nY) - (nY-1)/2.0) * dY, dtype=xp.float32)
    z = xp.asarray((np.arange(nZ) - (nZ-1)/2.0) * dZ, dtype=xp.float32)

    X2D, Y2D = xp.meshgrid(x, y, indexing='xy')
    volume   = xp.zeros((nZ, nY, nX), dtype=xp.float32)
    z_chunk  = 64 if (use_gpu and _GPU) else 16

    backend  = 'GPU' if (use_gpu and _GPU) else 'CPU'
    fft_note = '' if _CUFFT else ' (FFT on CPU)'
    print(f"  {len(ang)} angles → {nZ}×{nY}×{nX}  [{backend}{fft_note}]")

    # Transfer the entire filtered sinogram to GPU once, avoiding 800 individual
    # PCIe transfers inside the angle loop (~30-60s saving on this dataset).
    if use_gpu and _GPU:
        filtered_gpu = cp.asarray(filtered, dtype=cp.float32)

    t0 = time.perf_counter()

    for i, theta in enumerate(ang):
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        U2D   = DSO + X2D * sin_t - Y2D * cos_t
        U2D   = xp.where(xp.abs(U2D) < 1e-6, 1e-6, U2D)
        col2D = (DSD / U2D * (X2D * cos_t + Y2D * sin_t) - off_h) / du + c_cen
        w2D   = (DSO**2 / U2D**2).astype(xp.float32)
        aw    = float(d_theta[i]) / 2.0
        proj  = filtered_gpu[i] if (use_gpu and _GPU) else filtered[i].astype(np.float32)

        for z0 in range(0, nZ, z_chunk):
            z1   = min(z0 + z_chunk, nZ)
            v_p  = (DSD / U2D)[xp.newaxis] * z[z0:z1, xp.newaxis, xp.newaxis]
            row  = r_cen - (v_p - off_v) / dv
            col  = xp.broadcast_to(col2D, row.shape)

            if use_gpu and _GPU:
                vals = cpndi.map_coordinates(proj, xp.stack([row.ravel(), col.ravel()]),
                                             order=1, mode='constant', cval=0.0)
            else:
                vals = map_coordinates(proj.astype(np.float64),
                                       [row.ravel(), col.ravel()],
                                       order=1, mode='constant', cval=0.0)

            volume[z0:z1] += aw * vals.reshape(z1-z0, nY, nX).astype(xp.float32) * w2D[xp.newaxis]

        if (i+1) % max(1, len(ang)//10) == 0:
            el = time.perf_counter() - t0
            print(f"    {i+1:4d}/{len(ang)}  {el:5.0f}s  ETA {el/(i+1)*(len(ang)-i-1):5.0f}s")

    if use_gpu and _GPU:
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

    weighted = preweight(sinogram, geo, use_gpu);                            del sinogram;  gc.collect()
    filtered = ramp_filter(weighted, geo, cfg['filter_type'], use_gpu);      del weighted;  gc.collect()
    volume   = backproject(filtered, geo, angles, use_gpu);                  del filtered;  gc.collect()

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


# ── Config ────────────────────────────────────────────────────────────────────
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