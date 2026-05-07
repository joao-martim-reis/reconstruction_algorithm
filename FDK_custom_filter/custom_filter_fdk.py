"""Custom frequency-domain filter design and application for FDK cone-beam CT.

Provides an alternative to TIGRE's algs.fdk() that accepts user-designed
1D filter arrays instead of fixed string names. The filtering pipeline
replicates TIGRE's internal filtering() function exactly, then calls
TIGRE's Atb for GPU-accelerated cone-beam backprojection.

Why this module exists
----------------------
TIGRE's algs.fdk() hardcodes the bandwidth parameter d=1 inside
tigre/utilities/filtering.py (line 22) and single_pass_algorithms.py
imports filtering via `from ... import filtering` (name binding at load
time), so monkey-patching after import does not work. The only clean way
to use a custom filter is to pre-filter projections here and then call
TIGRE's Atb for backprojection.

Typical workflow
----------------
    # 1. Get the FFT length that matches the loaded geometry
    from custom_filter_fdk import design_ct_filter, filt_len_for_geometry
    from custom_filter_fdk import reconstruct_fdk_custom_filter, plot_filter_frequency_response

    n = filt_len_for_geometry(geo)          # geometry must be set up first

    # 2. Design base filter and modify as needed
    H = design_ct_filter('ram_lak', n_samples=n, cutoff_fraction=1.0)
    H[int(0.7 * len(H)):] = 0              # hard cutoff at 70% Nyquist

    # 3. Visualize before reconstructing
    plot_filter_frequency_response(H, geo, title='Ram-Lak truncado a 70% Nyquist')

    # 4. Reconstruct
    volume = reconstruct_fdk_custom_filter(input_data, geo, angles, H)

Input / output shapes
---------------------
    projections : (n_angles, H, W) float32  — TIGRE format (angles first)
    volume      : (nz, ny, nx)    float32  — TIGRE volume format
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.fft import fft as scipy_fft
from scipy.fft import ifft as scipy_ifft
from tigre.utilities.Atb import Atb

logger = logging.getLogger(__name__)

_VALID_FILTER_NAMES = frozenset({'ram_lak', 'shepp_logan', 'cosine', 'hamming', 'hann'})


# ─── Private helpers ──────────────────────────────────────────────────────────

def _nextpow2(n: int) -> int:
    i = 1
    while 2 ** i < n:
        i += 1
    return i


def _ramp_kernel(n_samples: int) -> np.ndarray:
    """Impulse response of the Ram-Lak ramp filter, length n_samples.

    Matches TIGRE's ramp_flat() exactly — sampled at integer positions,
    analytically derived from the continuous ramp via the Fourier slice theorem.
    When FFT'd and doubled, gives the piecewise-linear ramp |f|.
    """
    nn = np.arange(-n_samples / 2, n_samples / 2)
    h = np.zeros(nn.shape, dtype=np.float32)
    h[int(n_samples / 2)] = 1.0 / 4.0
    odd = (nn % 2 == 1)
    h[odd] = -1.0 / (np.pi * nn[odd]) ** 2
    return h


def _first_float(value) -> float:
    """Return the first numeric element whether value is scalar or array-like."""
    return float(np.asarray(value).reshape(-1)[0])


# ─── Public functions ─────────────────────────────────────────────────────────

def filt_len_for_geometry(geo) -> int:
    """Return the FFT length TIGRE would use internally for a given geometry.

    Use this to get the correct n_samples for design_ct_filter() so that
    the filter matches the geometry without needing to be resampled.

    Args:
        geo: TIGRE geometry object with a nDetector attribute.

    Returns:
        FFT length (power of 2, >= 64).
    """
    return max(64, int(2 ** _nextpow2(2 * int(max(geo.nDetector)))))


def design_ct_filter(
    filter_name: str,
    n_samples: int,
    cutoff_fraction: float = 1.0,
) -> np.ndarray:
    """Design a CT reconstruction filter as a 1D half-spectrum array.

    Replicates TIGRE's internal filter construction but exposes the bandwidth
    parameter (hardcoded to 1.0 in TIGRE) as cutoff_fraction.

    The returned array is a starting point — you can modify it freely
    (notch filters, asymmetric ramps, custom apodization windows) before
    passing it to reconstruct_fdk_custom_filter().

    Args:
        filter_name: Window applied on top of the Ram-Lak ramp. One of:
            'ram_lak'     — pure ramp, no additional windowing
            'shepp_logan' — sinc window; reduces ringing, mild smoothing
            'cosine'      — cosine window; moderate smoothing
            'hamming'     — Hamming window; good noise-resolution trade-off
            'hann'        — Hann window; strongest smoothing, most noise reduction
        n_samples: Total FFT length (should be a power of 2). Use
            filt_len_for_geometry(geo) for the geometrically correct value.
        cutoff_fraction: Bandwidth as a fraction of Nyquist.
            1.0 = full bandwidth (identical to TIGRE default).
            0.5 = half bandwidth — more noise suppression, lower resolution.
            Range: (0, 1].

    Returns:
        Half-spectrum filter, shape (n_samples // 2 + 1,), float32.
        Index 0 = DC (0 Hz), index n_samples // 2 = Nyquist.
        Frequencies above cutoff_fraction * Nyquist are zeroed.

    Raises:
        ValueError: If filter_name is unrecognized or cutoff_fraction not in (0, 1].

    Examples:
        # Identical to TIGRE's internal shepp_logan filter
        H = design_ct_filter('shepp_logan', n_samples=1024)

        # Ram-Lak with bandwidth limited to 60% of Nyquist
        H = design_ct_filter('ram_lak', n_samples=1024, cutoff_fraction=0.6)

        # Manual modification after design
        H = design_ct_filter('ram_lak', n_samples=1024, cutoff_fraction=1.0)
        H[int(0.5 * len(H)):] = 0   # hard cutoff at half Nyquist
    """
    if filter_name not in _VALID_FILTER_NAMES:
        raise ValueError(
            f"filter_name must be one of {sorted(_VALID_FILTER_NAMES)}, got {filter_name!r}"
        )
    if not (0.0 < cutoff_fraction <= 1.0):
        raise ValueError(
            f"cutoff_fraction must be in (0, 1], got {cutoff_fraction}"
        )

    kernel = _ramp_kernel(n_samples)
    f_kernel = np.abs(np.fft.fft(kernel)) * 2.0
    filt = f_kernel[:int(n_samples / 2) + 1].copy()

    # Angular frequency axis: w[k] = 2π k / n_samples (matches TIGRE's w)
    w = 2.0 * np.pi * np.arange(len(filt)) / n_samples
    d = cutoff_fraction

    if filter_name == 'shepp_logan':
        # sinc window: sin(w / 2d) / (w / 2d)
        filt[1:] *= np.sin(w[1:] / (2.0 * d)) / (w[1:] / (2.0 * d))
    elif filter_name == 'cosine':
        filt[1:] *= np.cos(w[1:] / (2.0 * d))
    elif filter_name == 'hamming':
        filt[1:] *= 0.54 + 0.46 * np.cos(w[1:] / d)
    elif filter_name == 'hann':
        filt[1:] *= (1.0 + np.cos(w[1:] / d)) / 2.0
    # ram_lak: pure ramp, no window applied

    # Hard cutoff: zero all frequencies above cutoff_fraction * Nyquist
    filt[w > np.pi * d] = 0.0

    return filt.astype(np.float32)


def apply_custom_filter_to_projections(
    projections: np.ndarray,
    geo,
    angles: np.ndarray,
    custom_filter: np.ndarray,
) -> np.ndarray:
    """Apply a user-designed filter to cone-beam projections in the frequency domain.

    Replicates TIGRE's filtering() function exactly — same zero-padding strategy,
    same complex-packing trick (two projections per FFT pass), same scale_factor —
    but uses custom_filter instead of computing the filter from a string name.

    Args:
        projections: shape (n_angles, H, W), float32. TIGRE projection format
            (angles first). Obtained via np.transpose(proj_HWA, (2, 0, 1)).
        geo: TIGRE geometry object. Used for nDetector, DSD, DSO, dDetector.
        angles: Projection angles in radians, shape (n_angles,).
        custom_filter: 1D half-spectrum array (positive frequencies only),
            float32. Typically from design_ct_filter(), optionally modified.
            Length must match filt_len_for_geometry(geo) // 2 + 1.
            If it differs, linear resampling is applied with a warning.

    Returns:
        Filtered projections, same shape as input (n_angles, H, W), float32.
        Ready to pass directly to tigre.Atb for backprojection.

    Raises:
        ValueError: If projections is not 3D or custom_filter is not 1D.
    """
    if projections.ndim != 3:
        raise ValueError(
            f"Expected projections shape (n_angles, H, W), got {projections.shape}"
        )
    if custom_filter.ndim != 1:
        raise ValueError(
            f"custom_filter must be a 1D array, got shape {custom_filter.shape}"
        )

    n_angles, H, W = projections.shape
    filt_len = filt_len_for_geometry(geo)
    expected_half_len = filt_len // 2 + 1

    if len(custom_filter) != expected_half_len:
        logger.warning(
            "custom_filter length %d does not match internal FFT half-length %d "
            "(filt_len=%d). Resampling via linear interpolation. "
            "Use filt_len_for_geometry(geo) when calling design_ct_filter() to avoid this.",
            len(custom_filter), expected_half_len, filt_len,
        )
        old_x = np.linspace(0.0, 1.0, len(custom_filter))
        new_x = np.linspace(0.0, 1.0, expected_half_len)
        custom_filter = np.interp(new_x, old_x, custom_filter).astype(np.float32)

    # Build full-spectrum (two-sided) filter and tile across all detector rows
    full_filt = np.hstack(
        [custom_filter, custom_filter[1:-1][::-1]]
    ).astype(np.float32)
    filt_2d = np.tile(full_filt, (H, 1))  # shape (H, filt_len)

    # Centering padding: same as TIGRE's `padding = (filt_len - nDetector[1]) // 2`
    padding = int((filt_len - W) // 2)

    # Scale factor matches TIGRE's filtering.py line 27 exactly
    scale_factor = (
        _first_float(geo.DSD) / _first_float(geo.DSO)
        * (2.0 * np.pi / n_angles)
        / (4.0 * _first_float(geo.dDetector))
    )

    proj_out = projections.copy()
    fproj = np.empty((H, filt_len), dtype=np.complex64)

    # Filter pairs of projections packed into real/imag of a complex array.
    # This halves the number of FFT calls — identical strategy to TIGRE.
    for i in range(0, n_angles - 1, 2):
        fproj.fill(0)
        fproj.real[:, padding:padding + W] = proj_out[i]
        fproj.imag[:, padding:padding + W] = proj_out[i + 1]

        fproj = scipy_fft(fproj, axis=1)
        fproj = fproj * filt_2d
        fproj = scipy_ifft(fproj, axis=1)

        proj_out[i]     = fproj.real[:, padding:padding + W] * scale_factor
        proj_out[i + 1] = fproj.imag[:, padding:padding + W] * scale_factor

    # Handle odd number of projections — last one filtered solo (same as TIGRE)
    if n_angles % 2:
        fproj.fill(0)
        fproj.real[:, padding:padding + W] = proj_out[n_angles - 1]

        fproj = scipy_fft(fproj, axis=1)
        fproj = fproj * filt_2d
        fproj = np.real(scipy_ifft(fproj, axis=1))

        proj_out[n_angles - 1] = fproj[:, padding:padding + W] * scale_factor

    logger.info(
        "Filtered %d projections: filt_len=%d, padding=%d, scale_factor=%.4e",
        n_angles, filt_len, padding, scale_factor,
    )
    return proj_out.astype(np.float32)


def reconstruct_fdk_custom_filter(
    projections: np.ndarray,
    geo,
    angles: np.ndarray,
    custom_filter: np.ndarray,
) -> np.ndarray:
    """FDK cone-beam reconstruction with a user-designed frequency-domain filter.

    Pipeline:
        1. apply_custom_filter_to_projections() — freq-domain filtering
           (identical to TIGRE's internal filtering(), but with custom_filter)
        2. tigre.Atb()                          — GPU cone-beam backprojection
           (same CUDA kernel TIGRE uses in algs.fdk())

    With a filter from design_ct_filter(..., cutoff_fraction=1.0) and no
    further modification, the result is numerically equivalent to algs.fdk().

    Args:
        projections: shape (n_angles, H, W), float32. TIGRE projection format.
        geo: TIGRE geometry object.
        angles: Projection angles in radians, shape (n_angles,).
        custom_filter: 1D half-spectrum filter array. Typically from
            design_ct_filter(), optionally modified before passing.

    Returns:
        Reconstructed volume, shape (nz, ny, nx), float32.

    Raises:
        RuntimeError: If the volume contains non-finite voxels.
    """
    logger.info(
        "Custom-filter FDK: %d projections, filter length %d",
        projections.shape[0], len(custom_filter),
    )

    filtered = apply_custom_filter_to_projections(projections, geo, angles, custom_filter)

    logger.info("Running TIGRE Atb backprojection...")
    volume = Atb(filtered, geo, angles)

    n_bad = int(np.sum(~np.isfinite(volume)))
    if n_bad > 0:
        raise RuntimeError(
            f"Reconstruction produced {n_bad} non-finite voxels. "
            "Check the filter design and geometry parameters."
        )

    return volume.astype(np.float32)


def plot_filter_frequency_response(
    filter_array: np.ndarray,
    geo,
    title: str = '',
) -> None:
    """Plot the frequency response of a custom CT filter.

    Converts the normalised frequency axis to physical units (cycles/mm)
    using the detector pixel size from the geometry object.

    Args:
        filter_array: 1D half-spectrum filter array (positive frequencies only).
            DC at index 0, Nyquist at the last index.
        geo: TIGRE geometry object. geo.dDetector[1] provides the detector
            pixel pitch in mm, used to compute the cycles/mm axis.
        title: Optional plot title. Defaults to 'CT Filter Frequency Response'.
    """
    import matplotlib.pyplot as plt

    detector_pixel_size_mm = float(geo.dDetector[1])
    n_half = len(filter_array)

    # Nyquist frequency = 1 / (2 * pixel_size_mm)
    nyquist_cycles_per_mm = 1.0 / (2.0 * detector_pixel_size_mm)
    freqs_cycles_per_mm = np.linspace(0.0, nyquist_cycles_per_mm, n_half)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs_cycles_per_mm, filter_array, linewidth=2, color='steelblue')
    ax.axvline(
        x=nyquist_cycles_per_mm,
        color='red', linestyle='--', alpha=0.7,
        label=f'Nyquist  ({nyquist_cycles_per_mm:.1f} cy/mm)',
    )
    ax.set_xlabel('Frequency (cycles/mm)')
    ax.set_ylabel('Filter amplitude')
    ax.set_title(title or 'CT Filter Frequency Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, nyquist_cycles_per_mm * 1.05)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()
