"""Phantom-based Hounsfield Unit calibration for CT reconstructed volumes.

Phantom calibration workflow:
    1. calibration = run_phantom_calibration()   # file dialog + interactive ROIs + fit
    2. vol_hu = apply_hu_calibration(volume, load_calibration())  # apply to any volume

The original two-point HU_conversion() is preserved for backward compatibility.
"""

import json
import logging
import os
from datetime import date

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Button, RectangleSelector

logger = logging.getLogger(__name__)

# Theoretical HU values derived from NIST XCOM mass attenuation coefficients at 45 keV.
# Reference: https://physics.nist.gov/PhysRefData/XrayMassCoef/tab4.html
# Formula: HU = 1000 * (mu_material - mu_water) / (mu_water - mu_air)
# where mu = (mu/rho) * rho  (linear attenuation coefficient, cm^-1)
# Update these constants if your phantom insert densities differ from literature values.
PHANTOM_MATERIALS_HU: dict[str, float] = {
    "Air":             -1000.0,
    "Distilled water":     0.0,
    "Paraffin":          -222.0,
    # HA computed by elemental mixture rule from NIST element data
    # (Ca 39.9%, P 18.5%, O 41.4%, H 0.2%, density 3.16 g/cm3).
    # This value is highly density-dependent — verify against your phantom datasheet.
    "Hydroxyapatite":  11900.0,
    "Rigid PU foam":    -736.0,
    # PMMA physical density (1.19 g/cm3) is higher than water, but at 45 keV its mass
    # attenuation coefficient mu/rho is slightly lower than water's, so HU is near zero
    # or slightly negative. This is a real low-energy effect; at 120 kV PMMA is ~+120 HU.
    "PMMA":               -8.0,
}

MATERIAL_COLORS: list[str] = [
    "lightblue", "deepskyblue", "khaki", "orange", "ghostwhite", "lightgreen"
]

_CALIB_FILENAME = "hu_calibration.json"
_CALIB_PATH = os.path.join(os.path.dirname(__file__), _CALIB_FILENAME)
_POLY_DEGREE = 2


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def load_volume_nii(filepath: str = None) -> np.ndarray:
    """Load a NIfTI volume, opening a file dialog if no path is given.

    Args:
        filepath: Absolute path to a .nii or .nii.gz file. If None, a file
            picker dialog opens for interactive selection.

    Returns:
        volume: 3-D float32 array, shape (nz, ny, nx).

    Raises:
        ValueError: If no file is selected or the loaded array is not 3-D.
    """
    if filepath is None:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select NIfTI volume",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")],
        )
        root.destroy()
        if not filepath:
            raise ValueError("No file selected.")

    volume = nib.load(filepath).get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}.")
    logger.info("Loaded volume: %s  shape=%s", os.path.basename(filepath), volume.shape)
    return volume


# ---------------------------------------------------------------------------
# Interactive ROI selection
# ---------------------------------------------------------------------------

def select_phantom_rois(volume: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    """Interactively draw one 2-D ROI per phantom material on the central slice.

    Shows the central z-slice. The user draws rectangular regions one material
    at a time, in the order of PHANTOM_MATERIALS_HU. Each confirmed ROI is
    rendered as a colored box with a label. "Undo Last" resets the previous
    material; "Confirm All" closes the window once all six are drawn.

    Args:
        volume: 3-D float32 array, shape (nz, ny, nx).

    Returns:
        rois: Dict mapping material name → (x1, y1, x2, y2) pixel coordinates
            on the central slice (x = column index, y = row index).
    """
    materials = list(PHANTOM_MATERIALS_HU.keys())
    center_z = volume.shape[0] // 2
    rois: dict = {}
    current_idx = [0]
    patch_artists: list = []
    text_artists: list = []

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.18)
    ax.imshow(volume[center_z], cmap="gray", origin="upper")
    title = ax.set_title(f"Draw ROI for:  {materials[0]}", fontsize=12)

    def _refresh_title() -> None:
        idx = current_idx[0]
        label = materials[idx] if idx < len(materials) else "All done — click Confirm All"
        title.set_text(f"Draw ROI for:  {label}")
        fig.canvas.draw_idle()

    def _on_select(eclick, erelease) -> None:
        idx = current_idx[0]
        if idx >= len(materials):
            return
        x1 = int(min(eclick.xdata, erelease.xdata))
        y1 = int(min(eclick.ydata, erelease.ydata))
        x2 = int(max(eclick.xdata, erelease.xdata))
        y2 = int(max(eclick.ydata, erelease.ydata))
        mat = materials[idx]
        color = MATERIAL_COLORS[idx % len(MATERIAL_COLORS)]
        rois[mat] = (x1, y1, x2, y2)
        patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        txt = ax.text(x1, max(y1 - 4, 0), mat, color=color, fontsize=8, clip_on=True)
        patch_artists.append(patch)
        text_artists.append(txt)
        current_idx[0] += 1
        _refresh_title()

    _selector_ref = [RectangleSelector(
        ax, _on_select, useblit=False, button=[1],
        minspanx=3, minspany=3, spancoords="pixels", interactive=False,
        props=dict(facecolor="white", edgecolor="white", alpha=0.2, fill=True),
    )]

    def _undo_last(_) -> None:
        if not rois:
            return
        mat = materials[current_idx[0] - 1]
        del rois[mat]
        patch_artists.pop().remove()
        text_artists.pop().remove()
        current_idx[0] -= 1
        _refresh_title()

    def _confirm_all(_) -> None:
        if len(rois) == len(materials):
            plt.close(fig)

    _btn_undo = Button(plt.axes([0.10, 0.05, 0.20, 0.07]), "Undo Last")
    _btn_confirm = Button(plt.axes([0.70, 0.05, 0.20, 0.07]), "Confirm All")
    _btn_undo.on_clicked(_undo_last)
    _btn_confirm.on_clicked(_confirm_all)

    plt.show(block=True)

    missing = [m for m in materials if m not in rois]
    if missing:
        logger.warning("No ROI drawn for: %s", missing)
    return rois


# ---------------------------------------------------------------------------
# Intensity extraction
# ---------------------------------------------------------------------------

def _extract_roi_mean(
    volume: np.ndarray, roi: tuple[int, int, int, int], center_z: int
) -> float:
    """Return the mean intensity within a 2-D ROI on the central slice.

    Args:
        volume: 3-D float32 array, shape (nz, ny, nx).
        roi: (x1, y1, x2, y2) pixel coordinates on the slice.
        center_z: Index of the z-slice to sample.

    Returns:
        Mean pixel value within the specified ROI region.
    """
    x1, y1, x2, y2 = roi
    return float(np.mean(volume[center_z, y1:y2, x1:x2]))


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def _fit_calibration_curves(
    hu_vals: np.ndarray, intensities: np.ndarray
) -> dict:
    """Fit linear, polynomial (deg 2), and exponential calibration models.

    All forward fits use HU as the independent variable and measured intensity
    as the dependent variable (matches the calibration plot axes). Inverse
    conversion coefficients are also stored for direct volume conversion.

    Args:
        hu_vals: Theoretical HU values per material, shape (n,), float64.
        intensities: Measured mean intensities per material, shape (n,), float64.

    Returns:
        fits: Dict with keys "linear", "polynomial", "exponential". Each entry
            is a dict containing: coeffs (forward, for plotting),
            convert_coeffs (inverse, for apply_hu_calibration), r2, rmse.
            "exponential" is None when any intensity is <= 0.
    """
    def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return r2, float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    fits: dict = {}

    # Linear: intensity = m*HU + b;  inverse: HU = (I - b)/m = polyval([1/m, -b/m], I)
    c_lin = np.polyfit(hu_vals, intensities, 1).tolist()
    m, b = c_lin
    r2, rmse = _r2_rmse(intensities, np.polyval(c_lin, hu_vals))
    fits["linear"] = {
        "coeffs": c_lin,
        "convert_coeffs": [1.0 / m, -b / m],
        "r2": r2, "rmse": rmse,
    }

    # Polynomial deg-2: forward fit for plot; separate inverse fit for conversion
    c_poly = np.polyfit(hu_vals, intensities, _POLY_DEGREE).tolist()
    c_inv_poly = np.polyfit(intensities, hu_vals, _POLY_DEGREE).tolist()
    r2, rmse = _r2_rmse(intensities, np.polyval(c_poly, hu_vals))
    fits["polynomial"] = {
        "coeffs": c_poly,
        "convert_coeffs": c_inv_poly,
        "r2": r2, "rmse": rmse,
    }

    # Exponential: I = A*exp(B*HU);  linearise as ln(I) = ln(A) + B*HU
    if np.all(intensities > 0):
        c_exp = np.polyfit(hu_vals, np.log(intensities), 1).tolist()  # [B, ln(A)]
        B_exp, ln_A = c_exp
        r2, rmse = _r2_rmse(intensities, np.exp(B_exp * hu_vals + ln_A))
        fits["exponential"] = {
            "coeffs": c_exp,
            "convert_coeffs": c_exp,  # HU = (ln(I) - ln(A)) / B applied in apply_hu_calibration
            "r2": r2, "rmse": rmse,
        }
    else:
        logger.warning("Exponential fit skipped: one or more measured intensities are <= 0.")
        fits["exponential"] = None

    return fits


# ---------------------------------------------------------------------------
# Interactive fit selection
# ---------------------------------------------------------------------------

def _choose_fit_interactively(
    hu_vals: np.ndarray, intensities: np.ndarray, fits: dict
) -> str:
    """Show calibration scatter and all fitted curves; return the user's choice.

    Displays a matplotlib window with one scatter point per material, three
    overlaid fit curves, and a stats box showing R² and RMSE. Three buttons
    let the user toggle the active model; "Save & Close" finalises.

    Args:
        hu_vals: Theoretical HU values per material.
        intensities: Measured mean intensities per material.
        fits: Output of _fit_calibration_curves().

    Returns:
        chosen: One of "linear", "polynomial", or "exponential".
    """
    materials = list(PHANTOM_MATERIALS_HU.keys())
    chosen = ["linear"]

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.22)

    for mat, hu, val, color in zip(materials, hu_vals, intensities, MATERIAL_COLORS):
        ax.scatter(hu, val, color=color, s=80, zorder=5)
        ax.annotate(mat, (hu, val), textcoords="offset points", xytext=(6, 4), fontsize=7)

    x_range = np.linspace(hu_vals.min() - 500, hu_vals.max() + 500, 400)
    plot_lines: dict = {}
    curve_styles = {
        "linear":      ("b-",  "Linear"),
        "polynomial":  ("r--", "Polynomial (deg 2)"),
        "exponential": ("g:",  "Exponential"),
    }
    for key, (style, label) in curve_styles.items():
        if not fits.get(key):
            continue
        c = fits[key]["coeffs"]
        if key == "exponential":
            B_exp, ln_A = c
            y_curve = np.exp(np.clip(B_exp * x_range + ln_A, -30, 30))
        else:
            y_curve = np.polyval(c, x_range)
        line, = ax.plot(x_range, y_curve, style, label=label, linewidth=2)
        plot_lines[key] = line

    stats_lines = "\n".join(
        f"{k[:4].capitalize()}: R²={fits[k]['r2']:.4f}  RMSE={fits[k]['rmse']:.5f}"
        for k in ("linear", "polynomial", "exponential") if fits.get(k)
    )
    ax.text(0.02, 0.98, stats_lines, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    ax.set_xlabel("HU (theoretical, 45 keV)")
    ax.set_ylabel("Measured intensity")
    ax.set_title("HU Calibration — select a fit model, then click Save & Close")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    def _highlight(key: str) -> None:
        chosen[0] = key
        for k, line in plot_lines.items():
            line.set_linewidth(3 if k == key else 1)
            line.set_alpha(1.0 if k == key else 0.3)
        fig.canvas.draw_idle()

    _highlight("linear")

    _btn_refs: list = []
    btn_positions = {"linear": [0.10, 0.05, 0.18, 0.07],
                     "polynomial": [0.33, 0.05, 0.18, 0.07],
                     "exponential": [0.56, 0.05, 0.18, 0.07]}
    for key, pos in btn_positions.items():
        if fits.get(key):
            btn = Button(plt.axes(pos), key.capitalize())
            btn.on_clicked(lambda _, k=key: _highlight(k))
            _btn_refs.append(btn)

    btn_save = Button(plt.axes([0.78, 0.05, 0.15, 0.07]), "Save & Close")
    btn_save.on_clicked(lambda _: plt.close(fig))
    _btn_refs.append(btn_save)

    plt.show(block=True)
    return chosen[0]


# ---------------------------------------------------------------------------
# Main calibration workflow
# ---------------------------------------------------------------------------

def run_phantom_calibration(
    volume: np.ndarray = None, save_path: str = None
) -> dict:
    """Full phantom calibration: ROI selection → fits → user choice → save JSON.

    Args:
        volume: 3-D float32 reconstructed CT array. If None, a .nii file dialog
            opens for selection.
        save_path: Override path for the calibration JSON. Defaults to
            hu_calibration.json in the same folder as this script.

    Returns:
        calibration: Dict with fit_type, coeffs, convert_coeffs, r2, rmse,
            energy_kv, created date, and per-material HU + intensity values.
    """
    if volume is None:
        volume = load_volume_nii()

    rois = select_phantom_rois(volume)
    center_z = volume.shape[0] // 2
    materials = list(PHANTOM_MATERIALS_HU.keys())
    hu_vals = np.array([PHANTOM_MATERIALS_HU[m] for m in materials], dtype=np.float64)
    intensities = np.array(
        [_extract_roi_mean(volume, rois[m], center_z) if m in rois else float("nan")
         for m in materials],
        dtype=np.float64,
    )

    for mat, hu, intensity in zip(materials, hu_vals, intensities):
        logger.info("%-20s  HU_ref=%8.1f  intensity=%.6f", mat, hu, intensity)

    fits = _fit_calibration_curves(hu_vals, intensities)
    chosen = _choose_fit_interactively(hu_vals, intensities, fits)

    fit_data = fits[chosen]
    calibration = {
        "fit_type": chosen,
        "coeffs": fit_data["coeffs"],
        "convert_coeffs": fit_data["convert_coeffs"],
        "r2": fit_data["r2"],
        "rmse": fit_data["rmse"],
        "energy_kv": 45,
        "created": date.today().isoformat(),
        "materials": {
            mat: {"hu_theoretical": float(hu), "measured_intensity": float(i)}
            for mat, hu, i in zip(materials, hu_vals, intensities)
        },
    }
    save_calibration(calibration, save_path or _CALIB_PATH)
    logger.info("Calibration saved | fit=%s  R²=%.4f", chosen, fit_data["r2"])
    return calibration


# ---------------------------------------------------------------------------
# Apply calibration to volumes
# ---------------------------------------------------------------------------

def apply_hu_calibration(volume: np.ndarray, calibration: dict) -> np.ndarray:
    """Convert a reconstructed volume to Hounsfield Units using a saved calibration.

    Args:
        volume: 3-D float32 array of raw attenuation values.
        calibration: Dict returned by run_phantom_calibration() or load_calibration().

    Returns:
        volume_hu: Float32 array in Hounsfield Units, same shape as input.

    Raises:
        ValueError: If calibration["fit_type"] is not recognised.
    """
    fit_type = calibration["fit_type"]
    cc = calibration["convert_coeffs"]

    if fit_type in ("linear", "polynomial"):
        volume_hu = np.polyval(cc, volume).astype(np.float32)
    elif fit_type == "exponential":
        B_exp, ln_A = cc
        log_vol = np.log(volume.astype(np.float64).clip(min=1e-9))
        volume_hu = ((log_vol - ln_A) / B_exp).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown fit_type '{fit_type}'. Expected linear, polynomial, or exponential."
        )

    if not np.isfinite(volume_hu).all():
        logger.warning("Non-finite values detected after HU conversion — check calibration.")

    logger.info("HU calibration applied | fit=%s", fit_type)
    return volume_hu


# ---------------------------------------------------------------------------
# Calibration persistence
# ---------------------------------------------------------------------------

def save_calibration(calibration: dict, filepath: str = None) -> None:
    """Write a calibration dict to a JSON file.

    Args:
        calibration: Dict returned by run_phantom_calibration().
        filepath: Destination path. Defaults to hu_calibration.json next to
            this script.
    """
    filepath = filepath or _CALIB_PATH
    with open(filepath, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info("Calibration written to %s", filepath)


def load_calibration(filepath: str = None) -> dict:
    """Load a previously saved HU calibration from JSON.

    Args:
        filepath: Path to hu_calibration.json. Defaults to the file next to
            this script.

    Returns:
        calibration: Dict with fit_type, coeffs, convert_coeffs, r2, rmse,
            and per-material measurements.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
    """
    filepath = filepath or _CALIB_PATH
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")
    with open(filepath) as f:
        calibration = json.load(f)
    logger.info(
        "Calibration loaded | fit=%s  R²=%.4f  path=%s",
        calibration.get("fit_type"), calibration.get("r2", float("nan")), filepath,
    )
    return calibration


# ---------------------------------------------------------------------------
# Original two-point calibration — kept for backward compatibility
# ---------------------------------------------------------------------------

def HU_conversion(volume: np.ndarray, water_val: float, air_val: float) -> np.ndarray:
    """Convert a reconstructed volume from raw attenuation values to Hounsfield Units.

    Applies a two-point linear calibration using measured air and water intensities.
    Generates and displays the calibration line plot.

    Args:
        volume: Reconstructed volume array, any shape, float32.
        water_val: Mean intensity measured in a water region of the reconstruction.
        air_val: Mean intensity measured in an air region of the reconstruction.
            Must differ from water_val.

    Returns:
        volume_hu: Array of same shape as input, values in Hounsfield Units (float32).

    Raises:
        ValueError: If water_val == air_val (would cause division by zero).
    """
    if water_val == air_val:
        raise ValueError(
            f"water_val and air_val must differ for HU calibration; both are {water_val}"
        )

    HU_AIR = -1000.0
    HU_WATER = 0.0
    m = (HU_WATER - HU_AIR) / (water_val - air_val)
    b = HU_WATER - m * water_val

    logger.info("HU calibration | slope=%.4f  intercept=%.4f", m, b)

    margin = abs(water_val - air_val) * 0.2
    x_range = np.linspace(min(air_val, water_val) - margin,
                          max(air_val, water_val) + margin, 100)

    plt.figure(figsize=(8, 6))
    plt.plot(x_range, m * x_range + b, "b-", label="Calibration Line")
    plt.scatter([air_val], [HU_AIR], color="red", s=100, zorder=5,
                label=f"Air ({HU_AIR:.0f} HU)")
    plt.scatter([water_val], [HU_WATER], color="green", s=100, zorder=5,
                label=f"Water ({HU_WATER:.0f} HU)")
    plt.title(f"Calibration: Intensity vs Hounsfield Units\n(Air={air_val}, Water={water_val})")
    plt.xlabel("Input Value (float32)")
    plt.ylabel("Hounsfield Units (HU)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

    return (volume * m + b).astype(np.float32)
