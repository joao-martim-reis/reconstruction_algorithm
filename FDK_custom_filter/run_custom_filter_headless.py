import builtins
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import MAIN_FDK_custom_filter as pipeline


PROJECTION_FOLDER = (
    r"C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados"
    r"\Projections_SDD_457+S0D_211\Bar_pattern_nivel_2"
)
MAX_PROJECTIONS = 800
SUBSET_FOLDER = Path(__file__).resolve().parent / "_tmp_subset_even120"


def build_even_projection_subset(
    source_folder: str,
    max_projections: int,
    target_folder: Path,
) -> str:
    """Create a folder with evenly spaced TIFF files across full angle span."""
    source_path = Path(source_folder)
    tif_files = sorted(source_path.glob("*.tif")) + sorted(source_path.glob("*.tiff"))

    if len(tif_files) <= max_projections:
        print(f"--> Using full projection set ({len(tif_files)} files)")
        return str(source_path)

    subset_indices = np.unique(
        np.linspace(0, len(tif_files) - 1, max_projections, dtype=int)
    )
    selected_files = [tif_files[idx] for idx in subset_indices]

    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for src in selected_files:
        shutil.copy2(src, target_folder / src.name)

    print(
        f"--> Built evenly sampled subset: {len(selected_files)} / {len(tif_files)} "
        f"projections at {target_folder}"
    )
    return str(target_folder)


def auto_select_i0_roi(sino_raw: np.ndarray):
    """Select a deterministic top-band ROI for I0 in headless runs."""
    detector_rows, n_angles = sino_raw.shape
    row_end = max(50, detector_rows // 20)
    angle_margin = max(5, n_angles // 20)
    return (0, row_end, angle_margin, n_angles - angle_margin)


def auto_select_crop(first_projection: np.ndarray):
    """Select a centered crop with dimensions used in prior manual run."""
    height, width = first_projection.shape
    target_height = min(329, height)
    target_width = min(296, width)

    row_start = (height - target_height) // 2
    row_end = row_start + target_height
    col_start = (width - target_width) // 2
    col_end = col_start + target_width

    return {
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end,
        "original_center_col": width // 2,
        "original_height": height,
        "original_width": width,
    }


def apply_crop_without_preview(projections: np.ndarray, crop_params):
    """Apply crop without opening matplotlib preview windows."""
    if crop_params is None:
        return projections

    rs = crop_params["row_start"]
    re = crop_params["row_end"]
    cs = crop_params["col_start"]
    ce = crop_params["col_end"]
    cropped = projections[rs:re, cs:ce, :]

    print(f"--> Applying deterministic headless crop: {cropped.shape}")
    return cropped


class _DummyViewer:
    def add_image(self, *args, **kwargs):
        return None


def main():
    pipeline.selecionar_roi_I0 = auto_select_i0_roi
    pipeline.select_crop_region = auto_select_crop
    pipeline.apply_crop_to_projections = apply_crop_without_preview
    pipeline.plot_filter_frequency_response = lambda *args, **kwargs: None
    pipeline.napari = SimpleNamespace(Viewer=lambda: _DummyViewer(), run=lambda: None)

    config = {
        "voxel_size": 25,
        "calibrated_shift_px": 19.5,
        "shift_sign": 1,
        "total_angle": 2 * np.pi,
        "DSD": 457,
        "DSO": 224,
        "downsample": 1,
        "filter_type": "hann",
        "cutoff_fraction": 0.60,
        "detector_tilt": 0,
        "output_folder_NiFT": (
            r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao"
            r"\reconstructed_volumes_Nift"
        ),
        "filtered_volumes_folder": (
            r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao"
            r"\Filtered_volumes.Nift"
        ),
        "custom_filter_fn": None,
        "custom_filter_array": None,
    }

    original_input = builtins.input
    builtins.input = lambda prompt="": "n"
    try:
        projection_folder_to_use = build_even_projection_subset(
            source_folder=PROJECTION_FOLDER,
            max_projections=MAX_PROJECTIONS,
            target_folder=SUBSET_FOLDER,
        )
        volume = pipeline.main(
            projection_folder_to_use,
            config,
            output_folder=config.get("output_folder_NiFT"),
        )
    finally:
        builtins.input = original_input

    if volume is None:
        raise RuntimeError("Pipeline returned None.")

    print("Headless pipeline completed.")
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Volume finite: {np.isfinite(volume).all()}")


if __name__ == "__main__":
    main()
