import gc
from pathlib import Path

import numpy as np
import tigre.algorithms as algs


from crop_projections import select_crop_and_ranges
from geometry_reconstruction_fbp import prepare_fbp_geometry
from data_processing_fbp_volume import (
    extract_sinogram_raw,
    get_I0_from_roi,
    load_images,
    load_projection_batch,
    normalize_sinogram,
    selecionar_roi_I0,
    validate_projection_set,
)
from volume_output import ask_and_export_nii, open_volume_in_napari


# Chunking and per-slice reconstruction rationale
# ---------------------------------------------
# This pipeline reconstructs the 3D volume by reconstructing detector rows
# (slices) one at a time and writing each reconstructed slice into a
# memmap-backed volume. We group contiguous rows into "chunks" to reduce I/O
# overhead: each chunk contains `chunk_slices` rows which are loaded together
# from disk (only the needed rows and columns). Within a chunk we still
# reconstruct each detector row independently and immediately patch it into
# the output memmap. This achieves a low peak memory footprint while keeping
# reasonable throughput.
#
# Why per-slice (and chunked) reconstruction?
# - Memory bounds: reconstructing one slice at a time means peak memory is
#   proportional to a single reconstructed slice plus the sinogram buffer for
#   that slice (not the full 3D volume).
# - I/O tradeoff: grouping rows into small chunks reduces repeated open/read
#   overhead compared with strictly per-slice loads. `chunk_slices` controls
#   this tradeoff (1 = minimal memory, higher values = fewer I/O ops).
# - Direct patching: each reconstructed slice is written into the memmap at
#   its final location, so no large in-memory stacking is required.



def print_input_assumptions(height, width, n_proj):
    print(f"  Detector size: {height} x {width} px")
    print(f"  Number of projections: {n_proj}")


def fbp_reconstruct_single_slice(sino_norm, geo, angles, filter_type):
    # Expect a 2D sinogram (angles x detector_u). Wrap to (angles, 1, detector_u)
    if sino_norm.ndim != 2:
        raise ValueError("sino_norm must be 2D: (angles, detector_u)")
    sino_input = sino_norm[:, None, :]

    rec = algs.fbp(sino_input, geo, angles, filter_type=filter_type)
    return np.squeeze(rec).astype(np.float32, copy=False)


def resolve_chunk_slices(cropped_h, requested_chunk):
    if requested_chunk is None:
        return cropped_h

    requested_chunk = int(requested_chunk)
    if requested_chunk <= 0:
        return cropped_h

    return min(requested_chunk, cropped_h)


def iter_chunk_ranges(total_slices, chunk_slices):
    for chunk_start in range(0, total_slices, chunk_slices):
        chunk_end = min(chunk_start + chunk_slices, total_slices)
        yield chunk_start, chunk_end


def reconstruct_chunk_to_volume(volume_mem, chunk_start, chunk_end, context):
    """
    Reconstruct one chunk of rows and write directly into the output memmap.
    This keeps peak memory low by processing only a subset of rows at a time.
    """
    # context is a simple dict with the required keys (keeps the helper lightweight)
    full_row_start = context["row_start"] + chunk_start
    full_row_end = context["row_start"] + chunk_end

    raw_batch = load_projection_batch(
        context["file_paths"],
        0,
        context["n_proj"],
        dtype=np.float32,
        row_range=(full_row_start, full_row_end),
        col_range=(context["col_start"], context["col_end"]),
    )

    # raw_batch shape: (chunk_slices, cropped_w, n_proj)
    sinogram_batch = np.transpose(raw_batch, (0, 2, 1))  # (chunk_slices, n_proj, cropped_w)
    del raw_batch
    gc.collect()

    for local_idx, sino_raw in enumerate(sinogram_batch):
        sino_norm = normalize_sinogram(sino_raw, context.get("mean_i0"))
        rec_slice = fbp_reconstruct_single_slice(
            sino_norm,
            context["geo"],
            context["angles"],
            context["filter_type"],
        )
        np.nan_to_num(rec_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        volume_mem[chunk_start + local_idx, :, :] = rec_slice

    del sinogram_batch
    gc.collect()


def reconstruct_fbp_volume(config):
    print("PHASE 1: INPUT VALIDATION")
    file_paths = load_images(config["tiff_folder"])
    full_h, full_w, n_proj = validate_projection_set(file_paths)
    print_input_assumptions(full_h, full_w, n_proj)

    print("\nPHASE 2: I0 ROI CALIBRATION (BEFORE CROP)")
    collapsed_sino = extract_sinogram_raw(file_paths, crop_params=None, angle_batch_size=n_proj)
    roi_background = selecionar_roi_I0(collapsed_sino)
    mean_i0 = get_I0_from_roi(collapsed_sino, roi_background, full_h)

    print("\nPHASE 3: CROP SELECTION (APPLIED BEFORE NORMALIZATION/RECON)")
    _crop_params, (row_start, row_end, col_start, col_end), cropped_h, cropped_w = (
        select_crop_and_ranges(file_paths, load_projection_batch, full_h, full_w)
    )


    print("\nPHASE 4: GEOMETRY SETUP (FBP/PARALLEL)")
    print(f"  voxel_size_um: {config['voxel_size_um']}")
    print(f"  calibrated_shift_px: {config['calibrated_shift_px']}")

    geo, angles = prepare_fbp_geometry(cropped_w, n_proj, config)
    if int(geo.nVoxel[0]) != 1 or int(geo.nDetector[0]) != 1:
        raise RuntimeError(
            "Geometry is not configured for single-slice FBP "
            f"(nVoxel[0]={geo.nVoxel[0]}, nDetector[0]={geo.nDetector[0]})."
        )
    print(f"  Slice-by-slice mode confirmed: nVoxel[0]={geo.nVoxel[0]}, nDetector[0]={geo.nDetector[0]}")

    print("\nPHASE 5: CHUNKED SEQUENTIAL SLICE RECONSTRUCTION")
    out_dir = Path(config["output_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_memmap_path = out_dir / "_fbp_volume_working_cache.memmap"
    volume_mem = np.memmap(
        volume_memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(cropped_h, cropped_w, cropped_w),
    )

    requested_chunk = config.get("chunk_slices", 32)
    chunk_slices = resolve_chunk_slices(cropped_h, requested_chunk)
    # Use a simple dict as the context to keep the helper lightweight and
    # avoid a dedicated dataclass. Keys are documented in reconstruct_chunk_to_volume.
    context = {
        "file_paths": file_paths,
        "row_start": row_start,
        "col_start": col_start,
        "col_end": col_end,
        "n_proj": n_proj,
        "mean_i0": mean_i0,
        "geo": geo,
        "angles": angles,
        "filter_type": config["filter_type"],
    }

    n_chunks = int(np.ceil(cropped_h / float(chunk_slices)))
    print(f"  total slices to reconstruct: {cropped_h}")
    print(f"  chunk_slices (memory batch size): {chunk_slices}")
    print(f"  number of chunks: {n_chunks}")

    for chunk_idx, (chunk_start, chunk_end) in enumerate(
        iter_chunk_ranges(cropped_h, chunk_slices),
        start=1,
    ):
        current_chunk = chunk_end - chunk_start

        print(
            f"\n  Reconstructing chunk {chunk_idx}/{n_chunks}: "
            f"cropped rows [{chunk_start}:{chunk_end}] ({current_chunk} slices in this chunk)"
        )
        reconstruct_chunk_to_volume(volume_mem, chunk_start, chunk_end, context)
        volume_mem.flush()

    # Copy memmap to RAM at the end to produce one contiguous 3D stack.
    volume_np = np.array(volume_mem, dtype=np.float32, copy=True)
    np.nan_to_num(volume_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nPHASE 6: STACK READY (NO AUTO-SAVE)")
    print(f"  Volume shape: {volume_np.shape}")

    return volume_np, str(volume_memmap_path)





if __name__ == "__main__":
    CONFIG = {
        "tiff_folder": r"C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_nivel_2",
        "output_folder": r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\FBP_memory_optimized_pipeline",
        "voxel_size_um": 25,
        "calibrated_shift_px": 19.5,
        "shift_sign": 1,
        "detector_pixel_size_mm": 0.050,
        "total_angle": 2 * np.pi,
        "filter_type": "shepp_logan",
        # chunk_slices is only the in-memory batch size, not final stack depth.
        # Use None or 0 (or <= 0) to process all cropped slices in one chunk.
        # Set to 1 to process one detector row (slice) at a time.
        "chunk_slices": 1,
        # angle_batch_size: number of angles to load per batch. Use 0 to load all angles.
        "angle_batch_size": 0,
        # Remove temporary memmap working file after pipeline ends.
        "cleanup_working_cache": True,
    }

    fbp_volume, working_cache_path = reconstruct_fbp_volume(CONFIG)
    open_volume_in_napari(fbp_volume, CONFIG["voxel_size_um"])
    ask_and_export_nii(fbp_volume, CONFIG)


