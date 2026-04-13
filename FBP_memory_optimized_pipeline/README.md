# FBP Memory-Optimized Volume Pipeline

This folder contains a standalone FBP slice-by-slice reconstruction pipeline with memory-aware preprocessing.

## Files
- `MAIN_TIGRE_FBP_Volume_memory_optimized.py`: Main executable pipeline.
- `crop_projections.py`: Crop UI and crop application copied from the FDK reduce-memory workflow.
- `data_processing_fbp_volume.py`: Projection batch loading, I0 ROI tools, and sinogram normalization.

## Chunking Note
- `chunk_slices` is the memory batch size, not the final number of reconstructed slices.
- Final stack depth is set by your crop height: `row_end - row_start`.
- Set `chunk_slices = None` (or `<= 0`) to reconstruct all cropped slices in one chunk.

## Preprocessing Order
1. Validate TIFF set and geometry assumptions.
2. Compute collapsed sinogram from full projections (no crop).
3. Select I0 ROI on that full collapsed sinogram and compute I0.
4. Select crop region on the first RAW projection (`select_crop_region`).
5. Apply that crop during batch loading (`apply_crop_to_projections`).
6. Normalize each slice sinogram with `-log(I/I0)` using the pre-crop I0 value.
7. Reconstruct each slice sequentially with TIGRE FBP.
8. Stack slices into a 3D volume in memory (no automatic final file export).
9. Open the stack in Napari for inspection (with robust contrast limits).
10. Optionally export to NIfTI (`.nii` / `.nii.gz`) when prompted.

## Saving Behavior
- The pipeline does **not** automatically save TIFF/NPY output files.
- A temporary memmap cache is used during reconstruction for memory efficiency.
- Final persistent output is created only if you choose NIfTI export.

## Run
```powershell
python FBP_memory_optimized_pipeline/MAIN_TIGRE_FBP_Volume_memory_optimized.py
```

## Required Parameters (already set in script)
- `voxel_size_um = 30`
- `calibrated_shift_px = 19.5`
- Input folder:
  `C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Analise_Resultados\Projections_SDD_457+S0D_211\Bar_pattern_nivel_2`
