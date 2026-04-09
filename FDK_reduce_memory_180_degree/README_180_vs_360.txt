FDK_reduce_memory_180 - What was changed and how it differs from full 360 reconstruction

Date: 2026-03-31

Goal
- Create an isolated reconstruction pipeline that uses only the first half of projections and reconstructs with 0 to 180 degrees.
- Keep original FDK_reduce_memory unchanged.

What was changed in this folder
1) Main configuration changed to half-scan mode
   File: MAIN_TIGRE_FDK_Voxel_size.py
   - total_angle set to np.pi (180 degrees)
   - projection_fraction set to 0.5

2) Projection loader now supports selecting only a fraction of files
   File: data_processing_FDK_3D.py
   - load_images signature changed to: load_images(tiff_folder, projection_fraction=1.0)
   - TIFF files are sorted as before, then only the first fraction is kept
   - input validation added: projection_fraction must be in (0, 1]
   - status print added showing selected_count/original_count

3) Main pipeline now passes projection_fraction to loader
   File: MAIN_TIGRE_FDK_Voxel_size.py
   - projections_raw = load_images(tiff_folder, projection_fraction=projection_fraction)

4) I0 ROI sinogram axis now follows the actual angular span
   File: data_processing_FDK_3D.py
   - selecionar_roi_I0 signature changed to: selecionar_roi_I0(sino_raw, angle_span_deg=360)
   - x-axis tick positions are generated from angle_span_deg
   - main now passes angle_span_deg = rad2deg(total_angle)

Comparison: 180 mode vs full 360 mode
1) Number of projections used
   - 360 mode: uses all files in folder (example: 800/800)
   - 180 mode: uses first 50 percent only (example: 400/800)

2) Angular range in reconstruction
   - 360 mode: total_angle = 2*pi
   - 180 mode: total_angle = pi

3) Sinogram display labeling for ROI selection
   - 360 mode: fixed labels 0 to 360 degrees
   - 180 mode: labels match configured span (0 to 180 degrees)

4) Expected practical effects
   - 180 mode can reduce long-scan drift effects and reduce memory/time
   - 180 mode can be more sensitive to geometry mismatch and may increase artifacts
   - In cone-beam systems, strict 180 degrees may be insufficient for best FDK quality; short-scan usually needs about 180 degrees plus fan angle

Important note
- This folder implements exactly the requested strict half-scan behavior.
- It does NOT duplicate or invert projections.
- If needed, a second mode can be added later: auto short-scan span (180 degrees + fan angle margin).
