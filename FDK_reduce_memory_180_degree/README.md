# Memory-Efficient FDK Cone-Beam CT Reconstruction Pipeline

## Abstract
This repository provides a memory-efficient reconstruction workflow for cone-beam computed tomography (CBCT) based on the Feldkamp–Davis–Kress (FDK) algorithm. The key optimization is the application of a spatial crop to raw projections prior to logarithmic normalization and reconstruction, thereby reducing peak memory footprint and computational cost. Geometric consistency is preserved by explicitly compensating the detector offset after cropping. Reconstruction is performed using the TIGRE toolbox; interactive visualization and post-processing are supported through Napari; and results can be exported to NIfTI with optional conversion to Hounsfield Units (HU).

## Introduction
FDK is a filtered backprojection method for circular cone-beam trajectories and remains widely used in micro-CT and industrial CT systems due to its efficiency and robustness for circular acquisition geometries. In practical experimental pipelines, the dominant constraints are frequently memory and I/O throughput, particularly when handling large detector arrays and high angular sampling. This implementation addresses those constraints by reducing the spatial size of the projection stack before the most expensive per-pixel operations (normalization and filtering), while preserving the correct reconstruction center through crop-aware geometric compensation.

## Technical Overview
The pipeline performs the following steps:

1. Load and numerically sort projection TIFF files.
2. Build a collapsed sinogram to support robust background selection.
3. Estimate the reference intensity $I_0$ from a user-selected background ROI.
4. Define a crop on a raw projection and apply it to all projections before normalization.
5. Normalize projections using Beer–Lambert: $-\log\left(\frac{I}{I_0}\right)$.
6. Optionally downsample via block-wise mean pooling with edge padding.
7. Configure cone-beam geometry in TIGRE with crop-aware detector offset correction.
8. Reconstruct the 3D volume with TIGRE's FDK implementation.
9. Visualize the volume in Napari; optionally apply interactive filters.
10. Export to NIfTI (and optionally HU) with provenance metadata.

## Detailed Technical Specifications

### Inputs

**Projection dataset**
- Data type: 2D TIFF files, one projection per file.
- Ordering: sorted by the first integer found in the filename to preserve angular consistency.
- Shape convention: raw stack is stored as `(H, W, N)` where `N` is the number of projection angles.
- Numeric type: TIFF dtype is preserved on load; subsequent computations use `float32` for a memory/precision trade-off.

**$I_0$ calibration**
- A collapsed sinogram is computed by summing the projection stack along the detector-row dimension.
- The user selects a rectangular background region (air) on the collapsed sinogram.
- $I_0$ is computed as the mean intensity inside that ROI and used as a global reference for normalization.

**Acquisition and reconstruction parameters (configuration)**
- `pixel_size` (mm): detector pixel pitch.
- `DSD` (mm): source-to-detector distance.
- `DSO` (mm): source-to-object distance.
- `total_angle` (rad): total rotation angle.
- `calibrated_shift_px` (px): pre-calibrated detector shift expressed in pixels.
- `shift_sign` (±1): sign convention for the detector shift.
- `filter_type`: reconstruction filter (e.g., `hann`).
- `downsample` (integer): spatial downsampling factor applied after cropping.
- `voxel_ratio`: voxel size scaling factor relative to the default geometric estimate.

### Outputs

**Reconstructed volume (in-memory)**
- Returned as a NumPy array.
- TIGRE output axis order is `(Z, Y, X)`.
- Physical spacing is reported by `geo.dVoxel`.

**NIfTI export**
- `.nii` volume exported in a conventional orientation with an affine derived from voxel spacing.
- A companion `metadata.txt` is generated containing geometry, shape, runtime, and TIGRE version information.

**HU export (optional)**
- If requested, the volume is converted to Hounsfield Units (HU) using user-provided intensity references for water and air.


## Pipeline Organization (Phases)

### Phase 1 — Data loading and preprocessing
**Goal:** reliable ingestion and preparation for calibration.
- Loads and sorts the TIFF stack.
- Computes a collapsed sinogram to facilitate robust selection of a background region for $I_0$.

### Phase 2 — Spatial optimization (memory reduction)
**Goal:** reduce the spatial domain before expensive operations.
- Interactive crop definition is performed on a raw projection (pre-normalization).
- The crop is applied across the entire projection stack.
- Optional downsampling is applied after cropping to maximize speed and memory savings.

### Phase 3 — Geometry setup and reconstruction
**Goal:** preserve geometric consistency after cropping.
- Detector geometry is updated to reflect the cropped detector size.
- Detector offset is corrected to account for the crop-induced shift of the detector center.
- Reconstruction is executed using TIGRE's FDK.

### Phase 4 — Post-processing and export
**Goal:** visual inspection, optional enhancement, and reproducible export.
- Napari is used for 2D/3D volume visualization.
- Optional interactive filtering is available.
- Export to NIfTI is supported, with optional HU conversion.

## Feature Specifications (Design and Behaviour)

### 1) Background ROI selection for $I_0$
- Implementation: interactive rectangle selection (Matplotlib).
- Input: collapsed sinogram.
- Output: ROI coordinates and scalar $I_0$ estimate (mean ROI intensity).
- Rationale: increases robustness against dataset-dependent background statistics and detector artefacts.

### 2) Interactive projection cropping
- User interaction:
	- Left-click defines a symmetric crop width around the detector center.
	- Right-click defines a crop height from the top down to the selected row.
- Output: `crop_params` dictionary storing crop bounds and the original detector center.
- Rationale: cropping before normalization reduces memory usage and accelerates all subsequent processing.

### 3) Beer–Lambert normalization
- Operation: $-\log\left(\frac{I}{I_0}\right)$ applied to cropped projections.
- Stability measures:
	- division uses an epsilon to avoid zero denominators;
	- ratio is clipped to avoid log singularities and extreme outliers;
	- negative values are truncated to zero.
- Output dtype: `float32`.

### 4) Block-mean downsampling with edge padding
- Operation: non-overlapping mean pooling over `f × f` pixel blocks.
- Padding: edge replication if detector dimensions are not divisible by `f`.
- Effect: reduces `(H, W)` while preserving `N` angles.
- Rationale: provides a controllable speed/memory–resolution trade-off and avoids aliasing compared with naive striding.

### 5) Crop-aware geometry and detector-offset compensation
- Updates: `geo.nDetector`, `geo.sDetector`, voxel size, voxel counts.
- Offset correction:
	- Crop changes the detector center; the resulting shift (in pixels) is converted to millimetres.
	- Total detector offset combines the pre-calibrated shift and crop-induced shift:
		$\Delta_{\mathrm{total}} = \Delta_{\mathrm{calibration}} + \Delta_{\mathrm{crop}}$.
	- The total is assigned to `geo.offDetector`.
- Rationale: prevents reconstruction-center drift and associated artefacts after cropping.

### 6) FDK reconstruction (TIGRE)
- Input formatting: projection data are transposed to `(N, H, W)`.
- Filter: configurable (e.g., Hann) as supported by TIGRE.
- Output: 3D volume compatible with Napari visualization and NIfTI export.

### 7) Interactive visualization and filtering
- Viewer: Napari with original and filtered layers.
- Filters (interactive): Gaussian, bilateral, simplified beam-hardening correction, cupping correction.
- Purpose: rapid qualitative assessment and post-processing parameter tuning.

### 8) NIfTI export and optional HU conversion
- NIfTI affine: derived from voxel spacing to preserve physical units (mm).
- Provenance: a `metadata.txt` file documents geometry, volume shape, and software versions.
- HU conversion (optional): requires user-provided intensity references for air and water.

## Repository Structure
- [MAIN_TIGRE_FDK_crop.py](MAIN_TIGRE_FDK_crop.py): end-to-end pipeline entry point.
- [data_processing_crop.py](data_processing_crop.py): projection loading, collapsed sinogram, $I_0$ ROI selection.
- [crop_projections.py](crop_projections.py): crop selection UI and crop application.
- [geometry_reconstruction.py](geometry_reconstruction.py): TIGRE geometry configuration with crop-aware offset correction.
- [export_volumes.py](export_volumes.py): NIfTI export and HU conversion.
- [napari_filters.py](napari_filters.py): interactive filtering and helper utilities.

## Configuration and Usage
Update the `CONFIG` dictionary and the input folder path in `MAIN_TIGRE_FDK_crop.py` to match your acquisition setup. The script will prompt for interactive ROI selection, cropping, and export choices.

## Limitations
- The beam-hardening and cupping corrections included here are simplified, heuristic post-processing methods intended for exploratory analysis; quantitative CT workflows typically require calibration-based or physics-informed corrections.
- The pipeline assumes a circular trajectory and uses FDK; this is not exact for non-circular trajectories or severely truncated projection data.

## Common Issues and Troubleshooting

### Negative Values in Reconstructed Volumes
It is common for reconstructed CT volumes to contain small negative values in background regions. These negatives can arise from numerical effects of the reconstruction filter (e.g., Ram-Lak ringing), slight mis-centering/shift errors, noise, or algorithmic artifacts. Small negative values are usually not a sign of catastrophic failure; however, if you require strictly non-negative data, it is possible to clip (`volume[volume<0]=0`), but clipping may hide underlying issues that are better fixed (I0 calibration, center of rotation, filter choice, etc.).

## References
When using this pipeline in academic work, please cite the canonical FDK literature and the TIGRE toolbox, as well as any relevant dataset or instrumentation sources.
