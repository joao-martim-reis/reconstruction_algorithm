# CT Detector Center Calibration

## Overview

This folder provides a modular workflow to estimate detector center misalignment
in cone-beam CT. The output is a detector shift in pixels, suitable for use in
reconstruction setup.

## Pipeline Summary

1. Load projection TIFF files from a target folder.
2. Build a collapsed sinogram.
3. Select background ROI and object ROI interactively.
4. Segment the object and estimate detector shift.
5. Save JSON and CSV outputs with summary statistics.

## Files and Responsibilities

- `calibration_main.py`: orchestrator for full calibration workflow.
- `calibration_images_sinogram.py`: TIFF loading and collapsed sinogram creation.
- `calibration_roi.py`: interactive ROI selection and operation mode.
- `calibration_segmentation.py`: segmentation and shift estimation.
- `calibration_visualization.py`: per-run visualization.
- `calibration_reporting.py`: statistics, exports, and final report.

## Dependencies

- numpy
- tifffile
- matplotlib
- scipy

## Supported Input Layouts

### Layout A: Multi-distance mode

`parent_folder` contains one or more numeric subfolders where each subfolder
name is a distance in cm (for example `-5`, `0`, `12.5`).

Example:

```
parent_folder/
  -5/
    0001.tif
    ...
  0/
    0001.tif
    ...
  10/
    0001.tif
    ...
```

Behavior:

- Numeric subfolders are parsed and processed in ascending distance order.
- If TIFF files also exist directly in `parent_folder`, they are ignored.

### Layout B: Single-folder mode

When no numeric distance subfolders are available, the workflow can process a
single position in either form:

1. `parent_folder` contains exactly one subfolder with TIFFs, regardless of the
   subfolder name.
2. `parent_folder` contains TIFFs directly.

In single-folder mode, distance is recorded as `0.0 cm` in outputs and the
summary distance-vs-shift plot is skipped.

## Usage

Edit the `parent_folder` path in `calibration_main.py`, then run:

```bash
python Calibration\calibration_main.py
```

## Output Files

For each run, files are written to `parent_folder`:

- `calibration_results_YYYYMMDD_HHMMSS.json`
- `calibration_results_YYYYMMDD_HHMMSS.csv`

The JSON includes calibration mode metadata and summary statistics.

## Notes

- ROI selection is interactive (Matplotlib widgets), so run with a display.
- Projection ordering depends on numeric tokens in TIFF filenames.
- Reliability mode is fail-fast: invalid TIFF shape, empty TIFF sets, NaN/Inf values,
  or ambiguous segmentation now raise explicit errors instead of silent fallbacks.
- Segmentation requires a valid object component overlapping the selected object ROI;
  if no valid overlap is found, calibration for that folder is skipped with an error.

