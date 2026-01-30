# CT Detector Center Calibration

## Overview

This script performs systematic calibration to determine the **detector center misalignment** in cone-beam CT systems. It quantifies the pixel offset between the rotation axis center and the detector center, which is critical for accurate image reconstruction.
"""
Calibration Module — Overview

This folder contains a modular CT detector-center calibration workflow. The goal
is to compute the pixel offset between the rotation axis and the detector
geometric center using collapsed sinograms from projection stacks.
"""

## Purpose
- Measure detector/rotation-center misalignment across multiple phantom
  positions and export results for further analysis.

## Files and responsibilities
- `calibration_process.py`: orchestrator that runs the full calibration flow.
- `calibration_io.py`: image I/O and collapsed sinogram generation.
- `calibration_roi.py`: interactive ROI selection (background + object).
- `calibration_segmentation.py`: segmentation, component selection and shift computation.
- `calibration_visualization.py`: plotting of sinogram, mask and final result.
- `calibration_reporting.py`: save JSON/CSV, compute statistics and summary plotting.

## Dependencies
- numpy, tifffile, matplotlib, scipy

Install with:

```bash
pip install numpy tifffile matplotlib scipy
```

## Quick usage

Run the orchestrator (from the workspace root or the `Calibration` folder):

```bash
python Calibration\calibration_process.py
```

The script expects a `parent_folder` containing subfolders named by distance
(e.g. `10`, `20` or `0.45`). For each folder it will:
1. Load TIFF projections
2. Create a collapsed sinogram
3. Ask the user to draw background and object ROIs
4. Compute the detector shift and show results
5. Save JSON/CSV summary and a plot

## Notes
- ROI selection uses Matplotlib interactive widgets — run locally or with an
  X server. For headless runs, consider adding a non-interactive ROI mode.
- I can add CLI flags to pass `parent_folder` and to skip GUI steps.

## Suggested next steps
- Add CLI argument parsing for `parent_folder` and non-interactive mode.
- Add unit tests for `calibration_segmentation.py` using synthetic sinograms.
- Replace prints with structured logging.
   - Area containing the **phantom sinogram signature**

   - Guides the segmentation algorithm to focus on the correct region



**Why two ROIs?**

- Background ROI → Calculates **threshold** for segmentation

- Object ROI → Acts as a **spatial hint** to avoid detecting wrong features

