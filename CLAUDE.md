# CLAUDE.md — CT Reconstruction Development Guide

## Project Overview
Domain: X-ray CT reconstruction (FBP, FDK, iterative). Backends: TIGRE, ASTRA, NumPy/CuPy.
Shapes: projections `(H, W, n_angles)` · volume `(nz, ny, nx)`.
Units: distances in **mm**, voxel size displayed in **μm**, angles in **radians**.

## Naming
- Functions: `snake_case`, verb-first — `load_projections_from_tiff`, `normalize_projections_beer_lambert`
- Variables encode units: `dsd_mm`, `voxel_size_um`, `i0_intensity`, `n_angles`
- Constants: `SCREAMING_SNAKE_CASE` at module level
- Domain terms: `projection`, `sinogram`, `attenuation`, `voxel`, `i0`, `dsd`, `dso` — use these exactly

## Functions & Modules
- ≤50 lines per function. Guard clauses first. One return type. No mixed I/O + computation.
- Every public function: Google-style docstring with Args/Returns/Raises (include shapes and units).
- Type hints on all new or modified function signatures.

## Reconstruction Standards
- Beer-Lambert `attenuation = -log(I/I0)` is an isolated step before reconstruction.
- I0 from open-beam ROI; 99th-percentile fallback requires `logger.warning`.
- Assert `np.isfinite` after normalization and after reconstruction.
- Default filter: `"ram-lak"`. Log DSD, DSO, voxel size, and filter at INFO level each run.

## Error Handling & Logging
- `logging` for diagnostics; `print()` only for final user-facing output.
- Raise `ValueError`/`RuntimeError` on bad input — never return `None` as a failure sentinel.
- Validate at public function entry. NaN/Inf check after normalization and reconstruction.
- Levels: DEBUG=shapes · INFO=pipeline milestones · WARNING=fallbacks · ERROR=pre-raise

## Code Quality
- All code and comments in English. No other languages.
- No commented-out code; no dead code — use git for history.
- Named module-level constants for all magic numbers (geometry, thresholds, figure sizes).
- No speculative abstraction. Solve the actual problem.

## Shared Utilities (`shared/`)
- `crop_projections.py`, `export_volumes.py`, `napari_filters.py`, `HU_conversion.py`,
  `data_processing_fdk.py` live in `shared/`. Do NOT copy into pipeline folders.
- Import via `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))`.

## GPU & Performance
- `float32` on GPU; `float64` only for accumulation. Free GPU memory explicitly after use.
- Chunk large volumes; never hold 3× the volume size in RAM. Profile before optimizing.

## Reproducibility
- Output folders: `YYYYMMDD_HHMMSS_<algorithm>_<voxel>um/` — always use `%Y%m%d_%H%M%S` (locale-safe).
- Record DSD, DSO, voxel size, filter, and iteration count in output metadata.
