"""
CT Calibration Process - Detector Center Alignment

This module orchestrates the calibration workflow and delegates
IO, ROI selection, segmentation, visualization, and reporting
to separate modules for better organization.
"""

import os
import sys
import logging

# Ensure local `Calibration` modules import when running this script directly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from calibration_images_sinogram import load_images, generate_collapsed_sinogram
from calibration_roi import selecionar_roi_interativamente
from calibration_segmentation import calculate_shift_segmentation
from calibration_visualization import show_results
from calibration_reporting import compute_statistics, save_results, plot_results, print_final_report


logger = logging.getLogger(__name__)


def _setup_logging():
    """Configure default logging format for calibration runs."""
    # Keep diagnostics structured and timestamped for easier troubleshooting.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _contains_tiff_files(folder_path):
    """Return True if folder contains at least one TIFF projection file."""
    try:
        entries = os.listdir(folder_path)
    except OSError:
        return False
    return any(name.lower().endswith((".tif", ".tiff")) for name in entries)


def _parse_distance_cm(folder_name: str) -> float:
    """Parse folder name as a numeric distance in centimeters."""
    return float(folder_name.replace(',', '.'))


def _build_processing_targets(parent_folder: str) -> tuple[list, str]:
    """Discover calibration targets and choose processing mode.

    Returns:
        targets: List of tuples (distance_cm, folder_name, folder_path).
        mode: Either "multi_distance" or "single_folder".
    """
    # Guard: fail early with a clear message for missing parent folders.
    if not os.path.isdir(parent_folder):
        raise FileNotFoundError(f"Parent folder does not exist or is not a directory: {parent_folder}")

    subfolders = [
        f for f in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, f))
    ]
    parent_has_tiffs = _contains_tiff_files(parent_folder)

    logger.info("Found %d subfolders", len(subfolders))
    for folder in subfolders:
        logger.info("  - %s", folder)

    folders_with_distances = []
    skipped_subfolders = []
    for folder in subfolders:
        try:
            distance_cm = _parse_distance_cm(folder)
            folder_path = os.path.join(parent_folder, folder)

            # Guard: numeric folders without TIFF files are ignored to avoid runtime load failures.
            if _contains_tiff_files(folder_path):
                folders_with_distances.append((distance_cm, folder, folder_path))
            else:
                skipped_subfolders.append(f"{folder} (numeric but no TIFF files)")
        except ValueError:
            skipped_subfolders.append(folder)

    if folders_with_distances:
        folders_with_distances.sort(key=lambda x: x[0])
        logger.info("Processing %d folders in order", len(folders_with_distances))
        for dist, folder, _ in folders_with_distances:
            logger.info("  %.2f cm -> %s", dist, folder)
        if skipped_subfolders:
            logger.info("Ignoring non-numeric subfolders (multi-distance mode)")
            for folder in skipped_subfolders:
                logger.info("  - %s", folder)
        if parent_has_tiffs:
            logger.info("TIFF files in parent folder were ignored because numeric distance folders were found")
        return folders_with_distances, "multi_distance"

    if len(subfolders) == 1:
        single_folder = subfolders[0]
        single_folder_path = os.path.join(parent_folder, single_folder)
        if _contains_tiff_files(single_folder_path):
            logger.info("Single-folder mode enabled (one subfolder detected, no numeric distance required)")
            return [(0.0, single_folder, single_folder_path)], "single_folder"

    if len(subfolders) == 0 and parent_has_tiffs:
        logger.info("Single-folder mode enabled (TIFF files found directly in parent folder)")
        return [(0.0, os.path.basename(parent_folder), parent_folder)], "single_folder"

    if skipped_subfolders:
        logger.warning("No numeric distance folders were found. Non-numeric subfolders detected")
        for folder in skipped_subfolders:
            logger.warning("  - %s", folder)

    raise ValueError(
        "No valid calibration input found. Provide numeric distance subfolders, "
        "or provide exactly one subfolder with TIFF projections, "
        "or place TIFF projections directly in the parent folder."
    )


def main(parent_folder: str | None = None) -> None:
    """Run the multi-position CT detector calibration workflow.

    For each target folder, loads projections, selects ROIs interactively,
    runs segmentation-based shift estimation, and saves results to JSON/CSV.

    Args:
        parent_folder: Path to directory containing numeric-distance subfolders
            (multi-distance mode) or a single projection folder (single-folder mode).
            If None or empty, the function logs an error and returns.
    """
    logger.info("=" * 60)
    logger.info("CT DETECTOR CENTER CALIBRATION")
    logger.info("=" * 60)

    if parent_folder is None:
        logger.error("parent_folder must be provided")
        return

    # Guard: reject empty/whitespace-only folder inputs before filesystem access.
    if not str(parent_folder).strip():
        logger.error("parent_folder is empty")
        return

    try:
        processing_targets, calibration_mode = _build_processing_targets(parent_folder)
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        return
    
    # Process each folder
    distances = []
    shifts = []
    bg_means = []
    
    for i, (distance, folder, folder_path) in enumerate(processing_targets):
        
        logger.info("%s", "=" * 60)
        logger.info("FOLDER %d/%d: %s", i + 1, len(processing_targets), folder)
        logger.info("Distance: %.2f cm", distance)
        logger.info("%s", "=" * 60)
        
        try:
            # 1. Load projections
            projections = load_images(folder_path)

            # 2. Create collapsed sinogram
            sino_raw = generate_collapsed_sinogram(projections)

            # 3. ROI selection
            roi_bg, roi_obj, calculate_mode = selecionar_roi_interativamente(sino_raw)

            if not calculate_mode:
                logger.info("Skipping folder because 'Already Centered' mode was selected")
                continue

            # 4. Calculate shift
            shift, mask, obj_center, det_min, det_max, bg_mean = calculate_shift_segmentation(
                sino_raw, roi_bg, roi_obj
            )

            # 5. Show results for this folder
            show_results(sino_raw, mask, shift, obj_center, det_min, det_max, calculate_mode)

            distances.append(distance)
            shifts.append(shift)
            bg_means.append(float(bg_mean))
            logger.info("Shift: %.2f pixels", shift)
        except Exception as error:
            # Guard: isolate per-folder failures so one bad folder does not abort full run.
            logger.exception("Calibration failed for folder %s", folder_path)
            logger.error("Error while processing folder '%s': %s", folder, error)
            continue
    
    # Guard: stop before reporting when no valid shifts were produced.
    if len(shifts) == 0:
        logger.error("No valid calibration shifts were computed. Nothing to report")
        return

    # Calculate statistics
    avg_shift, std_shift, avg_bg_mean = compute_statistics(shifts, bg_means)

    # Save results
    save_results(
        parent_folder,
        distances,
        shifts,
        bg_means,
        avg_shift,
        std_shift,
        avg_bg_mean,
        calibration_mode=calibration_mode,
    )

    # Plot summary results only for multi-distance mode
    if calibration_mode == "multi_distance":
        plot_results(distances, shifts, avg_shift, std_shift)
    else:
        logger.info("Skipping summary distance-vs-shift plot in single-folder mode")

    # Final report
    print_final_report(distances, shifts, avg_shift, std_shift, calibration_mode=calibration_mode)


if __name__ == "__main__":
    _setup_logging()
    #main(parent_folder=r"C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_roldanas")
    main(parent_folder=r"C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_roldanas\QRM_maquinar_2")
