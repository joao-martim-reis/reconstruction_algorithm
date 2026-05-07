"""NIfTI export utilities for CT reconstructed volumes."""

import logging
import os
import sys

import nibabel as nib
import numpy as np
from datetime import datetime

from HU_conversion import HU_conversion

logger = logging.getLogger(__name__)


def export_volume_to_nii(
    volume: np.ndarray,
    geo,
    source_folder: str,
    base_output: str | None = None,
) -> str:
    """Export a reconstructed volume to NIfTI format (.nii).

    Creates a unique timestamped subfolder inside base_output and writes both
    the .nii file and a metadata.txt with geometry parameters.

    Args:
        volume: Reconstructed volume, shape (nz, ny, nx), float32. TIGRE axis order.
        geo: Geometry object (TIGRE or SimpleNamespace) with attributes dVoxel,
            nVoxel, sVoxel, DSD, DSO (all in mm).
        source_folder: Path to the original TIFF projections folder; its basename
            is used to name the output file.
        base_output: Root directory for output. Defaults to "reconstructed_volumes".

    Returns:
        filepath: Absolute path to the saved .nii file.
    """
    if base_output is None:
        base_output = "reconstructed_volumes"

    dataset_name = os.path.basename(os.path.normpath(source_folder))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{dataset_name}_{timestamp}"
    output_folder = os.path.join(base_output, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    filepath = os.path.join(output_folder, f"{dataset_name}.nii")

    # Transpose TIGRE (nz, ny, nx) → NIfTI (nx, ny, nz) for conventional RAS orientation
    volume_export = np.transpose(volume.astype(np.float32), (2, 1, 0))

    # Build diagonal affine: voxel size in mm along each axis, origin at image centre
    affine = np.eye(4)
    affine[0, 0] = geo.dVoxel[2]
    affine[1, 1] = geo.dVoxel[1]
    affine[2, 2] = geo.dVoxel[0]
    affine[0, 3] = -(volume_export.shape[0] * geo.dVoxel[2]) / 2.0
    affine[1, 3] = -(volume_export.shape[1] * geo.dVoxel[1]) / 2.0
    affine[2, 3] = -(volume_export.shape[2] * geo.dVoxel[0]) / 2.0

    nii_img = nib.Nifti1Image(volume_export, affine)
    nii_img.header.set_xyzt_units('mm', 'sec')
    nii_img.header['descrip'] = f'FDK Reconstruction - {dataset_name}'

    nib.save(nii_img, filepath)
    logger.info("NIfTI saved: %s", filepath)

    metadata_path = os.path.join(output_folder, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("Volume Reconstruction Metadata\n")
        f.write(f"  Exported shape (X, Y, Z): {volume_export.shape}\n")
        f.write(f"  Original shape (TIGRE): {volume.shape} (Z, Y, X)\n")
        f.write(f"  Data type: {volume_export.dtype}\n")
        f.write(f"  Total size (bytes): {volume_export.nbytes}\n\n")
        f.write(f"  Voxel size (mm): {geo.dVoxel}\n")
        f.write(f"  Number of voxels: {geo.nVoxel}\n")
        f.write(f"  Physical dimensions (mm): {geo.sVoxel}\n")
        f.write(f"  DSD: {geo.DSD} mm\n")
        f.write(f"  DSO: {geo.DSO} mm\n\n")
        f.write(f"  Python: {sys.version.split()[0]} ({sys.platform})\n")

    return filepath


def export_volume_HU(
    original_nii_path: str,
    volume: np.ndarray,
    water_val: float,
    air_val: float,
) -> str:
    """Convert volume to Hounsfield Units and save as a new .nii file.

    Reuses the header and affine from the original .nii file. The output
    filename is the original name with a '_HU' suffix.

    Args:
        original_nii_path: Path to the previously saved .nii reconstruction.
        volume: Same volume that was exported, shape (nz, ny, nx), float32.
        water_val: Mean intensity of water in the reconstruction.
        air_val: Mean intensity of air in the reconstruction.

    Returns:
        filepath_HU: Path to the saved HU .nii file.
    """
    volume_HU = HU_conversion(volume, water_val, air_val)
    original_nii = nib.load(original_nii_path)

    volume_HU_export = np.transpose(volume_HU.astype(np.float32), (2, 1, 0))
    nii_HU = nib.Nifti1Image(volume_HU_export, original_nii.affine, original_nii.header)
    nii_HU.header['descrip'] = original_nii.header['descrip'].decode() + ' (HU)'

    filepath_HU = original_nii_path.replace('.nii', '_HU.nii')
    nib.save(nii_HU, filepath_HU)
    logger.info("HU volume saved: %s", os.path.basename(filepath_HU))

    return filepath_HU
