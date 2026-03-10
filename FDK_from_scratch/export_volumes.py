import numpy as np
import os
import sys
import tigre
import nibabel as nib
from datetime import datetime

# Import HU conversion function (local copy)
from HU_conversion import HU_conversion


def export_volume_to_nii(volume, geo, source_folder, base_output=None):
    """
    Exports the reconstructed volume to NIfTI format (.nii).
    Creates a unique subfolder for each reconstruction inside base_output.
    """
    if base_output is None:
        base_output = "reconstructed_volumes"
    
    dataset_name = os.path.basename(os.path.normpath(source_folder))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    now = datetime.now()
    month_abbr = now.strftime("%b").lower()
    timestamp = f"{now.day}_{month_abbr}_{now.hour}h{now.minute}"
    
    folder_name = f"{script_name}_{dataset_name}_{timestamp}"
    output_folder = os.path.join(base_output, folder_name)
    
    os.makedirs(output_folder, exist_ok=True)
    
    filename = f"{dataset_name}.nii"
    filepath = os.path.join(output_folder, filename)
    
    volume_export = volume.astype(np.float32)
    volume_export = np.transpose(volume_export, (2, 1, 0))
    
    affine = np.eye(4) # Create identity matrix for affine transformation which is necessary for NIfTI because of its coordinate system
    affine[0, 0] = geo.dVoxel[2] #use the voxel size information from the geometry to preserve spatial accuracy
    affine[1, 1] = geo.dVoxel[1] #use the voxel size information from the geometry to preserve spatial accuracy
    affine[2, 2] = geo.dVoxel[0] #use the voxel size information from the geometry to preserve spatial accuracy
    
    affine[0, 3] = -(volume_export.shape[0] * geo.dVoxel[2]) / 2.0
    affine[1, 3] = -(volume_export.shape[1] * geo.dVoxel[1]) / 2.0
    affine[2, 3] = -(volume_export.shape[2] * geo.dVoxel[0]) / 2.0
    
    nii_img = nib.Nifti1Image(volume_export, affine) #nib.Nifti1Image creates a NIfTI image object
    nii_img.header.set_xyzt_units('mm', 'sec')
    nii_img.header['descrip'] = f'FDK Reconstruction - {dataset_name}'
    
    print(f"    Saving NIfTI file...")
    nib.save(nii_img, filepath)
    print(f"    ✓ File saved successfully!")
    
    metadata_file = os.path.join(output_folder, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Volume Reconstruction Metadata\n")
        f.write(f"Volume Information (exported to NIfTI):\n")
        f.write(f"  Exported shape (X, Y, Z): {volume_export.shape}\n")
        f.write(f"  Original shape (TIGRE): {volume.shape} (Z, Y, X)\n")
        f.write(f"  Data type: {volume_export.dtype}\n")
        f.write(f"  Total size (bytes): {volume_export.nbytes}\n\n")
        f.write(f"  Geometry Information:\n")
        f.write(f"  Voxel size (mm): {geo.dVoxel}\n")
        f.write(f"  Number of voxels: {geo.nVoxel}\n")
        f.write(f"  Physical dimensions (mm): {geo.sVoxel}\n")
        f.write(f"  DSD: {geo.DSD} mm\n")
        f.write(f"  DSO: {geo.DSO} mm\n\n")
        f.write(f"  Python: {sys.version.split()[0]} ({sys.platform})\n")
        f.write(f"  TIGRE version: {getattr(tigre, '__version__', 'unknown')}\n")

    return filepath


def export_volume_HU(original_nii_path, volume, water_val, air_val):
    """
    Converts the volume to HU and saves with _HU suffix.
    Reuses the header/affine from the original .nii file.
    """
    volume_HU = HU_conversion(volume, water_val, air_val)
    original_nii = nib.load(original_nii_path)
    
    volume_HU_export = volume_HU.astype(np.float32)
    volume_HU_export = np.transpose(volume_HU_export, (2, 1, 0))
    
    nii_HU = nib.Nifti1Image(volume_HU_export, original_nii.affine, original_nii.header)
    nii_HU.header['descrip'] = original_nii.header['descrip'].decode() + ' (HU)'
    
    filepath_HU = original_nii_path.replace('.nii', '_HU.nii')
    nib.save(nii_HU, filepath_HU)
    
    filename_HU = os.path.basename(filepath_HU)
    print(f"    ✓ Volume in Hounsfield Units saved: {filename_HU}")
    
    return filepath_HU