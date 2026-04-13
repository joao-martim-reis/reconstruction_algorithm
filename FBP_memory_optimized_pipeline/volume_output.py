from pathlib import Path

import napari
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def _sanitize_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    np.nan_to_num(volume, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return volume


def open_volume_in_napari(volume, voxel_size_um):
    voxel_size_mm = float(voxel_size_um) / 1000.0
    volume = _sanitize_volume(volume)

    cmin = float(np.min(volume))
    cmax = float(np.max(volume))
    if cmax <= cmin:
        cmax = cmin + 1.0

    print("\nPHASE 7: NAPARI VISUALIZATION")
    print("Opening Napari viewer...")
    viewer = napari.Viewer()
    viewer.add_image(
        volume,
        scale=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
        name="FBP Volume",
        contrast_limits=(cmin, cmax),
        colormap="gray",
    )
    napari.run()


def export_volume_to_nii(volume, voxel_size_um, output_folder, filename="FBP_volume_stack.nii.gz"):
    if nib is None:
        print("nibabel is not available. Cannot export NIfTI.")
        return None

    voxel_size_mm = float(voxel_size_um) / 1000.0
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = voxel_size_mm
    affine[1, 1] = voxel_size_mm
    affine[2, 2] = voxel_size_mm

    nii_img = nib.Nifti1Image(_sanitize_volume(volume), affine)
    nib.save(nii_img, str(out_path))
    return str(out_path)


def ask_and_export_nii(volume, config):
    choice = input("\nDo you want to export the volume to .nii/.nii.gz? (y/n): ").strip().lower()
    if choice != "y":
        print("NIfTI export skipped")
        return None

    default_name = "FBP_volume_stack.nii"
    user_name = input(f"Output filename [{default_name}]: ").strip()
    filename = user_name if user_name else default_name
    if not filename.endswith(".nii"):
        filename = f"{filename}.nii"

    nii_path = export_volume_to_nii(
        volume,
        voxel_size_um=config["voxel_size_um"],
        output_folder=config["output_folder"],
        filename=filename,
    )

    if nii_path is not None:
        print(f"Volume exported to NIfTI: {nii_path}")
    return nii_path
