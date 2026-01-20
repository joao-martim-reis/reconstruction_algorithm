import numpy as np
import nibabel as nib
import napari
import sys
import os

from napari_filters import interactive_filter_viewer, print_applied_filters_summary


def load_nii_volume(nii_path):
    """Carrega volume .nii e retorna os dados."""
    print(f"\nLoading: {nii_path}")
    
    nii = nib.load(nii_path)
    volume = nii.get_fdata()
    
    # Transpose to napari convention (Z, Y, X)
    volume = np.transpose(volume, (2, 1, 0))
    
    # Get voxel size
    voxel_size = nii.header.get_zooms()
    voxel_size = (voxel_size[2], voxel_size[1], voxel_size[0])
    
    print(f"  Shape: {volume.shape}")
    print(f"  Voxel size: {voxel_size} mm")
    
    return volume, voxel_size, nii_path


def main(nift_folder):
    # Get path
    
    nii_path = nift_folder
    # Load volume
    volume, voxel_size, nii_path = load_nii_volume(nii_path)
    
    # Ask about filtering
    filter_choice = input("\nUse interactive filtering? (y/n): ").strip().lower()
    
    print("\nOpening napari...")
    
    if filter_choice == 'y':
        # Create output folder
        output_folder = os.path.join(os.path.dirname(nii_path), "filtered_volumes")
        os.makedirs(output_folder, exist_ok=True)
        
        # Open with filters
        viewer = interactive_filter_viewer(
            volume, 
            name="CT Volume", 
            scale=voxel_size, 
            nii_filepath=nii_path, 
            filtered_output_folder=output_folder
        )
        napari.run()
        
        # Print summary
        print_applied_filters_summary(viewer.current_filter_params)
    else:
        # Just view
        viewer = napari.Viewer()
        viewer.add_image(volume, name="Volume", colormap='gray', scale=voxel_size)
        napari.run()


if __name__ == "__main__":
    main(
        nift_folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift\MAIN_TIGRE_FDK_CROP_Phantom_800_1_20_jan_10h36'

    )
