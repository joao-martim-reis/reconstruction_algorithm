import numpy as np
import nibabel as nib
import napari


from napari_filters import interactive_filter_viewer, print_applied_filters_summary


def load_nii_volume(nii_path):
    print(f"\nLoading: {nii_path}")
    
    nii = nib.load(nii_path)
    volume = nii.get_fdata()
    
    # Transpose to napari convention (Z, Y, X)
    volume = np.transpose(volume, (2, 1, 0))
    
    # Get voxel size
    voxel_size = nii.header.get_zooms()
    voxel_size = (voxel_size[2], voxel_size[1], voxel_size[0])
    
    print(f"  Shape: {volume.shape}")
    print(f"  Voxel size: ({voxel_size[0]:.3f}, {voxel_size[1]:.3f}, {voxel_size[2]:.3f}) mm")
    
    return volume, voxel_size, nii_path


def main(nift_folder):

    nii_path = nift_folder
    volume, voxel_size, nii_path = load_nii_volume(nii_path)
    

    filter_choice = input("\nUse interactive filtering? (y/n): ").strip().lower()
    print("\nOpening napari...")
    
    if filter_choice == 'y':
        viewer, final_filter_params = interactive_filter_viewer(
            volume,
            name="CT Volume",
            scale=voxel_size,
            nii_filepath=nii_path,
            filtered_output_folder=None,
            allow_saving=False,
        )
        napari.run()
        print_applied_filters_summary(final_filter_params)
    else:
        
        viewer = napari.Viewer()
        viewer.add_image(volume, name="Volume", colormap='gray', scale=voxel_size)
        napari.run()

if __name__ == "__main__":

    nift_input = r"C:\Users\joaomartimreis\Desktop\Joao_CT\Volumes_reconstrucao\reconstructed_volumes_Nift\Versao1_Normalizacao_errada\MAIN_TIGRE_FDK_CROP_Phantom_800_1_20_jan_10h36\Phantom_800_1.nii"
    main(nift_input)
