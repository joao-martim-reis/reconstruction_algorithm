"""
Simplified Napari Filtering Module for FDK_crop_data
=====================================================
Simplified version with essential filters and interactive controls.
"""

import numpy as np
import napari
from scipy import ndimage
from skimage import restoration
import nibabel as nib
import os
from magicgui import magicgui


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def apply_gaussian_filter(volume, sigma=1.0):
    """Apply Gaussian smoothing filter to reduce noise."""
    print(f"    Applying Gaussian filter (sigma={sigma})...")
    filtered = ndimage.gaussian_filter(volume, sigma=sigma)
    return filtered


def apply_median_filter(volume, size=3):
    """Apply median filter to remove salt-and-pepper noise while preserving edges."""
    print(f"    Applying median filter (size={size})...")
    
    if volume.shape[0] > 100:
        filtered = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            filtered[i] = ndimage.median_filter(volume[i], size=size)
            if i % 20 == 0:
                print(f"      Processed {i}/{volume.shape[0]} slices...")
    else:
        filtered = ndimage.median_filter(volume, size=size)
    
    return filtered


def apply_bilateral_filter(volume, sigma_spatial=2.0, sigma_intensity=None):
    """Apply bilateral filter to smooth while preserving edges."""
    print(f"    Applying bilateral filter (spatial={sigma_spatial})...")
    
    if sigma_intensity is None:
        sigma_intensity = np.std(volume) * 0.1
    
    filtered = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        filtered[i] = restoration.denoise_bilateral(
            volume[i],
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_intensity,
            channel_axis=None
        )
        if i % 10 == 0:
            print(f"      Processed {i}/{volume.shape[0]} slices...")
    
    return filtered


# ============================================================================
# NAPARI INTERACTIVE VIEWER
# ============================================================================

def interactive_filter_viewer(volume, name="CT Volume", scale=None, nii_filepath=None):
    """
    Open napari viewer with interactive filter controls using sliders.
    Simplified version with Gaussian, Median, and Bilateral filters only.
    
    Args:
        volume: 3D numpy array
        name: Name for the volume layer
        scale: Tuple of voxel sizes (z, y, x)
        nii_filepath: Path to original .nii file (for saving filtered version)
    
    Returns:
        napari viewer object
    """
    print(f"\n==> Opening interactive napari viewer with filter controls...")
    print(f"    Volume shape: {volume.shape}")
    print(f"    Value range: [{np.min(volume):.6f}, {np.max(volume):.6f}]")
    
    viewer = napari.Viewer()
    
    # Add original and filtered layers
    if scale is not None:
        original_layer = viewer.add_image(volume, name="Original", colormap='gray', visible=True, scale=scale, opacity=0.5)
        filtered_layer = viewer.add_image(volume.copy(), name="Filtered", colormap='gray', scale=scale)
    else:
        original_layer = viewer.add_image(volume, name="Original", colormap='gray', visible=True, opacity=0.5)
        filtered_layer = viewer.add_image(volume.copy(), name="Filtered", colormap='gray')
    
    # Store current filter parameters
    current_params = {
        'filter_type': 'None',
        'gaussian_sigma': 1.0,
        'median_size': 3,
        'bilateral_spatial': 2.0
    }
    
    # Create interactive widget with sliders
    @magicgui(
        auto_call=False,
        call_button="Apply Filter",
        filter_type={
            'label': 'Filter Type',
            'choices': ['None', 'Gaussian', 'Median', 'Bilateral']
        },
        gaussian_sigma={
            'label': 'Gaussian Sigma',
            'widget_type': 'FloatSlider',
            'min': 0.1,
            'max': 5.0,
            'step': 0.1
        },
        median_size={
            'label': 'Median Size',
            'widget_type': 'Slider',
            'min': 3,
            'max': 9,
            'step': 2
        },
        bilateral_spatial={
            'label': 'Bilateral Spatial Sigma',
            'widget_type': 'FloatSlider',
            'min': 0.5,
            'max': 5.0,
            'step': 0.1
        }
    )
    def apply_filter(
        filter_type: str = 'None',
        gaussian_sigma: float = 1.0,
        median_size: int = 3,
        bilateral_spatial: float = 2.0
    ):
        """Apply selected filter with current parameters."""
        
        # Update current params
        current_params['filter_type'] = filter_type
        current_params['gaussian_sigma'] = gaussian_sigma
        current_params['median_size'] = median_size
        current_params['bilateral_spatial'] = bilateral_spatial
        
        print(f"\n--> Applying filter: {filter_type}")
        
        if filter_type == 'None':
            filtered_data = volume.copy()
            print("    No filter applied (showing original)")
            
        elif filter_type == 'Gaussian':
            print(f"    Sigma: {gaussian_sigma}")
            filtered_data = ndimage.gaussian_filter(volume, sigma=gaussian_sigma)
            
        elif filter_type == 'Median':
            print(f"    Size: {median_size}")
            if volume.shape[0] > 100:
                filtered_data = np.zeros_like(volume)
                for i in range(volume.shape[0]):
                    filtered_data[i] = ndimage.median_filter(volume[i], size=median_size)
            else:
                filtered_data = ndimage.median_filter(volume, size=median_size)
                
        elif filter_type == 'Bilateral':
            print(f"    Spatial Sigma: {bilateral_spatial}")
            sigma_intensity = np.std(volume) * 0.1
            filtered_data = np.zeros_like(volume)
            for i in range(volume.shape[0]):
                filtered_data[i] = restoration.denoise_bilateral(
                    volume[i],
                    sigma_spatial=bilateral_spatial,
                    sigma_color=sigma_intensity,
                    channel_axis=None
                )
        
        # Update the filtered layer
        filtered_layer.data = filtered_data
        print("    ✓ Filter applied! Adjust sliders and click 'Apply Filter' again to update")
    
    # Create save button
    @magicgui(call_button="💾 Save Filtered Volume")
    def save_results():
        """Save the filtered volume and parameter values."""
        if current_params['filter_type'] == 'None':
            print("\n⚠ No filter applied - nothing to save")
            return
            
        if nii_filepath is None:
            print("\n⚠ No original .nii filepath provided - cannot save")
            return
        
        # Save filtered volume
        filtered_path = nii_filepath.replace('.nii', '_filtered.nii')
        filtered_data = filtered_layer.data
        save_filtered_volume(filtered_data, filtered_path, nii_filepath)
        
        # Save parameters to text file
        params_path = nii_filepath.replace('.nii', '_filter_params.txt')
        with open(params_path, 'w') as f:
            f.write("FILTER PARAMETERS\n")
            f.write("="*50 + "\n")
            f.write(f"Filter Type: {current_params['filter_type']}\n\n")
            
            if current_params['filter_type'] == 'Gaussian':
                f.write(f"Gaussian Sigma: {current_params['gaussian_sigma']}\n")
            elif current_params['filter_type'] == 'Median':
                f.write(f"Median Size: {current_params['median_size']}\n")
            elif current_params['filter_type'] == 'Bilateral':
                f.write(f"Bilateral Spatial Sigma: {current_params['bilateral_spatial']}\n")
        
        print(f"\n✓ Parameters saved to: {params_path}")
        print(f"✓ Filtered volume saved to: {filtered_path}")
    
    # Add widgets to viewer
    viewer.window.add_dock_widget(apply_filter, area='right', name='🎛️ Filter Controls')
    viewer.window.add_dock_widget(save_results, area='right', name='💾 Save')
    
    print(f"\n    ✓ Interactive viewer ready!")
    print(f"\n    📋 INSTRUCTIONS:")
    print(f"    1. Select filter type from dropdown")
    print(f"    2. Adjust sliders for that filter")
    print(f"    3. Click 'Apply Filter' to see results")
    print(f"    4. Repeat until satisfied")
    print(f"    5. Click 'Save' button to save volume & parameters")
    print(f"    6. Toggle 'Original' layer to compare")
    
    return viewer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_filtered_volume(volume, output_path, original_nii_path=None):
    """Save filtered volume to NIfTI format."""
    print(f"\n==> Saving filtered volume to: {output_path}")
    
    volume_export = volume.astype(np.float32)
    volume_export = np.transpose(volume_export, (2, 1, 0))
    
    if original_nii_path and os.path.exists(original_nii_path):
        original_nii = nib.load(original_nii_path)
        nii_img = nib.Nifti1Image(volume_export, original_nii.affine, original_nii.header)
        nii_img.header['descrip'] = original_nii.header['descrip'].decode() + ' (filtered)'
    else:
        affine = np.eye(4)
        nii_img = nib.Nifti1Image(volume_export, affine)
        nii_img.header.set_xyzt_units('mm', 'sec')
        nii_img.header['descrip'] = 'Filtered CT Volume'
    
    nib.save(nii_img, output_path)
    print(f"    ✓ Filtered volume saved successfully!")
    
    return output_path
