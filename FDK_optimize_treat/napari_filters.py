"""
Napari Visualization and Artifact Filtering Module
====================================================
This module provides interactive visualization and filtering tools for CT reconstructed volumes.
Can be used both interactively in napari and programmatically in reconstruction pipelines.
"""

import numpy as np
import napari
from scipy import ndimage
from skimage import restoration
from skimage.filters import threshold_otsu
import warnings
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
    print(f"    Applying 3D median filter (size={size})...")
    
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


def remove_ring_artifacts_polar(volume, filter_size=5):
    """Remove ring artifacts using polar coordinate transformation."""
    print(f"    Removing ring artifacts (polar method)...")
    
    filtered = np.zeros_like(volume)
    
    for i in range(volume.shape[0]):
        slice_data = volume[i]
        center_y, center_x = np.array(slice_data.shape) // 2
        
        y, x = np.indices(slice_data.shape)
        y = y - center_y
        x = x - center_x
        r = np.sqrt(x**2 + y**2)
        
        r_flat = r.flatten()
        indices = np.argsort(r_flat)
        sorted_data = slice_data.flatten()[indices]
        
        filtered_data = ndimage.median_filter(sorted_data, size=filter_size)
        
        result = np.zeros_like(sorted_data)
        result[indices] = filtered_data
        filtered[i] = result.reshape(slice_data.shape)
        
        if i % 20 == 0:
            print(f"      Processed {i}/{volume.shape[0]} slices...")
    
    return filtered


def apply_combined_filter(volume, gaussian_sigma=1.0, median_size=3):
    """Apply combination of Gaussian and median filters for robust denoising."""
    print(f"    Applying combined filter (Gaussian + Median)...")
    filtered = apply_gaussian_filter(volume, sigma=gaussian_sigma)
    filtered = apply_median_filter(filtered, size=median_size)
    return filtered


# ============================================================================
# NAPARI INTERACTIVE FUNCTIONS
# ============================================================================

def interactive_filter_viewer(volume, name="CT Volume", scale=None, nii_filepath=None):
    """
    Open napari viewer with interactive filter controls using sliders.
    Adjust parameters in real-time and save when satisfied.
    
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
    
    # Add original volume (visible for comparison)
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
        'bilateral_spatial': 2.0,
        'ring_filter_size': 5,
        'combined_gaussian': 1.0,
        'combined_median': 3
    }
    
    # Create interactive widget with sliders
    @magicgui(
        auto_call=False,
        call_button="Apply Filter",
        filter_type={
            'label': 'Filter Type',
            'choices': ['None', 'Gaussian', 'Median', 'Bilateral', 'Ring Removal', 'Combined']
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
        },
        ring_filter_size={
            'label': 'Ring Filter Size',
            'widget_type': 'Slider',
            'min': 3,
            'max': 11,
            'step': 2
        },
        combined_gaussian={
            'label': 'Combined: Gaussian',
            'widget_type': 'FloatSlider',
            'min': 0.1,
            'max': 3.0,
            'step': 0.1
        },
        combined_median={
            'label': 'Combined: Median',
            'widget_type': 'Slider',
            'min': 3,
            'max': 7,
            'step': 2
        }
    )
    def apply_filter(
        filter_type: str = 'None',
        gaussian_sigma: float = 1.0,
        median_size: int = 3,
        bilateral_spatial: float = 2.0,
        ring_filter_size: int = 5,
        combined_gaussian: float = 1.0,
        combined_median: int = 3
    ):
        """Apply selected filter with current parameters."""
        
        # Update current params
        current_params['filter_type'] = filter_type
        current_params['gaussian_sigma'] = gaussian_sigma
        current_params['median_size'] = median_size
        current_params['bilateral_spatial'] = bilateral_spatial
        current_params['ring_filter_size'] = ring_filter_size
        current_params['combined_gaussian'] = combined_gaussian
        current_params['combined_median'] = combined_median
        
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
                
        elif filter_type == 'Ring Removal':
            print(f"    Filter Size: {ring_filter_size}")
            filtered_data = np.zeros_like(volume)
            for i in range(volume.shape[0]):
                slice_data = volume[i]
                center_y, center_x = np.array(slice_data.shape) // 2
                y, x = np.indices(slice_data.shape)
                y = y - center_y
                x = x - center_x
                r = np.sqrt(x**2 + y**2)
                r_flat = r.flatten()
                indices = np.argsort(r_flat)
                sorted_data = slice_data.flatten()[indices]
                filtered_data_1d = ndimage.median_filter(sorted_data, size=ring_filter_size)
                result = np.zeros_like(sorted_data)
                result[indices] = filtered_data_1d
                filtered_data[i] = result.reshape(slice_data.shape)
                
        elif filter_type == 'Combined':
            print(f"    Gaussian: {combined_gaussian}, Median: {combined_median}")
            filtered_data = ndimage.gaussian_filter(volume, sigma=combined_gaussian)
            if volume.shape[0] > 100:
                temp = np.zeros_like(filtered_data)
                for i in range(filtered_data.shape[0]):
                    temp[i] = ndimage.median_filter(filtered_data[i], size=combined_median)
                filtered_data = temp
            else:
                filtered_data = ndimage.median_filter(filtered_data, size=combined_median)
        
        # Update the filtered layer
        filtered_layer.data = filtered_data
        print("    ✓ Filter applied! Adjust sliders and click 'Apply Filter' again to update")
    
    # Create save button
    @magicgui(call_button="💾 Save Filtered Volume & Parameters")
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
            elif current_params['filter_type'] == 'Ring Removal':
                f.write(f"Ring Filter Size: {current_params['ring_filter_size']}\n")
            elif current_params['filter_type'] == 'Combined':
                f.write(f"Combined Gaussian Sigma: {current_params['combined_gaussian']}\n")
                f.write(f"Combined Median Size: {current_params['combined_median']}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("To reproduce this filter:\n")
            
            if current_params['filter_type'] == 'Gaussian':
                f.write(f"apply_gaussian_filter(volume, sigma={current_params['gaussian_sigma']})\n")
            elif current_params['filter_type'] == 'Median':
                f.write(f"apply_median_filter(volume, size={current_params['median_size']})\n")
            elif current_params['filter_type'] == 'Bilateral':
                f.write(f"apply_bilateral_filter(volume, sigma_spatial={current_params['bilateral_spatial']})\n")
            elif current_params['filter_type'] == 'Ring Removal':
                f.write(f"remove_ring_artifacts_polar(volume, filter_size={current_params['ring_filter_size']})\n")
            elif current_params['filter_type'] == 'Combined':
                f.write(f"apply_combined_filter(volume, gaussian_sigma={current_params['combined_gaussian']}, median_size={current_params['combined_median']})\n")
        
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


def view_volume_with_filters(volume, name="CT Volume", scale=None):
    """
    Open napari viewer with the volume.
    
    Args:
        volume: 3D numpy array
        name: Name for the volume layer
        scale: Tuple of voxel sizes (z, y, x)
    
    Returns:
        napari viewer object
    """
    print(f"\n==> Opening napari viewer...")
    print(f"    Volume shape: {volume.shape}")
    print(f"    Value range: [{np.min(volume):.6f}, {np.max(volume):.6f}]")
    
    viewer = napari.Viewer()
    
    if scale is not None:
        viewer.add_image(volume, name=name, colormap='gray', scale=scale)
    else:
        viewer.add_image(volume, name=name, colormap='gray')
    
    print(f"    ✓ Napari viewer opened!")
    
    return viewer


def add_filtered_layer(viewer, volume, filter_name, filter_func, scale=None, **filter_params):
    """Apply a filter and add it as a new layer in napari viewer."""
    print(f"\n==> Applying filter: {filter_name}")
    filtered_volume = filter_func(volume, **filter_params)
    
    if scale is not None:
        layer = viewer.add_image(filtered_volume, name=filter_name, colormap='gray', scale=scale)
    else:
        layer = viewer.add_image(filtered_volume, name=filter_name, colormap='gray')
    
    print(f"    ✓ Filtered layer added to napari")
    return layer, filtered_volume


def compare_filters_napari(volume, filters_config, scale=None):
    """
    Open napari viewer with multiple filtered versions for comparison.
    
    Args:
        volume: 3D numpy array
        filters_config: List of tuples (name, filter_func, params_dict)
        scale: Tuple of voxel sizes (z, y, x)
    
    Returns:
        napari viewer object and dict of filtered volumes
    """
    print(f"\n==> Comparing {len(filters_config)} filters in napari...")
    
    viewer = napari.Viewer()
    filtered_volumes = {}
    
    # Add original
    if scale is not None:
        viewer.add_image(volume, name="Original", colormap='gray', visible=False, scale=scale)
    else:
        viewer.add_image(volume, name="Original", colormap='gray', visible=False)
    
    # Apply and add each filter
    for filter_name, filter_func, params in filters_config:
        print(f"\n  Processing: {filter_name}")
        filtered = filter_func(volume, **params)
        filtered_volumes[filter_name] = filtered
        
        if scale is not None:
            viewer.add_image(filtered, name=filter_name, colormap='gray', visible=True, scale=scale)
        else:
            viewer.add_image(filtered, name=filter_name, colormap='gray', visible=True)
    
    print(f"\n    ✓ All filters applied!")
    print(f"    Use layer visibility toggles to compare results")
    
    return viewer, filtered_volumes


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_filter_recommendations(volume):
    """Analyze volume and provide filter recommendations."""
    print(f"\n==> Analyzing volume for filter recommendations...")
    
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    snr = mean_val / std_val if std_val > 0 else 0
    
    noise_estimate = np.median(np.abs(volume - ndimage.median_filter(volume, size=3)))
    
    print(f"    Volume statistics:")
    print(f"      Mean: {mean_val:.6f}")
    print(f"      Std:  {std_val:.6f}")
    print(f"      SNR estimate: {snr:.2f}")
    print(f"      Noise estimate: {noise_estimate:.6f}")
    
    recommendations = {}
    
    if snr < 5:
        recommendations['primary'] = {
            'name': 'Bilateral Filter',
            'function': apply_bilateral_filter,
            'params': {'sigma_spatial': 2.0},
            'reason': 'High noise level - bilateral filter preserves edges'
        }
    elif snr < 10:
        recommendations['primary'] = {
            'name': 'Combined Filter',
            'function': apply_combined_filter,
            'params': {'gaussian_sigma': 1.0, 'median_size': 3},
            'reason': 'Moderate noise - combined filter balances quality and speed'
        }
    else:
        recommendations['primary'] = {
            'name': 'Gaussian Filter',
            'function': apply_gaussian_filter,
            'params': {'sigma': 1.0},
            'reason': 'Low noise - simple Gaussian sufficient'
        }
    
    recommendations['ring_artifacts'] = {
        'name': 'Ring Artifact Removal',
        'function': remove_ring_artifacts_polar,
        'params': {'filter_size': 5},
        'reason': 'Common CT artifact - apply if rings visible'
    }
    
    print(f"\n    RECOMMENDATIONS:")
    for key, rec in recommendations.items():
        print(f"      {rec['name']}: {rec['reason']}")
    
    return recommendations


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


# ============================================================================
# QUICK REFERENCE
# ============================================================================

FILTER_GUIDE = """
FILTER SELECTION GUIDE:
-----------------------
1. Gaussian (fast) - General smoothing
   apply_gaussian_filter(volume, sigma=1.0)

2. Median (fast) - Remove outliers, preserve edges
   apply_median_filter(volume, size=3)

3. Bilateral (moderate) - Edge-preserving smoothing
   apply_bilateral_filter(volume, sigma_spatial=2.0)

4. Ring removal (moderate) - CT-specific artifacts
   remove_ring_artifacts_polar(volume, filter_size=5)

5. Combined (fast) - Robust general-purpose
   apply_combined_filter(volume, gaussian_sigma=1.0, median_size=3)
"""


if __name__ == "__main__":
    print(__doc__)
    print(FILTER_GUIDE)
