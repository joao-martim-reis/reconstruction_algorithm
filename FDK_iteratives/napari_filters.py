import numpy as np
import napari
from scipy import ndimage
from skimage import restoration
import nibabel as nib
import os
from magicgui import magicgui
from tqdm import tqdm


# FILTER FUNCTIONS

def apply_gaussian_filter(volume, sigma=1.0):
    """Gaussian Smoothing Filter - removes noise by averaging nearby voxels.
    
    WHEN TO USE:  Quick noise reduction for preview; High-frequency noise (speckle, random variations)

    TRADE-OFFS: Very fast (entire volume processed at once)
    Blurs edges and fine details; Loss of sharpness
    
    PARAMETER:
    sigma: Amount of smoothing (0.1-5.0)
           - Low (0.5-1.0): Subtle smoothing, preserves details
           - Medium (1.0-2.0): Balanced noise reduction
           - High (2.0-5.0): Strong smoothing, may lose features
    """
    print(f"    Applying Gaussian filter (sigma={sigma})...")
    filtered = ndimage.gaussian_filter(volume, sigma=sigma)
    return filtered


def apply_bilateral_filter(volume, sigma_spatial=2.0, sigma_intensity=None):
    """Bilateral Filter - edge-preserving smoothing (slower but better quality).
    
    WHEN TO USE: Need to reduce noise WITHOUT blurring edges; Preserving anatomical boundaries is critical
    Better quality than Gaussian (worth the wait)
    
    TRADE-OFFS:
    Preserves edges and fine structures
    Much slower (processes slice-by-slice)
    
    PARAMETER:
    sigma_spatial: Spatial smoothing strength (0.5-5.0)
                   - Low (0.5-1.5): Minimal smoothing, max detail
                   - Medium (1.5-3.0): Balanced (recommended)
                   - High (3.0-5.0): Strong smoothing
    """
    print(f"    Applying bilateral filter (spatial={sigma_spatial})...")
    
    if sigma_intensity is None:
        sigma_intensity = np.std(volume) * 0.1
    
    filtered = np.zeros_like(volume) # Initialize output volume
    for i in range(volume.shape[0]): # Process slice-by-slice
        filtered[i] = restoration.denoise_bilateral(
            volume[i],
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_intensity,
            channel_axis=None
        ) #restoration.denoise_bilateral is used to reduce noise in images while preserving edges by applying a bilateral filter.

    return filtered


def apply_beam_hardening_correction(volume, strength=0.3):
    """Beam Hardening Correction - compensates for non-linear X-ray attenuation.
    
    Beam hardening occurs when low-energy X-rays are absorbed more than high-energy ones, causing darker centers and cupping artifacts.
    
    WHEN TO USE: Visible dark streaks between dense objects; Non-uniform intensity in homogeneous materials
    
    LIMITATIONS (SIMPLIFIED APPROXIMATION):
    Uses basic polynomial model (beam hardening is material-specific)
    Works best for single-material phantoms
    
    PARAMETER:
    strength: Correction intensity (0.0-1.0)
              - Low (0.1-0.3): Subtle correction (recommended start)
              - Medium (0.3-0.5): Moderate correction
              - High (0.5-1.0): Aggressive (may overcorrect)
    """
    
    vol_min = np.min(volume)
    vol_max = np.max(volume)
    vol_norm = (volume - vol_min) / (vol_max - vol_min + 1e-10) # Normalize to [0, 1]
    
    # Polynomial correction: compensates non-linear attenuation
    # Second-order polynomial: I_corrected = I + strength * (I^2 - I)
    correction = strength * (vol_norm**2 - vol_norm)
    vol_corrected = vol_norm + correction
    
    filtered = vol_corrected * (vol_max - vol_min) + vol_min
    return filtered


def apply_cupping_correction(volume, strength=0.5):
    """Cupping Artifact Correction - increases center intensity radially.
    
    Cupping = darker center, brighter edges ("cup" intensity profile)
    Caused by scatter radiation, beam hardening, or detector effects.
    
    WHEN TO USE: Center systematically darker than edges

    ASSUMPTIONS: Object is CENTERED in field-of-view; Artifact is RADIALLY SYMMETRIC

    MAY NOT WORK WELL FOR: Non-circular/irregular phantoms; Asymmetric artifacts
    
    PARAMETER:
    strength: Correction intensity (0.0-1.0)
              - Low (0.2-0.4): Subtle correction
              - Medium (0.4-0.6): Typical cupping
              - High (0.6-1.0): Severe cupping
    """

    filtered = np.zeros_like(volume)
    
    for i in tqdm(range(volume.shape[0]), desc="    Processing slices", ncols=80): #tqdm is used to create a progress bar for loops
        slice_data = volume[i].astype(np.float32) # Ensure float for calculations
        
        center_y, center_x = np.array(slice_data.shape) // 2 # Center coordinates
        
        y, x = np.indices(slice_data.shape)
        y = y - center_y
        x = x - center_x
        r = np.sqrt(x**2 + y**2) # Radial distance from center
        
        max_radius = np.sqrt(center_y**2 + center_x**2) # Max possible radius
        r_norm = r / (max_radius + 1e-10) # Normalize to [0, 1]
        
        # Quadratic correction: boost center intensity
        correction_map = 1.0 + strength * (1.0 - r_norm**2)
        filtered[i] = slice_data * correction_map
    
    print(f"    ✓ Done")
    return filtered






# NAPARI INTERACTIVE VIEWER

def interactive_filter_viewer(volume, name="CT Volume", scale=None, nii_filepath=None, filtered_output_folder=None):
    """
    Open napari viewer with interactive filter controls using sliders.
    
    Args:
        volume: 3D volume array
        name: Name for the volume layer
        scale: Voxel size tuple (z, y, x)
        nii_filepath: Path to original .nii file (for metadata)
        filtered_output_folder: Custom folder path for saving filtered volumes (optional)
    """
    print(f"\n==> Opening interactive napari viewer with filter controls...")
    print(f"    Volume shape: {volume.shape}")
    print(f"    Value range: [{np.min(volume):.6f}, {np.max(volume):.6f}]")
    
    viewer = napari.Viewer()
    
    # Add original and filtered layers
    # scale is a tuple with voxel size in each dimension
    original_layer = viewer.add_image(volume, name="Original", colormap='gray', visible=True, scale=scale, opacity=0.5)
    filtered_layer = viewer.add_image(volume.copy(), name="Filtered", colormap='gray', scale=scale)
   
    # Store current filter parameters
    current_params = {
        'filter_type': 'None',
        'gaussian_sigma': 1.0,
        'bilateral_spatial': 2.0,
        'beam_hardening_strength': 0.3,
        'cupping_strength': 0.5
    }
    
    # Create interactive widget with sliders
    # magicgui is used to create graphical user interfaces (GUIs) for functions, allowing users to interactively adjust parameters and see results in real-time.
    @magicgui(
        auto_call=False,
        call_button="Apply Filter",
        filter_type={
            'label': 'Filter Type',
            'choices': ['None', 'Gaussian', 'Bilateral', 'Beam Hardening', 'Cupping']
        },
        gaussian_sigma={
            'label': 'Gaussian Sigma',
            'widget_type': 'FloatSlider',
            'min': 0.1,
            'max': 5.0,
            'step': 0.1
        },
        bilateral_spatial={
            'label': 'Bilateral Spatial Sigma',
            'widget_type': 'FloatSlider',
            'min': 0.5,
            'max': 5.0,
            'step': 0.1
        },
        beam_hardening_strength={
            'label': 'Beam Hardening Strength',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.05
        },
        cupping_strength={
            'label': 'Cupping Correction Strength',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.05
        }
    )




    def apply_filter(
        filter_type: str = 'None',
        gaussian_sigma: float = 1.0,
        bilateral_spatial: float = 2.0,
        beam_hardening_strength: float = 0.3,
        cupping_strength: float = 0.5
    ):
        """Apply selected filter with current parameters."""
        
        # Update current params
        current_params['filter_type'] = filter_type
        current_params['gaussian_sigma'] = gaussian_sigma
        current_params['bilateral_spatial'] = bilateral_spatial
        current_params['beam_hardening_strength'] = beam_hardening_strength
        current_params['cupping_strength'] = cupping_strength
        
        print(f"\n--> Applying filter: {filter_type}")
        
        # FILTER_MAP maps the filter name chosen in the GUI to a callable
        # that applies that filter and returns a tuple (filtered_volume, optional_message).
        # The lambdas capture current slider values by closure so they use
        # the up-to-date parameters when the filter is run.
        FILTER_MAP = {
            'None': lambda v: (v.copy(), "No filter applied (showing original)"),
            'Gaussian': lambda v: (apply_gaussian_filter(v, sigma=gaussian_sigma), None),
            'Bilateral': lambda v: (apply_bilateral_filter(v, sigma_spatial=bilateral_spatial), None),
            'Beam Hardening': lambda v: (apply_beam_hardening_correction(v, strength=beam_hardening_strength), None),
            'Cupping': lambda v: (apply_cupping_correction(v, strength=cupping_strength), None)
        }
        
        # Apply filter using dictionary lookup
        if filter_type in FILTER_MAP:
            filtered_data, msg = FILTER_MAP[filter_type](volume)
            if msg:
                print(f"    {msg}")
        else:
            filtered_data = volume.copy()
            
        # Update the filtered layer
        filtered_layer.data = filtered_data
        print(f"    ✓ Filter applied! Adjust sliders and click 'Apply Filter' to update")
    
    # Create save button
    @magicgui(call_button="Save Filtered Volume")

    def save_results():
        
        if current_params['filter_type'] == 'None':
            print("\n No filter applied - nothing to save")
            return
    
        # Save filtered volume (parameters saved inside save_filtered_volume)
        filtered_data = filtered_layer.data
        save_filtered_volume(filtered_data, nii_filepath, filtered_output_folder, current_params)
    
    # Add widgets to viewer
    viewer.window.add_dock_widget(apply_filter, area='right', name=' Filter Controls')
    viewer.window.add_dock_widget(save_results, area='right', name=' Save')

    # Return viewer and current params (don't attach to Viewer object)
    return viewer, current_params


# UTILITY FUNCTIONS

def print_applied_filters_summary(filter_params):
    """Print summary of filters applied during napari session."""
    if filter_params['filter_type'] == 'None':
        print("\n" + "="*60)
        print("FILTER SUMMARY: No filters were applied")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("FILTER SUMMARY - Applied Filters:")
    print("="*60)
    print(f"Filter Type: {filter_params['filter_type']}")
    print("")
    
    if filter_params['filter_type'] == 'Gaussian':
        print(f"  - Gaussian Sigma: {filter_params['gaussian_sigma']:.2f}")
    elif filter_params['filter_type'] == 'Bilateral':
        print(f"  - Bilateral Spatial Sigma: {filter_params['bilateral_spatial']:.2f}")
    elif filter_params['filter_type'] == 'Beam Hardening':
        print(f"  - Beam Hardening Strength: {filter_params['beam_hardening_strength']:.2f}")
    elif filter_params['filter_type'] == 'Cupping':
        print(f"  - Cupping Correction Strength: {filter_params['cupping_strength']:.2f}")
    
    print("="*60 + "\n")


def save_filtered_volume(volume, output_path, original_nii_path=None):
 
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
