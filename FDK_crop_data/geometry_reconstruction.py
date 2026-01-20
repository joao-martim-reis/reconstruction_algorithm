import numpy as np
import tigre



def setup_geometry(img_shape, pixel_size, DSD, DSO, shift_pixels, total_angle, 
                   shift_sign, voxel_ratio=1.0, crop_params=None):
    """
    Setup TIGRE geometry with support for cropped projections.
    
    CRITICAL: When projections are cropped, the detector geometry must be adjusted:
    - nDetector: new height and width after crop
    - sDetector: physical size changes proportionally
    - offDetector: must account for the shift of the cropped region center
    
    Args:
        img_shape: (height, width, n_angles) - AFTER crop and downsample
        pixel_size: Detector pixel size in mm (after downsample)
        DSD: Distance Source to Detector in mm
        DSO: Distance Source to Object in mm
        shift_pixels: Detector shift in pixels (for center correction)
        total_angle: Total rotation angle in radians
        shift_sign: Sign of the shift (+1 or -1)
        voxel_ratio: Multiplier for voxel size (1.0 = Nyquist optimal, <1 = higher res, >1 = lower res)
        crop_params: Dict with crop information (from select_crop_region)
        
    """
    
    print(f"--> Setting up geometry...")
    
    height, width, n_angles = img_shape
    geo = tigre.geometry(mode="cone")
    
    # Detector size after crop
    geo.nDetector = np.array([height, width])
    geo.dDetector = np.array([pixel_size, pixel_size])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    # Calculate voxel size
    magnification = DSD / DSO 
    voxel_size_base = pixel_size / magnification
    voxel_size = voxel_size_base * voxel_ratio
    
    print(f"    Pixel size: {pixel_size:.6f} mm")
    print(f"    Magnification: {magnification:.4f}")
    print(f"    Voxel size (ratio={voxel_ratio}): {voxel_size:.6f} mm")
    
    geo.dVoxel = np.array([voxel_size, voxel_size, voxel_size])
    
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size),
        int(geo.sDetector[1] / voxel_size),
        int(geo.sDetector[1] / voxel_size)
    ])
    
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = DSD
    geo.DSO = DSO
    
    # CRITICAL: Detector offset adjustment for cropped projections
    # The calibrated shift is relative to the ORIGINAL detector center
    # After crop, we need to account for:
    # 1. The horizontal shift from crop
    # 2. The original calibrated shift
    
    shift_mm = shift_pixels * pixel_size
    shift_mm_sign = shift_mm * shift_sign  # Apply sign
    
    if crop_params is not None:
        # Calculate how much the detector center moved due to crop
        original_center = crop_params['original_center_col']
        new_center = (crop_params['col_end'] + crop_params['col_start']) / 2.0
        crop_shift_pixels = new_center - original_center
        print(f" New center: {new_center:.2f} px (original: {original_center:.2f} px)")
        crop_shift_mm = crop_shift_pixels * pixel_size
        crop_shift_mm = crop_shift_mm * shift_sign  # Apply sign
        # Total offset = calibrated shift + crop shift
        total_shift_mm = shift_mm_sign + crop_shift_mm   
        geo.offDetector = np.array([0.0, total_shift_mm])
        
    else:
        # No crop, just use calibrated shift
        print(f"    Detector shift: {shift_pixels:.2f} px = {shift_mm:.4f} mm")
        geo.offDetector = np.array([0.0, shift_mm_sign])
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    
    return geo, angles
