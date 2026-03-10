import numpy as np
from geometry import Geometry

"""
    VOXEL-SIZE-FIRST APPROACH: Setup TIGRE geometry with voxel size as primary input.
    
    CRITICAL FIX: Detector pixel size is INDEPENDENT of voxel size!
    - geo.dDetector = effective_detector_pixel (hardware property, fixed by downsampling)
    - geo.dVoxel = voxel_size_mm (reconstruction parameter, user choice)
    
    These are INDEPENDENT. Changing voxel size should NOT change detector geometry.
"""


def setup_geometry(img_shape, voxel_size, DSD, DSO, shift_pixels, total_angle, 
                   shift_sign, downsample_factor=1, crop_params=None, detector_tilt=0):


    REAL_DETECTOR_PIXEL_SIZE = 0.050  # Physical hardware constraint - actual pixel size of our detector, (50 μm)
    effective_detector_pixel = REAL_DETECTOR_PIXEL_SIZE * downsample_factor # Effective detector pixel after downsampling. This is the ACTUAL pixel size of the projection images
    
    geo = Geometry(mode="cone")
    magnification = DSD / DSO

    voxel_size_mm = voxel_size / 1000.0 # Convert voxel size to mm
    Nyquist_limit_voxel_size = effective_detector_pixel / magnification # Nyquist limit: smallest voxel size achievable with current detector resolution


    print("GEOMETRY SETUP - VOXEL SIZE ANALYSIS")
    print(f"Physical detector pixel size: {REAL_DETECTOR_PIXEL_SIZE*1000:.2f} μm (hardware)")
    print(f"Downsampling factor: {downsample_factor}x")
    print(f"Effective detector pixel: {effective_detector_pixel*1000:.2f} μm (after downsampling)")
    print(f"Requested voxel size: {voxel_size_mm*1000:.2f} μm")
    print(f"Magnification (DSD/DSO): {magnification:.3f}x")


    height, width, n_angles = img_shape #this uses the cropped image shape
    
    geo.nDetector = np.array([height, width]) # Detector size after crop (if applied)

    # This should NEVER change based on voxel_size!
    geo.dDetector = np.array([effective_detector_pixel, effective_detector_pixel]) # CRITICAL FIX: Detector pixel spacing is the PHYSICAL pixel size
    geo.sDetector = geo.nDetector * geo.dDetector  # Physical size of detector in mm

    geo.dVoxel = np.array([voxel_size_mm, voxel_size_mm, voxel_size_mm]) # Voxel spacing: use voxel_size_mm directly (user's choice for reconstruction)

    geo.nVoxel = np.array([ # Calculate number of voxels needed to cover the detector field of view
        int(np.ceil(geo.sDetector[0] / voxel_size_mm)),
        int(np.ceil(geo.sDetector[1] / voxel_size_mm)),
        int(np.ceil(geo.sDetector[1] / voxel_size_mm))
    ])
    
    geo.sVoxel = geo.nVoxel * geo.dVoxel  # Real physical size of the volume in mm
    geo.DSD = DSD
    geo.DSO = DSO
    
    shift_mm = shift_pixels * effective_detector_pixel # Shift conversion using PHYSICAL detector pixel size
    shift_mm_sign = shift_mm * shift_sign
    

    print("SHIFT CALCULATION")
    print(f"Shift (downsampled pixels): {shift_pixels:.3f} px")
    print(f"Effective detector pixel: {effective_detector_pixel:.4f} mm")
    print(f"Shift (mm): {shift_mm:.4f} mm (physical hardware offset)")
    
    if crop_params is not None:
        original_center = crop_params['original_center_col']
        new_center = (crop_params['col_end'] + crop_params['col_start']) / 2.0
        crop_shift_pixels = new_center - original_center
        print(f"Crop adjustment: New center: {new_center:.2f} px (original: {original_center:.2f} px)")
        
        crop_shift_mm = crop_shift_pixels * effective_detector_pixel
        crop_shift_mm = crop_shift_mm * shift_sign
        
        # Total offset = calibrated shift + crop shift
        total_shift_mm = shift_mm_sign + crop_shift_mm   
        print(f"Crop shift (mm): {crop_shift_mm:.4f} mm")
        print(f"Total shift (mm): {total_shift_mm:.4f} mm")
        
        geo.offDetector = np.array([0.0, total_shift_mm])
    else:
        geo.offDetector = np.array([0.0, shift_mm_sign])
    
    geo.offOrigin = np.array([0, 0, 0])

    # --- DETECTOR TILT CORRECTION ---
    # Value obtained from the calculate_detector_tilt.py script
    # In TIGRE, in-plane rotation is the Z component (index 2)
    geo.rotDetector = np.array([0, 0, detector_tilt])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)


    print("GEOMETRY SUMMARY")
    print(f"DETECTOR (fixed by hardware + downsampling):")
    print(f"Detector size: {geo.nDetector}")
    print(f"Pixel size: {geo.dDetector[0]*1000:.3f} μm")
    print(f"Physical size: [{geo.sDetector[0]:.2f}, {geo.sDetector[1]:.2f}] mm")
    print(f"Offset: {geo.offDetector} mm")

    print(f"\nVOLUME (determined by voxel size choice):")
    print(f"Voxel size: {geo.dVoxel[0]*1000:.2f} μm")
    print(f"Volume size: {geo.nVoxel}")
    print(f"Physical size: [{geo.sVoxel[0]:.1f}, {geo.sVoxel[1]:.1f}, {geo.sVoxel[2]:.1f}] mm")

    print(f"\nDISTANCES:")
    print(f"Distance Source to Detector: {geo.DSD:.1f} mm")
    print(f"Distance Source to Object: {geo.DSO:.1f} mm")
    print(f"Number of angles: {n_angles}")

    
    return geo, angles