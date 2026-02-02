import numpy as np
import tigre


def setup_geometry(img_shape, voxel_size, DSD, DSO, shift_pixels, total_angle, 
                   shift_sign, downsample_factor=1, crop_params=None):
    """
    VOXEL-SIZE-FIRST APPROACH: Setup TIGRE geometry with voxel size as primary input.
    
    - voxel_size enters ALREADY ADJUSTED for downsampling (done in MAIN script)
    - Calculate: calculated_pixel_size = voxel_size_mm * magnification
    - Benefit: Direct comparison with published papers and commercial systems

    CRITICAL: When projections are cropped, the detector geometry must be adjusted:
    - nDetector: new height and width after crop
    - sDetector: physical size changes proportionally
    - offDetector: must account for the shift of the cropped region center
    
    SHIFT CALCULATION FIX:
    - shift_pixels comes in units of DOWNSAMPLED PROJECTION PIXELS
    - Must convert using calculated_pixel_size (which matches the reconstruction geometry)
    - NOT effective_detector_pixel (which would double-correct for downsampling)
    """
    
    # Physical hardware constraint - actual pixel size of our detector
    REAL_DETECTOR_PIXEL_SIZE = 0.050  # mm (50 μm)
    effective_detector_pixel = REAL_DETECTOR_PIXEL_SIZE * downsample_factor
    
    geo = tigre.geometry(mode="cone")
    magnification = DSD / DSO

    # Calculate required pixel size from voxel size 
    voxel_size_mm = voxel_size / 1000.0
    calculated_pixel_size = voxel_size_mm * magnification
    
    # Nyquist limit: smallest voxel size achievable with current effective detector resolution
    # Both voxel_size_mm and Nyquist_limit_voxel_size are at the same scale (after downsampling)
    Nyquist_limit_voxel_size = effective_detector_pixel / magnification


    print("VOXEL SIZE ANALYSIS")
    print(f"Physical detector pixel size: {REAL_DETECTOR_PIXEL_SIZE*1000:.2f} μm (hardware)")
    print(f"Magnification (DSD/DSO): {magnification:.3f}x")


    """ 
    Print related to oversampling/undersampling trade-offs, relation between voxel size and detector resolution.

    if voxel_size_mm < Nyquist_limit_voxel_size:
        oversampling_factor = Nyquist_limit_voxel_size / voxel_size_mm
        volume_increase = oversampling_factor ** 3
        
        print()
        print("OVERSAMPLING DETECTED")
        print(f"Oversampling factor: {oversampling_factor:.2f}x")
        print()
        print("TRADE-OFFS:")
        print(f"  ✗ No new spatial information (limited by detector resolution)")
        print(f"  ✗ Volume size increases by ({oversampling_factor:.1f})³ = {volume_increase:.1f}x")
        print(f"  ✗ Reconstruction time increases proportionally")
        print(f"  ✗ Memory usage increases {volume_increase:.1f}x")
        print(f"  ✓ Smoother visualization (less 'blocky' appearance)")
        print(f"  ✓ Better for image registration/alignment with other datasets")
        print(f"  ✓ Reduces aliasing artifacts (if moderate oversampling ~1.5-2x)")
        print()
        print("→ Reconstruction will proceed with requested voxel size")
        # We inform but don't restrict - user knows their requirements best
        
    
    # Undersampling: not using full detector resolution
    elif voxel_size_mm > Nyquist_limit_voxel_size:
        undersampling_factor = voxel_size_mm / Nyquist_limit_voxel_size
        volume_decrease = undersampling_factor ** 3
        
        print("UNDERSAMPLING (Larger voxels)")
        print(f"Undersampling factor: {undersampling_factor:.2f}x")
        print()
        print("TRADE-OFFS:")
        print(f"  ✓ Faster reconstruction (fewer voxels to compute)")
        print(f"  ✓ Smaller file sizes ({1/volume_decrease:.1f}x smaller)")
        print(f"  ✓ Less memory required ({1/volume_decrease:.1f}x less)")
        print(f"  ✗ May lose fine details that detector could capture")
        print(f"  ✗ Effective resolution lower than detector capability")
        print()
        print("→ Reconstruction will proceed with requested voxel size")
    
    else:
        print("✓ OPTIMAL SAMPLING (Nyquist-matched)")
        print("  Using full detector resolution without oversampling")
        print("  This is the sweet spot for resolution vs. computational efficiency")
    """

    height, width, n_angles = img_shape
    
    # Detector size after crop (if applied)
    geo.nDetector = np.array([height, width])

    # Detector pixel spacing: use calculated_pixel_size = voxel_size_mm * magnification (already correct for downsampling)
    geo.dDetector = np.array([calculated_pixel_size, calculated_pixel_size]) # geo.dDetector is the physical size of each pixel in mm
    geo.sDetector = geo.nDetector * geo.dDetector # physical size of the detector in mm
    
    # Voxel spacing: use voxel_size_mm directly (already adjusted for downsampling)
    geo.dVoxel = np.array([voxel_size_mm, voxel_size_mm, voxel_size_mm]) # is the physical size of each voxel in mm
    
    # Calculate number of voxels needed to cover the detector field of view
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size_mm),
        int(geo.sDetector[1] / voxel_size_mm),
        int(geo.sDetector[1] / voxel_size_mm)
    ])
    
    geo.sVoxel = geo.nVoxel * geo.dVoxel #real physical size of the volume in mm
    geo.DSD = DSD
    geo.DSO = DSO
    

    # CRITICAL FIX: Shift conversion using calculated_pixel_size
    # The shift_pixels comes in units of DOWNSAMPLED PROJECTION PIXELS
    # (already divided by downsample_factor in the MAIN script)
    # 
    # We must convert to mm using calculated_pixel_size, which is the pixel size
    # that matches our reconstruction geometry (voxel_size_mm * magnification)
    # This ensures the shift is in the same coordinate system as the geometry
    # ============================================================================
    
    shift_mm = shift_pixels * calculated_pixel_size
    shift_mm_sign = shift_mm * shift_sign  # Apply sign
    
    print(f"\nSHIFT CALCULATION:")
    print(f"  Shift (downsampled pixels): {shift_pixels:.3f} px")
    print(f"  Calculated pixel size: {calculated_pixel_size:.4f} mm")
    print(f"  Shift (mm): {shift_mm:.4f} mm")
    
    if crop_params is not None:
        # Calculate how much the detector center moved due to crop
        original_center = crop_params['original_center_col']
        new_center = (crop_params['col_end'] + crop_params['col_start']) / 2.0
        crop_shift_pixels = new_center - original_center
        print(f"  Crop adjustment: New center: {new_center:.2f} px (original: {original_center:.2f} px)")
        
        # Crop shift also uses calculated_pixel_size (same reasoning as above)
        crop_shift_mm = crop_shift_pixels * calculated_pixel_size
        crop_shift_mm = crop_shift_mm * shift_sign  # Apply sign
        
        # Total offset = calibrated shift + crop shift
        total_shift_mm = shift_mm_sign + crop_shift_mm   
        print(f"  Crop shift (mm): {crop_shift_mm:.4f} mm")
        print(f"  Total shift (mm): {total_shift_mm:.4f} mm")
        
        geo.offDetector = np.array([0.0, total_shift_mm])
        
    else:
        geo.offDetector = np.array([0.0, shift_mm_sign])
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    

    print(f"\nGEOMETRY SUMMARY:")
    print(f"  Voxel size: {geo.dVoxel[0]*1000:.2f} μm")
    print(f"  Detector pixel (calculated): {calculated_pixel_size*1000:.2f} μm")
    print(f"  Volume dimensions: {geo.nVoxel}")
    print(f"  Physical size: [{geo.sVoxel[0]:.1f}, {geo.sVoxel[1]:.1f}, {geo.sVoxel[2]:.1f}] mm\n")
    
    return geo, angles