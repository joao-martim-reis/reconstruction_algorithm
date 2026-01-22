import numpy as np
import tigre


def setup_geometry(img_shape, voxel_size, DSD, DSO, shift_pixels, total_angle, shift_sign, downsample_factor=1):
    """
    VOXEL-SIZE-FIRST APPROACH: Setup TIGRE geometry with voxel size as primary input.
    
    SIMPLIFIED APPROACH:
        - voxel_size enters ALREADY ADJUSTED for downsampling (done in MAIN script)
        - Calculate: required_pixel_size = voxel_size × magnification (simple!)
        - Use values directly (no extra scaling needed)
        - Benefit: Direct comparison with published papers and commercial systems
    
    WHY THIS MATTERS:
    - Micro-CT community reports results in voxel size, not pixel size
    - Makes it easy to match resolution of other systems (e.g., "reconstruct at 20 μm voxels")
    - Easier to understand trade-offs between resolution, memory, and computation time
    
    Args:
        img_shape: (height, width, n_angles)
        voxel_size: Effective voxel size in μm (ALREADY adjusted for downsampling in MAIN)
        DSD: Distance Source to Detector in mm
        DSO: Distance Source to Object in mm
        shift_pixels: Detector shift in pixels (for center correction)
        total_angle: Total rotation angle in radians
        shift_sign: Sign of the shift (+1 or -1)
        downsample_factor: For Nyquist limit calculation only
        
    Returns:
        geo: TIGRE geometry object
        angles: Array of projection angles
    """
    
    # Physical hardware constraint - actual pixel size of our detector
    REAL_DETECTOR_PIXEL_SIZE = 0.050  # mm (50 μm)
    
    print(f"--> Setting up VOXEL-SIZE-FIRST geometry...")
    
    # Convert voxel size from micrometers to millimeters
    # IMPORTANT: voxel_size already adjusted for downsampling in MAIN script
    # Example: user requests 20 μm, downsample=4 → voxel_size=80 μm enters here
    voxel_size_mm = voxel_size / 1000.0
    
    height, width, n_angles = img_shape
    geo = tigre.geometry(mode="cone")
    
    magnification = DSD / DSO
    
    # Calculate effective detector pixel size after downsampling
    # This affects the Nyquist limit (maximum achievable resolution)
    effective_detector_pixel = REAL_DETECTOR_PIXEL_SIZE * downsample_factor
    
    # Calculate required pixel size from voxel size (simple formula!)
    # Formula: pixel_size = voxel_size × magnification
    # Since voxel_size is already adjusted for downsampling, this gives us the correct pixel size
    required_pixel_size = voxel_size_mm * magnification
    
    # Nyquist limit: smallest voxel size achievable with current effective detector resolution
    # Both voxel_size_mm and Nyquist_limit are at the same scale (after downsampling)
    Nyquist_limit = effective_detector_pixel / magnification
    
    # ========== VALIDATION & INFORMATION DISPLAY ==========
    print("\n" + "="*60)
    print("VOXEL SIZE ANALYSIS")
    print("="*60)
    print(f"Physical detector pixel size: {REAL_DETECTOR_PIXEL_SIZE*1000:.2f} μm (hardware)")
    
    # Check if downsampling was applied
    if downsample_factor > 1:
        print(f"Downsampling applied: {downsample_factor:.0f}× → Effective pixel: {effective_detector_pixel*1000:.2f} μm")
        print(f"  → Each projection pixel = {downsample_factor:.0f}×{downsample_factor:.0f} = {downsample_factor**2:.0f} physical detector pixels averaged")
    
    print(f"Magnification (DSD/DSO): {magnification:.3f}x")
    print(f"Maximum achievable resolution (Nyquist limit): {Nyquist_limit*1000:.2f} μm voxels")
    print(f"  → Smallest voxel size where we use full effective detector resolution")
    print(f"  → Choosing smaller voxels = oversampling (no new information)")
    
    print()
    
    # ========== OVERSAMPLING DETECTION ==========
    # Oversampling means creating more voxels than detector can distinguish
    # Like upscaling a low-res image - more pixels, but no new detail
    if voxel_size_mm < Nyquist_limit:
        oversampling_factor = Nyquist_limit / voxel_size_mm
        volume_increase = oversampling_factor ** 3
        
        print("⚠️  OVERSAMPLING DETECTED")
        print(f"Oversampling factor: {oversampling_factor:.2f}x")
        print()
        print("CONSEQUENCES:")
        print(f"  ✗ No new spatial information (limited by detector resolution)")
        print(f"  ✗ Volume size increases by ({oversampling_factor:.1f})³ = {volume_increase:.1f}x")
        print(f"  ✗ Reconstruction time increases proportionally")
        print(f"  ✗ Memory usage increases {volume_increase:.1f}x")
        print()
        print("POTENTIAL BENEFITS:")
        print(f"  ✓ Smoother visualization (less 'blocky' appearance)")
        print(f"  ✓ Better for image registration/alignment with other datasets")
        print(f"  ✓ Reduces aliasing artifacts (if moderate oversampling ~1.5-2x)")
        print()
        print("→ Reconstruction will proceed with requested voxel size")
        # We inform but don't restrict - user knows their requirements best
        
    # ========== UNDERSAMPLING DETECTION ==========
    # Undersampling: not using full detector resolution
    # Useful when fine details aren't needed or for fast preview scans
    elif voxel_size_mm > Nyquist_limit:
        undersampling_factor = voxel_size_mm / Nyquist_limit
        volume_decrease = undersampling_factor ** 3
        
        print("ℹ️  UNDERSAMPLING (Larger voxels)")
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
    
    # ========== OPTIMAL SAMPLING ==========
    else:
        print("✓ OPTIMAL SAMPLING (Nyquist-matched)")
        print("  Using full detector resolution without oversampling")
        print("  This is the sweet spot for resolution vs. computational efficiency")
    
    print("="*60 + "\n")
    
    # ========== DETECTOR GEOMETRY SETUP ==========
    geo.nDetector = np.array([height, width]) # ndetector is the number of pixels in detector 
    # geo.dDetector: Detector pixel spacing (already correct for downsampling)
    geo.dDetector = np.array([required_pixel_size, required_pixel_size])
    geo.sDetector = geo.nDetector * geo.dDetector # Size of the detector 

    
    # ========== VOXEL GEOMETRY SETUP ==========
    # geo.dVoxel: Voxel spacing (already adjusted for downsampling)
    # This is the whole point of voxel-size-first approach!
    geo.dVoxel = np.array([voxel_size_mm, voxel_size_mm, voxel_size_mm])
    
    # geo.nVoxel = NUMBER of voxels in each dimension [nZ, nY, nX]
    # Calculate how many voxels fit in the detector field of view
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size_mm),  # Z (height): detector height / voxel_size
        int(geo.sDetector[1] / voxel_size_mm),  # Y (depth): detector width / voxel_size
        int(geo.sDetector[1] / voxel_size_mm)   # X (width): same as Y to create cubic FOV in axial plane
    ])
    # Note: nVoxel[1] == nVoxel[2] creates a square/cubic volume in the XY plane (axial slices)
    # You can change nVoxel[2] to be different (e.g., int(geo.sDetector[1] * 0.8 / voxel_size))
    # to create a rectangular FOV and reduce memory usage
    # Aumentar nVoxel (mais voxels com mesmo dVoxel) → maior detalhe espacial, mais RAM/VRAM e tempo.
    
    # geo.sVoxel = TOTAL physical size of volume in mm [sZ, sY, sX]
    # Total volume dimensions = number of voxels × size of each voxel
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    # ========== SOURCE-DETECTOR GEOMETRY ==========
    geo.DSD = DSD # Distance Source to Detector
    geo.DSO = DSO # Distance Source to Object
    
    # ========== DETECTOR OFFSET ADJUSTMENT ==========
    # APPLYING SHIFT 
    shift_mm = shift_pixels * required_pixel_size
    geo.offDetector = np.array([0.0, shift_mm * shift_sign]) 
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    
    print(f"    Final voxel size: {voxel_size:.2f} μm")
    print(f"    Required pixel size: {required_pixel_size*1000:.2f} μm")
    print(f"    Volume dimensions: {geo.nVoxel}")
    print(f"    Physical size: [{geo.sVoxel[0]:.1f}, {geo.sVoxel[1]:.1f}, {geo.sVoxel[2]:.1f}] mm\n")
    
    return geo, angles
