import numpy as np
import tigre


def setup_geometry(img_shape, pixel_size, DSD, DSO, shift_pixels, total_angle, shift_sign, voxel_ratio=1.0):
    """
    Args:
        img_shape: (height, width, n_angles)
        pixel_size: Detector pixel size in mm
        DSD: Distance Source to Detector in mm
        DSO: Distance Source to Object in mm
        shift_pixels: Detector shift in pixels (for center correction)
        voxel_ratio: Multiplier for voxel size (1.0 = Nyquist optimal, <1 = higher res, >1 = lower res)
    """
    
    print(f"--> Setting up geometry with shift: {shift_pixels:.2f} px")
    
    height, width, n_angles = img_shape # img_shape = (Height, Width, Angles)
    geo = tigre.geometry(mode="cone") # Cone beam geometry
    
    geo.nDetector = np.array([height, width]) #ndetector is the number of pixels in detector 
    geo.dDetector = np.array([pixel_size, pixel_size]) #ddetector is the size of each pixel 
    geo.sDetector = geo.nDetector * geo.dDetector # Size of the detector 

    
    # Calculate base voxel size from Nyquist criterion
    magnification = DSD / DSO 
    voxel_size_base = pixel_size / magnification
    
    # Apply user-defined voxel ratio (allows oversampling or downsampling)
    voxel_size = voxel_size_base * voxel_ratio
    
    print(f"--> Magnification: {magnification:.4f}")
    print(f"--> Base voxel size (Nyquist): {voxel_size_base:.6f} mm")
    print(f"--> Final voxel size (ratio={voxel_ratio}): {voxel_size:.6f} mm")
    
    # geo.dVoxel = Size of EACH voxel in mm [dZ, dY, dX]
    # Individual voxel dimensions (isotropic = same size in all directions)
    geo.dVoxel = np.array([voxel_size, voxel_size, voxel_size])
    
    # geo.nVoxel = NUMBER of voxels in each dimension [nZ, nY, nX]
    # Calculate how many voxels fit in the detector field of view
    geo.nVoxel = np.array([
        int(geo.sDetector[0] / voxel_size),  # Z (height): detector height / voxel_size
        int(geo.sDetector[1] / voxel_size),  # Y (depth): detector width / voxel_size
        int(geo.sDetector[1] / voxel_size)   # X (width): same as Y to create cubic FOV in axial plane
    ])
    # Note: nVoxel[1] == nVoxel[2] creates a square/cubic volume in the XY plane (axial slices)
    # You can change nVoxel[2] to be different (e.g., int(geo.sDetector[1] * 0.8 / voxel_size))
    # to create a rectangular FOV and reduce memory usage
    #Aumentar nVoxel (mais voxels com mesmo dVoxel) → maior detalhe espacial, mais RAM/VRAM e tempo.
    
    # geo.sVoxel = TOTAL physical size of volume in mm [sZ, sY, sX]
    # Total volume dimensions = number of voxels × size of each voxel
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.DSD = DSD # Distance Source to Detector
    geo.DSO = DSO # Distance Source to Object
    
    # APPLYING SHIFT 
    shift_mm = shift_pixels * pixel_size
    geo.offDetector = np.array([0.0, shift_mm * shift_sign]) 
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])
    
    angles = np.linspace(0, total_angle, n_angles, endpoint=False)
    
    return geo, angles