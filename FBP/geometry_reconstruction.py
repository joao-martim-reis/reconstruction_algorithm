import numpy as np
import tigre


def prepare_geometry(sino_normalized, pixel_size, shift_pixels, shift_sign):
    """
    Configure TIGRE geometry for FBP reconstruction with PARALLEL beam geometry.
    
    Note: Using "parallel" mode for FBP reconstruction because the Filtered Back Projection 
    algorithm assumes parallel-beam geometry. This simplifies the reconstruction process 
    as it does not account for cone-beam divergence.
    
    Args:
        sino_normalized: Normalized sinogram array
        pixel_size: Detector pixel size in mm
        shift_pixels: Detector shift in pixels (for center correction)
        shift_sign: Sign of the shift (+1 or -1)
        
    Returns:
        geo: TIGRE geometry object
        sino_input: Sinogram with added dimension for TIGRE
        angles: Array of projection angles
    """
    n_angles, width = sino_normalized.shape #sino_normalized shape = (Angles, Width)
    
    geo = tigre.geometry(mode="parallel", nVoxel=np.array([1, width, width])) # Parallel beam geometry
    geo.dVoxel = np.array([pixel_size, pixel_size, pixel_size]) # Size of EACH voxel in mm [dZ, dY, dX] 
    geo.dDetector = np.array([pixel_size, pixel_size]) # Size of EACH pixel in mm [dV, dU], dV and dU are detector directions
    geo.nDetector = np.array([1, width]) # ndetector is the number of pixels in detector [nV, nU]
    geo.sDetector = geo.nDetector * geo.dDetector # Size of the detector [sV, sU]
    geo.sVoxel = geo.nVoxel * geo.dVoxel # Total size of the volume [sZ, sY, sX]
    
    geo.offOrigin = np.array([0, 0, 0]) # Offset of image from origin (mm)
    geo.rotDetector = np.array([0, 0, 0]) # No rotation of detector

    shift_mm = shift_pixels * pixel_size # Convert pixel shift to mm
    geo.offDetector = np.array([0.0, shift_mm * shift_sign])  # Apply shift correction on detector offset
    
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # TIGRE expects sinograms shaped as (nAngles, nDetectorV, nDetectorU).
    # For 2D single-slice reconstructions the detector vertical size is 1, so we need to add a singleton dimension to produce shape (nAngles, 1, width).

    sino_input = sino_normalized[:, np.newaxis, :]  # shape -> (Angles, 1, Width)

    return geo, sino_input, angles

