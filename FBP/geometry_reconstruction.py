import numpy as np
import tigre


def prepare_geometry(sino_normalized, pixel_size, shift_pixels, shift_sign):
    """
    Configure TIGRE geometry for fbp reconstruction.
    
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
    n_angles, width = sino_normalized.shape
    
    geo = tigre.geometry(mode="parallel", nVoxel=np.array([1, width, width]))
    geo.dVoxel = np.array([pixel_size, pixel_size, pixel_size])
    geo.dDetector = np.array([pixel_size, pixel_size])
    geo.nDetector = np.array([1, width])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    geo.offOrigin = np.array([0, 0, 0])
    geo.rotDetector = np.array([0, 0, 0])

    shift_mm = shift_pixels * pixel_size
    geo.offDetector = np.array([0.0, shift_mm * shift_sign])  # Apply shift correction on detector offset
    
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    sino_input = sino_normalized[:, np.newaxis, :]
    
    return geo, sino_input, angles
