import numpy as np
import tigre


def prepare_fbp_geometry(width, n_angles, config):
    """
    Centralized FBP geometry definition for parallel-beam slice reconstruction.
    """
    voxel_size_mm = float(config["voxel_size_um"]) / 1000.0
    detector_pixel_size_mm = float(config["detector_pixel_size_mm"])
    shift_mm = float(config["calibrated_shift_px"]) * detector_pixel_size_mm * float(config["shift_sign"])

    geo = tigre.geometry(mode="parallel", nVoxel=np.array([1, width, width]))
    geo.dVoxel = np.array([voxel_size_mm, voxel_size_mm, voxel_size_mm], dtype=np.float32)
    geo.dDetector = np.array([detector_pixel_size_mm, detector_pixel_size_mm], dtype=np.float32)
    geo.nDetector = np.array([1, width]) # Only one detector pixel in the first dimension for parallel-beam geometry
    geo.sDetector = geo.nDetector * geo.dDetector # Size of the detector in mm
    geo.sVoxel = geo.nVoxel * geo.dVoxel # Size of the voxel grid in mm, only one voxel in the first dimension for slice reconstruction
    geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.rotDetector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.offDetector = np.array([0.0, shift_mm], dtype=np.float32)

    angles = np.linspace(0.0, float(config["total_angle"]), n_angles, endpoint=False).astype(np.float32)
    return geo, angles
