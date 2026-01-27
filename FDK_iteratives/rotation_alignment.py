"""
Simple Manual Rotation Correction for CT Projections

Applies a user-specified rotation angle to all projections to align the sample
perpendicular to the detector.
"""

import numpy as np
from scipy.ndimage import rotate


def apply_rotation_to_projections(projections, angle, order=3):
    """
    Rotates all projections by a specified angle.
    
    ROTATION MATRIX PARAMETERS EXPLAINED:
    =====================================
    
    angle : float (degrees)
        - Rotation angle in DEGREES
        - POSITIVE values = counterclockwise rotation
        - NEGATIVE values = clockwise rotation
        - Example: angle=2.5 rotates image 2.5° counterclockwise
        - Example: angle=-1.8 rotates image 1.8° clockwise
    
    order : int (interpolation quality)
        - Controls how pixel values are interpolated during rotation
        - 0 = Nearest neighbor (fastest, lowest quality - blocky)
        - 1 = Bilinear (fast, decent quality)
        - 3 = Cubic (slower, best quality - RECOMMENDED)
        - Higher order = smoother edges but slower computation
    
    ROTATION BEHAVIOR:
    - reshape=False: Keeps original image dimensions (some edges may be cropped)
    - mode='constant': Fills empty corners with zero (black)
    - Rotation is applied around the image center
    
    MEMORY EFFICIENCY:
    - Processes ONE projection at a time (not the entire stack)
    - Prevents memory overflow with large datasets
    
    Parameters:
    -----------
    projections : np.ndarray
        Projection stack with shape (Height × Width × N_angles)
    angle : float
        Rotation angle in degrees
        Positive = counterclockwise, Negative = clockwise
    order : int, default=3
        Interpolation order (0=nearest, 1=bilinear, 3=cubic)
    
    Returns:
    --------
    rotated_projections : np.ndarray
        Rotated projection stack with same shape as input
    """
    
    # Safety check
    if projections is None:
        raise ValueError("Projections array is None. Cannot apply rotation.")
    
    H, W, N = projections.shape
    rotated_projections = np.zeros_like(projections)
    
    print(f"\n{'='*60}")
    print(f"APPLYING MANUAL ROTATION")
    print(f"{'='*60}")
    print(f"  Rotation angle: {angle:.3f}° {'(counterclockwise)' if angle > 0 else '(clockwise)'}")
    print(f"  Number of projections: {N}")
    print(f"  Interpolation: order={order} ({'cubic' if order==3 else 'bilinear' if order==1 else 'nearest'})")
    print(f"  Processing...")
    
    # Apply rotation to each projection individually (memory efficient)
    for i in range(N):
        rotated_projections[:, :, i] = rotate(
            projections[:, :, i], 
            angle, 
            reshape=False,  # Keep same dimensions
            order=order,    # Interpolation quality
            mode='constant',  # Fill edges with zeros
            cval=0          # Background value
        )
        
        # Progress update every 50 projections
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{N} projections")
    
    print(f"  ✓ Rotation complete!")
    print(f"{'='*60}\n")
    
    return rotated_projections


if __name__ == "__main__":
    print("Manual Rotation Module")
    print("\nUsage:")
    print("  from rotation_alignment import apply_rotation_to_projections")
    print("  projections_rotated = apply_rotation_to_projections(projections, angle=2.5)")
