"""
Automated Projection Alignment Module

This module provides automatic rotation correction for CT projections where the 
sample rotation axis is not perfectly perpendicular to the detector.

Key Features:
- Automatic angle detection using edge detection and Hough transform
- Applies same rotation to all projections for consistency
- Memory-efficient processing (processes one projection at a time)
- Preserves data quality with high-quality interpolation
"""

import numpy as np
from scipy.ndimage import rotate
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt


def detect_rotation_angle(projection, visualize=False):
    """
    Automatically detects the rotation angle needed to align the sample vertically.
    
    Uses edge detection + Hough transform to find dominant vertical lines.
    The sample should be aligned so edges are perpendicular to the detector top.
    
    Parameters:
    -----------
    projection : np.ndarray
        Single 2D projection image (H × W)
    visualize : bool
        If True, displays the detection process
    
    Returns:
    --------
    angle : float
        Rotation angle in degrees (positive = counterclockwise)
    """
    
    # Step 1: Normalize projection for better edge detection
    proj_norm = (projection - projection.min()) / (projection.max() - projection.min() + 1e-6)
    
    # Step 2: Apply Canny edge detection
    # Low thresholds to capture subtle edges in CT data
    edges = canny(proj_norm, sigma=2.0, low_threshold=0.1, high_threshold=0.3)
    
    # Step 3: Hough transform to detect lines
    # Test angles: -45° to +45° (assuming misalignment is not extreme)
    tested_angles = np.linspace(-np.pi/4, np.pi/4, 360)
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # Step 4: Find the most prominent lines
    _, angles_detected, _ = hough_line_peaks(h, theta, d, num_peaks=5, threshold=0.3*h.max())
    
    # Step 5: Calculate median angle (robust to outliers)
    # Convert from radians to degrees
    angles_deg = np.degrees(angles_detected)
    
    # The detected angle represents the current tilt
    # We want to rotate by the NEGATIVE to correct it
    median_angle = np.median(angles_deg)
    correction_angle = -median_angle
    
    # Step 6: Optional visualization
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original projection
        axes[0].imshow(projection, cmap='gray')
        axes[0].set_title('Original Projection')
        axes[0].axis('off')
        
        # Detected edges
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title(f'Edge Detection\nDetected angles: {angles_deg}')
        axes[1].axis('off')
        
        # Hough space
        axes[2].imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                      cmap='hot', aspect='auto')
        axes[2].set_title(f'Hough Transform\nCorrection: {correction_angle:.2f}°')
        axes[2].set_xlabel('Angle (degrees)')
        axes[2].set_ylabel('Distance (pixels)')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n{'='*60}")
    print(f"AUTOMATIC ROTATION DETECTION")
    print(f"{'='*60}")
    print(f"  Detected tilt: {median_angle:.3f}°")
    print(f"  Correction angle: {correction_angle:.3f}°")
    print(f"{'='*60}\n")
    
    return correction_angle


def rotate_projection(projection, angle, order=3):
    """
    Rotates a single projection by the specified angle.
    
    Parameters:
    -----------
    projection : np.ndarray
        Single 2D projection (H × W)
    angle : float
        Rotation angle in degrees (positive = counterclockwise)
    order : int
        Interpolation order (0=nearest, 1=bilinear, 3=cubic)
        Higher order = better quality but slower
    
    Returns:
    --------
    rotated : np.ndarray
        Rotated projection with same shape as input
    """
    # Rotate without reshaping (keeps same dimensions, crops edges)
    rotated = rotate(projection, angle, reshape=False, order=order, mode='constant', cval=0)
    return rotated


def apply_rotation_to_all_projections(projections, angle, order=3, verbose=True):
    """
    Applies the same rotation angle to all projections in a stack.
    
    Memory-efficient: processes one slice at a time instead of the whole volume.
    
    Parameters:
    -----------
    projections : np.ndarray
        Projection stack (H × W × N_angles)
    angle : float
        Rotation angle in degrees
    order : int
        Interpolation order (3 = cubic, recommended)
    verbose : bool
        Print progress updates
    
    Returns:
    --------
    rotated_projections : np.ndarray
        Rotated projection stack with same shape
    """
    
    H, W, N = projections.shape
    rotated_projections = np.zeros_like(projections)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"APPLYING ROTATION TO ALL PROJECTIONS")
        print(f"{'='*60}")
        print(f"  Angle: {angle:.3f}°")
        print(f"  Number of projections: {N}")
        print(f"  Interpolation order: {order} (cubic)")
        print(f"  Processing...")
    
    # Process each projection individually (memory efficient)
    for i in range(N):
        rotated_projections[:, :, i] = rotate_projection(projections[:, :, i], angle, order=order)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{N} projections")
    
    if verbose:
        print(f"  ✓ Rotation complete!")
        print(f"{'='*60}\n")
    
    return rotated_projections


def rotation_correction_pipeline(projections, auto_detect=True, manual_angle=None, 
                                 visualize_detection=False, interpolation_order=3):
    """
    Complete pipeline for rotation correction.
    
    Parameters:
    -----------
    projections : np.ndarray
        Raw projection stack (H × W × N_angles)
    auto_detect : bool
        If True, automatically detect rotation angle from first projection
    manual_angle : float or None
        If provided, uses this angle instead of auto-detection
    visualize_detection : bool
        Show detection visualization (only if auto_detect=True)
    interpolation_order : int
        Quality of rotation interpolation (3 = cubic, best quality)
    
    Returns:
    --------
    rotated_projections : np.ndarray
        Corrected projection stack
    angle_used : float
        The rotation angle that was applied
    """
    
    # Determine which angle to use
    if manual_angle is not None:
        angle = manual_angle
        print(f"Using manual rotation angle: {angle:.3f}°")
    elif auto_detect:
        # Use middle projection for detection (often cleaner than first/last)
        middle_idx = projections.shape[2] // 2
        sample_projection = projections[:, :, middle_idx]
        angle = detect_rotation_angle(sample_projection, visualize=visualize_detection)
    else:
        print("No rotation angle provided and auto-detection disabled. Skipping rotation.")
        return projections, 0.0
    
    # Apply rotation to all projections
    rotated_projections = apply_rotation_to_all_projections(
        projections, 
        angle, 
        order=interpolation_order, 
        verbose=True
    )
    
    return rotated_projections, angle


# Quick test function
if __name__ == "__main__":
    print("Rotation Alignment Module")
    print("This module should be imported by main reconstruction scripts.")
    print("\nUsage:")
    print("  from rotation_alignment import rotation_correction_pipeline")
    print("  projections_aligned, angle = rotation_correction_pipeline(projections_raw)")
