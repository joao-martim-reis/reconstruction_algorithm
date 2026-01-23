# Iterative Reconstruction with OS (Ordered Subset) Algorithms

This directory contains iterative reconstruction algorithms using TIGRE's Ordered Subset (OS) methods. These algorithms are alternatives to the analytical FDK algorithm and can produce higher quality reconstructions in certain scenarios.

## Overview

Iterative algorithms iteratively update the volume estimate to minimize the difference between measured and simulated projections. While slower than FDK, they offer several advantages:

- **Better artifact reduction**: Handle limited angle, sparse data, and noisy projections
- **Regularization options**: TV (Total Variation) can reduce noise while preserving edges
- **Flexible constraints**: Can incorporate prior knowledge or physical constraints

## Available Algorithms

### 1. OSSART (Ordered Subset SART)
**Key:** `ossart`

Simultaneous Algebraic Reconstruction Technique with ordered subsets for faster convergence.

**Parameters:**
- `niter`: Number of iterations (default: 50)
- `blocksize`: Projections per subset (default: 20)
- `lmbda`: Relaxation parameter (default: 1.0)
- `lmbda_red`: Lambda reduction per iteration (default: 0.99)

**Best for:** General-purpose iterative reconstruction with good convergence speed

---

### 2. OSSART-TV (OSSART with Total Variation)
**Key:** `ossart_tv`

OSSART with Total Variation regularization for noise reduction.

**Parameters:**
- Same as OSSART, plus:
- `alpha`: TV regularization weight (default: 0.002)
- `alpha_red`: Alpha reduction per iteration (default: 0.95)
- `ng`: TV minimization iterations (default: 25)

**Best for:** Noisy data where edge preservation is important

---

### 3. SIRT (Simultaneous Iterative Reconstruction Technique)
**Key:** `sirt`

Classic iterative algorithm with simultaneous updates.

**Parameters:**
- `niter`: Number of iterations (default: 100)

**Best for:** Simple iterative reconstruction without subset acceleration

---

### 4. CGLS (Conjugate Gradient Least Squares)
**Key:** `cgls`

Krylov subspace method that minimizes least-squares error efficiently.

**Parameters:**
- `niter`: Number of iterations (default: 50)

**Best for:** Fast convergence to least-squares solution

---

### 5. SART (Simultaneous Algebraic Reconstruction Technique)
**Key:** `sart`

Standard SART without ordered subsets.

**Parameters:**
- `niter`: Number of iterations (default: 50)
- `lmbda`: Relaxation parameter (default: 1.0)
- `lmbda_red`: Lambda reduction per iteration (default: 0.99)

**Best for:** When subset acceleration is not needed

---

### 6. OS-ASD-POCS (Ordered Subset Adaptive Steepest Descent POCS)
**Key:** `os_asd_pocs`

Projection onto convex sets with adaptive steepest descent and TV regularization.

**Parameters:**
- `niter`: Number of iterations (default: 50)
- `blocksize`: Projections per subset (default: 20)
- `alpha`: Regularization parameter (default: 0.002)
- `alpha_red`: Alpha reduction (default: 0.95)
- `ng`: Gradient descent steps (default: 20)
- `epsilon`: Tolerance (default: 0.0)

**Best for:** Limited angle or sparse view reconstructions

---

### 7. OS-AwASD-POCS (Ordered Subset Adaptive-weighted ASD POCS)
**Key:** `os_awasd_pocs`

Advanced POCS variant with adaptive weighting.

**Parameters:**
- Same as OS-ASD-POCS, but with:
- `delta`: Adaptive weighting parameter (default: -0.005)

**Best for:** Challenging reconstruction problems with severe artifacts

---

## Usage

### Basic Example

```python
import numpy as np
from MAIN_TIGRE_iterative import main

CONFIG = {
    # Select algorithm
    'algorithm': 'ossart',
    
    # Geometry
    'voxel_size': 22,       # μm
    'DSD': 457,             # mm
    'DSO': 211,             # mm
    'total_angle': 2 * np.pi,
    
    # Calibration
    'calibrated_shift_px': 5.12,
    'shift_sign': 1,
    
    # Preprocessing
    'downsample': 2,
    
    # Output
    'output_folder_NiFT': './output'
}

folder = './projections'
volume = main(folder, CONFIG)
```

### Custom Algorithm Parameters

```python
CONFIG = {
    'algorithm': 'ossart_tv',
    
    # Override default parameters
    'algorithm_params': {
        'niter': 100,        # More iterations
        'blocksize': 10,     # Smaller subsets
        'lmbda': 0.8,        # Lower relaxation
        'alpha': 0.005,      # Stronger TV regularization
    },
    
    # ... rest of config
}
```

## Parameter Tuning Guide

### Number of Iterations (`niter`)
- **Too few:** Underconvergence, blurry results
- **Too many:** Overfitting, noise amplification
- **Typical range:** 50-200 depending on algorithm and data quality

### Block Size (`blocksize`)
- Only for OS algorithms (OSSART, OS-ASD-POCS, etc.)
- **Smaller blocks:** Faster convergence, less stable
- **Larger blocks:** Slower but more stable
- **Typical range:** 10-40 projections per subset
- **Formula:** `num_subsets = total_projections / blocksize`

### Relaxation Parameter (`lmbda`)
- Controls step size of updates
- **Higher (>1.0):** Faster convergence, risk of instability
- **Lower (<1.0):** Slower, more stable
- **Typical range:** 0.5-1.5
- **Reduction (`lmbda_red`)**: 0.95-0.99 to gradually reduce step size

### TV Regularization (`alpha`)
- Only for TV-based algorithms
- **Higher:** More smoothing, less noise but potential oversmoothing
- **Lower:** Preserves more detail but less noise reduction
- **Typical range:** 0.001-0.01
- Start low and increase if too noisy

## Performance Considerations

### Memory Usage
Iterative algorithms require more memory than FDK:
- Store volume estimate throughout iterations
- May need forward/backward projection buffers

**Tip:** Use `downsample` parameter to reduce memory if needed

### Computation Time
Approximate relative speeds (compared to FDK = 1x):
- FDK: 1x (fastest, baseline)
- CGLS: 10-20x slower
- SIRT: 15-30x slower
- OSSART: 20-40x slower (depends on subset size)
- OSSART-TV: 30-60x slower (TV adds overhead)
- OS-ASD-POCS: 40-80x slower

**Tips for faster reconstruction:**
- Increase `blocksize` (fewer subsets = fewer updates)
- Reduce `niter`
- Use downsampling for testing
- For TV algorithms, reduce `ng` (TV minimization steps)

## Comparison: FDK vs Iterative

| Aspect | FDK | Iterative (OS) |
|--------|-----|----------------|
| **Speed** | Very fast (seconds) | Slow (minutes to hours) |
| **Quality** | Good for clean data | Better for noisy/limited data |
| **Artifacts** | Ring artifacts, beam hardening | Better artifact suppression |
| **Memory** | Low | Higher |
| **Regularization** | None (analytical) | TV, constraints available |
| **Use case** | Standard full-angle scans | Limited angle, sparse, noisy |

## When to Use Iterative Algorithms

**Use iterative (OS) when:**
- Limited angle acquisitions (<180°)
- Sparse angular sampling (few projections)
- High noise levels
- Need for regularization (edge preservation)
- Artifacts are problematic in FDK
- Have time for longer reconstruction

**Stick with FDK when:**
- Standard 360° full acquisition
- Clean projections with good SNR
- Need fast preview or testing
- Memory or time constrained

## File Structure

```
FDK_iteratives/
├── MAIN_TIGRE_iterative.py              # Main script with OS presets
├── geometry_reconstruction_Voxel_size.py # Geometry setup (from FDK_reduce_memory)
├── data_processing_FDK_3D.py            # Data loading utilities
├── export_volumes.py                     # NIfTI export utilities
└── README.md                            # This file
```

## Examples

### Example 1: Fast Preview with CGLS

```python
CONFIG = {
    'algorithm': 'cgls',
    'algorithm_params': {'niter': 20},  # Quick preview
    'downsample': 4,  # Heavy downsampling for speed
    'voxel_size': 50,  # Larger voxels
    # ... other params
}
```

### Example 2: High-Quality with OSSART-TV

```python
CONFIG = {
    'algorithm': 'ossart_tv',
    'algorithm_params': {
        'niter': 100,
        'blocksize': 20,
        'alpha': 0.003,  # Moderate TV regularization
        'ng': 30,
    },
    'downsample': 1,  # No downsampling
    'voxel_size': 22,  # High resolution
    # ... other params
}
```

### Example 3: Limited Angle Reconstruction

```python
CONFIG = {
    'algorithm': 'os_asd_pocs',
    'algorithm_params': {
        'niter': 150,
        'blocksize': 15,
        'alpha': 0.005,  # Stronger regularization needed
    },
    'total_angle': np.pi,  # Only 180° scan
    # ... other params
}
```

## Troubleshooting

### Problem: Reconstruction is too slow
**Solutions:**
- Increase `blocksize` (fewer subsets)
- Reduce `niter`
- Use `downsample` > 1
- Try CGLS (faster than OSSART)

### Problem: Noisy reconstruction
**Solutions:**
- Use TV variant (`ossart_tv` instead of `ossart`)
- Increase `alpha` (TV weight)
- Reduce `lmbda` (relaxation)
- Increase `niter` (more smoothing over time)

### Problem: Blurry reconstruction
**Solutions:**
- Reduce `alpha` (less regularization)
- Decrease `niter` (avoid oversmoothing)
- Increase `lmbda` (larger steps)
- Try non-TV algorithm

### Problem: Convergence issues
**Solutions:**
- Reduce `lmbda` (smaller steps)
- Enable `lmbda_red` (gradual reduction)
- Check geometry parameters
- Verify projection normalization

## References

- TIGRE Documentation: https://github.com/CERN/TIGRE
- OSSART: Andersen & Kak, "Simultaneous Algebraic Reconstruction Technique"
- TV Regularization: Rudin et al., "Nonlinear total variation based noise removal algorithms"

## Support

For issues or questions:
1. Check TIGRE documentation
2. Verify geometry parameters match your scanner
3. Start with default parameters before tuning
4. Use FDK for comparison/validation
