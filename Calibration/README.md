# CT Detector Center Calibration

## Overview

This script performs systematic calibration to determine the **detector center misalignment** in cone-beam CT systems. It quantifies the pixel offset between the rotation axis center and the detector center, which is critical for accurate image reconstruction.


## Problem Statement

In cone-beam CT, perfect alignment requires that:
- The **rotation axis** (center of the rotating phantom)
- The **detector center** (geometric center of the detector panel)

...should coincide at the same pixel coordinate.

However, due to mechanical imperfections, these two centers are often **offset** by several pixels. This misalignment causes:
- **Reconstruction artifacts** (double edges, blurring)
- **Geometric distortions** in the reconstructed volume
- **Reduced image quality**



## How It Works

### Conceptual Approach

The script analyzes **sinogram data** to detect the position of a calibration phantom across multiple rotation positions.

**Key insight**: In a sinogram:
- The **vertical axis** represents the detector (pixels)
- The **horizontal axis** represents rotation angle (degrees)
- A centered object appears as a **horizontal band** centered at the detector midpoint
- An off-center object creates a **shifted horizontal band**

By measuring where the phantom appears in the sinogram, we can calculate the detector center offset.

### Multi-Position Strategy

The script processes **multiple phantom positions** (different distances from the rotation center) to:
1. **Average out noise** and segmentation errors
2. **Validate consistency** across different geometries
3. **Improve robustness** by statistical averaging



## Workflow

### Step 1: Data Loading
### Step 2: Sinogram Generation
### Step 3: Interactive ROI Selection
```
selecionar_roi_interativamente(sino_raw)
```
**User draws TWO regions of interest (ROIs)**:

1. **Background ROI** (green): 
   - Area with **no object** (air/background)
   - Used to estimate intensity of empty space
   
2. **Object ROI** (orange): 
   - Area containing the **phantom sinogram signature**
   - Guides the segmentation algorithm to focus on the correct region

**Why two ROIs?**
- Background ROI → Calculates **threshold** for segmentation
- Object ROI → Acts as a **spatial hint** to avoid detecting wrong features


### Step 4: Shift Calculation via Segmentation

**Adaptive thresholding**:
1. Calculate `mean` and `std` of the background ROI
2. Threshold = `2 * std_background`
3. Detect if phantom is **darker** or **brighter** than background:
   - If **darker**: `mask = (background_mean - sino) > threshold`
   - If **brighter**: `mask = (sino - background_mean) > threshold`

**Component selection**:
1. Apply **morphological opening** (noise removal)
2. Label all connected components
3. Select the component with **maximum overlap** with the object ROI
4. If no overlap, fallback to the **largest component**

**Shift measurement**:
```
object_center = (det_min + det_max) / 2
geometric_center = detector_height / 2
shift = geometric_center - object_center
```

**Output**: 
- `shift_val` (pixels): How many pixels the detector center is offset
- Positive = object appears **below** center → shift detector **down**
- Negative = object appears **above** center → shift detector **up**

### Step 5: Visualization
### Step 6: Multi-Folder Processing

Processes all subfolders (named by distance, e.g., `0.0`, `+2.5`, `-1.0`):
1. Extracts distance from folder name
2. Sorts folders by distance
3. Runs calibration for each position
4. Collects all shift measurements

### Step 7: Statistical Summary & Export

**Calculates**:
- Average shift across all positions
- Standard deviation (measurement uncertainty)
- Average background intensity (for I₀ normalization)





### Quality Indicators

**Good calibration**:
- ✅ Low standard deviation (<0.5 px) → Consistent measurements
- ✅ Shift doesn't vary much with distance → Systematic offset
- ✅ Clean segmentation in all positions






