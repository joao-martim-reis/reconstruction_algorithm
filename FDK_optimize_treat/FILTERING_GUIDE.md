# FDK Reconstruction with Interactive Napari Filtering

This folder contains the FDK reconstruction algorithm with **interactive slider-based filtering**.

## Files

- **`TIGRE_fdk_2.py`** - Main reconstruction script with filtering integration
- **`napari_filters.py`** - Interactive filtering module with real-time sliders
- **`data_processing_3D_2.py`** - Data processing utilities

## How to Use

### Run Reconstruction

```bash
python TIGRE_fdk_2.py
```

### Interactive Filtering Workflow

After reconstruction completes, you'll be prompted:

```
Do you want to use interactive filtering? (y/n): y
```

This opens **napari with interactive sliders** on the right panel!

## Interactive Controls

### Available Filters (select from dropdown):

1. **None** - Show original volume
2. **Gaussian** - General smoothing
   - Slider: `Sigma` (0.1 to 5.0)
   
3. **Median** - Remove outliers, preserve edges
   - Slider: `Size` (3, 5, 7, 9)
   
4. **Bilateral** - Edge-preserving smoothing
   - Slider: `Spatial Sigma` (0.5 to 5.0)
   
5. **Ring Removal** - CT-specific artifacts
   - Slider: `Filter Size` (3 to 11)
   
6. **Combined** - Gaussian + Median
   - Slider: `Gaussian Sigma` (0.1 to 3.0)
   - Slider: `Median Size` (3, 5, 7)

### How to Use the Interface:

1. **Select filter type** from the dropdown menu
2. **Adjust sliders** for that filter's parameters
3. **Click "Apply Filter"** to see the result in real-time
4. **Repeat** steps 1-3 to try different settings
5. **Toggle "Original" layer** visibility to compare before/after
6. **When satisfied**, click **"Save Filtered Volume & Parameters"**

### What Gets Saved:

✓ **Filtered volume** as `.nii` file  
✓ **Filter parameters** in readable `.txt` file  
✓ **Command to reproduce** the exact filter

Example saved parameters file:
```
FILTER PARAMETERS
==================================================
Filter Type: Gaussian

Gaussian Sigma: 1.5

==================================================
To reproduce this filter:
apply_gaussian_filter(volume, sigma=1.5)
```

## Quick Start Guide

### First Time Users:

1. Run reconstruction
2. Choose `y` for interactive filtering
3. Read the auto-detected recommendation
4. Try the suggested filter first
5. Experiment with sliders until satisfied
6. Save your results

### Parameter Starting Points:

| Artifact Type | Filter | Starting Values |
|--------------|--------|-----------------|
| Light noise | Gaussian | sigma=1.0 |
| Heavy noise | Bilateral | spatial=2.0 |
| Salt-and-pepper | Median | size=3 |
| Ring artifacts | Ring Removal | size=5 |
| Mixed artifacts | Combined | gaussian=1.0, median=3 |

**Tip**: Start conservative (lower values) and increase gradually!

## Examples

### Example 1: Removing Ring Artifacts
1. Select **"Ring Removal"** from dropdown
2. Set **Filter Size slider** to 5
3. Click **"Apply Filter"**
4. If rings still visible, increase to 7 or 9
5. When happy, click **"Save Filtered Volume & Parameters"**

### Example 2: General Denoising
1. Select **"Gaussian"** from dropdown
2. Start with **Sigma = 1.0**
3. Click **"Apply Filter"** and observe
4. Increase sigma if more smoothing needed
5. Decrease if losing too much detail
6. Save when optimal

### Example 3: Preserve Details While Denoising
1. Select **"Bilateral"** from dropdown
2. Start with **Spatial Sigma = 2.0**
3. Click **"Apply Filter"**
4. Toggle Original layer to compare edges
5. Adjust sigma based on edge preservation
6. Save when satisfied

## Tips & Tricks

✓ **Toggle layers**: Click eye icon to compare Original vs Filtered  
✓ **Start low**: Begin with conservative parameters  
✓ **Iterate**: Try → adjust → try again until perfect  
✓ **Save parameters**: You can reuse them for similar datasets  
✓ **Experiment freely**: Changes are non-destructive (original preserved)

✗ **Don't over-filter**: More isn't always better  
✗ **Watch for detail loss**: Check fine structures aren't blurred  
✗ **Be patient**: Some filters (Bilateral, Ring) take time to process

## Performance

- **Fast**: Gaussian, None
- **Moderate**: Median (size 3), Combined
- **Slower**: Bilateral, Ring Removal, Median (size >5)

Large volumes process slice-by-slice with progress indicators.

## Saved Files

After saving, you'll find:
- `your_volume_filtered.nii` - The filtered volume
- `your_volume_filter_params.txt` - Parameters used

You can reload filtered volumes in napari, ImageJ, or 3D Slicer.
