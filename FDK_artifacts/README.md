# CT Reconstruction Pipeline - FDK Algorithm

## Overview
This pipeline implements a complete 3D Cone Beam CT reconstruction workflow using the **Feldkamp-Davis-Kress (FDK)** algorithm via the TIGRE library. The process transforms raw X-ray projection images (TIFF files) into a calibrated 3D volume that can be optionally converted to Hounsfield Units (HU) for medical/material analysis.

---

## Files in This Module

### 1. `data_processing_3D_2.py`
Handles all **preprocessing** steps: loading projections, creating visualization tools (collapsed sinograms), and performing normalization through user-guided ROI selection.

### 2. `TIGRE_fdk_2.py`
Performs **geometry setup, reconstruction (FDK algorithm), and export** to NIfTI format (.nii). Optionally converts reconstructed volumes to Hounsfield Units (HU) for quantitative analysis.

---

## Pipeline Workflow (Step-by-Step)

### **STEP 1: Load Projection Images** 
**File:** `data_processing_3D_2.py` → `load_images()`

**What it does:**
- Scans the specified folder for `.tif` files
- Extracts numerical indices from filenames using regex to ensure proper angular sorting
- Loads all projections into a 3D NumPy array with shape `(Height, Width, N_projections)`
- Displays the first and last 5 filenames to verify correct ordering

**Why it's needed:**
Raw CT data consists of hundreds of 2D X-ray projections taken at different rotation angles. Proper loading and sorting is **critical** — incorrect angular ordering will produce severe artifacts in the reconstruction.

---

### **STEP 2: Generate Collapsed Sinogram**
**File:** `data_processing_3D_2.py` → `generate_collapsed_sinogram()`

**What it does:**
- Sums all projections along the height axis (vertical detector dimension)
- Creates a 2D "collapsed sinogram" with dimensions `(Width, N_angles)`
- This provides a quick preview showing how X-ray attenuation varies across detector positions and rotation angles

**Why it's needed:**
The collapsed sinogram is used for:
1. **Visual quality check** — identifying motion artifacts, incomplete rotations, or detector issues
2. **ROI selection interface** — allows the user to select background (air) regions for normalization

**Visualization Details:**
- X-axis represents rotation angle (0° to 360°, labeled every 30°)
- Y-axis represents detector pixel position
- Bright regions = high X-ray transmission (low attenuation, e.g., air)
- Dark regions = high attenuation (e.g., dense materials)

---

### **STEP 3: Select I₀ ROI for Normalization**
**File:** `data_processing_3D_2.py` → `selecionar_roi_I0()`

**What it does:**
- Displays the collapsed sinogram with an interactive ROI selection tool
- User draws a rectangular region on **background/air areas** (regions with no sample)
- The selected ROI defines the reference intensity **I₀** (unattenuated beam intensity)
- A "Confirm" button saves the ROI coordinates and closes the interface

**Why it's needed:**
CT reconstruction requires converting raw intensity values (I) to **attenuation coefficients** using the Beer-Lambert law:

$$\mu = -\ln\left(\frac{I}{I_0}\right)$$

Where:
- $I$ = measured intensity (with sample)
- $I_0$ = reference intensity (without sample, i.e., air/background)
- $\mu$ = linear attenuation coefficient

Without proper I₀ calibration, the reconstructed volume will have incorrect contrast and quantitative errors.

**Best Practices:**
- Select a region that spans **multiple angles** (wide X-range) to account for beam intensity variations
- Avoid edges of the detector where vignetting or scatter may occur
- The ROI should contain **only air**, not parts of the sample or support structures

---

### **STEP 4: Calculate Mean I₀ from ROI**
**File:** `data_processing_3D_2.py` → `get_I0_from_roi()`

**What it does:**
- Extracts the pixel values within the user-defined background ROI
- Computes the mean intensity value → this becomes the normalization reference **I₀**

**Why it's needed:**
Provides a single scalar value representing "unattenuated beam intensity" for all projections. This assumes the X-ray source and detector are stable across the entire scan.


### **STEP 5: Show Preprocessing Results**
**File:** `data_processing_3D_2.py` → `show_results()`

**What it does:**
- Displays two side-by-side plots:
  1. **Raw sinogram** (before normalization)
  2. **Normalized sinogram** (after applying I₀ correction)
- Overlays a red dashed line indicating the **geometric center** of the detector

**Why it's needed:**
Allows the user to visually verify that normalization worked correctly. The normalized sinogram should have:
- Uniform background intensity (air regions should be dark/near-zero after `-log` transformation)
- Clear contrast between sample and background
- No obvious ring artifacts or intensity gradients

--------------------------------------------------------------------------------------------------------------------

### **STEP 6: Normalize Projections (Beer-Lambert Transform)**
**File:** `TIGRE_fdk_2.py` → `normalize_projections()`

**What it does:**
1. Converts raw intensity images to attenuation coefficients using:
   $$\text{projections\_norm} = -\ln\left(\frac{I}{I_0}\right)$$
2. Applies numerical safeguards:
   - Clips the ratio $I/I_0$ to the range `[1e-6, 1.2]` to prevent logarithm errors
   - Sets any negative attenuation values to zero (physically impossible)
3. Uses the I₀ value obtained from the ROI (or falls back to 1st percentile if not provided)

**Why it's needed:**
The FDK algorithm operates on **line integrals of attenuation** (Radon transform), not raw intensities. This transformation is mathematically required for correct reconstruction.

**Potential Issues:**
- ⚠️ The clipping range `[1e-6, 1.2]` assumes low-noise data. If you have high detector noise or scatter, adjust these values.
- ⚠️ Clipping to 1.2 means the algorithm cannot handle "superattenuating" scenarios where $I > I_0$ (e.g., phase contrast effects). This should not occur in standard absorption CT.

---

### **STEP 7: Apply Downsampling (Optional)**
**File:** `TIGRE_fdk_2.py` → `main()` (lines ~267-273)

**What it does:**
- If `downsample > 1` in the configuration, reduces projection resolution by factor `f`
- Uses NumPy slicing: `projections[::f, ::f, :]` (takes every f-th pixel)
- Adjusts `pixel_size` accordingly: `pixel_size_new = pixel_size_original × f`

**Why it's needed:**
- **Memory optimization** — downsampling by 4× reduces memory usage by ~16×
- **Speed** — reconstruction time scales with $O(n^3)$ for volume size
- **Noise reduction** — averaging nearby pixels can improve SNR (though slicing doesn't average, true binning would be better)

**Calibration Adjustment:**
The `calibrated_shift_px` value (from detector center-of-rotation calibration) must be divided by the downsample factor:
$$\text{shift\_val} = \frac{\text{calibrated\_shift\_px}}{\text{downsample}}$$

This ensures the shift correction remains accurate at the new resolution.

---

### **STEP 8: Setup CT Geometry**
**File:** `TIGRE_fdk_2.py` → `setup_geometry()`

**What it does:**
Configures the TIGRE `geometry` object with all system parameters:

1. **Detector Configuration:**
   - `nDetector` = number of pixels `[Height, Width]`
   - `dDetector` = physical pixel size in mm
   - `sDetector` = total detector size (n × d)

2. **Voxel Configuration:**
   - Calculates voxel size based on geometric magnification:
     $$\text{voxel\_size} = \frac{\text{pixel\_size}}{\text{magnification}}$$
     where $\text{magnification} = \frac{\text{DSD}}{\text{DSO}}$
   - Sets `nVoxel` to match detector FOV when backprojected to isocenter
   - Defines cubic voxels (equal size in X, Y, Z)

3. **Source-Detector Geometry:**
   - `DSD` = Distance Source to Detector (mm)
   - `DSO` = Distance Source to Object/Isocenter (mm)
   - These define the cone beam geometry and magnification factor

4. **Center-of-Rotation Shift Correction:**
   - `offDetector = [0.0, shift_mm × shift_sign]`
   - Corrects for misalignment between the detector center and rotation axis
   - `shift_sign` allows inverting the correction direction if needed

5. **Rotation Angles:**
   - Generates array from 0 to `total_angle` (typically 2π radians = 360°)
   - Uses `endpoint=False` to avoid duplicate 0°/360° projection

**Why it's needed:**
Accurate geometry is **critical** for FDK reconstruction. Even small errors (e.g., 0.1mm shift, 1° angular error) will cause:
- Blurring
- Double edges
- Ring artifacts
- Loss of spatial resolution

**Potential Issues:**
- ⚠️ The current implementation assumes **circular cone beam** geometry with no gantry tilt or detector rotation (`rotDetector = [0, 0, 0]`). If your system has these, additional parameters must be configured.
- ⚠️ **Shift sign ambiguity:** The `shift_sign` parameter may need to be toggled depending on the detector's coordinate system convention. If reconstructions show persistent double edges, try flipping this value.

---

### **STEP 9: Run FDK Reconstruction**
**File:** `TIGRE_fdk_2.py` → `main()` (line ~287)

**What it does:**
1. Transposes normalized projections from `(Height, Width, Angles)` to `(Angles, DetectorV, DetectorU)` (TIGRE's expected format)
2. Calls `tigre.algorithms.fdk()` with:
   - Input projections (attenuation data)
   - Geometry object
   - Angle array
   - Filter type (e.g., `'hann'`, `'ram-lak'`, `'shepp-logan'`)

**Algorithm Details:**
The FDK algorithm performs:
1. **Filtering:** Applies a ramp filter in Fourier space (with optional window function like Hann)
2. **Weighting:** Applies cone-beam geometric weighting
3. **Backprojection:** Each filtered projection is "smeared back" through 3D space along ray paths

**Filter Selection:**
- `'ram-lak'` (ramp) — sharpest, but amplifies noise
- `'hann'` — balanced, recommended for most cases
- `'shepp-logan'` — smoothest, good for low-dose/noisy data
- `'cosine'` — intermediate smoothing

**Why it's needed:**
FDK is the standard analytical algorithm for cone-beam CT reconstruction. It's fast (GPU-accelerated in TIGRE) and produces high-quality results for circular trajectories.

**Output:**
- 3D volume with shape `(Z, Y, X)` where Z is the superior-inferior axis
- Values represent **linear attenuation coefficients** (μ) in units of mm⁻¹
- Typical range: 0.01–0.05 mm⁻¹ for soft tissues, 0.15–0.25 mm⁻¹ for bone

---

### **STEP 10: Volume Information Display**
**File:** `TIGRE_fdk_2.py` → `print_volume_info()`

**What it does:**
- Prints detailed summary of the reconstructed volume:
  - Data type (should be `float32`)
  - Dimensions (voxel count in each axis)
  - Voxel size (mm)
  - Physical dimensions (total FOV in mm)
  - Geometric parameters (DSD, DSO)

**Why it's needed:**
Provides immediate feedback to verify reconstruction succeeded and check if parameters are as expected.

---

### **STEP 11: Export to NIfTI Format (.nii)**
**File:** `TIGRE_fdk_2.py` → `export_volume_to_nii()`

**What it does:**
1. Creates a timestamped output folder:
   ```
   reconstructed_volumes/
   └── TIGRE_fdk_2_<dataset_name>_<day>_<month>_<hour>h<minute>/
       ├── <dataset_name>.nii
       └── metadata.txt
   ```

2. **Axis Transformation:**
   - TIGRE outputs volumes as `(Z, Y, X)` (slice, row, column)
   - NIfTI standard expects `(X, Y, Z)` in RAS orientation
   - Applies transpose: `np.transpose(volume, (2, 1, 0))`

3. **Creates Affine Matrix:**
   - Defines voxel spacing in X, Y, Z directions
   - Centers the volume at the origin
   - Ensures correct spatial registration in visualization software

4. **Saves Metadata:**
   - Reconstruction timestamp
   - Source dataset path
   - Volume statistics (min, max, mean, std)
   - Geometry parameters
   - Python/TIGRE version info

**Why it's needed:**
NIfTI is the standard format for medical imaging and is compatible with:
- **3D Slicer** (medical image analysis)
- **ImageJ/Fiji** (with NIfTI plugin)
- **Napari** (Python visualization)
- **FSL, SPM, AFNI** (neuroimaging tools)

**CRITICAL NOTE — Value Preservation:**
The function docstring explicitly states:
> **"NO scaling, normalization, or clipping is applied to the data"**

This means:
- ✅ Attenuation coefficient values are preserved exactly as computed by FDK
- ✅ Quantitative analysis (e.g., measuring μ-values) is valid
- ✅ Contrast adjustments in viewers are **display-only** and don't modify the file

**Potential Issues:**
- ⚠️ The affine matrix assumes **isotropic voxels** and **no rotation**. If your reconstruction uses anisotropic voxels or has spatial rotations, the affine must be adjusted.

---

### **STEP 12: Napari Visualization**
**File:** `TIGRE_fdk_2.py` → `main()` (lines ~300-314)

**What it does:**
- Opens an interactive 3D viewer (Napari) with the reconstructed volume
- Configures proper voxel spacing for accurate spatial representation
- Allows slicing, rotating, and inspecting the volume interactively

**Why it's needed:**
Immediate visual feedback to:
- Check for reconstruction artifacts (rings, streaks, blurring)
- Verify anatomical/sample features are correctly resolved
- Assess need for parameter adjustments (e.g., filter type, shift correction)

---

### **STEP 13: Hounsfield Unit (HU) Conversion (Optional)**
**File:** `TIGRE_fdk_2.py` → `export_volume_HU()` + `HU_conversion.py`

**What it does:**
1. Prompts user to provide **calibration values**:
   - Gray scale value for **water** (measured from reconstruction)
   - Gray scale value for **air** (measured from reconstruction)

2. Applies linear transformation:
   $$HU = 1000 \times \frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}} - \mu_{\text{air}}}$$

   This maps:
   - Water → 0 HU (by definition)
   - Air → -1000 HU (by definition)
   - Bone → ~+1000 HU (depending on density)

3. Saves a new file with `_HU.nii` suffix, preserving the original header/affine

**Why it's needed:**
Hounsfield Units are the **standard quantitative scale** in medical CT imaging:
- Allows comparing results across different scanners and acquisition parameters
- Enables material classification (e.g., fat = -100 to -50 HU, muscle = +10 to +40 HU)
- Required for many automated segmentation and analysis tools

**Calibration Requirements:**
- You must measure gray scale values from a known phantom (with water and air) or from recognizable regions in your sample
- Use Napari or ImageJ to measure mean values in ROI
- The conversion assumes **linear response** — this is valid for monochromatic sources but may introduce errors with polychromatic X-ray tubes (beam hardening)

**Potential Issues:**
- ⚠️ **Beam hardening not corrected** — if your reconstruction shows "cupping artifacts" (edges brighter than center), the HU conversion will be inaccurate. Consider applying beam hardening correction before HU conversion.

---

## Configuration Parameters (TIGRE_fdk_2.py)

Located in the `CONFIG` dictionary at the bottom of `TIGRE_fdk_2.py`:

| Parameter | Description | Units | Typical Values |
|-----------|-------------|-------|----------------|
| `pixel_size` | Physical size of detector pixel | mm | 0.05 - 0.2 |
| `DSD` | Distance Source to Detector | mm | 500 - 1500 |
| `DSO` | Distance Source to Object (isocenter) | mm | DSD - (50-200) |
| `downsample` | Resolution reduction factor | dimensionless | 1, 2, 4 |
| `total_angle` | Total rotation range | radians | 2π (360°) or π (180°) |
| `calibrated_shift_px` | Center-of-rotation offset | pixels | From calibration |
| `shift_sign` | Direction of shift correction | +1 or -1 | Empirically determined |
| `filter_type` | Reconstruction filter | string | `'hann'`, `'ram-lak'`, `'shepp-logan'` |
| `volume_roi` | Subvolume reconstruction (optimization) | dict or None | `None` = full volume |

---

## Execution Order Summary

```
[1] load_images()                      → Load all projections from TIFF files
          ↓
[2] (optional) downsample              → Reduce resolution for speed/memory
          ↓
[3] generate_collapsed_sinogram()      → Create sum-projection for visualization
          ↓
[4] selecionar_roi_I0()                → User selects background ROI
          ↓
[5] get_I0_from_roi()                  → Calculate mean I₀ value
          ↓
[6] normalize_projections()            → Apply Beer-Lambert transform: -ln(I/I₀)
          ↓
[7] generate_collapsed_sinogram()      → Preview normalized data
          ↓
[8] show_results()                     → Display before/after normalization
          ↓
[9] setup_geometry()                   → Configure TIGRE geometry (shift, angles, etc.)
          ↓
[10] algs.fdk()                        → Run FDK reconstruction (GPU-accelerated)
          ↓
[11] print_volume_info()               → Display volume statistics
          ↓
[12] export_volume_to_nii()            → Save as NIfTI with metadata
          ↓
[13] napari.Viewer()                   → Interactive 3D visualization
          ↓
[14] (optional) export_volume_HU()     → Convert to Hounsfield Units
```

---

## Known Issues and Limitations

### 1. **Beam Hardening Not Corrected**
**Problem:** Polychromatic X-ray sources cause "cupping artifacts" (center appears darker than edges)

**Impact:** 
- Incorrect attenuation values in the volume center
- Errors in HU conversion
- Reduced contrast for soft tissues

**Solution:**
- Implement linearization correction (requires calibration with known materials)
- Apply iterative beam hardening correction algorithms
- Use water-based correction LUTs

### 2. **Scatter Correction Missing**
**Problem:** X-rays scattered by the sample are detected alongside primary radiation

**Impact:**
- Reduced contrast
- Cupping artifacts
- Quantitative errors in attenuation coefficients

**Solution:**
- Use anti-scatter grids during acquisition
- Apply software scatter correction (e.g., kernel-based methods)
- Measure scatter profiles with beam-stop array

### 3. **Ring Artifacts Possible**
**Problem:** Defective or miscalibrated detector pixels cause circular artifacts

**Impact:**
- Concentric rings in reconstructed slices
- Reduced image quality

**Solution:**
- Apply flat-field correction before reconstruction
- Use ring artifact reduction filters (e.g., polar coordinate median filter)
- Calibrate detector gain/offset

### 4. **Shift Sign Ambiguity**
**Problem:** The `shift_sign` parameter (+1 or -1) must be empirically determined

**Impact:**
- If wrong, double edges and blurring will persist even with correct shift magnitude

**Solution:**
- Reconstruct a test sample twice (with shift_sign = +1 and -1)
- Choose the configuration with sharpest edges
- Document the correct sign for your system

### 5. **Single I₀ Value Limitation**
**Problem:** Assumes constant beam intensity across all projections

**Impact:**
- If source intensity drifts (tube heating, power fluctuations), normalization will be incorrect

**Solution:**
- Implement per-projection I₀ estimation (track background ROI across all angles)
- Monitor source stability during acquisition
- Apply temporal intensity correction

### 6. **No Truncation Handling**
**Problem:** If the sample extends beyond the detector FOV, projections are truncated

**Impact:**
- Severe streaking artifacts radiating from truncated regions
- Quantitative errors outside the FOV

**Solution:**
- Use iterative reconstruction with extrapolation (e.g., SART, CGLS)
- Implement truncation compensation algorithms
- Ensure sample fits within detector FOV during acquisition

---

## Recommended Workflow

1. **First-time setup:**
   - Run calibration to determine `calibrated_shift_px` (use `Calibration/calibration_process.py`)
   - Test reconstruction with a simple phantom
   - Verify `shift_sign` by checking edge sharpness

2. **Standard reconstruction:**
   - Set `downsample=4` for initial preview (fast)
   - Run full pipeline, check Napari visualization
   - If quality is good, re-run with `downsample=1` for final high-resolution output

3. **Parameter tuning:**
   - If noisy: use `filter_type='hann'` or `'shepp-logan'`
   - If blurry: use `filter_type='ram-lak'`
   - If double edges persist: adjust `calibrated_shift_px` or toggle `shift_sign`

4. **HU conversion (if needed):**
   - Reconstruct a calibration phantom (or identifiable sample with known materials)
   - Measure gray scale values in water and air regions using Napari
   - Apply HU conversion and verify values match expected ranges

---


## References

- **FDK Algorithm:** Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984). "Practical cone-beam algorithm." *JOSA A*, 1(6), 612-619.
- **TIGRE Toolbox:** Biguri, A., et al. (2016). "TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction." *Biomedical Physics & Engineering Express*, 2(5).
- **NIfTI Format:** https://nifti.nimh.nih.gov/
- **Hounsfield Units:** Hounsfield, G. N. (1973). "Computerized transverse axial scanning (tomography)." *British Journal of Radiology*, 46(552), 1016-1022.

---

## Author Notes

This pipeline was developed for micro-CT and lab-based CT systems. The code emphasizes:
- **Transparency:** All processing steps are explicit (no hidden preprocessing)
- **Reproducibility:** Timestamped outputs with full metadata
- **Flexibility:** Easy parameter adjustment via CONFIG dictionary

For questions or issues, review the "Known Issues" section or inspect intermediate outputs (sinograms, Napari visualization) to diagnose problems.

---

**Last Updated:** January 12, 2026
