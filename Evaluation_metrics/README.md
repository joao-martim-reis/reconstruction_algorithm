# Evaluation Metrics for Micro-CT Quality Control

This folder contains Python scripts for evaluating the performance of a micro-CT system using standardized quality control metrics. The analysis is performed on **grey scale values** (not Hounsfield Units) as no HU calibration phantom is currently available.

## Reference

These metrics are based on:
- **Mambrini et al. (2022)**: "The importance of routine quality control for reproducible pulmonary measurements by in vivo micro-CT." *Scientific Reports*. DOI: 10.1038/s41598-022-13477-7
- **AAPM Task Group 233**: Performance evaluation of CT systems

---

## Available Metrics

### 1. Uniformity, Noise & SNR (`Uniformity_coise_water.py`)

**Phantom Required:** Water Phantom (uniform material)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Uniformity** | `mean(\|μ_central - μ_peripheral\|)` | Lower is better - measures spatial homogeneity |
| **Noise** | `mean(σ_i)` for all ROIs | Lower is better - random pixel variations |
| **SNR** | `μ / σ` (central ROI) | Higher is better - signal vs noise ratio |

**What it detects:**
- Beam hardening artifacts
- Ring artifacts  
- Reconstruction errors
- Detector non-uniformities

**Arguments (in script configuration section):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tif_path` | - | Path to water phantom TIFF stack |
| `output_folder` | - | Folder for Excel and PNG outputs |
| `pixel_size_mm` | 0.05 | Pixel size in mm (50 μm) |
| `roi_radius` | 40 | ROI radius in pixels |
| `distance_from_center` | 100 | Peripheral ROI distance from center (pixels) |
| `slices_above_below` | 10 | Slices above/below center for 3D ROI |

**ROI Size Example (with defaults):**
- Diameter: 4.00 mm (40 px × 2 × 0.05 mm)
- Height: 1.05 mm (21 slices × 0.05 mm)
- Size: **4.00 × 4.00 × 1.05 mm³**

---

### 2. Contrast-to-Noise Ratio (`contrast_to_noise_Ratio.py`)

**Phantom Required:** Water Phantom (with air visible outside)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **CNR** | `\|μ_water - μ_air\| / √(σ_water² + σ_air²)` | Higher is better - ability to distinguish materials |

**What it measures:**
- Detectability of low-contrast structures
- Ability to distinguish between different materials/tissues

**Arguments (in script configuration section):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tif_path` | - | Path to water phantom TIFF stack |
| `output_folder` | - | Folder for Excel and PNG outputs |
| `pixel_size_mm` | 0.05 | Pixel size in mm (50 μm) |
| `roi_radius` | 25 | ROI radius in pixels |
| `slices_above_below` | 20 | Slices above/below center for 3D ROI |

**Interactive:** User clicks to select air ROI position.

**ROI Size Example (with defaults):**
- Diameter: 2.50 mm (25 px × 2 × 0.05 mm)
- Height: 2.05 mm (41 slices × 0.05 mm)
- Size: **2.50 × 2.50 × 2.05 mm³**

---

### 3. Spatial Resolution - MTF (`spactial_resolution.py`)

**Phantom Required:** Bar Pattern Phantom (line-pair phantom)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MTF (%)** | `[(CT_high - CT_low) / (CT_high + CT_low)] × 100` | Higher is better - 100% = ideal contrast |

**What it measures:**
- Spatial resolution capability
- Smallest detectable structures
- System's ability to resolve fine details

**Arguments (in script configuration section):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tif_path` | - | Path to bar pattern TIFF image/stack |
| `output_folder` | - | Folder for Excel and PNG outputs |
| `pixel_size_mm` | 0.05 | Pixel size in mm (50 μm) |
| `peak_distance` | 10 | Min distance between peaks (pixels) |

**Interactive:** 
1. User selects slice with slider
2. User draws line perpendicular to bar pattern (2 clicks)

---

## Usage

All scripts follow the same structure:

```python
# 1. Edit configuration section at the top of the script
tif_path = r"path/to/your/image.tif"
output_folder = r"path/to/output/folder"
pixel_size_mm = 0.05  # Your actual pixel size

# 2. Run the script
python script_name.py
```

Or run from command line:
```bash
python Uniformity_coise_water.py
python contrast_to_noise_Ratio.py
python spactial_resolution.py
```

---

## Outputs

Each script generates:

1. **PNG Graph** - Visualization of ROIs and results
2. **Excel File** - Detailed results including:
   - ROI size in mm³
   - Calculated metrics
   - ROI statistics (mean, std, position)
   - Measurement parameters

---

## ROI Configuration

All scripts use **cylindrical ROIs** (circular in XY plane, extended across multiple slices) for improved statistical reliability:

```
ROI Volume = π × radius² × height
```

The physical size is reported as: `D × D × H mm³` where:
- D = diameter (2 × radius × pixel_size)
- H = height (num_slices × pixel_size)

---

## Additional Metrics from Literature

### Currently Implemented ✓
- [x] **Uniformity** - Spatial homogeneity (Mambrini et al. 2022)
- [x] **Noise** - Random variations (Mambrini et al. 2022)
- [x] **SNR** - Signal-to-Noise Ratio (Mambrini et al. 2022)
- [x] **CNR** - Contrast-to-Noise Ratio (Standard QC)
- [x] **MTF** - Spatial Resolution (Bar pattern article)

### Future Metrics (Require Additional Phantoms)

| Metric | Phantom Required | Purpose | Reference |
|--------|-----------------|---------|-----------|
| **HU Calibration** | Tungsten rod + known materials | Convert grey to Hounsfield Units | Mambrini et al. |
| **CT Number Accuracy** | Multi-material phantom | Verify HU values match expected | Standard QC |
| **Geometric Accuracy** | Precision phantom with known dimensions | Verify dimensional accuracy | AAPM TG 233 |
| **Ring Artifact Severity** | Water phantom | Quantify ring artifacts | Image processing |
| **Beam Hardening** | Step wedge phantom | Assess cupping artifacts | Standard QC |

### Metrics Requiring Small Tungsten Rod (Planned)

**Contrast Resolution Test**: Using a small tungsten rod or wire
- Measure contrast between rod and water
- Assess ability to detect small high-contrast objects
- Calculate CNR with different materials

---

## File Structure

```
Evaluation_metrics/
├── README.md                    # This file
├── Uniformity_coise_water.py    # Uniformity, Noise, SNR (water phantom)
├── contrast_to_noise_Ratio.py   # CNR (water phantom + air)
└── spactial_resolution.py       # MTF (bar pattern phantom)
```
