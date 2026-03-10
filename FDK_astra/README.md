# reconstruction_comparison

Three self-contained FDK reconstruction scripts that benchmark alternative
backends against the TIGRE reference.  Every script:

- Imports pre-processing helpers **directly** from `FDK_reduce_memory/`
- Runs the **identical** Phase 1 & 2 pipeline (same GUI, same Beer-Lambert,
  same crop, same downsample) as `MAIN_TIGRE_FDK_Voxel_size.py`
- Builds its own geometry for the specific backend
- Shows the result in **Napari** and optionally exports to **.nii**

---

## Scripts

| File | Backend | Filter |
|---|---|---|
| `recon_tigre_fdk.py` | TIGRE `algs.fdk` (GPU) | ram_lak / shepp_logan / hann / cosine |
| `recon_astra_fdk.py` | ASTRA `FDK_CUDA` (GPU) | Ram-Lak only (ASTRA limitation) |
| `recon_fdk_scratch.py` | from-scratch NumPy/CuPy | ram_lak / shepp_logan / hann / cosine |

---

## How to run

Edit `CONFIG` and `FOLDER` at the bottom of whichever script you want, then:

```
python recon_tigre_fdk.py
python recon_astra_fdk.py
python recon_fdk_scratch.py
```

Each script is fully independent.

---

## Dependencies

```
pip install astra-toolbox cupy-cuda12x   # adjust cupy version for your CUDA
```

| Package | Purpose |
|---|---|
| tigre / pytigre | TIGRE backend + geometry struct |
| astra-toolbox ≥ 2.1 (CUDA build) | ASTRA backend |
| cupy-cuda12x ≥ 9 | GPU for scratch FDK |
| numpy, scipy | everywhere |
| napari | volume viewer |
| nibabel | NIfTI export |

---

## Geometry notes (TIGRE → ASTRA)

All distances are in **mm**.  Angles in **radians, CCW from above**.

| TIGRE field | ASTRA cone_vec mapping |
|---|---|
| `geo.DSD` | `src_pos + det_pos` distance |
| `geo.DSO` | radial distance from isocentre to source |
| `geo.dDetector` | u and v pixel vectors scaled by pitch |
| `geo.offDetector[1]` | horizontal offset baked into detector centre |
| `geo.offDetector[0]` | vertical offset baked into detector centre (fixed in Z) |
| `geo.rotDetector[2]` | in-plane tilt applied via Rodrigues rotation on the u-vector |
| `geo.nVoxel [Z,Y,X]` | `create_vol_geom(nY, nX, nZ, ...)` |

ASTRA `data3d.get()` returns `(nZ, nY, nX)` – same as TIGRE, no transpose needed.

### Known intensity difference: ASTRA vs TIGRE

ASTRA `FDK_CUDA` and TIGRE use different filter normalisation constants.
The reconstructed intensities will have a different absolute scale.
This is expected — compare relative contrast and structure first.
A linear rescale can be applied empirically once you observe the difference.
