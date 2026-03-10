"""
Minimal local shim for `tigre` to allow the project to run without the
upstream Tigre package installed. This provides a tiny `Geometry` container
and a `geometry()` factory so `import tigre` and `tigre.geometry(mode=...)`
work in code that only reads/writes geometry fields.

This is intentionally minimal — it only implements the fields the repo
reads (nDetector, dDetector, nVoxel, dVoxel, sDetector, sVoxel, DSD, DSO,
offDetector, offOrigin, rotDetector, __version__).
"""

__all__ = ["geometry", "Geometry", "__version__"]

__version__ = "0.0.local-shim"

import numpy as np


class Geometry:
    def __init__(self, mode="cone"):
        self.nDetector   = np.zeros(2, dtype=np.int64)
        self.dDetector   = np.zeros(2, dtype=np.float64)
        self.sDetector   = np.zeros(2, dtype=np.float64)

        self.nVoxel      = np.zeros(3, dtype=np.int64)
        self.dVoxel      = np.zeros(3, dtype=np.float64)
        self.sVoxel      = np.zeros(3, dtype=np.float64)

        self.DSD         = 0.0
        self.DSO         = 0.0

        self.offDetector = np.zeros(2, dtype=np.float64)
        self.offOrigin   = np.zeros(3, dtype=np.float64)
        self.rotDetector = np.zeros(3, dtype=np.float64)


def geometry(mode="cone"):
    """Factory returning a Geometry instance (signature compatible with tigre.geometry)."""
    return Geometry(mode=mode)
