"""Hounsfield Unit (HU) calibration for CT reconstructed volumes."""

import logging

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

HU_WATER_REFERENCE = 0.0
HU_AIR_REFERENCE = -1000.0
_PLOT_MARGIN_FRACTION = 0.2


def HU_conversion(volume: np.ndarray, water_val: float, air_val: float) -> np.ndarray:
    """Convert a reconstructed volume from raw attenuation values to Hounsfield Units.

    Applies a two-point linear calibration using measured air and water intensities.
    Generates and displays the calibration line plot.

    Args:
        volume: Reconstructed volume array, any shape, float32.
        water_val: Mean intensity measured in a water region of the reconstruction.
        air_val: Mean intensity measured in an air region of the reconstruction.
            Must differ from water_val.

    Returns:
        volume_hu: Array of same shape as input, values in Hounsfield Units (float32).

    Raises:
        ValueError: If water_val == air_val (would cause division by zero).
    """
    if water_val == air_val:
        raise ValueError(
            f"water_val and air_val must differ for HU calibration; both are {water_val}"
        )

    # Two-point calibration: y = mx + b, anchored at (air_val, HU_AIR) and (water_val, HU_WATER)
    m = (HU_WATER_REFERENCE - HU_AIR_REFERENCE) / (water_val - air_val)
    b = HU_WATER_REFERENCE - m * water_val

    logger.info("HU calibration | slope=%.4f  intercept=%.4f", m, b)
    logger.info("Equation: HU = %.4f * pixel + %.4f", m, b)

    margin = abs(water_val - air_val) * _PLOT_MARGIN_FRACTION
    x_range = np.linspace(min(air_val, water_val) - margin,
                          max(air_val, water_val) + margin, 100)
    y_range = m * x_range + b

    plt.figure(figsize=(8, 6))
    plt.plot(x_range, y_range, 'b-', label='Calibration Line')
    plt.scatter([air_val], [HU_AIR_REFERENCE], color='red', s=100, zorder=5,
                label=f'Air ({HU_AIR_REFERENCE:.0f} HU)')
    plt.scatter([water_val], [HU_WATER_REFERENCE], color='green', s=100, zorder=5,
                label=f'Water ({HU_WATER_REFERENCE:.0f} HU)')
    plt.title(f'Calibration: Intensity vs Hounsfield Units\n(Air={air_val}, Water={water_val})')
    plt.xlabel('Input Value (float32)')
    plt.ylabel('Hounsfield Units (HU)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    return (volume * m + b).astype(np.float32)
