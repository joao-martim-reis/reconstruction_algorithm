import numpy as np
import matplotlib.pyplot as plt

def HU_conversion(volume, water_val, air_val):
    """
    Converts a volume (numpy array) of 32-bit float values to Hounsfield Units (HU).
    Automatically generates the calibration line graph.

    """

    
    # 1. Define the reference physical values
    HU_water = 0.0
    HU_air = -1000.0
    
    # 2. Calculate the calibration line (y = mx + b)
    # m = (y2 - y1) / (x2 - x1)
    m = (HU_water - HU_air) / (water_val - air_val)
    
    # b = y - mx (using the water point to solve for b)
    b = HU_water - (m * water_val)
    
    print(f"--- Calibration Report ---")
    print(f"Slope (m): {m:.2f}")
    print(f"Intercept (b): {b:.2f}")
    print(f"Final Equation: HU = {m:.2f} * pixel + {b:.2f}")
    
    # 3. Display the calibration line
    plt.figure(figsize=(8, 6))
    
    # Create range for the X-axis of the graph
    # We extend 10% beyond the measured points for visualization
    margin = abs(water_val - air_val) * 0.2
    x_range = np.linspace(min(air_val, water_val) - margin, 
                          max(air_val, water_val) + margin, 100)
    y_range = m * x_range + b
    
    # Plot the line and points
    plt.plot(x_range, y_range, 'b-', label='Calibration Line')
    plt.scatter([air_val], [HU_air], color='red', s=100, zorder=5, label='Air (-1000 HU)')
    plt.scatter([water_val], [HU_water], color='green', s=100, zorder=5, label='Water (0 HU)')
    
    plt.title(f'Calibration: Intensity vs Hounsfield Units\n(Air={air_val}, Water={water_val})')
    plt.xlabel('Input Value (32-bit Float)')
    plt.ylabel('Hounsfield Units (HU)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()  # Display the graph
    
    # 4. Apply the conversion to the entire volume and return
    return (volume * m) + b

