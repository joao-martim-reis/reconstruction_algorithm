"""
IO utilities for calibration process.
"""

import os
import re

import numpy as np
import tifffile as tiff


def load_images(tiff_folder):
    print(f"--> Loading images from: {tiff_folder}")

    if not os.path.exists(tiff_folder):
        print("Error: Folder does not exist.")
        return None

    def extract_number(filename):
        match = re.search(r"\d+", filename)
        if match:
            return int(match.group())
        return 0

    file_list = [f for f in os.listdir(tiff_folder) if f.lower().endswith(".tif")]
    file_list.sort(key=extract_number)
    num_files = len(file_list)
    print(f"Number of files found: {num_files}")

    if not file_list:
        print("Error: The folder is empty")
        return None

    first_img = tiff.imread(os.path.join(tiff_folder, file_list[0]))
    height, width = first_img.shape
    print(f"Dimensions: {height} (H) x {width} (W) | {num_files} projections.")

    projections = np.zeros((height, width, num_files), dtype=first_img.dtype)
    for i, f in enumerate(file_list):
        projections[:, :, i] = tiff.imread(os.path.join(tiff_folder, f))

    print("--> Images Loaded Successfully")
    return projections


def generate_collapsed_sinogram(projections):
    print("--> Creating collapsed sinogram (sum projection)...")
    sino_sum = np.sum(projections, axis=0)
    return sino_sum
