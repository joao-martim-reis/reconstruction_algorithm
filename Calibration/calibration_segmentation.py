"""
Segmentation and shift computation for calibration.
"""

import numpy as np
from scipy.ndimage import label, binary_opening, find_objects


def calculate_shift_segmentation(sino_raw, roi_background, roi_object):
    print("--> Processing mask...")

    # Use background ROI for threshold stats
    r_start, r_end, c_start, c_end = roi_background
    roi_crop = sino_raw[r_start:r_end, c_start:c_end]

    mean_fundo = np.mean(roi_crop)
    std_fundo = np.std(roi_crop)
    threshold = 2 * std_fundo

    # Determine object contrast direction
    r_obj_start, r_obj_end, c_obj_start, c_obj_end = roi_object
    roi_object_crop = sino_raw[r_obj_start:r_obj_end, c_obj_start:c_obj_end]
    mean_object = np.mean(roi_object_crop)

    print(f"  Background mean: {mean_fundo:.2f} | Object mean: {mean_object:.2f}")

    if mean_object < mean_fundo:
        print("  Object is DARKER than background")
        mascara_inicial = (mean_fundo - sino_raw) > threshold
    else:
        print("  Object is BRIGHTER than background")
        mascara_inicial = (sino_raw - mean_fundo) > threshold

    mascara_limpa = binary_opening(mascara_inicial, structure=np.ones((3, 3)))
    labeled_array, num_features = label(mascara_limpa)
    print(f"Detected {num_features} connected components.")

    if num_features > 0:
        r_start, r_end, c_start, c_end = roi_object
        roi_mask = np.zeros_like(labeled_array, dtype=bool)
        roi_mask[r_start:r_end, c_start:c_end] = True

        best_label = None
        best_overlap = 0

        for i in range(1, num_features + 1):
            componente = labeled_array == i

            n_pixels = np.sum(componente)
            if n_pixels < 200:
                continue
            overlap = np.sum(componente & roi_mask)

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = i

        if best_label is not None:
            mascara_final = labeled_array == best_label
            print(f"Selected component {best_label} with {best_overlap} pixels inside ROI.")
        else:
            sizes = np.bincount(labeled_array.ravel())[1:]
            if len(sizes) > 0:
                best = np.argmax(sizes) + 1
                mascara_final = labeled_array == best
                print("Warning: No component overlaps ROI. Using largest component.")
            else:
                mascara_final = mascara_limpa
    else:
        print("Warning: No object detected.")
        mascara_final = mascara_inicial

    obj_slices = find_objects(mascara_final.astype(int))
    if obj_slices and obj_slices[0] is not None:
        det_min = obj_slices[0][0].start
        det_max = obj_slices[0][0].stop - 1
    else:
        coords = np.where(mascara_final)
        det_min = np.min(coords[0])
        det_max = np.max(coords[0])

    centro_objeto = (det_min + det_max) / 2.0
    centro_geo = sino_raw.shape[0] / 2.0

    shift_val = centro_geo - centro_objeto

    print("--- Shift Report ---")
    print(f"Object Center: {centro_objeto:.2f} | Geo Center: {centro_geo:.2f}")
    print(f"Calculated shift: {shift_val:.2f} pixels")

    return shift_val, mascara_final, centro_objeto, det_min, det_max, mean_fundo
