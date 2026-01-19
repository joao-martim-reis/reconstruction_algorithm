import numpy as np
import tigre
import tigre.algorithms as algs
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from geometry_reconstruction import prepare_geometry

from data_processing_2D import (
    load_images, 
    extract_sinogram_raw, 
    selecionar_roi_I0,
    get_I0_from_roi,
    normalize_sinogram,
    show_results
)


def reconstruct_single(sino, pixel_size, shift_px, shift_sign, filter_type):
    """
    Perform a single FBP reconstruction.
    """
    geo, sino_input, angles = prepare_geometry(sino, pixel_size, shift_px, shift_sign)
    img = algs.fbp(sino_input, geo, angles, filter_type=filter_type)
    return np.squeeze(img) # Remove singleton dimensions


def select_crop_roi(projections, line, config):
    """
    Interactive ROI selection for cropping reconstructed images.
    
    Note: This is only used for more detailed visualization of specific regions.
    It does not affect the actual reconstruction, only the displayed results.
    
    Returns crop coordinates (row_start, row_end, col_start, col_end) or None for full image.
    """
    print("\n--> Select ROI for image cropping...")
    print("    Click 2 points (opposite corners) or close window to skip")
    
    sino_raw = extract_sinogram_raw(projections, line)
    img_preview = reconstruct_single(sino_raw, config['pixel_size'], 
                                     config['calibrated_shift_px'], 
                                     config['shift_sign'], 
                                     config['default_filter'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_preview, cmap='gray', interpolation='bilinear')
    ax.set_title("Click 2 points (opposite corners) or close window for full image", fontsize=12)
    ax.axis('on')
    
    pts = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close(fig)
    
    if len(pts) == 2:
        x1, y1 = int(round(pts[0][0])), int(round(pts[0][1]))
        x2, y2 = int(round(pts[1][0])), int(round(pts[1][1]))
        
        col_start, col_end = sorted([x1, x2])
        row_start, row_end = sorted([y1, y2])
        
        crop_roi = (row_start, row_end, col_start, col_end)
        print(f"    Crop ROI selected: {crop_roi}")
        return crop_roi
    else:
        print("    Using full image (no crop)")
        return None


def apply_crop(img, crop_roi):
    """
    Apply crop to image if crop_roi is provided.
    """
    if crop_roi is None:
        return img
    row_start, row_end, col_start, col_end = crop_roi
    return img[row_start:row_end, col_start:col_end]


def show_shift_comparison(projections, line, mean_I0, config, crop_roi):
    """
    Phase 1: Show shift effect for ONE specific line.
    Compare RAW with and without shift.
    """
    print(f"Shift Analysis - Line {line}")
    
    sino_raw = extract_sinogram_raw(projections, line)
    
    shift_px = config['calibrated_shift_px']
    
    configs = [
        {"label": "RAW - No Shift", "shift": 0.0},
        {"label": f"RAW - Shift: {shift_px:.2f} px", "shift": shift_px}
    ]
    
    results = []
    for cfg in configs:
        print(f"  Reconstructing: {cfg['label']}...")
        img = reconstruct_single(sino_raw, config['pixel_size'], cfg['shift'], 
                                config['shift_sign'], config['default_filter'])
        img_cropped = apply_crop(img, crop_roi)
        results.append({"label": cfg['label'], "image": img_cropped})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Store image objects for updating
    images = []
    
    for ax, result in zip(axes, results):
        ax.set_title(result['label'], fontsize=11, fontweight='bold', pad=8)
        im = ax.imshow(result['image'], cmap='gray', interpolation='bilinear')
        ax.axis('off')
        images.append((im, result))
    
    plt.suptitle(f"Shift Comparison - Line {line} - Filter: {config['default_filter']}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add slider for contrast control
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Contrast', 0.1, 2.0, valinit=1.0, valstep=0.05)

    def update_contrast(val):
        contrast = slider.val
        for im, result in images:
            data = result['image']
            vmin = np.percentile(data, 1) / contrast
            vmax = np.percentile(data, 99) * contrast
            im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    slider.on_changed(update_contrast)
    
    plt.tight_layout(pad=2)
    plt.show()


def show_filter_comparison(projections, lines, mean_I0, config, crop_roi):
    """
    Show effect of different FILTERS for multiple lines.
    Use RAW sinogram with SHIFT applied (better visualization).
    
    To use normalization instead of RAW, replace:
        sino_raw -> sino_norm = normalize_sinogram(sino_raw, mean_I0)
    """
    print(f"Filter Comparison")
    print(f"Chosen lines in the sinogram: {lines}")
    print(f"Filters: {config['filters']}")
    
    shift_px = config['calibrated_shift_px']
    
    for line in lines:
        sino_raw = extract_sinogram_raw(projections, line)
        
        # Use RAW for better visualization
        # To normalize: sino_norm = normalize_sinogram(sino_raw, mean_I0)
        sino_to_use = sino_raw
        
        results = []
        for filt in config['filters']:
            img = reconstruct_single(sino_to_use, config['pixel_size'], shift_px, 
                                    config['shift_sign'], filt)
            img_cropped = apply_crop(img, crop_roi)
            results.append({"label": f"Filter: {filt}", "image": img_cropped})
        
        n_filters = len(config['filters'])
        fig, axes = plt.subplots(1, n_filters, figsize=(5*n_filters, 5))
        
        if n_filters == 1:
            axes = [axes]
        
        # Store image objects for updating
        images = []
        for ax, result in zip(axes, results):
            ax.set_title(result['label'], fontsize=11, fontweight='bold', pad=8)
            im = ax.imshow(result['image'], cmap='gray', interpolation='bilinear')
            ax.axis('off')
            images.append(im)
        
        plt.suptitle(f"Filter Comparison - Line {line}", 
                     fontsize=13, fontweight='bold', y=0.98)
        
        # Add slider for contrast control
        plt.subplots_adjust(bottom=0.15)
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        
        slider = Slider(ax_slider, 'Contrast', 0.1, 2.0, valinit=1.0, valstep=0.05)
        
        def update_contrast(val):
            contrast = slider.val
            for im, result in zip(images, results):
                data = result['image']
                vmin = np.percentile(data, 1) / contrast
                vmax = np.percentile(data, 99) * contrast
                im.set_clim(vmin, vmax)
            fig.canvas.draw_idle()
        
        slider.on_changed(update_contrast)
        
        plt.tight_layout(pad=2)
        plt.show()


def run_reconstruction(tiff_folder, line_shift, lines_filters, configurations):
    """
    Main pipeline:
    1. Load images
    2. Select ROI for I0
    3. PHASE 1: Show shift effect on one line
    4. PHASE 2: Show filter effect on multiple lines
    """

    print("STARTING RECONSTRUCTION PIPELINE")

    projections = load_images(tiff_folder)
    
    sino_raw_ref = extract_sinogram_raw(projections, line_shift)
    roi_coords = selecionar_roi_I0(sino_raw_ref)
    mean_I0 = get_I0_from_roi(sino_raw_ref, roi_coords)
    
    sino_norm_ref = normalize_sinogram(sino_raw_ref, mean_I0)
    show_results(sino_raw=sino_raw_ref, sino_final_norm=sino_norm_ref, shift_val=configurations['calibrated_shift_px'], linha_escolhida=line_shift)
    crop_roi = select_crop_roi(projections, line_shift, configurations)
    
    show_shift_comparison(projections, line_shift, mean_I0, configurations, crop_roi)
    show_filter_comparison(projections, lines_filters, mean_I0, configurations, crop_roi)
    
    print("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    CONFIG = {
        'pixel_size': 0.05,
        'calibrated_shift_px': 5.12,
        'shift_sign': -1,
        'default_filter': 'hann',
        'filters': ['ram_lak', 'shepp_logan', 'hann']
        #'filters': ['ram_lak', 'shepp_logan', 'hann', 'hamming', 'cosine', 'blackman']
    }

    tiff_folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_simples_5'
    #tiff_folder = r'C:\Users\joaomartimreis\Desktop\Joao_CT\Imagens\Sistema_calhas\45kv+0.45mA\Phantom_800_1'

    line_shift = 400
    lines_filters = [50, 400, 800]

    run_reconstruction(tiff_folder, line_shift, lines_filters, CONFIG)
