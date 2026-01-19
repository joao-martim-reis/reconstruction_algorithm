import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def select_crop_region(first_projection):
    """
    Interface interativa para selecionar região de crop nas projeções RAW (antes da normalização).
    
    - Botão ESQUERDO: Define linha vertical → largura do crop (simétrico em relação ao centro)
    - Botão DIREITO: Define linha horizontal → altura do crop (do topo até a linha clicada)
    
    Returns:
        crop_params: dict com 'row_start', 'row_end', 'col_start', 'col_end'
    """
    print("--> [4/5] Select crop region on first projection...")
    print("    Instructions:")
    print("    1. LEFT CLICK → define VERTICAL line (symmetric crop around center)")
    print("    2. RIGHT CLICK → define HORIZONTAL line (crop from top to that line)")
    print("    3. Click 'Confirm' when satisfied with the crop")
    
    height, width = first_projection.shape
    center_col = width // 2
    
    # Estado da seleção
    crop_state = {
        'vertical_line': None,   # Distância do centro (em pixels)
        'horizontal_line': None  # Linha horizontal (row index)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    
    # Plot original
    ax1.imshow(first_projection, cmap='gray', aspect='auto', origin='upper')
    ax1.set_title("Original Projection\n(Click to define crop)", fontsize=12)
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    
    # Plot preview
    im2 = ax2.imshow(first_projection, cmap='gray', aspect='auto', origin='upper')
    ax2.set_title("Crop Preview", fontsize=12)
    ax2.set_xlabel("Width (px)")
    ax2.set_ylabel("Height (px)")
    
    # Linhas de guia
    vline_left = ax1.axvline(x=0, color='cyan', linestyle='--', linewidth=2, visible=False)
    vline_right = ax1.axvline(x=width, color='cyan', linestyle='--', linewidth=2, visible=False)
    hline = ax1.axhline(y=0, color='yellow', linestyle='--', linewidth=2, visible=False)
    center_line = ax1.axvline(x=center_col, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update_preview():
        """Atualiza o preview do crop"""
        if crop_state['vertical_line'] is not None and crop_state['horizontal_line'] is not None:
            dist = crop_state['vertical_line']
            row_end = crop_state['horizontal_line']
            
            col_start = max(0, center_col - dist)
            col_end = min(width, center_col + dist)
            row_start = 0
            
            cropped = first_projection[row_start:row_end, col_start:col_end]
            im2.set_data(cropped)
            im2.set_extent([col_start, col_end, row_end, row_start])
            ax2.set_xlim(col_start, col_end)
            ax2.set_ylim(row_end, row_start)
            ax2.set_title(f"Crop Preview: {row_end-row_start}H x {col_end-col_start}W px", fontsize=12)
        else:
            im2.set_data(first_projection)
            im2.set_extent([0, width, height, 0])
            ax2.set_xlim(0, width)
            ax2.set_ylim(height, 0)
            ax2.set_title("Crop Preview (select both lines)", fontsize=12)
        
        fig.canvas.draw()
    
    def on_click(event):
        if event.inaxes != ax1:
            return
        
        x_click, y_click = event.xdata, event.ydata
        
        # Botão esquerdo (1) → linha VERTICAL
        if event.button == 1:
            # Clique VERTICAL → define largura simétrica
            dist_from_center = abs(x_click - center_col)
            crop_state['vertical_line'] = int(dist_from_center)
            
            col_left = center_col - crop_state['vertical_line']
            col_right = center_col + crop_state['vertical_line']
            
            vline_left.set_xdata([col_left, col_left])
            vline_right.set_xdata([col_right, col_right])
            vline_left.set_visible(True)
            vline_right.set_visible(True)
            
            width_crop = 2 * crop_state['vertical_line']
            info_text.set_text(f"Vertical: ±{crop_state['vertical_line']} px from center\n"
                             f"Width: {width_crop} px\n"
                             f"Horizontal: {'SELECTED' if crop_state['horizontal_line'] else 'Not selected'}")
        
        # Botão direito (3) → linha HORIZONTAL
        elif event.button == 3:
            # Clique HORIZONTAL → define altura (do topo até a linha)
            crop_state['horizontal_line'] = int(y_click)
            
            hline.set_ydata([crop_state['horizontal_line'], crop_state['horizontal_line']])
            hline.set_visible(True)
            
            height_crop = crop_state['horizontal_line']
            info_text.set_text(f"Horizontal: {crop_state['horizontal_line']} px from top\n"
                             f"Height: {height_crop} px\n"
                             f"Vertical: {'SELECTED' if crop_state['vertical_line'] else 'Not selected'}")
        
        update_preview()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_button, 'Confirm Crop')
    
    def confirm_crop(event):
        plt.close(fig)
    
    btn_confirm.on_clicked(confirm_crop)
    plt.show(block=True)
    
    # Valida e retorna parâmetros de crop
    if crop_state['vertical_line'] is None or crop_state['horizontal_line'] is None:
        print("    WARNING: Crop not fully defined. Using full image.")
        return None
    
    dist = crop_state['vertical_line']
    row_end = crop_state['horizontal_line']
    
    crop_params = {
        'row_start': 0,
        'row_end': row_end,
        'col_start': max(0, center_col - dist),
        'col_end': min(width, center_col + dist),
        'original_center_col': center_col,
        'original_height': height,
        'original_width': width
    }
    
    print(f"    ✓ Crop defined: H={crop_params['row_end']-crop_params['row_start']} x W={crop_params['col_end']-crop_params['col_start']} px")
    print(f"      Rows: {crop_params['row_start']} → {crop_params['row_end']}")
    print(f"      Cols: {crop_params['col_start']} → {crop_params['col_end']}")
    
    return crop_params


def apply_crop_to_projections(projections, crop_params):
    """
    Aplica crop a todas as projeções RAW (ANTES da normalização para reduzir processamento).
    
    Args:
        projections: array (H, W, n_angles) - RAW projections
        crop_params: dict retornado por select_crop_region()
    
    Returns:
        cropped_projections: array (H_new, W_new, n_angles)
    """
    if crop_params is None:
        print("--> [5/5] No crop applied (using full projections)")
        return projections
    
    print("--> [5/5] Applying crop to all projections...")
    
    rs = crop_params['row_start']
    re = crop_params['row_end']
    cs = crop_params['col_start']
    ce = crop_params['col_end']
    
    cropped = projections[rs:re, cs:ce, :]
    
    print(f"    Original shape: {projections.shape}")
    print(f"    Cropped shape:  {cropped.shape}")
    print(f"    Memory reduction: {projections.nbytes / 1e6:.1f} MB → {cropped.nbytes / 1e6:.1f} MB")
    print(f"    Reduction factor: {projections.nbytes / cropped.nbytes:.2f}x")
    
    return cropped
