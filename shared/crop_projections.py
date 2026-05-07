"""Interactive crop region selection and application for raw projection stacks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

_ANGLE_TICK_STEP_DEG = 30


def show_cropped_sinogram(cropped_projections: np.ndarray) -> None:
    """Display the collapsed sinogram after cropping for visual verification.

    Args:
        cropped_projections: Cropped projection stack, shape (H, W, n_angles).
    """
    if cropped_projections.size == 0:
        print("    WARNING: Cropped projections are empty. Sinogram preview skipped.")
        return

    sino_cropped = np.sum(cropped_projections, axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(sino_cropped, cmap='gray', aspect='auto', origin='upper')
    ax.set_title("Collapsed Sinogram After Cropping", fontsize=12)
    ax.set_ylabel("Detector (px)")
    ax.set_xlabel("θ (degree)")

    n_proj = sino_cropped.shape[1]
    ticks_deg = np.arange(0, 361, _ANGLE_TICK_STEP_DEG)
    tick_positions = [int(d * n_proj / 360.0) for d in ticks_deg]
    tick_labels = [f"{int(d)}°" for d in ticks_deg]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.show(block=True)


def select_crop_region(first_projection: np.ndarray) -> dict | None:
    """Interactive interface to define a crop region on the first raw projection.

    Interaction:
        - LEFT CLICK: Define vertical extent — symmetric crop around the image centre.
        - RIGHT CLICK (1st): Define the TOP horizontal boundary.
        - RIGHT CLICK (2nd): Define the BOTTOM horizontal boundary.
        - Confirm button: Accept the selection and close.

    Args:
        first_projection: First projection image, shape (H, W), used as reference.

    Returns:
        crop_params: Dict with keys row_start, row_end, col_start, col_end,
            original_center_col, original_height, original_width.
            Returns None if the crop was not fully defined before confirming.
    """
    print("--> Select crop region on first projection...")
    print("    1. LEFT CLICK  → define VERTICAL line (symmetric crop around centre)")
    print("    2. RIGHT CLICK (1st) → TOP horizontal line")
    print("    3. RIGHT CLICK (2nd) → BOTTOM horizontal line")

    height, width = first_projection.shape
    center_col = width // 2

    crop_state = {
        'vertical_line': None,
        'horizontal_line_top': None,
        'horizontal_line_bottom': None,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.15, wspace=0.3)

    ax1.imshow(first_projection, cmap='gray', aspect='auto', origin='upper')
    ax1.set_title("Original Projection\n(Click to define crop)", fontsize=12)
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)

    im2 = ax2.imshow(first_projection, cmap='gray', aspect='auto', origin='upper')
    ax2.set_title("Crop Preview", fontsize=12)
    ax2.set_xlabel("Width (px)")
    ax2.set_ylabel("Height (px)")

    vline_left = ax1.axvline(x=0, color='cyan', linestyle='--', linewidth=2, visible=False)
    vline_right = ax1.axvline(x=width, color='cyan', linestyle='--', linewidth=2, visible=False)
    hline_top = ax1.axhline(y=0, color='yellow', linestyle='--', linewidth=2, visible=False)
    hline_bottom = ax1.axhline(y=height, color='orange', linestyle='--', linewidth=2, visible=False)
    ax1.axvline(x=center_col, color='red', linestyle=':', linewidth=1, alpha=0.5)

    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update_preview() -> None:
        if all(v is not None for v in crop_state.values()):
            dist = crop_state['vertical_line']
            row_start = crop_state['horizontal_line_top']
            row_end = crop_state['horizontal_line_bottom']
            col_start = max(0, center_col - dist)
            col_end = min(width, center_col + dist)

            cropped = first_projection[row_start:row_end, col_start:col_end]
            im2.set_data(cropped)
            im2.set_extent([col_start, col_end, row_end, row_start])
            ax2.set_xlim(col_start, col_end)
            ax2.set_ylim(row_end, row_start)
            ax2.set_title(f"Crop Preview: {row_end - row_start}H x {col_end - col_start}W px",
                          fontsize=12)
        else:
            im2.set_data(first_projection)
            im2.set_extent([0, width, height, 0])
            ax2.set_xlim(0, width)
            ax2.set_ylim(height, 0)
            ax2.set_title("Crop Preview (select all lines)", fontsize=12)
        fig.canvas.draw()

    def on_click(event) -> None:
        if event.inaxes != ax1:
            return
        x_click, y_click = event.xdata, event.ydata

        if event.button == 1:
            # Symmetric crop around centre column
            dist_from_center = abs(x_click - center_col)
            crop_state['vertical_line'] = int(dist_from_center)

            col_left = center_col - crop_state['vertical_line']
            col_right = center_col + crop_state['vertical_line']
            vline_left.set_xdata([col_left, col_left])
            vline_right.set_xdata([col_right, col_right])
            vline_left.set_visible(True)
            vline_right.set_visible(True)

            h_top = crop_state['horizontal_line_top']
            h_bottom = crop_state['horizontal_line_bottom']
            info_text.set_text(
                f"Vertical: ±{crop_state['vertical_line']} px from centre\n"
                f"Width: {2 * crop_state['vertical_line']} px\n"
                f"Top: {h_top if h_top is not None else 'Not set'}, "
                f"Bottom: {h_bottom if h_bottom is not None else 'Not set'}"
            )

        elif event.button == 3:
            y_int = int(y_click)
            if crop_state['horizontal_line_top'] is None:
                crop_state['horizontal_line_top'] = y_int
                hline_top.set_ydata([y_int, y_int])
                hline_top.set_visible(True)
                info_text.set_text(
                    f"Top: {y_int} px\nBottom: Not set\n"
                    f"Vertical: {'set' if crop_state['vertical_line'] is not None else 'Not set'}"
                )
            elif crop_state['horizontal_line_bottom'] is None:
                crop_state['horizontal_line_bottom'] = y_int
                hline_bottom.set_ydata([y_int, y_int])
                hline_bottom.set_visible(True)
                # Ensure top < bottom
                if crop_state['horizontal_line_top'] > crop_state['horizontal_line_bottom']:
                    crop_state['horizontal_line_top'], crop_state['horizontal_line_bottom'] = (
                        crop_state['horizontal_line_bottom'], crop_state['horizontal_line_top']
                    )
                    hline_top.set_ydata([crop_state['horizontal_line_top']] * 2)
                    hline_bottom.set_ydata([crop_state['horizontal_line_bottom']] * 2)
                h_crop = crop_state['horizontal_line_bottom'] - crop_state['horizontal_line_top']
                info_text.set_text(
                    f"Top: {crop_state['horizontal_line_top']} px, "
                    f"Bottom: {crop_state['horizontal_line_bottom']} px\n"
                    f"Height: {h_crop} px\n"
                    f"Vertical: {'set' if crop_state['vertical_line'] is not None else 'Not set'}"
                )
            else:
                # Reset and start over from top
                crop_state['horizontal_line_top'] = y_int
                crop_state['horizontal_line_bottom'] = None
                hline_top.set_ydata([y_int, y_int])
                hline_bottom.set_visible(False)
                info_text.set_text(
                    f"Top: {y_int} px (reset — click again for bottom)\nBottom: Not set"
                )

        update_preview()

    fig.canvas.mpl_connect('button_press_event', on_click)

    ax_button = plt.axes([0.4, 0.02, 0.2, 0.06])
    btn_confirm = Button(ax_button, 'Confirm Crop')
    btn_confirm.on_clicked(lambda _: plt.close(fig))
    plt.show(block=True)

    if any(v is None for v in crop_state.values()):
        print("    WARNING: Crop not fully defined. Using full image.")
        return None

    dist = crop_state['vertical_line']
    crop_params = {
        'row_start': crop_state['horizontal_line_top'],
        'row_end': crop_state['horizontal_line_bottom'],
        'col_start': max(0, center_col - dist),
        'col_end': min(width, center_col + dist),
        'original_center_col': center_col,
        'original_height': height,
        'original_width': width,
    }
    print(f"    Crop defined: {crop_params['row_end'] - crop_params['row_start']}H "
          f"x {crop_params['col_end'] - crop_params['col_start']}W px")
    return crop_params


def apply_crop_to_projections(
    projections: np.ndarray,
    crop_params: dict | None,
) -> np.ndarray:
    """Apply a crop to all projections in the stack.

    Args:
        projections: Full projection stack, shape (H, W, n_angles).
        crop_params: Dict from select_crop_region, or None to skip cropping.

    Returns:
        cropped: Cropped projection stack, shape (H', W', n_angles).
    """
    if crop_params is None:
        print("--> No crop applied (using full projections)")
        return projections

    print("--> Applying crop to all projections...")
    rs = crop_params['row_start']
    re = crop_params['row_end']
    cs = crop_params['col_start']
    ce = crop_params['col_end']

    cropped = projections[rs:re, cs:ce, :]

    print(f"    Original: {projections.shape}  →  Cropped: {cropped.shape}")
    print(f"    Memory: {projections.nbytes / 1e6:.1f} MB → {cropped.nbytes / 1e6:.1f} MB "
          f"({projections.nbytes / cropped.nbytes:.1f}x reduction)")

    show_cropped_sinogram(cropped)
    return cropped
