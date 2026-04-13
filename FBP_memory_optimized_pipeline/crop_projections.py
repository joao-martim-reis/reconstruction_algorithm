import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def _as_2d_projection(first_projection):
    """
    Normalize loader output to a 2D projection.

    Supported inputs:
    - (H, W)
    - (H, W, 1)
    - (1, H, W)
    """
    if first_projection.ndim == 2:
        return first_projection

    if first_projection.ndim == 3 and first_projection.shape[2] == 1:
        return first_projection[:, :, 0]

    if first_projection.ndim == 3 and first_projection.shape[0] == 1:
        return first_projection[0, :, :]

    return np.squeeze(first_projection)


def select_crop_region(first_projection):
    """
    Interactive interface to select crop region on RAW projections (before normalization).
    - LEFT CLICK: Define vertical line -> crop width (symmetric around center)
    - RIGHT CLICK (1st): Define TOP horizontal line
    - RIGHT CLICK (2nd): Define BOTTOM horizontal line
    """
    print("--> Select crop region on first projection...")
    print("    1. LEFT CLICK -> define VERTICAL line (symmetric crop around center)")
    print("    2. RIGHT CLICK (1st) -> TOP horizontal line")
    print("    3. RIGHT CLICK (2nd) -> BOTTOM horizontal line")

    height, width = first_projection.shape
    center_col = width // 2

    crop_state = {
        "vertical_line": None,
        "horizontal_line_top": None,
        "horizontal_line_bottom": None,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.15, wspace=0.3)

    ax1.imshow(first_projection, cmap="gray", aspect="auto", origin="upper")
    ax1.set_title("Original Projection\n(Click to define crop)", fontsize=12)
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)

    im2 = ax2.imshow(first_projection, cmap="gray", aspect="auto", origin="upper")
    ax2.set_title("Crop Preview", fontsize=12)
    ax2.set_xlabel("Width (px)")
    ax2.set_ylabel("Height (px)")

    vline_left = ax1.axvline(x=0, color="cyan", linestyle="--", linewidth=2, visible=False)
    vline_right = ax1.axvline(x=width, color="cyan", linestyle="--", linewidth=2, visible=False)
    hline_top = ax1.axhline(y=0, color="yellow", linestyle="--", linewidth=2, visible=False)
    hline_bottom = ax1.axhline(y=height, color="orange", linestyle="--", linewidth=2, visible=False)
    ax1.axvline(x=center_col, color="red", linestyle=":", linewidth=1, alpha=0.5)

    info_text = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def update_preview():
        if (
            crop_state["vertical_line"] is not None
            and crop_state["horizontal_line_top"] is not None
            and crop_state["horizontal_line_bottom"] is not None
        ):
            dist = crop_state["vertical_line"]
            row_start = crop_state["horizontal_line_top"]
            row_end = crop_state["horizontal_line_bottom"]

            col_start = max(0, center_col - dist)
            col_end = min(width, center_col + dist)

            cropped = first_projection[row_start:row_end, col_start:col_end]
            im2.set_data(cropped)
            im2.set_extent([col_start, col_end, row_end, row_start])
            ax2.set_xlim(col_start, col_end)
            ax2.set_ylim(row_end, row_start)
            ax2.set_title(f"Crop Preview: {row_end - row_start}H x {col_end - col_start}W px", fontsize=12)
        else:
            im2.set_data(first_projection)
            im2.set_extent([0, width, height, 0])
            ax2.set_xlim(0, width)
            ax2.set_ylim(height, 0)
            ax2.set_title("Crop Preview (select all lines)", fontsize=12)

        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax1:
            return

        x_click, y_click = event.xdata, event.ydata

        if event.button == 1:
            dist_from_center = abs(x_click - center_col)
            crop_state["vertical_line"] = int(dist_from_center)

            col_left = center_col - crop_state["vertical_line"]
            col_right = center_col + crop_state["vertical_line"]

            vline_left.set_xdata([col_left, col_left])
            vline_right.set_xdata([col_right, col_right])
            vline_left.set_visible(True)
            vline_right.set_visible(True)

            width_crop = 2 * crop_state["vertical_line"]
            h_top = crop_state["horizontal_line_top"]
            h_bottom = crop_state["horizontal_line_bottom"]
            h_status = f"Top: {h_top if h_top else 'Not set'}, Bottom: {h_bottom if h_bottom else 'Not set'}"
            info_text.set_text(
                f"Vertical: +/-{crop_state['vertical_line']} px from center\n"
                f"Width: {width_crop} px\n"
                f"Horizontal: {h_status}"
            )

        elif event.button == 3:
            y_click_int = int(y_click)

            if crop_state["horizontal_line_top"] is None:
                crop_state["horizontal_line_top"] = y_click_int
                hline_top.set_ydata([y_click_int, y_click_int])
                hline_top.set_visible(True)
                info_text.set_text(
                    f"Top line: {y_click_int} px\n"
                    f"Bottom line: Not set\n"
                    f"Vertical: {'SELECTED' if crop_state['vertical_line'] else 'Not selected'}"
                )
            elif crop_state["horizontal_line_bottom"] is None:
                crop_state["horizontal_line_bottom"] = y_click_int
                hline_bottom.set_ydata([y_click_int, y_click_int])
                hline_bottom.set_visible(True)

                if crop_state["horizontal_line_top"] > crop_state["horizontal_line_bottom"]:
                    crop_state["horizontal_line_top"], crop_state["horizontal_line_bottom"] = (
                        crop_state["horizontal_line_bottom"],
                        crop_state["horizontal_line_top"],
                    )
                    hline_top.set_ydata([
                        crop_state["horizontal_line_top"],
                        crop_state["horizontal_line_top"],
                    ])
                    hline_bottom.set_ydata([
                        crop_state["horizontal_line_bottom"],
                        crop_state["horizontal_line_bottom"],
                    ])

                height_crop = crop_state["horizontal_line_bottom"] - crop_state["horizontal_line_top"]
                info_text.set_text(
                    f"Top: {crop_state['horizontal_line_top']} px, "
                    f"Bottom: {crop_state['horizontal_line_bottom']} px\n"
                    f"Height: {height_crop} px\n"
                    f"Vertical: {'SELECTED' if crop_state['vertical_line'] else 'Not selected'}"
                )
            else:
                crop_state["horizontal_line_top"] = y_click_int
                crop_state["horizontal_line_bottom"] = None
                hline_top.set_ydata([y_click_int, y_click_int])
                hline_bottom.set_visible(False)
                info_text.set_text(
                    f"Top line: {y_click_int} px (reset - click again for bottom)\n"
                    f"Bottom line: Not set\n"
                    f"Vertical: {'SELECTED' if crop_state['vertical_line'] else 'Not selected'}"
                )

        update_preview()

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_button = plt.axes([0.4, 0.02, 0.2, 0.06])
    btn_confirm = Button(ax_button, "Confirm Crop")

    def confirm_crop(_event):
        plt.close(fig)

    btn_confirm.on_clicked(confirm_crop)
    plt.show(block=True)

    if (
        crop_state["vertical_line"] is None
        or crop_state["horizontal_line_top"] is None
        or crop_state["horizontal_line_bottom"] is None
    ):
        print("    WARNING: Crop not fully defined. Using full image.")
        return None

    dist = crop_state["vertical_line"]
    row_start = crop_state["horizontal_line_top"]
    row_end = crop_state["horizontal_line_bottom"]

    crop_params = {
        "row_start": row_start,
        "row_end": row_end,
        "col_start": max(0, center_col - dist),
        "col_end": min(width, center_col + dist),
        "original_center_col": center_col,
        "original_height": height,
        "original_width": width,
    }

    print("    ✓ Crop defined")
    return crop_params


def apply_crop_to_projections(projections, crop_params):
    """
    Apply crop to all RAW projections (BEFORE normalization to reduce processing).
    """
    if crop_params is None:
        print("--> No crop applied (using full projections)")
        return projections

    rs = crop_params["row_start"]
    re = crop_params["row_end"]
    cs = crop_params["col_start"]
    ce = crop_params["col_end"]

    cropped = projections[rs:re, cs:ce, :]
    return cropped


def select_crop_and_ranges(file_paths, load_projection_batch, full_h, full_w):
    """
    Run interactive crop selection and return crop ranges and dimensions.

    Returns:
    - crop_params
    - (row_start, row_end, col_start, col_end)
    - cropped_h
    - cropped_w
    """
    first_projection = load_projection_batch(file_paths, 0, 1, dtype=np.float32)
    first_projection_2d = _as_2d_projection(first_projection)
    crop_params = select_crop_region(first_projection_2d)

    if crop_params is None:
        return None, (0, full_h, 0, full_w), full_h, full_w

    row_start = int(crop_params["row_start"])
    row_end = int(crop_params["row_end"])
    col_start = int(crop_params["col_start"])
    col_end = int(crop_params["col_end"])

    cropped_h = row_end - row_start
    cropped_w = col_end - col_start

    return crop_params, (row_start, row_end, col_start, col_end), cropped_h, cropped_w
