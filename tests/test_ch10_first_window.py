"""Unit tests for Chapter 10.2: First GUI Window Decoration & 3D Bevel Math (osbook_day10b)."""

from pathlib import Path
import subprocess
import numpy as np
import pytest

from common import (
    CLOSE_BUTTON_PATTERN,
    COLOR_WHITE,
    COLOR_LIGHT_GRAY,
    COLOR_DARK_GRAY,
    COLOR_BLACK,
    COLOR_TITLE_BAR,
    WindowBevelDecoratorSim,
    plot_fig10_04_window_bevel_and_light_source_math,
    plot_fig10_05_close_button_bitmap_matrix,
    plot_fig10_06_first_window_composition_render,
)


def test_window_bevel_decorator_dimensions_and_buffers():
    """Verify window simulator dimensions, buffer shape, and data types."""
    win = WindowBevelDecoratorSim(width=160, height=68, title="Hello Window")
    assert win.width == 160
    assert win.height == 68
    assert win.buffer.shape == (68, 160, 3)
    assert win.buffer.dtype == np.uint8


def test_window_bevel_border_lines_exact_colors():
    """Verify exact 3D Bevel highlight and shadow border pixel colors."""
    win = WindowBevelDecoratorSim(width=160, height=68)
    w, h = win.width, win.height

    # Outer border: Top (0xc6c6c6), Left (0xc6c6c6), Bottom (0x000000), Right (0x000000)
    assert win.get_pixel(0, 0) == COLOR_LIGHT_GRAY
    assert win.get_pixel(w // 2, 0) == COLOR_LIGHT_GRAY
    assert win.get_pixel(0, h // 2) == COLOR_LIGHT_GRAY

    assert win.get_pixel(w - 1, 0) == COLOR_BLACK
    assert win.get_pixel(w - 1, h // 2) == COLOR_BLACK
    assert win.get_pixel(0, h - 1) == COLOR_BLACK
    assert win.get_pixel(w // 2, h - 1) == COLOR_BLACK
    assert win.get_pixel(w - 1, h - 1) == COLOR_BLACK

    # Inner border: Top (0xffffff), Left (0xffffff), Bottom (0x848484), Right (0x848484)
    assert win.get_pixel(1, 1) == COLOR_WHITE
    assert win.get_pixel(w // 2, 1) == COLOR_WHITE
    assert win.get_pixel(1, h // 2) == COLOR_WHITE

    assert win.get_pixel(w - 2, 1) == COLOR_DARK_GRAY
    assert win.get_pixel(w - 2, h // 2) == COLOR_DARK_GRAY
    assert win.get_pixel(1, h - 2) == COLOR_DARK_GRAY
    assert win.get_pixel(w // 2, h - 2) == COLOR_DARK_GRAY
    assert win.get_pixel(w - 2, h - 2) == COLOR_DARK_GRAY

    # Interior background: (2, 2) is 0xc6c6c6
    assert win.get_pixel(2, 2) == COLOR_LIGHT_GRAY


def test_window_title_bar_region_and_string():
    """Verify title bar filling (0x000084 navy blue) and dimensions."""
    win = WindowBevelDecoratorSim(width=160, height=68, title="Hello Window")

    # Title bar spans (3, 3) to (w-4, 20)
    assert win.get_pixel(3, 3) == COLOR_TITLE_BAR
    assert win.get_pixel(10, 10) == COLOR_TITLE_BAR
    assert win.get_pixel(120, 15) == COLOR_TITLE_BAR

    # Title text at (24, 4) should render white text pixels
    text_region = win.buffer[4:20, 24:120]
    white_mask = (text_region[:, :, 0] == 255) & (text_region[:, :, 1] == 255) & (text_region[:, :, 2] == 255)
    assert np.any(white_mask), "Title text must produce white pixels in title bar"


def test_close_button_bitmap_matrix_and_palette():
    """Verify close button 16x14 pattern, symbol mapping, and placement."""
    assert len(CLOSE_BUTTON_PATTERN) == 14
    for row in CLOSE_BUTTON_PATTERN:
        assert len(row) == 16
        assert all(ch in ".@$:" for ch in row)

    win = WindowBevelDecoratorSim(width=160, height=68)
    btn_x = 160 - 5 - 16  # 139
    btn_y = 5

    # Top-left of close button is '.' -> COLOR_WHITE
    assert win.get_pixel(btn_x, btn_y) == COLOR_WHITE

    # Top-right is '@' -> COLOR_BLACK
    assert win.get_pixel(btn_x + 15, btn_y) == COLOR_BLACK

    # Bottom-left is '@' -> COLOR_BLACK
    assert win.get_pixel(btn_x, btn_y + 13) == COLOR_BLACK

    # Center cross glyph (row 6, cols 7 & 8) is '@' -> COLOR_BLACK
    assert win.get_pixel(btn_x + 7, btn_y + 6) == COLOR_BLACK
    assert win.get_pixel(btn_x + 8, btn_y + 6) == COLOR_BLACK

    # Inner shadow '$' -> COLOR_DARK_GRAY at row 12, col 1
    assert win.get_pixel(btn_x + 1, btn_y + 12) == COLOR_DARK_GRAY


def test_client_area_geometry_and_text_lines():
    """Verify client area computation and text line rendering."""
    win = WindowBevelDecoratorSim(width=160, height=68, lines=["Welcome to", " MikanOS world!"])
    client = win.get_client_area()

    assert client["x"] == 4
    assert client["y"] == 25
    assert client["width"] == 152
    assert client["height"] == 39

    # Verify black text pixels exist in client area (y in [28, 60], x in [24, 150])
    client_region = win.buffer[25:64, 4:156]
    black_mask = (client_region[:, :, 0] == 0) & (client_region[:, :, 1] == 0) & (client_region[:, :, 2] == 0)
    assert np.any(black_mask), "Client area text must produce black pixels"


def test_desktop_composition_and_layer_integrity():
    """Verify window and mouse composition onto full desktop canvas."""
    win = WindowBevelDecoratorSim(width=160, height=68)
    desktop = win.composite_onto_desktop(desktop_width=640, desktop_height=400, win_pos=(300, 100), mouse_pos=(200, 200))

    assert desktop.shape == (400, 640, 3)

    # Window region should have window pixels
    assert tuple(desktop[100 + 1, 300 + 1]) == COLOR_WHITE
    assert tuple(desktop[100 + 10, 300 + 10]) == COLOR_TITLE_BAR

    # Background area untouched
    assert tuple(desktop[50, 50]) == (45, 118, 137)


def test_cpp_binary_execution():
    """Verify C++20 standalone binary builds and passes all assertions."""
    binary = Path("build/bin/ch10_first_window")
    assert binary.exists(), f"Binary {binary} does not exist. Run cmake build."
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert "All Section 10.2 C++20 Verification Tests Passed Successfully!" in result.stdout


def test_ch10_2_generated_figures(tmp_path):
    """Verify high-resolution architectural figures are generated properly."""
    fig1 = tmp_path / "fig1.png"
    fig2 = tmp_path / "fig2.png"
    fig3 = tmp_path / "fig3.png"

    plot_fig10_04_window_bevel_and_light_source_math(fig1)
    plot_fig10_05_close_button_bitmap_matrix(fig2)
    plot_fig10_06_first_window_composition_render(fig3)

    assert fig1.exists() and fig1.stat().st_size > 50000
    assert fig2.exists() and fig2.stat().st_size > 50000
    assert fig3.exists() and fig3.stat().st_size > 50000


def test_notebook_structure_and_error_free_execution():
    """Verify Section 10.2 notebook exists, is valid JSON, and has zero execution errors."""
    import json
    nb_path = Path("10/10.2_Drawing_First_GUI_Window.ipynb")
    assert nb_path.is_file()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert len(nb_data["cells"]) >= 10

    # Ensure no cell threw an execution error
    for idx, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", (
                    f"Execution error in cell {idx}: {output.get('ename')}: {output.get('evalue')}"
                )

