"""Tests for Chapter 6.1: Mouse Cursor & Desktop GUI Simulation (osbook_day06a)."""

import json
import os
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    MOUSE_CURSOR_WIDTH,
    MOUSE_CURSOR_HEIGHT,
    MOUSE_CURSOR_SHAPE,
    Vector2DSim,
    Vector2D,
    DesktopLayoutConfig,
    MouseCursorStats,
    MouseCursorSimulator,
    plot_fig6_01_vector2d_and_rectangle_rendering,
    plot_fig6_02_desktop_layout_and_mouse_cursor_sprite,
)


def test_vector2d_operations():
    """Verify Vector2DSim arithmetic, equality, and tuple conversion."""
    v1 = Vector2DSim(10, 20)
    v2 = Vector2DSim(5, 15)

    assert v1.x == 10 and v1.y == 20
    assert (v1 + v2) == Vector2DSim(15, 35)
    assert (v1 - v2) == Vector2DSim(5, 5)
    assert v1.as_tuple() == (10, 20)
    assert Vector2D == Vector2DSim


def test_mouse_cursor_shape_dimensions_and_tokens():
    """Verify sprite dimension constants and character token set."""
    assert MOUSE_CURSOR_WIDTH == 15
    assert MOUSE_CURSOR_HEIGHT == 24
    assert len(MOUSE_CURSOR_SHAPE) == 24

    valid_tokens = {'@', '.', ' '}
    for y, row in enumerate(MOUSE_CURSOR_SHAPE):
        assert len(row) == 15, f"Row {y} must have width 15, got {len(row)}"
        for ch in row:
            assert ch in valid_tokens, f"Invalid token '{ch}' in row {y}"


def test_draw_rectangle_pixel_count_and_outline():
    """Verify DrawRectangle paints outline without modifying interior pixels."""
    sim = MouseCursorSimulator(width=100, height=100)
    # Background fill white
    sim.fill_rectangle(Vector2DSim(0, 0), Vector2DSim(100, 100), (255, 255, 255))

    pos = Vector2DSim(10, 20)
    size = Vector2DSim(30, 40)
    border_color = (255, 0, 0)  # Red

    pixels_drawn = sim.draw_rectangle(pos, size, border_color)
    expected_pixels = 2 * (size.x + size.y) - 4
    assert pixels_drawn == expected_pixels

    # Four corners must be border color
    assert sim.get_pixel(pos.x, pos.y) == border_color
    assert sim.get_pixel(pos.x + size.x - 1, pos.y) == border_color
    assert sim.get_pixel(pos.x, pos.y + size.y - 1) == border_color
    assert sim.get_pixel(pos.x + size.x - 1, pos.y + size.y - 1) == border_color

    # Interior pixel must remain untouched (white)
    assert sim.get_pixel(pos.x + 5, pos.y + 5) == (255, 255, 255)


def test_fill_rectangle_solid_fill():
    """Verify FillRectangle paints every pixel in the bounding box."""
    sim = MouseCursorSimulator(width=100, height=100)
    pos = Vector2DSim(5, 10)
    size = Vector2DSim(20, 15)
    fill_color = (0, 128, 255)

    pixels_drawn = sim.fill_rectangle(pos, size, fill_color)
    assert pixels_drawn == size.x * size.y

    # Check boundaries and interior
    for dy in range(size.y):
        for dx in range(size.x):
            assert sim.get_pixel(pos.x + dx, pos.y + dy) == fill_color

    # Adjacent pixel outside must still be black (initial)
    assert sim.get_pixel(pos.x - 1, pos.y) == (0, 0, 0)
    assert sim.get_pixel(pos.x + size.x, pos.y) == (0, 0, 0)


def test_mouse_cursor_sprite_rendering_and_color_keying():
    """Verify sprite pixels (black, white) and transparent color keying."""
    sim = MouseCursorSimulator(width=100, height=100)
    bg_color = (45, 118, 237)  # Desktop blue
    sim.fill_rectangle(Vector2DSim(0, 0), Vector2DSim(100, 100), bg_color)

    pos = Vector2DSim(20, 30)
    stats = sim.draw_mouse_cursor(pos)

    assert stats.total_pixels == MOUSE_CURSOR_WIDTH * MOUSE_CURSOR_HEIGHT
    assert stats.black_pixels > 0
    assert stats.white_pixels > 0
    assert stats.transparent_pixels > 0

    # Tip of arrow at pos is '@' (black)
    assert sim.get_pixel(pos.x, pos.y) == (0, 0, 0)
    # Inside arrow is '.' (white)
    assert sim.get_pixel(pos.x + 2, pos.y + 3) == (255, 255, 255)
    # Top-right corner of 15x24 bounding box is ' ' (transparent) -> must retain desktop background
    assert sim.get_pixel(pos.x + 14, pos.y) == bg_color


def test_desktop_layout_regions_and_colors():
    """Verify complete day06a desktop rendering regions and palette."""
    sim = MouseCursorSimulator(width=800, height=600)
    stats = sim.render_desktop()

    cfg = sim.config
    assert stats.black_pixels > 0
    assert stats.white_pixels > 0

    # Desktop background check (center)
    assert sim.get_pixel(400, 250) == cfg.desktop_bg_color
    # Taskbar check (right bottom)
    assert sim.get_pixel(700, 580) == cfg.taskbar_color
    # Start button check (left bottom)
    assert sim.get_pixel(50, 580) == cfg.start_button_color
    # Start icon border
    assert sim.get_pixel(cfg.start_icon_pos.x, cfg.start_icon_pos.y) == cfg.start_icon_color
    # Start icon interior (remains start button gray)
    assert sim.get_pixel(cfg.start_icon_pos.x + 5, cfg.start_icon_pos.y + 5) == cfg.start_button_color
    # Mouse cursor at (200, 100)
    assert sim.get_pixel(200, 100) == (0, 0, 0)


def test_cpp_standalone_simulation_execution():
    """Verify C++20 standalone simulator builds and passes tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch06" / "mouse_cursor_sim.cpp"
    assert cpp_src.exists()

    bin_path = repo_root / "build" / "bin" / "ch06_mouse_cursor_sim"
    if not bin_path.exists():
        # Compile if not present in build dir
        subprocess.run(
            ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", str(cpp_src), "-o", "/tmp/ch06_mouse_cursor_sim"],
            check=True
        )
        bin_path = Path("/tmp/ch06_mouse_cursor_sim")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert "All Chapter 6.1 mouse cursor & desktop simulation tests PASSED!" in res.stdout


def test_figure_generation_and_validity(tmp_path):
    """Verify generation of Figure 6.01 and 6.02 with high DPI and valid image properties."""
    fig1_path = tmp_path / "fig6_01.png"
    fig2_path = tmp_path / "fig6_02.png"

    plot_fig6_01_vector2d_and_rectangle_rendering(str(fig1_path))
    plot_fig6_02_desktop_layout_and_mouse_cursor_sprite(str(fig2_path))

    assert fig1_path.exists()
    assert fig2_path.exists()
    assert fig1_path.stat().st_size > 100_000
    assert fig2_path.stat().st_size > 100_000

    with Image.open(fig1_path) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 2000

    with Image.open(fig2_path) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 2000


def test_notebook_structure_and_error_free_execution():
    """Verify 6/6.1_Rendering_Mouse_Cursor.ipynb exists, is valid JSON, and has no execution errors."""
    nb_path = Path("6/6.1_Rendering_Mouse_Cursor.ipynb")
    assert nb_path.exists(), "Notebook 6.1 must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10, "Notebook must contain at least 10 cells"

    # Check for execution errors in cell outputs
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", f"Cell {idx} contains an error output: {output}"

