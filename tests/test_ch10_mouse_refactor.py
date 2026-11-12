"""Tests for Chapter 10.1: Refactoring Mouse Processing & Window Geometric Clipping (osbook_day10a)."""

import subprocess
from pathlib import Path
from PIL import Image
import pytest

from common import (
    MousePositionClampingSim,
    WindowDrawClippingSim,
    plot_fig10_01_mouse_clamping_and_screen_bounds,
    plot_fig10_02_window_drawto_intersection_clipping_math,
    plot_fig10_03_mouse_cursor_edge_rendering_simulation,
)


def test_mouse_position_clamping_basic():
    """Verify basic mouse coordinate clamping across screen boundaries."""
    sim = MousePositionClampingSim(screen_width=640, screen_height=400, pos_x=200, pos_y=200)

    # Initial position
    assert sim.pos_x == 200
    assert sim.pos_y == 200

    # Normal displacement
    cx, cy = sim.move(15, -25)
    assert cx == 215
    assert cy == 175

    # Clamp at left edge (X < 0)
    cx, cy = sim.move(-300, 0)
    assert cx == 0
    assert cy == 175

    # Clamp at top edge (Y < 0)
    cx, cy = sim.move(0, -300)
    assert cx == 0
    assert cy == 0

    # Clamp at right edge (X >= 640 -> 639)
    cx, cy = sim.move(1000, 0)
    assert cx == 639
    assert cy == 0

    # Clamp at bottom edge (Y >= 400 -> 399)
    cx, cy = sim.move(0, 1000)
    assert cx == 639
    assert cy == 399


def test_mouse_clamping_history():
    """Verify that clamping state transitions and records are correctly logged."""
    sim = MousePositionClampingSim(screen_width=640, screen_height=400, pos_x=10, pos_y=10)
    sim.move(-50, -50)
    assert len(sim.history) == 2
    last_record = sim.history[-1]
    assert last_record["raw_x"] == -40
    assert last_record["raw_y"] == -40
    assert last_record["clamped_x"] == 0
    assert last_record["clamped_y"] == 0
    assert last_record["was_clamped"] is True


def test_window_draw_clipping_math_interior():
    """Verify clipping bounds when window is completely inside the screen."""
    clipper = WindowDrawClippingSim(window_width=20, window_height=20, screen_width=100, screen_height=100)
    bounds = clipper.compute_clipping_bounds(pos_x=10, pos_y=10)

    assert bounds["is_visible"] is True
    assert bounds["x_range"] == (0, 20)
    assert bounds["y_range"] == (0, 20)
    assert bounds["drawn_pixels"] == 400
    assert bounds["clipped_pixels"] == 0
    assert bounds["screen_rect"] == (10, 10, 30, 30)


def test_window_draw_clipping_math_negative_pos():
    """Verify clipping bounds when window hangs off top-left (pos < 0)."""
    clipper = WindowDrawClippingSim(window_width=20, window_height=20, screen_width=100, screen_height=100)
    # pos_x = -10, pos_y = -5
    # y_start = max(0, 5) = 5, y_end = min(20, 100 - (-5)) = 20 -> y in [5, 20)
    # x_start = max(0, 10) = 10, x_end = min(20, 100 - (-10)) = 20 -> x in [10, 20)
    bounds = clipper.compute_clipping_bounds(pos_x=-10, pos_y=-5)

    assert bounds["is_visible"] is True
    assert bounds["x_range"] == (10, 20)
    assert bounds["y_range"] == (5, 20)
    assert bounds["visible_width"] == 10
    assert bounds["visible_height"] == 15
    assert bounds["drawn_pixels"] == 150
    assert bounds["clipped_pixels"] == 250
    assert bounds["screen_rect"] == (0, 0, 10, 15)


def test_window_draw_clipping_math_positive_overflow():
    """Verify clipping bounds when window hangs off bottom-right (pos + size > screen)."""
    clipper = WindowDrawClippingSim(window_width=20, window_height=20, screen_width=100, screen_height=100)
    # pos_x = 90, pos_y = 85
    # x_start = 0, x_end = min(20, 100 - 90) = 10 -> x in [0, 10)
    # y_start = 0, y_end = min(20, 100 - 85) = 15 -> y in [0, 15)
    bounds = clipper.compute_clipping_bounds(pos_x=90, pos_y=85)

    assert bounds["is_visible"] is True
    assert bounds["x_range"] == (0, 10)
    assert bounds["y_range"] == (0, 15)
    assert bounds["visible_width"] == 10
    assert bounds["visible_height"] == 15
    assert bounds["drawn_pixels"] == 150
    assert bounds["clipped_pixels"] == 250
    assert bounds["screen_rect"] == (90, 85, 100, 100)


def test_window_draw_clipping_completely_outside():
    """Verify clipping behavior when window is entirely outside the screen boundaries."""
    clipper = WindowDrawClippingSim(window_width=20, window_height=20, screen_width=100, screen_height=100)

    # Completely to the left
    bounds = clipper.compute_clipping_bounds(pos_x=-30, pos_y=10)
    assert bounds["is_visible"] is False
    assert bounds["drawn_pixels"] == 0
    assert bounds["clipped_pixels"] == 400

    # Completely below
    bounds = clipper.compute_clipping_bounds(pos_x=10, pos_y=150)
    assert bounds["is_visible"] is False
    assert bounds["drawn_pixels"] == 0
    assert bounds["clipped_pixels"] == 400


def test_render_clipped_transparent_and_illegal_access_zero():
    """Verify that rendering simulation performs zero illegal VRAM writes and handles transparent color."""
    clipper = WindowDrawClippingSim(window_width=16, window_height=24, screen_width=640, screen_height=400)

    # Hanging partially off top-left
    res = clipper.render_clipped(pos_x=-6, pos_y=-8, transparent_value=0)
    assert res["illegal_writes"] == 0
    assert res["bounds"]["visible_width"] == 10
    assert res["bounds"]["visible_height"] == 16
    assert res["bounds"]["drawn_pixels"] == 160


def test_cpp_binary_execution():
    """Verify that the C++20 test suite executes and passes cleanly."""
    bin_path = Path("build/bin/ch10_mouse_refactor")
    assert bin_path.exists(), f"Binary {bin_path} not found. Build target ch10_mouse_refactor first."

    proc = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert "ALL CHAPTER 10.1 TESTS AND VERIFICATIONS PASSED SUCCESSFULLY!" in proc.stdout
    assert "Zero illegal accesses" in proc.stdout


def test_ch10_1_generated_figures():
    """Verify that all generated figures for Section 10.1 exist and are valid images."""
    fig1 = Path("10/result/fig10_01_mouse_clamping_and_screen_bounds.png")
    fig2 = Path("10/result/fig10_02_window_drawto_intersection_clipping_math.png")
    fig3 = Path("10/result/fig10_03_mouse_cursor_edge_rendering_simulation.png")

    plot_fig10_01_mouse_clamping_and_screen_bounds(fig1)
    plot_fig10_02_window_drawto_intersection_clipping_math(fig2)
    plot_fig10_03_mouse_cursor_edge_rendering_simulation(fig3)

    for fig_path in [fig1, fig2, fig3]:
        assert fig_path.exists(), f"Figure {fig_path} does not exist"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_path} too small"
        with Image.open(fig_path) as img:
            assert img.format == "PNG"
            assert img.size[0] > 1000
            assert img.size[1] > 600
