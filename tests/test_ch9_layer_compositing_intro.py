"""Tests for Chapter 9.1: Introduction to Layer Compositing."""

from pathlib import Path
from PIL import Image
import pytest

from common import (
    Vector2DSim,
    PixelColorSim,
    WindowSim,
    LayerSim,
    LayerManagerSim,
    BackgroundDestructionComparator,
    plot_fig9_01_background_destruction_vs_layer_compositing,
    plot_fig9_02_layer_compositing_architecture,
    plot_fig9_03_layer_stack_updown_and_compositing_order,
)


def test_vector2d_sim_arithmetic():
    """Verify 2D vector coordinate arithmetic."""
    v1 = Vector2DSim(10, 20)
    v2 = Vector2DSim(5, -15)
    v3 = v1 + v2
    assert v3.x == 15
    assert v3.y == 5

    v4 = v1 - v2
    assert v4.x == 5
    assert v4.y == 35
    assert repr(v1) == "Vector2D(10, 20)"


def test_pixel_color_sim():
    """Verify PixelColorSim representations and equality."""
    c1 = PixelColorSim(45, 118, 237)
    c2 = PixelColorSim(45, 118, 237)
    c3 = PixelColorSim(0, 0, 0)

    assert c1 == c2
    assert c1 != c3
    assert c1.to_rgb_tuple() == (45, 118, 237)
    assert c1.to_hex() == "#2d76ed"
    norm = c1.to_normalized_tuple()
    assert pytest.approx(norm[0], 0.01) == 45 / 255.0


def test_window_sim_clipping_and_transparency():
    """Verify Window buffer clipping and transparent color skipping."""
    win = WindowSim(10, 10, PixelColorSim(255, 0, 0)) # Red
    assert win.width == 10
    assert win.height == 10
    assert win.at(0, 0) == PixelColorSim(255, 0, 0)

    # Modify a pixel
    c_blue = PixelColorSim(0, 0, 255)
    win.write(5, 5, c_blue)
    assert win.at(5, 5) == c_blue

    # Bounds check
    with pytest.raises(IndexError):
        win.at(10, 10)

    # Render to 20x20 target with transparency
    target = [[PixelColorSim(0, 0, 0) for _ in range(20)] for _ in range(20)]
    win.set_transparent_color(PixelColorSim(255, 0, 0)) # Red is transparent
    win.draw_to(target, Vector2DSim(5, 5))

    # At (5+0, 5+0) = (5, 5) was Red (transparent), so it should remain black (0, 0, 0)
    assert target[5][5] == PixelColorSim(0, 0, 0)
    # At (5+5, 5+5) = (10, 10) was Blue, so it should be written
    assert target[10][10] == c_blue


def test_layer_manager_z_order_and_updown():
    """Verify LayerManager layer stack and UpDown z-order reordering."""
    c_bg = PixelColorSim(45, 118, 237)
    c_red = PixelColorSim(255, 0, 0)
    c_green = PixelColorSim(0, 255, 0)

    mgr = LayerManagerSim(50, 50, c_bg)
    w_bottom = WindowSim(20, 20, c_red)
    w_top = WindowSim(20, 20, c_green)

    l1 = mgr.new_layer().set_window(w_bottom).move(Vector2DSim(10, 10))
    l2 = mgr.new_layer().set_window(w_top).move(Vector2DSim(15, 15))

    # Order: l1 at 0, l2 at 1
    mgr.up_down(l1.id, 0)
    mgr.up_down(l2.id, 1)
    assert mgr.layer_stack == [l1.id, l2.id]

    mgr.draw()
    # At overlapping region (15, 15), l2 (green) is on top
    assert mgr.frame_buffer[15][15] == c_green

    # Reorder: move l1 to top (height 1 or higher)
    mgr.up_down(l1.id, 1)
    assert mgr.layer_stack == [l2.id, l1.id]

    mgr.draw()
    # Now l1 (red) is on top at (15, 15)
    assert mgr.frame_buffer[15][15] == c_red

    # Hide l1
    mgr.hide(l1.id)
    assert mgr.layer_stack == [l2.id]
    mgr.draw()
    # At (10, 10) where l1 was, it's now back to c_bg
    assert mgr.frame_buffer[10][10] == c_bg


def test_background_destruction_resolution():
    """Verify background destruction in naive method vs complete resolution in layer compositing."""
    result = BackgroundDestructionComparator.simulate_naive_vs_layer_movement()
    assert result["naive_text_preserved"] is False
    assert result["layer_text_preserved"] is True
    assert result["layer_step1_cursor"] is True
    assert result["cursor_new_pos_correct"] is True


def test_generate_figures(tmp_path: Path):
    """Verify high-DPI figures 9.01, 9.02, 9.03 are generated and valid images."""
    f1 = plot_fig9_01_background_destruction_vs_layer_compositing(tmp_path / "fig9_01.png")
    assert f1.exists()
    assert f1.stat().st_size > 10000
    with Image.open(f1) as img:
        assert img.width > 2000

    f2 = plot_fig9_02_layer_compositing_architecture(tmp_path / "fig9_02.png")
    assert f2.exists()
    assert f2.stat().st_size > 10000
    with Image.open(f2) as img:
        assert img.width > 2000

    f3 = plot_fig9_03_layer_stack_updown_and_compositing_order(tmp_path / "fig9_03.png")
    assert f3.exists()
    assert f3.stat().st_size > 10000
    with Image.open(f3) as img:
        assert img.width > 2000
