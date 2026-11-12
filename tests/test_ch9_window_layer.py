"""Tests for Chapter 9.3: Window, Layer, and LayerManager Architecture with Smart Pointer Ownership."""

from pathlib import Path
from PIL import Image
import pytest

from common import (
    Vector2DSim,
    PixelColorSim,
    WindowSim,
    LayerSim,
    LayerManagerSim,
    SmartPointerType,
    ControlBlockSim,
    SharedPtrSim,
    UniquePtrSim,
    LayerStackTransitionLogger,
    MikanLayerArchitectureSim,
    plot_fig9_07_window_layer_layermanager_architecture,
    plot_fig9_08_smart_pointer_ownership_and_control_block,
    plot_fig9_09_layer_updown_stack_manipulation_and_compositing,
)


def test_control_block_sim_lifecycle():
    """Verify std::shared_ptr Control Block reference counting and destruction."""
    cb = ControlBlockSim(managed_id="Window_01")
    assert cb.strong_count == 1
    assert cb.weak_count == 0
    assert cb.is_alive is True

    # Strong count increment
    assert cb.increment_strong() == 2
    assert cb.strong_count == 2
    assert cb.is_alive is True

    # Weak count
    assert cb.increment_weak() == 1
    assert cb.decrement_weak() == 0

    # Decrement strong count to 1
    assert cb.decrement_strong() == 1
    assert cb.is_alive is True

    # Decrement strong count to 0 -> object destroyed
    assert cb.decrement_strong() == 0
    assert cb.is_alive is False

    # Cannot increment strong on dead object
    with pytest.raises(RuntimeError, match="Cannot increment strong count on dead object"):
        cb.increment_strong()


def test_shared_ptr_sim_cloning_and_reset():
    """Verify SharedPtrSim cloning, use_count tracking, and reset behavior."""
    win = WindowSim(20, 20)
    sp1 = SharedPtrSim(win, name="sp1")
    assert sp1.use_count() == 1
    assert sp1.is_valid() is True
    assert sp1.get() is win

    # Copy shared_ptr (cloning)
    sp2 = sp1.clone(new_name="sp2")
    assert sp1.use_count() == 2
    assert sp2.use_count() == 2
    assert sp2.get() is win

    sp3 = sp2.clone(new_name="sp3")
    assert sp1.use_count() == 3

    # Reset sp2
    sp2.reset()
    assert sp2.get() is None
    assert sp1.use_count() == 2
    assert sp3.use_count() == 2

    # Reset remaining pointers
    sp1.reset()
    assert sp3.use_count() == 1
    sp3.reset()
    assert sp3.get() is None


def test_unique_ptr_sim_move_semantics():
    """Verify UniquePtrSim exclusive ownership and move semantics."""
    layer = LayerSim(10)
    u1 = UniquePtrSim(layer, name="u1")
    assert u1.is_valid() is True
    assert u1.get() is layer

    # Move ownership to u2
    u2 = u1.move_to(new_name="u2")
    assert u1.is_valid() is False
    assert u1.get() is None
    assert u2.is_valid() is True
    assert u2.get() is layer

    # Attempting to move from u1 again must raise an error
    with pytest.raises(RuntimeError, match="Cannot move from moved/empty unique_ptr"):
        u1.move_to("u3")


def test_window_sim_writer_and_color_keying():
    """Verify Window pixel modification and transparent color skipping."""
    win = WindowSim(8, 8, fill_color=PixelColorSim(100, 100, 100))
    trans_key = PixelColorSim(99, 99, 99)
    win.set_transparent_color(trans_key)

    # Set key color and a foreground pixel
    win.write(0, 0, trans_key)
    win.write(1, 1, PixelColorSim(255, 255, 255))
    assert win.at(0, 0) == trans_key
    assert win.at(1, 1) == PixelColorSim(255, 255, 255)

    target_buf = [[PixelColorSim(0, 0, 0) for _ in range(8)] for _ in range(8)]
    win.draw_to(target_buf, Vector2DSim(0, 0))

    # (0, 0) was key color -> must remain background black (0, 0, 0)
    assert target_buf[0][0] == PixelColorSim(0, 0, 0)
    # (1, 1) was white -> must be white
    assert target_buf[1][1] == PixelColorSim(255, 255, 255)


def test_layer_sim_chaining_and_delegation():
    """Verify LayerSim fluent method chaining and coordinate offset rendering."""
    win = WindowSim(4, 4, fill_color=PixelColorSim(255, 0, 0))
    layer = LayerSim(5)
    layer.set_window(win).move(Vector2DSim(10, 10)).move_relative(Vector2DSim(2, -3))

    assert layer.id == 5
    assert layer.pos.x == 12
    assert layer.pos.y == 7
    assert layer.window is win

    target_buf = [[PixelColorSim(0, 0, 0) for _ in range(20)] for _ in range(20)]
    layer.draw_to(target_buf)
    assert target_buf[7][12] == PixelColorSim(255, 0, 0)
    assert target_buf[0][0] == PixelColorSim(0, 0, 0)


def test_layer_manager_updown_comprehensive():
    """Verify LayerManager UpDown logic: insertion, reordering, clipping, and Hide."""
    mgr = LayerManagerSim(50, 50)
    l1 = mgr.new_layer() # 1
    l2 = mgr.new_layer() # 2
    l3 = mgr.new_layer() # 3

    assert mgr.layer_stack == []

    # Insert l1 at 0 -> [1]
    mgr.up_down(l1.id, 0)
    assert mgr.layer_stack == [1]

    # Insert l2 at 1 -> [1, 2]
    mgr.up_down(l2.id, 1)
    assert mgr.layer_stack == [1, 2]

    # Insert l3 at height 1 -> [1, 3, 2]
    mgr.up_down(l3.id, 1)
    assert mgr.layer_stack == [1, 3, 2]

    # Reorder l1 to top with large height (clipping) -> [3, 2, 1]
    mgr.up_down(l1.id, 999)
    assert mgr.layer_stack == [3, 2, 1]

    # Reorder l2 to bottom (0) -> [2, 3, 1]
    mgr.up_down(l2.id, 0)
    assert mgr.layer_stack == [2, 3, 1]

    # Negative height hides layer -> [3, 1]
    mgr.up_down(l2.id, -1)
    assert mgr.layer_stack == [3, 1]

    # Explicit hide -> [1]
    mgr.hide(l3.id)
    assert mgr.layer_stack == [1]


def test_mikan_layer_architecture_sim_desktop():
    """Verify full MikanOS desktop simulation with background, taskbar, cursor and app window."""
    sim = MikanLayerArchitectureSim(screen_w=60, screen_h=40)
    desktop_info = sim.create_desktop_environment()

    assert desktop_info["bg_layer_id"] == 1
    assert desktop_info["cursor_layer_id"] == 2
    assert desktop_info["stack"] == [1, 2]

    # Background taskbar color check at bottom (x=25 is outside start button)
    assert sim.manager.frame_buffer[35][25] == PixelColorSim(1, 8, 17)
    # Start button color check at (x=5, y=35)
    assert sim.manager.frame_buffer[35][5] == PixelColorSim(80, 80, 80)

    # Add application window
    app_id = sim.add_app_window(20, 15, Vector2DSim(10, 10), title_color=PixelColorSim(180, 180, 180))
    # Stack should be [1 (BG), 3 (App), 2 (Cursor)]
    assert sim.manager.layer_stack == [1, 3, 2]

    # Check title bar rendered at (10, 10)
    assert sim.manager.frame_buffer[10][10] == PixelColorSim(180, 180, 180)


def test_generated_figures_dpi_and_dimensions(tmp_path):
    """Verify figure generation scripts produce valid 300 DPI images."""
    p7 = plot_fig9_07_window_layer_layermanager_architecture(tmp_path / "fig7.png")
    p8 = plot_fig9_08_smart_pointer_ownership_and_control_block(tmp_path / "fig8.png")
    p9 = plot_fig9_09_layer_updown_stack_manipulation_and_compositing(tmp_path / "fig9.png")

    for p in [p7, p8, p9]:
        assert p.exists()
        with Image.open(p) as img:
            assert img.width > 1000
            assert img.height > 600
            assert img.format == "PNG"
