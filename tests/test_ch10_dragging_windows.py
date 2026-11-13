"""Unit tests for Chapter 10.6: Window Dragging with Mouse (osbook_day10f)."""

import json
import subprocess
from pathlib import Path

import numpy as np

from common import (
    DoubleBufferLayerSpec,
    HidMouseReport,
    WindowDragSim,
    generate_ch10_16_usb_hid_mouse_report_and_drag_fsm,
    generate_ch10_17_find_layer_by_position_z_order_hit_test,
    generate_ch10_18_window_drag_trajectory_and_bg_bug,
)


def test_hid_mouse_report_decoding_and_button_bitmasks():
    """Verify 3-byte USB HID Boot Protocol report decoding and two's complement dx/dy."""
    r_idle = HidMouseReport.from_bytes(0x00, 0x0A, 0x05)
    assert r_idle.buttons == 0x00
    assert not r_idle.left_pressed
    assert not r_idle.right_pressed
    assert not r_idle.middle_pressed
    assert r_idle.displacement_x == 10
    assert r_idle.displacement_y == 5

    # Left (0x01) + Middle (0x04) = 0x05, dx = -128 (0x80), dy = -20 (0xEC)
    r_neg = HidMouseReport.from_bytes(0x05, 0x80, 0xEC)
    assert r_neg.left_pressed is True
    assert r_neg.right_pressed is False
    assert r_neg.middle_pressed is True
    assert r_neg.displacement_x == -128
    assert r_neg.displacement_y == -20


def test_find_layer_by_position_z_order_exclude_hidden_and_half_open():
    """Verify FindLayerByPosition excludes mouse, skips hidden/windowless, and uses half-open intervals."""
    sim = WindowDragSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(180, 72))

    # Add a windowless layer on top of HelloWindow (ID=3) -> must be skipped
    sim.add_layer(
        DoubleBufferLayerSpec(
            5, "NoWindowLayer", 120, 50, 120, 60, np.zeros((60, 120, 3), dtype=np.uint8)
        ),
        has_window=False,
        height=3,
    )
    # Add a hidden window layer -> must be skipped because removed from layer_stack
    sim.add_layer(
        DoubleBufferLayerSpec(
            6, "HiddenWindow", 120, 50, 120, 60, np.zeros((60, 120, 3), dtype=np.uint8)
        ),
        has_window=True,
        height=-1,
    )
    # Ensure mouse (ID=4) remains at top
    sim.up_down(4, 10)

    # Inside HelloWindow [130, 240) x [56, 104) at mouse position (180, 72)
    hit_win, _ = sim.find_layer_by_position((180, 72), exclude_id=4)
    assert hit_win == 3

    # Overlap with Console [0, 156) x [0, 108) -> HelloWindow (Z=2) wins over Console (Z=1)
    hit_overlap, _ = sim.find_layer_by_position((140, 80), exclude_id=4)
    assert hit_overlap == 3

    # Console-only area (60, 75) -> Console (ID=2)
    hit_con, _ = sim.find_layer_by_position((60, 75), exclude_id=4)
    assert hit_con == 2

    # Half-open boundary checks on HelloWindow [130, 240) x [56, 104)
    assert sim.find_layer_by_position((130, 56), exclude_id=4)[0] == 3
    assert sim.find_layer_by_position((239, 103), exclude_id=4)[0] == 3
    # Right edge x=240 and bottom edge y=104 are exclusive -> fall through to DesktopBG (ID=1)
    assert sim.find_layer_by_position((240, 80), exclude_id=4)[0] == 1
    assert sim.find_layer_by_position((180, 104), exclude_id=4)[0] == 1

    # Outside screen bounds -> None
    assert sim.find_layer_by_position((320, 100), exclude_id=4)[0] is None
    assert sim.find_layer_by_position((-1, 10), exclude_id=4)[0] is None


def test_mouse_observer_fsm_and_clamped_posdiff_sync():
    """Verify 4-state MouseObserver FSM and posdiff clamping synchronization at screen edges."""
    sim = WindowDragSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(180, 72))

    # 1. Idle move: HelloWindow remains at (130, 56)
    e1 = sim.mouse_observer(0x00, 10, 4)
    assert e1.transition == "IDLE (Cursor Move)"
    assert sim.mouse_position == (190, 76)
    assert (sim.db_sim.layers[3].x, sim.db_sim.layers[3].y) == (130, 56)

    # 2. Rising edge: left click selects HelloWindow (ID=3)
    e2 = sim.mouse_observer(0x01, 0, 0)
    assert e2.transition == "RISING_EDGE (Drag Start)"
    assert e2.hit_layer_id == 3
    assert sim.mouse_drag_layer_id == 3

    # 3. Hold: drag right beyond screen boundary (190 + 127 = 317, then +20 clamped to 319)
    e3 = sim.mouse_observer(0x01, 127, 10)
    assert e3.posdiff == (127, 10)
    assert (sim.db_sim.layers[3].x, sim.db_sim.layers[3].y) == (257, 66)

    e4 = sim.mouse_observer(0x01, 20, 0)
    # Clamped mouse_x = 319 -> posdiff.x = 319 - 317 = 2 (NOT raw 20!)
    assert sim.mouse_position == (319, 86)
    assert e4.posdiff == (2, 0)
    assert (sim.db_sim.layers[3].x, sim.db_sim.layers[3].y) == (259, 66)
    # Cursor offset relative to window origin remains constant (319 - 259 == 190 - 130 == 60)!
    assert sim.mouse_position[0] - sim.db_sim.layers[3].x == 60

    # 4. Falling edge: release clears drag layer ID
    e5 = sim.mouse_observer(0x00, -30, -10)
    assert e5.transition == "FALLING_EDGE (Drag End)"
    assert sim.mouse_drag_layer_id == 0
    assert (sim.db_sim.layers[3].x, sim.db_sim.layers[3].y) == (259, 66)

    # VRAM and back_buffer must remain identical
    assert np.array_equal(sim.vram, sim.back_buffer)


def test_osbook_day10f_background_drag_bug_reproduction():
    """Verify osbook_day10f behavior where clicking empty desktop drags DesktopBG (ID=1)."""
    sim = WindowDragSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(260, 150))
    e_press = sim.mouse_observer(0x01, 0, 0)
    assert e_press.hit_layer_id == 1
    assert sim.mouse_drag_layer_id == 1

    sim.mouse_observer(0x01, -40, -28)
    assert (sim.db_sim.layers[1].x, sim.db_sim.layers[1].y) == (-40, -28)


def test_figure_generation_outputs():
    """Verify all three Chapter 10.6 high-DPI PNG figures are generated cleanly."""
    p1 = generate_ch10_16_usb_hid_mouse_report_and_drag_fsm()
    p2 = generate_ch10_17_find_layer_by_position_z_order_hit_test()
    p3 = generate_ch10_18_window_drag_trajectory_and_bg_bug()

    for p in [p1, p2, p3]:
        path = Path(p)
        assert path.exists(), f"Missing generated figure: {path}"
        assert path.stat().st_size > 50_000, f"Figure too small: {path}"


def test_cpp_executable_execution():
    """Verify C++20 standalone executable builds and passes all assertions."""
    exe_path = Path("build/bin/ch10_dragging_windows")
    if not exe_path.exists():
        subprocess.run(
            ["cmake", "--build", "build", "--target", "ch10_dragging_windows"], check=True
        )

    result = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All C++20 Chapter 10.6 Assertions PASSED Successfully!" in result.stdout
    assert "FindLayerByPosition Z-order + half-open [pos, pos+size) checks: PASS" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 10.6 notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("10/10.6_Dragging_Windows_with_Mouse.ipynb")
    assert nb_path.is_file(), f"Notebook {nb_path} does not exist yet"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert len(nb_data["cells"]) >= 12
    for idx, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is not None, f"Cell {idx} has not been executed"
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", (
                    f"Execution error in cell {idx}: {output.get('ename')}: {output.get('evalue')}"
                )
