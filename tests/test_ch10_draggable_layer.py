"""Unit tests for Chapter 10.7: Draggable Layer Flags (osbook_day10g)."""

import json
import subprocess
from pathlib import Path

import numpy as np

from common import (
    DoubleBufferLayerSpec,
    DraggableLayerSim,
    generate_ch10_19_layer_draggable_flag_architecture,
    generate_ch10_20_occluding_non_draggable_click_through_analysis,
    generate_ch10_21_draggable_layer_desktop_simulation,
)


def test_fail_safe_default_and_fluent_set_draggable():
    """Verify whitelist default draggable_=False and fluent SetDraggable/IsDraggable."""
    sim = DraggableLayerSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(180, 72))

    # Only HelloWindow (ID=3) is draggable; DesktopBG (1), Console (2), Mouse (4) are False
    assert sim.is_draggable(1) is False
    assert sim.is_draggable(2) is False
    assert sim.is_draggable(3) is True
    assert sim.is_draggable(4) is False

    # Adding a brand-new layer without explicit draggable defaults to False
    sim.add_layer(
        DoubleBufferLayerSpec(
            5, "StatusPanel", 10, 170, 120, 24, np.zeros((24, 120, 3), dtype=np.uint8)
        ),
        height=3,
    )
    assert sim.is_draggable(5) is False

    # Fluent chaining returns self and toggles state
    ret = sim.set_draggable(5, True)
    assert ret is sim
    assert sim.is_draggable(5) is True
    sim.set_draggable(5, False)
    assert sim.is_draggable(5) is False


def test_desktop_bg_and_console_drag_rejection():
    """Verify osbook_day10g fixes osbook_day10f bug: BG and Console cannot be dragged."""
    sim = DraggableLayerSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(250, 140))

    # 1. Click empty DesktopBG (250, 140) and try dragging (-40, -28)
    e_bg_press = sim.mouse_observer(0x01, 0, 0)
    assert e_bg_press.hit_layer_id == 1
    assert e_bg_press.hit_layer_is_draggable is False
    assert e_bg_press.drag_accepted is False
    assert sim.mouse_drag_layer_id == 0

    e_bg_hold = sim.mouse_observer(0x01, -40, -28)
    assert e_bg_hold.transition == "HOLD (No Target)"
    assert sim.mouse_position == (210, 112)
    assert (sim.db_sim.layers[1].x, sim.db_sim.layers[1].y) == (0, 0)
    sim.mouse_observer(0x00, 0, 0)

    # 2. Move to Console (50, 50) and try dragging (+25, +15)
    sim.mouse_observer(0x00, -100, -62)
    sim.mouse_observer(0x00, -60, 0)
    assert sim.mouse_position == (50, 50)

    e_con_press = sim.mouse_observer(0x01, 0, 0)
    assert e_con_press.hit_layer_id == 2
    assert e_con_press.hit_layer_is_draggable is False
    assert e_con_press.drag_accepted is False
    assert sim.mouse_drag_layer_id == 0

    sim.mouse_observer(0x01, 25, 15)
    assert (sim.db_sim.layers[2].x, sim.db_sim.layers[2].y) == (0, 0)
    sim.mouse_observer(0x00, 0, 0)

    assert np.array_equal(sim.vram, sim.back_buffer)


def test_hello_window_dragging_and_edge_clamping():
    """Verify draggable HelloWindow still drags smoothly with clamped posdiff edge sync."""
    sim = DraggableLayerSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(180, 66))

    e_press = sim.mouse_observer(0x01, 0, 0)
    assert e_press.hit_layer_id == 3
    assert e_press.hit_layer_is_draggable is True
    assert e_press.drag_accepted is True
    assert sim.mouse_drag_layer_id == 3

    e_drag1 = sim.mouse_observer(0x01, 70, 50)
    assert e_drag1.posdiff == (70, 50)
    assert (sim.db_sim.layers[3].x, sim.db_sim.layers[3].y) == (200, 106)

    # Edge clamp: 250 + 100 -> clamped to 319 -> posdiff = (69, 0)
    e_drag2 = sim.mouse_observer(0x01, 100, 0)
    assert sim.mouse_position == (319, 116)
    assert e_drag2.posdiff == (69, 0)
    assert (sim.db_sim.layers[3].x, sim.db_sim.layers[3].y) == (269, 106)

    e_rel = sim.mouse_observer(0x00, -10, 0)
    assert e_rel.transition == "FALLING_EDGE (Drag End)"
    assert sim.mouse_drag_layer_id == 0
    assert np.array_equal(sim.vram, sim.back_buffer)


def test_srp_occlusion_prevents_click_through_bug():
    """Verify non-draggable top layer occludes lower draggable window (SRP design)."""
    sim = DraggableLayerSim.create_mikanos_desktop_sim(320, 200, init_mouse_pos=(140, 70))
    # Raise Console (ID=2, draggable=False) above HelloWindow (ID=3, draggable=True)
    sim.up_down(2, 2)  # Stack: 1(BG) < 3(HelloWindow) < 2(Console) < 4(Mouse)

    # MikanOS actual design: returns topmost visual layer (Console ID=2)
    hit_mikan, _ = sim.find_layer_by_position((140, 70), exclude_id=4, naive_filter_draggable=False)
    assert hit_mikan == 2

    # MouseObserver rejects drag without clicking through to HelloWindow
    e = sim.mouse_observer(0x01, 0, 0)
    assert e.drag_accepted is False
    assert sim.mouse_drag_layer_id == 0

    # Anti-pattern: filtering draggable inside FindLayerByPosition causes click-through to ID=3
    hit_bug, _ = sim.find_layer_by_position((140, 70), exclude_id=4, naive_filter_draggable=True)
    assert hit_bug == 3

    # Out-of-bounds click safely handles nullptr without error
    assert sim.find_layer_by_position((-1, -1), exclude_id=4)[0] is None


def test_figure_generation_outputs():
    """Verify all three Chapter 10.7 high-DPI PNG figures are generated cleanly."""
    p1 = generate_ch10_19_layer_draggable_flag_architecture()
    p2 = generate_ch10_20_occluding_non_draggable_click_through_analysis()
    p3 = generate_ch10_21_draggable_layer_desktop_simulation()

    for p in [p1, p2, p3]:
        path = Path(p)
        assert path.exists(), f"Missing generated figure: {path}"
        assert path.stat().st_size > 50_000, f"Figure too small: {path}"


def test_cpp_executable_execution():
    """Verify C++20 standalone executable builds and passes all assertions."""
    exe_path = Path("build/bin/ch10_draggable_layer")
    if not exe_path.exists():
        subprocess.run(
            ["cmake", "--build", "build", "--target", "ch10_draggable_layer"], check=True
        )

    result = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All C++20 Chapter 10.7 Assertions PASSED Successfully!" in result.stdout
    assert "SRP Occlusion vs Click-Through bug verification: PASS" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 10.7 notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("10/10.7_Draggable_Layer_Flags.ipynb")
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
