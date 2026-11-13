"""Unit tests for Chapter 10.4: Eliminating Flicker with Partial Draw (osbook_day10d)."""

from pathlib import Path
import subprocess
import pytest

from common import (
    SimRect,
    PartialDrawLayerInfo,
    PartialDrawLayerManagerSim,
    Rectangle3WayIntersectionSim,
    FlickerEliminationScopeAnalyzer,
    generate_ch10_10_layer_partial_draw_mechanism_and_stack,
    generate_ch10_11_rectangle_clipping_and_partial_copy_math,
    generate_ch10_12_flicker_reduction_effect_and_limitations,
)


def test_sim_rect_intersection_math():
    """Verify 2D rectangle intersection operator (&) properties and corner cases."""
    # Standard overlapping rectangles
    r1 = SimRect(10, 10, 50, 40)  # [10, 60) x [10, 50)
    r2 = SimRect(30, 20, 60, 50)  # [30, 90) x [20, 70)
    inter = r1 & r2
    assert inter.x == 30
    assert inter.y == 20
    assert inter.w == 30
    assert inter.h == 30
    assert inter.area == 900
    assert not inter.is_empty

    # Disjoint rectangles
    r_disjoint = SimRect(100, 100, 20, 20)
    inter_empty = r1 & r_disjoint
    assert inter_empty.is_empty
    assert inter_empty.area == 0

    # Containment
    r_inner = SimRect(20, 15, 10, 10)
    inter_inner = r1 & r_inner
    assert inter_inner == r_inner
    assert inter_inner.area == 100


def test_rectangle_3way_intersection_sim():
    """Verify 3-way clipping math: dst_outline & src_outline & src_area_shifted."""
    sim = Rectangle3WayIntersectionSim(
        dst_size=(320, 200),
        src_size=(160, 120),
        dst_pos=(150, 80),
        src_area=(20, 20, 120, 80),
    )
    res = sim.compute()

    assert res["dst_outline"] == (0, 0, 320, 200)
    assert res["src_outline"] == (130, 60, 160, 120)
    assert res["src_area_shifted"] == (150, 80, 120, 80)
    assert res["copy_area"] == (150, 80, 120, 80)
    assert res["src_start_pos"] == (20, 20)
    assert res["pixels_copied"] == 120 * 80


def test_partial_draw_layer_manager_stack_and_full_draw():
    """Verify LayerManager layer stack initialization and full screen redraw metrics."""
    mgr = PartialDrawLayerManagerSim(screen_w=1024, screen_h=768)

    # Add 4 layers
    l0 = PartialDrawLayerInfo(id=0, name="DesktopBG", x=0, y=0, w=1024, h=768)
    l1 = PartialDrawLayerInfo(id=1, name="Console", x=0, y=0, w=640, h=400)
    l2 = PartialDrawLayerInfo(id=2, name="MainWindow", x=300, y=100, w=160, h=52)
    l3 = PartialDrawLayerInfo(id=3, name="MouseCursor", x=350, y=110, w=15, h=24)

    mgr.add_layer(l0)
    mgr.add_layer(l1)
    mgr.add_layer(l2)
    mgr.add_layer(l3)

    assert mgr.layer_stack == [0, 1, 2, 3]

    # Full screen draw
    full_res = mgr.draw_all()
    assert full_res["layers_drawn"] == ["DesktopBG", "Console", "MainWindow", "MouseCursor"]
    assert full_res["layers_skipped"] == []
    # Total pixels = 1024*768 + 640*400 + 160*52 + 15*24 = 1,051,112
    assert full_res["total_pixels"] == (1024 * 768) + (640 * 400) + (160 * 52) + (15 * 24)
    assert full_res["bytes_transferred"] == full_res["total_pixels"] * 4


def test_partial_draw_selective_layer_skip_and_reduction():
    """Verify LayerManager::Draw(id) strictly skips lower layers and achieves >120x reduction."""
    mgr = PartialDrawLayerManagerSim(screen_w=1024, screen_h=768)

    l0 = PartialDrawLayerInfo(id=0, name="DesktopBG", x=0, y=0, w=1024, h=768)
    l1 = PartialDrawLayerInfo(id=1, name="Console", x=0, y=0, w=640, h=400)
    l2 = PartialDrawLayerInfo(id=2, name="MainWindow", x=300, y=100, w=160, h=52)
    l3 = PartialDrawLayerInfo(id=3, name="MouseCursor", x=350, y=110, w=15, h=24)

    mgr.add_layer(l0)
    mgr.add_layer(l1)
    mgr.add_layer(l2)
    mgr.add_layer(l3)

    # Redraw counter window (target_id = 2)
    part_res = mgr.draw_layer(target_id=2)

    assert part_res["layers_skipped"] == ["DesktopBG", "Console"]
    assert part_res["layers_drawn"] == ["MainWindow", "MouseCursor"]
    # Main window area is 160*52 = 8,320 pixels. Mouse intersects with 15*24 = 360 pixels.
    assert part_res["total_pixels"] == (160 * 52) + (15 * 24)

    full_res = mgr.draw_all()
    ratio = full_res["total_pixels"] / part_res["total_pixels"]
    assert ratio > 120.0, f"Expected reduction > 120x, got {ratio:.2f}x"


def test_layer_move_double_draw_behavior():
    """Verify LayerManager::Move restores old rectangle and draws new rectangle."""
    mgr = PartialDrawLayerManagerSim(screen_w=1024, screen_h=768)

    l0 = PartialDrawLayerInfo(id=0, name="DesktopBG", x=0, y=0, w=1024, h=768)
    l1 = PartialDrawLayerInfo(id=1, name="MainWindow", x=300, y=100, w=160, h=52)

    mgr.add_layer(l0)
    mgr.add_layer(l1)

    move_res = mgr.move_layer(layer_id=1, new_x=320, new_y=120)

    assert "old_draw" in move_res
    assert "new_draw" in move_res
    assert move_res["old_draw"]["mode"] == "all"
    assert move_res["new_draw"]["mode"] == "layer"
    assert mgr.find_layer(1).x == 320
    assert mgr.find_layer(1).y == 120


def test_flicker_elimination_scope_table():
    """Verify structured comparison of solved vs. unsolved flicker issues."""
    table = FlickerEliminationScopeAnalyzer.get_comparison_table()
    assert len(table) == 4

    regions = [item["region"] for item in table]
    assert any("デスクトップ背景" in r for r in regions)
    assert any("コンソール" in r for r in regions)
    assert any("カウンタ文字" in r for r in regions)

    counter_item = next(it for it in table if "カウンタ文字" in it["region"])
    assert "未解決" in counter_item["status"]
    assert "FillRectangle" in counter_item["reason"]


def test_figure_generation_outputs():
    """Verify that all three Chapter 10.4 figure generation functions output valid PNGs."""
    p1 = generate_ch10_10_layer_partial_draw_mechanism_and_stack()
    p2 = generate_ch10_11_rectangle_clipping_and_partial_copy_math()
    p3 = generate_ch10_12_flicker_reduction_effect_and_limitations()

    for p in [p1, p2, p3]:
        path = Path(p)
        assert path.exists(), f"Figure file must exist: {path}"
        assert path.stat().st_size > 50_000, f"Figure file must be substantial size (>50KB): {path}"


def test_cpp_executable_execution():
    """Verify that the C++20 standalone executable builds and passes all assertions."""
    exe_path = Path("build/bin/ch10_partial_draw")
    if not exe_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch10_partial_draw"], check=True)

    result = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All C++20 Chapter 10.4 Assertions PASSED Successfully!" in result.stdout
    assert "Pixel reduction ratio exceeds 100x" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 10.4 notebook exists, is valid JSON, and has zero execution errors."""
    import json
    nb_path = Path("10/10.4_Eliminating_Flicker_with_Partial_Draw.ipynb")
    assert nb_path.is_file(), f"Notebook {nb_path} does not exist yet"

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

