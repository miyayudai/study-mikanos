"""Unit tests for Chapter 10.5: Double Buffering & Back Buffer (osbook_day10e)."""

import json
import subprocess
from pathlib import Path

import numpy as np

from common import (
    BackBufferConfigSim,
    DoubleBufferLayerSpec,
    DoubleBufferingLayerSim,
    SimRect,
    VramAtomicityBenchmark,
    generate_ch10_13_single_vs_double_buffering_pipeline,
    generate_ch10_14_layermanager_back_buffer_architecture,
    generate_ch10_15_flicker_elimination_vram_atomicity_analysis,
)


def _make_solid_pixels(w: int, h: int, rgb: tuple[int, int, int]) -> np.ndarray:
    arr = np.empty((h, w, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return arr


def _build_demo_layers() -> tuple[DoubleBufferingLayerSim, tuple[int, int, int]]:
    sim = DoubleBufferingLayerSim(screen_w=320, screen_h=200)
    bg_color = (45, 110, 142)
    con_color = (16, 24, 32)
    win_color = (198, 198, 198)
    mouse_color = (255, 255, 255)
    mouse_trans = (0, 0, 1)

    mouse_pixels = _make_solid_pixels(12, 16, mouse_trans)
    mouse_pixels[2:12, 2:10] = mouse_color

    sim.add_layer(
        DoubleBufferLayerSpec(0, "DesktopBG", 0, 0, 320, 200, _make_solid_pixels(320, 200, bg_color))
    )
    sim.add_layer(
        DoubleBufferLayerSpec(1, "Console", 0, 0, 180, 100, _make_solid_pixels(180, 100, con_color))
    )
    sim.add_layer(
        DoubleBufferLayerSpec(2, "MainWindow", 100, 60, 120, 50, _make_solid_pixels(120, 50, win_color))
    )
    sim.add_layer(
        DoubleBufferLayerSpec(
            3, "MouseCursor", 130, 75, 12, 16, mouse_pixels, transparent_color=mouse_trans
        )
    )
    return sim, mouse_color


def test_back_buffer_config_dual_mode_and_stride_normalization():
    """Verify FrameBuffer::Initialize behavior for external VRAM vs nullptr RAM back buffer."""
    screen_cfg, back_cfg = BackBufferConfigSim.create_back_config_from_screen(
        screen_width=1024, screen_height=768, screen_stride=1088
    )

    assert screen_cfg.mode == "external_vram"
    assert screen_cfg.owns_memory is False
    assert screen_cfg.allocated_ram_bytes == 0
    assert screen_cfg.effective_pixels_per_scan_line == 1088
    assert screen_cfg.scan_line_bytes == 1088 * 4

    assert back_cfg.mode == "internal_back_buffer"
    assert back_cfg.owns_memory is True
    assert back_cfg.allocated_ram_bytes == 1024 * 768 * 4
    # Stride padding must be removed in RAM back_buffer_
    assert back_cfg.effective_pixels_per_scan_line == 1024
    assert back_cfg.scan_line_bytes == 1024 * 4


def test_single_vs_double_buffer_draw_area_atomicity():
    """Verify full/area Draw exposes transient VRAM mismatches in single buffer, 0 in double buffer."""
    sim_single, _ = _build_demo_layers()
    res_single = sim_single.draw_area(SimRect(0, 0, 320, 200), use_double_buffer=False)
    assert res_single["vram_transient_exposures"] > 0
    assert res_single["ram_pixel_writes"] == 0

    sim_double, _ = _build_demo_layers()
    res_double = sim_double.draw_area(SimRect(0, 0, 320, 200), use_double_buffer=True)
    assert res_double["vram_transient_exposures"] == 0
    assert res_double["vram_pixel_writes"] == 320 * 200
    assert np.array_equal(sim_double.vram, sim_double.back_buffer)


def test_draw_layer_eliminates_mouse_overlap_flicker():
    """Verify LayerManager::Draw(id) eliminates mouse cursor flicker when overlapping window."""
    sim_single, mouse_color = _build_demo_layers()
    sim_single.draw_area(SimRect(0, 0, 320, 200), use_double_buffer=True)
    # Single-buffer Draw(main_window) overwrites overlapping mouse in VRAM before mouse redraw
    res_single = sim_single.draw_layer(2, use_double_buffer=False)
    assert res_single["vram_transient_exposures"] > 0

    sim_double, _ = _build_demo_layers()
    sim_double.draw_area(SimRect(0, 0, 320, 200), use_double_buffer=True)
    res_double = sim_double.draw_layer(2, use_double_buffer=True)
    assert res_double["vram_transient_exposures"] == 0
    assert res_double["vram_pixel_writes"] == 120 * 50
    assert tuple(sim_double.vram[78, 134]) == mouse_color
    assert np.array_equal(sim_double.vram, sim_double.back_buffer)


def test_move_layer_atomicity_and_partial_edge_clipping():
    """Verify LayerManager::Move restores old area and draws new area with 0 VRAM flicker."""
    sim_single, _ = _build_demo_layers()
    sim_single.draw_area(SimRect(0, 0, 320, 200), use_double_buffer=True)
    res_move_single = sim_single.move_layer(3, new_x=314, new_y=192, use_double_buffer=False)
    assert res_move_single["vram_transient_exposures"] > 0

    sim_double, mouse_color = _build_demo_layers()
    sim_double.draw_area(SimRect(0, 0, 320, 200), use_double_buffer=True)
    res_move_double = sim_double.move_layer(3, new_x=314, new_y=192, use_double_buffer=True)
    assert res_move_double["vram_transient_exposures"] == 0
    # Old mouse position (134, 78) restored to MainWindow color (198, 198, 198)
    assert tuple(sim_double.vram[78, 134]) == (198, 198, 198)
    # New mouse position clipped safely inside screen edge (317, 195)
    assert tuple(sim_double.vram[195, 317]) == mouse_color
    assert np.array_equal(sim_double.vram, sim_double.back_buffer)


def test_vram_atomicity_benchmark_metrics():
    """Verify quantitative benchmark table across Ch 10.3, 10.4, and 10.5."""
    rows = VramAtomicityBenchmark.get_architecture_comparison()
    assert len(rows) == 3
    assert rows[0]["vram_copy_pixels"] == 1_051_112
    assert rows[1]["vram_copy_pixels"] == 8_680
    assert rows[2]["vram_copy_pixels"] == 8_320
    assert rows[2]["back_buffer_ram_mib"] == 3.0
    assert rows[2]["mouse_overlap_flicker_exposures"] == 0


def test_figure_generation_outputs():
    """Verify all three Chapter 10.5 high-DPI PNG figures are generated cleanly."""
    p1 = generate_ch10_13_single_vs_double_buffering_pipeline()
    p2 = generate_ch10_14_layermanager_back_buffer_architecture()
    p3 = generate_ch10_15_flicker_elimination_vram_atomicity_analysis()

    for p in [p1, p2, p3]:
        path = Path(p)
        assert path.exists(), f"Missing generated figure: {path}"
        assert path.stat().st_size > 50_000, f"Figure too small: {path}"


def test_cpp_executable_execution():
    """Verify C++20 standalone executable builds and passes all assertions."""
    exe_path = Path("build/bin/ch10_double_buffering")
    if not exe_path.exists():
        subprocess.run(
            ["cmake", "--build", "build", "--target", "ch10_double_buffering"], check=True
        )

    result = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All C++20 Chapter 10.5 Assertions PASSED Successfully!" in result.stdout
    assert "Double-Buffer Draw(main_id) VRAM transient flicker exposures: 0" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 10.5 notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("10/10.5_Double_Buffering_and_Shadow_Buffer.ipynb")
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
