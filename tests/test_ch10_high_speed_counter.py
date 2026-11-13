"""Unit tests for Chapter 10.3: High-Speed Counter & Screen Flicker Problem (osbook_day10c)."""

from pathlib import Path
import subprocess
import numpy as np
import pytest

from common import (
    COLOR_WHITE,
    COLOR_LIGHT_GRAY,
    COLOR_DARK_GRAY,
    COLOR_BLACK,
    COLOR_TITLE_BAR,
    HighSpeedCounterSimulator,
    FlickerTearingSimulator,
    BandwidthWasteAnalyzer,
    generate_ch10_07_counter_window_and_text_region_geometry,
    generate_ch10_08_screen_flicker_and_tearing_hardware_mechanism,
    generate_ch10_09_full_redraw_bandwidth_waste_and_bottleneck,
)


def test_high_speed_counter_window_dimensions_and_buffers():
    """Verify high-speed counter window simulator dimensions, buffer shape, and initial state."""
    sim = HighSpeedCounterSimulator(width=160, height=52, title="Hello Window", count=0)
    assert sim.width == 160
    assert sim.height == 52
    assert sim.buffer.shape == (52, 160, 3)
    assert sim.buffer.dtype == np.uint8

    # 3D Bevel border colors
    assert tuple(sim.buffer[0, 0]) == COLOR_LIGHT_GRAY
    assert tuple(sim.buffer[1, 1]) == COLOR_WHITE
    assert tuple(sim.buffer[51, 159]) == COLOR_BLACK
    assert tuple(sim.buffer[50, 158]) == COLOR_DARK_GRAY
    assert tuple(sim.buffer[3, 3]) == COLOR_TITLE_BAR


def test_counter_update_formatting_and_erase_geometry():
    """Verify counter increment, 10-digit zero padding, erase rect, and text rendering."""
    sim = HighSpeedCounterSimulator(width=160, height=52, count=0)

    # Initial counter: "0000000000"
    res0 = sim.update_counter(0)
    assert res0["formatted"] == "0000000000"
    assert res0["erased_pixels"] == 80 * 16

    # Update to 12345: "0000012345"
    res = sim.update_counter(12345)
    assert res["count"] == 12345
    assert res["formatted"] == "0000012345"
    assert res["erased_pixels"] == 80 * 16
    assert res["font_pixels"] > 0

    # Counter area (x in [24, 104], y in [28, 44]) should contain black font pixels and gray background
    counter_sub = sim.buffer[28:44, 24:104]
    black_pixels = np.all(counter_sub == [0, 0, 0], axis=-1)
    gray_pixels = np.all(counter_sub == [198, 198, 198], axis=-1)

    assert np.any(black_pixels), "Counter must render black digits"
    assert np.any(gray_pixels), "Counter background must be light gray"
    assert np.all(black_pixels | gray_pixels), "Counter subregion must only contain text or background"


def test_flicker_tearing_simulation_math():
    """Verify mathematical properties of display raster scan and flicker exposure."""
    sim = FlickerTearingSimulator(
        screen_height=768,
        refresh_rate_hz=60.0,
        window_rect=(300, 100, 160, 52),
        draw_duration_ms=8.0,
    )
    assert abs(sim.frame_period_ms - 16.666666) < 0.001

    # Scanline position calculations
    assert sim.scanline_y(0.0) == 0.0
    assert abs(sim.scanline_y(sim.frame_period_ms / 2.0) - 384.0) < 0.1

    # Exposure simulation
    exp = sim.simulate_exposure(num_cycles=5, draw_interval_ms=4.0)
    assert "scanline_y" in exp
    assert "exposed_state" in exp
    assert exp["window_y_bounds"] == (100, 152)
    assert 0.0 <= exp["flicker_exposure_ratio"] <= 1.0


def test_bandwidth_waste_math():
    """Verify bandwidth waste calculations comparing full redraw vs. localized diff."""
    analyzer = BandwidthWasteAnalyzer(screen_w=1024, screen_h=768, bpp=4, counter_w=80, counter_h=16)

    assert analyzer.full_redraw_bytes == 1024 * 768 * 4  # 3,145,728 bytes
    assert analyzer.diff_redraw_bytes == 80 * 16 * 4      # 5,120 bytes
    assert abs(analyzer.waste_factor - 614.4) < 0.01

    # At 60 Hz
    full_mb, diff_mb = analyzer.calculate_bandwidth_mb_s(60.0)
    assert abs(full_mb - 180.0) < 0.1
    assert abs(diff_mb - 0.29296875) < 0.01

    # At 1000 Hz
    full_1k, _ = analyzer.calculate_bandwidth_mb_s(1000.0)
    assert abs(full_1k - 3000.0) < 1.0


def test_desktop_composition():
    """Verify compositing the counter window onto a virtual desktop."""
    sim = HighSpeedCounterSimulator(width=160, height=52, count=999)
    desktop = sim.composite_onto_desktop(desktop_w=640, desktop_h=400, win_pos=(300, 100), mouse_pos=(200, 200))

    assert desktop.shape == (400, 640, 3)
    # Background cyan
    assert tuple(desktop[50, 50]) == (0, 128, 128)
    # Window region
    assert tuple(desktop[100 + 1, 300 + 1]) == COLOR_WHITE
    assert tuple(desktop[100 + 3, 300 + 3]) == COLOR_TITLE_BAR


def test_cpp_binary_execution():
    """Verify C++20 standalone binary builds and passes all assertions."""
    binary = Path("build/bin/ch10_high_speed_counter")
    assert binary.exists(), f"Binary {binary} does not exist. Run cmake build."
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert "All C++20 Chapter 10.3 Assertions PASSED Successfully!" in result.stdout


def test_ch10_3_generated_figures(tmp_path):
    """Verify high-resolution architectural figures are generated properly."""
    fig1 = tmp_path / "fig1.png"
    fig2 = tmp_path / "fig2.png"
    fig3 = tmp_path / "fig3.png"

    generate_ch10_07_counter_window_and_text_region_geometry(fig1)
    generate_ch10_08_screen_flicker_and_tearing_hardware_mechanism(fig2)
    generate_ch10_09_full_redraw_bandwidth_waste_and_bottleneck(fig3)

    assert fig1.exists() and fig1.stat().st_size > 50000
    assert fig2.exists() and fig2.stat().st_size > 50000
    assert fig3.exists() and fig3.stat().st_size > 50000


def test_notebook_structure_and_error_free_execution():
    """Verify Section 10.3 notebook exists, is valid JSON, and has zero execution errors."""
    import json
    nb_path = Path("10/10.3_High_Speed_Counter_Flicker_Problem.ipynb")
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
