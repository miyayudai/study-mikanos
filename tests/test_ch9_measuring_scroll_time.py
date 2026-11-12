"""Tests for Chapter 9.6: Measuring Console Scroll Time with Local APIC Timer (osbook_day09d)."""

import subprocess
from pathlib import Path
from PIL import Image
import pytest

from common import (
    ConsoleScrollLineResult,
    ConsoleScrollProfilerSim,
    plot_fig9_16_console_scroll_overhead_mechanism,
    plot_fig9_17_scroll_time_measurement_timeline,
    plot_fig9_18_scroll_bottleneck_breakdown_and_comparison,
)


def test_console_scroll_dimensions_and_constants():
    """Verify console dimensions and pixel count constants."""
    assert ConsoleScrollProfilerSim.ROWS == 25
    assert ConsoleScrollProfilerSim.COLUMNS == 80
    assert ConsoleScrollProfilerSim.CHAR_WIDTH == 8
    assert ConsoleScrollProfilerSim.CHAR_HEIGHT == 16
    assert ConsoleScrollProfilerSim.WIDTH == 640
    assert ConsoleScrollProfilerSim.HEIGHT == 400
    assert ConsoleScrollProfilerSim.TOTAL_PIXELS == 256000


def test_console_scroll_normal_lines():
    """Verify that rows 0..23 do not trigger scrolling and incur minimal overhead."""
    sim = ConsoleScrollProfilerSim()
    for row in range(24):
        msg = f"Line {row + 1}: Testing normal output without scroll\n"
        res = sim.put_string(msg)
        assert res.did_scroll is False
        assert res.bg_clear_writes == 0
        assert res.font_redraw_writes == 0
        assert res.total_pixel_writes < 5000
        assert res.elapsed_ticks < 50000


def test_console_scroll_trigger_at_row_25():
    """Verify that reaching the 25th row triggers a full scroll with 256k clear writes."""
    sim = ConsoleScrollProfilerSim()
    # Fill up rows 0..23 (24 lines) with representative log lines
    for row in range(24):
        sim.put_string(f"Line {row + 1:02d}: MikanOS Kernel log message filling buffer\n")

    # 25th line triggers scroll
    scroll_res = sim.put_string("Line 25: Triggering full screen scroll event!\n")
    assert scroll_res.did_scroll is True
    assert scroll_res.bg_clear_writes == ConsoleScrollProfilerSim.TOTAL_PIXELS
    assert scroll_res.font_redraw_writes > 20000
    assert scroll_res.total_pixel_writes > 256000
    assert scroll_res.elapsed_ticks > 1000000


def test_console_scroll_benchmark_suite_metrics():
    """Verify that benchmark suite captures quantitative slowdown of scrolling."""
    benchmark = ConsoleScrollProfilerSim.benchmark_scroll(num_lines=30)
    assert len(benchmark["results"]) == 30
    assert len(benchmark["normal_lines"]) == 24
    assert len(benchmark["scroll_lines"]) == 6

    # Verify significant slowdown
    assert benchmark["slowdown_time"] > 30.0
    assert benchmark["slowdown_writes"] > 100.0

    # Verify background clearing constitutes majority of scroll writes
    first_scroll = benchmark["scroll_lines"][0]
    assert first_scroll.bg_clear_writes == 256000
    assert first_scroll.bg_clear_writes / first_scroll.total_pixel_writes > 0.75


def test_cpp20_ch09_measuring_scroll_time_binary():
    """Execute compiled C++20 standalone verification binary."""
    bin_path = Path("build/bin/ch09_measuring_scroll_time")
    if not bin_path.exists():
        # Fallback to direct build if cmake build dir layout differs
        subprocess.run(["cmake", "--build", "build", "--target", "ch09_measuring_scroll_time"], check=True)

    result = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "MikanOS Chapter 9.6: Measuring Console Scroll Time" in result.stdout
    assert "All Chapter 9.6 C++20 self-verification assertions passed!" in result.stdout
    assert "Slowdown Factor (Time)" in result.stdout
    assert "Slowdown Factor (Write)" in result.stdout


def test_ch9_6_figure_generation_and_validity(tmp_path):
    """Verify generation, resolution, and valid image headers of all 3 figures."""
    f1 = plot_fig9_16_console_scroll_overhead_mechanism(tmp_path / "fig16.png")
    f2 = plot_fig9_17_scroll_time_measurement_timeline(tmp_path / "fig17.png")
    f3 = plot_fig9_18_scroll_bottleneck_breakdown_and_comparison(tmp_path / "fig18.png")

    for p in [f1, f2, f3]:
        assert p.exists()
        assert p.stat().st_size > 100000  # High resolution PNG (> 100 KB)
        with Image.open(p) as img:
            assert img.size[0] >= 3000  # Wide DPI 300 figure
            assert img.size[1] >= 1500


def test_notebook_structure_and_error_free_execution():
    """Verify Section 9.6 notebook exists, is valid JSON, and has zero execution errors."""
    import json
    nb_path = Path("9/9.6_Measuring_Console_Scroll_Time.ipynb")
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
