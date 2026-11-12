"""Tests for Chapter 9.7: Optimizing Console Scroll with memcpy/memmove (osbook_day09e)."""

import subprocess
from pathlib import Path
from PIL import Image
import pytest

from common import (
    FrameBufferMoveSim,
    ConsoleMemcpyScrollSim,
    ScrollBenchmarkComparisonResult,
    plot_fig9_19_framebuffer_move_overlap_mechanism,
    plot_fig9_20_console_scroll_optimization_architecture,
    plot_fig9_21_scroll_performance_comparison_benchmark,
)


def test_console_scroll_memcpy_constants():
    """Verify geometry, pixel, and byte constants for console scroll optimization."""
    assert ConsoleMemcpyScrollSim.ROWS == 25
    assert ConsoleMemcpyScrollSim.COLUMNS == 80
    assert ConsoleMemcpyScrollSim.CHAR_WIDTH == 8
    assert ConsoleMemcpyScrollSim.CHAR_HEIGHT == 16
    assert ConsoleMemcpyScrollSim.WIDTH == 640
    assert ConsoleMemcpyScrollSim.HEIGHT == 400
    assert ConsoleMemcpyScrollSim.TOTAL_PIXELS == 256000
    assert ConsoleMemcpyScrollSim.SCROLL_SRC_H == 384
    assert ConsoleMemcpyScrollSim.LINE_MEMCPY_BYTES == 2560
    assert ConsoleMemcpyScrollSim.TOTAL_SCROLL_MEMCPY_BYTES == 983040
    assert ConsoleMemcpyScrollSim.BOTTOM_LINE_CLEAR_PIXELS == 10240


def test_framebuffer_move_upward_shift():
    """Verify upward memory block shift with forward scan prevents data corruption."""
    sim = FrameBufferMoveSim(width=100, height=100, bytes_per_pixel=4)

    # Fill each line with its Y coordinate value
    for y in range(100):
        for x in range(100):
            sim.write_pixel(x, y, y)

    # Shift rows 10..49 up to destination y=5
    res = sim.move(dst_x=0, dst_y=5, src_x=0, src_y=10, src_w=100, src_h=40)
    assert res["direction"] == "forward (top-to-bottom)"
    assert res["copied_lines"] == 40
    assert res["bytes_per_line"] == 400
    assert res["total_bytes"] == 16000

    # Verify rows 5..44 now hold original rows 10..49
    for y in range(5, 45):
        val = sim.read_pixel(x=0, y=y)
        assert val == y + 5


def test_framebuffer_move_downward_shift():
    """Verify downward memory block shift with backward scan prevents data corruption."""
    sim = FrameBufferMoveSim(width=100, height=100, bytes_per_pixel=4)

    for y in range(100):
        for x in range(100):
            sim.write_pixel(x, y, y)

    # Shift rows 10..49 down to destination y=20
    res = sim.move(dst_x=0, dst_y=20, src_x=0, src_y=10, src_w=100, src_h=40)
    assert res["direction"] == "backward (bottom-to-top)"
    assert res["copied_lines"] == 40
    assert res["bytes_per_line"] == 400

    # Verify rows 20..59 now hold original rows 10..49
    for y in range(20, 60):
        val = sim.read_pixel(x=0, y=y)
        assert val == y - 10


def test_console_scroll_benchmark_comparison():
    """Verify quantitative speedup and pixel write reduction metrics."""
    res = ConsoleMemcpyScrollSim.benchmark_comparison(num_scrolls=100)
    assert isinstance(res, ScrollBenchmarkComparisonResult)
    assert res.scroll_count == 100
    assert res.speedup_factor > 30.0
    assert res.writes_reduction_ratio > 20.0
    assert res.fast_pixel_writes < res.legacy_pixel_writes
    assert res.fast_memcpy_bytes == 983040 * 100


def test_cpp_binary_execution():
    """Verify that the C++20 test suite executes and passes cleanly."""
    bin_path = Path("build/bin/ch09_optimizing_scroll")
    assert bin_path.exists(), f"Binary {bin_path} not found. Build target ch09_optimizing_scroll first."

    proc = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert "ALL CHAPTER 9.7 TESTS AND BENCHMARKS PASSED SUCCESSFULLY!" in proc.stdout
    assert "Speedup Factor:" in proc.stdout


def test_ch9_7_generated_figures():
    """Verify that all generated figures for Section 9.7 exist and are valid images."""
    fig1 = Path("9/result/fig9_19_framebuffer_move_overlap_mechanism.png")
    fig2 = Path("9/result/fig9_20_console_scroll_optimization_architecture.png")
    fig3 = Path("9/result/fig9_21_scroll_performance_comparison_benchmark.png")

    for fig in [fig1, fig2, fig3]:
        assert fig.exists(), f"Figure {fig} does not exist"
        assert fig.stat().st_size > 50000, f"Figure {fig} is unexpectedly small"
        with Image.open(fig) as img:
            assert img.width > 1000
            assert img.height > 500


def test_notebook_structure_and_error_free_execution():
    """Verify Section 9.7 notebook exists, is valid JSON, and has zero execution errors."""
    import json
    nb_path = Path("9/9.7_Optimizing_Scroll_with_memcpy.ipynb")
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

