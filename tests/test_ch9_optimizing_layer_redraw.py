"""Tests for Chapter 9.5: Optimizing Layer Compositing with FrameBuffer & Line-by-Line memcpy."""

import subprocess
from pathlib import Path
from PIL import Image
import pytest

from common import (
    PixelFormatSim,
    bits_per_pixel,
    LayerFrameBufferConfigSim,
    FrameBufferSim,
    OptimizedWindowSim,
    LayerCompositingSpeedupSim,
    plot_fig9_13_framebuffer_shadow_buffer_architecture,
    plot_fig9_14_framebuffer_copy_clipping_and_memcpy_math,
    plot_fig9_15_compositing_speedup_benchmark_comparison,
)


def test_framebuffer_bits_per_pixel():
    """Verify bits per pixel computation for supported formats."""
    assert bits_per_pixel(PixelFormatSim.RGB_RESV_8BIT) == 32
    assert bits_per_pixel(PixelFormatSim.BGR_RESV_8BIT) == 32


def test_framebuffer_sim_initialization():
    """Verify FrameBufferSim initialization for offscreen and shared VRAM buffers."""
    # 1. Off-screen buffer allocation
    fb_off = FrameBufferSim()
    cfg_off = LayerFrameBufferConfigSim(
        horizontal_resolution=640,
        vertical_resolution=480,
        pixel_format=PixelFormatSim.RGB_RESV_8BIT,
    )
    assert fb_off.initialize(cfg_off) is True
    assert fb_off.bytes_per_pixel == 4
    assert len(fb_off.buffer) == 640 * 480 * 4
    assert fb_off.config.pixels_per_scan_line == 640

    # 2. Pre-allocated VRAM buffer
    raw_vram = bytearray(800 * 600 * 4)
    fb_vram = FrameBufferSim()
    cfg_vram = LayerFrameBufferConfigSim(
        frame_buffer=raw_vram,
        pixels_per_scan_line=800,
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=PixelFormatSim.BGR_RESV_8BIT,
    )
    assert fb_vram.initialize(cfg_vram) is True
    assert fb_vram.buffer is raw_vram


def test_framebuffer_sim_pixel_read_write():
    """Verify pixel write/read and RGB packing/unpacking."""
    fb_rgb = FrameBufferSim()
    fb_rgb.initialize(LayerFrameBufferConfigSim(horizontal_resolution=10, vertical_resolution=10, pixel_format=PixelFormatSim.RGB_RESV_8BIT))
    fb_rgb.write_pixel(3, 4, 100, 150, 200)
    assert fb_rgb.read_pixel(3, 4) == (100, 150, 200)

    fb_bgr = FrameBufferSim()
    fb_bgr.initialize(LayerFrameBufferConfigSim(horizontal_resolution=10, vertical_resolution=10, pixel_format=PixelFormatSim.BGR_RESV_8BIT))
    fb_bgr.write_pixel(5, 6, 12, 34, 56)
    assert fb_bgr.read_pixel(5, 6) == (12, 34, 56)


def test_framebuffer_sim_copy_and_clipping():
    """Verify line-by-line copy and 2D clipping math."""
    dst = FrameBufferSim()
    dst.initialize(LayerFrameBufferConfigSim(horizontal_resolution=100, vertical_resolution=100, pixel_format=PixelFormatSim.RGB_RESV_8BIT))

    src = FrameBufferSim()
    src.initialize(LayerFrameBufferConfigSim(horizontal_resolution=40, vertical_resolution=30, pixel_format=PixelFormatSim.RGB_RESV_8BIT))
    for y in range(30):
        for x in range(40):
            src.write_pixel(x, y, 255, 128, 0)

    # Interior copy
    res_a = dst.copy(10, 20, src)
    assert res_a["lines_copied"] == 30
    assert res_a["bytes_copied"] == 30 * 40 * 4
    assert res_a["clipped"] is False
    assert dst.read_pixel(10, 20) == (255, 128, 0)
    assert dst.read_pixel(49, 49) == (255, 128, 0)
    assert dst.read_pixel(9, 19) == (0, 0, 0)

    # Partial clipping right/bottom
    res_b = dst.copy(80, 80, src)
    assert res_b["lines_copied"] == 20
    assert res_b["bytes_copied"] == 20 * 20 * 4
    assert res_b["clipped"] is True
    assert dst.read_pixel(80, 80) == (255, 128, 0)
    assert dst.read_pixel(99, 99) == (255, 128, 0)

    # Partial clipping top/left (negative coordinates)
    res_c = dst.copy(-10, -5, src)
    assert res_c["lines_copied"] == 25
    assert res_c["bytes_copied"] == 25 * 30 * 4
    assert res_c["clipped"] is True
    assert dst.read_pixel(0, 0) == (255, 128, 0)
    assert dst.read_pixel(29, 24) == (255, 128, 0)

    # Completely outside
    res_d = dst.copy(150, 150, src)
    assert res_d["lines_copied"] == 0
    assert res_d["bytes_copied"] == 0
    assert res_d["clipped"] is True


def test_optimized_window_sim_fast_path():
    """Verify OptimizedWindowSim fast-path (memcpy) vs slow-path (transparency)."""
    dst = FrameBufferSim()
    dst.initialize(LayerFrameBufferConfigSim(horizontal_resolution=100, vertical_resolution=100, pixel_format=PixelFormatSim.RGB_RESV_8BIT))

    # 1. Opaque window -> fast-path
    win_opaque = OptimizedWindowSim(50, 50)
    win_opaque.write(10, 10, (10, 20, 30))
    res1 = win_opaque.draw_to(dst, 0, 0)
    assert res1["mode"] == "memcpy"
    assert dst.read_pixel(10, 10) == (10, 20, 30)

    # 2. Transparent window -> slow-path
    win_trans = OptimizedWindowSim(16, 24)
    win_trans.set_transparent_color((0, 0, 0))
    win_trans.write(0, 0, (0, 0, 0))       # transparent
    win_trans.write(1, 1, (255, 255, 255)) # opaque
    res2 = win_trans.draw_to(dst, 20, 20)
    assert res2["mode"] == "pixel_by_pixel"
    assert dst.read_pixel(20, 20) == (0, 0, 0)
    assert dst.read_pixel(21, 21) == (255, 255, 255)


def test_layer_compositing_speedup_sim():
    """Verify speedup estimation model for desktop and mouse compositing."""
    est_desktop = LayerCompositingSpeedupSim.estimate_cycles(1024, 768, False)
    assert est_desktop["speedup"] > 50.0  # memcpy should estimate > 50x speedup
    assert est_desktop["mode"] == "line_memcpy"

    bench = LayerCompositingSpeedupSim.benchmark_multi_layer(1024, 768)
    assert bench["overall_speedup"] > 30.0
    assert bench["total_fast_cycles"] < bench["total_slow_cycles"]


def test_cpp20_ch09_optimizing_layer_redraw_executable():
    """Verify C++20 standalone executable compiles and passes all checks."""
    bin_path = Path("build/bin/ch09_optimizing_layer_redraw")
    if not bin_path.exists():
        # Build if not already built
        subprocess.run(["cmake", "--build", "build", "--target", "ch09_optimizing_layer_redraw"], check=True)
    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "All C++20 Section 9.5 validation checks successfully passed!" in res.stdout


def test_figure_generation_outputs():
    """Verify that Figures 9.13, 9.14, and 9.15 exist and are valid high-res PNGs."""
    p13 = plot_fig9_13_framebuffer_shadow_buffer_architecture()
    p14 = plot_fig9_14_framebuffer_copy_clipping_and_memcpy_math()
    p15 = plot_fig9_15_compositing_speedup_benchmark_comparison()

    for p in [p13, p14, p15]:
        assert Path(p).exists()
        assert Path(p).stat().st_size > 10000
        with Image.open(p) as img:
            assert img.format == "PNG"
            assert img.width >= 2000  # High-DPI verification


def test_notebook_structure_and_error_free_execution():
    """Verify Section 9.5 notebook exists, is valid JSON, and has zero execution errors."""
    import json
    nb_path = Path("9/9.5_Optimizing_Layer_Redraw_Bounding_Box.ipynb")
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

