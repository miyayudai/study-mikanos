"""Tests for Chapter 4.2: Drawing Pixels Freely and ABI (osbook_day04b)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    PixelFormat,
    FrameBufferConfigSim,
    PixelColorSim,
    WritePixelSimulator,
    AbiKind,
    AbiSpec,
    AbiComparisonAnalyzer,
    plot_fig4_03_framebuffer_config_and_coordinate_addressing,
    plot_fig4_04_abi_calling_convention_comparison,
)


def test_framebuffer_config_layout_and_offsets():
    """Verify FrameBufferConfig struct size, member offsets, and address math."""
    base = 0x8000_0000
    config = FrameBufferConfigSim(
        frame_buffer_base=base,
        pixels_per_scan_line=1024,
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )

    # 1. Struct layout properties
    assert config.STRUCT_SIZE == 24
    assert config.FIELD_OFFSETS["frame_buffer"] == (0, 8, "uint8_t*")
    assert config.FIELD_OFFSETS["pixels_per_scan_line"] == (8, 4, "uint32_t")
    assert config.FIELD_OFFSETS["horizontal_resolution"] == (12, 4, "uint32_t")
    assert config.FIELD_OFFSETS["vertical_resolution"] == (16, 4, "uint32_t")
    assert config.FIELD_OFFSETS["pixel_format"] == (20, 4, "enum PixelFormat")

    # 2. Stride padding
    assert config.get_stride_padding_pixels() == 224
    assert config.total_buffer_size_bytes() == 1024 * 600 * 4

    # 3. Coordinate calculation
    # Top-left (0, 0)
    assert config.calc_pixel_index(0, 0) == 0
    assert config.calc_byte_offset(0, 0) == 0
    assert config.calc_physical_address(0, 0) == base

    # Coordinate (10, 20) with stride 1024
    expected_idx = 20 * 1024 + 10  # 20490
    assert config.calc_pixel_index(10, 20) == expected_idx
    assert config.calc_byte_offset(10, 20) == expected_idx * 4  # 81960
    assert config.calc_physical_address(10, 20) == base + 81960

    # Bounds check
    assert config.is_in_bounds(0, 0) is True
    assert config.is_in_bounds(799, 599) is True
    assert config.is_in_bounds(800, 599) is False  # outside horizontal
    assert config.is_in_bounds(799, 600) is False  # outside vertical
    assert config.is_in_bounds(-1, 0) is False

    # Negative validation
    with pytest.raises(ValueError):
        FrameBufferConfigSim(base, 500, 800, 600, PixelFormat.RGB_RESV_8BIT_PER_COLOR)


def test_pixel_color_packing_rgb_and_bgr():
    """Verify PixelColor 8-bit bounds and RGB/BGR byte packing."""
    color = PixelColorSim(r=0xAA, g=0xBB, b=0xCC)

    # RGB: [R, G, B, Reserved]
    rgb_bytes = color.to_bytes(PixelFormat.RGB_RESV_8BIT_PER_COLOR)
    assert len(rgb_bytes) == 4
    assert rgb_bytes[0] == 0xAA
    assert rgb_bytes[1] == 0xBB
    assert rgb_bytes[2] == 0xCC
    assert rgb_bytes[3] == 0x00

    # BGR: [B, G, R, Reserved]
    bgr_bytes = color.to_bytes(PixelFormat.BGR_RESV_8BIT_PER_COLOR)
    assert len(bgr_bytes) == 4
    assert bgr_bytes[0] == 0xCC
    assert bgr_bytes[1] == 0xBB
    assert bgr_bytes[2] == 0xAA
    assert bgr_bytes[3] == 0x00

    # Range validation
    with pytest.raises(ValueError):
        PixelColorSim(256, 0, 0)
    with pytest.raises(ValueError):
        PixelColorSim(-1, 0, 0)


def test_write_pixel_simulator_drawing_and_bounds():
    """Verify WritePixel operation, error return on out-of-bounds, and raw VRAM state."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=10,
        horizontal_resolution=8,
        vertical_resolution=6,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    sim = WritePixelSimulator(config)

    # Normal write at (2, 3)
    c = PixelColorSim(100, 150, 200)
    ret = sim.write_pixel(2, 3, c)
    assert ret == 0

    read_back = sim.read_pixel_color(2, 3)
    assert read_back is not None
    assert read_back.r == 100
    assert read_back.g == 150
    assert read_back.b == 200

    # Out of bounds
    assert sim.write_pixel(-1, 0, c) == -1
    assert sim.write_pixel(8, 0, c) == -1
    assert sim.write_pixel(0, 6, c) == -1
    assert sim.read_pixel_color(8, 0) is None


def test_day04b_screen_fill_and_rectangle():
    """Verify complete MikanOS day04b drawing: White background and Green rectangle at (100, 100)."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=400,
        horizontal_resolution=400,
        vertical_resolution=300,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    sim = WritePixelSimulator(config)
    sim.execute_day04b_demo()

    # Background at (50, 50) is white
    c_bg = sim.read_pixel_color(50, 50)
    assert c_bg is not None
    assert (c_bg.r, c_bg.g, c_bg.b) == (255, 255, 255)

    # Inside rectangle at (100, 100) and (250, 150) is green (0, 255, 0)
    c_rect_corner = sim.read_pixel_color(100, 100)
    assert c_rect_corner is not None
    assert (c_rect_corner.r, c_rect_corner.g, c_rect_corner.b) == (0, 255, 0)

    c_rect_inner = sim.read_pixel_color(250, 150)
    assert c_rect_inner is not None
    assert (c_rect_inner.r, c_rect_inner.g, c_rect_inner.b) == (0, 255, 0)

    # Outside rectangle at (310, 100) is white
    c_outside = sim.read_pixel_color(310, 100)
    assert c_outside is not None
    assert (c_outside.r, c_outside.g, c_outside.b) == (255, 255, 255)

    # Verify numpy array conversion
    arr = sim.to_rgb_numpy()
    assert arr.shape == (300, 400, 3)
    assert (arr[100, 100] == [0, 255, 0]).all()
    assert (arr[50, 50] == [255, 255, 255]).all()


def test_abi_calling_convention_comparison():
    """Verify System V AMD64 ABI vs Microsoft x64 ABI specifications."""
    sysv = AbiComparisonAnalyzer.get_spec(AbiKind.SYSTEM_V_AMD64)
    ms = AbiComparisonAnalyzer.get_spec(AbiKind.MICROSOFT_X64)

    # Register ordering
    assert sysv.integer_arg_registers == ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
    assert ms.integer_arg_registers == ["RCX", "RDX", "R8", "R9"]

    # Special stack zones
    assert sysv.shadow_space_bytes == 0
    assert ms.shadow_space_bytes == 32
    assert sysv.red_zone_bytes == 128
    assert ms.red_zone_bytes == 0

    # RDI/RSI preservation
    assert "RDI" in ms.callee_saved_registers
    assert "RSI" in ms.callee_saved_registers
    assert "RDI" not in sysv.callee_saved_registers
    assert "RSI" not in sysv.callee_saved_registers
    assert "RDI" in sysv.caller_saved_registers
    assert "RSI" in sysv.caller_saved_registers

    # Argument mapping check
    mapping_sysv = AbiComparisonAnalyzer.map_arguments(AbiKind.SYSTEM_V_AMD64, 7)
    assert mapping_sysv[0]["register"] == "RDI"
    assert mapping_sysv[5]["register"] == "R9"
    assert mapping_sysv[6]["location"] == "stack"

    mapping_ms = AbiComparisonAnalyzer.map_arguments(AbiKind.MICROSOFT_X64, 5)
    assert mapping_ms[0]["register"] == "RCX"
    assert mapping_ms[3]["register"] == "R9"
    assert mapping_ms[4]["location"] == "stack"
    assert mapping_ms[4]["offset"] == 40  # 32 (shadow) + 8 (return addr)


def test_uefi_kernel_bridge_and_efiapi_trap():
    """Verify why omitting EFIAPI in Main.c ensures System V ABI register match."""
    # When EFIAPI is omitted (Correct MikanOS implementation)
    analysis_correct = AbiComparisonAnalyzer.analyze_uefi_kernel_bridge(use_efiapi=False)
    assert analysis_correct["is_compatible"] is True
    assert analysis_correct["passed_in_register"] == "RDI"
    assert analysis_correct["expected_in_register"] == "RDI"

    # When EFIAPI is mistakenly included
    analysis_trap = AbiComparisonAnalyzer.analyze_uefi_kernel_bridge(use_efiapi=True)
    assert analysis_trap["is_compatible"] is False
    assert analysis_trap["passed_in_register"] == "RCX"
    assert analysis_trap["expected_in_register"] == "RDI"


def test_figure_generation_outputs(tmp_path: Path):
    """Verify generation and integrity of Chapter 4.2 high-DPI architecture figures."""
    fig1 = tmp_path / "fig4_03.png"
    fig2 = tmp_path / "fig4_04.png"

    plot_fig4_03_framebuffer_config_and_coordinate_addressing(fig1)
    plot_fig4_04_abi_calling_convention_comparison(fig2)

    assert fig1.exists()
    assert fig2.exists()
    assert fig1.stat().st_size > 50_000
    assert fig2.stat().st_size > 50_000

    with Image.open(fig1) as img1:
        assert img1.width >= 1500
        assert img1.height >= 900

    with Image.open(fig2) as img2:
        assert img2.width >= 1500
        assert img2.height >= 900


def test_cpp20_drawing_pixels_sim_standalone():
    """Verify execution of standalone C++20 test executable."""
    binary_path = Path("build/bin/ch04_drawing_pixels_sim")
    if not binary_path.exists():
        pytest.skip("Binary ch04_drawing_pixels_sim not yet built")

    result = subprocess.run([str(binary_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All Chapter 4.2 C++20 tests passed successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Chapter 4.2 Jupyter notebook structure, Cell 0 setup, and error-free execution."""
    nb_path = Path("4/4.2_Drawing_Pixels_Freely_and_ABI.ipynb")
    if not nb_path.exists():
        pytest.skip("Notebook 4.2_Drawing_Pixels_Freely_and_ABI.ipynb not yet created")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10

    # Cell 0 must be Colab setup
    first_cell_src = "".join(nb["cells"][0].get("source", []))
    assert "google.colab" in first_cell_src or "colab" in first_cell_src.lower()

    # Check that no cell has output_type == "error"
    for idx, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", f"Cell {idx} contains execution error: {out}"

    # Verify coverage of core technical keywords
    all_text = json.dumps(nb, ensure_ascii=False)
    keywords = [
        "FrameBufferConfig",
        "pixels_per_scan_line",
        "WritePixel",
        "kPixelRGBResv8BitPerColor",
        "kPixelBGRResv8BitPerColor",
        "System V AMD64 ABI",
        "Microsoft x64 ABI",
        "EFIAPI",
        "RDI",
        "RCX",
        "Shadow Space",
    ]
    for kw in keywords:
        assert kw in all_text, f"Notebook missing key concept keyword: {kw}"
