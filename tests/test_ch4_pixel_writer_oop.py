"""Tests for Chapter 4.3: Refactoring with C++ OOP & Placement New (osbook_day04c)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    PixelFormat,
    FrameBufferConfigSim,
    PixelColorSim,
    PixelWriterSim,
    RGBResv8BitPerColorPixelWriterSim,
    BGRResv8BitPerColorPixelWriterSim,
    PlacementNewSimulator,
    BranchVsVtableBenchmark,
    PolymorphismCostMetrics,
    plot_fig4_05_pixel_writer_class_hierarchy_and_memory,
    plot_fig4_06_placement_new_and_runtime_cost,
)


def test_pixel_writer_class_hierarchy_and_abstract():
    """Verify that PixelWriterSim is an abstract base class and cannot be instantiated directly."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=800,
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )

    # PixelWriterSim is abstract because of pure virtual write()
    with pytest.raises(TypeError):
        PixelWriterSim(config)

    # Derived classes must be instantiable
    rgb_writer = RGBResv8BitPerColorPixelWriterSim(config)
    bgr_writer = BGRResv8BitPerColorPixelWriterSim(config)

    assert isinstance(rgb_writer, PixelWriterSim)
    assert isinstance(bgr_writer, PixelWriterSim)
    assert rgb_writer.config is config
    assert bgr_writer.config is config


def test_rgb_pixel_writer_rendering():
    """Verify RGB channel ordering and boundary checking in RGBResv8BitPerColorPixelWriter."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x1000_0000,
        pixels_per_scan_line=100,
        horizontal_resolution=100,
        vertical_resolution=100,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    writer = RGBResv8BitPerColorPixelWriterSim(config)

    # Write red pixel (255, 0, 0) at (10, 20)
    color = PixelColorSim(255, 128, 64)
    writer.write(10, 20, color)

    offset = config.calc_byte_offset(10, 20)
    assert writer.buffer[offset + 0] == 255  # R
    assert writer.buffer[offset + 1] == 128  # G
    assert writer.buffer[offset + 2] == 64   # B
    assert writer.buffer[offset + 3] == 0    # Reserved

    # Out of bounds coordinates must raise IndexError
    with pytest.raises(IndexError):
        writer.write(100, 20, color)
    with pytest.raises(IndexError):
        writer.write(10, 100, color)
    with pytest.raises(IndexError):
        writer.write(-1, 0, color)


def test_bgr_pixel_writer_rendering():
    """Verify BGR channel ordering and boundary checking in BGRResv8BitPerColorPixelWriter."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x2000_0000,
        pixels_per_scan_line=100,
        horizontal_resolution=100,
        vertical_resolution=100,
        pixel_format=PixelFormat.BGR_RESV_8BIT_PER_COLOR,
    )
    writer = BGRResv8BitPerColorPixelWriterSim(config)

    # Write pixel (255, 128, 64) at (5, 5) -> In BGR, memory is [64, 128, 255, 0]
    color = PixelColorSim(255, 128, 64)
    writer.write(5, 5, color)

    offset = config.calc_byte_offset(5, 5)
    assert writer.buffer[offset + 0] == 64   # B
    assert writer.buffer[offset + 1] == 128  # G
    assert writer.buffer[offset + 2] == 255  # R
    assert writer.buffer[offset + 3] == 0    # Reserved

    # Out of bounds coordinates must raise IndexError
    with pytest.raises(IndexError):
        writer.write(105, 5, color)


def test_placement_new_simulation():
    """Verify PlacementNewSimulator lifecycle, sizing, and error handling."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x3000_0000,
        pixels_per_scan_line=800,
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )

    # Normal allocation in 16-byte buffer
    sim = PlacementNewSimulator(buffer_capacity=16, alignment=8)
    assert not sim.is_active

    writer = sim.allocate_placement(RGBResv8BitPerColorPixelWriterSim, config)
    assert sim.is_active
    assert sim.object_type_name == "RGBResv8BitPerColorPixelWriterSim"
    assert isinstance(writer, RGBResv8BitPerColorPixelWriterSim)

    # Destructor
    sim.destroy_placement()
    assert not sim.is_active
    assert sim.allocated_object is None

    # Capacity overflow error
    tiny_sim = PlacementNewSimulator(buffer_capacity=8)
    with pytest.raises(MemoryError):
        tiny_sim.allocate_placement(RGBResv8BitPerColorPixelWriterSim, config)


def test_day04c_full_drawing_simulation():
    """Verify day04c full screen fill (white) and green rectangle drawing."""
    width, height = 800, 600
    config = FrameBufferConfigSim(
        frame_buffer_base=0x4000_0000,
        pixels_per_scan_line=width,
        horizontal_resolution=width,
        vertical_resolution=height,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    writer = RGBResv8BitPerColorPixelWriterSim(config)

    # 1. Fill entire screen white
    for x in range(width):
        for y in range(height):
            writer.write(x, y, PixelColorSim(255, 255, 255))

    # 2. Draw green rectangle @ (100, 100), 200x100
    for x in range(200):
        for y in range(100):
            writer.write(100 + x, 100 + y, PixelColorSim(0, 255, 0))

    # Check background pixel (0, 0) is white [255, 255, 255, 0]
    off_bg = config.calc_byte_offset(0, 0)
    assert writer.buffer[off_bg : off_bg + 4] == bytearray([255, 255, 255, 0])

    # Check inside green rectangle (150, 150) is green [0, 255, 0, 0]
    off_green = config.calc_byte_offset(150, 150)
    assert writer.buffer[off_green : off_green + 4] == bytearray([0, 255, 0, 0])

    # Check boundary edges of rectangle
    off_top_left = config.calc_byte_offset(100, 100)
    assert writer.buffer[off_top_left : off_top_left + 4] == bytearray([0, 255, 0, 0])
    off_bottom_right = config.calc_byte_offset(299, 199)
    assert writer.buffer[off_bottom_right : off_bottom_right + 4] == bytearray([0, 255, 0, 0])

    # Check pixel just outside rectangle (300, 200) is white
    off_outside = config.calc_byte_offset(300, 200)
    assert writer.buffer[off_outside : off_outside + 4] == bytearray([255, 255, 255, 0])


def test_branch_vs_vtable_benchmark():
    """Verify performance metrics model comparing if-else branch vs vtable calls."""
    metrics = BranchVsVtableBenchmark.evaluate(800, 600)
    assert isinstance(metrics, PolymorphismCostMetrics)
    assert metrics.total_pixels == 480_000
    assert metrics.branch_instructions_ifelse == 960_000
    assert metrics.vtable_indirect_calls == 480_000
    assert metrics.estimated_cycles_vtable < metrics.estimated_cycles_ifelse
    assert metrics.code_modularity_score >= 9.0


def test_figure_generation_outputs(tmp_path):
    """Verify generation, resolution, and valid image format for Figures 4.5 and 4.6."""
    fig1 = tmp_path / "fig4_05_test.png"
    fig2 = tmp_path / "fig4_06_test.png"

    path1 = plot_fig4_05_pixel_writer_class_hierarchy_and_memory(str(fig1))
    path2 = plot_fig4_06_placement_new_and_runtime_cost(str(fig2))

    assert Path(path1).exists()
    assert Path(path2).exists()

    with Image.open(path1) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 2000
        assert img1.format == "PNG"

    with Image.open(path2) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 2000
        assert img2.format == "PNG"


def test_cpp20_pixel_writer_oop_sim_standalone():
    """Verify standalone C++20 simulation binary builds and passes all tests."""
    binary_path = Path("build/bin/ch04_pixel_writer_oop_sim")
    assert binary_path.exists(), "Binary build/bin/ch04_pixel_writer_oop_sim must exist. Run cmake --build build."

    result = subprocess.run(
        [str(binary_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "Testing Placement New Construction on Static Buffer... PASSED" in result.stdout
    assert "Testing RGB vs BGR Polymorphic Pixel Writers... PASSED" in result.stdout
    assert "Testing Day04c Drawing (White Background & Green Rectangle)... PASSED" in result.stdout
    assert "Testing Column 4.2 Compile-Time Safety Principles... PASSED" in result.stdout
    assert "All Section 4.3 C++20 Verification Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 4.3 Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("4/4.3_Refactoring_with_Cpp_OOP.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 12, f"Notebook should have at least 12 detailed cells, got {len(cells)}"

    # Check cell 0 has Colab badge / setup
    first_cell_src = "".join(cells[0].get("source", []))
    assert "colab" in first_cell_src.lower() or "google colab" in first_cell_src.lower()

    # Verify every code cell executed without errors
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} must be executed"
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"Code cell {idx} failed with error: {out.get('evalue')}"
