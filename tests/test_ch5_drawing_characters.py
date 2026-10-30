"""Tests for Chapter 5.1: Drawing Characters & References (osbook_day05a)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    PixelFormat,
    FrameBufferConfigSim,
    PixelColorSim,
    RGBResv8BitPerColorPixelWriterSim,
    BGRResv8BitPerColorPixelWriterSim,
    FONT_A_RAW,
    BitmapFontGlyph,
    GLYPH_A,
    write_ascii_sim,
    write_string_sim,
    ReferenceVsPointerComparison,
    get_reference_vs_pointer_comparisons,
    plot_fig5_01_font_glyph_and_bit_expansion,
    plot_fig5_02_reference_vs_pointer_memory_model,
)


def test_font_a_raw_and_glyph_properties():
    """Verify 8x16 font 'A' raw data and BitmapFontGlyph calculations."""
    assert len(FONT_A_RAW) == 16
    assert all(0 <= b <= 255 for b in FONT_A_RAW)

    # Glyph properties
    assert GLYPH_A.char == "A"
    assert GLYPH_A.width == 8
    assert GLYPH_A.height == 16
    assert GLYPH_A.count_lit_pixels() == 34

    # Bit-test logic equivalence: (byte << dx) & 0x80u vs (byte >> (7 - dx)) & 1
    for dy in range(16):
        for dx in range(8):
            lit_left = GLYPH_A.is_pixel_lit(dx, dy)
            lit_right = GLYPH_A.is_pixel_lit_right_shift(dx, dy)
            assert lit_left == lit_right

    # Matrix representation
    mat = GLYPH_A.to_matrix()
    assert mat.shape == (16, 8)
    assert mat.sum() == 34

    # Crossbar row at dy = 9: 0b01111110 (dx = 1..6 are True)
    assert not mat[9, 0]
    assert all(mat[9, 1:7])
    assert not mat[9, 7]

    # ASCII art output
    ascii_art = GLYPH_A.to_ascii_art(lit_char="#", unlit_char=".")
    lines = ascii_art.splitlines()
    assert len(lines) == 16
    assert lines[9] == ".######."


def test_write_ascii_sim_rendering():
    """Verify WriteAscii rendering in RGB and BGR pixel formats."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=100,
        horizontal_resolution=100,
        vertical_resolution=100,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    rgb_writer = RGBResv8BitPerColorPixelWriterSim(config)

    # Fill white
    for y in range(100):
        for x in range(100):
            rgb_writer.write(x, y, PixelColorSim(255, 255, 255))

    # Render 'A' at (20, 30) with black color
    drawn_count = write_ascii_sim(rgb_writer, 20, 30, "A", PixelColorSim(0, 0, 0))
    assert drawn_count == 34

    # Verify crossbar pixel (20 + 3, 30 + 9) = (23, 39) is black
    offset = rgb_writer.pixel_at(23, 39)
    assert rgb_writer.buffer[offset : offset + 3] == bytearray([0, 0, 0])

    # Verify unlit pixel (20 + 0, 30 + 0) = (20, 30) remains white
    offset_unlit = rgb_writer.pixel_at(20, 30)
    assert rgb_writer.buffer[offset_unlit : offset_unlit + 3] == bytearray([255, 255, 255])

    # Unsupported character should not render anything
    buf_snapshot = bytearray(rgb_writer.buffer)
    count_other = write_ascii_sim(rgb_writer, 20, 30, "Z", PixelColorSim(255, 0, 0))
    assert count_other == 0
    assert rgb_writer.buffer == buf_snapshot


def test_write_string_sim_advance():
    """Verify sequential string rendering and horizontal character advance."""
    config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=100,
        horizontal_resolution=100,
        vertical_resolution=100,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    writer = RGBResv8BitPerColorPixelWriterSim(config)

    # Draw "AA" (2 letters, 34 pixels each)
    total_drawn = write_string_sim(writer, 50, 50, "AA", PixelColorSim(10, 20, 30))
    assert total_drawn == 68

    # First 'A' crossbar (50 + 2, 59)
    off1 = writer.pixel_at(52, 59)
    assert writer.buffer[off1 : off1 + 3] == bytearray([10, 20, 30])

    # Second 'A' crossbar (58 + 2, 59) = (60, 59)
    off2 = writer.pixel_at(60, 59)
    assert writer.buffer[off2 : off2 + 3] == bytearray([10, 20, 30])


def test_reference_vs_pointer_comparisons():
    """Verify systematic Reference vs Pointer comparison items (Column 5.1)."""
    comparisons = get_reference_vs_pointer_comparisons()
    assert len(comparisons) >= 5

    features = [c.feature for c in comparisons]
    assert any("NULL" in f or "nullptr" in f for f in features)
    assert any("再束縛" in f or "再代入" in f for f in features)
    assert any("アセンブリ" in f for f in features)


def test_figure_generation_outputs(tmp_path: Path):
    """Verify high-resolution figure generation for Section 5.1."""
    fig1 = tmp_path / "fig5_01_test.png"
    fig2 = tmp_path / "fig5_02_test.png"

    path1 = plot_fig5_01_font_glyph_and_bit_expansion(fig1)
    path2 = plot_fig5_02_reference_vs_pointer_memory_model(fig2)

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


def test_cpp20_font_rendering_sim_standalone():
    """Verify standalone C++20 simulation binary builds and passes all tests."""
    binary_path = Path("build/bin/ch05_font_rendering_sim")
    assert binary_path.exists(), "Binary build/bin/ch05_font_rendering_sim must exist. Run cmake --build build."

    result = subprocess.run(
        [str(binary_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "Testing Font 'A' Bitmap Structure & Bit-Test Mathematics... PASSED" in result.stdout
    assert "Testing WriteAscii Character Rendering (RGB & BGR)... PASSED" in result.stdout
    assert "Testing Column 5.1: C++ Reference vs Pointer Principles... PASSED" in result.stdout
    assert "Testing Day05a Full VRAM Scene Simulation (White bg + Green rect + 'AA')... PASSED" in result.stdout
    assert "All Section 5.1 C++20 Verification Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 5.1 Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("5/5.1_Drawing_Characters_and_References.ipynb")
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
