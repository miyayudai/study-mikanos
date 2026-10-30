"""Tests for Chapter 3.4: Drawing Pixels from Bootloader (UEFI GOP & Framebuffer Architecture)."""

import os
import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common import (
    UefiGraphicsPixelFormat,
    EfiPixelBitmask,
    EfiGraphicsOutputModeInformation,
    EfiGraphicsOutputProtocolMode,
    EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID,
    UefiGopSimulator,
    plot_uefi_gop_framebuffer_architecture,
    plot_pixel_format_and_scanline_stride,
)


def test_uefi_graphics_pixel_format_and_unicode_mapping():
    """Verify UEFI pixel format enumeration values and string representation matching Main.c."""
    assert UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor.value == 0
    assert UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor.value == 1
    assert UefiGraphicsPixelFormat.PixelBitMask.value == 2
    assert UefiGraphicsPixelFormat.PixelBltOnly.value == 3
    assert UefiGraphicsPixelFormat.PixelFormatMax.value == 4

    sim_rgb = UefiGopSimulator(
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor,
    )
    assert (
        sim_rgb.get_pixel_format_unicode()
        == "PixelRedGreenBlueReserved8BitPerColor"
    )

    sim_bgr = UefiGopSimulator(
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor,
    )
    assert (
        sim_bgr.get_pixel_format_unicode()
        == "PixelBlueGreenRedReserved8BitPerColor"
    )

    sim_mask = UefiGopSimulator(
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=UefiGraphicsPixelFormat.PixelBitMask,
    )
    assert sim_mask.get_pixel_format_unicode() == "PixelBitMask"


def test_efi_gop_mode_structures_layout_and_size():
    """Verify GOP mode information and protocol structures layout and frame buffer size derivation."""
    info = EfiGraphicsOutputModeInformation(
        version=0,
        horizontal_resolution=1024,
        vertical_resolution=768,
        pixel_format=UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor,
        pixels_per_scan_line=1024,
    )
    assert info.horizontal_resolution == 1024
    assert info.vertical_resolution == 768
    assert info.pixels_per_scan_line == 1024

    mode = EfiGraphicsOutputProtocolMode(
        max_mode=1,
        mode=0,
        info=info,
        size_of_info=36,
        frame_buffer_base=0x80000000,
    )
    # 1024 * 768 * 4 = 3,145,728 bytes
    assert mode.frame_buffer_size == 1024 * 768 * 4
    assert mode.frame_buffer_base == 0x80000000
    assert EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID == (
        "9042a9de-23dc-4a38-96fb-7aded080516a"
    )


def test_scanline_stride_and_padding_address_math():
    """Verify that scanline stride (pixels_per_scan_line) properly dictates memory offset and padding."""
    # Screen with 800 width, but GPU stride padded to 832 (common 128-byte/256-byte hardware alignment)
    sim = UefiGopSimulator(
        horizontal_resolution=800,
        vertical_resolution=600,
        pixels_per_scan_line=832,
    )

    assert sim.info.horizontal_resolution == 800
    assert sim.info.pixels_per_scan_line == 832
    assert sim.mode.frame_buffer_size == 832 * 600 * 4

    # Top-left corner (0, 0)
    assert sim.calculate_byte_offset(0, 0) == 0

    # End of first visible row (799, 0)
    assert sim.calculate_byte_offset(799, 0) == 799 * 4

    # Beginning of second row (0, 1): MUST jump by stride (832 * 4 = 3328 bytes)
    assert sim.calculate_byte_offset(0, 1) == 832 * 4

    # Difference between stride and width is 32 pixels = 128 bytes padding per scanline
    padding_bytes = (sim.info.pixels_per_scan_line - sim.info.horizontal_resolution) * 4
    assert padding_bytes == 128
    assert sim.calculate_byte_offset(0, 1) - (sim.info.horizontal_resolution * 4) == 128

    # Bounds check error on coordinates outside stride
    with pytest.raises(IndexError):
        sim.calculate_byte_offset(832, 0)


def test_pixel_color_packing_and_format_distinction():
    """Verify that RGB vs BGR formats write correct byte sequences into raw VRAM buffer."""
    sim_rgb = UefiGopSimulator(
        horizontal_resolution=10,
        vertical_resolution=10,
        pixel_format=UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor,
    )
    sim_rgb.write_pixel(1, 1, r=0xAA, g=0xBB, b=0xCC)
    off_rgb = sim_rgb.calculate_byte_offset(1, 1)
    assert sim_rgb.vram[off_rgb + 0] == 0xAA  # R
    assert sim_rgb.vram[off_rgb + 1] == 0xBB  # G
    assert sim_rgb.vram[off_rgb + 2] == 0xCC  # B
    assert sim_rgb.vram[off_rgb + 3] == 0x00  # Reserved
    assert sim_rgb.get_pixel_rgb(1, 1) == (0xAA, 0xBB, 0xCC)

    sim_bgr = UefiGopSimulator(
        horizontal_resolution=10,
        vertical_resolution=10,
        pixel_format=UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor,
    )
    sim_bgr.write_pixel(1, 1, r=0xAA, g=0xBB, b=0xCC)
    off_bgr = sim_bgr.calculate_byte_offset(1, 1)
    assert sim_bgr.vram[off_bgr + 0] == 0xCC  # B
    assert sim_bgr.vram[off_bgr + 1] == 0xBB  # G
    assert sim_bgr.vram[off_bgr + 2] == 0xAA  # R
    assert sim_bgr.vram[off_bgr + 3] == 0x00  # Reserved
    assert sim_bgr.get_pixel_rgb(1, 1) == (0xAA, 0xBB, 0xCC)


def test_mikanos_day03b_whole_screen_white_fill():
    """Verify MikanOS day03b implementation: filling entire VRAM with 255 (0xFF) results in pure white."""
    sim = UefiGopSimulator(
        horizontal_resolution=100,
        vertical_resolution=100,
        pixels_per_scan_line=100,
        pixel_format=UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor,
    )

    # Initially all bytes 0
    assert all(b == 0 for b in sim.vram)

    # Day03b logic: frame_buffer[i] = 255
    sim.fill_all(255)

    # Every byte must be 255
    assert all(b == 255 for b in sim.vram)

    # Sample pixels must all be pure white (255, 255, 255)
    assert sim.get_pixel_rgb(0, 0) == (255, 255, 255)
    assert sim.get_pixel_rgb(50, 50) == (255, 255, 255)
    assert sim.get_pixel_rgb(99, 99) == (255, 255, 255)

    # Image matrix conversion
    matrix = sim.to_rgb_matrix()
    assert matrix.shape == (100, 100, 3)
    assert (matrix == 255).all()


def test_cpp20_gop_sim_binary():
    """Verify that C++20 gop_sim binary builds and passes all assertions."""
    bin_path = Path("build/bin/ch03_gop_sim")
    if not bin_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch03_gop_sim"], check=True)

    result = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "ALL CHAPTER 3.4 UEFI GOP SIMULATION TESTS PASSED!" in result.stdout


def test_generated_figures_ch03_section4():
    """Verify that Chapter 3.4 figures are generated with expected DPI and dimensions."""
    fig7_path = Path("3/result/fig3_07_uefi_gop_framebuffer_architecture.png")
    fig8_path = Path("3/result/fig3_08_pixel_format_and_scanline_stride.png")

    plot_uefi_gop_framebuffer_architecture(fig7_path)
    plot_pixel_format_and_scanline_stride(fig8_path)

    assert fig7_path.exists()
    assert fig8_path.exists()

    assert fig7_path.stat().st_size > 100_000
    assert fig8_path.stat().st_size > 100_000

    with Image.open(fig7_path) as img:
        assert img.width >= 2000
        assert img.height >= 1000

    with Image.open(fig8_path) as img:
        assert img.width >= 2000
        assert img.height >= 1000


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 3.4 notebook exists and has zero cell errors."""
    import json
    nb_path = Path("3/3.4_Drawing_Pixels_from_Bootloader.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist"

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert len(nb["cells"]) >= 10
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5

    for idx, cell in enumerate(code_cells):
        for out in cell.get("outputs", []):
            assert (
                out.get("output_type") != "error"
            ), f"Error in code cell {idx}: {out.get('evalue')}"

