"""Tests for Chapter 3.5: Drawing Pixels from Kernel (day03c)."""

import json
import os
import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common import (
    CallingConventionKind,
    KernelArgumentBridge,
    MikanKernelDrawer,
    UefiGraphicsPixelFormat,
    plot_bootloader_to_kernel_handoff,
    plot_kernel_pixel_stripe_pattern,
)


def test_abi_calling_convention_bridge():
    """Verify ABI argument register mapping and the EFIAPI bridge mechanism."""
    dummy_base = 0x8000_0000
    dummy_size = 1024 * 768 * 4
    bridge = KernelArgumentBridge(dummy_base, dummy_size)

    # 1. MS x64 ABI registers
    ms_regs = bridge.get_ms_x64_registers()
    assert ms_regs["RCX"] == dummy_base
    assert ms_regs["RDX"] == dummy_size
    assert ms_regs.get("RDI", 0) == 0
    assert ms_regs.get("RSI", 0) == 0

    # 2. System V AMD64 ABI registers
    sysv_regs = bridge.get_system_v_registers()
    assert sysv_regs["RDI"] == dummy_base
    assert sysv_regs["RSI"] == dummy_size
    assert sysv_regs.get("RCX", 0) == 0
    assert sysv_regs.get("RDX", 0) == 0

    # 3. Simulate call without EFIAPI (Standard host Linux / System V ABI)
    sim_sysv = bridge.simulate_call(use_efiapi=False)
    assert sim_sysv["caller_abi"] == CallingConventionKind.SYSTEM_V_AMD64
    assert sim_sysv["received_base"] == dummy_base
    assert sim_sysv["received_size"] == dummy_size
    assert sim_sysv["success"] is True

    # 4. Simulate call WITH EFIAPI (__attribute__((ms_abi)))
    sim_ms = bridge.simulate_call(use_efiapi=True)
    assert sim_ms["caller_abi"] == CallingConventionKind.MICROSOFT_X64
    assert sim_ms["received_base"] == 0  # Kernel expected RDI, but caller put it in RCX
    assert sim_ms["received_size"] == 0  # Kernel expected RSI, but caller put it in RDX
    assert sim_ms["success"] is False


def test_mikan_kernel_drawer_modulo256_buffer():
    """Verify kernel drawing loop (frame_buffer[i] = i % 256) and pixel packing."""
    drawer = MikanKernelDrawer(
        frame_buffer_base=0x8000_0000,
        width=640,
        height=480,
        pixels_per_scan_line=640,
        pixel_format=UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor,
    )
    drawer.execute_modulo256()

    assert len(drawer.buffer) == 640 * 480 * 4

    # Verify first 256 bytes
    for i in range(256):
        assert drawer.get_byte_at(i) == i

    # Verify pixel 0 in BGR format: bytes (0, 1, 2, 3) -> (B=0, G=1, R=2)
    assert drawer.get_pixel_bytes(0, 0) == (0, 1, 2, 3)
    assert drawer.get_pixel_rgb(0, 0) == (2, 1, 0)  # (R, G, B)

    # Verify pixel 1: bytes (4, 5, 6, 7) -> (B=4, G=5, R=6)
    assert drawer.get_pixel_bytes(1, 0) == (4, 5, 6, 7)
    assert drawer.get_pixel_rgb(1, 0) == (6, 5, 4)

    # Verify pixel 63: bytes (252, 253, 254, 255) -> (B=252, G=253, R=254)
    assert drawer.get_pixel_bytes(63, 0) == (252, 253, 254, 255)
    assert drawer.get_pixel_rgb(63, 0) == (254, 253, 252)

    # Verify pixel 64: modulo wraps around to bytes 0, 1, 2, 3
    assert drawer.get_pixel_bytes(64, 0) == (0, 1, 2, 3)
    assert drawer.get_pixel_rgb(64, 0) == (2, 1, 0)

    # Verify RGB format swap
    drawer_rgb = MikanKernelDrawer(
        frame_buffer_base=0x8000_0000,
        width=640,
        height=480,
        pixels_per_scan_line=640,
        pixel_format=UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor,
    )
    drawer_rgb.execute_modulo256()
    assert drawer_rgb.get_pixel_rgb(0, 0) == (0, 1, 2)


def test_stride_geometry_and_periodicity_analysis():
    """Verify stride vs 64-pixel cycle analysis for vertical vs slanted moire patterns."""
    # Stride = 1024 (1024 % 64 == 0) -> Pure vertical stripes
    drawer_1024 = MikanKernelDrawer(width=1024, height=768, pixels_per_scan_line=1024)
    res_1024 = drawer_1024.analyze_stride_periodicity()
    assert res_1024["is_pure_vertical"] is True
    assert res_1024["stride_pixel_mod"] == 0
    assert res_1024["line_shift_bytes"] == 0

    # Stride = 800 (800 % 64 == 32) -> Slanted moire pattern with 32-pixel horizontal shift per line
    drawer_800 = MikanKernelDrawer(width=800, height=600, pixels_per_scan_line=800)
    res_800 = drawer_800.analyze_stride_periodicity()
    assert res_800["is_pure_vertical"] is False
    assert res_800["stride_pixel_mod"] == 32
    assert res_800["stride_byte_mod"] == 128
    assert res_800["line_shift_pixels"] == 32

    # Render numpy array verification
    drawer_800.execute_modulo256()
    img_arr = drawer_800.render_to_numpy_array()
    assert img_arr.shape == (600, 800, 3)
    assert img_arr.dtype == "uint8"


def test_cpp20_kernel_pixel_sim_binary():
    """Verify that C++20 kernel_pixel_sim binary builds and passes all assertions."""
    bin_path = Path("build/bin/ch03_kernel_pixel_sim")
    if not bin_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch03_kernel_pixel_sim"], check=True)

    result = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "[SUCCESS] All Chapter 3.5 C++20 verifications passed!" in result.stdout


def test_generated_figures_ch03_section5():
    """Verify that Chapter 3.5 figures exist and have high resolution."""
    fig9_path = Path("3/result/fig3_09_bootloader_to_kernel_handoff.png")
    fig10_path = Path("3/result/fig3_10_kernel_pixel_stripe_pattern.png")

    plot_bootloader_to_kernel_handoff(str(fig9_path))
    plot_kernel_pixel_stripe_pattern(str(fig10_path))

    assert fig9_path.exists()
    assert fig10_path.exists()

    assert fig9_path.stat().st_size > 100_000
    assert fig10_path.stat().st_size > 100_000

    with Image.open(fig9_path) as img:
        assert img.width >= 2000
        assert img.height >= 1000

    with Image.open(fig10_path) as img:
        assert img.width >= 2000
        assert img.height >= 1000


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 3.5 notebook exists and has zero cell errors."""
    nb_path = Path("3/3.5_Drawing_Pixels_from_Kernel.ipynb")
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
