"""Tests for Chapter 8.2: Inspecting UEFI MemoryMap in Kernel."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    UefiMemoryType,
    UefiMemoryDescriptor,
    UefiMemoryMapSimulator,
    KernelMemoryMapInspector,
    plot_fig8_04_uefi_memory_map_kernel_handoff,
    plot_fig8_05_memory_descriptor_stepping_math,
    plot_fig8_06_available_memory_distribution,
)


def test_uefi_memory_type_classification():
    """Verify classification of UEFI memory types into OS-reusable vs preserved/MMIO."""
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiConventionalMemory) is True
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiBootServicesCode) is True
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiBootServicesData) is True

    # Integer enum values
    assert KernelMemoryMapInspector.is_available(7) is True  # ConventionalMemory
    assert KernelMemoryMapInspector.is_available(3) is True  # BootServicesCode
    assert KernelMemoryMapInspector.is_available(4) is True  # BootServicesData

    # Non-reusable / preserved types
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiReservedMemoryType) is False
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiLoaderCode) is False
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiLoaderData) is False
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiRuntimeServicesCode) is False
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiACPIReclaimMemory) is False
    assert KernelMemoryMapInspector.is_available(UefiMemoryType.EfiMemoryMappedIO) is False
    assert KernelMemoryMapInspector.is_available(999) is False


def test_kernel_memory_map_inspector_scanning():
    """Verify scanning and physical memory size aggregation on simulated QEMU memory map."""
    sim = KernelMemoryMapInspector.create_typical_qemu_memmap()
    assert sim.descriptor_size == 48
    assert sim.descriptor_version == 1

    regions = KernelMemoryMapInspector.scan_available_memory(sim)
    # Among the 13 entries in typical QEMU map:
    # Entry 1: Conventional (16128 pages = 63 MiB)
    # Entry 2: BootServicesCode (2048 pages = 8 MiB)
    # Entry 3: BootServicesData (4096 pages = 16 MiB)
    # Entry 6: Conventional (755712 pages = 2952 MiB)
    # Exactly 4 regions must be recognized as available.
    assert len(regions) == 4

    r0 = regions[0]
    assert r0["type"] == UefiMemoryType.EfiConventionalMemory
    assert r0["physical_start"] == 0x0010_0000
    assert r0["number_of_pages"] == 16128
    assert r0["size_bytes"] == 16128 * 4096
    assert r0["physical_end"] == 0x0010_0000 + 16128 * 4096 - 1

    total_avail = KernelMemoryMapInspector.calculate_total_available_bytes(sim)
    expected_pages = 16128 + 2048 + 4096 + 755712
    expected_bytes = expected_pages * 4096
    assert total_avail == expected_bytes
    assert total_avail > int(2.9 * 1024 * 1024 * 1024)  # ~3 GiB total available RAM


def test_unsafe_sizeof_stepping_corruption_demonstration():
    """Verify that sizeof(MemoryDescriptor) == 40 bytes corrupts descriptor fields when descriptor_size == 48 bytes."""
    sim = KernelMemoryMapInspector.create_typical_qemu_memmap()
    stepping_results = KernelMemoryMapInspector.simulate_unsafe_stepping(sim)

    # First entry (index 0) starts at offset 0 in both stepping modes
    assert stepping_results[0]["is_corrupted"] is False
    assert stepping_results[0]["offset_read"] == 0

    # From index 1 onwards, naive sizeof(MemoryDescriptor)=40 causes off-by-8*i corruption
    assert stepping_results[1]["is_corrupted"] is True
    assert stepping_results[1]["offset_read"] == 40
    assert stepping_results[1]["expected_offset"] == 48

    assert stepping_results[2]["is_corrupted"] is True
    assert stepping_results[2]["offset_read"] == 80
    assert stepping_results[2]["expected_offset"] == 96


def test_figure_generation_outputs():
    """Verify Chapter 8.2 figures exist and are high-resolution valid PNG images."""
    f1 = Path(plot_fig8_04_uefi_memory_map_kernel_handoff())
    f2 = Path(plot_fig8_05_memory_descriptor_stepping_math())
    f3 = Path(plot_fig8_06_available_memory_distribution())

    for f in (f1, f2, f3):
        assert f.exists(), f"Figure not found: {f}"
        assert f.stat().st_size > 50_000, f"Figure too small: {f} ({f.stat().st_size} bytes)"
        with Image.open(f) as img:
            assert img.format == "PNG"
            assert img.width >= 1200
            assert img.height >= 700


def test_cpp20_standalone_uefi_memmap_sim():
    """Verify standalone C++20 memory map inspection simulator compiles and passes tests."""
    bin_path = Path("build/bin/ch08_uefi_memmap_sim")
    if not bin_path.exists():
        compile_cmd = [
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            "src/ch08/uefi_memmap_sim.cpp", "-o", "src/ch08/uefi_memmap_sim"
        ]
        res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"
        bin_path = Path("src/ch08/uefi_memmap_sim")

    run_res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "[PASS] All Chapter 8.2 memory map inspection tests passed successfully!" in run_res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify 8/8.2_Inspecting_UEFI_Memory_Map_in_Kernel.ipynb exists and is valid JSON."""
    nb_path = Path("8/8.2_Inspecting_UEFI_Memory_Map_in_Kernel.ipynb")
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 10

    # Cell 0 must be Google Colab setup
    first_cell = nb_data["cells"][0]
    assert first_cell["cell_type"] == "code"
    assert "Google Colab" in "".join(first_cell["source"])

    # Cell 1 must have Colab badge
    second_cell = nb_data["cells"][1]
    assert second_cell["cell_type"] == "markdown"
    assert "colab-badge.svg" in "".join(second_cell["source"])

    # Verify no execution errors recorded in cell outputs
    for idx, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] == "code":
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"Cell {idx} contains execution error: {out}"
