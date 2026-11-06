"""Tests for Chapter 8.3: Relocating Kernel Data Structures."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    UefiMemoryType,
    RelocatedFBConfigSim,
    MemoryMapSim,
    DataRelocationSimulator,
    plot_fig8_07_data_structure_relocation_danger,
    plot_fig8_08_copy_constructor_and_stack_isolation,
    plot_fig8_09_memory_map_buffer_lifetime_safety,
)


def test_is_available_logic():
    """Verify IsAvailable() inline function behavior for all UEFI memory types."""
    # The 3 reusable types
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiBootServicesCode) is True
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiBootServicesData) is True
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiConventionalMemory) is True

    # Integer enum values
    assert DataRelocationSimulator.is_available(3) is True
    assert DataRelocationSimulator.is_available(4) is True
    assert DataRelocationSimulator.is_available(7) is True

    # Preserved / Hardware reserved types
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiReservedMemoryType) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiLoaderCode) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiLoaderData) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiRuntimeServicesCode) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiRuntimeServicesData) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiUnusableMemory) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiACPIReclaimMemory) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiACPIMemoryNVS) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiMemoryMappedIO) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiMemoryMappedIOPortSpace) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiPalCode) is False
    assert DataRelocationSimulator.is_available(UefiMemoryType.EfiPersistentMemory) is False
    assert DataRelocationSimulator.is_available(9999) is False


def test_unprotected_reference_corruption_simulation():
    """Verify simulation of raw reference corruption when UEFI stack is overwritten."""
    sim = DataRelocationSimulator()
    result = sim.simulate_unprotected_reference_corruption()

    assert result["safe"] is False
    assert result["original_fb_addr"] == 0xE000_0000
    assert result["corrupted_fb_addr"] == 0x0000_0000
    assert result["original_map_size"] == 1200
    assert result["corrupted_map_size"] == 0
    assert "page fault" in result["failure_mode"].lower()


def test_relocated_value_copy_safety_simulation():
    """Verify that value-copy relocation protects kernel data against UEFI stack wipes."""
    sim = DataRelocationSimulator()
    result = sim.simulate_relocated_value_copy()

    assert result["safe"] is True
    assert result["kernel_fb_addr"] == 0xE000_0000
    assert result["kernel_map_size"] == 1200
    assert "intact" in result["kernel_stack_status"].lower()
    assert "safely isolated" in result["uefi_stack_status"].lower()


def test_shallow_copy_lifetime_analysis():
    """Verify phase isolation analysis for the shallow-copied memory_map.buffer pointer."""
    sim = DataRelocationSimulator()
    analysis = sim.analyze_shallow_copy_lifetime()

    assert "Kernel Stack" in analysis["struct_location"]
    assert "UEFI BootServicesData" in analysis["buffer_location"]

    # Phase 1: Scanning phase is safe before allocator runs
    p1 = analysis["phase1_scanning"]
    assert p1["active_allocator"] is False
    assert p1["is_buffer_safe"] is True

    # Phase 2: Allocation phase reclaims memory, but buffer is no longer accessed
    p2 = analysis["phase2_allocation"]
    assert p2["active_allocator"] is True
    assert p2["is_buffer_safe"] is False


def test_figure_generation_outputs():
    """Verify Chapter 8.3 figures exist and are high-resolution valid PNG images."""
    f1 = Path(plot_fig8_07_data_structure_relocation_danger())
    f2 = Path(plot_fig8_08_copy_constructor_and_stack_isolation())
    f3 = Path(plot_fig8_09_memory_map_buffer_lifetime_safety())

    for f in (f1, f2, f3):
        assert f.exists(), f"Figure not found: {f}"
        assert f.stat().st_size > 50_000, f"Figure too small: {f} ({f.stat().st_size} bytes)"
        with Image.open(f) as img:
            assert img.format == "PNG"
            assert img.width >= 1200


def test_cpp20_relocate_data_sim_standalone():
    """Verify standalone compilation and execution of C++20 data relocation simulator."""
    binary_path = Path("build/bin/ch08_relocate_data_sim")
    if not binary_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch08_relocate_data_sim"], check=True)

    result = subprocess.run(
        [str(binary_path), "--test"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[PASS] All Chapter 8.3 data structure relocation tests passed successfully!" in result.stdout
    assert "sizeof(FrameBufferConfig) = 24 bytes" in result.stdout
    assert "sizeof(MemoryMap)         = 48 bytes" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Chapter 8.3 notebook exists and has zero cell execution errors."""
    nb_path = Path("8/8.3_Relocating_Kernel_Data_Structures.ipynb")
    assert nb_path.exists(), "Notebook 8.3 does not exist yet"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, f"Notebook has too few cells: {len(cells)}"

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"Cell {idx} has an execution error: {out.get('ename')}: {out.get('evalue')}"
                )
