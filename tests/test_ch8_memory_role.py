"""Tests for Chapter 8.1: Role of OS Memory Management Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    PageFaultType,
    PageFaultException,
    PageFrameAddressMath,
    MemoryAllocationStrategyComparator,
    VirtualPageEntry,
    PhysicalVsVirtualMemorySim,
    plot_fig8_01_physical_vs_virtual_memory_architecture,
    plot_fig8_02_memory_management_data_structures_comparison,
    plot_fig8_03_chapter8_memory_subsystem_roadmap,
)


def test_page_frame_address_math_pfn_conversion():
    """Verify 4 KiB page frame arithmetic and boundary alignments."""
    assert PageFrameAddressMath.PAGE_SIZE == 4096
    assert PageFrameAddressMath.phys_to_pfn(0x0000) == 0
    assert PageFrameAddressMath.phys_to_pfn(0x1000) == 1
    assert PageFrameAddressMath.phys_to_pfn(0x1FFF) == 1
    assert PageFrameAddressMath.phys_to_pfn(0x2000) == 2
    assert PageFrameAddressMath.phys_to_pfn(0x100000) == 256  # 1 MiB

    assert PageFrameAddressMath.pfn_to_phys(0) == 0x0000
    assert PageFrameAddressMath.pfn_to_phys(1) == 0x1000
    assert PageFrameAddressMath.pfn_to_phys(256) == 0x100000

    assert PageFrameAddressMath.page_offset(0x1000) == 0x000
    assert PageFrameAddressMath.page_offset(0x1234) == 0x234
    assert PageFrameAddressMath.page_offset(0x1FFF) == 0xFFF

    assert PageFrameAddressMath.is_page_aligned(0x1000) is True
    assert PageFrameAddressMath.is_page_aligned(0x2000) is True
    assert PageFrameAddressMath.is_page_aligned(0x1001) is False
    assert PageFrameAddressMath.is_page_aligned(0x1008) is False

    assert PageFrameAddressMath.align_down(0x1FEE) == 0x1000
    assert PageFrameAddressMath.align_up(0x1001) == 0x2000
    assert PageFrameAddressMath.align_up(0x2000) == 0x2000


def test_canonical_virtual_address_decomposition():
    """Verify x86-64 48-bit canonical virtual address decomposition."""
    # Lower canonical address (User mode)
    user_vaddr = 0x00007FFFFFFF1234
    user_decomp = PageFrameAddressMath.decompose_virtual_address(user_vaddr)
    assert user_decomp["is_canonical"] is True
    assert user_decomp["offset"] == 0x234

    # Higher canonical address (Kernel mode)
    kernel_vaddr = 0xFFFF800012345678
    kernel_decomp = PageFrameAddressMath.decompose_virtual_address(kernel_vaddr)
    assert kernel_decomp["is_canonical"] is True
    assert kernel_decomp["pml4_index"] == 256
    assert kernel_decomp["pdp_index"] == 0
    assert kernel_decomp["pd_index"] == 145
    assert kernel_decomp["pt_index"] == 325
    assert kernel_decomp["offset"] == 0x678

    # Non-canonical hole address
    non_canonical = 0x0000800000000000
    assert PageFrameAddressMath.is_canonical(non_canonical) is False
    non_canon_decomp = PageFrameAddressMath.decompose_virtual_address(non_canonical)
    assert non_canon_decomp["is_canonical"] is False


def test_memory_allocation_strategy_comparator():
    """Verify PMM algorithm comparison metrics and bitmap overhead."""
    overhead_4g = MemoryAllocationStrategyComparator.calculate_bitmap_overhead(4 * (1024**3))
    assert overhead_4g["ram_bytes"] == 4294967296
    assert overhead_4g["total_pages"] == 1048576
    assert overhead_4g["bitmap_bytes"] == 131072  # 128 KiB
    assert overhead_4g["bitmap_kib"] == 128.0
    assert abs(overhead_4g["overhead_percent"] - 0.0030517578) < 1e-6

    strategies = MemoryAllocationStrategyComparator.compare_strategies()
    assert "Bitmap" in strategies
    assert "FreeList" in strategies
    assert "BuddySystem" in strategies
    assert "0.003%" in strategies["Bitmap"]["space_overhead"]
    assert "O(1)" in strategies["Bitmap"]["free_time"]


def test_physical_vs_virtual_memory_simulator_normal_mapping():
    """Verify normal page mapping and MMU address translation."""
    sim = PhysicalVsVirtualMemorySim(total_physical_pages=64)
    pfn0 = sim.allocate_physical_frame()
    assert pfn0 == 0
    pfn1 = sim.allocate_physical_frame()
    assert pfn1 == 1

    sim.map_page(0x1000, pfn0, is_user_space=False, writable=True, user_accessible=False)
    sim.map_page(0x2000, pfn1, is_user_space=False, writable=False, user_accessible=False)

    # Read/Write to writable kernel page
    phys0 = sim.translate(0x1050, is_user_mode=False, is_write=True)
    assert phys0 == 0x0050

    # Read from read-only page
    phys1 = sim.translate(0x2100, is_user_mode=False, is_write=False)
    assert phys1 == 0x1100

    # Free frame
    sim.free_physical_frame(pfn0)
    assert pfn0 not in sim.allocated_physical_frames


def test_physical_vs_virtual_memory_simulator_page_faults():
    """Verify CPU hardware protection checks and #PF fault dispatching."""
    sim = PhysicalVsVirtualMemorySim(total_physical_pages=32)
    pfn = sim.allocate_physical_frame()

    # Kernel-only, read-only, non-executable page
    sim.map_page(0x4000, pfn, is_user_space=False, writable=False, user_accessible=False, executable=False)

    # 1. Unmapped address -> NOT_PRESENT
    with pytest.raises(PageFaultException) as exc_info:
        sim.translate(0x5000, is_user_mode=False)
    assert exc_info.value.fault_type == PageFaultType.NOT_PRESENT

    # 2. User mode access -> USER_ACCESS_VIOLATION
    with pytest.raises(PageFaultException) as exc_info:
        sim.translate(0x4010, is_user_mode=True)
    assert exc_info.value.fault_type == PageFaultType.USER_ACCESS_VIOLATION

    # 3. Write to read-only -> WRITE_PROTECTION
    with pytest.raises(PageFaultException) as exc_info:
        sim.translate(0x4020, is_user_mode=False, is_write=True)
    assert exc_info.value.fault_type == PageFaultType.WRITE_PROTECTION

    # 4. Instruction fetch on NX page -> INSTRUCTION_FETCH_VIOLATION
    with pytest.raises(PageFaultException) as exc_info:
        sim.translate(0x4030, is_user_mode=False, is_exec=True)
    assert exc_info.value.fault_type == PageFaultType.INSTRUCTION_FETCH_VIOLATION


def test_figure_generation_outputs():
    """Verify high-resolution figure outputs (DPI 300)."""
    fig1 = plot_fig8_01_physical_vs_virtual_memory_architecture()
    assert Path(fig1).exists()
    assert Path(fig1).stat().st_size > 50000
    with Image.open(fig1) as img:
        assert img.size[0] >= 3000
        assert img.size[1] >= 2000

    fig2 = plot_fig8_02_memory_management_data_structures_comparison()
    assert Path(fig2).exists()
    assert Path(fig2).stat().st_size > 50000
    with Image.open(fig2) as img:
        assert img.size[0] >= 3000
        assert img.size[1] >= 2000

    fig3 = plot_fig8_03_chapter8_memory_subsystem_roadmap()
    assert Path(fig3).exists()
    assert Path(fig3).stat().st_size > 50000
    with Image.open(fig3) as img:
        assert img.size[0] >= 3000
        assert img.size[1] >= 1800


def test_cpp20_memory_role_sim_binary():
    """Verify standalone C++20 simulation executable exits with 0 and passes assertions."""
    bin_path = Path("src/ch08/memory_role_sim")
    if not bin_path.exists():
        compile_cmd = [
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            "src/ch08/memory_role_sim.cpp", "-o", str(bin_path)
        ]
        res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "[PASS] 4 KiB Page Frame arithmetic verified." in run_res.stdout
    assert "[PASS] Canonical 48-bit virtual address decomposition verified." in run_res.stdout
    assert "[PASS] Bitmap Memory Manager allocation and free verified." in run_res.stdout
    assert "✅ All Chapter 8.1 C++20 memory role verifications passed successfully!" in run_res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify 8/8.1_Role_of_OS_Memory_Management.ipynb exists and is valid JSON."""
    nb_path = Path("8/8.1_Role_of_OS_Memory_Management.ipynb")
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
