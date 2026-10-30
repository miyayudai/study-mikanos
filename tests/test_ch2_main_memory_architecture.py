"""Unit tests for Chapter 2.3: Main Memory Architecture & MMIO Simulation."""

import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common.os_sim import (
    MemoryRegionType,
    PhysicalMemoryRegion,
    PhysicalAddressSpace,
    plot_physical_address_space_and_mmio,
    plot_x86_legacy_and_modern_memory_layout_comparison,
)


def test_physical_memory_region_properties():
    """Verify memory region math, page count, and page alignment."""
    # 4KB aligned region
    r1 = PhysicalMemoryRegion(
        base_addr=0x1000,
        length_bytes=0x3000,
        region_type=MemoryRegionType.USABLE_RAM,
        description="Aligned RAM",
    )
    assert r1.end_addr == 0x4000
    assert r1.page_count == 3
    assert r1.is_page_aligned
    assert r1.is_usable_ram
    assert r1.contains_address(0x1000)
    assert r1.contains_address(0x3FFF)
    assert not r1.contains_address(0x4000)
    assert not r1.contains_address(0x0FFF)

    # Non-page aligned region
    r2 = PhysicalMemoryRegion(
        base_addr=0x1004,
        length_bytes=500,
        region_type=MemoryRegionType.RESERVED,
        description="Unaligned Reserved",
    )
    assert not r2.is_page_aligned
    assert r2.page_count == 1  # 500 bytes requires 1 page
    assert not r2.is_usable_ram


def test_region_overlap_detection():
    """Verify mutual exclusion and overlap algorithms between address regions."""
    r1 = PhysicalMemoryRegion(0x1000, 0x1000, MemoryRegionType.USABLE_RAM, "R1")  # [0x1000, 0x2000)
    r2 = PhysicalMemoryRegion(0x2000, 0x1000, MemoryRegionType.USABLE_RAM, "R2")  # [0x2000, 0x3000)
    r3 = PhysicalMemoryRegion(0x1800, 0x1000, MemoryRegionType.USABLE_RAM, "R3")  # [0x1800, 0x2800)

    assert not r1.overlaps_with(r2)
    assert not r2.overlaps_with(r1)
    assert r1.overlaps_with(r3)
    assert r3.overlaps_with(r1)
    assert r2.overlaps_with(r3)
    assert r3.overlaps_with(r2)


def test_physical_address_space_operations():
    """Verify address space registration, collision prevention, and address resolution."""
    pas = PhysicalAddressSpace(address_width_bits=48)
    ram = PhysicalMemoryRegion(0x0010_0000, 0x1000_0000, MemoryRegionType.USABLE_RAM, "Conventional RAM")
    mmio = PhysicalMemoryRegion(0xFEE0_0000, 0x1000, MemoryRegionType.MMIO, "Local APIC", is_cacheable=False)

    pas.add_region(ram)
    pas.add_region(mmio)

    # Resolving valid addresses
    res_ram = pas.resolve_address(0x0010_5000)
    assert res_ram is not None
    assert res_ram.region_type == MemoryRegionType.USABLE_RAM

    res_mmio = pas.resolve_address(0xFEE0_0020)
    assert res_mmio is not None
    assert res_mmio.region_type == MemoryRegionType.MMIO
    assert not res_mmio.is_cacheable

    # Resolving unmapped address (hole)
    assert pas.resolve_address(0x5000_0000) is None

    # Collision rejection
    conflict_reg = PhysicalMemoryRegion(0x0015_0000, 0x1000, MemoryRegionType.RESERVED, "Conflict")
    with pytest.raises(ValueError, match="Address collision"):
        pas.add_region(conflict_reg)


def test_memory_holes_and_free_ram_search():
    """Verify memory gap detection and aligned free RAM allocation simulation."""
    pas = PhysicalAddressSpace(address_width_bits=32)
    pas.add_region(PhysicalMemoryRegion(0x0000, 0x1000, MemoryRegionType.RESERVED, "Low Reserved"))
    pas.add_region(PhysicalMemoryRegion(0x1000, 0x10000, MemoryRegionType.USABLE_RAM, "Low RAM"))
    pas.add_region(PhysicalMemoryRegion(0x80000, 0x20000, MemoryRegionType.USABLE_RAM, "High RAM"))

    holes = pas.detect_memory_holes()
    assert len(holes) == 1
    # First hole between Low RAM end (0x11000) and High RAM start (0x80000)
    assert holes[0] == (0x11000, 0x80000, 0x6F000)

    # Find continuous RAM chunk
    found = pas.find_free_ram_chunk(size_bytes=0x8000, alignment=4096)
    assert found is not None
    assert found % 4096 == 0
    assert found == 0x1000

    # Request larger than any single block
    assert pas.find_free_ram_chunk(size_bytes=0x100000) is None


def test_standard_pc_and_legacy_models():
    """Verify built-in QEMU standard PC and legacy 1MB PC simulation models."""
    qemu_pc = PhysicalAddressSpace.simulate_standard_pc_4gb()
    summary = qemu_pc.get_memory_summary()

    # Usable DRAM is ~4GB (minus UEFI services and low reserved areas, typical for real hardware)
    assert summary["total_usable_ram_bytes"] >= int(3.9 * 1024 * 1024 * 1024)
    assert summary["total_mapped_bytes"] >= (4 * 1024 * 1024 * 1024)
    assert summary["total_mmio_bytes"] > 0  # MMIO space allocated
    assert summary["total_firmware_bytes"] > 0
    assert len(qemu_pc.detect_memory_holes()) > 0

    legacy_pc = PhysicalAddressSpace.simulate_legacy_1mb_layout()
    leg_sum = legacy_pc.get_memory_summary()
    assert leg_sum["total_mapped_bytes"] == 0x100000  # Exactly 1MB
    assert leg_sum["total_usable_ram_bytes"] > 500 * 1024  # >500KB DOS Conventional RAM


def test_cpp20_main_memory_sim_standalone():
    """Verify standalone C++20 simulator builds and passes self-tests."""
    bin_path = Path("build/bin/ch02_main_memory_architecture_sim")
    assert bin_path.exists(), f"Binary {bin_path} does not exist. Run cmake --build build first."

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0, f"C++ self-tests failed:\n{res.stderr}"
    assert "All self-tests passed successfully!" in res.stdout


def test_memory_architecture_figures_generation(tmp_path):
    """Verify generation of Figure 2.5 and Figure 2.6 with valid dimensions and resolution."""
    p1 = plot_physical_address_space_and_mmio(str(tmp_path / "test_fig2_05.png"))
    p2 = plot_x86_legacy_and_modern_memory_layout_comparison(str(tmp_path / "test_fig2_06.png"))

    assert Path(p1).exists()
    assert Path(p2).exists()

    with Image.open(p1) as img1:
        assert img1.width >= 3000
        assert img1.height >= 1800

    with Image.open(p2) as img2:
        assert img2.width >= 3000
        assert img2.height >= 1800
