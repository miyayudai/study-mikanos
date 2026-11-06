"""Tests for Chapter 8.6: Identity Paging & 4-Level Paging Architecture."""

import os
import subprocess
import json
import pytest
from common.os_sim import (
    LinearAddress4LevelSim,
    PageTableEntrySim,
    IdentityPagingSimulator,
    plot_fig8_16_x86_64_4level_paging_architecture,
    plot_fig8_17_identity_mapping_2mib_huge_pages,
    plot_fig8_18_cr3_and_address_translation_flow,
)

def test_linear_address_4level_decomposition():
    """Verify 48-bit canonical linear address decomposition into 4-level paging indexes."""
    # Test address: 0x0000_0001_8020_1234
    # Hex breakdown:
    #   [11:0]  Offset = 0x234 (12 bits)
    #   [20:12] Page   = 0x001 (9 bits)
    #   [29:21] Dir    = 0x001 (9 bits)
    #   [38:30] PDP    = 0x006 (9 bits)
    #   [47:39] PML4   = 0x000 (9 bits)
    #   [63:48] Sign   = 0x0000
    addr_val = 0x0000_0001_8020_1234
    addr = LinearAddress4LevelSim(addr_val)
    assert addr.offset == 0x234
    assert addr.page == 0x001
    assert addr.dir == 0x001
    assert addr.pdp == 0x006
    assert addr.pml4 == 0x000
    assert addr.is_canonical is True
    assert addr.offset_2mb == (addr_val & 0x1FFFFF)
    assert addr.part(0) == 0x234
    assert addr.part(1) == 0x001
    assert addr.part(2) == 0x001
    assert addr.part(3) == 0x006
    assert addr.part(4) == 0x000

    # Negative canonical address (Higher Half, e.g., kernel space 0xFFFF_8000_0000_0000)
    high_addr_val = 0xFFFF_8000_0000_0000
    high_addr = LinearAddress4LevelSim(high_addr_val)
    assert high_addr.pml4 == 256
    assert high_addr.is_canonical is True

    # Non-canonical address (bit 47 is 0 but upper bits are non-zero)
    non_canon_val = 0x0001_0000_0000_0000
    non_canon_addr = LinearAddress4LevelSim(non_canon_val)
    assert non_canon_addr.is_canonical is False


def test_page_table_entry_bit_packing():
    """Verify bitfield packing and unpacking of 64-bit Page Table Entries."""
    # 1. Directory entry pointing to next level: 0x003 (Present | Writable)
    pdp_addr = 0x1000_1000
    pml4e = PageTableEntrySim.create(pdp_addr, present=True, writable=True, huge_page=False)
    assert pml4e.present == 1
    assert pml4e.writable == 1
    assert pml4e.user == 0
    assert pml4e.huge_page == 0
    assert pml4e.physical_frame_address == pdp_addr
    assert (pml4e.data & 0xFFF) == 0x003

    # 2. 2MiB Huge Page entry: 0x083 (Present | Writable | HugePage)
    phys_frame = 0x4000_0000  # 1GiB
    pde = PageTableEntrySim.create(phys_frame, present=True, writable=True, huge_page=True)
    assert pde.present == 1
    assert pde.writable == 1
    assert pde.huge_page == 1
    assert pde.huge_page_base_address == phys_frame
    assert (pde.data & 0xFFF) == 0x083


def test_identity_paging_simulator_table_structure():
    """Verify IdentityPagingSimulator internal arrays and pointer structure."""
    sim = IdentityPagingSimulator(pml4_base=0x1000_0000, pdp_base=0x1000_1000, pd_start_base=0x1000_2000)

    # Alignments
    assert sim.verify_table_alignments() is True
    assert sim.cr3 == 0x1000_0000

    # PML4[0] points to PDP table with 0x003
    pml4_0 = PageTableEntrySim(sim.pml4_table[0])
    assert pml4_0.present == 1
    assert pml4_0.writable == 1
    assert pml4_0.physical_frame_address == 0x1000_1000

    # Entries 1..511 in PML4 must be 0
    assert all(entry == 0 for entry in sim.pml4_table[1:])

    # 64 PDP entries point to corresponding PDs
    for i in range(64):
        pdpe = PageTableEntrySim(sim.pdp_table[i])
        assert pdpe.present == 1
        assert pdpe.writable == 1
        assert pdpe.physical_frame_address == sim.pd_bases[i]

    # Entries 64..511 in PDP must be 0
    assert all(entry == 0 for entry in sim.pdp_table[64:])

    # Each PD has 512 entries with 0x083 and continuous 2MiB steps
    for i_pdpt in range(64):
        for i_pd in range(512):
            raw = sim.page_directory[i_pdpt][i_pd]
            assert (raw & 0xFFF) == 0x083
            expected_phys = i_pdpt * sim.PAGE_SIZE_1G + i_pd * sim.PAGE_SIZE_2M
            pde = PageTableEntrySim(raw)
            assert pde.huge_page_base_address == expected_phys


def test_identity_paging_translation_and_bounds():
    """Verify address translation matches Virtual == Physical across 64GiB space."""
    sim = IdentityPagingSimulator()

    test_addresses = [
        0x00000000,           # Start of physical memory
        0x00100000,           # 1MiB (Kernel load)
        0x001FFFFF,           # End of 1st 2MiB frame
        0x00200000,           # Start of 2nd 2MiB frame
        0x3FE00000,           # 1022MiB
        0x40000000,           # 1GiB (PDP[1])
        0x80000000,           # 2GiB (PDP[2])
        0x100000000,          # 4GiB (PDP[4])
        0xF00000000,          # 60GiB (PDP[60])
        0xFFFE00000,          # 64GiB - 2MiB
        0xFFFFFFFFF,          # 64GiB - 1
    ]

    for addr in test_addresses:
        res = sim.translate(addr)
        assert res["success"] is True
        assert res["is_identity"] is True
        assert res["physical_address"] == addr

    # Beyond 64GiB mapped capacity
    beyond_addr = 0x1000000000  # 64GiB
    res_beyond = sim.translate(beyond_addr)
    assert res_beyond["success"] is False
    assert res_beyond["error"] == "PageFaultBeyondMappedLimit"

    # Non-canonical address
    res_non_canon = sim.translate(0x0001000000000000)
    assert res_non_canon["success"] is False
    assert res_non_canon["error"] == "NonCanonicalAddress"


def test_memory_overhead_savings_calculation():
    """Verify calculation of 4KiB page overhead vs 2MiB huge page savings."""
    sim = IdentityPagingSimulator()
    savings = sim.calculate_memory_savings()
    assert savings["mapped_capacity_gib"] == 64
    assert savings["4k_page_table_overhead_mib"] > 128.0
    assert savings["2m_huge_page_overhead_kib"] < 300.0
    assert savings["memory_reduction_ratio"] > 400.0


def test_figure_generation_outputs(tmp_path):
    """Verify generation of Figure 8.16, 8.17, and 8.18."""
    fig16_path = str(tmp_path / "test_fig8_16.png")
    fig17_path = str(tmp_path / "test_fig8_17.png")
    fig18_path = str(tmp_path / "test_fig8_18.png")

    out16 = plot_fig8_16_x86_64_4level_paging_architecture(fig16_path)
    out17 = plot_fig8_17_identity_mapping_2mib_huge_pages(fig17_path)
    out18 = plot_fig8_18_cr3_and_address_translation_flow(fig18_path)

    assert os.path.exists(out16)
    assert os.path.getsize(out16) > 50000

    assert os.path.exists(out17)
    assert os.path.getsize(out17) > 50000

    assert os.path.exists(out18)
    assert os.path.getsize(out18) > 50000


def test_cpp20_identity_paging_sim_standalone():
    """Verify the standalone C++20 simulator binary executes without error."""
    bin_path = os.path.abspath("build/bin/ch08_identity_paging_sim")
    if not os.path.exists(bin_path):
        subprocess.run(["cmake", "--build", "build", "--target", "ch08_identity_paging_sim"], check=True)

    result = subprocess.run([bin_path, "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "All C++20 Verification Unit Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 8.6 notebook exists and contains executed cells with no errors."""
    nb_path = "8/8.6_Setting_Up_Identity_Paging.ipynb"
    assert os.path.exists(nb_path), f"Notebook file {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, "Notebook should have at least 10 rich content cells"

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"Cell {idx} failed with error: {out.get('ename')}: {out.get('evalue')}"
                )
