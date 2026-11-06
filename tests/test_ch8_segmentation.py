"""Tests for Chapter 8.5: Global Descriptor Table (GDT) & x86-64 Segmentation."""

import os
import subprocess
import json
import pytest
from common.os_sim import (
    SegmentDescriptorSim,
    SegmentSelectorSim,
    GdtSimulator,
    plot_fig8_13_gdt_descriptor_bit_layout,
    plot_fig8_14_gdt_table_and_gdtr_architecture,
    plot_fig8_15_cs_switch_via_iretq_mechanism,
)

def test_segment_descriptor_bit_packing():
    """Verify bitfield packing and unpacking of 64-bit SegmentDescriptor."""
    # 1. Null descriptor test
    null_desc = SegmentDescriptorSim.create_null_descriptor()
    assert null_desc.to_uint64() == 0

    # 2. Kernel Code Segment test
    k_desc = SegmentDescriptorSim.create_code_segment(dpl=0, base=0, limit=0xFFFFF, type_val=10)
    assert k_desc.long_mode == 1
    assert k_desc.default_op_size == 0
    assert k_desc.granularity == 1
    assert k_desc.present == 1
    assert k_desc.system_segment == 1
    assert k_desc.dpl == 0
    assert k_desc.type_val == 10
    assert k_desc.base_address == 0
    assert k_desc.raw_limit == 0xFFFFF
    assert k_desc.effective_limit == 0xFFFFFFFF  # 4GiB

    # Roundtrip check
    k_val = k_desc.to_uint64()
    decoded = SegmentDescriptorSim.from_uint64(k_val)
    assert decoded.to_uint64() == k_val
    assert decoded.dpl == 0
    assert decoded.long_mode == 1

    # 3. User Code Segment test
    u_desc = SegmentDescriptorSim.create_code_segment(dpl=3, base=0, limit=0xFFFFF, type_val=10)
    assert u_desc.dpl == 3
    u_val = u_desc.to_uint64()
    u_decoded = SegmentDescriptorSim.from_uint64(u_val)
    assert u_decoded.dpl == 3
    assert u_decoded.long_mode == 1


def test_segment_selector_decomposition():
    """Verify 16-bit Segment Selector bitfields (Index, TI, RPL)."""
    # Kernel CS selector: index 1, RPL 0 -> (1 << 3) | 0 = 0x08
    sel_k = SegmentSelectorSim.from_uint16(0x08)
    assert sel_k.index == 1
    assert sel_k.ti == 0
    assert sel_k.rpl == 0
    assert sel_k.to_uint16() == 0x08

    # User CS selector: index 2, RPL 3 -> (2 << 3) | 3 = 19 = 0x13
    sel_u = SegmentSelectorSim.from_uint16(0x13)
    assert sel_u.index == 2
    assert sel_u.ti == 0
    assert sel_u.rpl == 3
    assert sel_u.to_uint16() == 0x13

    # Null selector
    sel_null = SegmentSelectorSim.from_uint16(0x00)
    assert sel_null.index == 0
    assert sel_null.ti == 0
    assert sel_null.rpl == 0


def test_gdt_simulator_validation():
    """Verify GdtSimulator table structure and selector validation."""
    sim = GdtSimulator()
    assert len(sim.descriptors) == 3
    assert sim.gdtr_limit == 3 * 8 - 1  # 23 bytes

    # Kernel CS selector check
    valid_k, msg_k = sim.validate_code_selector(0x08)
    assert valid_k is True

    # User CS selector check
    valid_u, msg_u = sim.validate_code_selector(0x13)
    assert valid_u is True

    # Out-of-bounds selector check
    valid_inv, msg_inv = sim.validate_code_selector(0x50)  # index 10
    assert valid_inv is False
    assert "exceeds GDT limit" in msg_inv

    # Null selector cannot be used as code segment
    valid_null, msg_null = sim.validate_code_selector(0x00)
    assert valid_null is False
    assert "not present" in msg_null


def test_figure_generation_outputs(tmp_path):
    """Verify generation of Figure 8.13, 8.14, and 8.15."""
    fig13_path = str(tmp_path / "test_fig8_13.png")
    fig14_path = str(tmp_path / "test_fig8_14.png")
    fig15_path = str(tmp_path / "test_fig8_15.png")

    out13 = plot_fig8_13_gdt_descriptor_bit_layout(fig13_path)
    out14 = plot_fig8_14_gdt_table_and_gdtr_architecture(fig14_path)
    out15 = plot_fig8_15_cs_switch_via_iretq_mechanism(fig15_path)

    assert os.path.exists(out13)
    assert os.path.getsize(out13) > 50000

    assert os.path.exists(out14)
    assert os.path.getsize(out14) > 50000

    assert os.path.exists(out15)
    assert os.path.getsize(out15) > 50000


def test_cpp20_gdt_segmentation_sim_standalone():
    """Verify the standalone C++20 simulator binary executes without error."""
    bin_path = os.path.abspath("build/bin/ch08_gdt_segmentation_sim")
    if not os.path.exists(bin_path):
        subprocess.run(["cmake", "--build", "build", "--target", "ch08_gdt_segmentation_sim"], check=True)

    result = subprocess.run([bin_path, "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "All C++20 Verification Unit Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 8.5 notebook exists and contains executed cells with no errors."""
    nb_path = "8/8.5_Setting_Up_GDT_Segmentation.ipynb"
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
