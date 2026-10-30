"""Tests for Chapter 2.9: Struct Memory Layout, Arrow Operator, and Double Pointers."""

import os
import subprocess
import pytest
from common import (
    StructMember,
    StructMemoryLayout,
    DoublePointerSimulator,
    plot_figure_2_17_struct_layout_and_arrow_operator,
    plot_figure_2_18_double_pointer_and_uefi_interface,
)


def test_struct_memory_layout_alignment_and_padding():
    """Verifies member offset, alignment padding, and total size calculation."""
    layout = StructMemoryLayout("SampleStruct", [
        ("a", "uint32_t"),  # offset 0, size 4
        ("b", "uint8_t"),   # offset 4, size 1
        ("c", "uint16_t"),  # offset 6 (pad 1B), size 2
        ("d", "uint64_t"),  # offset 8, size 8
    ])

    assert layout.total_size == 16
    assert layout.struct_alignment == 8

    m_a = layout.get_member("a")
    assert m_a.offset == 0
    assert m_a.size == 4
    assert m_a.padding_before == 0

    m_b = layout.get_member("b")
    assert m_b.offset == 4
    assert m_b.size == 1
    assert m_b.padding_before == 0

    m_c = layout.get_member("c")
    assert m_c.offset == 6
    assert m_c.size == 2
    assert m_c.padding_before == 1

    m_d = layout.get_member("d")
    assert m_d.offset == 8
    assert m_d.size == 8
    assert m_d.padding_before == 0


def test_struct_memory_map_layout():
    """Verifies MikanOS Chapter 2 MemoryMap struct layout."""
    layout = StructMemoryLayout("MemoryMap", [
        ("buffer_size", "UINTN"),          # offset 0, size 8
        ("buffer", "void*"),               # offset 8, size 8
        ("map_size", "UINTN"),             # offset 16, size 8
        ("map_key", "UINTN"),              # offset 24, size 8
        ("descriptor_size", "UINTN"),      # offset 32, size 8
        ("descriptor_version", "UINT32"),  # offset 40, size 4
    ])

    assert layout.total_size == 48
    assert layout.struct_alignment == 8

    eff_addr, offset, asm = layout.simulate_arrow_operator(0x1000, "buffer")
    assert offset == 8
    assert eff_addr == 0x1008
    assert "MOV RAX, [RDI + 8]" in asm


def test_double_pointer_simulator_allocation_and_dereference():
    """Verifies allocation and single/double dereferencing of double pointers."""
    sim = DoublePointerSimulator(stack_base=0x7FFF_0020, heap_base=0x1000_5000)

    # Allocate target object on heap
    obj_addr = sim.allocate_target_object("file_obj", size=64)
    assert obj_addr == 0x1000_5000

    # Allocate pointer variable on stack initialized to nullptr
    ptr_addr = sim.allocate_local_pointer("root_dir", target_address=0)
    assert ptr_addr == 0x7FFF_0020
    assert sim.dereference_pointer("root_dir") == 0

    # Allocate double pointer variable on stack initialized to &root_dir
    pp_addr = sim.allocate_double_pointer("pp_root_dir", "root_dir")
    assert pp_addr == 0x7FFF_0028
    assert sim.dereference_double_pointer_level1("pp_root_dir") == ptr_addr
    assert sim.dereference_double_pointer_level2("pp_root_dir") == 0


def test_double_pointer_pass_by_value_failure():
    """Verifies that pass-by-value fails to modify caller's pointer."""
    sim = DoublePointerSimulator()
    obj_addr = sim.allocate_target_object("new_file", size=32)
    sim.allocate_local_pointer("file_ptr", target_address=0)

    result = sim.simulate_pass_by_value_failure("file_ptr", obj_addr)
    assert result["success"] is False
    assert result["is_caller_updated"] is False
    assert result["caller_ptr_value_before"] == 0
    assert result["caller_ptr_value_after"] == 0
    assert sim.dereference_pointer("file_ptr") == 0


def test_double_pointer_pass_by_double_pointer_success():
    """Verifies UEFI output parameter pattern: callee modifies caller's pointer."""
    sim = DoublePointerSimulator()
    obj_addr = sim.allocate_target_object("root_protocol", size=48)
    sim.allocate_local_pointer("root_dir", target_address=0)

    result = sim.simulate_pass_by_double_pointer_success("root_dir", obj_addr)
    assert result["success"] is True
    assert result["is_caller_updated"] is True
    assert result["caller_ptr_value_before"] == 0
    assert result["caller_ptr_value_after"] == obj_addr
    assert sim.dereference_pointer("root_dir") == obj_addr


def test_cpp20_arrow_operator_sim_binary():
    """Executes the standalone C++20 simulation binary."""
    binary_path = "./build/bin/ch02_arrow_operator_sim"
    if not os.path.exists(binary_path):
        subprocess.run(["cmake", "--build", "build", "--target", "ch02_arrow_operator_sim"], check=True)

    result = subprocess.run([binary_path, "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "[PASS] All ArrowOperatorSimulator tests passed successfully." in result.stdout


def test_figure_generation_outputs(tmp_path):
    """Verifies generation of Figures 2.17 and 2.18."""
    fig17_path = str(tmp_path / "fig2_17_test.png")
    fig18_path = str(tmp_path / "fig2_18_test.png")

    out17 = plot_figure_2_17_struct_layout_and_arrow_operator(fig17_path)
    assert os.path.exists(out17)
    assert os.path.getsize(out17) > 50_000

    out18 = plot_figure_2_18_double_pointer_and_uefi_interface(fig18_path)
    assert os.path.exists(out18)
    assert os.path.getsize(out18) > 50_000
