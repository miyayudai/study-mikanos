"""Tests for Chapter 2.8: Pointers, Addressing, and Memory Model."""

import os
import subprocess
from pathlib import Path
import pytest

from common import (
    MemoryCellModel,
    PointerModelSimulator,
    PointerVariable,
    plot_figure_2_15_pointer_memory_model,
    plot_figure_2_16_pointer_arithmetic_scaling,
)


def test_memory_cell_model_read_write_little_endian():
    """Verify byte-level read/write and little-endian representation."""
    mem = MemoryCellModel(base_address=0x1000, size=256)

    # Write 32-bit integer: 0x12345678
    mem.write_uint32(0x1000, 0x12345678)
    assert mem.read_uint32(0x1000) == 0x12345678

    # In little-endian, byte[0] is 0x78, byte[1] is 0x56, byte[2] is 0x34, byte[3] is 0x12
    assert mem.read_byte(0x1000) == 0x78
    assert mem.read_byte(0x1001) == 0x56
    assert mem.read_byte(0x1002) == 0x34
    assert mem.read_byte(0x1003) == 0x12

    # Write 64-bit integer
    mem.write_uint64(0x1010, 0xAABBCCDDEEFF0011)
    assert mem.read_uint64(0x1010) == 0xAABBCCDDEEFF0011
    assert mem.read_byte(0x1010) == 0x11
    assert mem.read_byte(0x1017) == 0xAA


def test_memory_cell_model_bounds_checking():
    """Verify bounds checking on invalid memory access."""
    mem = MemoryCellModel(base_address=0x1000, size=64)

    # Underflow
    with pytest.raises(IndexError):
        mem.read_byte(0x0FFF)

    # Overflow
    with pytest.raises(IndexError):
        mem.read_byte(0x1040)

    with pytest.raises(IndexError):
        mem.read_uint32(0x103E)  # 2 bytes overflow


def test_pointer_model_variable_allocation_and_pointer_size():
    """Verify 64-bit architecture invariant: all pointers are 8 bytes."""
    sim = PointerModelSimulator(base_address=0x1000, size=4096)

    v_char = sim.allocate_variable("c", "char", 65)
    v_int = sim.allocate_variable("n", "int", 42)
    v_qword = sim.allocate_variable("q", "uint64", 0x100000000)
    v_desc = sim.allocate_variable("desc", "EFI_MEMORY_DESCRIPTOR", b"\x00" * 48)

    assert v_char.size == 1
    assert v_int.size == 4
    assert v_qword.size == 8
    assert v_desc.size == 48

    p_char = sim.create_pointer("p_c", "c")
    p_int = sim.create_pointer("p_n", "n")
    p_qword = sim.create_pointer("p_q", "q")
    p_desc = sim.create_pointer("p_desc", "desc")

    # In 64-bit systems, all pointers are exactly 8 bytes regardless of target type
    assert p_char.pointer_size == 8
    assert p_int.pointer_size == 8
    assert p_qword.pointer_size == 8
    assert p_desc.pointer_size == 8


def test_pointer_dereference_read_and_store():
    """Verify dereferencing *p reads and mutates target variable memory."""
    sim = PointerModelSimulator(base_address=0x2000, size=4096)

    sim.allocate_variable("target", "int", 12345, address=0x2100)
    sim.create_pointer("ptr", "target", ptr_address=0x2800)

    # Read via pointer: *ptr == 12345
    assert sim.dereference("ptr") == 12345

    # Store via pointer: *ptr = 99999
    sim.assign_via_pointer("ptr", 99999)
    assert sim.dereference("ptr") == 99999
    assert sim.variables["target"].value == 99999
    assert sim.memory.read_int32(0x2100) == 99999


def test_null_pointer_dereference_exception():
    """Verify dereference on nullptr (0x0) raises appropriate error."""
    mem = MemoryCellModel(base_address=0x1000, size=256)
    null_ptr = PointerVariable(
        name="p_null",
        target_type="int",
        target_type_size=4,
        pointer_addr=0x1000,
        target_addr=0x0,
    )

    with pytest.raises(ValueError, match="Null pointer dereference"):
        null_ptr.dereference(mem)

    with pytest.raises(ValueError, match="Null pointer write"):
        null_ptr.store(mem, 42)


def test_pointer_arithmetic_scaling_by_type():
    """Verify Address(p + n) == Address(p) + n * sizeof(*p)."""
    sim = PointerModelSimulator(base_address=0x1000, size=4096)

    sim.allocate_variable("char_arr", "char", 0, address=0x1000)
    sim.allocate_variable("int_arr", "int", 0, address=0x1000)
    sim.allocate_variable("desc_arr", "EFI_MEMORY_DESCRIPTOR", b"\x00" * 48, address=0x1000)

    p_char = sim.create_pointer("p_c", "char_arr")
    p_int = sim.create_pointer("p_i", "int_arr")
    p_desc = sim.create_pointer("p_desc", "desc_arr")

    # char* step = 1
    assert sim.pointer_arithmetic("p_c", 1) == 0x1000 + 1 * 1
    assert sim.pointer_arithmetic("p_c", 5) == 0x1000 + 5 * 1

    # int* step = 4
    assert sim.pointer_arithmetic("p_i", 1) == 0x1000 + 1 * 4
    assert sim.pointer_arithmetic("p_i", 3) == 0x1000 + 3 * 4

    # EFI_MEMORY_DESCRIPTOR* step = 48
    assert sim.pointer_arithmetic("p_desc", 1) == 0x1000 + 1 * 48
    assert sim.pointer_arithmetic("p_desc", 2) == 0x1000 + 2 * 48


def test_uefi_descriptor_traversal_casting_math():
    """Verify the difference between direct descriptor pointer math and byte/uintptr_t stride."""
    base_addr = 0x100000
    descriptor_size = 48  # Runtime descriptor size reported by UEFI

    # Scenario A: Wrong direct pointer addition
    # desc_ptr + descriptor_size advances by descriptor_size * sizeof(desc)
    wrong_next_addr = base_addr + descriptor_size * 48
    assert wrong_next_addr == base_addr + 2304

    # Scenario B: Correct uintptr_t or byte pointer addition
    correct_next_addr = base_addr + descriptor_size
    assert correct_next_addr == base_addr + 48


def test_cpp20_pointer_basics_sim_binary():
    """Verify the standalone C++20 pointer basics binary compiles and passes all checks."""
    proj_root = Path(__file__).resolve().parent.parent
    bin_path = proj_root / "build" / "bin" / "ch02_pointer_basics_sim"
    assert bin_path.exists(), f"Binary {bin_path} not found. Please build CMake targets."

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0, f"Binary failed with error: {res.stderr}"
    assert "ALL 7 POINTER ARCHITECTURE VERIFICATIONS PASSED" in res.stdout


def test_figure_generation_outputs(tmp_path):
    """Verify figure 2.15 and 2.16 generation produces valid high-resolution PNGs."""
    fig1 = tmp_path / "test_fig2_15.png"
    fig2 = tmp_path / "test_fig2_16.png"

    plot_figure_2_15_pointer_memory_model(str(fig1))
    plot_figure_2_16_pointer_arithmetic_scaling(str(fig2))

    assert fig1.exists()
    assert fig1.stat().st_size > 50_000

    assert fig2.exists()
    assert fig2.stat().st_size > 50_000
