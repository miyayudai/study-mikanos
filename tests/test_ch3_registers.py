"""Tests for Chapter 3.2: x86-64 Architecture Registers, Sub-Registers, and ABI Infrastructure."""

import os
import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common import (
    CallingConventionSimulator,
    ControlRegisters,
    RFlagsRegister,
    X86_64RegisterFile,
    plot_rflags_and_control_registers,
    plot_x86_64_register_hierarchy,
)


def test_register_file_32bit_zero_extension_rule():
    """Verify that writing to a 32-bit sub-register clears the upper 32 bits (Zero Extension)."""
    rf = X86_64RegisterFile()

    # Fill RAX with all 1s
    rf.write_64("rax", 0xFFFFFFFFFFFFFFFF)
    assert rf.read_64("rax") == 0xFFFFFFFFFFFFFFFF

    # Write 32-bit: EAX
    rf.write_32("eax", 0x12345678)
    assert rf.read_32("eax") == 0x12345678
    # CRITICAL: Upper 32 bits MUST BE 0!
    assert rf.read_64("rax") == 0x0000000012345678

    # Extended register R8
    rf.write_64("r8", 0xAAAAAAAAAAAAAAAA)
    rf.write_32("r8d", 0x55555555)
    assert rf.read_32("r8d") == 0x55555555
    assert rf.read_64("r8") == 0x0000000055555555


def test_register_file_16bit_and_8bit_preservation():
    """Verify that writing to 16-bit and 8-bit registers preserves unaffected bits."""
    rf = X86_64RegisterFile()

    rf.write_64("rax", 0x1122334455667788)

    # Write 16-bit AX: Upper 48 bits MUST be preserved
    rf.write_16("ax", 0xAA55)
    assert rf.read_64("rax") == 0x112233445566AA55
    assert rf.read_16("ax") == 0xAA55

    # Write 8-bit AL: Upper 56 bits preserved
    rf.write_8("al", 0x99)
    assert rf.read_64("rax") == 0x112233445566AA99
    assert rf.read_8("al") == 0x99

    # Write 8-bit AH: Bits 8..15 modified, others preserved
    rf.write_8("ah", 0x42)
    assert rf.read_64("rax") == 0x1122334455664299
    assert rf.read_8("ah") == 0x42


def test_register_file_stack_operations():
    """Verify x86-64 stack push/pop LIFO operations and RSP movement."""
    initial_rsp = 0x0000000007FE3DC0
    rf = X86_64RegisterFile(initial_rsp=initial_rsp)

    # Push 1
    new_rsp = rf.push(0xDEADBEEFCAFEBABE)
    assert new_rsp == initial_rsp - 8
    assert rf.rsp == initial_rsp - 8

    # Push 2
    new_rsp2 = rf.push(0x1122334455667788)
    assert new_rsp2 == initial_rsp - 16
    assert rf.rsp == initial_rsp - 16

    # Pop LIFO
    val1 = rf.pop()
    assert val1 == 0x1122334455667788
    assert rf.rsp == initial_rsp - 8

    val2 = rf.pop()
    assert val2 == 0xDEADBEEFCAFEBABE
    assert rf.rsp == initial_rsp

    # Underflow check
    with pytest.raises(RuntimeError):
        rf.pop()


def test_rflags_arithmetic_flag_derivations():
    """Verify status flags calculation for ADD and SUB arithmetic."""
    rfl = RFlagsRegister()

    # Reserved bit 1 must be 1
    assert (rfl.raw & 0x2) != 0

    # 1. Zero result: 42 - 42
    rfl.update_arithmetic_sub(42, 42, width=64)
    assert rfl.zf is True
    assert rfl.cf is False
    assert rfl.sf is False
    assert rfl.of is False

    # 2. Unsigned Borrow: 5 - 10
    rfl.update_arithmetic_sub(5, 10, width=64)
    assert rfl.cf is True   # Borrow
    assert rfl.zf is False
    assert rfl.sf is True   # Most significant bit is 1 (negative in 2s complement)
    assert rfl.of is False

    # 3. Signed Overflow: 0x7FFFFFFF + 1 (32-bit)
    rfl.update_arithmetic_add(0x7FFFFFFF, 1, width=32)
    assert rfl.of is True   # Signed overflow (+ and + gives -)
    assert rfl.sf is True
    assert rfl.cf is False
    assert rfl.zf is False

    summary = rfl.format_flags_summary()
    assert "OF" in summary
    assert "SF" in summary


def test_control_registers_and_msr():
    """Verify Control Registers bitfield extraction and PML4 root address masking."""
    cr = ControlRegisters(
        cr0=0x80010033,       # PG=1, WP=1, ET=1, PE=1
        cr3=0x0000000007FA0000,
        cr4=0x000006F0,       # PAE=1, OSFXSR=1
        ia32_efer=0x00000D01, # LME=1, LMA=1, NXE=1, SCE=1
    )

    assert cr.is_paging_enabled is True
    assert cr.is_protected_mode is True
    assert cr.is_write_protect is True
    assert cr.is_pae_enabled is True
    assert cr.is_osfxsr_enabled is True
    assert cr.is_long_mode_active is True
    assert cr.is_no_execute_enabled is True
    assert cr.is_syscall_extension_enabled is True
    assert cr.pml4_base_address == 0x0000000007FA0000


def test_system_v_abi_calling_convention():
    """Verify System V AMD64 ABI register argument distribution."""
    rf = X86_64RegisterFile()
    args = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]

    reg_args, stack_args = CallingConventionSimulator.prepare_call(rf, args)

    assert reg_args["rdi"] == 0x10
    assert reg_args["rsi"] == 0x20
    assert reg_args["rdx"] == 0x30
    assert reg_args["rcx"] == 0x40
    assert reg_args["r8"]  == 0x50
    assert reg_args["r9"]  == 0x60

    # 7th and 8th arguments pushed on stack
    assert stack_args == [0x70, 0x80]
    assert rf.read_64("rdi") == 0x10
    assert rf.read_64("rsi") == 0x20

    # Callee-saved vs Caller-saved classifications
    assert "rbx" in CallingConventionSimulator.CALLEE_SAVED
    assert "rsp" in CallingConventionSimulator.CALLEE_SAVED
    assert "rbp" in CallingConventionSimulator.CALLEE_SAVED
    assert "r12" in CallingConventionSimulator.CALLEE_SAVED

    assert "rax" in CallingConventionSimulator.CALLER_SAVED
    assert "rdi" in CallingConventionSimulator.CALLER_SAVED


def test_cpp20_registers_sim_executable():
    """Verify that the C++20 standalone binary compiles and passes all internal tests."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch03_registers_sim"

    if not bin_path.exists():
        subprocess.run(["cmake", "-B", "build"], cwd=repo_root, check=True)
        subprocess.run(["cmake", "--build", "build"], cwd=repo_root, check=True)

    result = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "ALL REGISTER SIMULATION TESTS PASSED SUCCESSFULLY" in result.stdout


def test_figure_generation_outputs(tmp_path: Path):
    """Verify high-resolution figure generation for register hierarchy and control registers."""
    fig1_path = tmp_path / "fig3_03_x86_64_register_hierarchy.png"
    fig2_path = tmp_path / "fig3_04_rflags_and_control_registers.png"

    res1 = plot_x86_64_register_hierarchy(fig1_path)
    res2 = plot_rflags_and_control_registers(fig2_path)

    assert res1.exists()
    assert res2.exists()

    with Image.open(res1) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 1500

    with Image.open(res2) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 1500
