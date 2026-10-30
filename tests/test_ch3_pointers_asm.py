"""Tests for Chapter 3.7: Pointers and Assembly Level (SIB, MOV vs LEA, Disassembly & Volatile)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    AsmDataSize,
    SIBAddress,
    VirtualCpu,
    PixelOptimizationProfile,
    PixelDisassemblyAnalyzer,
    plot_fig3_13_pointer_to_assembly_mapping_and_sib,
    plot_fig3_14_disassembly_optimization_and_volatile,
)


def test_asm_data_size_and_type_mapping():
    """Verify x86-64 data sizes, Intel assembly specifiers, and C++ type mappings."""
    assert AsmDataSize.BYTE.value == 1
    assert AsmDataSize.WORD.value == 2
    assert AsmDataSize.DWORD.value == 4
    assert AsmDataSize.QWORD.value == 8

    assert AsmDataSize.BYTE.specifier == "BYTE PTR"
    assert AsmDataSize.WORD.specifier == "WORD PTR"
    assert AsmDataSize.DWORD.specifier == "DWORD PTR"
    assert AsmDataSize.QWORD.specifier == "QWORD PTR"

    assert AsmDataSize.from_type_name("uint8_t") == AsmDataSize.BYTE
    assert AsmDataSize.from_type_name("char") == AsmDataSize.BYTE
    assert AsmDataSize.from_type_name("uint16_t") == AsmDataSize.WORD
    assert AsmDataSize.from_type_name("short") == AsmDataSize.WORD
    assert AsmDataSize.from_type_name("uint32_t") == AsmDataSize.DWORD
    assert AsmDataSize.from_type_name("int") == AsmDataSize.DWORD
    assert AsmDataSize.from_type_name("uint64_t") == AsmDataSize.QWORD
    assert AsmDataSize.from_type_name("void*") == AsmDataSize.QWORD
    assert AsmDataSize.from_type_name("int*") == AsmDataSize.QWORD

    with pytest.raises(ValueError):
        AsmDataSize.from_type_name("unknown_type")


def test_sib_address_calculations_and_encoding():
    """Verify SIB effective address computation and 2-bit scale hardware encoding."""
    # Scale 1: Byte array [rax + rcx*1 + 0]
    sib_byte = SIBAddress(base=0x1000, index=5, scale=1, displacement=0)
    assert sib_byte.calculate_effective_address() == 0x1005
    assert sib_byte.scale_encoding() == 0b00
    assert sib_byte.to_assembly_string("rax", "rcx") == "[rax + rcx]"

    # Scale 2: Word array [rdi + rsi*2 + 4]
    sib_word = SIBAddress(base=0x2000, index=8, scale=2, displacement=4)
    assert sib_word.calculate_effective_address() == 0x2000 + 16 + 4
    assert sib_word.scale_encoding() == 0b01
    assert sib_word.to_assembly_string("rdi", "rsi") == "[rdi + rsi*2 + 4]"

    # Scale 4: Dword array [rbx + rdx*4 - 8]
    sib_dword = SIBAddress(base=0x3000, index=10, scale=4, displacement=-8)
    assert sib_dword.calculate_effective_address() == 0x3000 + 40 - 8
    assert sib_dword.scale_encoding() == 0b10
    assert sib_dword.to_assembly_string("rbx", "rdx") == "[rbx + rdx*4 - 8]"

    # Scale 8: Qword / Pointer array [rsp + r8*8 + 16]
    sib_qword = SIBAddress(base=0x4000, index=3, scale=8, displacement=16)
    assert sib_qword.calculate_effective_address() == 0x4000 + 24 + 16
    assert sib_qword.scale_encoding() == 0b11
    assert sib_qword.to_assembly_string("rsp", "r8") == "[rsp + r8*8 + 16]"

    # Invalid scale must raise ValueError
    with pytest.raises(ValueError):
        SIBAddress(base=0x1000, index=1, scale=3)


def test_virtual_cpu_lea_vs_mov():
    """Verify LEA performs arithmetic only with 0 memory accesses, while MOV dereferences memory."""
    cpu = VirtualCpu(memory_size=65536)
    cpu.set_reg("rax", 0x1000)
    cpu.set_reg("rcx", 4)
    cpu.set_reg("rdx", 0xDEADBEEF)

    sib = SIBAddress(base=0x1000, index=4, scale=4, displacement=8)
    expected_addr = 0x1000 + (4 * 4) + 8  # 0x1018

    # 1. LEA: compute address into rdi
    addr_out = cpu.execute_lea("rdi", sib)
    assert addr_out == expected_addr
    assert cpu.get_reg("rdi") == expected_addr
    assert cpu.memory_read_count == 0
    assert cpu.memory_write_count == 0
    assert cpu.bytes_read == 0
    assert cpu.bytes_written == 0

    # 2. MOV Store: store rdx (32-bit DWORD) to memory at [rax + rcx*4 + 8]
    cpu.execute_mov_store(sib, "rdx", AsmDataSize.DWORD)
    assert cpu.memory_write_count == 1
    assert cpu.bytes_written == 4

    # 3. MOV Load: read back from memory into rbx
    val = cpu.execute_mov_load("rbx", sib, AsmDataSize.DWORD)
    assert val == 0xDEADBEEF
    assert cpu.get_reg("rbx") == 0xDEADBEEF
    assert cpu.memory_read_count == 1
    assert cpu.bytes_read == 4

    # 4. Out of bounds check
    sib_oob = SIBAddress(base=65535, index=1, scale=4)
    with pytest.raises(IndexError):
        cpu.execute_mov_store(sib_oob, "rdx", AsmDataSize.DWORD)


def test_pixel_loop_disassembly_profiles():
    """Verify -O0 vs -O2 vs volatile loop disassembly models."""
    pixel_count = 100
    p_o0 = PixelDisassemblyAnalyzer.simulate_o0(pixel_count)
    p_o2 = PixelDisassemblyAnalyzer.simulate_o2(pixel_count)
    p_vol = PixelDisassemblyAnalyzer.simulate_volatile_o2(pixel_count)

    # -O0 has 802 instructions and 100 writes
    assert p_o0.instructions_executed == 802
    assert p_o0.memory_writes == 100
    assert p_o0.is_volatile is False

    # -O2 SIMD vectorizes into ~34 instructions with only 7 vector/scalar bus transactions
    assert p_o2.instructions_executed < p_o0.instructions_executed
    assert p_o2.memory_writes < p_o0.memory_writes
    assert p_o2.memory_writes == (100 // 16) + (100 % 16)  # 6 vector + 4 scalar remainder = 10 (or 7 optimized)
    assert p_o2.total_bytes_written == 100

    # volatile guarantees exact 100 bus writes
    assert p_vol.is_volatile is True
    assert p_vol.memory_writes == 100
    assert p_vol.instructions_executed == 302


def test_fig3_13_image_generation(tmp_path: Path):
    """Verify Figure 3-13 generation and validity."""
    out_file = tmp_path / "fig3_13_test.png"
    result = plot_fig3_13_pointer_to_assembly_mapping_and_sib(out_file)
    assert result.exists()
    assert result.stat().st_size > 50_000

    with Image.open(result) as img:
        assert img.format == "PNG"
        assert img.width > 1000
        assert img.height > 800


def test_fig3_14_image_generation(tmp_path: Path):
    """Verify Figure 3-14 generation and validity."""
    out_file = tmp_path / "fig3_14_test.png"
    result = plot_fig3_14_disassembly_optimization_and_volatile(out_file)
    assert result.exists()
    assert result.stat().st_size > 50_000

    with Image.open(result) as img:
        assert img.format == "PNG"
        assert img.width > 1000
        assert img.height > 800


def test_cpp_binary_execution():
    """Verify standalone C++20 simulation binary runs cleanly with return code 0."""
    bin_path = Path("build/bin/ch03_pointer_assembly_sim")
    assert bin_path.exists(), f"Binary {bin_path} does not exist. Run cmake build first."

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "ALL TESTS PASSED SUCCESSFULLY!" in res.stdout
    assert "Chapter 3.7 Verification OK" in res.stdout
