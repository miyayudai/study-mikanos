"""Unit tests for Chapter 1.1 Hello World Binary and PE32+ Parser."""

from pathlib import Path
import pytest

from common.os_sim import (
    PE32PlusHeader,
    PE32PlusParser,
    UefiCallingConventionSim,
    plot_pe32plus_structure,
    plot_uefi_calling_convention,
)

SAMPLE_EFI_PATH = Path("vendor/mikanos-build/day01/bin/hello.efi")


def test_sample_efi_file_exists() -> None:
    assert SAMPLE_EFI_PATH.exists(), f"Sample binary not found at {SAMPLE_EFI_PATH}"
    assert SAMPLE_EFI_PATH.stat().st_size == 1536


def test_pe32plus_parser_headers() -> None:
    pe = PE32PlusParser.parse_file(SAMPLE_EFI_PATH)

    # DOS Header
    assert pe.dos_magic == 0x5A4D
    assert pe.pe_header_offset == 0x80

    # COFF File Header
    assert pe.is_x86_64
    assert pe.machine == 0x8664
    assert pe.num_sections == 2
    assert pe.size_of_optional_header == 0xF0

    # Optional Header
    assert pe.is_pe32plus
    assert pe.magic == 0x020B
    assert pe.entry_point_rva == 0x1000
    assert pe.image_base == 0x0000000140000000
    assert pe.section_alignment == 0x1000
    assert pe.file_alignment == 0x200
    assert pe.is_efi_application
    assert pe.subsystem == 0x000A
    assert pe.size_of_image == 0x3000
    assert pe.size_of_headers == 0x200


def test_pe32plus_sections() -> None:
    pe = PE32PlusParser.parse_file(SAMPLE_EFI_PATH)
    assert len(pe.sections) == 2

    # Section 1: .text
    sec_text = pe.sections[0]
    assert sec_text.name == ".text"
    assert sec_text.virtual_address == 0x1000
    assert sec_text.virtual_size == 0x14
    assert sec_text.raw_data_offset == 0x200
    assert sec_text.raw_data_size == 0x200
    assert sec_text.is_executable
    assert sec_text.is_readable
    assert not sec_text.is_writable

    # Section 2: .rdata
    sec_rdata = pe.sections[1]
    assert sec_rdata.name == ".rdata"
    assert sec_rdata.virtual_address == 0x2000
    assert sec_rdata.virtual_size == 0x1C
    assert sec_rdata.raw_data_offset == 0x400
    assert sec_rdata.raw_data_size == 0x200
    assert not sec_rdata.is_executable
    assert sec_rdata.is_readable
    assert not sec_rdata.is_writable


def test_disassembly_and_utf16_extraction() -> None:
    data = SAMPLE_EFI_PATH.read_bytes()

    instructions = PE32PlusParser.disassemble_hello_instructions(data)
    assert len(instructions) == 5
    assert instructions[0]["mnemonic"] == "sub rsp, 0x28"
    assert instructions[1]["mnemonic"] == "mov rcx, [rdx + 0x40]"
    assert instructions[2]["mnemonic"] == "lea rdx, [rip + 0x0ff1]"
    assert instructions[3]["mnemonic"] == "call [rcx + 8]"
    assert instructions[4]["mnemonic"] == "jmp -2 (.fin)"

    msg = PE32PlusParser.decode_utf16_string(data, offset=0x400)
    assert msg == "Hello, world!"


def test_uefi_calling_convention_stack_alignment() -> None:
    sim = UefiCallingConventionSim(initial_rsp=0x00007FFF_FFFF0008)
    assert (sim.rsp % 16) == 8  # Unaligned entry condition

    sim.execute_sub_rsp(0x28)
    assert (sim.rsp % 16) == 0  # 16-byte aligned

    sim.execute_mov_rcx_from_system_table(0x00000000_C0000000)
    assert sim.registers["RCX"] == 0x00000000_C0000000

    sim.execute_lea_rdx_string(0x00000001_40002000)
    assert sim.registers["RDX"] == 0x00000001_40002000

    sim.execute_call_output_string()
    assert len(sim.log) == 4


def test_plot_generation(tmp_path: Path) -> None:
    pe = PE32PlusParser.parse_file(SAMPLE_EFI_PATH)

    fig1 = tmp_path / "fig1_1.png"
    fig2 = tmp_path / "fig2_2.png"

    res1 = plot_pe32plus_structure(pe, fig1)
    res2 = plot_uefi_calling_convention(fig2)

    assert res1.exists()
    assert res1.stat().st_size > 10000
    assert res2.exists()
    assert res2.stat().st_size > 10000
