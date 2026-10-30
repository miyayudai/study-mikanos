"""Tests for Chapter 4.5: ELF Loader & .bss Zero-Clear Simulator."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    EI_NIDENT,
    ELF_MAGIC,
    ELFCLASS64,
    ELFDATA2LSB,
    EM_X86_64,
    ET_EXEC,
    PF_R,
    PF_W,
    PF_X,
    PT_DYNAMIC,
    PT_LOAD,
    PT_NOTE,
    Elf64Ehdr,
    Elf64Phdr,
    ElfLoaderSimulator,
    generate_fig4_09_elf_segment_structure,
    generate_fig4_10_pt_load_and_bss_clearing,
)


def test_elf_header_serialization_and_parsing():
    """Verify Elf64_Ehdr binary layout, size, and validity checks."""
    ehdr = Elf64Ehdr(
        e_entry=0x100050,
        e_phoff=64,
        e_phnum=2,
    )
    raw = ehdr.to_bytes()
    assert len(raw) == 64

    parsed = Elf64Ehdr.from_bytes(raw)
    assert parsed.is_valid_magic is True
    assert parsed.is_64bit is True
    assert parsed.is_little_endian is True
    assert parsed.is_x86_64 is True
    assert parsed.e_entry == 0x100050
    assert parsed.e_phoff == 64
    assert parsed.e_phnum == 2


def test_elf_program_header_fields_and_flags():
    """Verify Elf64_Phdr fields, flags, and .bss sizing."""
    phdr = Elf64Phdr(
        p_type=PT_LOAD,
        p_flags=PF_R | PF_W,
        p_offset=0x1000,
        p_vaddr=0x101000,
        p_paddr=0x101000,
        p_filesz=0x400,
        p_memsz=0x1000,
        p_align=0x1000,
    )
    raw = phdr.to_bytes()
    assert len(raw) == 56

    parsed = Elf64Phdr.from_bytes(raw)
    assert parsed.is_load is True
    assert parsed.flags_str == "RW-"
    assert parsed.bss_size == 0xC00  # 0x1000 - 0x400 = 0xC00 (3072 bytes)


def test_calc_load_address_range_and_pages():
    """Verify CalcLoadAddressRange logic ignoring non-LOAD segments."""
    ehdr = Elf64Ehdr(e_phnum=3)
    phdrs = [
        Elf64Phdr(p_type=PT_LOAD, p_vaddr=0x100000, p_memsz=0x800),
        Elf64Phdr(p_type=PT_DYNAMIC, p_vaddr=0x500000, p_memsz=0x200),
        Elf64Phdr(p_type=PT_LOAD, p_vaddr=0x101000, p_memsz=0x1500),
    ]

    first, last = ElfLoaderSimulator.calc_load_address_range(ehdr, phdrs)
    assert first == 0x100000
    assert last == 0x102500  # 0x101000 + 0x1500

    # Pages: (0x102500 - 0x100000 + 0xfff) / 0x1000 = (0x2500 + 0xfff) / 0x1000 = 3 pages
    pages = ElfLoaderSimulator.calc_required_pages(first, last, 4096)
    assert pages == 3


def test_segment_loading_and_bss_zero_clear():
    """Verify CopyLoadSegments faithfully copies data and zeroes out BSS."""
    # Build synthetic ELF with code segment and data+BSS segment
    segments = [
        {
            "type": PT_LOAD,
            "flags": PF_R | PF_X,
            "vaddr": 0x100000,
            "data": b"\x90\x90\xc3\x00" * 32,  # 128 bytes
            "memsz": 128,
        },
        {
            "type": PT_LOAD,
            "flags": PF_R | PF_W,
            "vaddr": 0x101000,
            "data": b"\x42" * 64,  # 64 bytes initialized data
            "memsz": 256,          # 192 bytes BSS
        },
    ]

    elf_bytes = ElfLoaderSimulator.build_mock_elf(entry=0x100000, segments=segments)
    ehdr, phdrs = ElfLoaderSimulator.parse_elf(elf_bytes)

    first, last = ElfLoaderSimulator.calc_load_address_range(ehdr, phdrs)
    total_span = last - first

    # Allocate dirty memory buffer (filled with 0xFF)
    memory = bytearray(b"\xFF" * total_span)

    result = ElfLoaderSimulator.copy_load_segments(
        ehdr, phdrs, elf_bytes, memory, base_vaddr=first
    )
    assert result["loaded_count"] == 2

    # Verify code (offset 0 to 128)
    assert memory[0:128] == b"\x90\x90\xc3\x00" * 32

    # Verify data (offset 0x1000 to 0x1000+64)
    data_offset = 0x101000 - 0x100000
    assert memory[data_offset : data_offset + 64] == b"\x42" * 64

    # Verify BSS (offset 0x1000+64 to 0x1000+256) is strictly zeroed
    assert memory[data_offset + 64 : data_offset + 256] == b"\x00" * 192


def test_loader_error_handling():
    """Verify error checks on malformed ELF headers and out-of-bounds segments."""
    # Malformed magic
    with pytest.raises(ValueError, match="Invalid ELF magic"):
        ElfLoaderSimulator.parse_elf(b"NOT_AN_ELF_HEADER" + b"\x00" * 48)

    # 32-bit ELF rejected
    bad_class = bytearray(b"\x7fELF" + b"\x01" + b"\x01" + b"\x00" * 58)
    with pytest.raises(ValueError, match="Not a 64-bit ELF binary"):
        ElfLoaderSimulator.parse_elf(bytes(bad_class))

    # Architecture mismatch rejected
    ehdr_arm = Elf64Ehdr(e_machine=40)  # ARM
    with pytest.raises(ValueError, match="not x86-64"):
        ElfLoaderSimulator.parse_elf(ehdr_arm.to_bytes())


def test_two_stage_loader_flow_and_entry_point():
    """Verify complete 2-stage bootloader flow with synthetic ELF."""
    expected_entry = 0x100040
    segments = [
        {
            "type": PT_LOAD,
            "flags": PF_R | PF_X,
            "vaddr": 0x100000,
            "data": b"\xcc" * 256,
            "memsz": 256,
        }
    ]

    elf_bytes = ElfLoaderSimulator.build_mock_elf(entry=expected_entry, segments=segments)
    ehdr, phdrs = ElfLoaderSimulator.parse_elf(elf_bytes)
    assert ehdr.e_entry == expected_entry

    first, last = ElfLoaderSimulator.calc_load_address_range(ehdr, phdrs)
    pages = ElfLoaderSimulator.calc_required_pages(first, last)
    assert pages == 1


def test_figures_generation(tmp_path: Path):
    """Verify high-DPI figure generation for Chapter 4.5."""
    fig1 = tmp_path / "fig4_09_test.png"
    fig2 = tmp_path / "fig4_10_test.png"

    path1 = generate_fig4_09_elf_segment_structure(str(fig1))
    path2 = generate_fig4_10_pt_load_and_bss_clearing(str(fig2))

    assert Path(path1).exists()
    assert Path(path2).exists()

    with Image.open(path1) as img1:
        assert img1.size[0] >= 1800
        assert img1.size[1] >= 1000

    with Image.open(path2) as img2:
        assert img2.size[0] >= 1800
        assert img2.size[1] >= 1000


def test_cpp_standalone_executable():
    """Verify C++20 standalone executable passes all assertions."""
    exe_path = Path("build/bin/ch04_elf_loader_sim")
    assert exe_path.exists(), "C++ test binary not found"

    res = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert "All Section 4.5 C++20 Verification Tests Passed Successfully!" in res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 4.5 Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("4/4.5_Loading_ELF_Segments_Properly.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 12, f"Notebook should have at least 12 detailed cells, got {len(cells)}"

    # Check cell 0 has Colab badge / setup
    first_cell_src = "".join(cells[0].get("source", []))
    assert "colab" in first_cell_src.lower() or "google colab" in first_cell_src.lower()

    # Verify every code cell executed without errors
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} must be executed"
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"Code cell {idx} failed with error: {out.get('evalue')}"

