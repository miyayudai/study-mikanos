"""Unit tests for Section 1.9: Hello World in C & Column 1.1: PE, COFF, ELF."""

import os
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    UefiSystemTableModel,
    BinaryFormatKind,
    BinaryFormatComparator,
    MinimalUefiCCompilerModel,
    plot_uefi_c_system_table_layout,
    plot_pe32plus_coff_elf_comparison,
)


def test_uefi_system_table_layout():
    """Verify EFI_SYSTEM_TABLE struct layout and offset math (dummy[52], ConOut)."""
    model = UefiSystemTableModel(system_table_base=0x3F80_0000, con_out_base=0x3F80_1000)
    layout = model.inspect_system_table_layout()

    assert len(layout) == 3

    # dummy[52]
    assert layout[0]["field"] == "dummy"
    assert layout[0]["offset"] == 0
    assert layout[0]["size"] == 52

    # ConsoleOutHandle
    assert layout[1]["field"] == "ConsoleOutHandle"
    assert layout[1]["offset"] == 52
    assert layout[1]["size"] == 8

    # ConOut
    assert layout[2]["field"] == "ConOut"
    assert layout[2]["offset"] == 60
    assert layout[2]["size"] == 8


def test_uefi_text_output_protocol_layout():
    """Verify EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL layout and member offsets."""
    model = UefiSystemTableModel()
    proto_layout = model.inspect_text_output_protocol_layout()

    assert len(proto_layout) == 2

    # dummy (Reset)
    assert proto_layout[0]["field"] == "dummy"
    assert proto_layout[0]["offset"] == 0
    assert proto_layout[0]["size"] == 8

    # OutputString
    assert proto_layout[1]["field"] == "OutputString"
    assert proto_layout[1]["offset"] == 8
    assert proto_layout[1]["size"] == 8


def test_utf16le_encoding_and_decoding():
    """Verify UTF-16LE wide string encoding and null-termination handling."""
    model = UefiSystemTableModel()
    sample_text = "Hello, world!\n"
    encoded = model.encode_utf16le(sample_text)

    # 14 chars * 2 bytes + 2 bytes null terminator = 30 bytes
    assert len(encoded) == 30
    assert encoded[0:2] == b"H\x00"
    assert encoded[-2:] == b"\x00\x00"

    decoded = model.decode_utf16le(encoded)
    assert decoded == sample_text


def test_simulated_call_output_string():
    """Verify SystemTable->ConOut->OutputString invocation semantics and 'This' pointer check."""
    con_out_addr = 0x3F80_1000
    model = UefiSystemTableModel(system_table_base=0x3F80_0000, con_out_base=con_out_addr)

    raw_str = model.encode_utf16le("Hello, MikanOS!\n")

    # 1. Successful call with valid 'This' pointer
    status, rendered = model.simulate_call_output_string(this_ptr=con_out_addr, string_utf16_bytes=raw_str)
    assert status == 0  # EFI_SUCCESS
    assert rendered == "Hello, MikanOS!\n"
    assert "Hello, MikanOS!\n" in model.output_buffer

    # 2. Failed call with mismatched 'This' pointer
    bad_this = 0xDEAD_BEEF
    bad_status, bad_rendered = model.simulate_call_output_string(this_ptr=bad_this, string_utf16_bytes=raw_str)
    assert bad_status == 0x8000_0000_0000_0002  # EFI_INVALID_PARAMETER
    assert bad_rendered == ""


def test_pointer_dereference_trace():
    """Verify pointer dereference step derivation for C arrow operator."""
    model = UefiSystemTableModel()
    trace = model.trace_pointer_dereference()
    assert len(trace) == 5
    assert "SystemTable" in trace[0]
    assert "ConOut" in trace[1]
    assert "OutputString" in trace[3]
    assert "Microsoft x64" in trace[4]


def test_binary_format_comparator():
    """Verify PE32+ vs ELF64 structural and historical comparison entries."""
    comparisons = BinaryFormatComparator.get_comparison_table()
    assert len(comparisons) == 5

    categories = [c["category"] for c in comparisons]
    assert any("マジックナンバー" in cat for cat in categories)
    assert any("ヘッダ構造" in cat for cat in categories)
    assert any("エントリポイント" in cat for cat in categories)
    assert any("サブシステム" in cat for cat in categories)

    explanation = BinaryFormatComparator.explain_uefi_pe_adoption()
    assert "Itanium" in explanation
    assert "RVA" in explanation
    assert "IMAGE_SUBSYSTEM_EFI_APPLICATION" in explanation


def test_minimal_uefi_c_compiler_model():
    """Verify C freestanding compliance audit and Makefile command simulation."""
    # Read actual hello.c from vendor codebase
    hello_c_path = Path("vendor/mikanos-build/day01/c/hello.c")
    assert hello_c_path.exists()
    source_code = hello_c_path.read_text(encoding="utf-8")

    model = MinimalUefiCCompilerModel(source_code)
    audit = model.validate_freestanding_compliance()
    assert audit["is_compliant"] is True
    assert audit["has_forbidden_hosted_headers"] is False
    assert audit["has_efi_main_entry"] is True

    # Test with hosted code containing stdio.h
    bad_source = "#include <stdio.h>\nint main() { printf(\"hi\"); return 0; }"
    bad_model = MinimalUefiCCompilerModel(bad_source)
    bad_audit = bad_model.validate_freestanding_compliance()
    assert bad_audit["is_compliant"] is False
    assert bad_audit["has_forbidden_hosted_headers"] is True

    # Test build command simulation
    commands = model.simulate_build_commands()
    assert len(commands) == 2
    assert "clang -target x86_64-pc-win32-coff" in commands[0]["command"]
    assert "lld-link /subsystem:efi_application" in commands[1]["command"]


def test_cpp20_uefi_c_hello_analyzer_executable():
    """Verify standalone C++20 tool execution and assertion checks."""
    binary_path = Path("build/bin/ch01_uefi_c_hello_analyzer")
    if not binary_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch01_uefi_c_hello_analyzer"], check=True)

    result = subprocess.run([str(binary_path)], capture_output=True, text=True, check=True)
    assert "sizeof(EFI_SYSTEM_TABLE):               68 bytes" in result.stdout
    assert "offsetof(EFI_SYSTEM_TABLE, ConOut):     60" in result.stdout
    assert "All Section 1.9 C++20 checks successfully verified!" in result.stdout


def test_figure_generation_outputs(tmp_path):
    """Verify high-resolution figure plotting functions produce valid PNGs."""
    fig17_path = str(tmp_path / "fig1_17_test.png")
    fig18_path = str(tmp_path / "fig1_18_test.png")

    out17 = plot_uefi_c_system_table_layout(fig17_path)
    out18 = plot_pe32plus_coff_elf_comparison(fig18_path)

    assert os.path.exists(out17)
    assert os.path.exists(out18)
    assert os.path.getsize(out17) > 50_000
    assert os.path.getsize(out18) > 50_000
