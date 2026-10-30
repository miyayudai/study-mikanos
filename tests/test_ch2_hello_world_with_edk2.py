"""Unit tests for Chapter 2.2: Hello World with EDK II (osbook_day02a) & Preprocessor Expansion."""

import subprocess
from pathlib import Path
import pytest

from common.os_sim import (
    UefiStatus,
    UefiStringEncoder,
    CPreprocessorSimulator,
    PreprocessResult,
    plot_uefi_hello_world_execution_flow,
    plot_preprocessor_include_expansion_and_tree,
)


def test_uefi_status_bitmask():
    """Verify UEFI status 64-bit mask, error identification, and string names."""
    # EFI_SUCCESS is 0 (no error bit)
    assert not UefiStatus.is_error(UefiStatus.SUCCESS)
    assert UefiStatus.error_code(UefiStatus.SUCCESS) == 0
    assert UefiStatus.name(UefiStatus.SUCCESS) == "EFI_SUCCESS"
    assert UefiStatus.to_hex(UefiStatus.SUCCESS) == "0x0000000000000000"

    # Common error statuses have bit 63 set
    for status, name, code in [
        (UefiStatus.LOAD_ERROR, "EFI_LOAD_ERROR", 1),
        (UefiStatus.INVALID_PARAMETER, "EFI_INVALID_PARAMETER", 2),
        (UefiStatus.UNSUPPORTED, "EFI_UNSUPPORTED", 3),
        (UefiStatus.BAD_BUFFER_SIZE, "EFI_BAD_BUFFER_SIZE", 4),
        (UefiStatus.BUFFER_TOO_SMALL, "EFI_BUFFER_TOO_SMALL", 5),
    ]:
        assert UefiStatus.is_error(status)
        assert UefiStatus.error_code(status) == code
        assert UefiStatus.name(status) == name
        assert (status >> 63) == 1


def test_uefi_string_encoder_utf16():
    """Verify UTF-16LE encoding, Little-Endian layout, and null termination."""
    text = "Hello, Mikan World!\n"
    encoded = UefiStringEncoder.encode(text)

    # 20 characters + 1 null terminator = 21 CHAR16 words = 42 bytes
    assert len(encoded) == 42
    assert encoded[-2:] == b"\x00\x00"

    decoded = UefiStringEncoder.decode(encoded)
    assert decoded == text

    char16_list = UefiStringEncoder.to_char16_list(text)
    assert len(char16_list) == 21
    assert char16_list[0] == ord("H")
    assert char16_list[-1] == 0

    analysis = UefiStringEncoder.analyze_memory_layout(text)
    assert len(analysis) == 21
    assert analysis[0]["offset"] == 0
    assert analysis[0]["hex_bytes"] == "48 00"
    assert analysis[0]["char"] == "H"
    assert analysis[-1]["hex_bytes"] == "00 00"
    assert analysis[-1]["char"] == r"\0"


def test_preprocessor_quoted_vs_angled_resolution():
    """Verify standard C preprocessor search path difference between #include <...> and #include '...'."""
    cpp = CPreprocessorSimulator()

    # System header path
    cpp.add_include_dir("include/system")
    cpp.add_file("include/system/common.h", "// System common.h")

    # Local project path
    cpp.add_file("src/common.h", "// Local project common.h")
    cpp.add_file("src/main.c", '#include "common.h"\n#include <common.h>\nint x = 1;')

    # Quoted include resolves to local file first
    resolved_quote = cpp.resolve_include("common.h", current_dir="src", is_system=False)
    assert resolved_quote == "src/common.h"

    # Angled include bypasses current dir and resolves to system include dir
    resolved_angle = cpp.resolve_include("common.h", current_dir="src", is_system=True)
    assert resolved_angle == "include/system/common.h"


def test_preprocessor_include_guards_prevent_bloat():
    """Verify that multiple inclusion guards (#ifndef / #define) prevent duplicate expansions and AST errors."""
    cpp = CPreprocessorSimulator()
    cpp.add_include_dir("MdePkg/Include")

    # Base header with guard
    cpp.add_file("MdePkg/Include/Uefi/UefiBaseType.h", """#ifndef __UEFI_BASE_TYPE_H__
#define __UEFI_BASE_TYPE_H__
typedef unsigned long long UINTN;
typedef char16_t CHAR16;
typedef UINTN EFI_STATUS;
#endif
""")

    # Spec header includes BaseType
    cpp.add_file("MdePkg/Include/Uefi/UefiSpec.h", """#ifndef __UEFI_SPEC_H__
#define __UEFI_SPEC_H__
#include <Uefi/UefiBaseType.h>
struct EFI_SYSTEM_TABLE {
    void* ConOut;
};
#endif
""")

    # Main Uefi.h includes both
    cpp.add_file("MdePkg/Include/Uefi.h", """#ifndef __UEFI_H__
#define __UEFI_H__
#include <Uefi/UefiBaseType.h>
#include <Uefi/UefiSpec.h>
#endif
""")

    # UefiLib includes Uefi.h
    cpp.add_file("MdePkg/Include/Library/UefiLib.h", """#ifndef __UEFI_LIB_H__
#define __UEFI_LIB_H__
#include <Uefi.h>
EFI_STATUS Print(const CHAR16* Format, ...);
#endif
""")

    # MikanLoaderPkg Main.c
    cpp.add_file("MikanLoaderPkg/Main.c", """#include <Uefi.h>
#include <Library/UefiLib.h>

EFI_STATUS EFIAPI UefiMain(
    EFI_HANDLE ImageHandle,
    EFI_SYSTEM_TABLE *SystemTable)
{
  Print(L"Hello, Mikan World!\\n");
  while (1);
  return EFI_SUCCESS;
}
""")

    result = cpp.preprocess("MikanLoaderPkg/Main.c")
    assert result.source_line_count == 11
    assert result.expanded_line_count > result.source_line_count

    # Guard symbols must all be defined
    expected_symbols = {"__UEFI_BASE_TYPE_H__", "__UEFI_SPEC_H__", "__UEFI_H__", "__UEFI_LIB_H__"}
    for sym in expected_symbols:
        assert sym in result.guard_symbols_defined

    # Check that tree captures the dependencies
    assert result.include_tree["file"] == "MikanLoaderPkg/Main.c"
    assert len(result.include_tree["children"]) == 2  # Uefi.h and UefiLib.h


def test_uefi_hello_world_figure_generation(tmp_path):
    """Verify that both technical figures are generated with high quality."""
    fig1 = tmp_path / "fig2_03_test.png"
    fig2 = tmp_path / "fig2_04_test.png"

    path1 = plot_uefi_hello_world_execution_flow(fig1)
    path2 = plot_preprocessor_include_expansion_and_tree(fig2)

    assert path1.exists()
    assert path1.stat().st_size > 50_000

    assert path2.exists()
    assert path2.stat().st_size > 50_000


def test_cpp20_uefi_hello_world_sim_standalone(tmp_path):
    """Verify that the C++20 standalone simulator compiles and passes all internal asserts."""
    proj_root = Path(__file__).resolve().parent.parent
    cpp_source = proj_root / "src" / "ch02" / "uefi_hello_world_sim.cpp"
    assert cpp_source.exists()

    bin_path = tmp_path / "uefi_hello_world_sim"
    compile_cmd = [
        "g++", "-std=c++20", "-O2", "-Wall", "-Wextra",
        str(cpp_source), "-o", str(bin_path)
    ]
    res_compile = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert res_compile.returncode == 0, f"Compilation failed: {res_compile.stderr}"

    run_cmd = [str(bin_path), "--test"]
    res_run = subprocess.run(run_cmd, capture_output=True, text=True)
    assert res_run.returncode == 0, f"Execution failed: {res_run.stderr}"
    assert "[PASS]" in res_run.stdout
