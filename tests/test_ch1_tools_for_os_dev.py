"""Unit tests for Section 1.8: Tools for OS Development."""

import os
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    TargetTriple,
    FreestandingEnvironmentModel,
    CompilerToolchainPipeline,
    plot_llvm_toolchain_and_build_pipeline,
    plot_freestanding_vs_hosted_environment,
)


def test_target_triple_parsing():
    """Verify target triple parsing across various architecture, OS, and environment combinations."""
    t1 = TargetTriple.parse("x86_64-pc-win32-coff")
    assert t1.arch == "x86_64"
    assert t1.vendor == "pc"
    assert t1.os_name == "win32"
    assert t1.environment == "coff"
    assert t1.canonical_string == "x86_64-pc-win32-coff"

    t2 = TargetTriple.parse("x86_64-unknown-linux-gnu")
    assert t2.arch == "x86_64"
    assert t2.vendor == "unknown"
    assert t2.os_name == "linux"
    assert t2.environment == "gnu"

    t3 = TargetTriple.parse("x86_64-elf")
    assert t3.arch == "x86_64"
    assert t3.os_name == "elf"


def test_target_triple_cross_and_compatibility():
    """Verify cross-compilation detection and binary format resolution."""
    host = TargetTriple.parse("x86_64-unknown-linux-gnu")
    uefi_target = TargetTriple.parse("x86_64-pc-win32-coff")
    linux_target = TargetTriple.parse("x86_64-unknown-linux-gnu")
    kernel_target = TargetTriple.parse("x86_64-elf")

    # Cross compilation checks
    assert uefi_target.is_cross_compile(host) is True
    assert linux_target.is_cross_compile(host) is False
    assert kernel_target.is_cross_compile(host) is True

    # UEFI compatibility and binary formats
    assert uefi_target.is_uefi_compatible() is True
    assert uefi_target.object_format() == "COFF"
    assert uefi_target.executable_format() == "PE32+"

    assert linux_target.is_uefi_compatible() is False
    assert linux_target.object_format() == "ELF"
    assert linux_target.executable_format() == "ELF64"


def test_freestanding_environment_headers():
    """Verify ISO freestanding allowed vs hosted-only headers and source analysis."""
    # Header permission checks
    assert FreestandingEnvironmentModel.is_header_allowed("stdint.h") is True
    assert FreestandingEnvironmentModel.is_header_allowed("stddef.h") is True
    assert FreestandingEnvironmentModel.is_header_allowed("stdarg.h") is True
    assert FreestandingEnvironmentModel.is_header_allowed("stdio.h") is False
    assert FreestandingEnvironmentModel.is_header_allowed("stdlib.h") is False
    assert FreestandingEnvironmentModel.is_header_allowed("string.h") is False

    # Source code analysis: compliant freestanding code
    compliant_src = """
    #include <stdint.h>
    #include <stddef.h>
    
    uint64_t add(uint64_t a, uint64_t b) {
        return a + b;
    }
    """
    res1 = FreestandingEnvironmentModel.analyze_includes(compliant_src)
    assert res1["is_compliant"] is True
    assert len(res1["forbidden_hosted"]) == 0
    assert "stdint.h" in res1["allowed_freestanding"]

    # Source code analysis: non-compliant hosted code
    hosted_src = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    
    int main() {
        printf("Hello World\\n");
        return 0;
    }
    """
    res2 = FreestandingEnvironmentModel.analyze_includes(hosted_src)
    assert res2["is_compliant"] is False
    assert "stdio.h" in res2["forbidden_hosted"]
    assert "stdlib.h" in res2["forbidden_hosted"]


def test_build_pipeline_simulator_stages():
    """Verify the 4 stages of compilation pipeline."""
    stages = CompilerToolchainPipeline.trace_stages("hello.c", "x86_64-pc-win32-coff")
    assert len(stages) == 4

    assert "プリプロセッサ" in stages[0]["stage"]
    assert stages[0]["output"] == "hello.i / hello.ii"

    assert "コンパイラ" in stages[1]["stage"]
    assert "hello.s" in stages[1]["output"]

    assert "アセンブラ" in stages[2]["stage"]
    assert "COFF" in stages[2]["output"]

    assert "リンカ" in stages[3]["stage"]
    assert "PE32+" in stages[3]["output"]


def test_compiler_flags_safety_audit():
    """Verify auditing of essential bare-metal OS compilation flags."""
    # Full safety flags
    safe_flags = ["-ffreestanding", "-mno-red-zone", "-fno-stack-protector", "-fshort-wchar", "-O2"]
    audit1 = CompilerToolchainPipeline.validate_compiler_flags(safe_flags)
    assert audit1["is_safe_for_os_dev"] is True
    assert audit1["score_percent"] == 100.0
    assert len(audit1["missing_flags"]) == 0

    # Missing critical flags
    dangerous_flags = ["-O2", "-Wall"]
    audit2 = CompilerToolchainPipeline.validate_compiler_flags(dangerous_flags)
    assert audit2["is_safe_for_os_dev"] is False
    assert audit2["score_percent"] == 0.0
    assert "-ffreestanding" in audit2["missing_flags"]
    assert "-mno-red-zone" in audit2["missing_flags"]


def test_cpp20_toolchain_inspector_executable():
    """Verify execution of standalone C++20 toolchain_inspector binary."""
    binary_path = Path("build/bin/ch01_toolchain_inspector")
    assert binary_path.exists(), "Binary build/bin/ch01_toolchain_inspector must be built"

    proc = subprocess.run([str(binary_path)], capture_output=True, text=True, check=True)
    assert "Section 1.8: OSを作る道具" in proc.stdout
    assert "Target Triple Analysis" in proc.stdout
    assert "Freestanding Environment Header Compliance" in proc.stdout
    assert "Compiler Flags Safety Audit" in proc.stdout
    assert "All self-verification assertions passed successfully!" in proc.stdout


def test_figure_generation_outputs():
    """Verify that Figures 1.15 and 1.16 are generated and valid."""
    fig1 = plot_llvm_toolchain_and_build_pipeline()
    fig2 = plot_freestanding_vs_hosted_environment()

    assert os.path.exists(fig1)
    assert os.path.exists(fig2)
    assert os.path.getsize(fig1) > 100_000
    assert os.path.getsize(fig2) > 100_000
