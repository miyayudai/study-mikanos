"""Unit tests for Chapter 1.6: Hands-On Mindset and Binary Mutation."""

import os
import subprocess
from pathlib import Path

import pytest

from common import (
    MutationOutcome,
    BinaryPatch,
    BinaryDiffEntry,
    BinaryMutator,
    plot_hands_on_experiment_cycle,
    plot_binary_mutation_anatomy,
)


@pytest.fixture
def hello_efi_bytes() -> bytes:
    sample_path = Path("vendor/mikanos-build/day01/bin/hello.efi")
    if not sample_path.exists():
        pytest.skip("hello.efi sample binary not found")
    return sample_path.read_bytes()


def test_binary_mutator_initial_state(hello_efi_bytes):
    """Verifies baseline binary inspection and initial UTF-16 extraction."""
    mutator = BinaryMutator(hello_efi_bytes)
    assert len(mutator.current_bytes) == 1536

    # Verify UTF-16 "Hello" search
    matches = mutator.find_utf16le("Hello")
    assert len(matches) >= 1
    assert matches[0] == 0x400

    # Verify initial simulation execution
    result = mutator.simulate_execution()
    assert result["outcome"] == MutationOutcome.SUCCESS
    assert result["screen_output"] == "Hello, world!"
    assert result["halted"] is True


def test_binary_mutation_same_length(hello_efi_bytes):
    """Verifies Case 1: Same-length string mutation without corruption."""
    mutator = BinaryMutator(hello_efi_bytes)
    patch_bytes = BinaryMutator.encode_utf16le("Hello, Mikan!", null_terminate=True)
    patch = mutator.apply_patch(0x400, patch_bytes, "Change to Hello, Mikan!")

    assert patch.offset == 0x400
    assert len(patch.patched_bytes) == 28

    diffs = mutator.diff()
    assert len(diffs) > 0

    diff_table = mutator.format_diff_table()
    assert "0x00040E" in diff_table  # 'w' -> 'M'

    result = mutator.simulate_execution()
    assert result["outcome"] == MutationOutcome.SUCCESS
    assert result["screen_output"] == "Hello, Mikan!"
    assert result["halted"] is True


def test_binary_mutation_missing_null_terminator(hello_efi_bytes):
    """Verifies Case 2: Shortened string missing null terminator reveals trailing memory."""
    mutator = BinaryMutator(hello_efi_bytes)
    # Patch only "Hi!" (3 chars = 6 bytes) without null terminator (0x0000)
    patch_bytes = BinaryMutator.encode_utf16le("Hi!", null_terminate=False)
    mutator.apply_patch(0x400, patch_bytes, "Shortened string without null terminator")

    result = mutator.simulate_execution()
    # "Hi!" overwrites "Hel", leaving "lo, world!\0" intact
    assert result["screen_output"] == "Hi!lo, world!"


def test_binary_mutation_section_overflow(hello_efi_bytes):
    """Verifies Case 3: Binary extension beyond raw capacity is rejected."""
    mutator = BinaryMutator(hello_efi_bytes)
    # Inject large chunk beyond 1536 bytes
    large_payload = b"\x41" * 300
    mutator.apply_patch(1500, large_payload, "Oversized payload")

    assert len(mutator.current_bytes) > 1536
    result = mutator.simulate_execution()
    assert result["outcome"] == MutationOutcome.SECTION_OVERFLOW
    assert result["halted"] is False
    assert "exceeds raw section/file capacity" in result["message"]


def test_binary_mutation_infinite_loop_break(hello_efi_bytes):
    """Verifies Case 4: Overwriting infinite loop 'jmp $' leads to execution crash."""
    mutator = BinaryMutator(hello_efi_bytes)
    # Replace 'jmp $' (0xEB, 0xFE) at 0x212 with 'nop nop' (0x90, 0x90)
    mutator.apply_patch(0x212, b"\x90\x90", "Break infinite loop")

    result = mutator.simulate_execution()
    assert result["outcome"] == MutationOutcome.EXECUTION_CRASH
    assert result["halted"] is False
    assert "CPU_FAULT" in result["cpu_state"]


def test_cpp20_binary_mutation_tester_executable():
    """Verifies that the compiled C++20 binary mutation tester runs and passes all assertions."""
    exe_path = Path("build/bin/ch01_binary_mutation_tester")
    if not exe_path.exists():
        pytest.skip("Binary ch01_binary_mutation_tester not compiled yet")

    proc = subprocess.run([str(exe_path)], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert "All Chapter 1.6 C++20 Binary Mutation Verification Tests PASSED!" in proc.stdout
    assert "Hello, Mikan!" in proc.stdout
    assert "Hi!lo, world!" in proc.stdout


def test_figure_generation_outputs(tmp_path):
    """Verifies generation of high-resolution architectural figures for Chapter 1.6."""
    fig1 = tmp_path / "test_fig1_11.png"
    out1 = plot_hands_on_experiment_cycle(str(fig1))
    assert Path(out1).exists()
    assert Path(out1).stat().st_size > 50_000

    fig2 = tmp_path / "test_fig1_12.png"
    out2 = plot_binary_mutation_anatomy(str(fig2))
    assert Path(out2).exists()
    assert Path(out2).stat().st_size > 50_000
