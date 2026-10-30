"""Tests for Chapter 1 Comprehensive Exercises (Theory, Verification & Simulation)."""

import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter1Quiz,
    Chapter1TheoryValidator,
    plot_chapter1_exercises_summary,
    plot_chapter1_mastery_radar,
)


def test_chapter1_quiz_structure_and_perfect_score():
    """Verify that Chapter1Quiz has 10 comprehensive questions and evaluates correctly."""
    quiz = Chapter1Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 9 sections in Chapter 1
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"1.{i}" for i in range(1, 10)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter1_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter1Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter1_theory_validator_pe_headers():
    """Test Chapter1TheoryValidator for PE32+ header magic validation."""
    valid_magics = {
        "dos_magic": 0x5A4D,
        "pe_signature": 0x00004550,
        "optional_magic": 0x020B,
    }
    ok, msg = Chapter1TheoryValidator.validate_pe_headers(valid_magics)
    assert ok is True
    assert "正解" in msg

    # Invalid optional magic (e.g., 32-bit PE 0x010B)
    bad_magics = valid_magics.copy()
    bad_magics["optional_magic"] = 0x010B
    ok_bad, msg_bad = Chapter1TheoryValidator.validate_pe_headers(bad_magics)
    assert ok_bad is False
    assert "optional_magic" in msg_bad


def test_chapter1_theory_validator_calling_convention():
    """Test Chapter1TheoryValidator for Microsoft x64 calling convention."""
    valid_conv = {
        "arg1": "rcx",
        "arg2": "rdx",
        "arg3": "r8",
        "arg4": "r9",
        "shadow_space_bytes": 32,
        "stack_alignment_bytes": 16,
    }
    ok, msg = Chapter1TheoryValidator.validate_uefi_calling_convention(valid_conv)
    assert ok is True

    # Bad shadow space
    bad_conv = valid_conv.copy()
    bad_conv["shadow_space_bytes"] = 0
    ok_bad, msg_bad = Chapter1TheoryValidator.validate_uefi_calling_convention(bad_conv)
    assert ok_bad is False
    assert "shadow_space_bytes" in msg_bad


def test_chapter1_theory_validator_clang_flags():
    """Test Chapter1TheoryValidator for Clang freestanding UEFI flags."""
    flags_ok = [
        "-target x86_64-unknown-windows",
        "-ffreestanding",
        "-mno-red-zone",
        "-fno-stack-protector",
        "-fshort-wchar",
    ]
    ok, msg = Chapter1TheoryValidator.validate_clang_uefi_flags(flags_ok)
    assert ok is True

    # Missing -mno-red-zone
    flags_bad = [
        "-target x86_64-unknown-windows",
        "-ffreestanding",
        "-fno-stack-protector",
        "-fshort-wchar",
    ]
    ok_bad, msg_bad = Chapter1TheoryValidator.validate_clang_uefi_flags(flags_bad)
    assert ok_bad is False
    assert "-mno-red-zone" in msg_bad


def test_chapter1_theory_validator_mbr_vs_gpt():
    """Test Chapter1TheoryValidator for MBR vs GPT addressing math."""
    valid_math = {
        "mbr_lba_bits": 32,
        "gpt_lba_bits": 64,
        "mbr_max_tib": 2.0,
    }
    ok, msg = Chapter1TheoryValidator.validate_mbr_vs_gpt_math(valid_math)
    assert ok is True

    bad_math = valid_math.copy()
    bad_math["mbr_max_tib"] = 4.0
    ok_bad, msg_bad = Chapter1TheoryValidator.validate_mbr_vs_gpt_math(bad_math)
    assert ok_bad is False
    assert "mbr_max_tib" in msg_bad


def test_chapter1_theory_validator_system_table_layout():
    """Test Chapter1TheoryValidator for 64-bit EFI_SYSTEM_TABLE offsets."""
    valid_offsets = {
        "con_out_offset": 64,
        "output_string_offset": 8,
    }
    ok, msg = Chapter1TheoryValidator.validate_system_table_layout(valid_offsets)
    assert ok is True

    bad_offsets = {"con_out_offset": 52, "output_string_offset": 8}
    ok_bad, msg_bad = Chapter1TheoryValidator.validate_system_table_layout(bad_offsets)
    assert ok_bad is False
    assert "con_out_offset" in msg_bad


def test_cpp20_os_chapter1_exercises_execution():
    """Verify that the C++20 standalone exercises binary compiles and passes all asserts."""
    bin_path = Path("build/bin/ch01_os_chapter1_exercises")
    if not bin_path.exists():
        pytest.skip("Standalone binary not built yet; run cmake --build build first")

    result = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All Chapter 1 C++20 Exercises Passed Successfully!" in result.stdout
    assert "Microsoft x64 Calling Convention validated" in result.stdout
    assert "EFI_SYSTEM_TABLE->ConOut offset confirmed at: 64 bytes" in result.stdout


def test_chapter1_generated_figures_exist():
    """Verify generated high-DPI figure files exist in 1/result/."""
    p1 = Path("1/result/fig1_ex1_exercises_summary.png")
    p2 = Path("1/result/fig1_ex2_ch1_mastery_radar.png")
    if not p1.exists() or not p2.exists():
        plot_chapter1_exercises_summary(p1)
        plot_chapter1_mastery_radar(p2)

    assert p1.exists()
    assert p2.exists()
    assert p1.stat().st_size > 10000
    assert p2.stat().st_size > 10000
