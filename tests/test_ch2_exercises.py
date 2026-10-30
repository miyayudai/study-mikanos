"""Tests for Chapter 2 Comprehensive Exercises (Theory, Verification & Simulation)."""

import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter2Quiz,
    Chapter2TheoryValidator,
    plot_chapter2_exercises_summary,
    plot_chapter2_mastery_radar,
)


def test_chapter2_quiz_structure_and_perfect_score():
    """Verify that Chapter2Quiz has 10 comprehensive questions and evaluates correctly."""
    quiz = Chapter2Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 9 sections in Chapter 2 (2.1 to 2.9)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"2.{i}" for i in range(1, 10)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter2_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter2Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter2_theory_validator_edk2_files():
    """Test Chapter2TheoryValidator for EDK II package configuration files."""
    valid_roles = {
        ".dec": "Declaration: 公開プロトコル/GUID宣言",
        ".inf": "Module Information: モジュール情報とソースコード定義",
        ".dsc": "Platform Description: パッケージ全体ビルド構成とライブラリ解決",
    }
    ok, msg = Chapter2TheoryValidator.validate_edk2_files(valid_roles)
    assert ok is True
    assert "正解" in msg

    # Bad roles
    bad_roles = valid_roles.copy()
    bad_roles[".dec"] = "バイナリ実行ファイル"
    ok_bad, msg_bad = Chapter2TheoryValidator.validate_edk2_files(bad_roles)
    assert ok_bad is False
    assert ".dec" in msg_bad


def test_chapter2_theory_validator_memory_descriptor_layout():
    """Test Chapter2TheoryValidator for EFI_MEMORY_DESCRIPTOR layout."""
    valid_layout = {
        "struct_size": 40,
        "type_size": 4,
        "pad_size": 4,
        "physical_start_size": 8,
        "number_of_pages_size": 8,
        "attribute_size": 8,
    }
    ok, msg = Chapter2TheoryValidator.validate_memory_descriptor_layout(valid_layout)
    assert ok is True
    assert "正解" in msg

    bad_layout = valid_layout.copy()
    bad_layout["struct_size"] = 32
    ok_bad, msg_bad = Chapter2TheoryValidator.validate_memory_descriptor_layout(bad_layout)
    assert ok_bad is False
    assert "struct_size" in msg_bad


def test_chapter2_theory_validator_exit_boot_services_reclaimable():
    """Test Chapter2TheoryValidator for reclaimable memory types."""
    valid_types = ["EfiConventionalMemory", "EfiBootServicesCode", "EfiBootServicesData"]
    ok, msg = Chapter2TheoryValidator.validate_exit_boot_services_reclaimable(valid_types)
    assert ok is True
    assert "正解" in msg

    # Invalid: missing BootServicesCode, includes RuntimeServicesCode (forbidden to free)
    bad_types = ["EfiConventionalMemory", "EfiRuntimeServicesCode"]
    ok_bad, msg_bad = Chapter2TheoryValidator.validate_exit_boot_services_reclaimable(bad_types)
    assert ok_bad is False
    assert "不足している種別" in msg_bad or "再利用不可" in msg_bad


def test_chapter2_theory_validator_pointer_arithmetic():
    """Test Chapter2TheoryValidator for 64-bit pointer arithmetic and scaling."""
    valid_math = {
        "pointer_size_bytes": 8,
        "uint32_ptr_add_1_bytes": 4,
        "uint64_ptr_add_1_bytes": 8,
    }
    ok, msg = Chapter2TheoryValidator.validate_pointer_arithmetic(valid_math)
    assert ok is True
    assert "正解" in msg

    bad_math = valid_math.copy()
    bad_math["pointer_size_bytes"] = 4
    ok_bad, msg_bad = Chapter2TheoryValidator.validate_pointer_arithmetic(bad_math)
    assert ok_bad is False
    assert "pointer_size_bytes" in msg_bad


def test_chapter2_theory_validator_struct_alignment():
    """Test Chapter2TheoryValidator for natural alignment and padding computation."""
    valid_align = {
        "total_size": 12,
        "pad_after_a": 3,
        "pad_after_c": 2,
    }
    ok, msg = Chapter2TheoryValidator.validate_struct_alignment(valid_align)
    assert ok is True
    assert "正解" in msg

    bad_align = valid_align.copy()
    bad_align["total_size"] = 10
    ok_bad, msg_bad = Chapter2TheoryValidator.validate_struct_alignment(bad_align)
    assert ok_bad is False
    assert "total_size" in msg_bad


def test_cpp20_os_chapter2_exercises_execution():
    """Verify that the C++20 standalone exercises binary compiles and passes all asserts."""
    bin_path = Path("build/bin/ch02_os_chapter2_exercises")
    if not bin_path.exists():
        pytest.skip("Standalone binary not built yet; run cmake --build build first")

    result = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "ALL CHAPTER 2 C++20 EXERCISES PASSED SUCCESSFULLY!" in result.stdout
    assert "Verified descriptor layout & 40-byte natural alignment!" in result.stdout
    assert "Verified double pointer out-parameter semantics!" in result.stdout


def test_chapter2_generated_figures_exist():
    """Verify generated high-DPI figure files exist in 2/result/."""
    p1 = Path("2/result/fig2_ex1_exercises_summary.png")
    p2 = Path("2/result/fig2_ex2_ch2_mastery_radar.png")
    if not p1.exists() or not p2.exists():
        plot_chapter2_exercises_summary(p1)
        plot_chapter2_mastery_radar(p2)

    assert p1.exists()
    assert p2.exists()
    assert p1.stat().st_size > 10000
    assert p2.stat().st_size > 10000
