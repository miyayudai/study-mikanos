"""Tests for Chapter 0 Comprehensive Exercises (Theory, Verification & Simulation)."""

import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter0Quiz,
    TheoryExerciseValidator,
    plot_chapter0_exercises_summary,
    plot_chapter0_mastery_radar,
)


def test_chapter0_quiz_structure_and_perfect_score():
    """Verify that Chapter0Quiz has 10 comprehensive questions and evaluates correctly."""
    quiz = Chapter0Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 5 sections
    sections = {q.section for q in quiz.questions}
    assert sections == {"0.1", "0.2", "0.3", "0.4", "0.5"}

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter0_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter0Quiz()
    # Provide answers that intentionally fail
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_theory_exercise_validator_os_roles():
    """Test TheoryExerciseValidator for OS roles."""
    # Correct roles
    ok, msg = TheoryExerciseValidator.validate_os_roles(["ハードウェアの抽象化", "リソースの管理"])
    assert ok is True
    assert "正解" in msg

    # Missing abstraction
    ok_bad, msg_bad = TheoryExerciseValidator.validate_os_roles(["CPUリソースの管理", "メモリ調停"])
    assert ok_bad is False
    assert "抽象化" in msg_bad


def test_theory_exercise_validator_syscall_registers():
    """Test TheoryExerciseValidator for System V AMD64 ABI syscall registers."""
    good_regs = {
        "syscall_number": "rax",
        "arg1": "rdi",
        "arg2": "rsi",
        "arg3": "rdx",
        "arg4": "r10",
    }
    ok, msg = TheoryExerciseValidator.validate_syscall_registers(good_regs)
    assert ok is True

    bad_regs = good_regs.copy()
    bad_regs["arg4"] = "rcx"  # Standard C call uses rcx, but syscall uses r10!
    ok_bad, msg_bad = TheoryExerciseValidator.validate_syscall_registers(bad_regs)
    assert ok_bad is False
    assert "r10" in msg_bad


def test_theory_exercise_validator_freestanding_flags():
    """Test TheoryExerciseValidator for freestanding compiler flags."""
    flags_ok = ["-O2", "-ffreestanding", "-Wall", "-nostdlib", "-mno-red-zone"]
    ok, msg = TheoryExerciseValidator.validate_freestanding_flags(flags_ok)
    assert ok is True

    flags_bad = ["-O2", "-ffreestanding", "-Wall"]
    ok_bad, msg_bad = TheoryExerciseValidator.validate_freestanding_flags(flags_bad)
    assert ok_bad is False
    assert "-nostdlib" in msg_bad


def test_cpp20_os_chapter0_exercises_execution():
    """Verify that the C++20 standalone exercises binary compiles and passes all asserts."""
    bin_path = Path("build/bin/ch00_os_chapter0_exercises")
    if not bin_path.exists():
        pytest.skip("Standalone binary not built yet; run cmake --build build first")

    result = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert result.returncode == 0
    assert "All Chapter 0 C++20 Exercises Passed Successfully!" in result.stdout
    assert "Isolated in 10 probes" in result.stdout or "isolated in 10 probes" in result.stdout


def test_generated_figures_exist():
    """Verify generated high-DPI figure files exist in 0/result/."""
    p1 = Path("0/result/fig0_ex1_exercises_summary.png")
    p2 = Path("0/result/fig0_ex2_ch0_mastery_radar.png")
    if not p1.exists() or not p2.exists():
        plot_chapter0_exercises_summary(p1)
        plot_chapter0_mastery_radar(p2)

    assert p1.exists()
    assert p2.exists()
    assert p1.stat().st_size > 10000
    assert p2.stat().st_size > 10000
