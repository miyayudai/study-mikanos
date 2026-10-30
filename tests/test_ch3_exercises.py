"""Tests for Chapter 3 Comprehensive Exercises (Theory, Verification & Simulation)."""

import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter3Quiz,
    Chapter3TheoryValidator,
    plot_chapter3_exercises_summary,
    plot_chapter3_mastery_radar,
)


def test_chapter3_quiz_structure_and_perfect_score():
    """Verify that Chapter3Quiz has 10 comprehensive questions and evaluates correctly."""
    quiz = Chapter3Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 7 sections in Chapter 3 (3.1 to 3.7)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"3.{i}" for i in range(1, 8)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter3_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter3Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter3_theory_validator_qemu_monitor_commands():
    """Test Chapter3TheoryValidator for QEMU monitor debugging commands."""
    valid_cmds = {
        "info registers": "全レジスタおよび制御レジスタのダンプ (Dump CPU Registers)",
        "xp": "物理メモリアドレスのダンプ (eXamine Physical memory)",
        "x": "仮想メモリアドレスのダンプ (eXamine Virtual memory)",
    }
    ok, msg = Chapter3TheoryValidator.validate_qemu_monitor_commands(valid_cmds)
    assert ok is True
    assert "正解" in msg

    # Bad commands
    bad_cmds = valid_cmds.copy()
    bad_cmds["xp"] = "仮想メモリダンプ"
    ok_bad, msg_bad = Chapter3TheoryValidator.validate_qemu_monitor_commands(bad_cmds)
    assert ok_bad is False
    assert "xp" in msg_bad


def test_chapter3_theory_validator_register_zero_extension():
    """Test Chapter3TheoryValidator for x86-64 32-bit zero-extension rule."""
    sim_data = {
        "initial_rax": 0xFFFFFFFFFFFFFFFF,
        "write_eax": 0x12345678,
        "final_rax": 0x0000000012345678,
    }
    ok, msg = Chapter3TheoryValidator.validate_register_zero_extension(sim_data)
    assert ok is True
    assert "正解" in msg

    # Incorrect zero extension (e.g. keeping upper bits)
    bad_sim = sim_data.copy()
    bad_sim["final_rax"] = 0xFFFFFFFF12345678
    ok_bad, msg_bad = Chapter3TheoryValidator.validate_register_zero_extension(bad_sim)
    assert ok_bad is False
    assert "final_rax" in msg_bad


def test_chapter3_theory_validator_red_zone_rules():
    """Test Chapter3TheoryValidator for System V AMD64 ABI Red Zone properties."""
    valid_rules = {
        "size_bytes": 128,
        "direction": "RSP より下 (below RSP, negative offset)",
        "compiler_flag": "-mno-red-zone",
        "safe_for_interrupts": False,
    }
    ok, msg = Chapter3TheoryValidator.validate_red_zone_rules(valid_rules)
    assert ok is True
    assert "正解" in msg

    bad_rules = valid_rules.copy()
    bad_rules["compiler_flag"] = "-O3"
    ok_bad, msg_bad = Chapter3TheoryValidator.validate_red_zone_rules(bad_rules)
    assert ok_bad is False
    assert "compiler_flag" in msg_bad


def test_chapter3_theory_validator_gop_framebuffer_math():
    """Test Chapter3TheoryValidator for GOP framebuffer stride and pixel offset math."""
    valid_math = {
        "h_res": 800,
        "v_res": 600,
        "ppsl": 832,
        "bytes_per_pixel": 4,
        "target_x": 100,
        "target_y": 50,
        "pixel_byte_offset": 4 * (50 * 832 + 100),
        "total_fb_size": 4 * 832 * 600,
    }
    ok, msg = Chapter3TheoryValidator.validate_gop_framebuffer_math(valid_math)
    assert ok is True
    assert "正解" in msg

    bad_math = valid_math.copy()
    bad_math["pixel_byte_offset"] = 4 * (50 * 800 + 100) # Wrong: used h_res instead of ppsl
    ok_bad, msg_bad = Chapter3TheoryValidator.validate_gop_framebuffer_math(bad_math)
    assert ok_bad is False
    assert "pixel_byte_offset" in msg_bad


def test_chapter3_theory_validator_system_v_abi_args():
    """Test Chapter3TheoryValidator for System V AMD64 ABI argument registers."""
    valid_regs = ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
    ok, msg = Chapter3TheoryValidator.validate_system_v_abi_args(valid_regs)
    assert ok is True
    assert "正解" in msg

    bad_regs = ["RCX", "RDX", "R8", "R9"] # Microsoft x64 ABI order
    ok_bad, msg_bad = Chapter3TheoryValidator.validate_system_v_abi_args(bad_regs)
    assert ok_bad is False


def test_chapter3_theory_validator_pointer_assembly_sib():
    """Test Chapter3TheoryValidator for SIB effective address computation."""
    valid_sib = {
        "base": 0x10000000,
        "index": 42,
        "scale": 4,
        "disp": 16,
        "effective_address": 0x10000000 + 42 * 4 + 16,
    }
    ok, msg = Chapter3TheoryValidator.validate_pointer_assembly_sib(valid_sib)
    assert ok is True
    assert "正解" in msg

    bad_sib = valid_sib.copy()
    bad_sib["effective_address"] = 0x10000000
    ok_bad, msg_bad = Chapter3TheoryValidator.validate_pointer_assembly_sib(bad_sib)
    assert ok_bad is False


def test_chapter3_figures_generation(tmp_path: Path):
    """Test generation of Chapter 3 exercise summary and radar charts."""
    summary_path = tmp_path / "fig3_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig3_ex2_ch3_mastery_radar.png"

    res_sum = plot_chapter3_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter3Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter3_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch03_exercises_binary():
    """Verify that the standalone C++20 Chapter 3 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch03_os_chapter3_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 3 Exercises C++20 Verifications PASSED!" in res.stdout
