"""Tests for Chapter 5 Comprehensive Exercises (Theory, Verification & Simulation)."""

import json
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter5Quiz,
    Chapter5TheoryValidator,
    plot_chapter5_exercises_summary,
    plot_chapter5_mastery_radar,
)


def test_chapter5_quiz_structure_and_perfect_score():
    """Verify that Chapter5Quiz has 10 comprehensive questions covering Sections 5.1 to 5.6 and evaluates correctly."""
    quiz = Chapter5Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 6 sections in Chapter 5 (5.1 to 5.6)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"5.{i}" for i in range(1, 7)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter5_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter5Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter5_theory_validator_font_bit_expansion():
    """Test Chapter5TheoryValidator for 8x16 bitmap font bit-expansion math."""
    # font_byte = 0b10010000: dx=0 is lit (MSB), dx=1 is not lit, dx=3 is lit
    math_dx0 = {"font_byte": 0b10010000, "dx": 0, "is_lit": True}
    ok0, msg0 = Chapter5TheoryValidator.validate_font_bit_expansion(math_dx0)
    assert ok0 is True
    assert "正解" in msg0

    math_dx1 = {"font_byte": 0b10010000, "dx": 1, "is_lit": False}
    ok1, msg1 = Chapter5TheoryValidator.validate_font_bit_expansion(math_dx1)
    assert ok1 is True
    assert "正解" in msg1

    bad_math = {"font_byte": 0b10010000, "dx": 1, "is_lit": True}
    ok_bad, msg_bad = Chapter5TheoryValidator.validate_font_bit_expansion(bad_math)
    assert ok_bad is False
    assert "期待値" in msg_bad


def test_chapter5_theory_validator_reference_vs_pointer():
    """Test Chapter5TheoryValidator for C++ reference vs pointer language rules."""
    valid_comp = {
        "ref_can_be_null": False,
        "ref_can_rebind": False,
        "ref_access_operator": ".",
        "ptr_access_operator": "->",
    }
    ok, msg = Chapter5TheoryValidator.validate_reference_vs_pointer(valid_comp)
    assert ok is True
    assert "正解" in msg

    bad_comp = valid_comp.copy()
    bad_comp["ref_can_be_null"] = True
    ok_bad, msg_bad = Chapter5TheoryValidator.validate_reference_vs_pointer(bad_comp)
    assert ok_bad is False
    assert "nullptr" in msg_bad


def test_chapter5_theory_validator_objcopy_symbols():
    """Test Chapter5TheoryValidator for GNU objcopy linker symbol naming conventions."""
    valid_syms = {
        "start_symbol": "_binary_hankaku_bin_start",
        "end_symbol": "_binary_hankaku_bin_end",
        "size_symbol": "_binary_hankaku_bin_size",
    }
    ok, msg = Chapter5TheoryValidator.validate_objcopy_symbols(valid_syms)
    assert ok is True
    assert "正解" in msg

    bad_syms = valid_syms.copy()
    bad_syms["start_symbol"] = "hankaku_start"
    ok_bad, msg_bad = Chapter5TheoryValidator.validate_objcopy_symbols(bad_syms)
    assert ok_bad is False
    assert "start_symbol" in msg_bad


def test_chapter5_theory_validator_system_v_varargs_abi():
    """Test Chapter5TheoryValidator for System V AMD64 ABI va_list struct fields."""
    valid_abi = {
        "struct_fields": ["gp_offset", "fp_offset", "overflow_arg_area", "reg_save_area"],
        "num_gp_regs": 6,
    }
    ok, msg = Chapter5TheoryValidator.validate_system_v_varargs_abi(valid_abi)
    assert ok is True
    assert "正解" in msg

    bad_abi = {
        "struct_fields": ["gp_offset", "stack_ptr"],
        "num_gp_regs": 4,
    }
    ok_bad, msg_bad = Chapter5TheoryValidator.validate_system_v_varargs_abi(bad_abi)
    assert ok_bad is False


def test_chapter5_theory_validator_console_scroll_math():
    """Test Chapter5TheoryValidator for Console 25-row scroll shift and line clearing."""
    valid_scroll = {
        "total_rows": 25,
        "shifted_rows": 24,
        "cleared_row": 24,
        "final_cursor_row": 24,
    }
    ok, msg = Chapter5TheoryValidator.validate_console_scroll_math(valid_scroll)
    assert ok is True
    assert "正解" in msg

    bad_scroll = valid_scroll.copy()
    bad_scroll["shifted_rows"] = 25
    ok_bad, msg_bad = Chapter5TheoryValidator.validate_console_scroll_math(bad_scroll)
    assert ok_bad is False


def test_chapter5_theory_validator_printk_buffer_safety():
    """Test Chapter5TheoryValidator for printk vsnprintf buffer truncation safety."""
    valid_safety = {
        "buffer_size": 1024,
        "truncation_detected_when": "ret >= sizeof(buf)",
        "null_terminated": True,
    }
    ok, msg = Chapter5TheoryValidator.validate_printk_buffer_safety(valid_safety)
    assert ok is True
    assert "正解" in msg

    bad_safety = valid_safety.copy()
    bad_safety["null_terminated"] = False
    ok_bad, msg_bad = Chapter5TheoryValidator.validate_printk_buffer_safety(bad_safety)
    assert ok_bad is False


def test_chapter5_figures_generation(tmp_path: Path):
    """Test generation of Chapter 5 exercise summary and radar charts."""
    summary_path = tmp_path / "fig5_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig5_ex2_ch5_mastery_radar.png"

    res_sum = plot_chapter5_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter5Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter5_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch05_exercises_binary():
    """Verify that the standalone C++20 Chapter 5 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch05_os_chapter5_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 5 Exercises C++20 Verifications PASSED!" in res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 5 Exercises Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("5/5_Exercises.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 15, f"Notebook should have at least 15 detailed cells, got {len(cells)}"

    # Check cell 0 has Colab setup
    first_cell_src = "".join(cells[0].get("source", []))
    assert "colab" in first_cell_src.lower() or "google colab" in first_cell_src.lower()

    # Verify every code cell executed without errors
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} must be executed"
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"Code cell {idx} failed with error: {out.get('evalue')}"
