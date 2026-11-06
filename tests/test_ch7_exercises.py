"""Tests for Chapter 7 Comprehensive Exercises (Theory, Verification & Simulation)."""

import json
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter7Quiz,
    Chapter7TheoryValidator,
    plot_chapter7_exercises_summary,
    plot_chapter7_mastery_radar,
)


def test_chapter7_quiz_structure_and_perfect_score():
    """Verify that Chapter7Quiz has 10 comprehensive questions covering Sections 7.1 to 7.10 and evaluates correctly."""
    quiz = Chapter7Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 10 sections in Chapter 7 (7.1 to 7.10)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"7.{i}" for i in range(1, 11)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter7_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter7Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter7_theory_validator_interrupt_frame():
    """Test Chapter7TheoryValidator for x86-64 interrupt stack frame layout."""
    valid_data = {
        "rip": 0x102040,
        "cs": 0x08,
        "rflags": 0x202,
        "rsp": 0x200FF0,
        "ss": 0x10,
    }
    ok, msg = Chapter7TheoryValidator.validate_interrupt_frame(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["rip"] = None
    ok_bad, msg_bad = Chapter7TheoryValidator.validate_interrupt_frame(bad_data)
    assert ok_bad is False
    assert "欠落" in msg_bad


def test_chapter7_theory_validator_idt_descriptor():
    """Test Chapter7TheoryValidator for 16-byte IDT descriptor decomposition."""
    func_addr = 0xFFFF800012345678
    valid_data = {
        "func_addr": func_addr,
        "offset_low": 0x5678,
        "offset_mid": 0x1234,
        "offset_high": 0xFFFF8000,
        "segment_selector": 0x08,
        "attr": (1 << 15) | (14 << 8), # Present=1, Type=14 (Interrupt Gate)
    }
    ok, msg = Chapter7TheoryValidator.validate_idt_descriptor(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["offset_low"] = 0x0000
    ok_bad, msg_bad = Chapter7TheoryValidator.validate_idt_descriptor(bad_data)
    assert ok_bad is False
    assert "offset_low" in msg_bad


def test_chapter7_theory_validator_msi_address_data():
    """Test Chapter7TheoryValidator for PCI MSI Message Address & Data."""
    valid_data = {
        "dest_id": 0,
        "vector": 0x40,
        "msg_addr": 0xFEE00000,
        "msg_data": 0x0040,
    }
    ok, msg = Chapter7TheoryValidator.validate_msi_address_data(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["msg_addr"] = 0x00000000
    ok_bad, msg_bad = Chapter7TheoryValidator.validate_msi_address_data(bad_data)
    assert ok_bad is False
    assert "Message Address" in msg_bad


def test_chapter7_theory_validator_array_queue_logic():
    """Test Chapter7TheoryValidator for ring buffer capacity, full, empty, and count calculation."""
    valid_empty = {
        "capacity": 8,
        "head": 2,
        "tail": 2,
        "is_empty": True,
        "is_full": False,
        "count": 0,
    }
    ok, msg = Chapter7TheoryValidator.validate_array_queue_logic(valid_empty)
    assert ok is True
    assert "正解" in msg

    valid_full = {
        "capacity": 8,
        "head": 0,
        "tail": 7,
        "is_empty": False,
        "is_full": True,
        "count": 7,
    }
    ok2, msg2 = Chapter7TheoryValidator.validate_array_queue_logic(valid_full)
    assert ok2 is True
    assert "正解" in msg2

    bad_data = valid_full.copy()
    bad_data["is_full"] = False
    ok_bad, msg_bad = Chapter7TheoryValidator.validate_array_queue_logic(bad_data)
    assert ok_bad is False
    assert "is_full" in msg_bad


def test_chapter7_theory_validator_race_condition_sti_hlt():
    """Test Chapter7TheoryValidator for CLI / STI; HLT interrupt shadow mechanism."""
    valid_data = {
        "has_cli_check": True,
        "uses_sti_hlt_macro": True,
        "understands_interrupt_shadow": True,
    }
    ok, msg = Chapter7TheoryValidator.validate_race_condition_sti_hlt(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = {
        "has_cli_check": False,
        "uses_sti_hlt_macro": True,
        "understands_interrupt_shadow": True,
    }
    ok_bad, msg_bad = Chapter7TheoryValidator.validate_race_condition_sti_hlt(bad_data)
    assert ok_bad is False


def test_chapter7_figures_generation(tmp_path: Path):
    """Test generation of Chapter 7 exercise summary and radar charts."""
    summary_path = tmp_path / "fig7_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig7_ex2_ch7_mastery_radar.png"

    res_sum = plot_chapter7_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter7Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter7_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch07_exercises_binary():
    """Verify that the standalone C++20 Chapter 7 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch07_os_chapter7_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 7 exercises verified successfully" in res.stdout


def test_chapter7_notebook_structure_and_error_free_execution():
    """Verify that Chapter 7 Exercises Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("7/7_Exercises.ipynb")
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
