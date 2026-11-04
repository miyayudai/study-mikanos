"""Tests for Chapter 6 Comprehensive Exercises (Theory, Verification & Simulation)."""

import json
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter6Quiz,
    Chapter6TheoryValidator,
    plot_chapter6_exercises_summary,
    plot_chapter6_mastery_radar,
)


def test_chapter6_quiz_structure_and_perfect_score():
    """Verify that Chapter6Quiz has 10 comprehensive questions covering Sections 6.1 to 6.4 and evaluates correctly."""
    quiz = Chapter6Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 4 sections in Chapter 6 (6.1 to 6.4)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"6.{i}" for i in range(1, 5)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter6_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter6Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter6_theory_validator_vector2d_math():
    """Test Chapter6TheoryValidator for Vector2D relative movement addition."""
    valid_data = {
        "pos_x": 200,
        "pos_y": 150,
        "disp_x": 15,
        "disp_y": -20,
        "new_x": 215,
        "new_y": 130,
    }
    ok, msg = Chapter6TheoryValidator.validate_vector2d_math(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["new_x"] = 999
    ok_bad, msg_bad = Chapter6TheoryValidator.validate_vector2d_math(bad_data)
    assert ok_bad is False
    assert "期待値" in msg_bad


def test_chapter6_theory_validator_xhci_trb_type():
    """Test Chapter6TheoryValidator for xHCI 16-byte TRB Type and Cycle bit extraction."""
    # Control: TRB Type 32 (TransferEvent: 32 << 10 = 0x8000), Cycle Bit = 1 -> 0x8001
    valid_data = {
        "control": (32 << 10) | 1,
        "trb_type": 32,
        "cycle_bit": True,
    }
    ok, msg = Chapter6TheoryValidator.validate_xhci_trb_type(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["trb_type"] = 1
    ok_bad, msg_bad = Chapter6TheoryValidator.validate_xhci_trb_type(bad_data)
    assert ok_bad is False
    assert "TRB Type" in msg_bad


def test_chapter6_theory_validator_pci_make_address():
    """Test Chapter6TheoryValidator for PCI MakeAddress 32-bit bitshift derivation."""
    # Bus 0, Device 20 (0x14), Function 0, Reg 0 -> 0x8000A000
    valid_data = {
        "bus": 0,
        "device": 0x14,
        "function": 0,
        "reg_addr": 0,
        "address": 0x8000A000,
    }
    ok, msg = Chapter6TheoryValidator.validate_pci_make_address(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["address"] = 0x00000000
    ok_bad, msg_bad = Chapter6TheoryValidator.validate_pci_make_address(bad_data)
    assert ok_bad is False


def test_chapter6_theory_validator_pci_class_code():
    """Test Chapter6TheoryValidator for USB 3.0 xHCI controller class code."""
    valid_data = {
        "base_class": 0x0C,
        "sub_class": 0x03,
        "interface": 0x30,
    }
    ok, msg = Chapter6TheoryValidator.validate_pci_class_code(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = {
        "base_class": 0x01,
        "sub_class": 0x01,
        "interface": 0x8A,
    }
    ok_bad, msg_bad = Chapter6TheoryValidator.validate_pci_class_code(bad_data)
    assert ok_bad is False


def test_chapter6_theory_validator_bar0_mmio():
    """Test Chapter6TheoryValidator for PCI BAR0 MMIO 64-bit base address extraction."""
    # 64-bit BAR: BAR0 = 0xFE000004 (bits 2:1 = 10b), BAR1 = 0x00000001 -> MMIO = 0x1FE000000
    valid_data = {
        "bar0": 0xFE000004,
        "bar1": 0x00000001,
        "mmio_addr": 0x1FE000000,
    }
    ok, msg = Chapter6TheoryValidator.validate_bar0_mmio(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["mmio_addr"] = 0xFE000004
    ok_bad, msg_bad = Chapter6TheoryValidator.validate_bar0_mmio(bad_data)
    assert ok_bad is False


def test_chapter6_theory_validator_sign_extension_safety():
    """Test Chapter6TheoryValidator for ~static_cast<uint64_t>(0xf) sign extension prevention."""
    valid_data = {
        "mask_value": 0xFFFFFFFFFFFFFFF0,
    }
    ok, msg = Chapter6TheoryValidator.validate_sign_extension_safety(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = {
        "mask_value": 0x00000000FFFFFFF0,
    }
    ok_bad, msg_bad = Chapter6TheoryValidator.validate_sign_extension_safety(bad_data)
    assert ok_bad is False


def test_chapter6_figures_generation(tmp_path: Path):
    """Test generation of Chapter 6 exercise summary and radar charts."""
    summary_path = tmp_path / "fig6_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig6_ex2_ch6_mastery_radar.png"

    res_sum = plot_chapter6_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter6Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter6_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch06_exercises_binary():
    """Verify that the standalone C++20 Chapter 6 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch06_os_chapter6_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 6 exercises verified successfully" in res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 6 Exercises Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("6/6_Exercises.ipynb")
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

