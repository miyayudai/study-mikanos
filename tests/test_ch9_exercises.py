"""Tests for Chapter 9 Comprehensive Exercises (Theory, Verification & Simulation)."""

import json
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter9Quiz,
    Chapter9TheoryValidator,
    plot_chapter9_exercises_summary,
    plot_chapter9_mastery_radar,
)


def test_chapter9_quiz_structure_and_perfect_score():
    """Verify that Chapter9Quiz has 10 comprehensive questions covering Sections 9.1 to 9.7 and evaluates correctly."""
    quiz = Chapter9Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 7 sections in Chapter 9 (9.1 to 9.7)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"9.{i}" for i in range(1, 8)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter9_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter9Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter9_theory_validator_layer_compositing():
    """Test Chapter9TheoryValidator for layer compositing and transparency checks."""
    valid_data = {
        "z_order": [0, 1, 2],
        "expected_z_order": [0, 1, 2],
        "trans_color": 0x010101,
        "test_pixel": 0x00FF00,
        "expected_color": 0x00FF00,
    }
    ok, msg = Chapter9TheoryValidator.validate_layer_compositing(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["z_order"] = [2, 1, 0]  # Inverted order
    ok_bad, msg_bad = Chapter9TheoryValidator.validate_layer_compositing(bad_data)
    assert ok_bad is False
    assert "Zオーダー描画順序" in msg_bad


def test_chapter9_theory_validator_sbrk_alignment():
    """Test Chapter9TheoryValidator for sbrk program break and 4KiB page allocation math."""
    valid_data = {
        "prev_break": 0x10000000,
        "increment": 5000,
        "page_size": 4096,
        "new_break": 0x10000000 + 5000,
        "allocated_pages": 2,  # 5000 bytes spans 2 frames (4096 + 904)
    }
    ok, msg = Chapter9TheoryValidator.validate_sbrk_alignment(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["allocated_pages"] = 1  # Incorrect allocation
    ok_bad, msg_bad = Chapter9TheoryValidator.validate_sbrk_alignment(bad_data)
    assert ok_bad is False
    assert "新規確保 4KiB ページ数" in msg_bad


def test_chapter9_theory_validator_clipping_rect():
    """Test Chapter9TheoryValidator for 2D geometric clipping intersection."""
    valid_data = {
        "layer_x": 100,
        "layer_y": 100,
        "layer_w": 200,
        "layer_h": 150,
        "redraw_x": 150,
        "redraw_y": 80,
        "redraw_w": 100,
        "redraw_h": 100,
        "has_intersection": True,
        "intersect_width": 100,  # min(300, 250) - max(100, 150) = 250 - 150 = 100
        "intersect_height": 80,  # min(250, 180) - max(100, 80) = 180 - 100 = 80
        "src_offset_x": 50,      # 150 - 100 = 50
        "src_offset_y": 0,       # 100 - 100 = 0
    }
    ok, msg = Chapter9TheoryValidator.validate_clipping_rect(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["intersect_width"] = 90
    ok_bad, msg_bad = Chapter9TheoryValidator.validate_clipping_rect(bad_data)
    assert ok_bad is False
    assert "交差幅" in msg_bad


def test_chapter9_theory_validator_lapic_timer_math():
    """Test Chapter9TheoryValidator for Local APIC timer ticks and latency math."""
    valid_data = {
        "initial_count": 0xFFFFFFFF,
        "current_count": 0xFFFE795F,
        "bus_frequency_hz": 100_000_000,
        "divide_ratio": 1,
        "calculated_ticks": 100_000,
        "calculated_ms": 1.0,
    }
    ok, msg = Chapter9TheoryValidator.validate_lapic_timer_math(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["calculated_ticks"] = 99_999
    ok_bad, msg_bad = Chapter9TheoryValidator.validate_lapic_timer_math(bad_data)
    assert ok_bad is False
    assert "経過 Ticks" in msg_bad


def test_chapter9_theory_validator_scroll_memcpy_math():
    """Test Chapter9TheoryValidator for fast console scroll buffer shift and bottom line clear math."""
    valid_data = {
        "console_columns": 80,
        "console_rows": 25,
        "font_width": 8,
        "font_height": 16,
        "bytes_per_pixel": 4,
        "calculated_shift_bytes": 24 * 16 * 640 * 4,  # 384 lines * 640 pixels * 4B = 983,040 bytes
        "calculated_clear_bytes": 16 * 640 * 4,       # 16 lines * 640 pixels * 4B = 40,960 bytes
        "is_safe_memcpy": True,
    }
    ok, msg = Chapter9TheoryValidator.validate_scroll_memcpy_math(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["is_safe_memcpy"] = False
    ok_bad, msg_bad = Chapter9TheoryValidator.validate_scroll_memcpy_math(bad_data)
    assert ok_bad is False
    assert "順方向 memcpy は安全" in msg_bad


def test_chapter9_figures_generation(tmp_path: Path):
    """Test generation of Chapter 9 exercise summary and radar charts."""
    summary_path = tmp_path / "fig9_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig9_ex2_ch9_mastery_radar.png"

    res_sum = plot_chapter9_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter9Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter9_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch09_exercises_binary():
    """Verify that the standalone C++20 Chapter 9 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch09_os_chapter9_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 9 exercises verified successfully" in res.stdout


def test_chapter9_notebook_structure_and_error_free_execution():
    """Verify that Chapter 9 Exercises Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("9/9_Exercises.ipynb")
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
