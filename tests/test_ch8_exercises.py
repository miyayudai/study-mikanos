"""Tests for Chapter 8 Comprehensive Exercises (Theory, Verification & Simulation)."""

import json
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter8Quiz,
    Chapter8TheoryValidator,
    plot_chapter8_exercises_summary,
    plot_chapter8_mastery_radar,
)


def test_chapter8_quiz_structure_and_perfect_score():
    """Verify that Chapter8Quiz has 10 comprehensive questions covering Sections 8.1 to 8.7 and evaluates correctly."""
    quiz = Chapter8Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 7 sections in Chapter 8 (8.1 to 8.7)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"8.{i}" for i in range(1, 8)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter8_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter8Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter8_theory_validator_uefi_memmap_stepping():
    """Test Chapter8TheoryValidator for UEFI memory descriptor pointer stepping arithmetic."""
    valid_data = {
        "descriptor_size": 48,
        "sizeof_descriptor": 40,
        "num_descriptors": 10,
        "calculated_offset_for_idx_5": 240, # 5 * 48
        "available_types": [1, 3, 4], # EfiBootServicesCode, EfiBootServicesData, EfiConventionalMemory
    }
    ok, msg = Chapter8TheoryValidator.validate_uefi_memmap_stepping(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["calculated_offset_for_idx_5"] = 200 # 5 * 40 (Off-by-8 bug!)
    ok_bad, msg_bad = Chapter8TheoryValidator.validate_uefi_memmap_stepping(bad_data)
    assert ok_bad is False
    assert "calculated_offset_for_idx_5" in msg_bad


def test_chapter8_theory_validator_stack_alignment():
    """Test Chapter8TheoryValidator for kernel stack allocation and System V AMD64 ABI 16-byte alignment."""
    valid_data = {
        "stack_base": 0x100000,
        "stack_size": 1024 * 1024, # 1MiB
        "rsp": 0x200000, # 0x100000 + 1MiB
    }
    ok, msg = Chapter8TheoryValidator.validate_stack_alignment(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["rsp"] = 0x200008 # Not 16-byte aligned for entry
    ok_bad, msg_bad = Chapter8TheoryValidator.validate_stack_alignment(bad_data)
    assert ok_bad is False
    assert "16 バイト境界" in msg_bad


def test_chapter8_theory_validator_gdt_descriptor():
    """Test Chapter8TheoryValidator for 64-bit code segment descriptor bit layout."""
    # Standard x86-64 Kernel Code Segment: Base=0, Limit=0xFFFF (ignored),
    # G=1, L=1, D=0, P=1, DPL=0, S=1, Type=0xA (Execute/Read) -> 0x00AF9A000000FFFF
    valid_data = {
        "selector": 0x08,
        "descriptor_raw": 0x00AF9A000000FFFF,
    }
    ok, msg = Chapter8TheoryValidator.validate_gdt_descriptor(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["selector"] = 0x10
    ok_bad, msg_bad = Chapter8TheoryValidator.validate_gdt_descriptor(bad_data)
    assert ok_bad is False
    assert "セレクタ" in msg_bad


def test_chapter8_theory_validator_identity_paging_entry():
    """Test Chapter8TheoryValidator for 4-level paging entries and CR3 alignment."""
    valid_data = {
        "cr3": 0x100000,
        "pml4_entry": 0x101003, # Base + P=1, W=1
        "pdp_entry": 0x102003,  # Base + P=1, W=1
        "pd_entry": 0x000083,   # 0MiB mapping, P=1, W=1, PS=1 (2MiB Huge Page)
    }
    ok, msg = Chapter8TheoryValidator.validate_identity_paging_entry(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["cr3"] = 0x100010 # Not 4KiB aligned
    ok_bad, msg_bad = Chapter8TheoryValidator.validate_identity_paging_entry(bad_data)
    assert ok_bad is False
    assert "4KiB 境界" in msg_bad


def test_chapter8_theory_validator_bitmap_skip_math():
    """Test Chapter8TheoryValidator for Bitmap indexing and First-Fit skip optimization."""
    valid_data = {
        "frame_id": 130,
        "start_frame_id": 10,
        "failed_offset_i": 5,
        "calculated_line": 2, # 130 // 64
        "calculated_bit": 2,  # 130 % 64
        "next_start_frame": 16, # 10 + 5 + 1
    }
    ok, msg = Chapter8TheoryValidator.validate_bitmap_skip_math(valid_data)
    assert ok is True
    assert "正解" in msg

    bad_data = valid_data.copy()
    bad_data["next_start_frame"] = 11 # Non-optimized: start_frame_id + 1
    ok_bad, msg_bad = Chapter8TheoryValidator.validate_bitmap_skip_math(bad_data)
    assert ok_bad is False
    assert "First-Fit スキップ先" in msg_bad


def test_chapter8_figures_generation(tmp_path: Path):
    """Test generation of Chapter 8 exercise summary and radar charts."""
    summary_path = tmp_path / "fig8_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig8_ex2_ch8_mastery_radar.png"

    res_sum = plot_chapter8_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter8Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter8_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch08_exercises_binary():
    """Verify that the standalone C++20 Chapter 8 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch08_os_chapter8_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 8 exercises verified successfully" in res.stdout


def test_chapter8_notebook_structure_and_error_free_execution():
    """Verify that Chapter 8 Exercises Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("8/8_Exercises.ipynb")
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
