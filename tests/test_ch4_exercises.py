"""Tests for Chapter 4 Comprehensive Exercises (Theory, Verification & Simulation)."""

import json
import subprocess
from pathlib import Path
import pytest
from common.os_sim import (
    Chapter4Quiz,
    Chapter4TheoryValidator,
    plot_chapter4_exercises_summary,
    plot_chapter4_mastery_radar,
)


def test_chapter4_quiz_structure_and_perfect_score():
    """Verify that Chapter4Quiz has 10 comprehensive questions covering Sections 4.1 to 4.5 and evaluates correctly."""
    quiz = Chapter4Quiz()
    assert len(quiz.questions) == 10

    # Ensure coverage of all 5 sections in Chapter 4 (4.1 to 4.5)
    sections = {q.section for q in quiz.questions}
    expected_sections = {f"4.{i}" for i in range(1, 6)}
    assert sections == expected_sections

    # Perfect score test
    perfect_answers = {q.question_id: q.correct_index for q in quiz.questions}
    result = quiz.evaluate(perfect_answers)
    assert result["total_questions"] == 10
    assert result["correct_count"] == 10
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert all(rate == 100.0 for rate in result["mastery_rates"].values())


def test_chapter4_quiz_partial_score_and_failure():
    """Verify quiz scoring with incorrect answers and fail threshold."""
    quiz = Chapter4Quiz()
    failing_answers = {q.question_id: (q.correct_index + 1) % len(q.options) for q in quiz.questions}
    result = quiz.evaluate(failing_answers)
    assert result["correct_count"] == 0
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_chapter4_theory_validator_makefile_rules():
    """Test Chapter4TheoryValidator for GNU Makefile automatic variables and .PHONY."""
    valid_rules = {
        "target_var": "$@",
        "first_dep_var": "$<",
        "all_deps_var": "$^",
        "phony_purpose": "同名ファイルとの衝突防止 (Collision Prevention)",
    }
    ok, msg = Chapter4TheoryValidator.validate_makefile_rules(valid_rules)
    assert ok is True
    assert "正解" in msg

    # Bad rules
    bad_rules = valid_rules.copy()
    bad_rules["target_var"] = "$<"
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_makefile_rules(bad_rules)
    assert ok_bad is False
    assert "target_var" in msg_bad


def test_chapter4_theory_validator_baremetal_cxx_flags():
    """Test Chapter4TheoryValidator for essential bare-metal C++ compiler flags."""
    valid_flags = [
        "-O2", "-Wall", "-g", "--target=x86_64-elf",
        "-ffreestanding", "-mno-red-zone", "-fno-exceptions", "-fno-rtti",
        "-std=c++17",
    ]
    ok, msg = Chapter4TheoryValidator.validate_baremetal_cxx_flags(valid_flags)
    assert ok is True
    assert "正解" in msg

    # Missing -mno-red-zone
    bad_flags = ["-ffreestanding", "-fno-exceptions", "-fno-rtti"]
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_baremetal_cxx_flags(bad_flags)
    assert ok_bad is False
    assert "-mno-red-zone" in msg_bad


def test_chapter4_theory_validator_framebuffer_pixel_math():
    """Test Chapter4TheoryValidator for FrameBufferConfig VRAM coordinate-to-byte offset math."""
    valid_math = {
        "x": 100,
        "y": 50,
        "pixels_per_scan_line": 832,
        "bytes_per_pixel": 4,
        "byte_offset": 4 * (50 * 832 + 100),
    }
    ok, msg = Chapter4TheoryValidator.validate_framebuffer_pixel_math(valid_math)
    assert ok is True
    assert "正解" in msg

    bad_math = valid_math.copy()
    bad_math["byte_offset"] = 4 * (50 * 800 + 100) # Wrong stride
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_framebuffer_pixel_math(bad_math)
    assert ok_bad is False
    assert "期待値" in msg_bad


def test_chapter4_theory_validator_abi_registers():
    """Test Chapter4TheoryValidator for System V AMD64 vs Microsoft x64 ABI calling conventions."""
    valid_abi = {
        "system_v": ["RDI", "RSI", "RDX", "RCX", "R8", "R9"],
        "ms_x64": ["RCX", "RDX", "R8", "R9"],
    }
    ok, msg = Chapter4TheoryValidator.validate_abi_registers(valid_abi)
    assert ok is True
    assert "正解" in msg

    bad_abi = {
        "system_v": ["RCX", "RDX", "R8", "R9"],
        "ms_x64": ["RDI", "RSI"],
    }
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_abi_registers(bad_abi)
    assert ok_bad is False
    assert "system_v" in msg_bad


def test_chapter4_theory_validator_placement_new_buffer():
    """Test Chapter4TheoryValidator for Placement new static memory buffer safety."""
    valid_layout = {
        "class_size": 32,
        "buffer_size": 64,
        "alignment": 8,
        "requires_free": False,
    }
    ok, msg = Chapter4TheoryValidator.validate_placement_new_buffer(valid_layout)
    assert ok is True
    assert "正解" in msg

    bad_layout = valid_layout.copy()
    bad_layout["buffer_size"] = 16 # Too small
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_placement_new_buffer(bad_layout)
    assert ok_bad is False


def test_chapter4_theory_validator_vtable_layout():
    """Test Chapter4TheoryValidator for Itanium C++ ABI virtual method table structure."""
    valid_vtable = {
        "vptr_offset": 0,
        "vptr_size": 8,
        "pure_virtual_handler": "__cxa_pure_virtual",
    }
    ok, msg = Chapter4TheoryValidator.validate_vtable_layout(valid_vtable)
    assert ok is True
    assert "正解" in msg

    bad_vtable = valid_vtable.copy()
    bad_vtable["vptr_offset"] = 8
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_vtable_layout(bad_vtable)
    assert ok_bad is False


def test_chapter4_theory_validator_elf_bss_clear():
    """Test Chapter4TheoryValidator for ELF64 PT_LOAD segment .bss zero-clearing math."""
    valid_bss = {
        "p_vaddr": 0x100000,
        "p_filesz": 0x2400,
        "p_memsz": 0x3000,
        "bss_start": 0x100000 + 0x2400,
        "bss_bytes": 0x3000 - 0x2400,
    }
    ok, msg = Chapter4TheoryValidator.validate_elf_bss_clear(valid_bss)
    assert ok is True
    assert "正解" in msg

    bad_bss = valid_bss.copy()
    bad_bss["bss_bytes"] = 0x3000
    ok_bad, msg_bad = Chapter4TheoryValidator.validate_elf_bss_clear(bad_bss)
    assert ok_bad is False


def test_chapter4_figures_generation(tmp_path: Path):
    """Test generation of Chapter 4 exercise summary and radar charts."""
    summary_path = tmp_path / "fig4_ex1_exercises_summary.png"
    radar_path = tmp_path / "fig4_ex2_ch4_mastery_radar.png"

    res_sum = plot_chapter4_exercises_summary(summary_path)
    assert res_sum.exists()
    assert res_sum.stat().st_size > 10000

    quiz = Chapter4Quiz()
    answers = {q.question_id: q.correct_index for q in quiz.questions}
    eval_result = quiz.evaluate(answers)

    res_radar = plot_chapter4_mastery_radar(radar_path, eval_result["mastery_rates"])
    assert res_radar.exists()
    assert res_radar.stat().st_size > 10000


def test_cpp20_ch04_exercises_binary():
    """Verify that the standalone C++20 Chapter 4 exercises verification binary executes with exit code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch04_os_chapter4_exercises"
    if not bin_path.exists():
        pytest.skip(f"Binary {bin_path} not found. Run cmake --build build first.")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "All Chapter 4 Exercises C++20 Verifications PASSED!" in res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 4 Exercises Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("4/4_Exercises.ipynb")
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
