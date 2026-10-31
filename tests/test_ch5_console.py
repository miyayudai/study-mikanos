"""Tests for Chapter 5.5: Implementing Console Class (osbook_day05e)."""

import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import pytest

from common import (
    ConsoleSim,
    plot_fig5_09_console_buffer_and_cursor_model,
    plot_fig5_10_console_scroll_sequence,
)


def test_console_initial_state():
    """Verify Console initial state (cursor at 0,0, empty buffer, no scroll)."""
    console = ConsoleSim()
    assert console.cursor_row == 0
    assert console.cursor_column == 0
    assert console.scroll_count == 0
    assert len(console.scroll_history) == 0

    state = console.dump_state()
    assert state["cursor_row"] == 0
    assert state["cursor_column"] == 0
    assert state["scroll_count"] == 0
    assert all(line == "" for line in state["lines"])


def test_console_put_string_and_cursor_advance():
    """Verify writing string advances cursor column and updates buffer & framebuffer."""
    console = ConsoleSim()
    console.put_string("Hello")

    assert console.cursor_row == 0
    assert console.cursor_column == 5
    assert console.get_line(0) == "Hello"

    # Framebuffer should contain black pixels
    black_pixels = np.sum(
        (console.framebuffer[:, :, 0] == 0) &
        (console.framebuffer[:, :, 1] == 0) &
        (console.framebuffer[:, :, 2] == 0)
    )
    assert black_pixels > 0


def test_console_newline_advance():
    """Verify newline resets column and increments row without scroll."""
    console = ConsoleSim()
    console.put_string("Row 0\nRow 1\n")

    assert console.cursor_row == 2
    assert console.cursor_column == 0
    assert console.get_line(0) == "Row 0"
    assert console.get_line(1) == "Row 1"
    assert console.scroll_count == 0


def test_console_scroll_behavior():
    """Verify scroll mechanics when filling beyond 24 lines."""
    console = ConsoleSim()

    # 1. Print 24 lines (line 0 to line 23)
    for i in range(24):
        console.put_string(f"line {i}\n")

    assert console.cursor_row == 24
    assert console.cursor_column == 0
    assert console.scroll_count == 0
    assert console.get_line(0) == "line 0"
    assert console.get_line(23) == "line 23"
    assert console.get_line(24) == ""

    # 2. Print line 24 (25th line, triggers 1st scroll upon '\n')
    console.put_string("line 24\n")
    assert console.scroll_count == 1
    assert console.cursor_row == 24
    assert console.cursor_column == 0
    assert console.get_line(0) == "line 1"
    assert console.get_line(23) == "line 24"
    assert console.get_line(24) == ""
    assert console.scroll_history == ["line 0"]

    # 3. Print line 25 and line 26 (completing 27 lines from day05e main.cpp)
    console.put_string("line 25\n")
    console.put_string("line 26\n")
    assert console.scroll_count == 3
    assert console.cursor_row == 24
    assert console.get_line(0) == "line 3"
    assert console.get_line(22) == "line 25"
    assert console.get_line(23) == "line 26"
    assert console.scroll_history == ["line 0", "line 1", "line 2"]


def test_console_column_clipping():
    """Verify clipping at kColumns - 1 (79 chars) prevents buffer overflow."""
    console = ConsoleSim()
    long_str = "A" * 100
    console.put_string(long_str)

    assert console.cursor_column == ConsoleSim.kColumns - 1
    assert len(console.get_line(0)) == ConsoleSim.kColumns - 1


def test_console_clear_screen():
    """Verify clearing screen resets buffer, cursor and framebuffer."""
    console = ConsoleSim()
    console.put_string("Some text\nAnother line")
    console.clear_screen()

    assert console.cursor_row == 0
    assert console.cursor_column == 0
    assert all(line == "" for line in console.get_buffer_text())
    assert np.all(console.framebuffer == 255)


def test_figure_generation_outputs(tmp_path):
    """Verify Figure 5.9 and 5.10 generation and image validity."""
    fig9_path = tmp_path / "fig5_09.png"
    fig10_path = tmp_path / "fig5_10.png"

    res9 = plot_fig5_09_console_buffer_and_cursor_model(str(fig9_path))
    res10 = plot_fig5_10_console_scroll_sequence(str(fig10_path))

    assert Path(res9).exists()
    assert Path(res10).exists()

    with Image.open(res9) as img:
        assert img.size[0] >= 1000
        assert img.size[1] >= 600

    with Image.open(res10) as img:
        assert img.size[0] >= 1000
        assert img.size[1] >= 600

    # Also verify repo result files exist
    assert Path("5/result/fig5_09_console_buffer_and_cursor_model.png").exists()
    assert Path("5/result/fig5_10_console_scroll_sequence.png").exists()


def test_cpp20_console_sim_standalone():
    """Compile and run C++20 standalone console simulation."""
    src_path = Path("src/ch05/console_sim.cpp")
    bin_path = Path("src/ch05/console_sim")

    cmd_compile = [
        "g++", "-std=c++20", "-O2", "-Wall", "-Wextra",
        str(src_path), "-o", str(bin_path)
    ]
    compile_res = subprocess.run(cmd_compile, capture_output=True, text=True)
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    run_res = subprocess.run([str(bin_path), "--json"], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"

    # Extract JSON from output
    stdout = run_res.stdout
    json_start = stdout.find("{")
    json_end = stdout.rfind("}") + 1
    assert json_start != -1 and json_end != 0
    data = json.loads(stdout[json_start:json_end])

    assert data["kRows"] == 25
    assert data["kColumns"] == 80
    assert data["scroll_count"] == 3
    assert data["top_line"] == "line 3"
    assert data["status"] == "SUCCESS"


def test_notebook_structure_and_error_free_execution():
    """Verify notebook existence and valid execution structure."""
    nb_path = Path("5/5.5_Implementing_Console_Class.ipynb")
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 15

    # Check for execution error outputs
    for idx, cell in enumerate(nb_data["cells"]):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", (
                    f"Cell {idx} has an execution error: {output.get('evalue')}"
                )
