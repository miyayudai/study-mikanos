"""Tests for Chapter 5.6: Implementing printk() (osbook_day05f)."""

import json
from pathlib import Path
import subprocess
import pytest

from common import (
    VaListAbiModel,
    PrintkPipelineSim,
    ConsoleSim,
    plot_fig5_11_varargs_abi_and_va_list_model,
    plot_fig5_12_global_placement_new_and_printk_pipeline,
)


def test_valist_abi_model_register_consumption():
    """Verify System V AMD64 ABI va_list register consumption and stack overflow."""
    model = VaListAbiModel()
    assert model.gp_offset == 8  # First vararg is at offset 8 (RSI)
    assert model.fp_offset == 48

    # Consuming 5 GP varargs (RSI, RDX, RCX, R8, R9)
    regs_expected = ["RSI", "RDX", "RCX", "R8", "R9"]
    offsets_expected = [8, 16, 24, 32, 40]

    for i in range(5):
        rec = model.consume_gp_arg(100 + i)
        assert regs_expected[i] in rec.passed_via
        assert rec.gp_offset == offsets_expected[i]

    assert model.gp_offset == 48  # All 6 GP regs consumed

    # 6th vararg (7th parameter overall) must overflow to stack
    overflow_rec1 = model.consume_gp_arg(999)
    assert "Stack" in overflow_rec1.passed_via
    assert "overflow+0" in overflow_rec1.passed_via

    overflow_rec2 = model.consume_gp_arg(1000)
    assert "Stack" in overflow_rec2.passed_via
    assert "overflow+8" in overflow_rec2.passed_via


def test_valist_abi_model_state_dump():
    """Verify va_list state dump dictionary structure."""
    model = VaListAbiModel(reg_save_base=0x1000, overflow_base=0x2000)
    model.consume_gp_arg(42)

    state = model.dump_va_list_state()
    assert state["gp_offset"] == 16
    assert state["fp_offset"] == 48
    assert state["reg_save_area"] == hex(0x1000)
    assert state["args_consumed_count"] == 1


def test_printk_pipeline_basic_formatting():
    """Verify printk pipeline correctly formats strings and updates console."""
    console = ConsoleSim()
    sim = PrintkPipelineSim(console=console)

    chars = sim.printk("Hello %s, val=%d\n", "World", 42)
    assert chars == len("Hello World, val=42\n")
    assert console.get_line(0) == "Hello World, val=42"
    assert console.cursor_row == 1
    assert console.cursor_column == 0
    assert console.scroll_count == 0


def test_printk_pipeline_buffer_truncation():
    """Verify printk safely truncates output to buffer_size - 1 like vsnprintf."""
    console = ConsoleSim()
    sim = PrintkPipelineSim(console=console, buffer_size=16)

    # String longer than 15 characters
    long_str = "0123456789ABCDEFEXTRA"
    sim.printk("%s", long_str)

    # Must be truncated to 15 characters
    assert len(sim.call_history[-1]["formatted_string"]) == 15
    assert sim.call_history[-1]["formatted_string"] == "0123456789ABCDE"


def test_day05f_27line_experiment_and_scroll():
    """Verify the 27-line printing experiment triggers exactly 3 scrolls."""
    console = ConsoleSim()
    sim = PrintkPipelineSim(console=console)

    sim.run_day05f_experiment()

    # 27 lines printed with '\n'
    # Lines 0..23 fill rows 0..23 (cursor_row becomes 24)
    # Line 24 triggers 1st scroll
    # Line 25 triggers 2nd scroll
    # Line 26 triggers 3rd scroll
    assert console.scroll_count == 3
    assert console.cursor_row == 24
    assert console.cursor_column == 0

    # Oldest visible line in Row 0 is "printk: 3"
    assert console.get_line(0) == "printk: 3"
    # Latest printed line is shifted to Row 23
    assert console.get_line(23) == "printk: 26"
    # Row 24 is waiting for next input
    assert console.get_line(24) == ""


def test_figure_generation_outputs():
    """Verify that Figures 5.11 and 5.12 are generated at high resolution."""
    fig11_path = Path("5/result/fig5_11_varargs_abi_and_va_list_model.png")
    fig12_path = Path("5/result/fig5_12_global_placement_new_and_printk_pipeline.png")

    out11 = plot_fig5_11_varargs_abi_and_va_list_model(str(fig11_path))
    out12 = plot_fig5_12_global_placement_new_and_printk_pipeline(str(fig12_path))

    assert Path(out11).exists()
    assert Path(out12).exists()
    assert fig11_path.stat().st_size > 100_000
    assert fig12_path.stat().st_size > 100_000


def test_cpp20_printk_sim_standalone():
    """Verify C++20 standalone binary execution and assertions."""
    bin_path = Path("build/bin/ch05_printk_sim")
    if not bin_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch05_printk_sim"], check=True)

    result = subprocess.run([str(bin_path), "--json"], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    assert data["status"] == "success"
    assert data["chapter"] == "5.6"
    assert data["scroll_count"] == 4
    assert data["line_0"] == "printk: 4"
    assert data["line_23"] == "Hex: 0x1234abcd, Str: MikanOS"
    assert data["placement_new_addr"] == data["console_buf_addr"]
    assert data["total_abi_trace_items"] == 8


def test_notebook_structure_and_error_free_execution():
    """Verify 5/5.6_Implementing_printk.ipynb exists, is valid JSON, and has no execution errors."""
    nb_path = Path("5/5.6_Implementing_printk.ipynb")
    assert nb_path.exists(), "Notebook 5.6 must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10, "Notebook must contain at least 10 cells"

    # Check for execution errors in cell outputs
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", f"Cell {idx} contains an error output: {output}"
