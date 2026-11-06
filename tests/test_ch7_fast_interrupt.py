"""Tests for Chapter 7.7: Fast Interrupt Handling & Top/Bottom Half Architecture."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    FastInterruptEvent,
    FastInterruptTimelineEntry,
    FastInterruptSimResult,
    simulate_monolithic_interrupts,
    simulate_split_interrupts,
    plot_monolithic_vs_split_interrupt_timeline,
    plot_top_half_bottom_half_architecture,
)


def test_fast_interrupt_dataclasses():
    """Verify FastInterruptEvent and timeline dataclasses."""
    ev = FastInterruptEvent(
        event_id=1,
        arrival_us=10.5,
        duration_us=120.0,
        name="Mouse Move",
        event_type="xhci"
    )
    assert ev.event_id == 1
    assert ev.arrival_us == 10.5
    assert ev.duration_us == 120.0
    assert ev.name == "Mouse Move"

    entry = FastInterruptTimelineEntry(
        start_us=0.0,
        end_us=10.0,
        state="IDLE",
        event_id=-1,
        desc="CPU Idle"
    )
    assert entry.state == "IDLE"
    assert entry.end_us - entry.start_us == 10.0


def test_monolithic_interrupt_bottleneck():
    """Verify monolithic handler bottleneck and packet drops under burst conditions."""
    events = [
        FastInterruptEvent(1, arrival_us=0.0, duration_us=100.0),
        FastInterruptEvent(2, arrival_us=10.0, duration_us=100.0),
        FastInterruptEvent(3, arrival_us=20.0, duration_us=100.0),
        FastInterruptEvent(4, arrival_us=30.0, duration_us=100.0),
        FastInterruptEvent(5, arrival_us=40.0, duration_us=100.0),
    ]

    # Queue capacity = 2
    res = simulate_monolithic_interrupts(events, hw_queue_capacity=2, prologue_epilogue_us=1.0)
    assert res.mode == "Monolithic (day07a)"
    assert res.processed_count < len(events)
    assert res.dropped_count > 0
    assert res.max_latency_us >= 100.0
    assert len(res.drop_events) == res.dropped_count


def test_split_interrupt_zero_drops_and_low_latency():
    """Verify split top/bottom half architecture eliminates drops and achieves microsecond latency."""
    events = [
        FastInterruptEvent(1, arrival_us=0.0, duration_us=100.0),
        FastInterruptEvent(2, arrival_us=10.0, duration_us=100.0),
        FastInterruptEvent(3, arrival_us=20.0, duration_us=100.0),
        FastInterruptEvent(4, arrival_us=30.0, duration_us=100.0),
        FastInterruptEvent(5, arrival_us=40.0, duration_us=100.0),
    ]

    res = simulate_split_interrupts(events, sw_queue_capacity=16, top_half_duration_us=0.5)
    assert res.mode == "Split Top/Bottom Half (day07b)"
    assert res.processed_count == len(events)
    assert res.dropped_count == 0
    assert res.max_latency_us < 5.0  # Top Half responds almost instantaneously
    assert len(res.drop_events) == 0


def test_split_interrupt_preemption():
    """Verify bottom half preemption by incoming hardware interrupt."""
    events = [
        FastInterruptEvent(1, arrival_us=0.0, duration_us=200.0),
        FastInterruptEvent(2, arrival_us=50.0, duration_us=100.0),
    ]

    res = simulate_split_interrupts(events, sw_queue_capacity=8, top_half_duration_us=1.0)
    assert res.processed_count == 2
    assert res.dropped_count == 0

    # Ensure timeline records Top Half for Event 2 cutting in
    states = [ent.state for ent in res.timeline]
    assert "TOP_HALF" in states
    assert "BOTTOM_HALF" in states

    top_half_entries = [ent for ent in res.timeline if ent.state == "TOP_HALF"]
    assert len(top_half_entries) == 2


def test_figure_7_13_monolithic_vs_split_timeline_generation(tmp_path):
    """Verify Figure 7.13 generates high-resolution valid image."""
    out_file = str(tmp_path / "fig7_13_test.png")
    result_path = plot_monolithic_vs_split_interrupt_timeline(out_file)

    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 10000

    with Image.open(result_path) as img:
        assert img.width >= 1800
        assert img.height >= 1200
        assert img.format == "PNG"


def test_figure_7_14_top_half_bottom_half_architecture_generation(tmp_path):
    """Verify Figure 7.14 generates high-resolution valid image."""
    out_file = str(tmp_path / "fig7_14_test.png")
    result_path = plot_top_half_bottom_half_architecture(out_file)

    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 10000

    with Image.open(result_path) as img:
        assert img.width >= 1800
        assert img.height >= 1000
        assert img.format == "PNG"


def test_cpp_fast_interrupt_sim_executable():
    """Verify C++20 standalone executable builds and passes all tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch07" / "fast_interrupt_sim.cpp"
    assert cpp_src.exists()

    bin_path = repo_root / "build" / "bin" / "ch07_fast_interrupt_sim"
    if not bin_path.exists():
        # Compile directly to temporary path
        bin_path = Path("/tmp/ch07_fast_interrupt_sim")
        compile_cmd = [
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic",
            str(cpp_src), "-o", str(bin_path)
        ]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "All Chapter 7.7 C++20 verification tests PASSED successfully!" in run_res.stdout
    assert "ArrayQueue ring buffer wrap-around, full/empty guards PASSED" in run_res.stdout
    assert "Monolithic handler bottleneck and packet drops confirmed" in run_res.stdout
    assert "Split Top/Bottom Half zero drops & sub-microsecond latency confirmed" in run_res.stdout


def test_chapter7_7_notebook_validity_and_execution():
    """Verify Chapter 7.7 Jupyter Notebook exists, has valid structure and executes cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "7" / "7.7_Fast_Interrupt_Handling_Top_Bottom_Half.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 15

    # Check that code cells have been executed and have zero errors
    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 6

    for idx, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Error in cell {idx}: {out.get('ename')}: {out.get('evalue')}\n"
                f"{''.join(out.get('traceback', []))}"
            )

    # Check key technical sections and terms exist in notebook source
    full_text = json.dumps(nb_data, ensure_ascii=False)
    assert "7.7 割り込みハンドラの高速化" in full_text
    assert "Top Half" in full_text
    assert "Bottom Half" in full_text
    assert "ArrayQueue" in full_text
    assert "Interrupt Latency" in full_text
    assert "osbook_day07b" in full_text
