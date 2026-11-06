"""Tests for Chapter 7.8: FIFO vs FILO Data Structures and Event Ordering."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    EventRecord,
    ProcessedRecord,
    EventOrderSimResult,
    FifoQueueSim,
    FiloStackSim,
    simulate_fifo_vs_filo_ordering,
    plot_fifo_vs_filo_concept,
    plot_event_order_inversion_timeline,
    plot_data_structure_comparison,
)


def test_fifo_queue_simulator_basic_ops():
    """Verify FIFO queue behavior: push, pop, front, count, full/empty limits."""
    q = FifoQueueSim(capacity=3)
    assert q.is_empty()
    assert not q.is_full()
    assert q.count() == 0
    assert q.pop() is None
    assert q.front() is None

    ev1 = EventRecord(1, "Event A", arrival_us=0.0)
    ev2 = EventRecord(2, "Event B", arrival_us=10.0)
    ev3 = EventRecord(3, "Event C", arrival_us=20.0)
    ev4 = EventRecord(4, "Event D", arrival_us=30.0)

    assert q.push(ev1)
    assert q.push(ev2)
    assert q.push(ev3)
    assert q.is_full()
    assert not q.push(ev4)  # Capacity reached

    assert q.front().name == "Event A"
    assert q.pop().name == "Event A"
    assert q.front().name == "Event B"
    assert q.pop().name == "Event B"
    assert q.front().name == "Event C"
    assert q.pop().name == "Event C"
    assert q.is_empty()
    assert q.pop() is None


def test_filo_stack_simulator_basic_ops():
    """Verify FILO stack behavior: LIFO order, push, pop, top, full/empty limits."""
    s = FiloStackSim(capacity=3)
    assert s.is_empty()
    assert not s.is_full()
    assert s.count() == 0
    assert s.pop() is None
    assert s.top() is None

    ev1 = EventRecord(1, "Event A", arrival_us=0.0)
    ev2 = EventRecord(2, "Event B", arrival_us=10.0)
    ev3 = EventRecord(3, "Event C", arrival_us=20.0)
    ev4 = EventRecord(4, "Event D", arrival_us=30.0)

    assert s.push(ev1)
    assert s.push(ev2)
    assert s.push(ev3)
    assert s.is_full()
    assert not s.push(ev4)  # Capacity reached

    # Stack pops in reverse order: C -> B -> A
    assert s.top().name == "Event C"
    assert s.pop().name == "Event C"
    assert s.top().name == "Event B"
    assert s.pop().name == "Event B"
    assert s.top().name == "Event A"
    assert s.pop().name == "Event A"
    assert s.is_empty()
    assert s.pop() is None


def test_event_ordering_simulation_inversions_and_starvation():
    """Verify that FIFO strictly preserves order while FILO causes inversions and high variance."""
    events = [
        EventRecord(1, "K", arrival_us=0.0, duration_us=20.0),
        EventRecord(2, "E", arrival_us=10.0, duration_us=20.0),
        EventRecord(3, "Y", arrival_us=15.0, duration_us=20.0),
        EventRecord(4, "1", arrival_us=25.0, duration_us=20.0),
    ]

    fifo_res = simulate_fifo_vs_filo_ordering(events, mode="fifo")
    filo_res = simulate_fifo_vs_filo_ordering(events, mode="filo")

    # FIFO must have zero inversions and process in order 1, 2, 3, 4
    assert fifo_res.inversion_count == 0
    assert [p.event.event_id for p in fifo_res.processed_events] == [1, 2, 3, 4]

    # FILO must have inversions and process later events before earlier ones
    assert filo_res.inversion_count > 0
    filo_order = [p.event.event_id for p in filo_res.processed_events]
    assert filo_order != [1, 2, 3, 4]


def test_figure_generations_chapter7_8(tmp_path):
    """Verify that all Chapter 7.8 figures generate valid, high-resolution PNGs."""
    fig16_path = tmp_path / "fig7_16.png"
    fig17_path = tmp_path / "fig7_17.png"
    fig18_path = tmp_path / "fig7_18.png"

    res16 = plot_fifo_vs_filo_concept(str(fig16_path))
    res17 = plot_event_order_inversion_timeline(str(fig17_path))
    res18 = plot_data_structure_comparison(str(fig18_path))

    for p in [fig16_path, fig17_path, fig18_path]:
        assert p.exists()
        assert p.stat().st_size > 50_000
        with Image.open(p) as img:
            assert img.format == "PNG"
            assert img.width >= 2000
            assert img.height >= 1000


def test_cpp20_fifo_filo_sim_standalone():
    """Verify compilation and execution of C++20 Section 7.8 standalone verification binary."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch07" / "fifo_filo_sim.cpp"
    assert cpp_src.exists(), f"Source file not found: {cpp_src}"

    bin_path = repo_root / "build" / "bin" / "ch07_fifo_filo_sim"
    if not bin_path.exists():
        bin_path = repo_root / "src" / "ch07" / "fifo_filo_sim"
        compile_cmd = [
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic",
            str(cpp_src), "-o", str(bin_path)
        ]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "All Chapter 7.8 C++20 verification tests PASSED successfully!" in run_res.stdout
    assert "FifoRingQueue vs FiloArrayStack invariant guarantees CONFIRMED" in run_res.stdout
    assert "Causality preservation and starvation-freedom of FIFO PROVEN" in run_res.stdout


def test_chapter7_8_notebook_validity_and_execution():
    """Verify Chapter 7.8 Jupyter Notebook exists, has valid structure and executes cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "7" / "7.8_FIFO_vs_FILO_Data_Structures.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 15

    # Check that code cells have been executed and have zero errors
    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 5

    for idx, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Error in cell {idx}: {out.get('ename')}: {out.get('evalue')}\n"
                f"{''.join(out.get('traceback', []))}"
            )

    # Check key technical sections and terms exist in notebook source
    full_text = json.dumps(nb_data, ensure_ascii=False)
    assert "7.8 FIFOとFILO" in full_text
    assert "FIFO" in full_text
    assert "FILO" in full_text
    assert "ArrayQueue" in full_text
    assert "リングバッファ" in full_text or "Ring Buffer" in full_text
    assert "因果律" in full_text or "Causality" in full_text
