"""Tests for Chapter 7.9: Implementing Ring Buffer ArrayQueue."""

import json
import os
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    ArrayQueueSim,
    OneSlotWasteQueueSim,
    RingBufferSnapshot,
    plot_arrayqueue_ring_buffer_structure,
    plot_push_pop_state_transition_sequence,
    plot_buffer_boundary_and_full_empty_logic,
)


def test_array_queue_sim_basic_ops():
    """Verify ArrayQueueSim: push, pop, front, count, full/empty limits, and invariants."""
    q = ArrayQueueSim(capacity=3)
    assert q.capacity == 3
    assert q.count == 0
    assert q.is_empty()
    assert not q.is_full()
    assert q.read_pos == 0
    assert q.write_pos == 0
    assert q.front() is None
    assert q.pop() is None

    # Push 1
    assert q.push("Alpha")
    assert q.count == 1
    assert not q.is_empty()
    assert not q.is_full()
    assert q.read_pos == 0
    assert q.write_pos == 1
    assert q.front() == "Alpha"

    # Push 2
    assert q.push("Beta")
    assert q.count == 2
    assert q.read_pos == 0
    assert q.write_pos == 2
    assert q.front() == "Alpha"

    # Push 3 -> Full
    assert q.push("Gamma")
    assert q.count == 3
    assert q.is_full()
    assert q.read_pos == 0
    assert q.write_pos == 0  # Wrapped around to 0
    assert not q.push("Delta")  # Rejected: buffer full

    # Pop 1
    assert q.pop() == "Alpha"
    assert q.count == 2
    assert q.read_pos == 1
    assert q.write_pos == 0
    assert q.front() == "Beta"

    # Pop 2
    assert q.pop() == "Beta"
    assert q.count == 1
    assert q.read_pos == 2
    assert q.front() == "Gamma"

    # Pop 3 -> Empty
    assert q.pop() == "Gamma"
    assert q.count == 0
    assert q.is_empty()
    assert q.read_pos == 0  # Wrapped around to 0
    assert q.write_pos == 0
    assert q.pop() is None


def test_array_queue_sim_wrap_around():
    """Verify circular buffer wrap-around mechanics over multiple push/pop cycles."""
    q = ArrayQueueSim(capacity=4)

    # Push 4 elements: [10, 20, 30, 40]
    for v in [10, 20, 30, 40]:
        assert q.push(v)
    assert q.is_full()
    assert q.read_pos == 0
    assert q.write_pos == 0

    # Pop 2 elements: [None, None, 30, 40]
    assert q.pop() == 10
    assert q.pop() == 20
    assert q.count == 2
    assert q.read_pos == 2
    assert q.write_pos == 0

    # Push 2 elements: [50, 60, 30, 40] (write wrapped)
    assert q.push(50)
    assert q.push(60)
    assert q.is_full()
    assert q.read_pos == 2
    assert q.write_pos == 2

    # Pop all 4 in exact FIFO sequence: 30, 40, 50, 60
    assert q.pop() == 30
    assert q.read_pos == 3
    assert q.pop() == 40
    assert q.read_pos == 0  # read wrapped!
    assert q.pop() == 50
    assert q.read_pos == 1
    assert q.pop() == 60
    assert q.read_pos == 2
    assert q.is_empty()


def test_array_queue_sim_full_and_empty_errors():
    """Verify error records and boundary integrity in ArrayQueueSim."""
    q = ArrayQueueSim(capacity=2)
    assert q.pop() is None
    last_snap = q.history[-1]
    assert not last_snap.success
    assert "kEmpty" in last_snap.description

    assert q.push("A")
    assert q.push("B")
    assert not q.push("C")
    last_snap = q.history[-1]
    assert not last_snap.success
    assert "kFull" in last_snap.description
    assert q.count == 2
    assert q.front() == "A"


def test_one_slot_waste_queue_sim_comparison():
    """Verify difference between MikanOS 100% capacity design and One-Slot Waste design."""
    cap = 5
    mikan_q = ArrayQueueSim(capacity=cap)
    waste_q = OneSlotWasteQueueSim(capacity=cap)

    assert mikan_q.capacity == cap
    assert waste_q.capacity == cap
    assert waste_q.usable_capacity == cap - 1

    # Fill both until full
    m_count = 0
    while mikan_q.push(f"M{m_count}"):
        m_count += 1

    w_count = 0
    while waste_q.push(f"W{w_count}"):
        w_count += 1

    assert m_count == cap  # 5 items
    assert w_count == cap - 1  # 4 items (1 slot sacrificed)
    assert mikan_q.is_full()
    assert waste_q.is_full()


def test_figure_generations_chapter7_9(tmp_path):
    """Verify generation of Figure 7.19, 7.20, and 7.21 with valid dimensions and resolution."""
    f19 = tmp_path / "test_fig7_19.png"
    f20 = tmp_path / "test_fig7_20.png"
    f21 = tmp_path / "test_fig7_21.png"

    plot_arrayqueue_ring_buffer_structure(str(f19))
    plot_push_pop_state_transition_sequence(str(f20))
    plot_buffer_boundary_and_full_empty_logic(str(f21))

    for fpath in [f19, f20, f21]:
        assert fpath.exists(), f"Figure was not created: {fpath}"
        with Image.open(fpath) as img:
            width, height = img.size
            assert width >= 1200
            assert height >= 800
            assert img.format == "PNG"


def test_cpp20_array_queue_sim_binary():
    """Verify standalone C++20 ArrayQueue implementation builds and passes all tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch07" / "array_queue_sim.cpp"
    assert cpp_src.exists(), f"Source file not found: {cpp_src}"

    bin_path = repo_root / "build" / "bin" / "ch07_array_queue_sim"
    if not bin_path.exists():
        bin_path = repo_root / "src" / "ch07" / "array_queue_sim"
        compile_cmd = [
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic",
            str(cpp_src), "-o", str(bin_path)
        ]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "ALL CHAPTER 7.9 TESTS COMPLETED SUCCESSFULLY!" in run_res.stdout
    assert "Invariants preserved during normal operations" in run_res.stdout
    assert "Read and write pointers wrap around accurately" in run_res.stdout
    assert "Error codes kFull and kEmpty returned without corruption" in run_res.stdout
    assert "MikanOS design achieves full 100% capacity utilization" in run_res.stdout


def test_chapter7_9_notebook_validity_and_execution():
    """Verify Chapter 7.9 Jupyter Notebook exists, has valid structure and executes cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "7" / "7.9_Implementing_Ring_Buffer_ArrayQueue.ipynb"
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
    assert "7.9 キューの実装" in full_text
    assert "ArrayQueue" in full_text
    assert "read_pos_" in full_text
    assert "write_pos_" in full_text
    assert "count_" in full_text
    assert "capacity_" in full_text
    assert "kFull" in full_text
    assert "kEmpty" in full_text
    assert "リングバッファ" in full_text or "Ring Buffer" in full_text
