"""Tests for Chapter 7.10: Event Loop with Interrupt Queue."""

import json
import os
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    MessageType,
    SimMessage,
    EventLoopSim,
    plot_sti_hlt_race_condition_and_interrupt_shadow,
    plot_top_half_bottom_half_queue_sequence,
    plot_event_driven_os_state_machine,
)


def test_event_loop_sim_basic_ops():
    """Verify EventLoopSim: CLI, STI, Top Half ISR push, Bottom Half pop, and state tracking."""
    sim = EventLoopSim(capacity=8)
    assert sim.capacity == 8
    assert sim.queue.count == 0
    assert sim.interrupt_flag is True
    assert sim.cpu_halted is False
    assert len(sim.processed_events) == 0

    # Top Half ISR fires
    assert sim.int_handler_xhci(event_data=101, timestamp_us=50.0) is True
    assert sim.queue.count == 1
    assert sim.overflow_count == 0

    # Step event loop safely (Bottom Half)
    msg = sim.step_event_loop_safe()
    assert msg is not None
    assert msg.msg_type == MessageType.kInterruptXHCI
    assert msg.data == 101
    assert msg.timestamp_us == 50.0
    assert sim.queue.count == 0
    assert sim.interrupt_flag is True
    assert len(sim.processed_events) == 1

    # Loop again when empty: transitions to HLT safely
    empty_msg = sim.step_event_loop_safe()
    assert empty_msg is None
    assert sim.cpu_halted is True

    # Hardware interrupt arrives while CPU is halted: immediately wakes CPU!
    assert sim.int_handler_xhci(event_data=102, timestamp_us=100.0) is True
    assert sim.cpu_halted is False
    assert sim.queue.count == 1

    # Consume new event
    msg2 = sim.step_event_loop_safe()
    assert msg2 is not None
    assert msg2.data == 102
    assert sim.queue.count == 0


def test_event_loop_race_condition_lost_wakeup():
    """Verify that naive separate STI followed by HLT reliably reproduces the Lost Wakeup deadlock."""
    sim = EventLoopSim(capacity=8)
    # Start with empty queue
    assert sim.queue.count == 0

    # Execute flawed step where interrupt arrives after STI but before HLT
    res = sim.step_event_loop_flawed_with_injected_interrupt(event_data=999)
    assert res["deadlock"] is True
    assert res["pending_count"] == 1
    assert sim.lost_wakeup_occurred is True
    assert sim.cpu_halted is True
    assert sim.queue.count == 1

    # Verify trace steps confirm sequence
    trace_actions = [step[0] for step in res["trace"]]
    assert "CLI" in trace_actions
    assert "QUEUE_EMPTY" in trace_actions
    assert "STI" in trace_actions
    assert "INTERRUPT_FIRED" in trace_actions
    assert "HLT_LOST_WAKEUP" in trace_actions


def test_figure_generations_chapter7_10():
    """Verify Figure 7.22, 7.23, and 7.24 are generated at 300 DPI with valid dimensions."""
    repo_root = Path(__file__).resolve().parent.parent
    fig_dir = repo_root / "7" / "result"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig7_22 = fig_dir / "fig7_22_sti_hlt_race_condition_and_interrupt_shadow.png"
    fig7_23 = fig_dir / "fig7_23_top_half_bottom_half_queue_sequence.png"
    fig7_24 = fig_dir / "fig7_24_event_driven_os_state_machine.png"

    plot_sti_hlt_race_condition_and_interrupt_shadow(str(fig7_22))
    plot_top_half_bottom_half_queue_sequence(str(fig7_23))
    plot_event_driven_os_state_machine(str(fig7_24))

    for fpath in [fig7_22, fig7_23, fig7_24]:
        assert fpath.exists(), f"Figure not found: {fpath}"
        assert fpath.stat().st_size > 50_000, f"File too small: {fpath}"
        with Image.open(fpath) as img:
            width, height = img.size
            assert width >= 1200
            assert height >= 800
            assert img.format == "PNG"


def test_cpp20_event_loop_sim_binary():
    """Verify standalone C++20 Event Loop simulator builds and passes all assertions."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch07" / "event_loop_sim.cpp"
    assert cpp_src.exists(), f"Source file not found: {cpp_src}"

    bin_path = repo_root / "build" / "bin" / "ch07_event_loop_sim"
    if not bin_path.exists():
        bin_path = repo_root / "src" / "ch07" / "event_loop_sim"
        compile_cmd = [
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", "-pthread",
            str(cpp_src), "-o", str(bin_path)
        ]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "ALL C++20 EVENT LOOP & QUEUE TESTS PASSED SUCCESSFULLY!" in run_res.stdout
    assert "Queue correctly rejected overflow when full" in run_res.stdout
    assert "Lost wakeup vulnerability triggered" in run_res.stdout
    assert "Atomic wake-up handled perfectly" in run_res.stdout
    assert "End-to-End MikanOS Day 07b Event Loop Execution" in run_res.stdout


def test_chapter7_10_notebook_validity_and_execution():
    """Verify Chapter 7.10 Jupyter Notebook exists, has valid structure and executes cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "7" / "7.10_Event_Loop_with_Interrupt_Queue.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 15

    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 5

    for idx, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Error in cell {idx}: {out.get('ename')}: {out.get('evalue')}\n"
                f"{''.join(out.get('traceback', []))}"
            )

    full_text = json.dumps(nb_data, ensure_ascii=False)
    assert "7.10 キューを使った割り込み高速化" in full_text
    assert "Message" in full_text
    assert "IntHandlerXHCI" in full_text
    assert "ProcessEvent" in full_text
    assert "cli" in full_text.lower()
    assert "sti" in full_text.lower()
    assert "hlt" in full_text.lower()
    assert "Interrupt Shadow" in full_text or "割り込みシャドウ" in full_text
