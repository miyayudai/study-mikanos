"""Tests for Chapter 7.1: Concept of Interrupts Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    X86InterruptCategory,
    InterruptStackFrame,
    PollingVsInterruptBenchmarker,
    plot_fig7_01_polling_vs_interrupt_timeline,
    plot_fig7_02_interrupt_classification_and_lifecycle,
)


def test_x86_interrupt_categories():
    """Verify x86-64 interrupt and exception categorization."""
    assert X86InterruptCategory.EXTERNAL_HARDWARE.name == "EXTERNAL_HARDWARE"
    assert "Asynchronous" in X86InterruptCategory.EXTERNAL_HARDWARE.value
    assert "Synchronous" in X86InterruptCategory.SOFTWARE_INTERRUPT.value
    assert "Restartable" in X86InterruptCategory.CPU_EXCEPTION_FAULT.value
    assert "Next-Instruction" in X86InterruptCategory.CPU_EXCEPTION_TRAP.value
    assert "Unrecoverable" in X86InterruptCategory.CPU_EXCEPTION_ABORT.value


def test_interrupt_stack_frame_layout():
    """Verify hardware interrupt stack frame layout and size."""
    frame_std = InterruptStackFrame(
        rip=0x104000,
        cs=0x08,
        rflags=0x202,
        rsp=0x7FFFFFFFE000,
        ss=0x10,
    )
    assert frame_std.total_bytes == 40
    dump = frame_std.dump_hex()
    assert len(dump) == 5
    assert "+40 SS" in dump[0]
    assert "+32 RSP" in dump[1]
    assert "+24 RFLAGS" in dump[2]
    assert "+16 CS" in dump[3]
    assert "+08 RIP" in dump[4]

    frame_err = InterruptStackFrame(
        rip=0x104000,
        cs=0x08,
        rflags=0x202,
        rsp=0x7FFFFFFFE000,
        ss=0x10,
        error_code=0x02,  # Page fault write violation
    )
    assert frame_err.total_bytes == 48
    dump_err = frame_err.dump_hex()
    assert len(dump_err) == 6
    assert "+00 ERR" in dump_err[5]


def test_polling_vs_interrupt_benchmark_simulation():
    """Verify mathematical benchmark comparison between polling and interrupt-driven modes."""
    benchmarker = PollingVsInterruptBenchmarker(
        loop_interval_ms=10.0,
        task_workload_ms=8.0,
        handler_execution_us=25.0,
        cpu_idle_power_w=5.0,
        cpu_active_power_w=65.0,
    )
    events = [12.0, 28.0, 45.0, 63.0, 81.0]
    pol_res, int_res = benchmarker.run_simulation(total_time_ms=100.0, event_times_ms=events)

    # 1. Events count
    assert pol_res.num_events == len(events)
    assert int_res.num_events == len(events)

    # 2. Polling vs Interrupt Latencies
    assert pol_res.avg_latency_ms > 0.0
    assert int_res.avg_latency_ms < 0.01  # ~0.005 ms
    assert int_res.avg_latency_ms < pol_res.avg_latency_ms
    assert int_res.max_latency_ms < pol_res.max_latency_ms

    # 3. CPU Utilization: Polling is 100%, Interrupt is fractional
    assert pol_res.cpu_utilization_pct == 100.0
    assert int_res.cpu_utilization_pct < 1.0  # < 1%

    # 4. Energy Consumption
    assert int_res.energy_consumed_joules < pol_res.energy_consumed_joules


def test_cpp20_standalone_interrupt_simulation():
    """Verify standalone C++20 simulation compilation and execution."""
    src_file = Path("src/ch07/interrupt_concept_sim.cpp")
    bin_file = Path("src/ch07/interrupt_concept_sim")

    assert src_file.exists(), f"Source file {src_file} does not exist"

    # Compile
    compile_cmd = f"g++ -std=c++20 -O2 -Wall -Wextra -pedantic {src_file} -o {bin_file}"
    res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    # Run
    run_res = subprocess.run(str(bin_file), capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "sizeof(InterruptFrame) = 40 bytes" in run_res.stdout
    assert "sizeof(InterruptFrameWithError) = 48 bytes" in run_res.stdout
    assert "All Chapter 7.1 C++20 assertions passed successfully!" in run_res.stdout


def test_figure_generation_and_validity():
    """Verify that Figures 7.01 and 7.02 are generated and valid images."""
    p1 = Path(plot_fig7_01_polling_vs_interrupt_timeline("7/result/fig7_01_polling_vs_interrupt_timeline.png"))
    p2 = Path(plot_fig7_02_interrupt_classification_and_lifecycle("7/result/fig7_02_interrupt_classification_and_lifecycle.png"))

    assert p1.exists(), f"Figure 7.01 missing: {p1}"
    assert p2.exists(), f"Figure 7.02 missing: {p2}"

    assert p1.stat().st_size > 50000
    assert p2.stat().st_size > 50000

    with Image.open(p1) as img1:
        assert img1.width >= 1000
        assert img1.height >= 800

    with Image.open(p2) as img2:
        assert img2.width >= 1000
        assert img2.height >= 600


def test_notebook_structure_and_error_free_execution():
    """Verify Chapter 7.1 Jupyter Notebook execution integrity."""
    nb_path = Path("7/7.1_Concept_of_Interrupts.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    cells = nb_data["cells"]
    assert len(cells) >= 10

    # Verify Colab setup in cell 0
    assert "google.colab" in cells[0]["source"][0] or any("google.colab" in line for line in cells[0]["source"])

    # Verify no execution error outputs in any code cells
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"Cell {idx} has an execution error: {out.get('ename')}: {out.get('evalue')}"
                )
