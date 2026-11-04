"""Tests for Chapter 7.2: Interrupt Handlers and Compiler Attribute Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    RegisterProtectionType,
    X86RegisterSpec,
    SYSTEM_V_AMD64_REGISTERS,
    StackFrameAlignmentSimulator,
    InterruptHandlerSimulator,
    plot_regular_call_vs_interrupt_stack,
    plot_compiler_attribute_interrupt_mechanism,
)


def test_abi_register_classification():
    """Verify System V AMD64 ABI register classification and counts."""
    caller_saved = [name for name, spec in SYSTEM_V_AMD64_REGISTERS.items()
                    if spec.protection_type == RegisterProtectionType.CALLER_SAVED]
    callee_saved = [name for name, spec in SYSTEM_V_AMD64_REGISTERS.items()
                    if spec.protection_type == RegisterProtectionType.CALLEE_SAVED]
    special = [name for name, spec in SYSTEM_V_AMD64_REGISTERS.items()
               if spec.protection_type == RegisterProtectionType.SPECIAL]

    assert len(caller_saved) == 9  # rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11
    assert "rax" in caller_saved
    assert "rdi" in caller_saved
    assert "r11" in caller_saved

    assert len(callee_saved) == 6  # rbx, rbp, r12, r13, r14, r15
    assert "rbx" in callee_saved
    assert "rbp" in callee_saved
    assert "r15" in callee_saved

    assert len(special) == 3       # rsp, rip, rflags
    assert "rsp" in special
    assert "rip" in special
    assert "rflags" in special


def test_stack_frame_alignment_simulator():
    """Verify 16-byte stack alignment transitions and frame offsets."""
    assert StackFrameAlignmentSimulator.is_16byte_aligned(0x7FFF_FFFF_E000)
    assert not StackFrameAlignmentSimulator.is_16byte_aligned(0x7FFF_FFFF_E008)

    # Standard hardware interrupt (no error code: 40 bytes)
    sim_std = StackFrameAlignmentSimulator.simulate_interrupt_entry(0x7FFF_FFFF_E000, has_error_code=False)
    assert sim_std["hardware_push_size"] == 40
    assert sim_std["rsp_after_hw_push"] == 0x7FFF_FFFF_DFD8
    assert not sim_std["is_hw_pushed_rsp_16aligned"]  # -40 % 16 == 8
    assert sim_std["frame_offsets"]["rip"] == 0
    assert sim_std["frame_offsets"]["cs"] == 8
    assert sim_std["frame_offsets"]["rflags"] == 16
    assert sim_std["frame_offsets"]["rsp"] == 24
    assert sim_std["frame_offsets"]["ss"] == 32

    # Exception with error code: 48 bytes (48 % 16 == 0)
    sim_err = StackFrameAlignmentSimulator.simulate_interrupt_entry(0x7FFF_FFFF_E000, has_error_code=True)
    assert sim_err["hardware_push_size"] == 48
    assert sim_err["rsp_after_hw_push"] == 0x7FFF_FFFF_DFD0
    assert sim_err["is_hw_pushed_rsp_16aligned"]  # -48 % 16 == 0


def test_interrupt_handler_simulator_corrupting_hazard():
    """Verify detection of register corruption when using regular C functions as ISRs."""
    sim = InterruptHandlerSimulator()
    # Handler uses caller-saved registers without saving them
    hazard_res = sim.simulate_corrupting_regular_handler(["rax", "rcx", "rdx", "rbx"])

    assert not hazard_res["preserved"]
    assert hazard_res["corrupted_count"] == 3  # rax, rcx, rdx corrupted (rbx is callee-saved)
    assert "rax" in hazard_res["corrupted_registers"]
    assert "rcx" in hazard_res["corrupted_registers"]
    assert "rdx" in hazard_res["corrupted_registers"]
    assert "rbx" not in hazard_res["corrupted_registers"]


def test_interrupt_handler_simulator_attributed_preservation():
    """Verify that __attribute__((interrupt)) ensures complete context preservation."""
    sim = InterruptHandlerSimulator()
    safe_res = sim.simulate_attributed_interrupt_handler(["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9"])

    assert safe_res["preserved"]
    assert safe_res["corrupted_count"] == 0
    assert safe_res["final_registers"] == sim.initial_registers


def test_cpp20_standalone_interrupt_handler_simulation():
    """Verify standalone C++20 simulation compilation and execution."""
    src_file = Path("src/ch07/interrupt_handler_sim.cpp")
    bin_file = Path("src/ch07/interrupt_handler_sim")

    assert src_file.exists(), f"Source file {src_file} does not exist"

    # Compile
    compile_cmd = f"g++ -std=c++20 -O2 -Wall -Wextra -pedantic {src_file} -o {bin_file}"
    res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    # Run
    run_res = subprocess.run(str(bin_file), capture_output=True, text=True)
    assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
    assert "InterruptFrame Total Size: 40 bytes" in run_res.stdout
    assert "Context 100% intact after IRETQ:      YES" in run_res.stdout
    assert "ALL CHAPTER 7.2 VERIFICATIONS COMPLETED SUCCESSFULLY!" in run_res.stdout


def test_figure_generation_and_validity():
    """Verify that Figures 7.03 and 7.04 are generated and valid images."""
    p1 = Path(plot_regular_call_vs_interrupt_stack("7/result/fig7_03_regular_call_vs_interrupt_stack.png"))
    p2 = Path(plot_compiler_attribute_interrupt_mechanism("7/result/fig7_04_compiler_attribute_interrupt_mechanism.png"))

    assert p1.exists(), f"Figure 7.03 missing: {p1}"
    assert p2.exists(), f"Figure 7.04 missing: {p2}"

    assert p1.stat().st_size > 50000
    assert p2.stat().st_size > 50000

    with Image.open(p1) as img1:
        assert img1.width >= 1000
        assert img1.height >= 800

    with Image.open(p2) as img2:
        assert img2.width >= 1000
        assert img2.height >= 800


def test_notebook_structure_and_error_free_execution():
    """Verify Chapter 7.2 Jupyter Notebook execution integrity."""
    nb_path = Path("7/7.2_Interrupt_Handlers_and_Compiler_Attribute.ipynb")
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
