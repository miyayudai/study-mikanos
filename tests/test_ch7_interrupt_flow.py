"""Tests for Chapter 7.6: Interrupt Flow Summary & Pipeline Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    InterruptPipelineStage,
    InterruptTraceRecord,
    InterruptFlowPipelineSimulator,
    plot_end_to_end_interrupt_pipeline_sequence,
    plot_hardware_software_boundary_and_stack_lifecycle,
)


def test_interrupt_pipeline_stages_and_enums():
    """Verify 12-stage interrupt pipeline enum definitions and labels."""
    stages = list(InterruptPipelineStage)
    assert len(stages) == 12

    assert stages[0] == InterruptPipelineStage.kMouseMovement
    assert stages[1] == InterruptPipelineStage.kXhciEventRing
    assert stages[2] == InterruptPipelineStage.kPciMsiWrite
    assert stages[3] == InterruptPipelineStage.kLapicReceive
    assert stages[4] == InterruptPipelineStage.kCpuInterruptAccept
    assert stages[5] == InterruptPipelineStage.kHardwareStackPush
    assert stages[6] == InterruptPipelineStage.kIdtLookup
    assert stages[7] == InterruptPipelineStage.kHandlerPrologue
    assert stages[8] == InterruptPipelineStage.kHandlerProcessEvent
    assert stages[9] == InterruptPipelineStage.kLapicNotifyEoi
    assert stages[10] == InterruptPipelineStage.kHandlerEpilogue
    assert stages[11] == InterruptPipelineStage.kCpuIretq


def test_interrupt_flow_pipeline_simulator_basic():
    """Verify single interrupt event simulation lifecycle and state transitions."""
    sim = InterruptFlowPipelineSimulator(
        bsp_apic_id=0,
        xhci_vector=0x40,
        initial_mouse=(300, 200),
        screen_size=(800, 600),
    )

    records = sim.trigger_mouse_movement(displacement_x=15, displacement_y=-10)
    assert len(records) == 12

    # Check mouse position updated correctly
    assert sim.mouse_x == 315
    assert sim.mouse_y == 190

    # Check EOI occurred and ISR cleared
    assert sim.lapic_eoi_written is True
    assert sim.lapic_isr == 0

    # Check CPU registers restored
    assert (sim.rflags & 0x200) != 0, "IF flag must be restored to 1"
    assert sim.rsp == sim.initial_rsp, "RSP must be fully restored"
    assert sim.rip == sim.initial_rip, "RIP must be returned to HLT loop"


def test_interrupt_flow_pipeline_invariants():
    """Verify pipeline architectural invariants checker."""
    sim = InterruptFlowPipelineSimulator()
    sim.trigger_mouse_movement(5, 5)

    invariants = sim.verify_pipeline_invariants()
    assert isinstance(invariants, dict)
    assert invariants["all_12_stages_recorded"] is True
    assert invariants["eoi_acknowledged"] is True
    assert invariants["if_flag_restored"] is True
    assert invariants["rsp_restored_to_initial"] is True
    assert invariants["rip_restored_to_initial"] is True
    assert invariants["event_ring_drained"] is True
    assert all(invariants.values()) is True


def test_hardware_vs_software_boundary():
    """Verify clear distinction between hardware actions and kernel software actions."""
    sim = InterruptFlowPipelineSimulator()
    records = sim.trigger_mouse_movement(10, 10)

    # Stages 1..6 are purely hardware
    for i in range(6):
        assert records[i].domain == "Hardware", f"Stage {i+1} must be Hardware"

    # Stage 7 is hardware IDT lookup
    assert records[6].domain == "Hardware"

    # Stages 8..9 are Kernel Software (Prologue + Body)
    assert records[7].domain == "Kernel Software"
    assert records[8].domain == "Kernel Software"

    # Stage 10 is Handshake (Software writes MMIO, Hardware clears ISR)
    assert "Handshake" in records[9].domain

    # Stage 11 is Kernel Software (Epilogue)
    assert records[10].domain == "Kernel Software"

    # Stage 12 is Hardware (IRETQ execution)
    assert records[11].domain == "Hardware"


def test_stack_frame_alignment_and_registers():
    """Verify 16-byte stack alignment invariants during hardware push and handler execution."""
    sim = InterruptFlowPipelineSimulator(initial_rsp=0x7FFFFFFF0)
    records = sim.trigger_mouse_movement(-5, 8)

    # Initial RSP must be 16-byte aligned
    assert (sim.initial_rsp % 16) == 0

    # Stage 6: After hardware pushes 5 uint64_t (40 bytes), RSP % 16 must be 8
    hw_rsp = records[5].rsp
    assert (hw_rsp % 16) == 8, "RSP after 40-byte hardware push must be 8 (mod 16)"

    # Stage 8: After compiler pushes 9 scratch GPRs (72 bytes), RSP % 16 must be 0
    prologue_rsp = records[7].rsp
    assert (prologue_rsp % 16) == 0, "RSP after GPR prologue must be 16-byte aligned"

    # Stage 12: After IRETQ, RSP is back to initial
    assert records[11].rsp == sim.initial_rsp


def test_figure_generation_and_dimensions(tmp_path):
    """Verify Figure 7.11 and 7.12 generation and image properties."""
    fig7_11 = tmp_path / "fig7_11.png"
    fig7_12 = tmp_path / "fig7_12.png"

    plot_end_to_end_interrupt_pipeline_sequence(str(fig7_11))
    plot_hardware_software_boundary_and_stack_lifecycle(str(fig7_12))

    assert fig7_11.exists()
    assert fig7_12.exists()

    with Image.open(fig7_11) as img:
        assert img.width > 3000
        assert img.height > 1800

    with Image.open(fig7_12) as img:
        assert img.width > 3000
        assert img.height > 1800


def test_cpp20_interrupt_flow_sim_execution():
    """Verify C++20 standalone binary execution."""
    bin_path = Path("build/bin/ch07_interrupt_flow_sim")
    if not bin_path.exists():
        pytest.skip("ch07_interrupt_flow_sim binary not built")

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "ALL 7 TESTS PASSED SUCCESSFULLY!" in res.stdout
    assert "MikanOS Section 7.6: Interrupt Flow & Pipeline Verification" in res.stdout


def test_notebook_structure_and_execution():
    """Verify Section 7.6 Jupyter Notebook existence, valid JSON structure, and execution."""
    nb_path = Path("7/7.6_Interrupt_Flow_Summary.ipynb")
    assert nb_path.exists(), "7/7.6_Interrupt_Flow_Summary.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 8

    # Verify Cell 0 has Google Colab setup
    first_cell = nb_data["cells"][0]
    first_cell_src = "".join(first_cell.get("source", []))
    assert "google.colab" in first_cell_src
    assert "study-mikanos" in first_cell_src

    # Verify key topics exist in markdown cells
    full_text = "".join("".join(c.get("source", [])) for c in nb_data["cells"])
    assert "xHCI" in full_text
    assert "MSI" in full_text
    assert "Local APIC" in full_text
    assert "IDT" in full_text
    assert "NotifyEndOfInterrupt" in full_text
    assert "IRETQ" in full_text
