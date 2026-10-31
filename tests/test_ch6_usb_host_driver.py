"""Tests for Chapter 6.2: USB 3.0 & xHCI Architecture Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    TRBType,
    TRBCompletionCode,
    TRBSim,
    LinkTRBSim,
    RingSimulator,
    SlotContextSim,
    EndpointContextSim,
    xHCIArchitectureSimulator,
    plot_fig6_03_xhci_data_structures_and_ring_hierarchy,
    plot_fig6_04_trb_binary_layout_and_cycle_bit_transition,
)


def test_trb_enum_definitions():
    """Verify standard xHCI TRB types and completion codes."""
    assert TRBType.NORMAL == 1
    assert TRBType.SETUP_STAGE == 2
    assert TRBType.DATA_STAGE == 3
    assert TRBType.STATUS_STAGE == 4
    assert TRBType.LINK == 6
    assert TRBType.ENABLE_SLOT_COMMAND == 9
    assert TRBType.TRANSFER_EVENT == 32
    assert TRBType.COMMAND_COMPLETION_EVENT == 33

    assert TRBCompletionCode.SUCCESS == 1
    assert TRBCompletionCode.STALL_ERROR == 6


def test_trb_binary_packing_and_unpacking():
    """Verify 16-byte TRB binary serialization and bitfield alignment."""
    trb = TRBSim(
        parameter=0x123456789ABCDEF0,
        status=0x00000040,
        cycle_bit=1,
        evaluate_next_trb=0,
        trb_type=TRBType.NORMAL,
        control_flags=0x0000,
    )
    raw = trb.to_bytes()
    assert len(raw) == 16, "TRB must be exactly 16 bytes"

    unpacked = TRBSim.from_bytes(raw)
    assert unpacked.parameter == 0x123456789ABCDEF0
    assert unpacked.status == 0x00000040
    assert unpacked.cycle_bit == 1
    assert unpacked.trb_type == TRBType.NORMAL


def test_link_trb_configuration():
    """Verify Link TRB specifies next ring segment pointer and toggle cycle flag."""
    link = LinkTRBSim(ring_segment_pointer=0x2000, toggle_cycle=True)
    raw = link.to_bytes()
    assert len(raw) == 16

    unpacked = TRBSim.from_bytes(raw)
    assert unpacked.trb_type == TRBType.LINK
    assert unpacked.parameter == 0x2000


def test_ring_simulator_cycle_bit_toggle():
    """Verify ring buffer manages cycle bit inversion when traversing Link TRB."""
    ring = RingSimulator(buf_size=4, ring_address=0x1000)
    assert ring.cycle_bit == 1
    assert ring.write_index == 0

    # Push 3 Normal TRBs (slots 0, 1, 2)
    for i in range(3):
        trb = TRBSim(parameter=0x100 + i, status=0, cycle_bit=0, trb_type=TRBType.NORMAL)
        ring.push(trb)

    # After 3 items in a 4-slot ring (last slot is Link TRB),
    # write_index should wrap to 0 and cycle_bit should toggle to 0!
    assert ring.write_index == 0
    assert ring.cycle_bit == 0

    # Verify that the Link TRB was auto-injected at index 3
    link_trb = ring.buf[3]
    assert link_trb.trb_type == TRBType.LINK
    assert link_trb.parameter == 0x1000  # Wrap back to start

    # Next push should be written with cycle_bit = 0 (lap 2)
    trb_lap2 = TRBSim(parameter=0x200, status=0, cycle_bit=1, trb_type=TRBType.NORMAL)
    ring.push(trb_lap2)
    assert ring.buf[0].cycle_bit == 0
    assert ring.write_index == 1
    assert ring.cycle_bit == 0


def test_slot_and_endpoint_context_sizes():
    """Verify context data structures conform to xHCI 32-byte layout."""
    slot = SlotContextSim(route_string=0, speed=4, context_entries=1, root_hub_port_num=1)
    raw_slot = slot.to_bytes()
    assert len(raw_slot) == 32, "SlotContext must be 32 bytes"

    ep = EndpointContextSim(ep_type=4, max_packet_size=64, tr_dequeue_pointer=0x4000)
    raw_ep = ep.to_bytes()
    assert len(raw_ep) == 32, "EndpointContext must be 32 bytes"


def test_xhci_architecture_simulator_pipeline():
    """Verify command ring push, doorbell notification, and event ring completion generation."""
    sim = xHCIArchitectureSimulator(max_slots=8, command_ring_size=8, event_ring_size=16)

    # Initial DCBAA size
    assert len(sim.dcbaa) == 9  # slot 0 + 8 device slots

    # Enqueue Enable Slot Command
    cmd = TRBSim(parameter=0, status=0, cycle_bit=0, trb_type=TRBType.ENABLE_SLOT_COMMAND)
    sim.command_ring.push(cmd)

    # Ring host controller doorbell (slot 0, target 0)
    msg = sim.ring_doorbell(slot_id=0, target=0)
    assert "Controller executed 1 command TRBs" in msg

    # Verify event ring has received Command Completion Event
    event = sim.event_ring[0]
    assert event.trb_type == TRBType.COMMAND_COMPLETION_EVENT
    assert event.cycle_bit == 1

    # Verify device context address assigned for newly enabled slot (slot 1)
    assert sim.dcbaa[1] == 0x10000 + 1 * 1024
    assert len(sim.command_history) == 1
    assert "Allocated Slot ID 1" in sim.command_history[0]


def test_cpp_standalone_simulation_execution():
    """Verify C++20 standalone simulator builds and passes all tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch06" / "xhci_arch_sim.cpp"
    assert cpp_src.exists()

    bin_path = repo_root / "build" / "bin" / "ch06_xhci_arch_sim"
    if not bin_path.exists():
        subprocess.run(
            ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", str(cpp_src), "-o", "/tmp/ch06_xhci_arch_sim"],
            check=True
        )
        bin_path = Path("/tmp/ch06_xhci_arch_sim")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert "All Chapter 6.2 xHCI architecture simulation tests PASSED!" in res.stdout


def test_figure_generation_and_validity(tmp_path):
    """Verify generation of Figure 6.03 and 6.04 with high DPI and valid image properties."""
    fig1_path = tmp_path / "fig6_03.png"
    fig2_path = tmp_path / "fig6_04.png"

    plot_fig6_03_xhci_data_structures_and_ring_hierarchy(str(fig1_path))
    plot_fig6_04_trb_binary_layout_and_cycle_bit_transition(str(fig2_path))

    assert fig1_path.exists()
    assert fig2_path.exists()
    assert fig1_path.stat().st_size > 100_000
    assert fig2_path.stat().st_size > 100_000

    with Image.open(fig1_path) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 2000

    with Image.open(fig2_path) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 2000


def test_notebook_structure_and_error_free_execution():
    """Verify 6/6.2_USB_3_0_xHCI_Architecture.ipynb exists, is valid JSON, and has no execution errors."""
    nb_path = Path("6/6.2_USB_3_0_xHCI_Architecture.ipynb")
    assert nb_path.exists(), "Notebook 6.2 must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10, "Notebook must contain at least 10 cells"

    # Check for execution errors in cell outputs
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", f"Cell {idx} contains an error output: {output}"
