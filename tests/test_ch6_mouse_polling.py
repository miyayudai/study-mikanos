"""Tests for Chapter 6.4: Mouse Input via xHCI Polling Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    LogLevel,
    KernelLogger,
    PCIBAR,
    MouseMovementSimulator,
    XHCIPollingSimulator,
    plot_fig6_07_pci_bar_and_mmio_mapping,
    plot_fig6_08_xhci_polling_sequence_and_mouse_event_flow,
)


def test_pci_bar_decoding_and_mmio_base():
    """Verify PCI BAR decoding for 32-bit and 64-bit MMIO addresses and flag clearing."""
    # 32-bit Memory BAR: Base = 0xC0000000, Type = 00b, PF = 0, Memory = 0
    bar32 = PCIBAR(raw_lower=0xC0000000, raw_upper=0, bar_index=0)
    assert bar32.is_mmio is True
    assert bar32.is_io is False
    assert bar32.is_64bit is False
    assert bar32.is_prefetchable is False
    assert bar32.base_address == 0xC0000000

    # 64-bit Memory BAR: Base = 0xFEA00000, Type = 10b (Bit 2 == 1 -> value & 4u), PF = 0
    # Lower DWORD has flags: Bit 2 is set (value = 0xFEA00004), Upper DWORD = 0x00000001
    bar64 = PCIBAR(raw_lower=0xFEA00004, raw_upper=0x00000001, bar_index=0)
    assert bar64.is_mmio is True
    assert bar64.is_64bit is True
    assert bar64.raw_combined == 0x0000_0001_FEA0_0004
    # Column 6.2 math: clearing lower 4 bits (flags) gives physical base address
    assert bar64.base_address == 0x0000_0001_FEA0_0000

    # Prefetchable BAR
    bar_pf = PCIBAR(raw_lower=0xD0000008, raw_upper=0)
    assert bar_pf.is_prefetchable is True
    assert bar_pf.base_address == 0xD0000000


def test_column6_2_static_cast_mystery():
    """Verify Column 6.2 explanation of static_cast<uint64_t>(0xf)."""
    expl = PCIBAR.explain_static_cast_mystery()
    assert "static_cast<uint64_t>(0xf)" in expl["issue"]
    assert "0xFFFFFFFFFFFFFFF0" in expl["cast_then_tilde"]

    # Python bitwise demonstration of ~0xf in 64-bit unsigned space
    mask_64 = (~0xF) & 0xFFFF_FFFF_FFFF_FFFF
    assert mask_64 == 0xFFFF_FFFF_FFFF_FFF0
    addr = 0x1234_5678_FEA0_0004
    assert (addr & mask_64) == 0x1234_5678_FEA0_0000


def test_kernel_logger_thresholds():
    """Verify KernelLogger filtering by log levels (Column 6.1)."""
    assert int(LogLevel.ERROR) < int(LogLevel.WARN) < int(LogLevel.INFO) < int(LogLevel.DEBUG)

    logger = KernelLogger(level=LogLevel.WARN)
    # ERROR (3) <= WARN (4) -> logged
    assert logger.log(LogLevel.ERROR, "Critical test error") is True
    # WARN (4) <= WARN (4) -> logged
    assert logger.log(LogLevel.WARN, "Port warning") is True
    # INFO (6) > WARN (4) -> filtered out
    assert logger.log(LogLevel.INFO, "Informational message") is False
    # DEBUG (7) > WARN (4) -> filtered out
    assert logger.log(LogLevel.DEBUG, "Debug message") is False

    assert len(logger.logs) == 2
    assert logger.logs[0][0] == LogLevel.ERROR
    assert logger.logs[1][0] == LogLevel.WARN

    # Raise threshold to DEBUG
    logger.set_level(LogLevel.DEBUG)
    assert logger.log(LogLevel.DEBUG, "Now debug is logged") is True
    assert len(logger.logs) == 3


def test_mouse_movement_simulator_relative_and_erase():
    """Verify MouseMovementSimulator position tracking, movement, and erase recording."""
    sim = MouseMovementSimulator(fb_width=800, fb_height=600, initial_pos=(300, 200))
    assert sim.current_position == (300, 200)
    assert len(sim.history) == 1
    assert len(sim.erased_positions) == 0

    # Move mouse relative
    sim.move_relative(15, -10)
    assert sim.current_position == (315, 190)
    assert len(sim.erased_positions) == 1
    assert sim.erased_positions[0] == (300, 200)
    assert len(sim.drawn_positions) == 2
    assert sim.drawn_positions[-1] == (315, 190)

    # Render to numpy array and verify dimensions
    img = sim.render_to_numpy()
    assert img.shape == (600, 800, 3)
    # Cursor at (315, 190) has top-left '@' = (0, 0, 0)
    assert tuple(img[190, 315]) == (0, 0, 0)
    # Point outside cursor is background color
    assert tuple(img[10, 10]) == (45, 118, 237)


def test_xhci_polling_simulator_event_queue():
    """Verify XHCIPollingSimulator event queuing, dispatch, and polling."""
    xhc = XHCIPollingSimulator(mmio_base=0xFEA00000)
    assert xhc.initialize() is True
    xhc.run()
    assert xhc.is_running is True

    # Empty queue poll
    assert xhc.process_event() is False

    # Queue events
    xhc.queue_mouse_packet(3, -2)
    xhc.queue_mouse_packet(1, 4)
    assert len(xhc.event_queue) == 2

    # Process events one by one
    assert xhc.process_event() is True
    assert len(xhc.observer_events) == 1
    assert xhc.observer_events[0] == (3, -2)

    assert xhc.process_event() is True
    assert len(xhc.observer_events) == 2
    assert xhc.observer_events[1] == (1, 4)

    # Now empty again
    assert xhc.process_event() is False

    # Batch polling
    xhc.queue_mouse_packet(5, 5)
    xhc.queue_mouse_packet(-1, -1)
    polled = xhc.poll_all()
    assert polled == 2
    assert len(xhc.observer_events) == 4


def test_cpp20_standalone_mouse_polling_simulation():
    """Verify C++20 standalone simulation binary builds and passes all unit tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch06" / "mouse_polling_sim.cpp"
    assert cpp_src.exists()

    bin_path = repo_root / "build" / "bin" / "ch06_mouse_polling_sim"
    if not bin_path.exists():
        subprocess.run(
            ["cmake", "--build", str(repo_root / "build"), "--target", "ch06_mouse_polling_sim"],
            check=True
        )

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert "All Chapter 6.4 mouse polling simulation tests PASSED" in res.stdout


def test_figure_generation_and_validity(tmp_path):
    """Verify generation of Figure 6.07 and 6.08 with high DPI and valid image properties."""
    fig1_path = tmp_path / "fig6_07.png"
    fig2_path = tmp_path / "fig6_08.png"

    plot_fig6_07_pci_bar_and_mmio_mapping(str(fig1_path))
    plot_fig6_08_xhci_polling_sequence_and_mouse_event_flow(str(fig2_path))

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
    """Verify 6/6.4_Mouse_Input_via_xHCI_Polling.ipynb exists, is valid JSON, and has no execution errors."""
    nb_path = Path("6/6.4_Mouse_Input_via_xHCI_Polling.ipynb")
    assert nb_path.exists(), "Notebook 6.4 must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10, "Notebook must contain at least 10 cells"

    # Check for execution errors in cell outputs
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", f"Cell {idx} contains an error output: {output}"
