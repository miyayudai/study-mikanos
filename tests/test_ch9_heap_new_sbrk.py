"""Tests for Chapter 9.2: Dynamic Memory Allocation, sbrk(), and MikanOS Heap Architecture."""

import json
import os
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    SbrkHeapSimulator,
    MikanBitmapMemoryManagerSim,
    plot_fig9_04_traditional_unix_heap_vs_mikanos_heap,
    plot_fig9_05_dynamic_allocation_pipeline_new_to_sbrk,
    plot_fig9_06_sbrk_program_break_transition_and_fragmentation,
)


def test_sbrk_heap_simulator_initialization():
    """Verify heap initialization via BitmapMemoryManager frames."""
    mem_mgr = MikanBitmapMemoryManagerSim(frame_count=65536)
    sim = SbrkHeapSimulator(memory_manager=mem_mgr, heap_frames=32768)

    info = sim.initialize_heap()
    assert info["heap_frames"] == 32768
    assert info["total_mib"] == 128
    assert info["total_bytes"] == 128 * 1024 * 1024
    assert sim.heap_start > 0
    assert sim.current_break == sim.heap_start
    assert sim.used_bytes == 0
    assert sim.remaining_bytes == 128 * 1024 * 1024
    assert sim.errno == 0


def test_sbrk_advancement_and_query():
    """Verify sbrk(0) query and sbrk(incr) advancement."""
    sim = SbrkHeapSimulator()
    sim.initialize_heap()

    # Query without advance
    b0 = sim.sbrk(0)
    assert b0 == sim.current_break

    # Advance by 4096 bytes
    b1 = sim.sbrk(4096)
    assert b1 == b0
    assert sim.current_break == b0 + 4096
    assert sim.used_bytes == 4096
    assert sim.remaining_bytes == 128 * 1024 * 1024 - 4096

    # Advance again
    b2 = sim.sbrk(8192)
    assert b2 == b0 + 4096
    assert sim.current_break == b0 + 4096 + 8192
    assert sim.used_bytes == 12288


def test_sbrk_enomem_bounds():
    """Verify sbrk bounds checking and ENOMEM setting on overflow."""
    sim = SbrkHeapSimulator()
    sim.initialize_heap()

    rem = sim.remaining_bytes
    # Requesting rem + 1 should fail
    fail_res = sim.sbrk(rem + 1)
    assert fail_res == -1
    assert sim.errno == SbrkHeapSimulator.ENOMEM

    # Requesting negative beyond heap_start should fail
    sim.sbrk(4096)
    fail_neg = sim.sbrk(-10000)
    assert fail_neg == -1
    assert sim.errno == SbrkHeapSimulator.ENOMEM


def test_malloc_and_free_simulation():
    """Verify chunk allocation, 16-byte alignment, and free coalescing."""
    sim = SbrkHeapSimulator()
    sim.initialize_heap()

    p1 = sim.malloc_sim(32)
    p2 = sim.malloc_sim(64)
    p3 = sim.malloc_sim(15)

    assert p1 > 0 and p2 > 0 and p3 > 0
    # 16-byte alignment
    assert p1 % 16 == 0
    assert p2 % 16 == 0
    assert p3 % 16 == 0

    stats = sim.get_stats()
    assert stats["active_allocated_chunks"] == 3
    assert stats["user_payload_bytes"] == 32 + 64 + 15

    # Free p1 and p2, verify coalescing
    assert sim.free_sim(p1) is True
    assert sim.free_sim(p2) is True

    # After freeing both adjacent chunks, coalesced free space can serve a larger request
    p_large = sim.malloc_sim(80)
    assert p_large == p1  # Reused from the coalesced hole


def test_cpp20_ch09_heap_sbrk_executable():
    """Execute host C++20 heap and sbrk verification binary."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch09_heap_sbrk"

    if not bin_path.exists():
        # Build if not already built
        subprocess.run(["cmake", "--build", str(repo_root / "build"), "--target", "ch09_heap_sbrk"], check=True)

    proc = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert "All C++ tests passed successfully!" in proc.stdout


def test_figure_generation_outputs(tmp_path: Path):
    """Verify high-resolution figure generators produce valid PNG files."""
    f1 = tmp_path / "fig9_04.png"
    f2 = tmp_path / "fig9_05.png"
    f3 = tmp_path / "fig9_06.png"

    plot_fig9_04_traditional_unix_heap_vs_mikanos_heap(str(f1))
    plot_fig9_05_dynamic_allocation_pipeline_new_to_sbrk(str(f2))
    plot_fig9_06_sbrk_program_break_transition_and_fragmentation(str(f3))

    for p in (f1, f2, f3):
        assert p.exists()
        assert p.stat().st_size > 10000
        with Image.open(p) as img:
            assert img.format == "PNG"
            assert img.size[0] >= 1000
            assert img.size[1] >= 600


def test_notebook_structure_and_error_free_execution():
    """Verify Section 9.2 notebook exists and contains executed cells with zero errors."""
    nb_path = "9/9.2_Enabling_Dynamic_Allocation_sbrk.ipynb"
    assert os.path.exists(nb_path), f"Notebook file {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, "Notebook should have at least 10 rich content cells"

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} was not executed"
            outputs = cell.get("outputs", [])
            assert len(outputs) > 0, f"Code cell {idx} produced no output"
            for out in outputs:
                assert out.get("output_type") != "error", f"Error found in cell {idx}: {out}"
