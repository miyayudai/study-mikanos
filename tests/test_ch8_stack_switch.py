"""Unit tests for MikanOS Chapter 8.4: Switching Kernel Stack."""

import json
import subprocess
from pathlib import Path
from PIL import Image

from common import (
    KernelStackSimulator,
    plot_fig8_10_kernel_stack_relocation_architecture,
    plot_fig8_11_system_v_abi_stack_alignment_rules,
    plot_fig8_12_stack_growth_and_pointer_arithmetics,
)


def test_kernel_stack_simulator_alignment():
    """Verify System V AMD64 ABI 16-byte stack alignment transitions in KernelStackSimulator."""
    sim = KernelStackSimulator()
    result = sim.simulate_stack_switch()

    # Pre-call RSP must be 16-byte aligned
    assert result["pre_call_aligned"] is True
    assert (result["pre_call_rsp"] % 16) == 0

    # Inside callee function entry (after 8B return address pushed), RSP % 16 == 8
    assert result["func_entry_valid"] is True
    assert result["func_entry_offset"] == 8

    # After 'push rbp' prologue, RSP % 16 == 0 again
    assert result["post_prologue_aligned"] is True
    assert (result["post_prologue_rsp"] % 16) == 0

    # Local allocations maintaining 16-byte alignment
    assert result["local_alloc_aligned"] is True
    assert (result["local_alloc_rsp"] % 16) == 0


def test_stack_switch_argument_preservation():
    """Verify that RDI and RSI registers pass through stack switch undisturbed and .fin catches return."""
    sim = KernelStackSimulator()
    custom_fb = 0xFFFF_9000_1234_0000
    custom_map = 0x0000_0000_0006_F000
    result = sim.simulate_stack_switch(rdi_val=custom_fb, rsi_val=custom_map)

    assert result["rdi_preserved"] is True
    assert result["rsi_preserved"] is True
    assert result["halted_at_fin"] is True
    assert result["epilogue_rsp"] == result["pre_call_rsp"]


def test_sse_alignment_fault_simulation():
    """Verify SSE aligned store instruction (movaps) behavior under aligned and misaligned RSP."""
    sim = KernelStackSimulator()

    # Aligned RSP executes without faults
    res_aligned = sim.simulate_sse_alignment_fault(misaligned=False)
    assert res_aligned["is_aligned_16"] is True
    assert res_aligned["causes_general_protection_fault"] is False
    assert res_aligned["exception"] == "None (Execution Success)"

    # Misaligned RSP causes #GP
    res_misaligned = sim.simulate_sse_alignment_fault(misaligned=True)
    assert res_misaligned["is_aligned_16"] is False
    assert res_misaligned["causes_general_protection_fault"] is True
    assert "#GP" in res_misaligned["exception"]


def test_stack_capacity_comparison():
    """Verify stack recursion depth and 1MiB capacity compared to UEFI 64KiB stack."""
    sim = KernelStackSimulator()
    cap = sim.simulate_stack_capacity(frame_size=64)

    assert cap["uefi_stack_size_bytes"] == 64 * 1024
    assert cap["kernel_stack_size_bytes"] == 1024 * 1024
    assert cap["uefi_max_frames"] == 1024
    assert cap["kernel_max_frames"] == 16384
    assert cap["depth_multiplier"] == 16.0


def test_figure_generation_outputs():
    """Verify Chapter 8.4 figures exist and are high-resolution valid PNG images."""
    f1 = Path(plot_fig8_10_kernel_stack_relocation_architecture())
    f2 = Path(plot_fig8_11_system_v_abi_stack_alignment_rules())
    f3 = Path(plot_fig8_12_stack_growth_and_pointer_arithmetics())

    for f in (f1, f2, f3):
        assert f.exists(), f"Figure not found: {f}"
        assert f.stat().st_size > 50_000, f"Figure too small: {f} ({f.stat().st_size} bytes)"
        with Image.open(f) as img:
            assert img.format == "PNG"
            assert img.width >= 1200


def test_cpp20_kernel_stack_sim_standalone():
    """Verify standalone compilation and execution of C++20 kernel stack simulator."""
    binary_path = Path("build/bin/ch08_kernel_stack_sim")
    if not binary_path.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch08_kernel_stack_sim"], check=True)

    result = subprocess.run(
        [str(binary_path), "--test"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[PASS] All Chapter 8.4 kernel stack switching tests passed successfully!" in result.stdout
    assert "kernel_main_stack 16-byte alignment verified" in result.stdout
    assert "1,048,576 bytes = 1 MiB" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Chapter 8.4 notebook exists and has zero cell execution errors."""
    nb_path = Path("8/8.4_Switching_Kernel_Stack.ipynb")
    assert nb_path.exists(), "Notebook 8.4 does not exist yet"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, f"Notebook has too few cells: {len(cells)}"

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"Cell {idx} has an execution error: {out.get('ename')}: {out.get('evalue')}"
                )
