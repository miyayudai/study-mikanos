"""Tests for Chapter 7.3: x86-64 Interrupt Vectors and Exceptions Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    ExceptionType,
    CPUExceptionSpec,
    InterruptVectorSpace,
    plot_interrupt_vector_space_map,
    plot_cpu_exception_types_and_handling,
)


def test_vector_space_ranges_and_partitioning():
    """Verify vector space boundary classification and MikanOS vector 0x40."""
    # CPU Exceptions: 0 to 31
    assert InterruptVectorSpace.is_cpu_exception(0)
    assert InterruptVectorSpace.is_cpu_exception(14)
    assert InterruptVectorSpace.is_cpu_exception(31)
    assert not InterruptVectorSpace.is_cpu_exception(32)
    assert not InterruptVectorSpace.is_cpu_exception(64)
    assert not InterruptVectorSpace.is_cpu_exception(255)

    # User Defined: 32 to 255
    assert not InterruptVectorSpace.is_user_defined(0)
    assert not InterruptVectorSpace.is_user_defined(31)
    assert InterruptVectorSpace.is_user_defined(32)
    assert InterruptVectorSpace.is_user_defined(64)
    assert InterruptVectorSpace.is_user_defined(255)

    # MikanOS xHCI MSI Interrupt Vector
    assert InterruptVectorSpace.MIKANOS_VECTOR_XHCI == 0x40
    assert InterruptVectorSpace.MIKANOS_VECTOR_XHCI == 64
    assert InterruptVectorSpace.is_user_defined(InterruptVectorSpace.MIKANOS_VECTOR_XHCI)

    # Out of range checks
    with pytest.raises(ValueError):
        InterruptVectorSpace.get_vector_description(-1)
    with pytest.raises(ValueError):
        InterruptVectorSpace.get_vector_description(256)


def test_cpu_exceptions_specifications():
    """Verify architectural specifications of key CPU exceptions."""
    # #DE (Divide Error, vector 0): Fault, No Error Code
    de = InterruptVectorSpace.get_exception_spec(0)
    assert de is not None
    assert de.mnemonic == "#DE"
    assert de.exception_type == ExceptionType.FAULT
    assert not de.has_error_code

    # #BP (Breakpoint, vector 3): Trap, No Error Code
    bp = InterruptVectorSpace.get_exception_spec(3)
    assert bp is not None
    assert bp.mnemonic == "#BP"
    assert bp.exception_type == ExceptionType.TRAP
    assert not bp.has_error_code

    # #DF (Double Fault, vector 8): Abort, With Error Code
    df = InterruptVectorSpace.get_exception_spec(8)
    assert df is not None
    assert df.mnemonic == "#DF"
    assert df.exception_type == ExceptionType.ABORT
    assert df.has_error_code

    # #GP (General Protection, vector 13): Fault, With Error Code
    gp = InterruptVectorSpace.get_exception_spec(13)
    assert gp is not None
    assert gp.mnemonic == "#GP"
    assert gp.exception_type == ExceptionType.FAULT
    assert gp.has_error_code

    # #PF (Page Fault, vector 14): Fault, With Error Code
    pf = InterruptVectorSpace.get_exception_spec(14)
    assert pf is not None
    assert pf.mnemonic == "#PF"
    assert pf.exception_type == ExceptionType.FAULT
    assert pf.has_error_code

    # Verify all exceptions that push an error code
    error_code_vectors = [vec for vec, spec in InterruptVectorSpace.EXCEPTIONS.items() if spec.has_error_code]
    expected_ec_vectors = [8, 10, 11, 12, 13, 14, 17, 21]
    for vec in expected_ec_vectors:
        assert vec in error_code_vectors


def test_dispatch_simulation_semantics():
    """Verify hardware dispatch behavior for Fault, Trap, Abort, and Interrupt."""
    cur_rip = 0x100020
    next_rip = 0x100025

    # 1. Fault (#PF, vector 14): re-executes instruction
    sim_pf = InterruptVectorSpace.simulate_dispatch(14, cur_rip, next_rip, error_code_val=0x02)
    assert sim_pf["type"] == ExceptionType.FAULT.value
    assert sim_pf["has_error_code"] is True
    assert sim_pf["saved_rip"] == hex(cur_rip)
    assert sim_pf["return_target"] == hex(cur_rip)
    assert sim_pf["stack_bytes_pushed"] == 48

    # 2. Trap (#BP, vector 3): proceeds to next instruction
    sim_bp = InterruptVectorSpace.simulate_dispatch(3, cur_rip, next_rip)
    assert sim_bp["type"] == ExceptionType.TRAP.value
    assert sim_bp["has_error_code"] is False
    assert sim_bp["saved_rip"] == hex(next_rip)
    assert sim_bp["return_target"] == hex(next_rip)
    assert sim_bp["stack_bytes_pushed"] == 40

    # 3. Abort (#DF, vector 8): unrecoverable
    sim_df = InterruptVectorSpace.simulate_dispatch(8, cur_rip, next_rip)
    assert sim_df["type"] == ExceptionType.ABORT.value
    assert sim_df["has_error_code"] is True
    assert sim_df["return_target"] == "CRASH / HALT"
    assert sim_df["stack_bytes_pushed"] == 48

    # 4. Hardware Interrupt (xHCI 0x40): proceeds to next instruction
    sim_xhci = InterruptVectorSpace.simulate_dispatch(InterruptVectorSpace.MIKANOS_VECTOR_XHCI, cur_rip, next_rip)
    assert sim_xhci["type"] == ExceptionType.INTERRUPT.value
    assert sim_xhci["has_error_code"] is False
    assert sim_xhci["saved_rip"] == hex(next_rip)
    assert sim_xhci["return_target"] == hex(next_rip)
    assert sim_xhci["stack_bytes_pushed"] == 40


def test_figure_generation_and_validity(tmp_path):
    """Verify Figure 7.05 and 7.06 generation, resolution, and valid image bytes."""
    fig1_path = str(tmp_path / "fig7_05_test.png")
    fig2_path = str(tmp_path / "fig7_06_test.png")

    p1 = plot_interrupt_vector_space_map(fig1_path)
    p2 = plot_cpu_exception_types_and_handling(fig2_path)

    assert Path(p1).exists()
    assert Path(p2).exists()

    with Image.open(p1) as img1:
        assert img1.format == "PNG"
        assert img1.width >= 3000
        assert img1.height >= 2000

    with Image.open(p2) as img2:
        assert img2.format == "PNG"
        assert img2.width >= 3000
        assert img2.height >= 2000


def test_cpp20_interrupt_vector_sim_executable():
    """Verify C++20 standalone simulator compilation and successful execution."""
    repo_root = Path(__file__).resolve().parent.parent
    src_file = repo_root / "src/ch07/interrupt_vector_sim.cpp"
    bin_file = repo_root / "src/ch07/interrupt_vector_sim_test_bin"

    assert src_file.exists()

    compile_cmd = [
        "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic",
        str(src_file), "-o", str(bin_file)
    ]
    res_compile = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert res_compile.returncode == 0, f"Compilation failed: {res_compile.stderr}"

    res_run = subprocess.run([str(bin_file)], capture_output=True, text=True)
    assert res_run.returncode == 0
    assert "All C++20 Interrupt Vector tests PASSED successfully!" in res_run.stdout
    assert "kXHCI" in res_run.stdout

    if bin_file.exists():
        bin_file.unlink()


def test_notebook_execution_and_integrity():
    """Verify Section 7.3 Jupyter Notebook structure, cell outputs, and Colab compatibility."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "7/7.3_Interrupt_Vectors_in_x86_64.ipynb"

    assert nb_path.exists(), "Notebook 7/7.3_Interrupt_Vectors_in_x86_64.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 10, "Notebook must contain at least 10 rich cells"

    # Cell 0 must be Colab auto-setup
    cell0 = cells[0]
    assert cell0["cell_type"] == "code"
    cell0_source = "".join(cell0["source"])
    assert "google.colab" in cell0_source
    assert "study-mikanos" in cell0_source

    # Check for error outputs across all code cells
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", f"Error output detected in cell {i}: {out.get('evalue')}"

    # Verify key topics in Markdown cells
    full_text = "\n".join("".join(c["source"]) for c in cells)
    assert "7.3 割り込みベクタ" in full_text
    assert "InterruptVector" in full_text
    assert "0x40" in full_text
    assert "Fault" in full_text
    assert "Trap" in full_text
    assert "Abort" in full_text
    assert "ErrorCode" in full_text
