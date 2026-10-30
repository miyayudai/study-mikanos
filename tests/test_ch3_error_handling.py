"""Tests for Chapter 3.6: Error Handling and Pointer Casts (osbook_day03d)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    BootStepAction,
    BootloaderErrorSimulator,
    CppCastKind,
    EFIStatus,
    EFI_ERROR_MASK,
    PointerCastSimulator,
    plot_pointer_casts_type_safety_memory,
    plot_uefi_status_and_error_handling_flow,
)


def test_efi_status_error_bit_and_severity():
    """Verify EFI_STATUS 64-bit layout, EFI_ERROR mask, and severity classification."""
    # 1. Success status
    success = EFIStatus.success()
    assert success.is_success is True
    assert success.is_error is False
    assert success.is_warning is False
    assert success.code_number == 0
    assert success.name == "EFI_SUCCESS"
    assert success.raw_value == 0

    # 2. Critical error statuses
    err_not_found = EFIStatus.error(14)
    assert err_not_found.is_error is True
    assert err_not_found.is_success is False
    assert err_not_found.is_warning is False
    assert err_not_found.code_number == 14
    assert err_not_found.name == "EFI_NOT_FOUND"
    assert err_not_found.raw_value == (14 | EFI_ERROR_MASK)
    assert err_not_found.raw_value == 0x8000_0000_0000_000E

    err_buf_small = EFIStatus.error(5)
    assert err_buf_small.name == "EFI_BUFFER_TOO_SMALL"
    assert err_buf_small.code_number == 5

    # 3. Two's complement signed evaluation (EFI_ERROR macro logic)
    # In C: #define EFI_ERROR(Status) (((INTN)(Status)) < 0)
    raw_signed = int.from_bytes(err_not_found.raw_value.to_bytes(8, "little", signed=False), "little", signed=True)
    assert raw_signed < 0

    raw_success_signed = int.from_bytes(success.raw_value.to_bytes(8, "little", signed=False), "little", signed=True)
    assert raw_success_signed >= 0

    # 4. Warning status (Bit 63 is 0, but non-zero code)
    warn_glyph = EFIStatus(1)
    assert warn_glyph.is_warning is True
    assert warn_glyph.is_error is False
    assert warn_glyph.is_success is False
    assert warn_glyph.name == "EFI_WARN_UNKNOWN_GLYPH"


def test_bootloader_error_simulator_nominal_boot():
    """Verify full successful boot sequence without any error injection."""
    sim = BootloaderErrorSimulator()
    results = sim.run_simulation()

    assert sim.boot_completed is True
    assert sim.is_halted is False
    assert len(results) == len(BootloaderErrorSimulator.STEPS)

    for r in results:
        assert r.passed is True
        assert r.status.is_success is True
        assert "OK" in r.output_message


def test_bootloader_error_simulator_graceful_memmap_failure():
    """Verify that \\memmap open failure is non-fatal (Graceful Degradation).

    In MikanLoaderPkg (day03d), if \\memmap cannot be opened, it outputs
    'Ignored.' and proceeds with GOP initialization and kernel loading.
    """
    sim = BootloaderErrorSimulator()
    injected = {
        "OpenMemmapFile": EFIStatus.error(14),  # EFI_NOT_FOUND
    }
    results = sim.run_simulation(injected_errors=injected)

    assert sim.boot_completed is True
    assert sim.is_halted is False

    # Check that OpenMemmapFile was marked as not passed, but ignored
    memmap_res = next(r for r in results if r.step_name == "OpenMemmapFile")
    assert memmap_res.passed is False
    assert "Ignored." in memmap_res.output_message
    assert "Graceful degradation" in memmap_res.action_taken

    # SaveMemoryMap and CloseMemmapFile should be safely skipped
    save_res = next(r for r in results if r.step_name == "SaveMemoryMap")
    close_res = next(r for r in results if r.step_name == "CloseMemmapFile")
    assert save_res.output_message == "[SKIPPED]"
    assert close_res.output_message == "[SKIPPED]"

    # Subsequent steps like OpenGOP, OpenKernelFile, ReadKernel must succeed
    kernel_res = next(r for r in results if r.step_name == "OpenKernelFile")
    assert kernel_res.passed is True


def test_bootloader_error_simulator_fatal_failures():
    """Verify fatal errors cause CPU Halt and stop boot sequence."""
    sim = BootloaderErrorSimulator()

    # Case A: Missing kernel ELF
    results_a = sim.run_simulation({"OpenKernelFile": EFIStatus.error(14)})
    assert sim.boot_completed is False
    assert sim.is_halted is True
    last_step_a = results_a[-1]
    assert last_step_a.step_name == "OpenKernelFile"
    assert "Halt." in last_step_a.output_message
    assert "Halt() executed" in last_step_a.action_taken

    # Case B: Out of memory on AllocatePages
    results_b = sim.run_simulation({"AllocatePages": EFIStatus.error(9)})  # EFI_OUT_OF_RESOURCES
    assert sim.boot_completed is False
    assert sim.is_halted is True
    last_step_b = results_b[-1]
    assert last_step_b.step_name == "AllocatePages"
    assert "EFI_OUT_OF_RESOURCES" in last_step_b.output_message

    # Case C: ExitBootServices failure
    results_c = sim.run_simulation({"ExitBootServices": EFIStatus.error(2)})  # EFI_INVALID_PARAMETER
    assert sim.boot_completed is False
    assert sim.is_halted is True
    last_step_c = results_c[-1]
    assert last_step_c.step_name == "ExitBootServices"
    assert "Failed -> Halt()" in last_step_c.action_taken


def test_pointer_cast_simulator_matrix():
    """Verify C++ cast rules against C-style cast for low-level OS operations."""
    # 1. uint64_t to function pointer (void (*)(uint64_t, uint64_t))
    src = "uint64_t"
    target_fn = "void (*)(uint64_t, uint64_t)"

    res_static = PointerCastSimulator.analyze(src, target_fn, CppCastKind.STATIC_CAST)
    assert res_static.is_legal is False
    assert "invalid 'static_cast'" in res_static.compile_error_reason

    res_reinterp = PointerCastSimulator.analyze(src, target_fn, CppCastKind.REINTERPRET_CAST)
    assert res_reinterp.is_legal is True
    assert res_reinterp.is_safe is True
    assert "エントリポイント呼び出し" in res_reinterp.low_level_usage

    res_c_style = PointerCastSimulator.analyze(src, target_fn, CppCastKind.C_STYLE_CAST)
    assert res_c_style.is_legal is True
    assert res_c_style.is_safe is False

    # 2. void* to concrete structure pointer (EFI_FILE_INFO*)
    res_void_static = PointerCastSimulator.analyze("void*", "EFI_FILE_INFO*", CppCastKind.STATIC_CAST)
    assert res_void_static.is_legal is True

    # 3. const char* to char* (stripping const)
    res_const = PointerCastSimulator.analyze("const char*", "char*", CppCastKind.CONST_CAST)
    assert res_const.is_legal is True

    res_const_static = PointerCastSimulator.analyze("const char*", "char*", CppCastKind.STATIC_CAST)
    assert res_const_static.is_legal is False
    assert "const qualifier" in res_const_static.compile_error_reason


def test_cpp20_error_handling_sim_binary():
    """Verify that the C++20 standalone verification binary runs and passes all assertions."""
    bin_path = Path("build/bin/ch03_error_handling_sim")
    if not bin_path.exists():
        pytest.skip("ch03_error_handling_sim binary not built yet.")

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert res.returncode == 0
    assert "ALL C++20 Chapter 3.6 VERIFICATIONS PASSED!" in res.stdout


def test_ch3_6_figures(tmp_path: Path):
    """Verify figure generation functions for Figure 3-11 and Figure 3-12."""
    fig11_path = tmp_path / "fig3_11.png"
    fig12_path = tmp_path / "fig3_12.png"

    plot_uefi_status_and_error_handling_flow(fig11_path)
    assert fig11_path.exists()
    assert fig11_path.stat().st_size > 50_000

    plot_pointer_casts_type_safety_memory(fig12_path)
    assert fig12_path.exists()
    assert fig12_path.stat().st_size > 50_000

    # Image validation
    with Image.open(fig11_path) as img:
        assert img.width > 1000
        assert img.height > 800

    with Image.open(fig12_path) as img:
        assert img.width > 1000
        assert img.height > 800


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 3.6 notebook exists, has proper structure, and has zero cell errors."""
    nb_path = Path("3/3.6_Error_Handling_and_Pointer_Casts.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist"

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert len(nb["cells"]) >= 10
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5

    for idx, cell in enumerate(code_cells):
        assert cell.get("execution_count") is not None, f"Code cell {idx} is not executed"
        for out in cell.get("outputs", []):
            assert (
                out.get("output_type") != "error"
            ), f"Error in code cell {idx}: {out.get('evalue')}"

