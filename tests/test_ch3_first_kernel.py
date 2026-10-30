"""Tests for Chapter 3.3: First Kernel, ExitBootServices, and Red Zone Architecture."""

import os
import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common import (
    Elf64Header,
    KernelLoaderStep,
    KernelLoaderSimulator,
    RedZoneSimulator,
    plot_kernel_loading_and_entry_flow,
    plot_red_zone_mechanism_and_trap,
)


def test_elf64_header_structure_and_offset24():
    """Verify that ELF64 header aligns e_entry precisely at offset 24 (0x18)."""
    header = Elf64Header(e_entry=0x00100000)
    raw_bytes = header.to_bytes()

    assert len(raw_bytes) == 64
    assert raw_bytes[0:4] == b"\x7fELF"
    assert raw_bytes[4] == 2  # 64-bit architecture
    assert raw_bytes[5] == 1  # Little endian
    assert raw_bytes[6] == 1  # ELF version 1
    assert raw_bytes[7] == 0  # System V ABI
    assert raw_bytes[18:20] == b"\x3e\x00"  # EM_X86_64 (62)

    # Offset 24 check
    assert Elf64Header.get_entry_point_offset() == 24
    entry_val = int.from_bytes(raw_bytes[24:32], byteorder="little")
    assert entry_val == 0x00100000

    # Parse round-trip
    parsed = Elf64Header.parse(raw_bytes)
    assert parsed.e_entry == 0x00100000
    assert parsed.e_machine == 0x3E
    assert parsed.e_type == 2


def test_bootloader_kernel_load_and_allocation():
    """Verify UEFI bootloader page allocation at 1MiB (0x100000) and kernel file read."""
    loader = KernelLoaderSimulator(target_base_addr=0x100000)

    # 1. Open kernel file
    assert loader.open_kernel("\\kernel.elf", fake_binary_size=8192)
    assert loader.current_step == KernelLoaderStep.OPEN_KERNEL_FILE

    # 2. Query file size
    assert loader.get_kernel_info() == 8192
    assert loader.current_step == KernelLoaderStep.GET_KERNEL_INFO

    # 3. Allocate pages at 0x100000 (1MiB)
    base_addr, num_pages = loader.allocate_pages()
    assert base_addr == 0x100000
    # (8192 + 0xfff) // 0x1000 = 2 pages (8192 bytes)
    assert num_pages == 2
    assert loader.current_step == KernelLoaderStep.ALLOCATE_PAGES

    # 4. Read kernel binary
    buf = loader.read_kernel_data()
    assert len(buf) == 8192
    assert loader.current_step == KernelLoaderStep.READ_KERNEL_DATA


def test_exit_boot_services_mapkey_mismatch_and_retry():
    """Verify ExitBootServices MapKey invalidation and mandatory retry logic."""
    loader = KernelLoaderSimulator()
    loader.open_kernel()
    loader.allocate_pages()  # Modifies MapKey due to allocation

    stale_map_key = 42

    # Without retry, stale MapKey MUST fail
    fail_res = loader.exit_boot_services(passed_map_key=stale_map_key, retry_if_mismatch=False)
    assert fail_res is False
    assert loader.boot_services_active is True

    # With retry (MikanOS pattern), re-fetches memory map and succeeds
    success_res = loader.exit_boot_services(passed_map_key=stale_map_key, retry_if_mismatch=True)
    assert success_res is True
    assert loader.boot_services_active is False
    assert loader.current_step == KernelLoaderStep.EXIT_BOOT_SERVICES


def test_kernel_jump_and_execution():
    """Verify extracting e_entry from offset 24 and transferring control to KernelMain."""
    loader = KernelLoaderSimulator(target_base_addr=0x100000)
    loader.open_kernel(fake_binary_size=4096)
    loader.allocate_pages()

    hdr = Elf64Header(e_entry=0x00100020)
    raw = hdr.to_bytes() + b"\x90" * (4096 - 64)
    loader.read_kernel_data(raw)

    # Calling kernel before ExitBootServices must raise error
    with pytest.raises(RuntimeError, match="Cannot execute kernel while UEFI Boot Services"):
        loader.call_kernel_entry()

    # Now exit boot services
    loader.exit_boot_services(loader.map_key)

    result = loader.call_kernel_entry()
    assert result["entry_point"] == "0x100020"
    assert result["executed"] is True
    assert loader.kernel_executed is True
    assert loader.current_step == KernelLoaderStep.KERNEL_HALT_LOOP


def test_red_zone_stack_corruption_on_interrupt():
    """Verify that System V ABI Red Zone leads to silent data corruption when interrupts arrive in kernel."""
    sim = RedZoneSimulator()
    sim.configure_compiler_mode(use_red_zone=True)

    # Leaf function stores variables in Red Zone [RSP-8 .. RSP-32] without modifying RSP
    sim.allocate_leaf_function_vars({
        "temp_pointer": 0xDEADBEEFCAFEBABE,
        "loop_counter": 0x0000000000000064,
        "buffer_index": 0x0000000000000008,
    })

    # RSP is untouched by compiler!
    assert sim.rsp == sim.initial_rsp

    # Asynchronous hardware interrupt arrives! CPU pushes 40 bytes onto current RSP
    sim.trigger_hardware_interrupt(vector=32)

    # Critical: All 3 variables were destroyed by the hardware interrupt push!
    assert sim.count_corrupted_variables() == 3


def test_no_red_zone_protects_kernel_stack():
    """Verify that compiling with -mno-red-zone protects local variables from interrupt clobbering."""
    sim = RedZoneSimulator()
    sim.configure_compiler_mode(use_red_zone=False)  # -mno-red-zone

    # Compiler explicitly executes `sub rsp, N` before saving variables
    sim.allocate_leaf_function_vars({
        "temp_pointer": 0xDEADBEEFCAFEBABE,
        "loop_counter": 0x0000000000000064,
        "buffer_index": 0x0000000000000008,
    })

    # RSP was decremented by frame size (16-byte aligned)
    assert sim.rsp < sim.initial_rsp

    # Interrupt arrives: CPU pushes below the new, safe RSP
    sim.trigger_hardware_interrupt(vector=32)

    # ZERO corruption: Local variables above RSP remain completely safe!
    assert sim.count_corrupted_variables() == 0


def test_cpp20_first_kernel_sim_binary():
    """Verify C++20 first kernel simulator executes and passes all internal tests."""
    binary_path = Path("build/bin/ch03_first_kernel_sim")
    assert binary_path.exists(), f"Binary {binary_path} does not exist. Run cmake build first."

    res = subprocess.run([str(binary_path), "--test"], capture_output=True, text=True)
    assert res.returncode == 0, f"ch03_first_kernel_sim failed: {res.stderr}"
    assert "ALL FIRST KERNEL & RED ZONE SIMULATION TESTS PASSED!" in res.stdout


def test_generated_figures_ch03_section3():
    """Verify that high-resolution explanation figures exist and meet publication quality."""
    fig5 = Path("3/result/fig3_05_kernel_loading_and_entry_flow.png")
    fig6 = Path("3/result/fig3_06_red_zone_mechanism_and_trap.png")

    assert fig5.exists(), f"Figure {fig5} was not found"
    assert fig6.exists(), f"Figure {fig6} was not found"

    with Image.open(fig5) as img:
        assert img.size[0] >= 3000, f"Expected high resolution width >= 3000, got {img.size[0]}"
        assert img.size[1] >= 1800, f"Expected high resolution height >= 1800, got {img.size[1]}"

    with Image.open(fig6) as img:
        assert img.size[0] >= 3000, f"Expected high resolution width >= 3000, got {img.size[0]}"
        assert img.size[1] >= 1800, f"Expected high resolution height >= 1800, got {img.size[1]}"
