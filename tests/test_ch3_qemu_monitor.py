"""Tests for Chapter 3.1: QEMU Monitor & x86-64 Debugging Infrastructure."""

import os
import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common import (
    CpuRegisterState,
    PhysicalMemoryInspector,
    QemuMonitorEngine,
    plot_qemu_monitor_architecture,
    plot_qemu_registers_and_memory_dump,
)


def test_cpu_register_state_bitfields_and_modes():
    """Verify x86-64 CPU control and flag register bitfield interpretations."""
    regs = CpuRegisterState(
        cr0=0x80010033,  # PG=1, PE=1
        cr3=0x00100000,
        rflags=0x00000202,  # IF=1, bit 1=1
    )
    assert regs.is_paging_enabled is True
    assert regs.is_protected_mode is True
    assert regs.is_interrupt_enabled is True

    # Paging off
    regs_no_pg = CpuRegisterState(cr0=0x00000001, rflags=0x00000002)
    assert regs_no_pg.is_paging_enabled is False
    assert regs_no_pg.is_protected_mode is True
    assert regs_no_pg.is_interrupt_enabled is False


def test_cpu_register_state_format_and_parse_roundtrip():
    """Verify formatting registers to QEMU info registers string and parsing back."""
    regs = CpuRegisterState(
        rax=0x123456789ABCDEF0,
        rbx=0x00000000067a3018,
        rip=0x0000000000100000,
        rsp=0x0000000007fe3dc0,
        cr0=0x80010033,
        cr3=0x0000000007fa0000,
        rflags=0x00000206,
    )
    dump_text = regs.format_qemu_info_registers()
    assert "RAX=123456789abcdef0" in dump_text
    assert "RIP=0000000000100000" in dump_text
    assert "CR3=0000000007fa0000" in dump_text
    assert "[----I---]" in dump_text

    # Parse back
    parsed = CpuRegisterState.parse_qemu_info_registers(dump_text)
    assert parsed.rax == 0x123456789ABCDEF0
    assert parsed.rbx == 0x00000000067A3018
    assert parsed.rip == 0x0000000000100000
    assert parsed.rsp == 0x0000000007FE3DC0
    assert parsed.cr0 == 0x80010033
    assert parsed.cr3 == 0x0000000007FA0000
    assert parsed.rflags == 0x00000206


def test_physical_memory_inspector_read_write_and_bounds():
    """Verify raw byte read/write and memory boundary bounds checks."""
    mem = PhysicalMemoryInspector(size_bytes=1024 * 1024)
    data = bytes([0x48, 0x89, 0xE5, 0xC3])
    mem.write_bytes(0x1000, data)

    readback = mem.read_bytes(0x1000, 4)
    assert readback == data

    # Out of bounds
    with pytest.raises(ValueError):
        mem.read_bytes(1024 * 1024 - 2, 4)

    with pytest.raises(ValueError):
        mem.write_bytes(1024 * 1024, b"\x00")


def test_physical_memory_inspector_execute_xp():
    """Verify QEMU `xp /nuf addr` format specifications (units b, h, w, g and formats x, d, c)."""
    mem = PhysicalMemoryInspector(size_bytes=4 * 1024 * 1024)
    # Write: 48 83 ec 28 e8 12 00 00
    # In Little-Endian:
    #   16-bit: [0x8348, 0x28ec, 0x12e8, 0x0000]
    #   32-bit: [0x28ec8348, 0x000012e8]
    #   64-bit: [0x000012e828ec8348]
    code = bytes([0x48, 0x83, 0xEC, 0x28, 0xE8, 0x12, 0x00, 0x00])
    mem.write_bytes(0x100000, code)

    # Byte format hex
    out_b = mem.execute_xp("/4bx", 0x100000)
    assert len(out_b) == 1
    assert "0x0000000000100000: 0x48 0x83 0xec 0x28" in out_b[0]

    # Halfword (16-bit)
    out_h = mem.execute_xp("/2hx", 0x100000)
    assert len(out_h) == 1
    assert "0x0000000000100000: 0x8348 0x28ec" in out_h[0]

    # Word (32-bit)
    out_w = mem.execute_xp("/2wx", 0x100000)
    assert len(out_w) == 1
    assert "0x0000000000100000: 0x28ec8348 0x000012e8" in out_w[0]

    # Giant word (64-bit)
    out_g = mem.execute_xp("/1gx", 0x100000)
    assert len(out_g) == 1
    assert "0x0000000000100000: 0x000012e828ec8348" in out_g[0]

    # Character dump
    mem.write_bytes(0x2000, b"HelloOS!")
    out_c = mem.execute_xp("/8bc", 0x2000)
    assert len(out_c) == 1
    assert "'H' 'e' 'l' 'l' 'o' 'O' 'S' '!'" in out_c[0]


def test_qemu_monitor_engine_commands():
    """Verify QEMU Monitor command dispatcher operations."""
    engine = QemuMonitorEngine(ram_size=2 * 1024 * 1024)
    engine.regs.rip = 0x100000
    engine.regs.rax = 0x42

    # info registers
    res_regs = engine.execute_command("info registers")
    assert "RAX=0000000000000042" in res_regs
    assert "RIP=0000000000100000" in res_regs

    # xp
    engine.ram.write_bytes(0x100000, bytes([0x90, 0x90, 0xC3, 0xCC]))
    res_xp = engine.execute_command("xp /4bx 0x100000")
    assert "0x90 0x90 0xc3 0xcc" in res_xp

    # help
    res_help = engine.execute_command("help")
    assert "QEMU Monitor commands" in res_help

    # print
    res_p = engine.execute_command("print rax + 10")
    assert "$0x4c" in res_p  # 0x42 + 10 = 66 + 10 = 76 = 0x4c

    # stop and cont
    res_stop = engine.execute_command("stop")
    assert "CPU stopped" in res_stop
    assert engine.is_running is False

    res_cont = engine.execute_command("cont")
    assert "Execution continued" in res_cont
    assert engine.is_running is True


def test_cpp20_qemu_monitor_sim_executable():
    """Verify C++20 standalone binary execution and test suite."""
    binary_path = Path("build/bin/ch03_qemu_monitor_sim")
    if not binary_path.exists():
        subprocess.run(["cmake", "-B", "build", "-S", "."], check=True)
        subprocess.run(["cmake", "--build", "build", "--target", "ch03_qemu_monitor_sim"], check=True)

    result = subprocess.run([str(binary_path), "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "[PASS] All QemuMonitorVerification tests passed successfully." in result.stdout


def test_figure_generation_outputs(tmp_path):
    """Verify high-DPI figure generation functions."""
    fig1_path = tmp_path / "fig3_01_arch.png"
    fig2_path = tmp_path / "fig3_02_regs.png"

    plot_qemu_monitor_architecture(fig1_path)
    assert fig1_path.exists()
    assert fig1_path.stat().st_size > 50_000

    plot_qemu_registers_and_memory_dump(fig2_path)
    assert fig2_path.exists()
    assert fig2_path.stat().st_size > 50_000

    # Verify high resolution (DPI 300)
    with Image.open(fig1_path) as img:
        assert img.width >= 2000
    with Image.open(fig2_path) as img:
        assert img.width >= 2000
