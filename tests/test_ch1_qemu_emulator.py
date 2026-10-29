"""Tests for Chapter 1.4: QEMU & OVMF UEFI Emulation & Virtual FAT Architecture."""

import os
import subprocess
from pathlib import Path

import pytest

from common import (
    QemuFirmwareMode,
    QemuDriveFormat,
    QemuVirtualHardwareConfig,
    QemuEmulatorConfig,
    VirtualFatSimulator,
    UefiBootPhase,
    UefiBootStep,
    QemuUefiBootSequenceSimulator,
    QemuMonitorSimulator,
    plot_qemu_ovmf_boot_architecture,
    plot_virtual_fat_and_boot_flow,
)


def test_qemu_emulator_config_default_build():
    """Verify default QEMU command arguments for standard MikanOS boot."""
    config = QemuEmulatorConfig(
        ovmf_code_path="/usr/share/OVMF/OVMF_CODE.fd",
        ovmf_vars_path="/usr/share/OVMF/OVMF_VARS.fd",
        disk_target="disk_img",
    )
    args = config.build_command()
    cmd_str = config.build_command_str()

    assert args[0] == "qemu-system-x86_64"
    assert "-m" in args and "1024M" in args
    assert "-cpu" in args and "qemu64" in args
    assert "-monitor" in args and "stdio" in args
    assert "-net" in args and "none" in args
    assert "-device" in args and "nec-usb-xhci,id=xhci" in args

    # Check pflash split
    assert "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd" in cmd_str
    assert "if=pflash,format=raw,file=/usr/share/OVMF/OVMF_VARS.fd" in cmd_str

    # Check virtual FAT drive
    assert "format=raw,file=fat:rw:disk_img" in cmd_str

    # No validation errors
    assert len(config.validate()) == 0


def test_qemu_firmware_modes():
    """Verify different OVMF firmware loading modes (-bios vs pflash)."""
    # 1. BIOS direct mode
    config_bios = QemuEmulatorConfig(
        firmware_mode=QemuFirmwareMode.BIOS_DIRECT,
        ovmf_code_path="/usr/share/OVMF/OVMF.fd",
        disk_target="disk_img",
    )
    args_bios = config_bios.build_command()
    idx_bios = args_bios.index("-bios")
    assert args_bios[idx_bios + 1] == "/usr/share/OVMF/OVMF.fd"
    assert not any("if=pflash" in arg for arg in args_bios)

    # 2. PFLASH unified mode
    config_pflash_uni = QemuEmulatorConfig(
        firmware_mode=QemuFirmwareMode.PFLASH_UNIFIED,
        ovmf_code_path="/usr/share/OVMF/OVMF.fd",
        disk_target="disk_img",
    )
    args_pflash_uni = config_pflash_uni.build_command()
    assert "if=pflash,format=raw,file=/usr/share/OVMF/OVMF.fd" in config_pflash_uni.build_command_str()


def test_qemu_drive_formats():
    """Verify virtual FAT directory vs raw image vs qcow2 disk formats."""
    # Raw disk
    config_raw = QemuEmulatorConfig(
        drive_format=QemuDriveFormat.RAW_DISK,
        disk_target="mikanos.img",
    )
    assert "format=raw,file=mikanos.img" in config_raw.build_command_str()

    # Qcow2 disk
    config_qcow2 = QemuEmulatorConfig(
        drive_format=QemuDriveFormat.QCOW2_DISK,
        disk_target="mikanos.qcow2",
    )
    assert "format=qcow2,file=mikanos.qcow2" in config_qcow2.build_command_str()


def test_qemu_advanced_hardware_options():
    """Verify hardware options like GDB, USB mouse, KVM, and SMP cores."""
    hw = QemuVirtualHardwareConfig(
        memory_mb=2048,
        smp_cores=4,
        enable_kvm=True,
        enable_xhci=True,
        enable_usb_mouse=True,
        gdb_debug=True,
        gdb_port=1234,
    )
    config = QemuEmulatorConfig(hardware=hw)
    args = config.build_command()

    assert "-enable-kvm" in args
    assert "-m" in args and "2048M" in args
    assert "-smp" in args and "4" in args
    assert "usb-mouse,bus=xhci.0" in config.build_command_str()
    assert "-gdb" in args and "tcp::1234" in args
    assert "-S" in args


def test_qemu_config_validation():
    """Verify detection of invalid configurations."""
    invalid_config = QemuEmulatorConfig(
        ovmf_code_path="",
        firmware_mode=QemuFirmwareMode.PFLASH_SPLIT,
        ovmf_vars_path=None,
        disk_target="",
        hardware=QemuVirtualHardwareConfig(memory_mb=64),
    )
    errors = invalid_config.validate()
    assert len(errors) >= 3
    assert any("memory" in e.lower() for e in errors)
    assert any("ovmf" in e.lower() for e in errors)
    assert any("disk_target" in e.lower() for e in errors)


def test_virtual_fat_simulator():
    """Verify on-the-fly FAT32 synthesis metadata and cluster assignments."""
    vfat = VirtualFatSimulator(host_dir="disk_img")
    files = {
        "EFI/BOOT/BOOTX64.EFI": 1488, # Single cluster
        "kernel.elf": 12500,          # 4 clusters (12500 / 4096 = 3.05 -> 4)
    }
    result = vfat.simulate_vfat_layout(files)

    assert result["sector_size"] == 512
    assert result["cluster_size_bytes"] == 4096
    assert result["root_cluster"] == 2
    assert result["num_fats"] == 2

    # Verify cluster allocation for BOOTX64.EFI
    boot_info = result["cluster_allocations"]["EFI/BOOT/BOOTX64.EFI"]
    assert boot_info["size_bytes"] == 1488
    assert len(boot_info["clusters"]) == 1
    assert boot_info["start_cluster"] == 3

    # Verify cluster allocation for kernel.elf
    kernel_info = result["cluster_allocations"]["kernel.elf"]
    assert kernel_info["size_bytes"] == 12500
    assert len(kernel_info["clusters"]) == 4
    assert kernel_info["start_cluster"] == 4

    assert len(result["vfat_limitations"]) == 3


def test_uefi_boot_sequence_simulation():
    """Verify the 6-stage UEFI platform boot sequence from Reset Vector to EfiMain."""
    steps = QemuUefiBootSequenceSimulator.get_boot_sequence()
    assert len(steps) == 6

    # Step 1: SEC
    assert steps[0].phase == UefiBootPhase.SEC
    assert steps[0].hardware_state["RIP"] == "0x00000000FFFFFFF0"

    # Step 2: PEI
    assert steps[1].phase == UefiBootPhase.PEI

    # Step 3: DXE
    assert steps[2].phase == UefiBootPhase.DXE
    assert "64-bit Long Mode" in steps[2].hardware_state["CR0"]

    # Step 4: BDS
    assert steps[3].phase == UefiBootPhase.BDS

    # Step 5: TSL
    assert steps[4].phase == UefiBootPhase.TSL
    assert steps[4].hardware_state["Path"] == "\\EFI\\BOOT\\BOOTX64.EFI"
    assert steps[4].is_mikanos_target

    # Step 6: RT (EfiMain)
    assert steps[5].phase == UefiBootPhase.RT
    assert steps[5].hardware_state["RCX"] == "ImageHandle"
    assert steps[5].is_mikanos_target


def test_qemu_monitor_simulator():
    """Verify QEMU monitor interactive commands simulation."""
    reg_res = QemuMonitorSimulator.execute_command("info registers")
    assert reg_res["status"] == "ok"
    assert "RAX=" in reg_res["output"]
    assert "RIP=" in reg_res["output"]
    assert "CR0=" in reg_res["output"]

    xp_res = QemuMonitorSimulator.execute_command("xp /16xb 0x067e4000")
    assert xp_res["status"] == "ok"
    assert "0x4d 0x5a" in xp_res["output"] # MZ Header

    pci_res = QemuMonitorSimulator.execute_command("info pci")
    assert pci_res["status"] == "ok"
    assert "xHCI" in pci_res["output"]

    block_res = QemuMonitorSimulator.execute_command("info block")
    assert block_res["status"] == "ok"
    assert "fat:rw:disk_img" in block_res["output"]


def test_cpp_qemu_launch_builder_binary():
    """Verify host-side C++20 qemu launch builder binary execution."""
    bin_path = Path("build/bin/ch01_qemu_launch_builder")
    assert bin_path.exists(), f"Binary {bin_path} not found. Please run cmake --build build first."

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "MikanOS C++20 QEMU Launch Builder" in res.stdout
    assert "Fat32Bpb size: 90 bytes" in res.stdout
    assert "FatDirectoryEntry size: 32 bytes" in res.stdout
    assert "All C++20 QEMU Launch Builder & FAT32 tests passed" in res.stdout


def test_figure_generation():
    """Verify high-resolution architecture figures are correctly saved."""
    fig1 = Path("1/result/fig1_7_qemu_ovmf_boot_architecture.png")
    fig2 = Path("1/result/fig1_8_virtual_fat_and_boot_flow.png")

    p1 = plot_qemu_ovmf_boot_architecture(str(fig1))
    p2 = plot_virtual_fat_and_boot_flow(str(fig2))

    assert Path(p1).exists()
    assert Path(p2).exists()
    assert Path(p1).stat().st_size > 50000
    assert Path(p2).stat().st_size > 50000
