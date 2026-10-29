"""Unit tests for Chapter 1.5: Behind the Scenes Boot Sequence and UEFI Phases."""

import os
import subprocess
from pathlib import Path

import pytest

from common import (
    CpuOperatingMode,
    CpuResetVector,
    NvramBootEntry,
    StorageVolume,
    TargetArchitecture,
    UefiBootManagerSimulator,
    UefiPhaseDetail,
    UefiPlatformPhase,
    get_uefi_platform_phases,
    plot_cpu_reset_to_uefi_phases,
    plot_uefi_boot_device_and_file_resolution,
)


def test_cpu_reset_vector_calculation():
    """Verifies x86-64 CPU hardware reset vector calculation and register state."""
    reset = CpuResetVector()
    assert reset.rip == 0xFFF0
    assert reset.cs_selector == 0xF000
    assert reset.cs_base == 0xFFFF_0000
    assert reset.physical_address == 0xFFFF_FFF0
    assert reset.distance_from_4gib == 16

    summary = reset.summary()
    assert summary["physical_reset_vector"] == "0xFFFFFFF0"
    assert summary["distance_from_4gib_bytes"] == 16
    assert summary["pe_bit"] == 0  # Real Mode
    assert summary["bist_passed"] is True


def test_target_architectures_and_fallback_specs():
    """Verifies canonical UEFI fallback filenames and paths per UEFI Spec Section 3.5."""
    assert TargetArchitecture.X86_64.fallback_filename == "BOOTX64.EFI"
    assert TargetArchitecture.X86_64.fallback_path == "\\EFI\\BOOT\\BOOTX64.EFI"

    assert TargetArchitecture.AARCH64.fallback_filename == "BOOTAA64.EFI"
    assert TargetArchitecture.AARCH64.fallback_path == "\\EFI\\BOOT\\BOOTAA64.EFI"

    assert TargetArchitecture.IA32.fallback_filename == "BOOTIA32.EFI"
    assert TargetArchitecture.IA32.fallback_path == "\\EFI\\BOOT\\BOOTIA32.EFI"

    assert TargetArchitecture.ARM32.fallback_filename == "BOOTARM.EFI"
    assert TargetArchitecture.ARM32.fallback_path == "\\EFI\\BOOT\\BOOTARM.EFI"

    assert TargetArchitecture.RISCV64.fallback_filename == "BOOTRISCV64.EFI"
    assert TargetArchitecture.RISCV64.fallback_path == "\\EFI\\BOOT\\BOOTRISCV64.EFI"


def test_uefi_platform_phases_integrity():
    """Verifies that all 5 UEFI platform initialization phases are defined correctly."""
    phases = get_uefi_platform_phases()
    assert len(phases) == 5

    phase_enums = [p.phase for p in phases]
    assert phase_enums == [
        UefiPlatformPhase.SEC,
        UefiPlatformPhase.PEI,
        UefiPlatformPhase.DXE,
        UefiPlatformPhase.BDS,
        UefiPlatformPhase.TSL,
    ]

    sec_phase = phases[0]
    assert sec_phase.cpu_mode == CpuOperatingMode.PROTECTED_MODE_32
    assert "Cache-as-RAM" in sec_phase.memory_type

    dxe_phase = phases[2]
    assert dxe_phase.cpu_mode == CpuOperatingMode.LONG_MODE_64
    assert "Identity Paging" in dxe_phase.memory_type

    tsl_phase = phases[4]
    assert "EfiMain" in tsl_phase.key_events[0]


def test_uefi_boot_manager_nvram_resolution():
    """Verifies that BDS resolves target from NVRAM BootOrder when present and valid."""
    bm = UefiBootManagerSimulator(TargetArchitecture.X86_64)

    vol = StorageVolume(
        label="NVME_OS_DRIVE",
        partition_type="GPT ESP",
        fs_type="FAT32",
        is_removable=False,
        files={"EFI/UBUNTU/SHIMX64.EFI": 1024000},
    )
    bm.add_volume(vol)

    entry1 = NvramBootEntry(
        boot_id=1,
        description="Disabled Fedora Entry",
        target_volume_label="NVME_OS_DRIVE",
        target_file_path="\\EFI\\fedora\\grubx64.efi",
        is_enabled=False,
    )
    entry2 = NvramBootEntry(
        boot_id=2,
        description="Ubuntu Production Boot",
        target_volume_label="NVME_OS_DRIVE",
        target_file_path="\\EFI\\ubuntu\\shimx64.efi",
        is_enabled=True,
    )
    bm.add_nvram_entry(entry1)
    bm.add_nvram_entry(entry2)

    report = bm.resolve()
    assert report.success is True
    assert report.selected_volume == "NVME_OS_DRIVE"
    assert report.resolved_file_path == "\\EFI\\ubuntu\\shimx64.efi"
    assert "Boot0002" in report.resolution_method


def test_uefi_boot_manager_removable_fallback():
    """Verifies that BDS falls back to removable media \\EFI\\BOOT\\BOOTX64.EFI."""
    bm = UefiBootManagerSimulator(TargetArchitecture.X86_64)

    # Incompatible non-FAT partition (e.g. Linux ext4 root)
    ext4_vol = StorageVolume(
        label="LINUX_ROOT",
        partition_type="GPT",
        fs_type="EXT4",
        is_removable=False,
        files={"EFI/BOOT/BOOTX64.EFI": 1440},
    )
    bm.add_volume(ext4_vol)

    # Removable USB stick with MikanOS FAT32
    usb_vol = StorageVolume(
        label="MIKANOS_USB",
        partition_type="MBR FAT",
        fs_type="FAT32",
        is_removable=True,
        files={"EFI/BOOT/BOOTX64.EFI": 1440},
    )
    bm.add_volume(usb_vol)

    report = bm.resolve()
    assert report.success is True
    assert report.selected_volume == "MIKANOS_USB"
    assert report.resolved_file_path == "\\EFI\\BOOT\\BOOTX64.EFI"
    assert "Removable Media Fallback" in report.resolution_method


def test_uefi_boot_manager_failure_case():
    """Verifies that BDS reports failure when no compatible volume or file exists."""
    bm = UefiBootManagerSimulator(TargetArchitecture.X86_64)
    empty_vol = StorageVolume(
        label="BLANK_DRIVE",
        partition_type="GPT",
        fs_type="NTFS",
        is_removable=True,
        files={},
    )
    bm.add_volume(empty_vol)

    report = bm.resolve()
    assert report.success is False
    assert report.selected_volume is None
    assert report.resolved_file_path is None


def test_cpp20_boot_sequence_tracer_executable():
    """Verifies that the C++20 boot_sequence_tracer binary compiles and passes self-tests."""
    binary_path = Path("build/bin/ch01_boot_sequence_tracer")
    if not binary_path.exists():
        # Build if not built
        subprocess.run(["cmake", "--build", "build", "--target", "ch01_boot_sequence_tracer"], check=True)

    assert binary_path.exists()

    # Test mode
    res_test = subprocess.run([str(binary_path), "--test"], capture_output=True, text=True, check=True)
    assert "All C++20 Boot Sequence Tracer Self-Tests Passed Successfully!" in res_test.stdout

    # Normal mode output verification
    res_normal = subprocess.run([str(binary_path)], capture_output=True, text=True, check=True)
    out = res_normal.stdout
    assert "0xFFFFFFF0" in out
    assert "SEC (Security Phase)" in out
    assert "PEI (Pre-EFI Initialization Phase)" in out
    assert "DXE (Driver Execution Environment Phase)" in out
    assert "BDS (Boot Device Selection Phase)" in out
    assert "TSL (Transient System Load Phase)" in out
    assert "BOOTX64.EFI" in out
    assert "Hello, world!" in out


def test_figure_generation():
    """Verifies generation of high-resolution architectural figures for Chapter 1.5."""
    fig1 = "1/result/fig1_9_cpu_reset_to_uefi_phases.png"
    fig2 = "1/result/fig1_10_uefi_boot_device_and_file_resolution.png"

    path1 = plot_cpu_reset_to_uefi_phases(fig1)
    path2 = plot_uefi_boot_device_and_file_resolution(fig2)

    assert Path(path1).exists()
    assert Path(path2).exists()
    assert Path(path1).stat().st_size > 50_000
    assert Path(path2).stat().st_size > 50_000
