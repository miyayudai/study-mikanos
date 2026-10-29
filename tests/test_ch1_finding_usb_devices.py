"""Tests for Chapter 1.2: Finding USB Device Names & Linux Block Subsystem (DoD verification)."""

from pathlib import Path
import subprocess
import pytest
from common.os_sim import (
    LinuxBlockDevice,
    LinuxPartition,
    LinuxBlockSubsystemSimulator,
    plot_linux_block_layer_and_usb_detection,
    plot_uefi_usb_fat_structure,
)


def test_dev_t_encoding_and_decoding():
    """Verify Linux dev_t 32-bit major/minor encoding and decoding bit-level symmetry."""
    test_cases = [
        (8, 0),       # /dev/sda
        (8, 16),      # /dev/sdb
        (8, 17),      # /dev/sdb1
        (259, 0),     # /dev/nvme0n1
        (259, 1),     # /dev/nvme0n1p1
        (7, 0),       # /dev/loop0
        (1, 1),       # /dev/ram1
    ]
    for maj, min_ in test_cases:
        encoded = LinuxBlockSubsystemSimulator.encode_dev_t(maj, min_)
        dec_maj, dec_min = LinuxBlockSubsystemSimulator.decode_dev_t(encoded)
        assert dec_maj == maj, f"Decoded major {dec_maj} != {maj}"
        assert dec_min == min_, f"Decoded minor {dec_min} != {min_}"


def test_major_device_canonical_names():
    """Verify identification of standard Linux block subsystems."""
    assert "sd" in LinuxBlockSubsystemSimulator.get_standard_major_name(8)
    assert "nvme" in LinuxBlockSubsystemSimulator.get_standard_major_name(259)
    assert "loop" in LinuxBlockSubsystemSimulator.get_standard_major_name(7)
    assert "ram" in LinuxBlockSubsystemSimulator.get_standard_major_name(1)


def test_workstation_state_and_internal_drive_safety():
    """Verify workstation state correctly identifies internal non-removable drives as dangerous."""
    devices = LinuxBlockSubsystemSimulator.create_typical_workstation_state()
    assert len(devices) == 1
    host_nvme = devices[0]
    assert host_nvme.name == "nvme0n1"
    assert not host_nvme.is_removable
    assert not host_nvme.is_usb_boot_candidate
    assert host_nvme.size_gib > 900.0  # ~1TB drive

    # Safety check on internal drive
    safety = LinuxBlockSubsystemSimulator.verify_target_safety("nvme0n1", devices)
    assert safety["found"] is True
    assert safety["is_safe"] is False
    assert "DANGER" in str(safety["reason"])


def test_usb_insertion_and_detection():
    """Verify USB insertion differential detection logic."""
    before = LinuxBlockSubsystemSimulator.create_typical_workstation_state()
    after = LinuxBlockSubsystemSimulator.simulate_usb_insertion(before)
    assert len(after) == len(before) + 1

    new_devs = LinuxBlockSubsystemSimulator.detect_inserted_devices(before, after)
    assert len(new_devs) == 1
    usb_dev = new_devs[0]
    assert usb_dev.name == "sdb"
    assert usb_dev.is_removable is True
    assert usb_dev.transport == "USB"
    assert usb_dev.is_usb_boot_candidate is True
    assert 14.0 < usb_dev.size_gib < 17.0  # 16GB USB drive

    # Partitions
    assert len(usb_dev.partitions) == 1
    assert usb_dev.partitions[0].name == "sdb1"

    # Safety verification
    safety = LinuxBlockSubsystemSimulator.verify_target_safety("sdb", after)
    assert safety["found"] is True
    assert safety["is_safe"] is True
    assert "Safe Removable USB" in str(safety["reason"])


def test_safety_check_nonexistent_device():
    """Verify safety check gracefully handles non-existent devices."""
    devices = LinuxBlockSubsystemSimulator.create_typical_workstation_state()
    safety = LinuxBlockSubsystemSimulator.verify_target_safety("sdz", devices)
    assert safety["found"] is False
    assert safety["is_safe"] is False


def test_figure_generation_outputs(tmp_path: Path):
    """Verify figure plotting outputs valid non-empty PNG files."""
    f1 = tmp_path / "fig1_3_test.png"
    f2 = tmp_path / "fig1_4_test.png"

    plot_linux_block_layer_and_usb_detection(f1)
    plot_uefi_usb_fat_structure(f2)

    assert f1.exists() and f1.stat().st_size > 100_000
    assert f2.exists() and f2.stat().st_size > 100_000


def test_cpp20_block_device_inspector_executable():
    """Verify C++20 block_device_inspector runs cleanly with --demo argument."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch01_block_device_inspector"
    if not bin_path.exists():
        pytest.skip("ch01_block_device_inspector binary not built yet in build/bin/")

    proc = subprocess.run([str(bin_path), "--demo"], capture_output=True, text=True, check=True)
    out = proc.stdout
    assert "Linux Block Device Inspector" in out
    assert "nvme0n1" in out
    assert "sdb" in out
    assert "SAFE (USB Boot Candidate)" in out
    assert "DANGER (Host Disk)" in out
