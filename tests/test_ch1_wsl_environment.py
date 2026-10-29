"""Tests for Chapter 1.3: WSL Environment & Cross-Filesystem Mount Simulation."""

import os
import subprocess
from pathlib import Path

import pytest

from common import (
    WSLArchType,
    DrvFsMountOptions,
    WSLPathConverter,
    WSLStorageSimulator,
    plot_wsl1_vs_wsl2_architecture,
    plot_wsl_storage_and_usb_mount,
)


def test_wsl_environment_detection():
    """Verify detection of WSL1, WSL2, and Native Linux from kernel version strings."""
    wsl2_str = "Linux version 5.15.153.1-microsoft-standard-WSL2 (x86_64)"
    assert WSLStorageSimulator.detect_environment(wsl2_str) == WSLArchType.WSL2

    wsl1_str = "Linux version 4.4.0-19041-Microsoft (x86_64)"
    assert WSLStorageSimulator.detect_environment(wsl1_str) == WSLArchType.WSL1

    native_str = "Linux version 6.8.0-45-generic (buildd@lcy02-amd64-077) (x86_64)"
    assert WSLStorageSimulator.detect_environment(native_str) == WSLArchType.NATIVE_LINUX


def test_wsl_path_converter_windows_to_wsl():
    """Verify conversion from Windows paths to WSL POSIX paths."""
    # Standard drive path
    assert WSLPathConverter.windows_to_wsl("C:\\Users\\uchan\\mikanos") == "/mnt/c/Users/uchan/mikanos"
    assert WSLPathConverter.windows_to_wsl("D:/EFI/BOOT/BOOTX64.EFI") == "/mnt/d/EFI/BOOT/BOOTX64.EFI"

    # Forward slash path
    assert WSLPathConverter.windows_to_wsl("E:\\") == "/mnt/e/"
    assert WSLPathConverter.windows_to_wsl("E:") == "/mnt/e/"

    # UNC path from Windows to WSL
    assert WSLPathConverter.windows_to_wsl(r"\\wsl$\Ubuntu\home\student\study-mikanos") == "/home/student/study-mikanos"
    assert WSLPathConverter.windows_to_wsl("//wsl$/Ubuntu/etc/wsl.conf") == "/etc/wsl.conf"


def test_wsl_path_converter_wsl_to_windows():
    """Verify conversion from WSL POSIX paths to Windows paths."""
    # Mounted drive letter
    assert WSLPathConverter.wsl_to_windows("/mnt/c/Users/uchan/mikanos") == "C:\\Users\\uchan\\mikanos"
    assert WSLPathConverter.wsl_to_windows("/mnt/d/EFI/BOOT/BOOTX64.EFI") == "D:\\EFI\\BOOT\\BOOTX64.EFI"

    # Native Linux path into UNC
    assert WSLPathConverter.wsl_to_windows("/home/student/mikanos", distro_name="Ubuntu") == r"\\wsl$\Ubuntu\home\student\mikanos"


def test_drvfs_mount_options_and_permissions():
    """Verify DrvFs mount options generation and effective POSIX permission calculations."""
    opts = DrvFsMountOptions(
        metadata=True,
        case="dir",
        uid=1000,
        gid=1000,
        umask=0o022,
        fmask=0o111,
        dmask=0o000,
    )
    flag_str = opts.to_mount_flags_string()
    assert "metadata" in flag_str
    assert "case=dir" in flag_str
    assert "uid=1000" in flag_str
    assert "fmask=111" in flag_str

    # Without metadata: file gets masked execute permissions
    file_perm = opts.calculate_file_permissions(is_directory=False, has_metadata=False)
    assert file_perm == 0o644  # 0777 & ~(0111 | 0022) = 0644

    dir_perm = opts.calculate_file_permissions(is_directory=True, has_metadata=False)
    assert dir_perm == 0o755   # 0777 & ~(0000 | 0022) = 0755

    # With metadata: stored permission preserved
    stored_script_perm = opts.calculate_file_permissions(is_directory=False, has_metadata=True, stored_mode=0o755)
    assert stored_script_perm == 0o755


def test_proc_mounts_parser():
    """Verify parsing of /proc/mounts style data for DrvFs detection."""
    sample_mounts = """
    /dev/sdb / ext4 rw,relatime,discard,errors=remount-ro,data=ordered 0 0
    tools /init 9p ro,relatime,dirsync,aname=tools;fmask=0222,mmap 0 0
    C:\\ /mnt/c 9p rw,noatime,dirsync,aname=drvfs;path=C:\\;uid=1000;gid=1000;metadata;case=dir 0 0
    D:\\ /mnt/d 9p rw,noatime,dirsync,aname=drvfs;path=D:\\;uid=1000;gid=1000;metadata;case=dir 0 0
    """
    entries = WSLStorageSimulator.parse_proc_mounts(sample_mounts)
    assert len(entries) == 4

    drvfs_entries = [e for e in entries if e["is_drvfs"]]
    assert len(drvfs_entries) == 2
    assert drvfs_entries[0]["mount_point"] == "/mnt/c"
    assert drvfs_entries[1]["mount_point"] == "/mnt/d"


def test_storage_performance_matrix():
    """Verify storage performance matrix entries."""
    matrix = WSLStorageSimulator.get_storage_performance_matrix()
    assert "ext4_vhdx" in matrix
    assert "drvfs_9p" in matrix
    assert "usbip_raw" in matrix
    assert matrix["ext4_vhdx"]["case_sensitive"] is True


def test_boot_media_copy_script_generation():
    """Verify boot media deployment shell script generation."""
    script_lines = WSLStorageSimulator.generate_boot_media_copy_script("hello.efi", "D:")
    combined = "\n".join(script_lines)
    assert "mount -t drvfs D: /mnt/d" in combined
    assert "/mnt/d/EFI/BOOT" in combined
    assert "sync" in combined


def test_cpp20_wsl_environment_inspector_executable():
    """Verify C++20 WSL environment inspector runs and passes deterministic verification."""
    bin_path = Path("build/bin/ch01_wsl_environment_inspector")
    if not bin_path.exists():
        bin_path = Path("build/ch01_wsl_environment_inspector")

    assert bin_path.exists(), f"Executable {bin_path} must exist after CMake build"

    res = subprocess.run([str(bin_path), "--demo"], capture_output=True, text=True, check=True)
    assert "MikanOS WSL & Cross-Filesystem Inspector" in res.stdout
    assert "[SUCCESS] All WSL cross-filesystem checks and validations passed successfully." in res.stdout


def test_wsl_figures_generation():
    """Verify high-resolution WSL architecture and storage figures are created."""
    fig1 = Path("1/result/fig1_5_wsl1_vs_wsl2_architecture.png")
    fig2 = Path("1/result/fig1_6_wsl_storage_and_usb_mount.png")

    plot_wsl1_vs_wsl2_architecture(str(fig1))
    plot_wsl_storage_and_usb_mount(str(fig2))

    assert fig1.exists()
    assert fig1.stat().st_size > 10_000
    assert fig2.exists()
    assert fig2.stat().st_size > 10_000
