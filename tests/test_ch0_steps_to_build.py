"""Unit tests for Chapter 0 Section 0.3: Steps to Build an OS."""

from pathlib import Path
import subprocess
import pytest

from common import (
    FREESTANDING_FLAGS,
    BuildArtifact,
    BuildPipelineSimulator,
    BuildStage,
    CompilerFlag,
    QemuLaunchConfig,
    UEFIBootMediaValidator,
)


def test_freestanding_compiler_flags() -> None:
    """Verify bare-metal freestanding compiler flags specification."""
    flags_by_name = {flag.name: flag for flag in FREESTANDING_FLAGS}

    assert "-ffreestanding" in flags_by_name
    assert "-nostdlib" in flags_by_name
    assert "-mno-red-zone" in flags_by_name
    assert "-fno-exceptions" in flags_by_name
    assert "-fno-rtti" in flags_by_name

    assert flags_by_name["-ffreestanding"].is_freestanding_required is True
    assert flags_by_name["-mno-red-zone"].is_freestanding_required is True
    assert "red zone" in flags_by_name["-mno-red-zone"].explanation.lower()


def test_build_pipeline_simulator() -> None:
    """Verify build pipeline stage execution and artifact generation."""
    sim = BuildPipelineSimulator(source_lines=300, is_freestanding=True, disable_red_zone=True)

    # 1. Validation of compiler flags
    valid_flags = ["-ffreestanding", "-nostdlib", "-mno-red-zone", "-fno-exceptions", "-fno-rtti", "-O2"]
    ok, errors = sim.validate_flags(valid_flags)
    assert ok is True
    assert len(errors) == 0

    bad_flags = ["-ffreestanding", "-O2"]  # missing -nostdlib, -mno-red-zone, etc.
    ok_bad, errors_bad = sim.validate_flags(bad_flags)
    assert ok_bad is False
    assert len(errors_bad) == 4

    # 2. Build UEFI Bootloader artifacts
    uefi_artifacts = sim.run_pipeline(target_type="uefi_bootloader")
    assert len(uefi_artifacts) == 4
    stages = [a.stage for a in uefi_artifacts]
    assert stages == [BuildStage.PREPROCESS, BuildStage.COMPILE, BuildStage.ASSEMBLE, BuildStage.LINK]
    assert uefi_artifacts[-1].filename == "BOOTX64.EFI"
    assert "PE32+" in uefi_artifacts[-1].file_format

    # 3. Build Kernel ELF artifacts
    kernel_artifacts = sim.run_pipeline(target_type="kernel")
    assert len(kernel_artifacts) == 4
    assert kernel_artifacts[-1].filename == "kernel.elf"
    assert "ELF64" in kernel_artifacts[-1].file_format


def test_uefi_boot_media_validator() -> None:
    """Verify UEFI boot media layout validation logic."""
    validator = UEFIBootMediaValidator(filesystem_type="FAT32")

    # Empty media must fail validation
    ok_empty, errs_empty = validator.validate()
    assert ok_empty is False
    assert any("Missing mandatory UEFI" in e for e in errs_empty)

    # Add standard UEFI bootloader and kernel
    validator.add_file(r"\EFI\BOOT\BOOTX64.EFI", size_bytes=32768, file_type="PE32+ EFI")
    validator.add_file(r"\kernel.elf", size_bytes=65536, file_type="ELF64 Kernel")

    ok_valid, errs_valid = validator.validate()
    assert ok_valid is True
    assert len(errs_valid) == 0
    assert len(validator.get_file_list()) == 2

    # Invalid filesystem type must fail
    bad_fs_validator = UEFIBootMediaValidator(filesystem_type="ext4")
    bad_fs_validator.add_file("/EFI/BOOT/BOOTX64.EFI", 1024, "PE32+ EFI")
    ok_fs, errs_fs = bad_fs_validator.validate()
    assert ok_fs is False
    assert any("Invalid filesystem" in e for e in errs_fs)


def test_qemu_launch_config() -> None:
    """Verify QEMU launch configuration and command-line assembly."""
    cfg = QemuLaunchConfig(
        ovmf_path="/usr/share/OVMF/OVMF_CODE.fd",
        disk_image="build/disk.img",
        memory="2G",
        monitor="stdio",
        gdb_debug=True,
    )
    cmd = cfg.build_command()
    cmd_str = " ".join(cmd)

    assert "qemu-system-x86_64" in cmd_str
    assert "-m 2G" in cmd_str
    assert "OVMF_CODE.fd" in cmd_str
    assert "build/disk.img" in cmd_str
    assert "-monitor stdio" in cmd_str
    assert "-gdb tcp::1234 -S" in cmd_str

    explanations = cfg.explain_options()
    assert len(explanations) == 5


def test_cpp20_os_build_steps_execution(tmp_path: Path) -> None:
    """Compile and execute standalone C++20 src/ch00/os_build_steps.cpp."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch00" / "os_build_steps.cpp"
    assert cpp_src.exists()

    bin_path = tmp_path / "os_build_steps"
    compile_res = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", str(cpp_src), "-o", str(bin_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0
    assert "Freestanding Environment Check Passed" in run_res.stdout
    assert "Pipeline Execution Passed" in run_res.stdout
    assert "UEFI Media Validation Passed" in run_res.stdout


def test_generated_figures_exist() -> None:
    """Ensure Section 0.3 high-resolution figures exist and are sufficiently large (> 100KB)."""
    repo_root = Path(__file__).resolve().parent.parent
    fig5 = repo_root / "0" / "result" / "fig0_5_os_build_pipeline.png"
    fig6 = repo_root / "0" / "result" / "fig0_6_uefi_boot_media_and_execution.png"

    assert fig5.exists() and fig5.stat().st_size > 100_000
    assert fig6.exists() and fig6.stat().st_size > 100_000
