"""Unit tests for Chapter 2.5: UEFI GetMemoryMap Protocol & MemoryMap Struct."""

import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common.os_sim import (
    MemoryMap,
    UefiBootServicesSimulator,
    UefiMemoryType,
    UefiStatus,
    plot_uefi_get_memory_map_protocol_and_buffer_flow,
    plot_uefi_mapkey_lifecycle_and_exit_boot_services_consistency,
)


def test_memory_map_struct_basics():
    """Verify MemoryMap struct allocation, initial fields, and entry counting."""
    memmap = MemoryMap.allocate(16384)
    assert memmap.buffer_size == 16384
    assert memmap.buffer is not None
    assert len(memmap.buffer) == 16384
    assert memmap.map_size == 0
    assert memmap.map_key == 0
    assert memmap.entry_count == 0

    # Null buffer handling
    null_map = MemoryMap(buffer_size=1024, buffer=None)
    with pytest.raises(ValueError, match="Buffer is null"):
        null_map.get_descriptor(0)


def test_uefi_boot_services_get_memory_map_insufficient_buffer():
    """Verify that buffer size smaller than needed returns EFI_BUFFER_TOO_SMALL and sets required size."""
    bs = UefiBootServicesSimulator(initial_map_key=42)
    small_map = MemoryMap(buffer_size=64, buffer=bytearray(64))

    status = bs.get_memory_map(small_map)
    assert status == UefiStatus.BUFFER_TOO_SMALL
    assert UefiStatus.is_error(status)
    assert small_map.map_size == bs.memory_map.map_size_bytes
    assert small_map.map_size > 64


def test_uefi_boot_services_get_memory_map_success():
    """Verify successful GetMemoryMap() retrieval with standard 16KiB buffer and descriptor iteration."""
    bs = UefiBootServicesSimulator(initial_map_key=77)
    memmap = MemoryMap.allocate(16384)

    status = bs.get_memory_map(memmap)
    assert status == UefiStatus.SUCCESS
    assert not UefiStatus.is_error(status)
    assert memmap.map_size == bs.memory_map.map_size_bytes
    assert memmap.map_key == 77
    assert memmap.descriptor_size == 48
    assert memmap.descriptor_version == 1
    assert memmap.entry_count == bs.memory_map.total_entries

    # Iterate over descriptors
    descriptors = list(memmap)
    assert len(descriptors) == memmap.entry_count

    # Check that conventional memory entries match simulator expectation
    conv_descriptors = [d for d in descriptors if d.is_conventional]
    assert len(conv_descriptors) > 0
    conv_bytes = sum(d.size_bytes for d in conv_descriptors)
    assert conv_bytes == bs.memory_map.total_conventional_bytes()


def test_uefi_mapkey_invalidation_and_exit_boot_services():
    """Verify MapKey increment on memory allocation and ExitBootServices consistency validation."""
    bs = UefiBootServicesSimulator(initial_map_key=100)
    memmap = MemoryMap.allocate(16384)

    # 1. Fetch initial memory map
    status = bs.get_memory_map(memmap)
    assert status == UefiStatus.SUCCESS
    assert memmap.map_key == 100

    # 2. Allocate pages (alters map and increments internal key)
    allocated_addr = bs.allocate_pages(UefiMemoryType.EfiBootServicesData, pages=8)
    assert allocated_addr > 0
    assert bs.map_key == 101

    # 3. Attempt ExitBootServices with stale MapKey (100) -> Must fail with EFI_INVALID_PARAMETER
    exit_status = bs.exit_boot_services(image_handle=1, map_key=memmap.map_key)
    assert exit_status == UefiStatus.INVALID_PARAMETER
    assert not bs.boot_services_exited

    # 4. Re-fetch memory map -> MapKey matches current (101)
    status = bs.get_memory_map(memmap)
    assert status == UefiStatus.SUCCESS
    assert memmap.map_key == 101

    # 5. ExitBootServices with matching MapKey -> Must succeed
    exit_status = bs.exit_boot_services(image_handle=1, map_key=memmap.map_key)
    assert exit_status == UefiStatus.SUCCESS
    assert bs.boot_services_exited

    # Post-exit: subsequent allocation or memory map fetch is rejected
    with pytest.raises(RuntimeError, match="Cannot allocate pages after ExitBootServices"):
        bs.allocate_pages(UefiMemoryType.EfiLoaderData, 4)

    post_status = bs.get_memory_map(memmap)
    assert post_status == UefiStatus.UNSUPPORTED


def test_cpp20_get_memory_map_sim_standalone():
    """Verify host-side C++20 simulation executable builds and passes all tests."""
    proj_root = Path(__file__).resolve().parent.parent
    bin_path = proj_root / "build" / "bin" / "ch02_get_memory_map_sim"

    if not bin_path.exists():
        subprocess.run(["cmake", "--build", str(proj_root / "build"), "--target", "ch02_get_memory_map_sim"], check=True)

    result = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert "[PASS] TestBufferTooSmallHandling" in result.stdout
    assert "[PASS] TestFixed16KiBBufferSuccess" in result.stdout
    assert "[PASS] TestMapKeyLifecycleAndExitBootServices" in result.stdout
    assert "[PASS] TestPointerSteppingVsTypeIncrement" in result.stdout
    assert "[ALL C++20 TESTS PASSED SUCCESSFULLY]" in result.stdout


def test_figure_generation_outputs():
    """Verify that both Section 2.5 technical diagrams are generated at high resolution."""
    proj_root = Path(__file__).resolve().parent.parent
    fig9_path = proj_root / "2" / "result" / "fig2_09_uefi_get_memory_map_protocol_and_buffer_flow.png"
    fig10_path = proj_root / "2" / "result" / "fig2_10_uefi_mapkey_lifecycle_and_exit_boot_services_consistency.png"

    plot_uefi_get_memory_map_protocol_and_buffer_flow(str(fig9_path))
    plot_uefi_mapkey_lifecycle_and_exit_boot_services_consistency(str(fig10_path))

    assert fig9_path.exists()
    assert fig10_path.exists()
    assert fig9_path.stat().st_size > 100_000
    assert fig10_path.stat().st_size > 100_000

    with Image.open(fig9_path) as img:
        assert img.width >= 1800
        assert img.height >= 800

    with Image.open(fig10_path) as img:
        assert img.width >= 1800
        assert img.height >= 1000
