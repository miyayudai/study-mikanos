"""Unit tests for Chapter 2.4: UEFI Memory Map & EFI_MEMORY_DESCRIPTOR Concepts."""

import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common.os_sim import (
    UefiMemoryType,
    UefiMemoryDescriptor,
    UefiMemoryMapSimulator,
    EFI_MEMORY_UC,
    EFI_MEMORY_WC,
    EFI_MEMORY_WT,
    EFI_MEMORY_WB,
    EFI_MEMORY_XP,
    EFI_MEMORY_RUNTIME,
    plot_efi_memory_descriptor_layout_and_attributes,
    plot_uefi_memory_types_classification_and_lifecycle,
)


def test_uefi_memory_type_enumeration_and_classification():
    """Verify all 16 EFI_MEMORY_TYPE enumerations and their OS usability properties."""
    assert len(UefiMemoryType) == 16
    assert UefiMemoryType.EfiReservedMemoryType.value == 0
    assert UefiMemoryType.EfiLoaderCode.value == 1
    assert UefiMemoryType.EfiLoaderData.value == 2
    assert UefiMemoryType.EfiBootServicesCode.value == 3
    assert UefiMemoryType.EfiBootServicesData.value == 4
    assert UefiMemoryType.EfiRuntimeServicesCode.value == 5
    assert UefiMemoryType.EfiRuntimeServicesData.value == 6
    assert UefiMemoryType.EfiConventionalMemory.value == 7
    assert UefiMemoryType.EfiUnusableMemory.value == 8
    assert UefiMemoryType.EfiACPIReclaimMemory.value == 9
    assert UefiMemoryType.EfiACPIMemoryNVS.value == 10
    assert UefiMemoryType.EfiMemoryMappedIO.value == 11
    assert UefiMemoryType.EfiMemoryMappedIOPortSpace.value == 12
    assert UefiMemoryType.EfiPalCode.value == 13
    assert UefiMemoryType.EfiPersistentMemory.value == 14
    assert UefiMemoryType.EfiMaxMemoryType.value == 15

    # Conventional memory is the only immediately usable RAM before ExitBootServices
    for mem_type in UefiMemoryType:
        if mem_type == UefiMemoryType.EfiConventionalMemory:
            assert mem_type.is_usable_by_os_initially
        else:
            assert not mem_type.is_usable_by_os_initially

    # Reclaimable after ExitBootServices
    reclaimables = {
        UefiMemoryType.EfiBootServicesCode,
        UefiMemoryType.EfiBootServicesData,
        UefiMemoryType.EfiLoaderCode,
        UefiMemoryType.EfiLoaderData,
    }
    for mem_type in UefiMemoryType:
        if mem_type in reclaimables:
            assert mem_type.is_reclaimable_after_exit_boot_services
        else:
            assert not mem_type.is_reclaimable_after_exit_boot_services

    # Runtime required
    runtime_required = {
        UefiMemoryType.EfiRuntimeServicesCode,
        UefiMemoryType.EfiRuntimeServicesData,
        UefiMemoryType.EfiMemoryMappedIO,
        UefiMemoryType.EfiACPIMemoryNVS,
        UefiMemoryType.EfiPalCode,
    }
    for mem_type in UefiMemoryType:
        if mem_type in runtime_required:
            assert mem_type.is_runtime_required
        else:
            assert not mem_type.is_runtime_required


def test_uefi_memory_descriptor_struct_layout_and_math():
    """Verify EFI_MEMORY_DESCRIPTOR sizing, page conversion, and binary serialization."""
    desc = UefiMemoryDescriptor(
        type=UefiMemoryType.EfiConventionalMemory,
        physical_start=0x00100000,
        virtual_start=0,
        number_of_pages=256,  # 1MB
        attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
    )

    assert desc.size_bytes == 256 * 4096  # 1048576 bytes = 1 MiB
    assert desc.physical_end == 0x00100000 + 0x00100000  # 0x00200000
    assert desc.is_page_aligned
    assert desc.is_conventional
    assert not desc.is_reclaimable_after_exit_boot_services
    assert not desc.is_runtime_service
    assert not desc.is_runtime_mapped_required
    assert "WB" in desc.attribute_names()
    assert "UC" in desc.attribute_names()

    # Serialization with 40 bytes (tight packing)
    raw40 = desc.to_bytes(descriptor_size=40)
    assert len(raw40) == 40
    restored40 = UefiMemoryDescriptor.from_bytes(raw40, descriptor_size=40)
    assert restored40.type == desc.type
    assert restored40.physical_start == desc.physical_start
    assert restored40.number_of_pages == desc.number_of_pages
    assert restored40.attribute == desc.attribute

    # Serialization with 48 bytes (standard OVMF descriptor size with 8 bytes padding)
    raw48 = desc.to_bytes(descriptor_size=48)
    assert len(raw48) == 48
    restored48 = UefiMemoryDescriptor.from_bytes(raw48, descriptor_size=48)
    assert restored48.type == desc.type
    assert restored48.physical_start == desc.physical_start
    assert restored48.number_of_pages == desc.number_of_pages
    assert restored48.attribute == desc.attribute


def test_uefi_memory_map_pointer_arithmetic_and_stepping():
    """Demonstrate the critical bug of C++ desc++ vs iter += descriptor_size."""
    # Create two consecutive descriptors
    d0 = UefiMemoryDescriptor(
        type=UefiMemoryType.EfiBootServicesCode,
        physical_start=0x00000000,
        virtual_start=0,
        number_of_pages=159,
        attribute=EFI_MEMORY_WB,
    )
    d1 = UefiMemoryDescriptor(
        type=UefiMemoryType.EfiConventionalMemory,
        physical_start=0x00100000,
        virtual_start=0,
        number_of_pages=1024,
        attribute=EFI_MEMORY_WB,
    )

    descriptor_size = 48  # Firmware returned descriptor_size = 48
    buffer = d0.to_bytes(descriptor_size) + d1.to_bytes(descriptor_size)
    assert len(buffer) == 96

    # CORRECT stepping: using byte offset increment (iter += descriptor_size)
    first_desc = UefiMemoryDescriptor.from_bytes(buffer[0:48], descriptor_size)
    second_desc_correct = UefiMemoryDescriptor.from_bytes(buffer[48:96], descriptor_size)

    assert first_desc.type == UefiMemoryType.EfiBootServicesCode
    assert second_desc_correct.type == UefiMemoryType.EfiConventionalMemory
    assert second_desc_correct.physical_start == 0x00100000

    # INCORRECT stepping: pretending buffer is an array of sizeof(EFI_MEMORY_DESCRIPTOR) = 40
    # Next element would start at offset 40 instead of 48!
    corrupted_bytes = buffer[40:80]
    corrupted_desc = UefiMemoryDescriptor.from_bytes(corrupted_bytes, 40)
    # The Type field in corrupted_desc now reads the 8-byte padding from d0!
    # Because padding is 0x00000000, it misinterprets it as EfiReservedMemoryType!
    assert corrupted_desc.type != UefiMemoryType.EfiConventionalMemory
    assert corrupted_desc.physical_start != 0x00100000


def test_uefi_memory_map_simulator_standard_qemu_ovmf():
    """Verify standard QEMU OVMF memory map parsing, aggregation, and lifecycle tracking."""
    sim = UefiMemoryMapSimulator.create_standard_uefi_map()
    assert sim.total_entries == 13
    assert sim.map_size_bytes == 13 * 48

    conv_bytes = sim.total_conventional_bytes()
    # 32512 + 498816 + 524288 = 1055616 pages * 4096 = 4,323,803,136 bytes (~4.026 GB)
    assert conv_bytes == 1055616 * 4096

    reclaim_bytes = sim.total_boot_services_reclaimable_bytes()
    # 159 + 128 + 256 = 543 pages * 4096 = 2,224,128 bytes (~2.12 MiB)
    assert reclaim_bytes == 543 * 4096

    runtime_bytes = sim.total_runtime_bytes()
    # 128 + 192 = 320 pages * 4096 = 1,310,720 bytes (1.25 MiB)
    assert runtime_bytes == 320 * 4096

    # Serialization round-trip
    buf = sim.serialize()
    assert len(buf) == 13 * 48
    deserialized = UefiMemoryMapSimulator.deserialize(buf, descriptor_size=48)
    assert deserialized.total_entries == 13
    assert deserialized.total_conventional_bytes() == conv_bytes
    assert deserialized.total_boot_services_reclaimable_bytes() == reclaim_bytes


def test_cpp20_uefi_memory_map_concept_sim_standalone():
    """Execute the compiled C++20 memory map simulation binary with --test flag."""
    bin_path = Path("build/bin/ch02_uefi_memory_map_concept_sim")
    if not bin_path.exists():
        bin_path = Path("build/ch02_uefi_memory_map_concept_sim")

    assert bin_path.exists(), f"Target binary {bin_path} not found. Run 'cmake --build build' first."
    result = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True)
    assert result.returncode == 0, f"C++ self-test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "All UEFI Memory Map self-tests passed cleanly" in result.stdout


def test_figure_generation_outputs(tmp_path):
    """Verify that high-DPI memory map figures are generated correctly and meet quality specs."""
    fig7_path = tmp_path / "fig2_07_test.png"
    fig8_path = tmp_path / "fig2_08_test.png"

    plot_efi_memory_descriptor_layout_and_attributes(str(fig7_path))
    plot_uefi_memory_types_classification_and_lifecycle(str(fig8_path))

    assert fig7_path.exists()
    assert fig8_path.exists()

    with Image.open(fig7_path) as img:
        assert img.size[0] >= 1800
        assert img.size[1] >= 1000

    with Image.open(fig8_path) as img:
        assert img.size[0] >= 1800
        assert img.size[1] >= 1000
