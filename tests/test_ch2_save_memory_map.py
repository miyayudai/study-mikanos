"""Unit tests for Chapter 2.6: Saving Memory Map to File (EFI_FILE_PROTOCOL & CSV Export)."""

import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common.os_sim import (
    EFI_FILE_MODE_READ,
    EFI_FILE_MODE_WRITE,
    EFI_FILE_MODE_CREATE,
    EFI_LOADED_IMAGE_PROTOCOL_GUID,
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID,
    MockFileProtocol,
    MockSimpleFileSystemProtocol,
    MockLoadedImageProtocol,
    MockUefiEnvironment,
    MemoryMap,
    MemoryMapCsvExporter,
    UefiBootServicesSimulator,
    UefiMemoryType,
    parse_memory_map_csv,
    plot_uefi_file_system_protocol_hierarchy,
    plot_memory_map_csv_export_and_layout,
)


def test_efi_file_mode_flags():
    """Verify UEFI file open mode flags bit patterns and bitwise combinations."""
    assert EFI_FILE_MODE_READ == 0x0000_0000_0000_0001
    assert EFI_FILE_MODE_WRITE == 0x0000_0000_0000_0002
    assert EFI_FILE_MODE_CREATE == 0x8000_0000_0000_0000

    combined = EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE
    assert combined & EFI_FILE_MODE_READ != 0
    assert combined & EFI_FILE_MODE_WRITE != 0
    assert combined & EFI_FILE_MODE_CREATE != 0


def test_mock_file_protocol_write_and_close():
    """Verify MockFileProtocol write, read, close, and error states."""
    file = MockFileProtocol(name="test.txt", mode=EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE)
    assert not file.is_closed

    # Write data
    data = "Hello, UEFI!\n"
    written = file.write(data)
    assert written == len(data)
    assert file.get_content_as_str() == data

    # Read data after rewinding cursor
    file.set_position(0)
    read_data = file.read(len(data))
    assert read_data == data.encode("utf-8")

    # Close file
    file.close()
    assert file.is_closed

    # Write after close must raise error
    with pytest.raises(RuntimeError, match="Cannot write to closed file"):
        file.write("More data\n")

    # Read after close must raise error
    with pytest.raises(RuntimeError, match="Cannot read from closed file"):
        file.read(10)


def test_mock_simple_file_system_root_dir_and_open():
    """Verify SimpleFileSystemProtocol volume opening and sub-file creation."""
    fs = MockSimpleFileSystemProtocol(volume_label="BOOT_PART")
    root = fs.open_volume()
    assert root.is_directory
    assert root.name == "\\"

    # Open nonexistent file without CREATE should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="not found in directory"):
        root.open("nonexistent.txt", mode=EFI_FILE_MODE_READ)

    # Create file with CREATE mode
    memmap_file = root.open("\\memmap", mode=EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE)
    assert memmap_file.name == "memmap"
    assert not memmap_file.is_directory
    assert not memmap_file.is_closed

    memmap_file.write("Header\n")
    assert memmap_file.get_content_as_str() == "Header\n"


def test_uefi_environment_open_protocol_chain():
    """Verify UEFI Handle Database protocol lookup and OpenRootDir chain."""
    env = MockUefiEnvironment()

    # Loaded image protocol lookup
    loaded_image = env.open_protocol(env.image_handle, EFI_LOADED_IMAGE_PROTOCOL_GUID)
    assert isinstance(loaded_image, MockLoadedImageProtocol)
    assert loaded_image.device_handle == env.device_handle

    # Simple file system protocol lookup
    fs = env.open_protocol(loaded_image.device_handle, EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID)
    assert isinstance(fs, MockSimpleFileSystemProtocol)

    # OpenRootDir convenience method
    root_dir = env.open_root_dir(env.image_handle)
    assert root_dir.is_directory
    assert root_dir.name == "\\"


def test_memory_map_csv_exporter_header_and_formatting():
    """Verify SaveMemoryMap CSV header, column order, and descriptor row formatting."""
    bs = UefiBootServicesSimulator()
    memmap = MemoryMap.allocate(16384)
    bs.get_memory_map(memmap)

    file = MockFileProtocol(name="memmap", mode=EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE)
    csv_output = MemoryMapCsvExporter.save_memory_map(bs.memory_map, file)

    assert csv_output.startswith("Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n")
    lines = csv_output.strip().splitlines()
    assert len(lines) == bs.memory_map.total_entries + 1

    # Check first row
    first_row = lines[1].split(",")
    assert len(first_row) == 6
    assert first_row[0].strip() == "0"


def test_memory_map_attribute_mask_upper_bits():
    """Verify that SaveMemoryMap applies & 0xfffff mask to descriptor attributes."""
    bs = UefiBootServicesSimulator()
    file = MockFileProtocol(name="memmap", mode=EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE)

    # Modify an entry to have upper runtime bit (Bit 63)
    desc = bs.memory_map.descriptors[0]
    desc.attribute |= 0x8000_0000_0000_0000

    csv_output = MemoryMapCsvExporter.save_memory_map(bs.memory_map, file)
    lines = csv_output.strip().splitlines()
    first_row_attr = lines[1].split(",")[5].strip()

    # Attribute string should NOT contain 8000...
    assert "8000" not in first_row_attr
    val = int(first_row_attr, 16)
    assert val <= 0xFFFFF


def test_parse_memory_map_csv_roundtrip():
    """Verify parse_memory_map_csv parses exported CSV records accurately."""
    bs = UefiBootServicesSimulator()
    file = MockFileProtocol(name="memmap", mode=EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE)
    csv_text = MemoryMapCsvExporter.save_memory_map(bs.memory_map, file)

    records = parse_memory_map_csv(csv_text)
    assert len(records) == bs.memory_map.total_entries

    total_conventional_bytes = sum(r["SizeBytes"] for r in records if r["TypeName"] == "EfiConventionalMemory")
    assert total_conventional_bytes > 0

    # Verify fields of first record
    rec0 = records[0]
    assert rec0["Index"] == 0
    assert rec0["SizeBytes"] == rec0["NumberOfPages"] * 4096
    assert rec0["PhysicalEnd"] == rec0["PhysicalStart"] + rec0["SizeBytes"]


def test_parse_memory_map_csv_invalid_header():
    """Verify parse_memory_map_csv rejects malformed CSV header."""
    invalid_csv = "Foo, Bar, Baz\n1, 2, 3\n"
    with pytest.raises(ValueError, match="Invalid memory map CSV header"):
        parse_memory_map_csv(invalid_csv)

    # Empty CSV returns empty list
    assert parse_memory_map_csv("") == []


def test_cpp20_save_memory_map_sim_standalone():
    """Verify that C++20 save_memory_map_sim compiles and runs with all assertions passing."""
    build_dir = Path("build")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build target
    cmake_build = subprocess.run(
        ["cmake", "--build", "build", "--target", "ch02_save_memory_map_sim"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert cmake_build.returncode == 0

    # Execute binary
    bin_path = build_dir / "bin" / "ch02_save_memory_map_sim"
    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert run_res.returncode == 0
    assert "All Section 2.6 C++20 verifications passed successfully!" in run_res.stdout
    assert "[PASS] TestFileProtocolOpenAndModes" in run_res.stdout
    assert "[PASS] TestSaveMemoryMapCsvFormatting" in run_res.stdout
    assert "[PASS] TestVariableDescriptorSizeStepping" in run_res.stdout


def test_figure_generation_outputs(tmp_path: Path):
    """Verify generation, resolution, and dimensions of Section 2.6 technical figures."""
    fig11_path = str(tmp_path / "fig2_11.png")
    fig12_path = str(tmp_path / "fig2_12.png")

    out11 = plot_uefi_file_system_protocol_hierarchy(fig11_path)
    out12 = plot_memory_map_csv_export_and_layout(fig12_path)

    assert Path(out11).exists()
    assert Path(out12).exists()

    with Image.open(out11) as img11:
        assert img11.width >= 1600
        assert img11.height >= 1000

    with Image.open(out12) as img12:
        assert img12.width >= 1600
        assert img12.height >= 1000
