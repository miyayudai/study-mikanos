"""Unit tests for Chapter 1 Section 1.7: Booting with UEFI BIOS.

Verifies the mathematical correctness of 16-bit real mode addressing,
A20 gate 1MB wraparound logic, MBR (512B) vs GPT (GUID Partition Table)
binary structures, C++20 comparator binary execution, and architecture figures.
"""

import os
import subprocess
import pytest
from common import (
    FirmwareType,
    CpuMode,
    PartitionScheme,
    BootComparisonFeature,
    RealModeAddress,
    MbrPartitionEntry,
    GptHeader,
    GptPartitionEntry,
    LegacyBiosVsUefiComparator,
    plot_legacy_bios_vs_uefi_boot_flow,
    plot_mbr_vs_gpt_disk_layout,
)


def test_firmware_and_cpu_enums():
    """Verify core firmware and CPU operating mode enumerations."""
    assert FirmwareType.LEGACY_BIOS.value == "Legacy BIOS"
    assert FirmwareType.UEFI.value == "UEFI"

    assert CpuMode.REAL_MODE_16.value == "16-bit Real Mode"
    assert CpuMode.PROTECTED_MODE_32.value == "32-bit Protected Mode"
    assert CpuMode.LONG_MODE_64.value == "64-bit Long Mode (IA-32e)"

    assert PartitionScheme.MBR.value == "MBR (Master Boot Record)"
    assert PartitionScheme.GPT.value == "GPT (GUID Partition Table)"


def test_real_mode_address_calculator():
    """Test 16-bit real-mode linear addressing and A20 gate wraparound."""
    # 1. Normal addressing inside 1MB (0x0000:0x7C00 -> 0x07C00)
    addr_mbr = RealModeAddress(segment=0x0000, offset=0x7C00, a20_enabled=True)
    assert addr_mbr.linear_address_raw == 0x7C00
    assert addr_mbr.effective_address == 0x7C00
    assert not addr_mbr.is_wrapped

    # 2. Reset vector (0xFFFF:0x0000 -> 0xFFFF0)
    addr_reset = RealModeAddress(segment=0xFFFF, offset=0x0000, a20_enabled=True)
    assert addr_reset.linear_address_raw == 0xFFFF0
    assert addr_reset.effective_address == 0xFFFF0
    assert not addr_reset.is_wrapped

    # 3. 1MB boundary exact wrap (0xFFFF:0x0010 -> raw 0x100000)
    # When A20 is enabled, 0x100000 is accessible.
    addr_1mb_on = RealModeAddress(segment=0xFFFF, offset=0x0010, a20_enabled=True)
    assert addr_1mb_on.linear_address_raw == 0x100000
    assert addr_1mb_on.effective_address == 0x100000
    assert not addr_1mb_on.is_wrapped

    # When A20 is disabled, bit 20 is masked -> wraps around to 0x000000!
    addr_1mb_off = RealModeAddress(segment=0xFFFF, offset=0x0010, a20_enabled=False)
    assert addr_1mb_off.linear_address_raw == 0x100000
    assert addr_1mb_off.effective_address == 0x000000
    assert addr_1mb_off.is_wrapped

    # 4. Above 1MB (0xFFFF:0x0020 -> raw 0x100010 -> wrapped 0x000010)
    addr_above_off = RealModeAddress(segment=0xFFFF, offset=0x0020, a20_enabled=False)
    assert addr_above_off.linear_address_raw == 0x100010
    assert addr_above_off.effective_address == 0x000010
    assert addr_above_off.is_wrapped


def test_mbr_partition_entry_structure():
    """Verify MBR partition table calculations and 2.0 TiB capacity wall."""
    entry = MbrPartitionEntry(
        boot_indicator=0x80,
        starting_chs=b"\x00\x02\x00",
        partition_type=0xEF,  # ESP
        ending_chs=b"\xff\xff\xff",
        starting_lba=2048,
        size_in_lba=204800,  # 100 MiB
    )
    assert entry.is_bootable is True
    assert entry.size_bytes(sector_size=512) == 104857600  # 100 MiB exactly
    assert pytest.approx(entry.size_gib(sector_size=512), rel=1e-3) == 0.09765625

    # 32-bit LBA max capacity limit: 2^32 * 512 = 2.0 TiB
    max_tib = MbrPartitionEntry.max_addressable_capacity_tib(sector_size=512)
    assert pytest.approx(max_tib, rel=1e-4) == 2.0


def test_gpt_header_and_entry_structure():
    """Verify GPT header, ESP partition recognition, and 9.4 ZB capacity."""
    header = GptHeader(
        signature=b"EFI PART",
        current_lba=1,
        backup_lba=1000000,
        first_usable_lba=34,
        last_usable_lba=999966,
        disk_guid="12345678-1234-1234-1234-123456789ABC",
    )
    assert header.signature == b"EFI PART"
    assert header.header_size == 92
    assert header.current_lba == 1
    assert header.num_partition_entries == 128
    assert header.size_partition_entry == 128

    # 64-bit LBA max capacity limit: 2^64 * 512 bytes ≈ 9.44 ZB
    max_zb = GptHeader.max_addressable_capacity_zb(sector_size=512)
    assert pytest.approx(max_zb, rel=1e-2) == 9.44

    # ESP entry
    esp_entry = GptPartitionEntry(
        type_guid=GptPartitionEntry.ESP_TYPE_GUID,
        unique_guid="AABBCCDD-EEFF-1122-3344-556677889900",
        starting_lba=2048,
        ending_lba=206847,  # 204800 sectors
        attributes=0,
        name="EFI System Partition",
    )
    assert esp_entry.is_esp is True
    assert esp_entry.sector_count == 204800
    assert esp_entry.size_bytes(sector_size=512) == 104857600
    assert pytest.approx(esp_entry.size_mib(sector_size=512), rel=1e-4) == 100.0

    # Non-ESP entry
    data_entry = GptPartitionEntry(
        type_guid=GptPartitionEntry.LINUX_DATA_GUID,
        unique_guid="11223344-5566-7788-99AA-BBCCDDEEFF00",
        starting_lba=206848,
        ending_lba=411647,
        attributes=0,
        name="MikanOS Data",
    )
    assert data_entry.is_esp is False


def test_legacy_bios_vs_uefi_comparator_features():
    """Verify architectural features and quantitative comparisons."""
    features = LegacyBiosVsUefiComparator.get_comparison_features()
    assert len(features) >= 5

    feature_names = [f.feature_name for f in features]
    assert "CPU起動時動作モード" in feature_names
    assert "アクセス可能メモリ空間" in feature_names
    assert "ブートローダサイズ制限" in feature_names
    assert "ストレージ管理 & 容量限界" in feature_names
    assert "ファイルシステムの直接認識" in feature_names
    assert "ファームウェアサービス" in feature_names


def test_cpp20_bios_uefi_comparator_executable():
    """Verify standalone C++20 comparator tool builds and executes with zero error."""
    bin_path = os.path.abspath("build/bin/ch01_bios_uefi_comparator")
    assert os.path.exists(bin_path), f"Binary not found: {bin_path}"

    res = subprocess.run([bin_path], capture_output=True, text=True, check=True)
    out = res.stdout

    assert "Chapter 1.7: Booting with UEFI BIOS" in out
    assert "sizeof(MbrSector)         : 512 bytes" in out
    assert "sizeof(GptHeader)         : 92 bytes" in out
    assert "sizeof(GptEntry)          : 128 bytes" in out
    assert "All static assertions passed successfully!" in out
    assert "0x007C00" in out
    assert "1MB Boundary Exact Wrap Test (0x100000) [WRAPPED!]" in out
    assert "2.00 TiB" in out
    assert "9.44 Zettabytes" in out
    assert "Boot Signature: 0xAA55 (VALID)" in out


def test_figure_generation_outputs():
    """Verify high-resolution architectural figures are correctly created."""
    fig13_path = "1/result/fig1_13_legacy_bios_vs_uefi_boot_flow.png"
    fig14_path = "1/result/fig1_14_mbr_vs_gpt_disk_layout.png"

    out13 = plot_legacy_bios_vs_uefi_boot_flow(fig13_path)
    out14 = plot_mbr_vs_gpt_disk_layout(fig14_path)

    assert os.path.exists(out13)
    assert os.path.exists(out14)
    assert os.path.getsize(out13) > 100_000  # At least 100KB (300 DPI high-res)
    assert os.path.getsize(out14) > 100_000
