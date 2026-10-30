"""Unit tests for Chapter 2.7: Analyzing Memory Map Output (Usable RAM, Reclaimability, Layout)."""

import subprocess
from pathlib import Path
import pytest
from PIL import Image

from common.os_sim import (
    MemoryMapAnalyzer,
    parse_memory_map_csv,
    plot_memory_map_analysis,
    plot_physical_memory_layout_map,
)


@pytest.fixture
def sample_csv() -> str:
    """Fixture providing realistic QEMU OVMF memory map CSV text."""
    return MemoryMapAnalyzer.create_sample_qemu_memmap_csv(total_ram_gib=2.0)


def test_memory_map_analyzer_from_sample_csv(sample_csv: str):
    """Verify that MemoryMapAnalyzer parses CSV records and computes accurate totals."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    assert analyzer.total_records >= 10
    assert analyzer.total_pages > 0
    assert analyzer.total_bytes == analyzer.total_pages * 4096

    # 2GiB configuration should yield around 1.8 - 2.5 GiB reported space
    assert 1.5 <= analyzer.total_gib <= 3.0
    assert analyzer.total_mib == analyzer.total_bytes / (1024 * 1024)


def test_memory_type_breakdown_metrics(sample_csv: str):
    """Verify memory type classification, count, pages, and percentage breakdown."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    breakdown = analyzer.type_breakdown()

    assert "EfiConventionalMemory" in breakdown
    assert "EfiBootServicesCode" in breakdown
    assert "EfiBootServicesData" in breakdown
    assert "EfiRuntimeServicesCode" in breakdown
    assert "EfiMemoryMappedIO" in breakdown

    total_pct = sum(item["percentage"] for item in breakdown.values())
    assert pytest.approx(total_pct, rel=1e-3) == 100.0

    total_pages_sum = sum(item["pages"] for item in breakdown.values())
    assert total_pages_sum == analyzer.total_pages


def test_conventional_memory_usable_summary(sample_csv: str):
    """Verify analysis of immediately usable EfiConventionalMemory (Type 7)."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    conv = analyzer.conventional_memory_summary()

    assert conv["record_count"] >= 2
    assert conv["pages"] > 0
    assert conv["bytes"] == conv["pages"] * 4096
    assert conv["mib"] > 1000.0  # Main RAM is substantial
    assert conv["percentage"] > 70.0

    assert conv["max_block"] is not None
    assert conv["max_block"]["PhysicalStart"] == 0x0010_0000  # Starts at 1MiB
    assert conv["max_block"]["NumberOfPages"] > 0x10000


def test_reclaimable_boot_services_and_post_boot_pool(sample_csv: str):
    """Verify calculation of boot services reclaimable memory after ExitBootServices()."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    conv = analyzer.conventional_memory_summary()
    bs = analyzer.reclaimable_boot_services_summary()
    post_boot = analyzer.available_memory_after_boot()

    assert bs["record_count"] >= 2
    assert bs["pages"] > 0
    assert bs["mib"] > 0

    assert post_boot["pages"] == conv["pages"] + bs["pages"]
    assert post_boot["bytes"] == conv["bytes"] + bs["bytes"]
    assert post_boot["mib"] == conv["mib"] + bs["mib"]
    assert post_boot["percentage"] > conv["percentage"]


def test_fragmentation_metrics_calculation(sample_csv: str):
    """Verify fragmentation index and block size distribution."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    frag = analyzer.fragmentation_metrics()

    assert frag["num_blocks"] >= 2
    assert frag["avg_block_mib"] > 0
    assert 0.0 <= frag["largest_block_ratio"] <= 1.0
    assert 0.0 <= frag["fragmentation_index"] <= 1.0
    assert pytest.approx(frag["largest_block_ratio"] + frag["fragmentation_index"], rel=1e-5) == 1.0


def test_free_regions_filter_and_kernel_placement(sample_csv: str):
    """Verify filtering of free regions sufficient to load an OS kernel."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    # Search for free region >= 16 MiB (16 * 1024 * 1024 bytes)
    min_bytes = 16 * 1024 * 1024
    regions = analyzer.find_free_regions_above(min_bytes)

    assert len(regions) >= 1
    for r in regions:
        assert r["SizeBytes"] >= min_bytes
        assert r["TypeName"] == "EfiConventionalMemory"


def test_gap_analysis_and_mmio_hole(sample_csv: str):
    """Verify address continuity checking and detection of gaps / holes."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    gaps = analyzer.gap_analysis()

    # The sample CSV has a large MMIO gap between RAM (<0x80000000) and MMIO (0x80000000, 0xFED00000)
    assert len(gaps) >= 1
    for g in gaps:
        assert g["gap_bytes"] > 0
        assert g["gap_end"] > g["gap_start"]


def test_summary_table_formatting(sample_csv: str):
    """Verify text table summary formatting."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    table_str = analyzer.summary_table()

    assert "Type (Name)" in table_str
    assert "EfiConventionalMemory" in table_str
    assert "TOTAL" in table_str
    assert "100.00%" in table_str


def test_cpp20_analyze_memory_map_sim_standalone():
    """Verify that C++20 analyze_memory_map_sim compiles and runs with all assertions passing."""
    build_dir = Path("build")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build target
    cmake_build = subprocess.run(
        ["cmake", "--build", "build", "--target", "ch02_analyze_memory_map_sim"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert cmake_build.returncode == 0

    # Execute binary
    bin_path = build_dir / "bin" / "ch02_analyze_memory_map_sim"
    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert run_res.returncode == 0
    assert "All Section 2.7 C++20 verifications passed successfully!" in run_res.stdout
    assert "[PASS] CSV parsing and numeric conversion verified." in run_res.stdout
    assert "[PASS] Usable RAM and post-boot reclaimability verified." in run_res.stdout
    assert "[PASS] Fragmentation and gap analysis verified." in run_res.stdout


def test_figure_generation_outputs(tmp_path: Path, sample_csv: str):
    """Verify generation, resolution, and dimensions of Section 2.7 technical figures."""
    analyzer = MemoryMapAnalyzer.from_csv(sample_csv)
    fig13_path = str(tmp_path / "fig2_13.png")
    fig14_path = str(tmp_path / "fig2_14.png")

    out13 = plot_memory_map_analysis(analyzer, fig13_path)
    out14 = plot_physical_memory_layout_map(analyzer, fig14_path)

    assert Path(out13).exists()
    assert Path(out14).exists()

    with Image.open(out13) as img13:
        assert img13.width >= 1600
        assert img13.height >= 1000

    with Image.open(out14) as img14:
        assert img14.width >= 1600
        assert img14.height >= 800
