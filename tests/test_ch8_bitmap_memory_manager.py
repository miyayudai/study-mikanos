"""Tests for Chapter 8.7: Implementing BitmapMemoryManager (osbook_day08c)."""

import json
import os
import subprocess

import pytest

from common import (
    NULL_FRAME_SIM,
    FrameIDSim,
    MikanBitmapMemoryManagerSim,
    plot_fig8_19_bitmap_memory_manager_architecture,
    plot_fig8_20_first_fit_allocation_and_skip_math,
    plot_fig8_21_uefi_memmap_to_bitmap_initialization,
)


def test_frame_id_and_full_spec_metrics():
    """Verify FrameIDSim strong wrapper, kNullFrame sentinel, and 128GiB footprint math."""
    f0 = FrameIDSim(0)
    f256 = FrameIDSim(256)
    assert f0.frame_address == 0x0
    assert f256.frame_address == 0x0010_0000  # 1 MiB Kernel load base
    assert NULL_FRAME_SIM.id == (1 << 64) - 1
    assert "kNullFrame" in repr(NULL_FRAME_SIM)

    with pytest.raises(ValueError):
        FrameIDSim(-1)

    metrics = MikanBitmapMemoryManagerSim.full_spec_metrics()
    assert metrics["max_physical_gib"] == 128
    assert metrics["bytes_per_frame"] == 4096
    assert metrics["total_frames"] == 33_554_432
    assert metrics["bits_per_map_line"] == 64
    assert metrics["map_line_count"] == 524_288
    assert metrics["alloc_map_bytes"] == 4_194_304
    assert metrics["alloc_map_mib"] == 4.0
    assert metrics["sizeof_bitmap_memory_manager"] == 4_194_320
    assert metrics["exceeds_1mib_stack"] is True
    assert metrics["stack_overflow_excess_bytes"] == 4_194_320 - 1_048_576
    assert pytest.approx(metrics["overhead_percentage"], rel=1e-6) == (1.0 / 32768.0) * 100.0


def test_2d_bit_coordinate_and_signed_shift_bug():
    """Verify (line_index, bit_index) decomposition and C++ 32-bit signed shift pitfall model."""
    test_frames = [0, 1, 31, 32, 63, 64, 65, 256, 33_554_431]
    for fid in test_frames:
        line_idx, bit_idx = MikanBitmapMemoryManagerSim.line_and_bit(fid)
        assert line_idx == fid // 64
        assert bit_idx == fid % 64
        reconstructed = MikanBitmapMemoryManagerSim.frame_from_line_and_bit(line_idx, bit_idx)
        assert reconstructed.id == fid
        assert (
            MikanBitmapMemoryManagerSim.physical_address_from_line_and_bit(line_idx, bit_idx)
            == fid * 4096
        )

    # Bits 0..30: 32-bit `1 << bit_index` happens to match 64-bit unsigned shift
    for b in (0, 15, 30):
        res = MikanBitmapMemoryManagerSim.simulate_signed_32bit_shift_bug(b)
        assert res["is_corrupted"] is False
        assert res["safe_mask_64"] == (1 << b)

    # Bit 31: `1 << 31` sign-extends to 0xFFFFFFFF80000000, corrupting bits 31..63!
    res31 = MikanBitmapMemoryManagerSim.simulate_signed_32bit_shift_bug(31)
    assert res31["is_corrupted"] is True
    assert res31["safe_mask_64"] == 0x0000_0000_8000_0000
    assert res31["buggy_mask_64"] == 0xFFFF_FFFF_8000_0000
    assert "SignExtensionCorruption" in res31["status"]

    # Bits 32..63: Shift >= 32 on 32-bit int is C++ Undefined Behavior
    for b in (32, 63):
        res_ub = MikanBitmapMemoryManagerSim.simulate_signed_32bit_shift_bug(b)
        assert res_ub["is_corrupted"] is True
        assert res_ub["safe_mask_64"] == (1 << b)
        assert "UndefinedBehavior" in res_ub["status"]


def test_official_osbook_day08c_7_cases():
    """Verify exact equivalence with all 7 official CppUTest cases in kernel/test/test_memory_manager.cpp."""
    # 1. Allocate
    mgr = MikanBitmapMemoryManagerSim(frame_count=256)
    f1, err1 = mgr.allocate(3)
    f2, err2 = mgr.allocate(1)
    assert err1 is None and f1.id == 0
    assert err2 is None and f2.id == 3
    assert mgr.alloc_map[0] == 0b1111

    # 2. AllocateMultipleLine (crosses 64-bit MapLineType boundary)
    mgr = MikanBitmapMemoryManagerSim(frame_count=256)
    f1, err1 = mgr.allocate(64 + 3)  # 67 frames -> line 0 full + bits 0..2 of line 1
    f2, err2 = mgr.allocate(1)
    assert err1 is None and f1.id == 0
    assert err2 is None and f2.id == 67
    assert mgr.alloc_map[0] == 0xFFFF_FFFF_FFFF_FFFF
    assert mgr.alloc_map[1] == 0b1111

    # 3. AllocateNoEnoughMemory
    mgr = MikanBitmapMemoryManagerSim(frame_count=128)
    f_oom, err_oom = mgr.allocate(129)
    assert f_oom == NULL_FRAME_SIM
    assert err_oom == "kNoEnoughMemory"

    # 4. Free
    mgr = MikanBitmapMemoryManagerSim(frame_count=128)
    f1, _ = mgr.allocate(3)
    assert mgr.free(f1, 3) is None
    f2, err2 = mgr.allocate(2)
    assert err2 is None and f2.id == 0

    # 5. FreeDifferentFrames
    mgr = MikanBitmapMemoryManagerSim(frame_count=128)
    f1, _ = mgr.allocate(3)  # 0..2
    f2, _ = mgr.allocate(2)  # 3..4
    f3, _ = mgr.allocate(4)  # 5..8
    assert f1.id == 0 and f2.id == 3 and f3.id == 5
    mgr.free(f2, 2)  # Free 3..4 (only 2 frames free there)
    f4, _ = mgr.allocate(3)  # Needs 3 contiguous -> skips 3..4 and lands at 9!
    f5, _ = mgr.allocate(2)  # Needs 2 contiguous -> fits into freed 3..4!
    assert f4.id == 9
    assert f5.id == 3

    # 6. MarkAllocated
    mgr = MikanBitmapMemoryManagerSim(frame_count=128)
    mgr.mark_allocated(FrameIDSim(5), 5)  # Frames 5..9 occupied
    fa, _ = mgr.allocate(6)  # Cannot fit in 0..4 (only 5 frames), must start at 10!
    fb, _ = mgr.allocate(5)  # Fits cleanly in 0..4!
    assert fa.id == 10
    assert fb.id == 0

    # 7. SetMemoryRange
    mgr = MikanBitmapMemoryManagerSim(frame_count=128)
    mgr.set_memory_range(FrameIDSim(10), FrameIDSim(64))
    fr1, err_r1 = mgr.allocate(54)
    fr2, err_r2 = mgr.allocate(1)
    assert err_r1 is None and fr1.id == 10
    assert fr2 == NULL_FRAME_SIM and err_r2 == "kNoEnoughMemory"


def test_first_fit_jump_optimization_and_trace():
    """Verify O(N) start_frame_id += i + 1 skip requires strictly fewer inspections than naive +1."""
    sim = MikanBitmapMemoryManagerSim(frame_count=512)
    # Create 8 blocked windows of length 16 (15 free frames followed by 1 busy frame at 15, 31, ..., 127)
    for blk in range(1, 9):
        sim.mark_allocated(FrameIDSim(blk * 16 - 1), 1)

    res = sim.allocate_with_trace(16)
    assert res["error"] is None
    assert res["frame"].id == 128
    # Jump method checks each frame in 0..143 exactly once: 144 checks!
    assert res["jump_checks"] == 144
    # Naive method does dramatically more redundant checks across every window
    assert res["naive_checks"] > res["jump_checks"] * 5
    assert len(res["probes"]) == 9
    assert all(p["skip_distance"] == 16 for p in res["probes"][:-1])


def test_external_fragmentation_and_auto_coalescing():
    """Verify external fragmentation causes kNoEnoughMemory and Free() auto-coalesces adjacent runs."""
    sim = MikanBitmapMemoryManagerSim(frame_count=64)
    sim.set_memory_range(FrameIDSim(0), FrameIDSim(12))

    fa, _ = sim.allocate(3)  # 0..2
    fb, _ = sim.allocate(2)  # 3..4
    fc, _ = sim.allocate(3)  # 5..7
    fd, _ = sim.allocate(2)  # 8..9

    sim.free(fa, 3)
    sim.free(fc, 3)

    m1 = sim.fragmentation_metrics()
    assert m1["free_frames"] == 8
    assert m1["free_fragments"] == 3
    assert m1["largest_contiguous_free"] == 3
    assert pytest.approx(m1["external_fragmentation_ratio"]) == 1.0 - (3.0 / 8.0)

    # 8 free frames total, yet Allocate(5) must fail due to external fragmentation
    f_fail, err = sim.allocate(5)
    assert f_fail == NULL_FRAME_SIM
    assert err == "kNoEnoughMemory"

    # Freeing fb (3..4) automatically merges [0..2] + [3..4] + [5..7] -> [0..7] (8 contiguous free frames)
    sim.free(fb, 2)
    m2 = sim.fragmentation_metrics()
    assert m2["free_frames"] == 10
    assert m2["free_fragments"] == 2
    assert m2["largest_contiguous_free"] == 8

    f_ok, err2 = sim.allocate(5)
    assert err2 is None
    assert f_ok.id == 0
    assert fd.id == 8


def test_uefi_memory_map_initialization_invariants():
    """Verify hole protection, EfiLoaderData kernel protection, FrameID{1} start, and MMIO non-inflation."""
    descriptors = [
        (3, 0x0000_0000, 1),     # Frame 0: BootServicesCode (excluded by range_begin=1)
        (7, 0x0000_1000, 159),   # Frames 1..159: Conventional RAM
        # Unlisted hole: Frames 160..255 (0xA0000..0x100000 VGA/ROM)
        (2, 0x0010_0000, 272),   # Frames 256..527: EfiLoaderData (MikanOS Kernel)
        (7, 0x0021_0000, 1520),  # Frames 528..2047: Conventional RAM
        (6, 0x0080_0000, 32),    # Frames 2048..2079: RuntimeServicesData
        (4, 0x0082_0000, 2016),  # Frames 2080..4095: BootServicesData (reclaimed)
        # Unlisted gap + trailing MMIO
        (11, 0x0140_0000, 16),   # Frames 5120..5135: MemoryMappedIO
    ]

    sim = MikanBitmapMemoryManagerSim(frame_count=8192)
    summary = sim.initialize_from_uefi_descriptors(descriptors)

    assert summary["range_begin"] == 1
    assert summary["range_end"] == 4096
    assert summary["available_end"] == 0x0100_0000
    assert (0x000A_0000, 0x0010_0000) in summary["unlisted_holes"]
    assert summary["allocated_frames"] == 96 + 272 + 32
    assert summary["free_frames"] == (4096 - 1) - 400

    # Verify VGA hole and Kernel ELF frames are all marked 1 (Allocated)
    assert all(sim.get_bit(FrameIDSim(fid)) for fid in range(160, 528))
    # Verify RuntimeServices frames are marked 1
    assert all(sim.get_bit(FrameIDSim(fid)) for fid in range(2048, 2080))
    # Verify First-Fit starts at FrameID{1} (0x1000 != nullptr)
    first_frame, err = sim.allocate(1)
    assert err is None
    assert first_frame.id == 1
    assert first_frame.frame_address == 0x1000


def test_invalid_inputs_and_edge_cases():
    """Verify defensive validation on invalid constructor/method arguments."""
    with pytest.raises(ValueError):
        MikanBitmapMemoryManagerSim(frame_count=0)
    with pytest.raises(ValueError):
        MikanBitmapMemoryManagerSim(frame_count=100)  # Not a multiple of 64

    sim = MikanBitmapMemoryManagerSim(frame_count=64)
    with pytest.raises(ValueError):
        sim.allocate(0)
    with pytest.raises(IndexError):
        sim.get_bit(FrameIDSim(64))
    with pytest.raises(ValueError):
        sim.set_memory_range(FrameIDSim(32), FrameIDSim(16))


def test_figure_generation_outputs(tmp_path):
    """Verify generation of Figures 8.19, 8.20, and 8.21 at high DPI."""
    f19 = plot_fig8_19_bitmap_memory_manager_architecture(str(tmp_path / "fig8_19.png"))
    f20 = plot_fig8_20_first_fit_allocation_and_skip_math(str(tmp_path / "fig8_20.png"))
    f21 = plot_fig8_21_uefi_memmap_to_bitmap_initialization(str(tmp_path / "fig8_21.png"))

    for path in (f19, f20, f21):
        assert os.path.exists(path)
        assert os.path.getsize(path) > 50_000

    for canonical in (
        "8/result/fig8_19_bitmap_memory_manager_architecture.png",
        "8/result/fig8_20_first_fit_allocation_and_skip_math.png",
        "8/result/fig8_21_uefi_memmap_to_bitmap_initialization.png",
    ):
        assert os.path.exists(canonical), f"Missing saved figure {canonical}"
        assert os.path.getsize(canonical) > 50_000


def test_cpp20_bitmap_memory_manager_sim_standalone():
    """Verify the standalone C++20 BitmapMemoryManager simulator binary passes all unit tests."""
    bin_path = os.path.abspath("build/bin/ch08_bitmap_memory_manager_sim")
    if not os.path.exists(bin_path):
        subprocess.run(
            ["cmake", "--build", "build", "--target", "ch08_bitmap_memory_manager_sim"],
            check=True,
        )

    result = subprocess.run([bin_path, "--test"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "All C++20 BitmapMemoryManager Verification Unit Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Section 8.7 notebook exists and contains executed cells with zero errors."""
    nb_path = "8/8.7_Implementing_BitmapMemoryManager.ipynb"
    assert os.path.exists(nb_path), f"Notebook file {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, "Notebook should have at least 10 rich content cells"

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) >= 4

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} was not executed"
            outputs = cell.get("outputs", [])
            assert len(outputs) > 0, f"Code cell {idx} produced no output"
            for out in outputs:
                assert out.get("output_type") != "error", f"Error found in cell {idx}: {out}"
