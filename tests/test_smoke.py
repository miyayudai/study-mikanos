"""Smoke and unit tests for common MikanOS simulation utilities."""

from pathlib import Path
import pytest

from common import (
    BitmapMemoryManager,
    FrameBufferSimulator,
    PageTableAddress,
    PixelFormat,
)


def test_smoke() -> None:
    """Repository initialization smoke test."""
    assert True


def test_page_table_address_canonical_higher_half() -> None:
    """Verify 4-level paging decomposition and reconstruction for MikanOS app space."""
    # MikanOS application base address from Chapter 19: 0xffff8000ffff0000
    app_base = 0xFFFF_8000_FFFF_0000
    parsed = PageTableAddress.from_linear_address(app_base)
    assert parsed.pml4_index == 256
    assert parsed.pdp_index == 3
    assert parsed.pd_index == 511
    assert parsed.pt_index == 496
    assert parsed.offset == 0
    assert parsed.to_linear_address() == app_base


def test_page_table_address_lower_half() -> None:
    """Verify decomposition of lower-half kernel identity-mapped addresses."""
    kernel_addr = 0x0000_0000_0010_6008
    parsed = PageTableAddress.from_linear_address(kernel_addr)
    assert parsed.pml4_index == 0
    assert parsed.pdp_index == 0
    assert parsed.pd_index == 0
    assert parsed.pt_index == 0x106
    assert parsed.offset == 0x008
    assert parsed.to_linear_address() == kernel_addr


def test_bitmap_memory_manager_first_fit() -> None:
    """Verify first-fit allocation and deallocation of 4KiB page frames."""
    mgr = BitmapMemoryManager(num_frames=64)
    mgr.mark_allocated(0, 2)  # Reserve frames 0..1

    f1 = mgr.allocate(4)
    assert f1 == 2

    f2 = mgr.allocate(3)
    assert f2 == 6

    # Free the 4 frames at index 2..5 and re-allocate 3 frames -> should reuse index 2
    mgr.free(f1, 4)
    f3 = mgr.allocate(3)
    assert f3 == 2


def test_bitmap_memory_manager_out_of_memory() -> None:
    """Verify MemoryError is raised when contiguous frames are unavailable."""
    mgr = BitmapMemoryManager(num_frames=8)
    mgr.allocate(6)
    with pytest.raises(MemoryError):
        mgr.allocate(4)


def test_framebuffer_simulator_draw_and_save(tmp_path: Path) -> None:
    """Verify pixel drawing, rectangle filling, and image export."""
    fb = FrameBufferSimulator(
        width=64, height=48, pixel_format=PixelFormat.BGR_RESV_8BIT_PER_COLOR
    )
    fb.fill_rectangle(10, 10, 20, 15, (255, 128, 0))
    fb.write_pixel(5, 5, 10, 20, 30)

    assert tuple(fb.buffer[5, 5]) == (10, 20, 30)
    assert tuple(fb.buffer[15, 15]) == (255, 128, 0)
    assert tuple(fb.buffer[0, 0]) == (0, 0, 0)

    out_file = fb.save_image(tmp_path / "test_fb.png", title="Test FrameBuffer")
    assert out_file.exists()
    assert out_file.stat().st_size > 0
