"""Core simulation and visualization utilities for MikanOS architecture study."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class PixelFormat(Enum):
    """UEFI GOP Pixel Format supported by MikanOS."""

    RGB_RESV_8BIT_PER_COLOR = "kPixelRGBResv8BitPerColor"
    BGR_RESV_8BIT_PER_COLOR = "kPixelBGRResv8BitPerColor"


def setup_plot_style() -> None:
    """Configure matplotlib style for clean, high-DPI technical figures."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["font.size"] = 10


@dataclass(frozen=True)
class PageTableAddress:
    """x86-64 4-level paging virtual address breakdown (PML4 / PDP / PD / PT / Offset)."""

    pml4_index: int
    pdp_index: int
    pd_index: int
    pt_index: int
    offset: int

    @classmethod
    def from_linear_address(cls, addr: int) -> PageTableAddress:
        """Decompose a 64-bit linear (virtual) address into 4-level paging indices."""
        if addr < 0:
            raise ValueError("Linear address must be non-negative")
        offset = addr & 0xFFF
        pt_index = (addr >> 12) & 0x1FF
        pd_index = (addr >> 21) & 0x1FF
        pdp_index = (addr >> 30) & 0x1FF
        pml4_index = (addr >> 39) & 0x1FF
        return cls(
            pml4_index=pml4_index,
            pdp_index=pdp_index,
            pd_index=pd_index,
            pt_index=pt_index,
            offset=offset,
        )

    def to_linear_address(self) -> int:
        """Reconstruct the canonical 64-bit linear address from paging indices."""
        raw = (
            ((self.pml4_index & 0x1FF) << 39)
            | ((self.pdp_index & 0x1FF) << 30)
            | ((self.pd_index & 0x1FF) << 21)
            | ((self.pt_index & 0x1FF) << 12)
            | (self.offset & 0xFFF)
        )
        # Canonical sign-extension for bit 47
        if raw & (1 << 47):
            raw |= 0xFFFF_0000_0000_0000
        return raw


class FrameBufferSimulator:
    """Simulates MikanOS UEFI GOP FrameBuffer and PixelWriter operations."""

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        pixel_format: PixelFormat = PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        # 4 bytes per pixel: [R, G, B, Reserved] in display order
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

    def write_pixel(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """Write a single pixel at (x, y) if within bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.buffer[y, x] = (r & 0xFF, g & 0xFF, b & 0xFF)

    def fill_rectangle(
        self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]
    ) -> None:
        """Fill a rectangle starting at (x, y) with size (w, h)."""
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        if x0 < x1 and y0 < y1:
            self.buffer[y0:y1, x0:x1] = color

    def draw_rectangle(
        self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]
    ) -> None:
        """Draw a 1-pixel border rectangle."""
        if w <= 0 or h <= 0:
            return
        self.fill_rectangle(x, y, w, 1, color)
        self.fill_rectangle(x, y + h - 1, w, 1, color)
        self.fill_rectangle(x, y, 1, h, color)
        self.fill_rectangle(x + w - 1, y, 1, h, color)

    def save_image(self, path: str | Path, title: str | None = None) -> Path:
        """Save the current framebuffer state to a high-resolution PNG file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.buffer)
        if title:
            ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path


class BitmapMemoryManager:
    """Simulates MikanOS BitmapMemoryManager (4 KiB page frame allocator)."""

    BYTES_PER_FRAME: int = 4096

    def __init__(self, num_frames: int = 1024) -> None:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames
        # False = free (0), True = allocated (1)
        self.alloc_map = np.zeros(num_frames, dtype=bool)
        self.range_begin = 0
        self.range_end = num_frames

    def mark_allocated(self, start_frame: int, num_frames: int) -> None:
        """Mark a contiguous range of frames as allocated."""
        if start_frame < 0 or num_frames < 0 or start_frame + num_frames > self.num_frames:
            raise ValueError("Frame range out of bounds")
        self.alloc_map[start_frame : start_frame + num_frames] = True

    def allocate(self, num_frames: int) -> int:
        """Allocate contiguous frames using first-fit search. Returns start frame ID."""
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        start = self.range_begin
        while start + num_frames <= self.range_end:
            window = self.alloc_map[start : start + num_frames]
            if not np.any(window):
                self.mark_allocated(start, num_frames)
                return start
            # Advance past the first allocated frame in the window
            first_busy = int(np.argmax(window))
            start += first_busy + 1
        raise MemoryError(f"Cannot allocate {num_frames} contiguous frames")

    def free(self, start_frame: int, num_frames: int) -> None:
        """Free a previously allocated contiguous range of frames."""
        if start_frame < 0 or num_frames <= 0 or start_frame + num_frames > self.num_frames:
            raise ValueError("Frame range out of bounds")
        self.alloc_map[start_frame : start_frame + num_frames] = False


@dataclass(frozen=True)
class OSCodebaseMetric:
    """Represents codebase size and architectural characteristics of an OS."""

    name: str
    release_year: int
    lines_of_code: int
    architecture: str
    category: str

    def lines_per_day(self, days: int = 30) -> float:
        """Compute average lines of code written per day over a given study period."""
        if days <= 0:
            raise ValueError("days must be positive")
        return self.lines_of_code / float(days)


HISTORICAL_OS_METRICS: tuple[OSCodebaseMetric, ...] = (
    OSCodebaseMetric("Haribote OS (30-day OS)", 2006, 4_249, "x86 (32-bit BIOS)", "Educational"),
    OSCodebaseMetric("Linux 0.01", 1991, 10_239, "x86 (32-bit)", "Early Monolithic"),
    OSCodebaseMetric("MikanOS (Final Kernel)", 2021, 11_071, "x86-64 (64-bit UEFI)", "Educational"),
    OSCodebaseMetric("Linux 4.10", 2017, 21_000_000, "Multi-arch", "Production Monolithic"),
)

# Measured kernel + loader file counts and LOC across official MikanOS git tags
MIKANOS_GROWTH_METRICS: tuple[tuple[str, int, int, str], ...] = (
    ("day02b", 1, 144, "EDK II Bootloader & Memory Map"),
    ("day05f", 13, 725, "PixelWriter, Font & Console printk"),
    ("day10g", 79, 7_119, "PCI, xHCI USB Mouse, IDT, Memory & Windows"),
    ("day15d", 84, 8_431, "ACPI Timer, Keyboard, Multitasking & Terminal"),
    ("day20e", 89, 9_597, "FAT Filesystem, ELF App, 4-Level Paging & Syscall"),
    ("day25c", 90, 10_417, "GUI Events, Multi-Terminal, File Read & Regex"),
    ("day30f", 92, 11_596, "Demand Paging, CoW, UTF-8/FreeType, Pipes & Apps"),
)


@dataclass(frozen=True)
class SyscallSpec:
    """Represents a system call specification in MikanOS and POSIX."""

    number: int
    name: str
    abi_register: str
    description: str
    is_posix_standard: bool


MIKANOS_SYSCALL_SPECS: tuple[SyscallSpec, ...] = (
    SyscallSpec(1, "LogString", "RDI, RSI, RDX", "Print string to kernel debug log", False),
    SyscallSpec(2, "PutString", "RDI, RSI, RDX", "Print string to terminal window", True),
    SyscallSpec(3, "Exit", "RDI", "Terminate running application", True),
    SyscallSpec(4, "OpenWindow", "RDI, RSI, RDX, R10, R8", "Create GUI window and allocate layer", False),
    SyscallSpec(5, "WinWriteString", "RDI, RSI, RDX, R10, R8", "Draw text inside window", False),
    SyscallSpec(6, "WinFillRectangle", "RDI, RSI, RDX, R10, R8, R9", "Fill rectangular area in window", False),
    SyscallSpec(7, "GetCurrentTick", "-", "Get elapsed system timer ticks", False),
    SyscallSpec(8, "WinRedraw", "RDI", "Redraw modified window buffer", False),
    SyscallSpec(9, "WinDrawLine", "RDI, RSI, RDX, R10, R8, R9", "Draw line using Bresenham algorithm", False),
    SyscallSpec(10, "CloseWindow", "RDI", "Close window and destroy layer", False),
    SyscallSpec(11, "ReadEvent", "RDI, RSI", "Block until next event (key/mouse/timer)", False),
)


class MockKernelSyscallDispatcher:
    """Simulates user-to-kernel mode switch and system call dispatching for Section 0.2."""

    def __init__(self) -> None:
        self.stdout_buffer: list[str] = []
        self.stderr_buffer: list[str] = []
        self.syscall_history: list[tuple[int, str, int]] = []  # (num, name, ret)

    def dispatch(self, syscall_num: int, arg1: int, arg2: str | bytes | int, arg3: int = 0) -> int:
        """Simulate kernel entry via SYSCALL instruction.

        Args:
            syscall_num: System call number passed in RAX.
            arg1: First argument (e.g. file descriptor or window ID).
            arg2: Second argument (e.g. buffer pointer / string content).
            arg3: Third argument (e.g. byte count).

        Returns:
            Return code returned in RAX to user space.
        """
        # MikanOS PutString (2) / POSIX write (1 = stdout, 2 = stderr)
        if syscall_num == 2:
            content = str(arg2)
            if arg1 == 1:
                self.stdout_buffer.append(content)
            elif arg1 == 2:
                self.stderr_buffer.append(content)
            else:
                self.syscall_history.append((syscall_num, "PutString", -9))  # EBADF
                return -9
            ret = len(content.encode("utf-8"))
            self.syscall_history.append((syscall_num, "PutString", ret))
            return ret
        elif syscall_num == 3:  # Exit
            self.syscall_history.append((syscall_num, "Exit", arg1))
            return 0
        else:
            self.syscall_history.append((syscall_num, "Unknown", -38))  # ENOSYS
            return -38

    def get_stdout(self) -> str:
        """Return accumulated standard output."""
        return "".join(self.stdout_buffer)


