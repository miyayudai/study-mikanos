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


# ==============================================================================
# Chapter 0 Section 0.3: Build Pipeline, Freestanding Environment & UEFI Media
# ==============================================================================

class BuildStage(Enum):
    """Phases in the bare-metal OS compilation and linking pipeline."""
    PREPROCESS = "Preprocessing (cpp)"
    COMPILE = "Compilation (clang++ / gcc)"
    ASSEMBLE = "Assembly (as / nasm)"
    LINK = "Linking (ld.lld)"


@dataclass
class CompilerFlag:
    """Represents a compiler / linker flag used in bare-metal OS development."""
    name: str
    purpose: str
    is_freestanding_required: bool
    explanation: str


FREESTANDING_FLAGS: list[CompilerFlag] = [
    CompilerFlag(
        name="-ffreestanding",
        purpose="Require freestanding environment without host standard library runtime",
        is_freestanding_required=True,
        explanation=(
            "Directs the compiler to assume that standard library headers and runtime startup "
            "(such as the host main() wrapper and glibc crt0) do not exist."
        ),
    ),
    CompilerFlag(
        name="-nostdlib",
        purpose="Do not use standard system startup files or libraries when linking",
        is_freestanding_required=True,
        explanation=(
            "Prevents linking against the host C runtime (crt0.o, libc.so, libpthread.so), "
            "which makes userland system calls to the host Linux kernel."
        ),
    ),
    CompilerFlag(
        name="-mno-red-zone",
        purpose="Disable the 128-byte red zone below the stack pointer (%rsp)",
        is_freestanding_required=True,
        explanation=(
            "The System V AMD64 ABI reserves 128 bytes below %rsp as a scratchpad known as the 'Red Zone'. "
            "In bare-metal OS kernels, asynchronous hardware interrupts will clobber this space if it is not explicitly disabled."
        ),
    ),
    CompilerFlag(
        name="-fno-exceptions",
        purpose="Disable C++ exception handling tables and runtime unwinder",
        is_freestanding_required=True,
        explanation=(
            "C++ exception handling (try/catch/throw) requires libunwind and complex personality routines, "
            "which are unavailable in freestanding kernel code."
        ),
    ),
    CompilerFlag(
        name="-fno-rtti",
        purpose="Disable Run-Time Type Information (dynamic_cast and typeid)",
        is_freestanding_required=True,
        explanation=(
            "Eliminates overhead and runtime dependency on type_info structures and vtable metadata "
            "when bare-metal kernels do not provide host C++ ABI support."
        ),
    ),
]


@dataclass
class BuildArtifact:
    """Artifact produced by a stage in the build pipeline."""
    stage: BuildStage
    filename: str
    file_format: str
    description: str
    size_bytes: int


class BuildPipelineSimulator:
    """Simulates the 4-phase transformation from C++/ASM source to bare-metal binary."""

    def __init__(self, source_lines: int = 500, is_freestanding: bool = True, disable_red_zone: bool = True) -> None:
        self.source_lines = source_lines
        self.is_freestanding = is_freestanding
        self.disable_red_zone = disable_red_zone

    def validate_flags(self, flags: list[str]) -> tuple[bool, list[str]]:
        """Validate if required freestanding flags are present."""
        errors: list[str] = []
        flag_set = set(flags)

        for required in FREESTANDING_FLAGS:
            if required.is_freestanding_required and required.name not in flag_set:
                errors.append(f"Missing required freestanding flag '{required.name}': {required.purpose}")

        return (len(errors) == 0, errors)

    def run_pipeline(self, target_type: str = "uefi_bootloader") -> list[BuildArtifact]:
        """Execute simulation of the build stages and return generated artifacts."""
        artifacts: list[BuildArtifact] = []

        # Stage 1: Preprocessing
        preprocessed_lines = self.source_lines * 4 if self.is_freestanding else self.source_lines * 50
        preprocessed_size = preprocessed_lines * 32
        artifacts.append(
            BuildArtifact(
                stage=BuildStage.PREPROCESS,
                filename="main.i",
                file_format="Preprocessed C++ Text",
                description="Macro expansion, comment stripping, and freestanding header inclusion (<stdint.h>, <stddef.h>).",
                size_bytes=preprocessed_size,
            )
        )

        # Stage 2: Compilation (AST -> LLVM IR -> x86-64 Assembly)
        asm_lines = int(self.source_lines * 2.5)
        asm_size = asm_lines * 28
        artifacts.append(
            BuildArtifact(
                stage=BuildStage.COMPILE,
                filename="main.s",
                file_format="x86-64 Assembly (AT&T or Intel syntax)",
                description=f"Clang++ compilation (-O2) with -mno-red-zone={'enabled' if self.disable_red_zone else 'disabled'}.",
                size_bytes=asm_size,
            )
        )

        # Stage 3: Assembly (Assembly -> Relocatable Machine Code Object)
        obj_size = int(asm_size * 0.4) + 1024  # ELF/COFF headers + sections + symbol table
        artifacts.append(
            BuildArtifact(
                stage=BuildStage.ASSEMBLE,
                filename="main.o",
                file_format="COFF / ELF64 Relocatable Object",
                description="Machine code opcodes, .text, .data, .rodata, .bss, and unresolved relocation symbols.",
                size_bytes=obj_size,
            )
        )

        # Stage 4: Linking
        if target_type == "uefi_bootloader":
            out_name = "BOOTX64.EFI"
            out_fmt = "PE32+ (UEFI x86-64 Executable)"
            out_desc = "Linked PE32+ application adhering to UEFI specification, entry point EfiMain/UefiMain."
            final_size = obj_size + 4096  # PE header + Data Directories
        else:
            out_name = "kernel.elf"
            out_fmt = "ELF64 Executable"
            out_desc = "Position-independent 64-bit kernel binary linked with custom linker script (KernelEntry)."
            final_size = obj_size + 8192

        artifacts.append(
            BuildArtifact(
                stage=BuildStage.LINK,
                filename=out_name,
                file_format=out_fmt,
                description=out_desc,
                size_bytes=final_size,
            )
        )

        return artifacts


@dataclass
class BootMediaFile:
    """Represents a file placed on a UEFI FAT filesystem boot partition."""
    path: str
    size_bytes: int
    file_type: str


class UEFIBootMediaValidator:
    """Validates UEFI FAT-formatted boot media directory structure and mandatory binaries."""

    REQUIRED_BOOTLOADER_PATH = "/EFI/BOOT/BOOTX64.EFI"
    KERNEL_PATH = "/kernel.elf"

    def __init__(self, filesystem_type: str = "FAT32") -> None:
        self.filesystem_type = filesystem_type.upper()
        self.files: dict[str, BootMediaFile] = {}

    def add_file(self, path: str, size_bytes: int, file_type: str) -> None:
        """Add a file to the simulated boot media layout."""
        norm_path = path.replace("\\", "/").strip()
        if not norm_path.startswith("/"):
            norm_path = "/" + norm_path
        self.files[norm_path.upper()] = BootMediaFile(path=norm_path, size_bytes=size_bytes, file_type=file_type)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate if the boot media meets UEFI specifications."""
        errors: list[str] = []

        if self.filesystem_type not in ("FAT12", "FAT16", "FAT32"):
            errors.append(f"Invalid filesystem '{self.filesystem_type}'; UEFI specification requires FAT12/FAT16/FAT32.")

        if self.REQUIRED_BOOTLOADER_PATH.upper() not in self.files:
            errors.append(f"Missing mandatory UEFI fallback bootloader at '{self.REQUIRED_BOOTLOADER_PATH}'.")
        else:
            bl = self.files[self.REQUIRED_BOOTLOADER_PATH.upper()]
            if bl.size_bytes == 0:
                errors.append(f"Bootloader '{self.REQUIRED_BOOTLOADER_PATH}' is empty (0 bytes).")
            if "PE32+" not in bl.file_type and "EFI" not in bl.file_type:
                errors.append(
                    f"Bootloader format must be PE32+ (x86-64 UEFI executable), got '{bl.file_type}'."
                )

        return (len(errors) == 0, errors)

    def get_file_list(self) -> list[BootMediaFile]:
        """Return list of files installed on media."""
        return list(self.files.values())


@dataclass
class QemuLaunchConfig:
    """Configuration generator for booting MikanOS under QEMU and OVMF UEFI firmware."""

    ovmf_path: str = "/usr/share/OVMF/OVMF_CODE.fd"
    disk_image: str = "disk.img"
    memory: str = "2G"
    cpu: str = "qemu64"
    enable_kvm: bool = False
    gdb_debug: bool = False
    gdb_port: int = 1234
    monitor: str = "stdio"

    def build_command(self) -> list[str]:
        """Construct full qemu-system-x86_64 command invocation."""
        cmd = ["qemu-system-x86_64", "-m", self.memory, "-cpu", self.cpu]
        if self.enable_kvm:
            cmd.append("-enable-kvm")
        cmd.extend(["-drive", f"if=pflash,format=raw,readonly=on,file={self.ovmf_path}"])
        cmd.extend(["-drive", f"if=ide,format=raw,file={self.disk_image}"])
        if self.monitor:
            cmd.extend(["-monitor", self.monitor])
        if self.gdb_debug:
            cmd.extend(["-gdb", f"tcp::{self.gdb_port}", "-S"])
        return cmd

    def explain_options(self) -> list[tuple[str, str]]:
        """Explain the rationale behind each critical QEMU flag."""
        return [
            ("-m 2G", "Allocates 2 Gigabytes of virtual guest RAM to match modern PC specifications."),
            ("-drive if=pflash,...", "Loads OVMF UEFI BIOS firmware into simulated parallel flash memory."),
            ("-drive if=ide,...", "Attaches FAT32 disk.img containing EFI/BOOT/BOOTX64.EFI as primary drive."),
            ("-monitor stdio", "Enables interactive QEMU monitor in terminal for CPU/MMU/register inspection."),
            ("-gdb tcp::1234 -S", "Freezes CPU at reset vector and waits for GDB remote debugger connection."),
        ]



