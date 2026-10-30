"""Core simulation and visualization utilities for MikanOS architecture study."""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, ArrowStyle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
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
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans", "sans-serif"]



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


class DebugStrategy(Enum):
    """Bare-metal debugging techniques used before printf is available."""

    FRAMEBUFFER_COLOR = "framebuffer_color"  # Filling screen/VRAM with a solid color to verify execution
    HALT_LOOP = "halt_loop"                  # Inserting while(1) or hlt / cli; hlt to freeze CPU at checkpoint
    SERIAL_PORT = "serial_port"              # Writing ASCII characters directly to UART 0x3F8 (COM1)
    QEMU_MONITOR = "qemu_monitor"            # Dumping registers and physical memory via QEMU monitor / GDB
    BINARY_SEARCH = "binary_search"          # Pinpointing freeze/crash lines using binary search instrumentation


class DevelopmentMethod(Enum):
    """Software development philosophies in operating system construction."""

    BIG_BANG = "big_bang"      # Writing thousands of lines before first boot (prone to untraceable bugs)
    SMALL_STEP = "small_step"  # Incremental addition of small verifiable features (MikanOS principle)


@dataclass(frozen=True)
class SmallStepPhase:
    """A single verifiable milestone in the incremental OS development journey."""

    day: str
    chapter: int
    title: str
    added_lines: int
    primary_feature: str
    verification_method: str


MIKANOS_KEY_MILESTONES: list[SmallStepPhase] = [
    SmallStepPhase("day01a", 1, "最小UEFIバイナリ", 4, "UEFI BIOSからの直接ロード・即座に正常終了", "QEMU画面が一瞬開いて終了コード0で戻る"),
    SmallStepPhase("day02a", 2, "EDK II Hello World", 18, "SystemTable->ConOut を用いた文字列出力", "UEFI画面に Hello, world! が描画される"),
    SmallStepPhase("day02b", 2, "メモリマップ取得", 54, "GetMemoryMap による物理メモリ構成の把握とファイル出力", "memmap ファイルにメモリディスクリプタCSVが保存される"),
    SmallStepPhase("day03a", 3, "最小C++カーネル呼び出し", 32, "ブートローダからELFカーネルへの関数ポインタジャンプ", "カーネル側無限ループでCPUが意図通り停止する"),
    SmallStepPhase("day03c", 3, "ピクセル描画", 48, "GOP フレームバッファの物理アドレスへ直書き", "画面全体が鮮やかな白色/黄色に塗りつぶされる"),
    SmallStepPhase("day04b", 4, "長方形・幾何学模様", 85, "座標計算とピクセル描画ループの抽象化", "画面中央にカラフルな長方形が整然と描画される"),
    SmallStepPhase("day05e", 5, "コンソールクラス", 130, "フォント描画・改行処理・画面スクロールの実装", "カーネルログがターミナル風に下方向へスクロールする"),
    SmallStepPhase("day06c", 6, "マウスカーソル描画", 240, "PCI探索・xHCIホストコントローラからのHIDマウスパケット取得", "マウスを動かすと矢印カーソルが滑らかに追従する"),
    SmallStepPhase("day07b", 7, "IDT割り込みとリングバッファ", 190, "ポーリング廃止、MSI割り込み＋ArrayQueueによる高速処理", "マウスが遅延ゼロで瞬時に反応しCPU負荷が激減する"),
    SmallStepPhase("day08c", 8, "ビットマップ物理メモリ管理", 160, "4KiBフレーム単位の First-Fit アロケータ", "カーネルデータ構造と動的ヒープ領域が安全に確保される"),
    SmallStepPhase("day09e", 9, "ウィンドウ重ね合わせ処理", 220, "LayerManager による透過色付き複数ウィンドウ合成", "マウスをウィンドウの上に乗せても背景が壊れず重なる"),
    SmallStepPhase("day13d", 13, "マルチタスクとスケジューラ", 280, "コンテキストスイッチと時分割マルチタスク", "複数のカウンタタスクが並行して数字をカウントアップする"),
    SmallStepPhase("day17b", 17, "FAT32ファイルシステム", 310, "BPB・ディレクトリエントリ解析と ls コマンド", "ターミナルで ls と打つとディスク内のファイル一覧が出る"),
    SmallStepPhase("day18b", 18, "ELFアプリケーション実行", 260, "ディスク上の別体実行ファイルを読み込んで実行", "ユーザーランドの hello アプリがターミナル上で動く"),
    SmallStepPhase("day20e", 20, "システムコール (Ring 0/3)", 340, "SYSCALL/SYSRET命令と特権保護・CPU例外ハンドラ", "ユーザー権限のアプリが安全にOS機能を呼び出して文字表示する"),
]


class SmallStepProgressTracker:
    """Analyzer comparing incremental small-step OS construction against big-bang development."""

    def __init__(self, milestones: list[SmallStepPhase] | None = None) -> None:
        self.milestones = milestones if milestones is not None else list(MIKANOS_KEY_MILESTONES)

    def total_milestones(self) -> int:
        """Return total number of tracked milestones."""
        return len(self.milestones)

    def total_cumulative_lines(self) -> int:
        """Return cumulative added lines across all milestones."""
        return sum(m.added_lines for m in self.milestones)

    def calculate_debugging_cost(self, total_lines: int, method: DevelopmentMethod) -> dict[str, float]:
        """Estimate bug-hunting cost (expected steps to locate a bug).

        In Big-Bang development, finding a bug among N lines with no runtime feedback
        requires inspecting the entire codebase or linearly debugging: O(N) worst-case.
        In Small-Step development with K checkpoints (each adding n = N/K lines),
        the search space is immediately constrained to the latest delta n: O(n) = O(N/K).
        """
        if method == DevelopmentMethod.BIG_BANG:
            return {
                "worst_case_search_lines": float(total_lines),
                "expected_inspection_steps": total_lines / 2.0,
                "feedback_delay_days": 30.0,
            }
        else:
            k = max(1, len(self.milestones))
            avg_delta = total_lines / k
            return {
                "worst_case_search_lines": float(avg_delta),
                "expected_inspection_steps": avg_delta / 2.0,
                "feedback_delay_days": 0.1,
            }


@dataclass
class DebugInspectionStep:
    """Record of a single probe during binary search debugging."""

    step_number: int
    low_line: int
    high_line: int
    probe_line: int
    outcome: str  # "PASS" (reached probe line successfully) or "CRASH" (hung before reaching)


class BareMetalDebugSimulator:
    """Simulates bare-metal debugging strategies when printf/standard output are unavailable."""

    def __init__(self, total_lines: int = 100, bug_line: int = 42) -> None:
        if not (1 <= bug_line <= total_lines):
            raise ValueError(f"bug_line must be between 1 and {total_lines}, got {bug_line}")
        self.total_lines = total_lines
        self.bug_line = bug_line
        self.vram_color: tuple[int, int, int] = (0, 0, 0)
        self.vram_history: list[tuple[int, tuple[int, int, int]]] = []
        self.serial_buffer: list[str] = []

    def set_vram_color(self, checkpoint_line: int, rgb: tuple[int, int, int]) -> bool:
        """Simulate setting framebuffer color at a code checkpoint.

        Returns True if the checkpoint was reached before the bug line, False if crashed before.
        """
        if checkpoint_line < self.bug_line:
            self.vram_color = rgb
            self.vram_history.append((checkpoint_line, rgb))
            return True
        else:
            return False

    def write_serial(self, checkpoint_line: int, char: str) -> bool:
        """Simulate outputting a character to UART serial port 0x3F8."""
        if checkpoint_line < self.bug_line:
            self.serial_buffer.append(char)
            return True
        return False

    def locate_bug_by_binary_search(self) -> list[DebugInspectionStep]:
        """Simulate pinpointing the bug line using binary search halting instrumentation.

        Places a freeze / marker (e.g. while(1) or color change) at mid = (low + high) // 2.
        If reached, bug is after mid (low = mid + 1). If crashed, bug is at or before mid (high = mid).
        """
        low = 1
        high = self.total_lines
        steps: list[DebugInspectionStep] = []
        step_idx = 1

        while low < high:
            mid = (low + high) // 2
            if mid < self.bug_line:
                outcome = "PASS"
                steps.append(DebugInspectionStep(step_idx, low, high, mid, outcome))
                low = mid + 1
            else:
                outcome = "CRASH"
                steps.append(DebugInspectionStep(step_idx, low, high, mid, outcome))
                high = mid
            step_idx += 1

        return steps


@dataclass(frozen=True)
class HardwareEnvironmentFeature:
    """Comparison metric between QEMU emulator and real bare-metal hardware."""

    category: str
    qemu_ovmf: str
    real_hardware: str
    impact_on_os_dev: str


class QemuVsHardwareComparator:
    """Structured evaluation of emulator-based vs bare-metal OS development."""

    @staticmethod
    def get_comparison_table() -> list[HardwareEnvironmentFeature]:
        """Return detailed comparison matrix between QEMU and real hardware."""
        return [
            HardwareEnvironmentFeature(
                category="起動・イテレーション時間",
                qemu_ovmf="約 1〜2 秒（即座に再起動可能）",
                real_hardware="約 30〜60 秒（PC再起動・USB認識待ち）",
                impact_on_os_dev="QEMUは1時間に数十回の試行錯誤が可能。実機直結は試行回数が激減する。",
            ),
            HardwareEnvironmentFeature(
                category="デバッグ容易性",
                qemu_ovmf="QEMUモニタ、GDB接続、レジスタ・メモリ完全可視化",
                real_hardware="画面が真っ黒になると原因究明が極めて困難",
                impact_on_os_dev="開発初期（カーネル呼び出し・ページング・割り込み）はQEMUが絶対的に有利。",
            ),
            HardwareEnvironmentFeature(
                category="安全性",
                qemu_ovmf="どれだけ不正ポインタを破壊してもホストOSは無傷",
                real_hardware="ストレージの誤書き込みやマザーボードの再起動ループリスク",
                impact_on_os_dev="初心者が安全に実験・学習するための防護壁となる。",
            ),
            HardwareEnvironmentFeature(
                category="ハードウェア互換性の検証",
                qemu_ovmf="仕様に忠実な仮想デバイス（理想的・標準的挙動）",
                real_hardware="メーカー固有のUEFI実装差、xHCIコントローラの癖",
                impact_on_os_dev="QEMUで動くOSを実機に持っていくことで、真のハードウェア制御の奥深さを学べる。",
            ),
            HardwareEnvironmentFeature(
                category="達成感・学習体験",
                qemu_ovmf="「画面上で動いた」という達成感",
                real_hardware="「手元の物理マシンを自分が支配した」という圧倒的感動",
                impact_on_os_dev="QEMUで動作を固めたあと、節目（章末）で実機ブートするのが最高の学習リズム。",
            ),
        ]


@dataclass(frozen=True)
class TechStackFeature:
    """Evaluation metric for architectural technology selection."""

    technology: str
    category: str
    dimension: str
    score: int  # 1 to 5
    advantages: str
    disadvantages: str
    mikanos_rationale: str


@dataclass(frozen=True)
class ArchitecturePhase:
    """One of the 7 evolutionary architectural phases of MikanOS."""

    phase_id: int
    name: str
    chapter_range: str
    start_chapter: int
    end_chapter: int
    key_milestones: list[str]
    core_data_structures: list[str]
    hardware_targets: list[str]


@dataclass(frozen=True)
class ArchitectureModule:
    """Core kernel module and its architectural layer/dependency."""

    module_name: str
    chapter: int
    layer: str  # "HAL", "KernelCore", "Subsystem", "Userland"
    dependencies: list[str]
    description: str


class MikanOSArchitectureRoadmap:
    """Roadmap model covering all 30 chapters and 7 evolutionary phases."""

    @staticmethod
    def get_phases() -> list[ArchitecturePhase]:
        """Return the 7 structural phases of MikanOS development."""
        return [
            ArchitecturePhase(
                phase_id=1,
                name="ブートとグラフィックス基盤",
                chapter_range="第1章 〜 第5章",
                start_chapter=1,
                end_chapter=5,
                key_milestones=[
                    "UEFI EDK II ブートローダ (MikanLoaderPkg)",
                    "GOP (Graphics Output Protocol) フレームバッファ取得",
                    "PixelWriter による点・矩形描画ポリモーフィズム",
                    "Hankaku ビットマップフォント展開と Console クラス",
                    "Newlib 連携と printk 書式付き文字列出力",
                ],
                core_data_structures=["PixelWriter", "RGB/BGR PixelFormat", "Console", "FrameBufferConfig"],
                hardware_targets=["UEFI BIOS", "Linear FrameBuffer (GOP)", "x86-64 CPU"],
            ),
            ArchitecturePhase(
                phase_id=2,
                name="入力デバイスと割り込み機構",
                chapter_range="第6章 〜 第7章",
                start_chapter=6,
                end_chapter=7,
                key_milestones=[
                    "PCI コンフィギュレーション空間の体系的探索",
                    "xHCI (USB 3.0) ホストコントローラの発見と初期化",
                    "IDT (割り込み記述子表) 登録と割り込みハンドラ",
                    "MSI (Message Signaled Interrupts) によるハードウェア通知",
                    "ArrayQueue (リングバッファ/FIFO) によるイベント非同期配送",
                ],
                core_data_structures=["IDTEntry", "InterruptDescriptorTable", "ArrayQueue", "PCIConfigAddress"],
                hardware_targets=["PCI Express Bus", "xHCI Controller", "USB Mouse", "CPU Interrupt Line"],
            ),
            ArchitecturePhase(
                phase_id=3,
                name="メモリ管理とGUIレイヤ重ね合わせ",
                chapter_range="第8章 〜 第10章",
                start_chapter=8,
                end_chapter=10,
                key_milestones=[
                    "GDT (Global Descriptor Table) セグメント初期化",
                    "BitmapMemoryManager による 4KiB 物理フレーム管理",
                    "new / delete 演算子オーバーロードとカーネルヒープ",
                    "Layer および LayerManager による前後関係（Z-order）管理",
                    "ShadowBuffer (ダブルバッファ) によるチラつきのないウィンドウ描画",
                ],
                core_data_structures=["BitmapMemoryManager", "SegmentDescriptor", "Layer", "LayerManager", "Window"],
                hardware_targets=["Physical RAM", "GDT Register (GDTR)", "CR0/CR3 Registers"],
            ),
            ArchitecturePhase(
                phase_id=4,
                name="時間計測・キーボードとマルチタスク",
                chapter_range="第11章 〜 第14章",
                start_chapter=11,
                end_chapter=14,
                key_milestones=[
                    "Local APIC タイマ割り込みと ACPI PM タイマ較正",
                    "TimerManager (優先度付きキュー) によるミリ秒タイマ",
                    "USB HID キーボードドライバとモディファイアキー解析",
                    "TaskContext と SwitchContext アセンブリによる協調的タスク切り替え",
                    "マルチレベル優先度付きランキューと HLT アイドルタスク",
                ],
                core_data_structures=["TaskContext", "Task", "TaskManager", "Timer", "TimerManager"],
                hardware_targets=["Local APIC Timer", "ACPI FADT/PM Timer", "USB Keyboard"],
            ),
            ArchitecturePhase(
                phase_id=5,
                name="ターミナル・コマンドとファイルシステム",
                chapter_range="第15章 〜 第18章",
                start_chapter=15,
                end_chapter=18,
                key_milestones=[
                    "Terminal タスクの独立スレッド化とアクティブ入力制御",
                    "キーイベントディスパッチと echo / clear / lspci コマンド",
                    "FAT32 BPB (BIOS Parameter Block) とディレクトリエントリ解析",
                    "クラスタチェーン追跡と FAT ファイル読み込み (ls コマンド)",
                    "ELF アプリケーションバイナリのロードと実行",
                ],
                core_data_structures=["Terminal", "FAT32Header", "DirectoryEntry", "Elf64_Ehdr", "Elf64_Phdr"],
                hardware_targets=["FAT32 Storage (USB/Virtual Disk)", "ELF Binary Image"],
            ),
            ArchitecturePhase(
                phase_id=6,
                name="4階層ページングとシステムコール保護",
                chapter_range="第19章 〜 第23章",
                start_chapter=19,
                end_chapter=23,
                key_milestones=[
                    "48ビット正規アドレスと 4階層ページテーブル (PML4/PDP/PD/PT)",
                    "カーネル空間とアプリケーション空間の仮想メモリ分離",
                    "Ring 0 (Kernel) と Ring 3 (User) の特権レベル分離",
                    "TSS (Task State Segment) と IST (Interrupt Stack Table)",
                    "SYSCALL / SYSRET 命令による低オーバーヘッドシステムコール",
                    "GUI アプリケーション API (ウィンドウ描画・イベント配送)",
                ],
                core_data_structures=["PML4Table", "PDPTable", "PageDirectory", "TSS64", "SyscallArgs"],
                hardware_targets=["x86-64 MMU (CR3)", "IA32_EFER / STAR / LSTAR / SFMASK MSRs"],
            ),
            ArchitecturePhase(
                phase_id=7,
                name="実用OSへの洗練・IPCとアプリケーション",
                chapter_range="第24章 〜 第30章",
                start_chapter=24,
                end_chapter=30,
                key_milestones=[
                    "複数ターミナルの同時起動と PML4 仮想空間の独立化",
                    "ファイル書き込み (FAT クラスタ割り当て) と標準入力抽象化",
                    "デマンドページング (Demand Paging) と Copy-on-Write (CoW)",
                    "UTF-8 デコードと FreeType による日本語フォント描画",
                    "パイプ (Pipe) と標準入出力リダイレクトによる IPC",
                    "実用アプリ群 (テキストビューア tview, 画像ビューア gview, blocks)",
                ],
                core_data_structures=["FileDescriptor", "PipeDescriptor", "DemandPageManager", "FreeTypeFont"],
                hardware_targets=["Page Fault (#PF Exception)", "Unicode Fonts", "Multiple App Spaces"],
            ),
        ]

    @staticmethod
    def get_core_modules() -> list[ArchitectureModule]:
        """Return major architectural kernel modules and their layer hierarchy."""
        return [
            ArchitectureModule("PixelWriter", 4, "HAL", [], "GOPフレームバッファへのピクセル描画抽象化"),
            ArchitectureModule("Console", 5, "KernelCore", ["PixelWriter"], "画面文字描画・スクロール・printk基盤"),
            ArchitectureModule("PCIManager", 6, "HAL", [], "PCIバス探索・デバイス特定・BAR空間読み取り"),
            ArchitectureModule("xHCIDriver", 6, "HAL", ["PCIManager"], "USB 3.0 コントローラ初期化とデバイス認識"),
            ArchitectureModule("IDTManager", 7, "KernelCore", [], "割り込み記述子表登録・CPU例外ハンドラ設定"),
            ArchitectureModule("ArrayQueue", 7, "KernelCore", [], "リングバッファ型スレッドセーフ非同期キュー"),
            ArchitectureModule("BitmapMemoryManager", 8, "KernelCore", [], "4KiB物理ページフレーム追跡と割り当て"),
            ArchitectureModule("LayerManager", 9, "Subsystem", ["PixelWriter"], "複数ウィンドウ・カーソルのZ-order重ね合わせ合成"),
            ArchitectureModule("TimerManager", 11, "KernelCore", ["IDTManager"], "Local APICタイマ割り込みとコールバック管理"),
            ArchitectureModule("TaskManager", 13, "KernelCore", ["TimerManager"], "コンテキストスイッチと優先度スケジューラ"),
            ArchitectureModule("Terminal", 15, "Subsystem", ["Console", "TaskManager", "LayerManager"], "独立ターミナルウィンドウとコマンド実行"),
            ArchitectureModule("FATFileSystem", 17, "Subsystem", [], "FAT32ファイルシステム解析・ファイル読み書き"),
            ArchitectureModule("ELFLoader", 18, "Subsystem", ["FATFileSystem"], "ELF64バイナリ検証・セグメント配置と実行"),
            ArchitectureModule("PagingManager", 19, "KernelCore", ["BitmapMemoryManager"], "4階層ページテーブル構成と仮想アドレス保護"),
            ArchitectureModule("SyscallHandler", 20, "KernelCore", ["TaskManager", "PagingManager"], "SYSCALL/SYSRETによるRing 0/3特権遷移"),
            ArchitectureModule("AppWindowManager", 21, "Subsystem", ["LayerManager", "SyscallHandler"], "ユーザーアプリ向けGUI API提供"),
            ArchitectureModule("PipeManager", 29, "Subsystem", ["TaskManager"], "プロセス間通信(IPC)用パイプバッファ"),
        ]

    @staticmethod
    def calculate_module_dependencies() -> dict[str, list[str]]:
        """Map each module name to its dependent modules."""
        modules = MikanOSArchitectureRoadmap.get_core_modules()
        return {m.module_name: m.dependencies for m in modules}


class TechStackComparison:
    """Comprehensive comparison of firmware, CPU architecture, and language choices."""

    @staticmethod
    def get_firmware_comparison() -> list[TechStackFeature]:
        """Comparison between UEFI BIOS and Legacy BIOS (MBR)."""
        return [
            TechStackFeature(
                technology="UEFI (EDK II)",
                category="Firmware",
                dimension="起動時のCPUモード",
                score=5,
                advantages="電源投入直後から64ビットロングモードで動作。16ビットリアルモードの制約がない。",
                disadvantages="仕様書が数千ページと巨大で、EDK II のビルドシステムが複雑。",
                mikanos_rationale="現代のPCハードウェアに合致し、面倒なリアルモード→プロテクトモード遷移パズルを回避できる。",
            ),
            TechStackFeature(
                technology="Legacy BIOS",
                category="Firmware",
                dimension="起動時のCPUモード",
                score=2,
                advantages="極めてシンプルで歴史的資料が多い。",
                disadvantages="16ビットリアルモードから起動し、A20ゲート解除やGDT手動設定など歴史的負債の処理が必須。",
                mikanos_rationale="本質的でない歴史的パズルに時間を浪費するため、現代OS入門としては非推奨。",
            ),
            TechStackFeature(
                technology="UEFI (EDK II)",
                category="Firmware",
                dimension="グラフィックス・ファイルアクセス",
                score=5,
                advantages="GOPによる高解像度フレームバッファとFATファイルシステムドライバが標準装備。",
                disadvantages="ファームウェア依存のAPIであり、ExitBootServices後は自前実装が必要。",
                mikanos_rationale="ブートローダ段階でカーネルELFファイルをFATから直接読み込め、初期から美麗GUIを構築可能。",
            ),
            TechStackFeature(
                technology="Legacy BIOS",
                category="Firmware",
                dimension="グラフィックス・ファイルアクセス",
                score=2,
                advantages="INT 10h / INT 13h などBIOS割り込みが利用可能。",
                disadvantages="VGA 320x200 256色モードにとどまり、ディスクはCHSセクタ単位の手動読み込みが必須。",
                mikanos_rationale="512バイトのMBR制限があり、複雑なカーネルを読み込むためのステージングが難解。",
            ),
        ]

    @staticmethod
    def get_architecture_comparison() -> list[TechStackFeature]:
        """Comparison between x86-64, 32-bit x86, and ARM64."""
        return [
            TechStackFeature(
                technology="x86-64 (AMD64)",
                category="Architecture",
                dimension="メモリ空間とページング",
                score=5,
                advantages="48ビット正規仮想アドレス、4階層ページング、フラットメモリモデル。",
                disadvantages="ページテーブルが4階層あり、データ構造の初期化ステップが多い。",
                mikanos_rationale="現代のPCで標準的に動作し、洗練された仮想メモリ保護を完全体験できる。",
            ),
            TechStackFeature(
                technology="32-bit x86 (IA-32)",
                category="Architecture",
                dimension="メモリ空間とページング",
                score=3,
                advantages="2階層ページング（10-10-12分割）で直感的・シンプル。",
                disadvantages="4GiBメモリの壁、PAEなどの拡張が煩雑、現代のハードウェアで直接動かしにくい。",
                mikanos_rationale="過去の自作OS本では定番だったが、現代の実機で動かすには時代遅れ。",
            ),
            TechStackFeature(
                technology="x86-64 (AMD64)",
                category="Architecture",
                dimension="システムコール高速化",
                score=5,
                advantages="SYSCALL / SYSRET 専用命令によりIDT割り込みゲート不要でRing 0/3遷移が可能。",
                disadvantages="MSR (Model Specific Register) の厳密な設定が必要。",
                mikanos_rationale="最新のCPU設計に基づき、低オーバーヘッドな現代的システムコールを学べる。",
            ),
        ]

    @staticmethod
    def get_language_comparison() -> list[TechStackFeature]:
        """Comparison between C++ (C++17/20), Pure C, and Rust."""
        return [
            TechStackFeature(
                technology="C++ (C++17/20)",
                category="Language",
                dimension="抽象化能力とハードウェア制御",
                score=5,
                advantages="クラス、多態性(vtable)、テンプレートによるゼロコスト抽象化とポインタ直接制御を両立。",
                disadvantages="例外やRTTI、重厚なlibstdc++に依存できないため、フリースタンディング制約が必要。",
                mikanos_rationale="低レイヤのハードウェアレジスタ制御と、GUIやタスク管理のオブジェクト指向設計が完璧に調和する。",
            ),
            TechStackFeature(
                technology="C言語 (C99/C11)",
                category="Language",
                dimension="抽象化能力とハードウェア制御",
                score=3,
                advantages="言語仕様が極小で、出力される機械語が最も予測しやすい。",
                disadvantages="オブジェクト指向や多態性を手動の関数ポインタ構造体で模倣する必要があり、コードが肥大化・脆弱化。",
                mikanos_rationale="ウィンドウやレイヤ管理が巨大化するにつれて手動vtableコードが破綻しやすい。",
            ),
            TechStackFeature(
                technology="Rust",
                category="Language",
                dimension="抽象化能力とハードウェア制御",
                score=4,
                advantages="所有権システムによるメモリ安全性、強力な型推論とマクロ。",
                disadvantages="ベアメタルのMMIOや割り込み・ページングなど `unsafe` が頻発し、学習曲線が急峻。",
                mikanos_rationale="OS入門としての教育的見通しにおいては、C++の方がハードウェアの真の姿を直接直感しやすい。",
            ),
        ]

    @staticmethod
    def get_radar_metrics() -> dict[str, dict[str, int]]:
        """Return multidimensional scores (1-5 scale) for radar chart visualization."""
        return {
            "UEFI (EDK II)": {
                "64bit親和性": 5,
                "GUIフレームバッファ": 5,
                "ファイルアクセス容易度": 5,
                "仕様のシンプルさ": 2,
                "実機互換性": 5,
            },
            "Legacy BIOS": {
                "64bit親和性": 1,
                "GUIフレームバッファ": 2,
                "ファイルアクセス容易度": 2,
                "仕様のシンプルさ": 4,
                "実機互換性": 2,
            },
            "C++ (C++17/20)": {
                "ハードウェア直結度": 5,
                "オブジェクト指向抽象度": 5,
                "機械語透過性": 4,
                "GUIモデリング容易度": 5,
                "学習導入の平易さ": 4,
            },
            "Pure C (C99)": {
                "ハードウェア直結度": 5,
                "オブジェクト指向抽象度": 2,
                "機械語透過性": 5,
                "GUIモデリング容易度": 2,
                "学習導入の平易さ": 4,
            },
        }


# ==============================================================================
# Chapter 0 Exercises: Data Structures and Verification Helpers
# ==============================================================================

@dataclass(frozen=True)
class QuizQuestion:
    """A multiple-choice quiz question for Chapter 0 self-assessment."""

    question_id: int
    section: str
    topic: str
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class Chapter0Quiz:
    """Comprehensive multiple-choice quiz engine covering Sections 0.1 to 0.5."""

    def __init__(self) -> None:
        self.questions: list[QuizQuestion] = [
            QuizQuestion(
                question_id=1,
                section="0.1",
                topic="OS開発規模と成長曲線",
                question="名著『ゼロからのOS自作入門』で開発するMikanOSの最終段階（day30f）における総コード行数（LoC）の規模感として最も適切なものはどれか？",
                options=[
                    "約 1,000 行（最小限のシェルのみ）",
                    "約 15,000〜18,000 行（GUI、USB、マルチタスク、アプリ実行を備えた完全自作OS）",
                    "約 100,000 行（Linux 0.99 同等の規模）",
                    "約 1,000,000 行以上（モダン汎用OSと同等）",
                ],
                correct_index=1,
                explanation="MikanOSは付録まで含めて約1.5万〜1.8万行の規模であり、個人が一冊を通じて全行を把握しながら読破・改造できる絶妙なサイズに設計されています（Linux 0.01は約1万行、最新Linuxは3000万行以上）。",
            ),
            QuizQuestion(
                question_id=2,
                section="0.2",
                topic="特権レベルとハードウェア保護",
                question="x86-64 アーキテクチャにおいて、OSカーネルが Ring 0 で動作し、ユーザーアプリケーションが Ring 3 で動作する主目的は何か？",
                options=[
                    "Ring 0 では浮動小数点数演算が高速化されるため",
                    "アプリケーションの不具合や悪意あるメモリアクセスからカーネルと他プロセスをハードウェア機構で保護するため",
                    "Ring 3 ではコンパイラの最適化が無効化されるため",
                    "64ビットLong ModeではすべてのプログラムをRing 0で動かすことが禁止されているため",
                ],
                correct_index=1,
                explanation="CPUの特権リング（Privilege Ring）機構は、ハードウェアレベルでI/Oポート、ページテーブル、制御レジスタ（CR0/CR3等）への直接アクセスをRing 0（スーパバイザモード）に制限し、ユーザーアプリ（Ring 3）の暴走からシステム全体を保護します。",
            ),
            QuizQuestion(
                question_id=3,
                section="0.2",
                topic="System V AMD64 ABI とシステムコール",
                question="x86-64 System V AMD64 ABI において、`syscall` 命令発行時にカーネルに渡すシステムコール番号および第1〜第3引数を格納するレジスタの正しい組み合わせはどれか？",
                options=[
                    "システムコール番号: eax / 第1引数: ebx / 第2引数: ecx / 第3引数: edx",
                    "システムコール番号: rax / 第1引数: rdi / 第2引数: rsi / 第3引数: rdx",
                    "システムコール番号: rdx / 第1引数: rax / 第2引数: rbx / 第3引数: rcx",
                    "システムコール番号: r8  / 第1引数: r9  / 第2引数: r10 / 第3引数: r11",
                ],
                correct_index=1,
                explanation="System V AMD64 ABI では、システムコール番号を `rax` に格納し、引数は順に `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9` を用いて渡します（通常のC関数呼び出しでの第4引数 `rcx` がシステムコールでは `r10` に置き換わる点にも注意）。",
            ),
            QuizQuestion(
                question_id=4,
                section="0.3",
                topic="ビルドパイプライン",
                question="C++ソースコード（`.cpp`）から実行可能バイナリ（`.elf` や `.efi`）が生成される4つのステージの正しい順序はどれか？",
                options=[
                    "コンパイル → アセンブル → プリプロセス → リンク",
                    "プリプロセス → コンパイル → アセンブル → リンク",
                    "リンク → プリプロセス → コンパイル → アセンブル",
                    "プリプロセス → リンク → コンパイル → アセンブル",
                ],
                correct_index=1,
                explanation="ビルドパイプラインは「①プリプロセス（マクロ展開・#include展開）→ ②コンパイル（C++構文解析・最適化・アセンブリコード生成）→ ③アセンブル（機械語バイナリ・再配置可能オブジェクト生成）→ ④リンク（シンボル解決・セクション結合）」の順序で進行します。",
            ),
            QuizQuestion(
                question_id=5,
                section="0.3",
                topic="Freestanding 環境",
                question="OSカーネルをコンパイルする際、`-ffreestanding` や `-nostdlib` を指定する理由として最も正しいものはどれか？",
                options=[
                    "実行可能ファイルのサイズを最小化してコンパイル時間を短縮するため",
                    "ホストOSのC/C++標準ライブラリ（printfやmallocなどOSが提供するAPI）への依存を断ち、ベアメタル上で動作可能にするため",
                    "すべての関数呼び出しをインライン展開させるため",
                    "C++の仮想関数（vtable）の生成を抑止するため",
                ],
                correct_index=1,
                explanation="OSカーネル自身がシステムコールやメモリ管理を提供する基盤であるため、ホストOS（LinuxやWindows）を前提とした標準C/C++ライブラリ（Hosted環境）は使えません。`-ffreestanding` と `-nostdlib` によって裸のハードウェアで動作するバイナリを生成します。",
            ),
            QuizQuestion(
                question_id=6,
                section="0.3",
                topic="UEFIブートメディア構造",
                question="UEFIファームウェアが起動時にFAT32フォーマットのUSBメモリから自動検出して実行する、x86-64向けの標準ブートローダファイルの配置パスはどれか？",
                options=[
                    "`/boot/vmlinuz`",
                    "`/EFI/BOOT/BOOTX64.EFI`",
                    "`/System/Library/CoreServices/boot.efi`",
                    "`/EFI/MICROSOFT/BOOT/BCD`",
                ],
                correct_index=1,
                explanation="UEFI仕様書において、デフォルトの自動起動ファイルはFATファイルシステムの `\\EFI\\BOOT\\BOOTX64.EFI`（x86-64の場合）と定められています。MikanOSのブートローダもこのファイル名で配置されます。",
            ),
            QuizQuestion(
                question_id=7,
                section="0.4",
                topic="ベアメタルデバッグ手法",
                question="画面描画クラスやシリアル出力がまだ未完成の極初期カーネルにおいて、コード内の特定地点までCPUが到達したかを視覚的に確認する古典的かつ確実な手法はどれか？",
                options=[
                    "カーネル内から GDB のブレークポイントを動的に注入する",
                    "VRAM（フレームバッファ）の先頭バイトに直接特定の色データを書き込み、画面全体または特定ピクセルを着色する",
                    "CPUのクロック周波数を計測してループ回数を推定する",
                    "例外ハンドラを登録してゼロ除算例外を故意に発生させる",
                ],
                correct_index=1,
                explanation="VRAMの物理アドレス（フレームバッファ）はブートローダから渡されており、ポインタキャストして `*(vram_ptr) = 0xFF;` などと書くだけで画面に色が付きます。文字列表示が不可能な極初期の生存確認として最強の武器となります。",
            ),
            QuizQuestion(
                question_id=8,
                section="0.4",
                topic="二分探索デバッグ戦略",
                question="1,000 行の初期化コードのどこかでカーネルがトリプルフォールト（CPUハング）を起こしている。VRAM着色マーカーを用いた二分探索デバッグを行う場合、原因箇所を特定するために必要な最大プローブ回数は理論的におよそ何回か？",
                options=[
                    "約 1,000 回（線形探索）",
                    "約 500 回",
                    "約 10 回（ceil(log2(1000)) = 10）",
                    "約 100 回",
                ],
                correct_index=2,
                explanation="二分探索（Binary Search）では、毎回候補区間を半分に絞り込めるため、計算量は O(log2 N) です。1,000行であれば log2(1000) ≈ 9.97 より、わずか10回のプローブで原因行を特定できます。",
            ),
            QuizQuestion(
                question_id=9,
                section="0.5",
                topic="UEFI vs レガシーBIOS",
                question="MikanOSがレガシーBIOSではなくUEFI（EDK II）を採用した最大の技術的メリットとして、誤っているものはどれか？",
                options=[
                    "起動時点でCPUが64ビットLong Mode（または32ビット保護モード）で動作しており、16ビットリアルモードの制約がない",
                    "GOP（Graphics Output Protocol）により、起動直後から高解像度・フルカラーのフレームバッファ情報が取得できる",
                    "EDK IIを採用することで、Linuxカーネルの全ドライバがそのまま再コンパイルなしで動作する",
                    "FATファイルシステムドライバがファームウェアに内蔵されており、カーネルファイルのロードが容易である",
                ],
                correct_index=2,
                explanation="Linuxカーネルドライバがそのまま動くわけではありません。UEFIの利点は「64ビットモードで起動」「GOPによる高解像度フレームバッファ」「FATファイルアクセス」「メモリマップ取得」をブートローダに提供してくれる点にあります。",
            ),
            QuizQuestion(
                question_id=10,
                section="0.5",
                topic="C++20採用の必然性",
                question="MikanOSがカーネル開発言語としてC言語ではなくC++（C++17/20）を採用した最大のアーキテクチャ上の理由はどれか？",
                options=[
                    "C++のガベージコレクションによってメモリリークが完全に防止できるため",
                    "ウィンドウ・レイヤ・タイマ・タスクなどのGUI/OS構成要素をクラス・カプセル化・インターフェース多態性で明快にモデル化できるため",
                    "C++の例外処理（try-catch）がベアメタルのハードウェア割り込み処理に必須であるため",
                    "C++コードはC言語コードよりも常に機械語命令数が少なくなるため",
                ],
                correct_index=1,
                explanation="MikanOSはGUIを備えたOSであり、PixelWriter多態性、LayerManager、TimerManager、Taskスケジューラなど、オブジェクト指向の恩恵を最大限に活用しています。なおベアメタルC++では例外（try-catch）やRTTIは無効化（-fno-exceptions -fno-rtti）して運用します。",
            ),
        ]

    def evaluate(self, answers: dict[int, int]) -> dict[str, any]:
        """Evaluate user answers and return comprehensive scores, explanations, and mastery breakdown."""
        details = []
        correct_count = 0
        section_scores: dict[str, dict[str, int]] = {
            "0.1": {"correct": 0, "total": 0},
            "0.2": {"correct": 0, "total": 0},
            "0.3": {"correct": 0, "total": 0},
            "0.4": {"correct": 0, "total": 0},
            "0.5": {"correct": 0, "total": 0},
        }

        for q in self.questions:
            user_ans = answers.get(q.question_id, -1)
            is_correct = (user_ans == q.correct_index)
            if is_correct:
                correct_count += 1
                section_scores[q.section]["correct"] += 1
            section_scores[q.section]["total"] += 1

            details.append({
                "question_id": q.question_id,
                "section": q.section,
                "topic": q.topic,
                "question": q.question,
                "user_answer": user_ans,
                "correct_answer": q.correct_index,
                "is_correct": is_correct,
                "explanation": q.explanation,
            })

        total = len(self.questions)
        percentage = (correct_count / total) * 100.0 if total > 0 else 0.0

        mastery_rates: dict[str, float] = {}
        for sec, stat in section_scores.items():
            rate = (stat["correct"] / stat["total"]) * 100.0 if stat["total"] > 0 else 0.0
            mastery_rates[sec] = rate

        return {
            "total_questions": total,
            "correct_count": correct_count,
            "percentage": percentage,
            "details": details,
            "mastery_rates": mastery_rates,
            "passed": percentage >= 80.0,
        }


class TheoryExerciseValidator:
    """Validator for fill-in-the-blank theory exercises."""

    @staticmethod
    def validate_os_roles(user_roles: list[str]) -> tuple[bool, str]:
        """Validate the two primary roles of an Operating System."""
        cleaned = [r.strip().lower() for r in user_roles]
        has_resource = any("リソース" in r or "資源" in r or "resource" in r for r in cleaned)
        has_abstract = any("抽象" in r or "abstract" in r or "仮想" in r for r in cleaned)
        if has_resource and has_abstract:
            return True, "正解！OSの2大役割は「ハードウェアの抽象化（仮想化）」と「リソース管理（調停）」です。"
        missing = []
        if not has_resource:
            missing.append("リソース管理（CPU・メモリ・I/O調停）")
        if not has_abstract:
            missing.append("ハードウェアの抽象化（デバイス非依存API提供）")
        return False, f"不正解です。不足している観点: {', '.join(missing)}"

    @staticmethod
    def validate_syscall_registers(reg_map: dict[str, str]) -> tuple[bool, str]:
        """Validate x86-64 System V AMD64 ABI syscall register conventions."""
        expected = {
            "syscall_number": "rax",
            "arg1": "rdi",
            "arg2": "rsi",
            "arg3": "rdx",
            "arg4": "r10",
        }
        errors = []
        for k, exp in expected.items():
            user_val = reg_map.get(k, "").strip().lower()
            if user_val != exp:
                errors.append(f"{k}: 期待値 '{exp}', 入力値 '{user_val}'")
        if not errors:
            return True, "正解！System V AMD64 ABI のシステムコールレジスタ規約（rax, rdi, rsi, rdx, r10）に完全準拠しています。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_freestanding_flags(flags: list[str]) -> tuple[bool, str]:
        """Validate essential compiler flags for freestanding OS kernel builds."""
        flag_set = {f.strip() for f in flags}
        required = {"-ffreestanding", "-nostdlib"}
        missing = required - flag_set
        if not missing:
            return True, "正解！-ffreestanding と -nostdlib により、ホストOS依存を遮断したベアメタルバイナリが生成されます。"
        return False, f"不正解です。OSカーネルビルドに必須のフラグが不足しています: {missing}"


def plot_chapter0_exercises_summary(output_path: Path | str) -> Path:
    """Generate high-DPI summary diagram showing Chapter 0 exercise structure and learning taxonomy."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#FAFAFA")

    # Categories and exercises
    exercises = [
        {"id": "Ex 1", "sec": "0.1 / 0.2", "title": "理論・穴埋め検証", "desc": "OSの二大役割、特権Ring分離、System V ABIレジスタ規約", "color": "#1E88E5", "y": 0.82},
        {"id": "Ex 2", "sec": "0.3", "title": "ビルド & UEFI配置", "desc": "Freestandingフラグ判定、FAT32 EFIブートパス検証ロジック", "color": "#43A047", "y": 0.64},
        {"id": "Ex 3", "sec": "0.4", "title": "二分探索デバッグ", "desc": "ベアメタル初期化クラッシュ行をO(log N)で特定するアルゴリズム", "color": "#FB8C00", "y": 0.46},
        {"id": "Ex 4", "sec": "0.5", "title": "C++20 カーネル検証", "desc": "システムコールディスパッチャ & アーキテクチャ依存検証コード", "color": "#8E24AA", "y": 0.28},
        {"id": "Ex 5", "sec": "0.1-0.5", "title": "第0章 総合テスト", "desc": "全節網羅の10問自己採点クイズ & レーダーチャート可視化", "color": "#D81B60", "y": 0.10},
    ]

    import matplotlib.patches as patches

    for ex in exercises:
        y = ex["y"]
        # Outer box
        rect = patches.FancyBboxPatch(
            (0.05, y - 0.06), 0.90, 0.12,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="white", edgecolor=ex["color"], linewidth=2, zorder=2
        )
        ax.add_patch(rect)

        # Badge
        badge = patches.FancyBboxPatch(
            (0.07, y - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=ex["color"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(0.13, y, ex["id"], color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

        # Text
        ax.text(0.22, y + 0.02, f"[{ex['sec']}] {ex['title']}", color="#212121", fontsize=12, fontweight="bold", va="center")
        ax.text(0.22, y - 0.025, ex["desc"], color="#555555", fontsize=9.5, va="center")

        # Status check circle
        circle = plt.Circle((0.91, y), 0.025, facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(0.91, y, "✓", color="#2E7D32", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.0, 0.95)
    ax.axis("off")
    ax.set_title("MikanOS Study — Chapter 0 Exercises Curriculum & Assessment Map",
                 fontsize=14, fontweight="bold", pad=20, color="#1A237E")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_chapter0_mastery_radar(output_path: Path | str, mastery_rates: dict[str, float] | None = None) -> Path:
    """Generate high-DPI 5-axis radar chart showing Chapter 0 concept mastery."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        "0.1 OS規模・開発思想\n(Code Scale & Growth)",
        "0.2 POSIX・特権モード\n(POSIX & Ring Separation)",
        "0.3 ビルド & UEFI\n(Build Pipeline & UEFI)",
        "0.4 デバッグ戦略\n(Bare-metal Debugging)",
        "0.5 アーキテクチャ\n(Architecture Roadmap)",
    ]
    num_vars = len(categories)

    if mastery_rates is None:
        mastery_rates = {"0.1": 100.0, "0.2": 100.0, "0.3": 100.0, "0.4": 100.0, "0.5": 100.0}

    values = [
        mastery_rates.get("0.1", 100.0),
        mastery_rates.get("0.2", 100.0),
        mastery_rates.get("0.3", 100.0),
        mastery_rates.get("0.4", 100.0),
        mastery_rates.get("0.5", 100.0),
    ]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFC")

    # Draw grid
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#757575", size=9)
    plt.ylim(0, 110)

    # Category labels
    plt.xticks(angles[:-1], categories, color="#1A237E", size=10, fontweight="bold")

    # Plot user scores
    ax.plot(angles, values, color="#1E88E5", linewidth=2.5, linestyle="solid", label="習得度スコア (Mastery Rate)")
    ax.fill(angles, values, color="#42A5F5", alpha=0.35)

    # Reference benchmark (80% pass line)
    benchmark = [80] * (num_vars + 1)
    ax.plot(angles, benchmark, color="#E53935", linewidth=1.5, linestyle="--", label="合格基準 (80% Benchmark)")

    plt.title("Chapter 0 Comprehension & Skills Mastery Radar\n（第0章 概念理解度・実践スキル評価）",
              size=13, fontweight="bold", color="#0D47A1", y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 1.1: PE32+ (UEFI Executable) Binary Parser & Hardware Simulation
# ==============================================================================


@dataclass(frozen=True)
class PE32PlusSection:
    """Represents a COFF Section in a PE32+ binary."""

    name: str
    virtual_size: int
    virtual_address: int
    raw_data_size: int
    raw_data_offset: int
    characteristics: int

    @property
    def is_executable(self) -> bool:
        return bool(self.characteristics & 0x20000000)

    @property
    def is_readable(self) -> bool:
        return bool(self.characteristics & 0x40000000)

    @property
    def is_writable(self) -> bool:
        return bool(self.characteristics & 0x80000000)


@dataclass(frozen=True)
class PE32PlusHeader:
    """Parsed headers of a 64-bit PE32+ executable (UEFI application)."""

    dos_magic: int
    pe_header_offset: int
    machine: int
    num_sections: int
    time_date_stamp: int
    size_of_optional_header: int
    characteristics: int
    magic: int
    entry_point_rva: int
    image_base: int
    section_alignment: int
    file_alignment: int
    subsystem: int
    size_of_image: int
    size_of_headers: int
    sections: list[PE32PlusSection]

    @property
    def is_pe32plus(self) -> bool:
        """Magic 0x020B indicates PE32+ (64-bit), whereas 0x010B indicates PE32 (32-bit)."""
        return self.magic == 0x020B

    @property
    def is_x86_64(self) -> bool:
        """Machine 0x8664 corresponds to IMAGE_FILE_MACHINE_AMD64."""
        return self.machine == 0x8664

    @property
    def is_efi_application(self) -> bool:
        """Subsystem 0x000A corresponds to IMAGE_SUBSYSTEM_EFI_APPLICATION."""
        return self.subsystem == 0x000A


class PE32PlusParser:
    """Parser and validator for UEFI PE32+ executables (e.g., BOOTX64.EFI)."""

    @classmethod
    def parse_bytes(cls, data: bytes) -> PE32PlusHeader:
        if len(data) < 0x200:
            raise ValueError(f"Binary too small for PE32+ headers: {len(data)} bytes (minimum 512 bytes required)")

        # 1. DOS Header
        dos_magic = struct.unpack_from("<H", data, 0x00)[0]
        if dos_magic != 0x5A4D:  # 'MZ'
            raise ValueError(f"Invalid DOS Magic: {hex(dos_magic)} (expected 0x5a4d 'MZ')")

        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
        if pe_offset + 0x18 > len(data):
            raise ValueError(f"PE header offset 0x{pe_offset:x} points outside data range")

        # 2. PE Signature
        pe_sig = struct.unpack_from("<I", data, pe_offset)[0]
        if pe_sig != 0x00004550:  # 'PE\\0\\0'
            raise ValueError(f"Invalid PE Signature: {hex(pe_sig)} (expected 0x4550 'PE\\0\\0')")

        # 3. COFF File Header (20 bytes)
        coff_offset = pe_offset + 4
        machine, num_sections, time_date_stamp, _, _, size_of_opt_hdr, characteristics = struct.unpack_from(
            "<HHIIIHH", data, coff_offset
        )

        # 4. Optional Header (PE32+ 64-bit)
        opt_offset = coff_offset + 20
        magic = struct.unpack_from("<H", data, opt_offset)[0]
        if magic != 0x020B:
            raise ValueError(f"Invalid Optional Header Magic: {hex(magic)} (expected 0x20b for PE32+)")

        entry_point_rva = struct.unpack_from("<I", data, opt_offset + 16)[0]
        image_base = struct.unpack_from("<Q", data, opt_offset + 24)[0]
        section_alignment = struct.unpack_from("<I", data, opt_offset + 32)[0]
        file_alignment = struct.unpack_from("<I", data, opt_offset + 36)[0]
        size_of_image = struct.unpack_from("<I", data, opt_offset + 56)[0]
        size_of_headers = struct.unpack_from("<I", data, opt_offset + 60)[0]
        subsystem = struct.unpack_from("<H", data, opt_offset + 68)[0]

        # 5. Section Table (40 bytes per section)
        sec_table_offset = opt_offset + size_of_opt_hdr
        sections: list[PE32PlusSection] = []
        for i in range(num_sections):
            cur = sec_table_offset + i * 40
            sec_name_raw = data[cur : cur + 8].split(b"\x00")[0]
            sec_name = sec_name_raw.decode("ascii", errors="replace")
            vsize, vaddr, raw_size, raw_ptr, _, _, _, _, chars = struct.unpack_from(
                "<IIIIIIHHI", data, cur + 8
            )
            sections.append(
                PE32PlusSection(
                    name=sec_name,
                    virtual_size=vsize,
                    virtual_address=vaddr,
                    raw_data_size=raw_size,
                    raw_data_offset=raw_ptr,
                    characteristics=chars,
                )
            )

        return PE32PlusHeader(
            dos_magic=dos_magic,
            pe_header_offset=pe_offset,
            machine=machine,
            num_sections=num_sections,
            time_date_stamp=time_date_stamp,
            size_of_optional_header=size_of_opt_hdr,
            characteristics=characteristics,
            magic=magic,
            entry_point_rva=entry_point_rva,
            image_base=image_base,
            section_alignment=section_alignment,
            file_alignment=file_alignment,
            subsystem=subsystem,
            size_of_image=size_of_image,
            size_of_headers=size_of_headers,
            sections=sections,
        )

    @classmethod
    def parse_file(cls, path: str | Path) -> PE32PlusHeader:
        with open(path, "rb") as f:
            data = f.read()
        return cls.parse_bytes(data)

    @classmethod
    def disassemble_hello_instructions(cls, data: bytes, code_offset: int = 0x200) -> list[dict]:
        """Disassemble the minimal x86-64 machine code instructions in hello.efi."""
        instructions = [
            {
                "offset": code_offset + 0,
                "bytes": "48 83 ec 28",
                "mnemonic": "sub rsp, 0x28",
                "explanation": "シャドウスペース(32B)+8B整列パディング確保 (16Bスタック整列維持)",
            },
            {
                "offset": code_offset + 4,
                "bytes": "48 8b 4a 40",
                "mnemonic": "mov rcx, [rdx + 0x40]",
                "explanation": "SystemTable(rdx)のオフセット64(0x40)からConOutポインタをrcxに取得",
            },
            {
                "offset": code_offset + 8,
                "bytes": "48 8d 15 f1 0f 00 00",
                "mnemonic": "lea rdx, [rip + 0x0ff1]",
                "explanation": "RIP相対で.rdataのUTF-16文字列「Hello, world!」アドレスを第2引数rdxに設定",
            },
            {
                "offset": code_offset + 15,
                "bytes": "ff 51 08",
                "mnemonic": "call [rcx + 8]",
                "explanation": "ConOutプロトコル構造体のオフセット8(OutputString関数ポインタ)を間接呼び出し",
            },
            {
                "offset": code_offset + 18,
                "bytes": "eb fe",
                "mnemonic": "jmp -2 (.fin)",
                "explanation": "自命令への相対ジャンプによるCPU無限ループ（ハングアップ待機）",
            },
        ]
        return instructions

    @classmethod
    def decode_utf16_string(cls, data: bytes, offset: int = 0x400) -> str:
        """Decode null-terminated UTF-16LE string from binary at specified offset."""
        end = offset
        while end + 1 < len(data):
            if data[end] == 0 and data[end + 1] == 0:
                break
            end += 2
        return data[offset:end].decode("utf-16-le")


class UefiCallingConventionSim:
    """Simulation of Microsoft x64 Calling Convention under UEFI execution."""

    def __init__(self, initial_rsp: int = 0x00007FFF_FFFF0008):
        # Entry convention: RSP % 16 == 8 upon function entry (due to 8-byte ret address pushed)
        self.rsp = initial_rsp
        self.registers: dict[str, int] = {
            "RCX": 0x00000000_A0000000,  # ImageHandle
            "RDX": 0x00000000_B0000000,  # SystemTable
            "R8": 0x0,
            "R9": 0x0,
            "RIP": 0x00000001_40001000,  # Entry point RVA 0x1000
        }
        self.stack: dict[int, int] = {}
        self.log: list[str] = []

    def execute_sub_rsp(self, amount: int = 0x28) -> None:
        self.rsp -= amount
        is_aligned = (self.rsp % 16) == 0
        self.log.append(f"sub rsp, {hex(amount)} -> RSP={hex(self.rsp)} (16B Aligned: {is_aligned})")

    def execute_mov_rcx_from_system_table(self, con_out_ptr: int = 0x00000000_C0000000) -> None:
        # SystemTable + 0x40 -> ConOut
        self.registers["RCX"] = con_out_ptr
        self.log.append(f"mov rcx, [rdx + 0x40] -> RCX={hex(con_out_ptr)} (ConOut Protocol)")

    def execute_lea_rdx_string(self, string_ptr: int = 0x00000001_40002000) -> None:
        self.registers["RDX"] = string_ptr
        self.log.append(f"lea rdx, [rip + offset] -> RDX={hex(string_ptr)} (UTF-16 String)")

    def execute_call_output_string(self) -> None:
        # Pushes 8-byte return address, stack becomes RSP - 8
        self.log.append(
            f"call [rcx + 8] -> Invoking OutputString(This={hex(self.registers['RCX'])}, String={hex(self.registers['RDX'])})"
        )


def plot_pe32plus_structure(pe: PE32PlusHeader, output_path: str | Path) -> Path:
    """Generate high-resolution architecture diagram of PE32+ binary layout (fig1_1)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.1, 1.3]})

    # Left: Binary File vs Memory Map
    y_coords = [0, 1, 2, 3, 4, 5]
    heights = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    labels = [
        "DOS Header (0x0000 - 0x003F)\nMagic: 'MZ' (0x5A4D), e_lfanew = 0x80",
        "DOS Stub / Padding (0x0040 - 0x007F)\nUnused zero padding",
        "PE Header & COFF File Header (0x0080 - 0x0097)\nSig: 'PE\\0\\0', Machine: 0x8664, Sec: 2",
        "Optional Header PE32+ (0x0098 - 0x0187)\nMagic: 0x20B, Entry: 0x1000, Subsys: 0x0A (EFI)",
        "Section Table: .text & .rdata (0x0188 - 0x01D7)\n2x 40-byte Section Headers",
        "Payload Sections (.text: 0x200, .rdata: 0x400)\nMachine code (20B) + UTF-16 String (28B)",
    ]
    colors = ["#FFE082", "#ECEFF1", "#81D4FA", "#80CBC4", "#C5CAE9", "#FFCC80"]

    for y, h, lbl, c in zip(y_coords, heights, labels, colors):
        ax1.broken_barh([(0.1, 0.8)], (y, h), facecolors=c, edgecolor="#37474F", linewidth=1.5)
        ax1.text(0.5, y + h / 2, lbl, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#263238")

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.5, 6.2)
    ax1.axis("off")
    ax1.set_title("PE32+ Binary Layout on Disk\n(1536 Bytes = 0x600 Bytes)", fontsize=12, fontweight="bold", color="#0D47A1")

    # Right: Header Field Key Metrics
    ax2.axis("off")
    info_text = (
        "═══════════════════════════════════════════════════════════════════\n"
        "  PE32+ Header Field Verification & UEFI Standard Conformity\n"
        "═══════════════════════════════════════════════════════════════════\n\n"
        f"  • DOS Signature (e_magic)      : 0x{pe.dos_magic:04X} ('MZ')\n"
        f"  • PE Header Offset (e_lfanew)  : 0x{pe.pe_header_offset:04X} (Decimal 128)\n"
        f"  • Machine Architecture         : 0x{pe.machine:04X} (IMAGE_FILE_MACHINE_AMD64)\n"
        f"  • Number of Sections           : {pe.num_sections} (.text, .rdata)\n"
        f"  • Size of Optional Header      : 0x{pe.size_of_optional_header:04X} ({pe.size_of_optional_header} bytes)\n"
        f"  • Optional Header Magic        : 0x{pe.magic:04X} (PE32+ 64-bit Executable)\n"
        f"  • Address of Entry Point       : 0x{pe.entry_point_rva:08X} (RVA = +4096 bytes)\n"
        f"  • Preferred Image Base         : 0x{pe.image_base:016X}\n"
        f"  • Section Alignment (Memory)   : 0x{pe.section_alignment:08X} (4 KB Page Boundary)\n"
        f"  • File Alignment (Disk)        : 0x{pe.file_alignment:08X} (512 B Sector Boundary)\n"
        f"  • Subsystem                    : 0x{pe.subsystem:04X} (0x000A = EFI Application)\n"
        f"  • Size of Image in Memory      : 0x{pe.size_of_image:08X} (12 KB)\n"
        f"  • Size of Headers              : 0x{pe.size_of_headers:08X} (512 bytes)\n\n"
        "───────────────────────────────────────────────────────────────────\n"
        "  Section Headers Summary:\n"
        f"    1. [{pe.sections[0].name:8s}] VAddr: 0x{pe.sections[0].virtual_address:04X}  VSize: 0x{pe.sections[0].virtual_size:02X}  Offset: 0x{pe.sections[0].raw_data_offset:04X}  (CODE, EXEC, READ)\n"
        f"    2. [{pe.sections[1].name:8s}] VAddr: 0x{pe.sections[1].virtual_address:04X}  VSize: 0x{pe.sections[1].virtual_size:02X}  Offset: 0x{pe.sections[1].raw_data_offset:04X}  (DATA, READ)\n"
        "═══════════════════════════════════════════════════════════════════"
    )
    ax2.text(
        0.02,
        0.5,
        info_text,
        family="monospace",
        fontsize=9.2,
        va="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#90A4AE", linewidth=1.5),
    )
    ax2.set_title("PE32+ COFF & Optional Header Detailed Inspection", fontsize=12, fontweight="bold", color="#0D47A1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_uefi_calling_convention(output_path: str | Path) -> Path:
    """Generate high-resolution diagram of Microsoft x64 calling convention & stack frame (fig1_2)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # Left: Register state transitions
    steps = [
        "1. Entry by UEFI Firmware\n(RCX=ImageHandle, RDX=SystemTable, RSP%16==8)",
        "2. sub rsp, 0x28\n(Alloc 32B Shadow Space + 8B Padding -> RSP%16==0)",
        "3. mov rcx, [rdx + 0x40]\n(Load SystemTable->ConOut pointer into RCX)",
        "4. lea rdx, [rip + 0xff1]\n(Load UTF-16 'Hello, world!' address into RDX)",
        "5. call [rcx + 8]\n(Indirect call OutputString(This=RCX, String=RDX))",
        "6. jmp -2 (.fin)\n(Infinite busy loop EB FE halting execution)",
    ]
    y_pos = np.arange(len(steps))[::-1]

    for y, text in zip(y_pos, steps):
        ax1.broken_barh([(0.05, 0.9)], (y - 0.35, 0.7), facecolors="#E1F5FE", edgecolor="#0288D1", linewidth=1.5)
        ax1.text(0.5, y, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#01579B")

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.8, len(steps) - 0.2)
    ax1.axis("off")
    ax1.set_title("x86-64 Execution Flow & Instruction Semantics", fontsize=12, fontweight="bold", color="#0D47A1")

    # Right: Stack Frame Visualization
    stack_blocks = [
        ("Previous Stack / Caller Return Addr\n(Pushed by UEFI Firmware Call)", "#CFD8DC"),
        ("Shadow Space Slot 4 (0x18[RSP])\nReserved for 4th arg (R9 home)", "#FFF9C4"),
        ("Shadow Space Slot 3 (0x10[RSP])\nReserved for 3rd arg (R8 home)", "#FFF9C4"),
        ("Shadow Space Slot 2 (0x08[RSP])\nReserved for 2nd arg (RDX home)", "#FFF9C4"),
        ("Shadow Space Slot 1 (0x00[RSP])\nReserved for 1st arg (RCX home)", "#FFF9C4"),
        ("16-byte Alignment Padding (8 Bytes)\n(RSP alignment guarantee)", "#FFCCBC"),
    ]
    y_s = np.arange(len(stack_blocks))[::-1]
    for y, (name, color) in zip(y_s, stack_blocks):
        ax2.broken_barh([(0.1, 0.8)], (y - 0.4, 0.8), facecolors=color, edgecolor="#455A64", linewidth=1.5)
        ax2.text(0.5, y, name, ha="center", va="center", fontsize=9, fontweight="bold", color="#263238")

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.8, len(stack_blocks) - 0.2)
    ax2.axis("off")
    ax2.set_title("Microsoft x64 Stack Frame (sub rsp, 0x28)\nShadow Space & 16-Byte Alignment", fontsize=12, fontweight="bold", color="#0D47A1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 1.2: Linux Block Subsystem & USB Device Detection Simulation
# ==============================================================================


@dataclass(frozen=True)
class LinuxPartition:
    """Represents a disk partition detected by Linux kernel."""

    name: str
    major: int
    minor: int
    sector_count: int
    size_bytes: int
    mountpoint: str = ""

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024**3)


@dataclass(frozen=True)
class LinuxBlockDevice:
    """Represents a Linux block device in /sys/block."""

    name: str
    major: int
    minor: int
    is_removable: bool
    transport: str
    sector_count: int
    size_bytes: int
    vendor: str = "Unknown"
    model: str = "Unknown"
    partitions: tuple[LinuxPartition, ...] = ()

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def is_usb_boot_candidate(self) -> bool:
        """Removable USB drive is the target candidate for MikanOS boot media."""
        return self.is_removable and self.transport == "USB"


class LinuxBlockSubsystemSimulator:
    """Simulates Linux kernel block layer, dev_t decoding, and USB hotplug events."""

    @staticmethod
    def encode_dev_t(major: int, minor: int) -> int:
        """Encode major/minor into 32-bit Linux dev_t: (major << 20) | (minor & 0xFFFFF)."""
        return ((major & 0xFFF) << 20) | (minor & 0xFFFFF)

    @staticmethod
    def decode_dev_t(dev: int) -> tuple[int, int]:
        """Decode 32-bit Linux dev_t into (major, minor)."""
        major = (dev >> 20) & 0xFFF
        minor = dev & 0xFFFFF
        return major, minor

    @staticmethod
    def get_standard_major_name(major: int) -> str:
        """Return canonical Linux subsystem name for known major device numbers."""
        mapping = {
            1: "ram (RAM disk driver)",
            2: "fd (Floppy disk driver)",
            3: "hda/hdb (Legacy IDE primary)",
            7: "loop (Loopback device driver)",
            8: "sd (SCSI disk subsystem: SATA/SAS/USB)",
            9: "md (Multiple Devices / RAID)",
            179: "mmcblk (MMC/SD card driver)",
            259: "nvme (NVM Express block driver)",
        }
        return mapping.get(major, f"custom / dynamic driver (major {major})")

    @classmethod
    def create_typical_workstation_state(cls) -> list[LinuxBlockDevice]:
        """Create a typical developer workstation state before inserting USB drive."""
        nvme = LinuxBlockDevice(
            name="nvme0n1",
            major=259,
            minor=0,
            is_removable=False,
            transport="NVMe",
            sector_count=1953525168,
            size_bytes=1953525168 * 512,
            vendor="Samsung",
            model="SSD 980 PRO 1TB",
            partitions=(
                LinuxPartition("nvme0n1p1", 259, 1, 1048576, 1048576 * 512, "/boot/efi"),
                LinuxPartition("nvme0n1p2", 259, 2, 33554432, 33554432 * 512, "[SWAP]"),
                LinuxPartition("nvme0n1p3", 259, 3, 1918920704, 1918920704 * 512, "/"),
            ),
        )
        return [nvme]

    @classmethod
    def simulate_usb_insertion(cls, devices_before: list[LinuxBlockDevice]) -> list[LinuxBlockDevice]:
        """Simulate hotplugging an 8GB/16GB USB flash drive into the system."""
        usb = LinuxBlockDevice(
            name="sdb",
            major=8,
            minor=16,
            is_removable=True,
            transport="USB",
            sector_count=30031872,
            size_bytes=30031872 * 512,
            vendor="SanDisk",
            model="Ultra Fit USB 3.0",
            partitions=(
                LinuxPartition("sdb1", 8, 17, 30029824, 30029824 * 512, "/media/student/MIKAN_USB"),
            ),
        )
        return devices_before + [usb]

    @classmethod
    def detect_inserted_devices(
        cls, before: list[LinuxBlockDevice], after: list[LinuxBlockDevice]
    ) -> list[LinuxBlockDevice]:
        """Find newly appeared block devices by diffing names."""
        before_names = {d.name for d in before}
        return [d for d in after if d.name not in before_names]

    @classmethod
    def verify_target_safety(cls, target_name: str, devices: list[LinuxBlockDevice]) -> dict[str, str | bool]:
        """Evaluate if writing to target_name is safe (Accidental Data Loss Prevention)."""
        for d in devices:
            if d.name == target_name:
                is_safe = d.is_usb_boot_candidate
                reason = "Safe Removable USB Flash Drive" if is_safe else "DANGER: Host Internal Disk or Non-USB Device"
                return {
                    "found": True,
                    "target": target_name,
                    "is_safe": is_safe,
                    "removable": d.is_removable,
                    "transport": d.transport,
                    "size_gib": f"{d.size_gib:.2f} GiB",
                    "reason": reason,
                }
        return {
            "found": False,
            "target": target_name,
            "is_safe": False,
            "reason": f"Device {target_name} not found in system",
        }


def plot_linux_block_layer_and_usb_detection(output_path: str | Path) -> Path:
    """Generate high-resolution architecture diagram of Linux block layer & USB enumeration (fig1_3)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.1, 1.2]})

    # Left: Linux Storage & Block I/O Layer Architecture
    layers = [
        ("User Space Applications & System Tools\n(cp, dd, lsblk, dmesg, fdisk)", "#E3F2FD", "#1565C0"),
        ("Virtual File System (VFS)\n(/dev/sdb, /media/student/MIKAN_USB, open/read/write)", "#E8F5E9", "#2E7D32"),
        ("Generic Block Layer & Page Cache\n(bio structures, I/O Request Queues, major:minor dispatch)", "#FFF3E0", "#E65100"),
        ("SCSI Subsystem Core (scsi_mod & sd_mod)\n(SCSI disk driver, translates block I/O to SCSI commands)", "#F3E5F5", "#6A1B9A"),
        ("USB Storage Transport (usb-storage / uas)\n(Bulk-Only Transport BOT, Command/Data/Status blocks)", "#FCE4EC", "#C2185B"),
        ("USB Core & Host Controller Driver (xHCI / EHCI)\n(PCIe MMIO registers, USB packets, URB transfer)", "#ECEFF1", "#37474F"),
        ("Physical Hardware (USB Flash Drive)\n(NAND Flash, Flash Controller, USB 3.0 PHY)", "#CFD8DC", "#263238"),
    ]

    y_pos = np.arange(len(layers))[::-1]
    for y, (title, fcolor, ecolor) in zip(y_pos, layers):
        ax1.broken_barh([(0.05, 0.9)], (y - 0.38, 0.76), facecolors=fcolor, edgecolor=ecolor, linewidth=2)
        ax1.text(0.5, y, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1A237E")

    # Connect arrows between layers
    for i in range(len(layers) - 1):
        ax1.annotate(
            "",
            xy=(0.5, y_pos[i + 1] + 0.38),
            xytext=(0.5, y_pos[i] - 0.38),
            arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.5),
        )

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.8, len(layers) - 0.2)
    ax1.axis("off")
    ax1.set_title("Linux Storage Subsystem Architecture\n(Everything is a File -> Block Device I/O Stack)", fontsize=12, fontweight="bold", color="#0D47A1")

    # Right: USB Insertion & Hotplug Enumeration Sequence
    steps = [
        ("Step 1: Physical Attachment & Electrical Signal", "xHCI Root Hub port detects voltage change on D+/D- lines.\nPort Status & Control register sets Current Connect Status (CCS = 1).", "#FFEBEE", "#C62828"),
        ("Step 2: USB Bus Enumeration (usbcore)", "xHCI issues SET_ADDRESS, reads Device/Config Descriptors.\nIdentifies Class 0x08 (Mass Storage), Protocol 0x50 (Bulk-Only).", "#FFF8E1", "#F57F17"),
        ("Step 3: Driver Binding (usb-storage & sd_mod)", "Kernel binds usb-storage driver, registers virtual SCSI Host.\nsd_mod issues SCSI INQUIRY (Vendor/Model) & READ CAPACITY.", "#E8F5E9", "#2E7D32"),
        ("Step 4: sysfs Registration & major:minor Assignment", "Kernel assigns major=8 (SCSI disk), minor=16 (sdb).\nCreates sysfs entries at /sys/block/sdb with size, removable=1.", "#E1F5FE", "#0277BD"),
        ("Step 5: uevent Broadcast & udev Device Node Creation", "Kernel fires kobject_uevent via Netlink.\nsystemd-udevd reads rules, creates /dev/sdb, /dev/sdb1, and /dev/disk/by-id/*.", "#EDE7F6", "#512DA8"),
        ("Step 6: User Space Discovery & Safe Verification", "User runs lsblk or dmesg | tail.\nChecks NAME=sdb, RM=1, VENDOR/MODEL, prevents wiping host disk!", "#FBE9E7", "#D84315"),
    ]

    y_seq = np.arange(len(steps))[::-1]
    for y, (title, desc, fcolor, ecolor) in zip(y_seq, steps):
        ax2.broken_barh([(0.05, 0.9)], (y - 0.40, 0.80), facecolors=fcolor, edgecolor=ecolor, linewidth=1.8)
        full_text = f"▶ {title}\n{desc}"
        ax2.text(0.08, y, full_text, ha="left", va="center", fontsize=8.8, color="#212121", linespacing=1.3)

    for i in range(len(steps) - 1):
        ax2.annotate(
            "",
            xy=(0.5, y_seq[i + 1] + 0.40),
            xytext=(0.5, y_seq[i] - 0.40),
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.8),
        )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.8, len(steps) - 0.2)
    ax2.axis("off")
    ax2.set_title("USB Hotplug & Device Node Generation Sequence\n(Hardware Connect -> dmesg / sysfs -> /dev/sdX)", fontsize=12, fontweight="bold", color="#0D47A1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_uefi_usb_fat_structure(output_path: str | Path) -> Path:
    """Generate high-resolution layout diagram of USB drive partitioning & FAT32 UEFI structure (fig1_4)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5), gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Left: Physical Disk Layout (MBR / GPT & Partition 1)
    disk_segments = [
        ("LBA 0: Protective MBR / MBR\n(Legacy Partition Table + 0x55AA signature)", "#CFD8DC", "#37474F", 0.12),
        ("LBA 1: GPT Header\n(Signature 'EFI PART', Disk GUID, Partition Table LBA)", "#ECEFF1", "#455A64", 0.10),
        ("LBA 2-33: GPT Partition Entries\n(128 entries x 128 bytes, ESP GUID, Start/End LBA)", "#E0E0E0", "#616161", 0.14),
        ("LBA 2048+: Partition 1 (EFI System Partition)\nFAT32 Volume for MikanOS Boot Media", "#C8E6C9", "#2E7D32", 0.50),
        ("Disk Tail: Backup GPT Partition Entries & Header", "#EEEEEE", "#757575", 0.10),
    ]

    cur_y = 0.0
    for title, fcolor, ecolor, height in disk_segments:
        ax1.broken_barh([(0.1, 0.8)], (cur_y, height), facecolors=fcolor, edgecolor=ecolor, linewidth=1.8)
        ax1.text(0.5, cur_y + height / 2.0, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#263238")
        cur_y += height + 0.02

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.05, cur_y + 0.05)
    ax1.axis("off")
    ax1.set_title("Physical Storage Layout: USB Flash Drive\n(GPT / MBR Partitioning Scheme)", fontsize=12, fontweight="bold", color="#0D47A1")

    # Right: FAT32 Internal File System Structure & Directory Path
    fat_blocks = [
        ("Reserved Region (LBA 0 of Partition)\n• VBR (Volume Boot Record) / BPB (BIOS Parameter Block)\n• Bytes per Sector: 512, Sectors per Cluster: 8 (4KB Clusters)\n• Reserved Sector Count: 32 (contains FSInfo Sector)", "#FFF9C4", "#FBC02D"),
        ("FAT Region (File Allocation Table)\n• FAT #1: Cluster allocation table (32-bit entries, cluster chain tracking)\n• FAT #2: Exact mirror duplicate for data redundancy & integrity", "#FFE0B2", "#FB8C00"),
        ("Data Region (Cluster #2 Root Directory)\n• Directory Entry: 32 bytes per file/dir (8.3 name, attributes, cluster)\n• Root contains entry: 'EFI' (Attribute: Directory, Points to Cluster #3)", "#E1BEE7", "#8E24AA"),
        ("Directory Cluster: /EFI/\n• Directory Entry: 'BOOT' (Attribute: Directory, Points to Cluster #4)", "#D1C4E9", "#5E35B1"),
        ("Directory Cluster: /EFI/BOOT/\n• Directory Entry: 'BOOTX64.EFI' (Points to Cluster #5, Size: 1536 Bytes)\n• Standard default fallback boot path per UEFI Specification 2.10!", "#BBDEFB", "#1976D2"),
        ("File Data Clusters: BOOTX64.EFI Binary\n• Contains PE32+ MZ/PE header and x86-64 machine code\n• Executed directly by UEFI Firmware Boot Manager", "#C8E6C9", "#388E3C"),
    ]

    y_fat = np.arange(len(fat_blocks))[::-1]
    for y, (desc, fcolor, ecolor) in zip(y_fat, fat_blocks):
        ax2.broken_barh([(0.05, 0.9)], (y - 0.40, 0.80), facecolors=fcolor, edgecolor=ecolor, linewidth=1.8)
        ax2.text(0.08, y, desc, ha="left", va="center", fontsize=8.8, color="#212121", linespacing=1.25)

    for i in range(len(fat_blocks) - 1):
        ax2.annotate(
            "",
            xy=(0.5, y_fat[i + 1] + 0.40),
            xytext=(0.5, y_fat[i] - 0.40),
            arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.6),
        )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.8, len(fat_blocks) - 0.2)
    ax2.axis("off")
    ax2.set_title("FAT32 Volume Internal Architecture & Boot Path\n(/EFI/BOOT/BOOTX64.EFI Traversal per UEFI Spec)", fontsize=12, fontweight="bold", color="#0D47A1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 1.3: Windows Subsystem for Linux (WSL) Architecture & Storage Simulation
# ==============================================================================


class WSLArchType(Enum):
    """WSL Architecture classification."""
    WSL1 = "WSL1 (System Call Translation / LXCore.sys)"
    WSL2 = "WSL2 (Hyper-V Lightweight Utility VM + Native Linux Kernel)"
    NATIVE_LINUX = "Native Linux (Direct Hardware / Bare Metal)"


@dataclass
class DrvFsMountOptions:
    """DrvFs (Windows Drive Mount on WSL) mount options configuration."""
    metadata: bool = True
    case: str = "dir"  # "off", "dir", "force"
    uid: int = 1000
    gid: int = 1000
    umask: int = 0o022
    fmask: int = 0o111
    dmask: int = 0o000

    def to_mount_flags_string(self) -> str:
        """Generate mount options argument string for mount command or wsl.conf."""
        opts = []
        if self.metadata:
            opts.append("metadata")
        if self.case in ("off", "dir", "force"):
            opts.append(f"case={self.case}")
        opts.append(f"uid={self.uid}")
        opts.append(f"gid={self.gid}")
        opts.append(f"umask={oct(self.umask)[2:]}")
        opts.append(f"fmask={oct(self.fmask)[2:]}")
        opts.append(f"dmask={oct(self.dmask)[2:]}")
        return ",".join(opts)

    def calculate_file_permissions(self, is_directory: bool = False, has_metadata: bool = False, stored_mode: Optional[int] = None) -> int:
        """Calculate effective POSIX permission mode (e.g. 0o755)."""
        if has_metadata and stored_mode is not None:
            return stored_mode & 0o7777

        base_mode = 0o777 if is_directory else 0o777
        mask = self.dmask if is_directory else self.fmask
        # Also apply general umask
        effective = base_mode & ~(mask | self.umask)
        return effective


class WSLPathConverter:
    """Bi-directional path converter between Windows and WSL2 Linux."""

    @staticmethod
    def is_windows_path(path: str) -> bool:
        """Check if path is a Windows style path (e.g. C:\\... or D:/...)."""
        if len(path) >= 2 and path[0].isalpha() and path[1] == ":":
            return True
        if path.startswith("\\\\") or path.startswith("//"):
            return True
        return False

    @staticmethod
    def is_wsl_path(path: str) -> bool:
        """Check if path is a standard Unix/WSL POSIX path."""
        return path.startswith("/")

    @staticmethod
    def windows_to_wsl(win_path: str, mount_root: str = "/mnt") -> str:
        """Convert a Windows path (e.g. D:\\EFI\\BOOT\\BOOTX64.EFI) to WSL path (/mnt/d/EFI/BOOT/BOOTX64.EFI)."""
        clean_path = win_path.replace("\\", "/")
        # UNC WSL path check: \\wsl$\Ubuntu\home\... -> /home/...
        unc_wsl_prefix = "//wsl$/"
        if clean_path.lower().startswith(unc_wsl_prefix):
            parts = clean_path[len(unc_wsl_prefix):].split("/", 1)
            if len(parts) > 1:
                return "/" + parts[1]
            return "/"

        # Drive letter check: C:/path or D:/path
        if len(clean_path) >= 2 and clean_path[0].isalpha() and clean_path[1] == ":":
            drive = clean_path[0].lower()
            rest = clean_path[2:]
            if not rest.startswith("/"):
                rest = "/" + rest
            root = mount_root.rstrip("/")
            return f"{root}/{drive}{rest}"

        return clean_path

    @staticmethod
    def wsl_to_windows(wsl_path: str, distro_name: str = "Ubuntu", mount_root: str = "/mnt") -> str:
        """Convert a WSL path (/mnt/d/EFI/BOOT) to Windows path (D:\\EFI\\BOOT or \\\\wsl$\\Distro\\...)."""
        clean_path = wsl_path.replace("\\", "/")
        root = mount_root.rstrip("/") + "/"
        if clean_path.startswith(root):
            # Mounted Windows drive: /mnt/c/path -> C:\path
            sub = clean_path[len(root):]
            if len(sub) > 0 and sub[0].isalpha():
                drive = sub[0].upper()
                rest = sub[1:].replace("/", "\\")
                if not rest.startswith("\\") and len(rest) > 0:
                    rest = "\\" + rest
                return f"{drive}:{rest}"

        # Native Linux path: /home/student/mikanos -> \\wsl$\Ubuntu\home\student\mikanos
        rel_posix = clean_path.lstrip("/").replace("/", "\\")
        return f"\\\\wsl$\\{distro_name}\\{rel_posix}"


class WSLStorageSimulator:
    """Simulator for WSL1 vs WSL2 storage architectures, DrvFs, 9P/virtio-fs, and USB/IP."""

    @staticmethod
    def detect_environment(version_string: str) -> WSLArchType:
        """Detect WSL version from /proc/version or uname output."""
        v_lower = version_string.lower()
        if "microsoft-standard" in v_lower or "wsl2" in v_lower:
            return WSLArchType.WSL2
        elif "microsoft" in v_lower or "wsl" in v_lower:
            return WSLArchType.WSL1
        return WSLArchType.NATIVE_LINUX

    @staticmethod
    def parse_proc_mounts(mounts_data: str) -> List[Dict[str, Any]]:
        """Parse Linux /proc/mounts to find 9p, virtiofs, and drvfs entries."""
        results = []
        for line in mounts_data.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                dev, mnt_point, fstype, options = parts[0], parts[1], parts[2], parts[3]
                opts_list = options.split(",")
                results.append({
                    "device": dev,
                    "mount_point": mnt_point,
                    "fstype": fstype,
                    "options": opts_list,
                    "is_drvfs": fstype in ("drvfs", "9p", "virtiofs") and mnt_point.startswith("/mnt/"),
                })
        return results

    @staticmethod
    def get_storage_performance_matrix() -> Dict[str, Dict[str, Any]]:
        """Return I/O performance characteristics of WSL2 storage backends."""
        return {
            "ext4_vhdx": {
                "name": "WSL2 Virtual Hard Disk (ext4 inside VHDX)",
                "location": "/home/username/... or /root/...",
                "read_iops": "High (Native Linux Block I/O)",
                "write_iops": "High (Native Linux Block I/O)",
                "git_status_speed": "Instant (~0.05s)",
                "case_sensitive": True,
                "posix_metadata": True,
                "recommended_for": "Compiling MikanOS, EDK II build, Git clone",
            },
            "drvfs_9p": {
                "name": "Windows Host Drive via DrvFs (9P / virtio-fs)",
                "location": "/mnt/c/..., /mnt/d/...",
                "read_iops": "Moderate (Inter-VM IPC via 9P/virtio-fs)",
                "write_iops": "Moderate to Low (NTFS Filter & IPC roundtrips)",
                "git_status_speed": "Noticeably slower (~1.5s - 5.0s)",
                "case_sensitive": "Configurable (case=dir / case=off)",
                "posix_metadata": "Configurable (metadata option required)",
                "recommended_for": "Copying final BOOTX64.EFI to USB, Host editor access",
            },
            "usbip_raw": {
                "name": "USB/IP Direct Hardware Attach (usbipd-win)",
                "location": "/dev/sdb, /dev/sdc",
                "read_iops": "High (Raw USB Block SCSI Pass-through)",
                "write_iops": "High (Raw USB Block SCSI Pass-through)",
                "git_status_speed": "N/A (Raw device)",
                "case_sensitive": "Filesystem dependent",
                "posix_metadata": "Filesystem dependent",
                "recommended_for": "QEMU direct disk pass-through, Raw partition management",
            },
        }

    @staticmethod
    def generate_boot_media_copy_script(efi_binary_source: str, windows_drive_letter: str) -> List[str]:
        """Generate safe, idempotent bash commands to copy BOOTX64.EFI to mounted USB drive."""
        clean_drive = windows_drive_letter.rstrip(":").upper()
        drive_lower = clean_drive.lower()
        mount_point = f"/mnt/{drive_lower}"
        target_dir = f"{mount_point}/EFI/BOOT"
        target_file = f"{target_dir}/BOOTX64.EFI"

        script = [
            f"# Check if Windows drive {clean_drive}: is mounted at {mount_point}",
            f"if ! mountpoint -q {mount_point}; then",
            f"    sudo mkdir -p {mount_point}",
            f"    sudo mount -t drvfs {clean_drive}: {mount_point} -o metadata,case=dir",
            "fi",
            f"# Ensure target UEFI directory hierarchy exists (case-preserving)",
            f"sudo mkdir -p {target_dir}",
            f"# Copy built EFI application to the standard UEFI fallback boot path",
            f"sudo cp {efi_binary_source} {target_file}",
            f"# Flush OS buffers to ensure complete physical disk persistence",
            "sync",
            f"echo 'Successfully deployed {efi_binary_source} -> {target_file}'",
        ]
        return script


def plot_wsl1_vs_wsl2_architecture(
    output_path: str = "1/result/fig1_5_wsl1_vs_wsl2_architecture.png",
) -> str:
    """Generate high-resolution architecture comparison diagram between WSL1 and WSL2."""
    setup_plot_style()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={"width_ratios": [1, 1]})

    # Left: WSL1 Architecture (System Call Emulation)
    wsl1_layers = [
        ("User Space (Linux ELF64 Binaries)\nBash, GCC, Clang, Python, Git (Unmodified)", "#E1F5FE", "#0288D1", 0.14),
        ("POSIX System Call API (SYS_read, SYS_write, fork, clone...)", "#FFF9C4", "#FBC02D", 0.09),
        ("Translation Layer: LXSS / LXCore.sys\n(Translates Linux Syscalls -> Windows NT Kernel APIs in real-time)\n• No real Linux kernel running\n• Pico Process / Pico Provider architecture", "#FFE0B2", "#E65100", 0.22),
        ("Windows NT Kernel & Native Drivers (ntoskrnl.exe)\nThread Scheduler, Object Manager, Virtual Memory Manager", "#E8EAF6", "#3F51B5", 0.16),
        ("Direct NTFS / FAT File System Drivers\nDirect disk access via Windows I/O subsystem", "#C8E6C9", "#2E7D32", 0.14),
        ("Physical x86-64 Hardware (CPU, RAM, USB Host Controller, Storage)", "#CFD8DC", "#37474F", 0.12),
    ]

    cur_y = 0.0
    for title, fcolor, ecolor, height in wsl1_layers:
        ax1.broken_barh([(0.05, 0.9)], (cur_y, height), facecolors=fcolor, edgecolor=ecolor, linewidth=1.8)
        ax1.text(0.5, cur_y + height / 2.0, title, ha="center", va="center", fontsize=9.2, color="#212121", linespacing=1.2)
        cur_y += height + 0.02

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.04, cur_y + 0.04)
    ax1.axis("off")
    ax1.set_title("WSL1: System Call Translation Architecture\n(LXCore.sys Emulation, No Linux Kernel, High NTFS Compatibility)", fontsize=11.5, fontweight="bold", color="#C62828")

    # Right: WSL2 Architecture (Hyper-V Utility VM + Native Linux Kernel)
    wsl2_layers = [
        ("User Space (Linux ELF64 Binaries)\nBash, GCC, Clang, EDK II, QEMU, Docker (100% Native Syscalls)", "#E1F5FE", "#0288D1", 0.14),
        ("Native Linux Kernel (x86-64 v5.x / v6.x MS-Standard)\nReal Page Tables, Real cgroups, Real Netfilter, ext4 Driver", "#C8E6C9", "#2E7D32", 0.18),
        ("Inter-VM Storage & I/O Communication Bridge\n• 9P / virtio-fs File Sharing Bridge -> DrvFs (/mnt/c, /mnt/d)\n• Virtual ext4 VHDX (Direct raw block I/O at native Linux speed!)\n• USB/IP Driver (usbipd-win attach /dev/sdX)", "#FFF9C4", "#F57F17", 0.20),
        ("Hyper-V Lightweight Utility VM (Hypervisor Type-1)\nDynamic Memory Allocation (vHost), Micro-VM Container (<1s boot)", "#FFE0B2", "#FB8C00", 0.14),
        ("Host Windows 11/10 OS & Hyper-V Hypervisor\nHardware Virtualization (Intel VT-x / AMD-V, SLAT / EPT)", "#E8EAF6", "#3F51B5", 0.14),
        ("Physical x86-64 Hardware (CPU, RAM, USB 3.0 xHCI Controller, Storage)", "#CFD8DC", "#37474F", 0.12),
    ]

    cur_y = 0.0
    for title, fcolor, ecolor, height in wsl2_layers:
        ax2.broken_barh([(0.05, 0.9)], (cur_y, height), facecolors=fcolor, edgecolor=ecolor, linewidth=1.8)
        ax2.text(0.5, cur_y + height / 2.0, title, ha="center", va="center", fontsize=9.2, color="#212121", linespacing=1.2)
        cur_y += height + 0.02

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.04, cur_y + 0.04)
    ax2.axis("off")
    ax2.set_title("WSL2: Hyper-V Utility VM + Native Linux Kernel\n(100% System Call Compatibility, VHDX ext4 Speed, USB/IP Support)", fontsize=11.5, fontweight="bold", color="#1565C0")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_wsl_storage_and_usb_mount(
    output_path: str = "1/result/fig1_6_wsl_storage_and_usb_mount.png",
) -> str:
    """Generate high-resolution diagram of WSL2 storage sharing, DrvFs mount, and USB/IP access."""
    setup_plot_style()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 9))

    # Containers representing Windows Host vs WSL2 VM
    # Windows Host Box
    rect_win = plt.Rectangle((0.03, 0.08), 0.42, 0.84, facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2.2, linestyle="--", alpha=0.5)
    ax.add_patch(rect_win)
    ax.text(0.24, 0.88, "Windows Host Operating System\n(NTFS, Explorer, Hardware Management)", ha="center", va="center", fontsize=11, fontweight="bold", color="#4A148C")

    # WSL2 VM Box
    rect_wsl = plt.Rectangle((0.55, 0.08), 0.42, 0.84, facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=2.2, linestyle="--", alpha=0.5)
    ax.add_patch(rect_wsl)
    ax.text(0.76, 0.88, "WSL2 Virtual Machine (Linux Kernel)\n(ext4, DrvFs, GNU/Clang Toolchain)", ha="center", va="center", fontsize=11, fontweight="bold", color="#0D47A1")

    # Windows Components
    win_nodes = [
        ("Host C: Drive (NTFS)\nC:\\Users\\uchan\\mikanos", 0.68, "#FFFFFF", "#7B1FA2"),
        ("USB Flash Drive (FAT32, D:)\nPhysical UEFI Boot Media", 0.46, "#FFF9C4", "#F57F17"),
        ("usbipd-win Service\nUSB/IP Server Daemon for Windows", 0.24, "#FFE0B2", "#E65100"),
    ]
    for text, y, fcol, ecol in win_nodes:
        ax.broken_barh([(0.06, 0.36)], (y - 0.07, 0.14), facecolors=fcol, edgecolor=ecol, linewidth=1.6)
        ax.text(0.24, y, text, ha="center", va="center", fontsize=9.2, color="#212121", linespacing=1.2)

    # WSL2 Components
    wsl_nodes = [
        ("DrvFs Mount: /mnt/c\n(9P / virtio-fs file protocol bridge)", 0.68, "#FFFFFF", "#1976D2"),
        ("DrvFs USB Mount: /mnt/d\n(sudo mount -t drvfs D: /mnt/d -o metadata)", 0.46, "#FFF9C4", "#F57F17"),
        ("USB/IP Client (vhci_hcd)\nRaw Block Device: /dev/sdb (SCSI Disk)", 0.24, "#C8E6C9", "#2E7D32"),
    ]
    for text, y, fcol, ecol in wsl_nodes:
        ax.broken_barh([(0.58, 0.36)], (y - 0.07, 0.14), facecolors=fcol, edgecolor=ecol, linewidth=1.6)
        ax.text(0.76, y, text, ha="center", va="center", fontsize=9.2, color="#212121", linespacing=1.2)

    # Connecting arrows between Windows and WSL2
    # 1. /mnt/c bridging
    ax.annotate(
        "9P / virtio-fs Bridge\n(Cross-VM File I/O)",
        xy=(0.58, 0.68), xytext=(0.42, 0.68),
        arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=2.2),
        ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#4A148C"
    )

    # 2. DrvFs USB Mount (Method A)
    ax.annotate(
        "Method A: DrvFs Mount\n(File-level FAT32 Access)\nsudo mount -t drvfs D: /mnt/d",
        xy=(0.58, 0.46), xytext=(0.42, 0.46),
        arrowprops=dict(arrowstyle="->", color="#E65100", lw=2.2),
        ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#BF360C"
    )

    # 3. USB/IP Pass-through (Method B)
    ax.annotate(
        "Method B: usbipd-win Pass-through\n(Raw Hardware Block Pass-through)\nusbipd attach --wsl --busid X-Y",
        xy=(0.58, 0.24), xytext=(0.42, 0.24),
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.2),
        ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1B5E20"
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.05, 0.96)
    ax.axis("off")
    ax.set_title("WSL2 Storage Interoperability & USB Media Access Architecture\n(DrvFs File Bridge vs USB/IP Hardware Pass-through)", fontsize=13, fontweight="bold", color="#0D47A1", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 1.4: QEMU & OVMF UEFI Virtual PC Emulation & Virtual FAT Architecture
# ==============================================================================

class QemuFirmwareMode(Enum):
    """Firmware loading mode for UEFI boot in QEMU."""
    BIOS_DIRECT = "bios_direct"        # -bios OVMF.fd (Single file, simplest)
    PFLASH_SPLIT = "pflash_split"      # -drive if=pflash (Split CODE.fd and VARS.fd)
    PFLASH_UNIFIED = "pflash_unified"  # -drive if=pflash (Single unified file)


class QemuDriveFormat(Enum):
    """Drive format and mounting scheme in QEMU."""
    VFAT_DIR = "vfat_dir"        # format=raw,file=fat:rw:<dir> (Virtual FAT)
    RAW_DISK = "raw_disk"        # format=raw,file=<img_path> (Disk image)
    QCOW2_DISK = "qcow2_disk"    # format=qcow2,file=<img_path> (Copy-on-write image)


@dataclass
class QemuVirtualHardwareConfig:
    """Virtual hardware devices and topology configuration for QEMU."""
    memory_mb: int = 1024
    smp_cores: int = 1
    cpu_model: str = "qemu64"
    enable_kvm: bool = False
    enable_xhci: bool = True           # NEC uPD720200 USB 3.0 / xHCI controller
    enable_usb_mouse: bool = False
    enable_usb_kbd: bool = False
    monitor_mode: str = "stdio"        # Terminal QEMU monitor
    display_mode: str = "default"      # GUI display backend ("none", "default")
    net_mode: str = "none"             # Network configuration ("none", "user")
    gdb_debug: bool = False            # Freeze CPU at reset and wait for GDB
    gdb_port: int = 1234
    extra_args: list[str] = field(default_factory=list)


@dataclass
class QemuEmulatorConfig:
    """Comprehensive QEMU + OVMF execution configuration generator & validator."""
    ovmf_code_path: str = "/usr/share/OVMF/OVMF_CODE.fd"
    ovmf_vars_path: Optional[str] = "/usr/share/OVMF/OVMF_VARS.fd"
    disk_target: str = "disk_img"
    drive_format: QemuDriveFormat = QemuDriveFormat.VFAT_DIR
    firmware_mode: QemuFirmwareMode = QemuFirmwareMode.PFLASH_SPLIT
    hardware: QemuVirtualHardwareConfig = field(default_factory=QemuVirtualHardwareConfig)

    def build_command(self) -> list[str]:
        """Construct full qemu-system-x86_64 argument list."""
        cmd = ["qemu-system-x86_64"]

        # 1. CPU and RAM topology
        cmd.extend(["-m", f"{self.hardware.memory_mb}M"])
        if self.hardware.smp_cores > 1:
            cmd.extend(["-smp", str(self.hardware.smp_cores)])
        cmd.extend(["-cpu", self.hardware.cpu_model])

        if self.hardware.enable_kvm:
            cmd.append("-enable-kvm")

        # 2. Firmware (OVMF)
        if self.firmware_mode == QemuFirmwareMode.BIOS_DIRECT:
            cmd.extend(["-bios", self.ovmf_code_path])
        elif self.firmware_mode == QemuFirmwareMode.PFLASH_SPLIT:
            cmd.extend(["-drive", f"if=pflash,format=raw,readonly=on,file={self.ovmf_code_path}"])
            if self.ovmf_vars_path:
                cmd.extend(["-drive", f"if=pflash,format=raw,file={self.ovmf_vars_path}"])
        elif self.firmware_mode == QemuFirmwareMode.PFLASH_UNIFIED:
            cmd.extend(["-drive", f"if=pflash,format=raw,file={self.ovmf_code_path}"])

        # 3. Storage Drive (Virtual FAT or Disk Image)
        if self.drive_format == QemuDriveFormat.VFAT_DIR:
            cmd.extend(["-drive", f"format=raw,file=fat:rw:{self.disk_target}"])
        elif self.drive_format == QemuDriveFormat.RAW_DISK:
            cmd.extend(["-drive", f"format=raw,file={self.disk_target}"])
        elif self.drive_format == QemuDriveFormat.QCOW2_DISK:
            cmd.extend(["-drive", f"format=qcow2,file={self.disk_target}"])

        # 4. USB / xHCI controllers and devices
        if self.hardware.enable_xhci:
            cmd.extend(["-device", "nec-usb-xhci,id=xhci"])
            if self.hardware.enable_usb_mouse:
                cmd.extend(["-device", "usb-mouse,bus=xhci.0"])
            if self.hardware.enable_usb_kbd:
                cmd.extend(["-device", "usb-kbd,bus=xhci.0"])

        # 5. Monitor and Display
        if self.hardware.monitor_mode:
            cmd.extend(["-monitor", self.hardware.monitor_mode])
        if self.hardware.display_mode != "default":
            cmd.extend(["-display", self.hardware.display_mode])

        # 6. Network
        if self.hardware.net_mode == "none":
            cmd.extend(["-net", "none"])
        elif self.hardware.net_mode == "user":
            cmd.extend(["-net", "nic", "-net", "user"])

        # 7. GDB Debugging
        if self.hardware.gdb_debug:
            cmd.extend(["-gdb", f"tcp::{self.hardware.gdb_port}", "-S"])

        # 8. Extra arguments
        if self.hardware.extra_args:
            cmd.extend(self.hardware.extra_args)

        return cmd

    def build_command_str(self) -> str:
        """Return formatted single-line or multi-line shell command string."""
        args = self.build_command()
        return " ".join(args)

    def validate(self) -> list[str]:
        """Validate configuration settings and return list of warnings/errors."""
        issues = []
        if self.hardware.memory_mb < 128:
            issues.append(f"Guest memory ({self.hardware.memory_mb}MB) is too low for UEFI OVMF (minimum 128MB recommended).")
        if not self.ovmf_code_path:
            issues.append("OVMF code path cannot be empty.")
        if self.firmware_mode == QemuFirmwareMode.PFLASH_SPLIT and not self.ovmf_vars_path:
            issues.append("PFLASH_SPLIT mode selected but ovmf_vars_path is not specified.")
        if not self.disk_target:
            issues.append("disk_target path cannot be empty.")
        return issues

    def explain_parameters(self) -> list[dict[str, str]]:
        """Return human-readable explanation table for each QEMU argument."""
        explanations = [
            {"flag": f"-m {self.hardware.memory_mb}M", "category": "Memory", "explanation": "仮想ゲストRAMの容量。UEFIファームウェアおよびOSカーネル実行空間を確保"},
            {"flag": f"-cpu {self.hardware.cpu_model}", "category": "CPU", "explanation": "エミュレートするCPUモデル（64ビットLong Modeサポート必須）"},
        ]
        if self.firmware_mode == QemuFirmwareMode.PFLASH_SPLIT:
            explanations.append({"flag": f"-drive if=pflash,...,file={self.ovmf_code_path}", "category": "Firmware", "explanation": "OVMF UEFIコードROMを読み取り専用パラレルフラッシュメモリとしてマッピング"})
            if self.ovmf_vars_path:
                explanations.append({"flag": f"-drive if=pflash,...,file={self.ovmf_vars_path}", "category": "Firmware", "explanation": "UEFI NVRAM不揮発性変数領域（ブート順序や環境設定）を書き換え可能フラッシュとしてマッピング"})
        elif self.firmware_mode == QemuFirmwareMode.BIOS_DIRECT:
            explanations.append({"flag": f"-bios {self.ovmf_code_path}", "category": "Firmware", "explanation": "単一バイナリとしてOVMFファームウェアをBIOS ROM空間に直接読み込み"})

        if self.drive_format == QemuDriveFormat.VFAT_DIR:
            explanations.append({"flag": f"-drive format=raw,file=fat:rw:{self.disk_target}", "category": "Storage", "explanation": f"ホスト上のディレクトリ '{self.disk_target}' をQEMUがオンザフライでFAT32仮想ブロックデバイスとしてマウント"})
        elif self.drive_format == QemuDriveFormat.RAW_DISK:
            explanations.append({"flag": f"-drive format=raw,file={self.disk_target}", "category": "Storage", "explanation": f"生ディスクイメージ '{self.disk_target}' をストレージドライブとしてアタッチ"})

        if self.hardware.enable_xhci:
            explanations.append({"flag": "-device nec-usb-xhci,id=xhci", "category": "I/O Device", "explanation": "USB 3.0 ホストコントローラ (xHCI) をPCIバス上に増設（MikanOS 第6章以降でマウス・キーボードを制御）"})

        if self.hardware.monitor_mode:
            explanations.append({"flag": f"-monitor {self.hardware.monitor_mode}", "category": "Debugging", "explanation": "対話型QEMUモニタを標準入出力（stdio）にバインド（CPUレジスタや物理メモリを動的検査可能）"})

        if self.hardware.net_mode == "none":
            explanations.append({"flag": "-net none", "category": "Network", "explanation": "ネットワークデバイスを無効化（不要なPXEブート遅延や割り込みオーバーヘッドを抑止）"})

        return explanations


class VirtualFatSimulator:
    """Simulates QEMU's internal on-the-fly Virtual VFAT filesystem generator."""

    def __init__(self, host_dir: str = "disk_img"):
        self.host_dir = host_dir
        self.sector_size = 512
        self.sectors_per_cluster = 8  # 4096 bytes per cluster
        self.reserved_sectors = 32
        self.num_fats = 2
        self.fat32_root_cluster = 2

    def simulate_vfat_layout(self, files: dict[str, int]) -> dict[str, Any]:
        """Simulate FAT32 partition metadata and cluster allocations for provided file size map.

        files: dict mapping relative path (e.g. 'EFI/BOOT/BOOTX64.EFI') to file size in bytes.
        """
        cluster_size = self.sector_size * self.sectors_per_cluster
        cluster_allocations = {}
        current_cluster = self.fat32_root_cluster + 1  # Root dir uses cluster 2

        # 1. Root directory allocation
        cluster_allocations["/"] = {
            "start_cluster": self.fat32_root_cluster,
            "clusters": [self.fat32_root_cluster],
            "size_bytes": cluster_size,
        }

        # 2. Allocate clusters for subdirectories and files
        total_data_bytes = 0
        file_entries = []

        for rel_path, file_size in files.items():
            parts = rel_path.replace("\\", "/").strip("/").split("/")
            num_clusters = max(1, (file_size + cluster_size - 1) // cluster_size)
            clusters = list(range(current_cluster, current_cluster + num_clusters))
            current_cluster += num_clusters
            total_data_bytes += num_clusters * cluster_size

            cluster_allocations[rel_path] = {
                "start_cluster": clusters[0],
                "clusters": clusters,
                "size_bytes": file_size,
                "allocated_bytes": num_clusters * cluster_size,
            }

            file_entries.append({
                "path": rel_path,
                "name": parts[-1],
                "size": file_size,
                "start_cluster": clusters[0],
                "clusters_count": num_clusters,
                "attributes": 0x20 if "." in parts[-1] else 0x10,  # 0x20: Archive, 0x10: Directory
            })

        # Calculate FAT sectors
        total_clusters = current_cluster + 128  # Slack clusters
        fat_size_sectors = max(32, (total_clusters * 4 + self.sector_size - 1) // self.sector_size)
        total_sectors = self.reserved_sectors + (self.num_fats * fat_size_sectors) + (total_clusters * self.sectors_per_cluster)

        return {
            "sector_size": self.sector_size,
            "sectors_per_cluster": self.sectors_per_cluster,
            "cluster_size_bytes": cluster_size,
            "reserved_sectors": self.reserved_sectors,
            "num_fats": self.num_fats,
            "fat_size_sectors": fat_size_sectors,
            "total_sectors": total_sectors,
            "total_capacity_mb": round((total_sectors * self.sector_size) / (1024 * 1024), 2),
            "root_cluster": self.fat32_root_cluster,
            "files": file_entries,
            "cluster_allocations": cluster_allocations,
            "vfat_limitations": [
                "書き込み遅延: QEMUプロセスのメモリ上にキャッシュされるため、ホストとゲストで同時編集するとデータ不整合が発生する",
                "動的反映の制限: QEMU起動後にホスト側で追加した新規ファイルは再起動するまでゲストUEFIから見えない場合がある",
                "ファイル名の大文字小文字: FAT32 Short Name (8.3形式) と Long File Name (LFN) の変換が発生する",
            ],
        }


class UefiBootPhase(Enum):
    """Execution phases in the UEFI standard platform boot sequence."""
    SEC = "Security (SEC)"                    # Reset vector, CPU initialization, Cache-As-RAM
    PEI = "Pre-EFI Initialization (PEI)"      # Memory detection, chipset initialization
    DXE = "Driver Execution Environment (DXE)"# Core services, protocol database, GOP, FAT driver
    BDS = "Boot Device Selection (BDS)"       # Finding bootable devices, ESP detection
    TSL = "Transient System Load (TSL)"       # Loading \\EFI\\BOOT\\BOOTX64.EFI into RAM
    RT = "Runtime (RT)"                       # OS kernel execution after ExitBootServices


@dataclass
class UefiBootStep:
    """A distinct verification step in the QEMU + OVMF boot lifecycle."""
    step_number: int
    phase: UefiBootPhase
    title: str
    description: str
    hardware_state: dict[str, str]
    is_mikanos_target: bool = False


class QemuUefiBootSequenceSimulator:
    """Simulates the step-by-step UEFI boot sequence under QEMU + OVMF."""

    @staticmethod
    def get_boot_sequence() -> list[UefiBootStep]:
        """Return the complete sequence of boot stages up to MikanOS hello.efi launch."""
        return [
            UefiBootStep(
                step_number=1,
                phase=UefiBootPhase.SEC,
                title="CPU Reset Vector Execution",
                description="QEMU仮想CPUにリセット信号が投入され、x86-64プロセッサは0xFFFFFFF0番地（Reset Vector）からOVMF SECコードを実行開始。CAR（Cache-As-RAM）を設定。",
                hardware_state={"RIP": "0x00000000FFFFFFF0", "CR0": "0x00000010 (Real Mode / No Paging)", "RAM": "未初期化", "Firmware": "OVMF_CODE.fd"},
            ),
            UefiBootStep(
                step_number=2,
                phase=UefiBootPhase.PEI,
                title="Pre-EFI Initialization & Memory Discovery",
                description="OVMF PEIモジュールがQEMU仮想マザーボードのQ35/i440FXチップセットおよび仮想RAM（-m 1024M）を検出し、4GiB空間の物理メモリ初期化とページングを有効化。",
                hardware_state={"RIP": "0x00000000008xxxxx", "CR0": "0x8001003B (Protected Mode / Paging)", "RAM": "1024 MiB 有効化", "Firmware": "OVMF PEI"},
            ),
            UefiBootStep(
                step_number=3,
                phase=UefiBootPhase.DXE,
                title="Driver Execution Environment & Protocol Dispatch",
                description="OVMF DXEコアが起動。64ビットLong Modeへ移行し、GOP（Graphics Output Protocol）、PCIドライバ、xHCIドライバ、FATファイルシステムドライバをメモリ上に順次ロード。",
                hardware_state={"RIP": "0x0000000007xxxxxx", "CR0": "0x80010033 (64-bit Long Mode)", "CR3": "PML4 Page Table", "Firmware": "OVMF DXE Protocols"},
            ),
            UefiBootStep(
                step_number=4,
                phase=UefiBootPhase.BDS,
                title="Boot Device Selection & ESP Partition Scan",
                description="OVMF BDSフェーズが起動デバイスを走査。QEMU仮想FATドライブ（fat:rw:disk_img）を検出し、EFI System Partition (ESP) としてファイルシステムをマウント。",
                hardware_state={"Device": "QEMU VFAT Virtual Disk", "Protocol": "EFI_SIMPLE_FILE_SYSTEM_PROTOCOL", "Status": "ESP Volume Mounted"},
            ),
            UefiBootStep(
                step_number=5,
                phase=UefiBootPhase.TSL,
                title="Locating and Loading \\EFI\\BOOT\\BOOTX64.EFI",
                description="BDSはUEFI規格仕様に従い、ボリューム内のデフォルトブートローダパス '\\EFI\\BOOT\\BOOTX64.EFI' を検索。見つかったPE32+バイナリをメインメモリの空きページに展開。",
                hardware_state={"Path": "\\EFI\\BOOT\\BOOTX64.EFI", "Action": "PE32+ Header Parsing & Image Relocation", "Status": "Success"},
                is_mikanos_target=True,
            ),
            UefiBootStep(
                step_number=6,
                phase=UefiBootPhase.RT,
                title="Transferring Control to EfiMain (MikanOS Execution)",
                description="UEFIブートマネージャが展開先エントリポイント（EfiMain）に関数ポインタジャンプ。RCXにImageHandle、RDXにEFI_SYSTEM_TABLEポインタを渡して実行開始！",
                hardware_state={"RCX": "ImageHandle", "RDX": "EFI_SYSTEM_TABLE*", "RSP": "16-byte aligned Stack", "Output": "Hello, world!"},
                is_mikanos_target=True,
            ),
        ]


class QemuMonitorSimulator:
    """Simulates QEMU monitor interactive commands and their outputs."""

    @staticmethod
    def execute_command(command: str) -> dict[str, Any]:
        """Simulate output for common QEMU monitor inspection commands."""
        cmd = command.strip().lower()
        if cmd == "info registers":
            return {
                "command": command,
                "status": "ok",
                "output": (
                    "RAX=0000000000000000 RBX=00000000067ea018 RCX=00000000067ea018 RDX=00000000067ed018\n"
                    "RSI=0000000000000000 RDI=0000000000000000 RBP=0000000007fa7f50 RSP=0000000007fa7f10\n"
                    "R8 =0000000000000000 R9 =0000000000000000 R10=0000000000000000 R11=0000000000000000\n"
                    "R12=0000000000000000 R13=0000000000000000 R14=0000000000000000 R15=0000000000000000\n"
                    "RIP=00000000067e4010 RFL=00000202 [-------] CPL=0 II=0 A20=1 SMM=0 HLT=0\n"
                    "CR0=80010033 CR2=0000000000000000 CR3=0000000007e01000 CR4=00000668"
                ),
                "explanation": "64ビット汎用レジスタ、命令ポインタ(RIP)、およびページング制御レジスタ(CR0/CR3/CR4)の状態を表示",
            }
        elif cmd.startswith("xp"):
            return {
                "command": command,
                "status": "ok",
                "output": "00000000067e4000: 0x4d 0x5a 0x90 0x00 0x03 0x00 0x00 0x00  MZ......\n00000000067e4008: 0x04 0x00 0x00 0x00 0xff 0xff 0x00 0x00  ........",
                "explanation": "物理メモリ番地（Physical Memory）の生バイト列を直接ダンプ（PE32+ MZマジックナンバー 0x5A4D の確認）",
            }
        elif cmd == "info pci":
            return {
                "command": command,
                "status": "ok",
                "output": (
                    "  Bus  0, device   0, function 0:\n"
                    "    Host bridge: PCI device 8086:1237\n"
                    "  Bus  0, device   1, function 0:\n"
                    "    ISA bridge: PCI device 8086:7000\n"
                    "  Bus  0, device   2, function 0:\n"
                    "    VGA controller: PCI device 1234:1111\n"
                    "  Bus  0, device   3, function 0:\n"
                    "    USB controller: PCI device 1033:0194 (NEC xHCI USB 3.0)"
                ),
                "explanation": "QEMU仮想マザーボード上のPCIデバイス一覧を表示（xHCIコントローラのBus/Device/Functionの確認）",
            }
        elif cmd == "info block":
            return {
                "command": command,
                "status": "ok",
                "output": "drive0 (#block165): fat:rw:disk_img (raw)\n  ro=0, drv=raw, backing_file=fat:rw:disk_img",
                "explanation": "QEMUにアタッチされている仮想ストレージブロックデバイスの状態を表示",
            }
        elif cmd == "system_reset":
            return {
                "command": command,
                "status": "ok",
                "output": "Resetting guest CPU and motherboard registers...",
                "explanation": "仮想マシン全体を即座にコールドリセットし、Reset Vectorから再ブート",
            }
        elif cmd == "quit" or cmd == "q":
            return {
                "command": command,
                "status": "ok",
                "output": "QEMU emulator process terminating.",
                "explanation": "QEMUプロセスを終了しホストシェルに戻る",
            }
        else:
            return {
                "command": command,
                "status": "unknown",
                "output": f"unknown command: '{command}'",
                "explanation": "未定義のモニタコマンド",
            }


def plot_qemu_ovmf_boot_architecture(output_path: str = "1/result/fig1_7_qemu_ovmf_boot_architecture.png") -> str:
    """Generate high-resolution architecture diagram of QEMU virtual PC & OVMF UEFI emulation."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 9.8), dpi=300)

    # Background Canvas
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.94, facecolor="#F8F9FA", edgecolor="#CFD8DC", lw=2, zorder=0))

    # 1. Host OS Layer (Bottom Container)
    ax.add_patch(Rectangle((0.05, 0.05), 0.90, 0.22, facecolor="#ECEFF1", edgecolor="#90A4AE", lw=1.8, linestyle="--", zorder=1))
    ax.text(0.07, 0.24, "Host Operating System (Linux / WSL2 / macOS)", fontsize=11.5, fontweight="bold", color="#37474F")

    # Host Components
    # Host disk_img Directory
    ax.add_patch(Rectangle((0.08, 0.08), 0.26, 0.14, facecolor="#FFF3E0", edgecolor="#FF9800", lw=1.5, zorder=2))
    ax.text(0.21, 0.18, "Host Directory (disk_img/)", fontsize=10, fontweight="bold", color="#E65100", ha="center")
    ax.text(0.21, 0.12, "disk_img/\n └── EFI/\n      └── BOOT/\n           └── BOOTX64.EFI", fontsize=8.2, family="monospace", color="#5D4037", ha="center")

    # Host OVMF Firmware Files
    ax.add_patch(Rectangle((0.38, 0.08), 0.26, 0.14, facecolor="#E1F5FE", edgecolor="#0288D1", lw=1.5, zorder=2))
    ax.text(0.51, 0.18, "OVMF Firmware Binary Files", fontsize=10, fontweight="bold", color="#01579B", ha="center")
    ax.text(0.51, 0.12, "OVMF_CODE.fd (Code ROM, read-only)\nOVMF_VARS.fd (NVRAM vars, rw)\n(EDK II / TianoCore build)", fontsize=8.2, family="monospace", color="#0D47A1", ha="center")

    # Host Terminal & GUI Window
    ax.add_patch(Rectangle((0.68, 0.08), 0.24, 0.14, facecolor="#EDE7F6", edgecolor="#673AB7", lw=1.5, zorder=2))
    ax.text(0.80, 0.18, "Host Display & I/O", fontsize=10, fontweight="bold", color="#4A148C", ha="center")
    ax.text(0.80, 0.12, "• QEMU GUI Window (GOP display)\n• Terminal stdio (QEMU Monitor)\n• GDB Server (:1234)", fontsize=8.2, color="#311B92", ha="center")

    # 2. QEMU Emulator Process Container (Middle to Top)
    ax.add_patch(Rectangle((0.05, 0.31), 0.90, 0.63, facecolor="#E8EAF6", edgecolor="#3F51B5", lw=2.2, zorder=1))
    ax.text(0.07, 0.91, "QEMU Virtual PC Process (qemu-system-x86_64)", fontsize=13, fontweight="bold", color="#1A237E")

    # Virtual Hardware Box 1: CPU & Memory
    ax.add_patch(Rectangle((0.08, 0.60), 0.40, 0.28, facecolor="#FFFFFF", edgecolor="#3949AB", lw=1.6, zorder=2))
    ax.text(0.10, 0.84, "Virtual CPU & Memory Subsystem", fontsize=11, fontweight="bold", color="#283593")
    ax.text(0.10, 0.77, "• Architecture: x86-64 (qemu64 / host)\n• Modes: Real Mode → Protected Mode → 64-bit Long Mode\n• RAM: 1024 MiB Guest Physical Memory (-m 1024M)\n• MMU: 4-Level Paging (CR3), CR0.PG=1\n• Execution Engine: TCG (JIT) / KVM (Hardware Assisted)", fontsize=8.5, color="#1A237E")

    # Virtual Hardware Box 2: OVMF UEFI Firmware ROM
    ax.add_patch(Rectangle((0.52, 0.60), 0.40, 0.28, facecolor="#E0F2F1", edgecolor="#00897B", lw=1.6, zorder=2))
    ax.text(0.54, 0.84, "OVMF UEFI BIOS Firmware Space", fontsize=11, fontweight="bold", color="#004D40")
    ax.text(0.54, 0.76, "• Parallel Flash 0: OVMF_CODE.fd (Mapped to Top of 4GB Space)\n• Parallel Flash 1: OVMF_VARS.fd (Simulated NVRAM variables)\n• Reset Vector: 0xFFFFFFF0 (Jumps directly into SEC core)\n• Boot Manager (BDS): Scans attached block devices for ESP\n• EFI Protocols: GOP, SimpleFileSystem, BlockIo, xHCI", fontsize=8.5, color="#004D40")

    # Virtual Hardware Box 3: Storage & Virtual FAT Driver
    ax.add_patch(Rectangle((0.08, 0.35), 0.40, 0.22, facecolor="#FFF8E1", edgecolor="#FFA000", lw=1.6, zorder=2))
    ax.text(0.10, 0.53, "Storage Controller: Virtual VFAT", fontsize=11, fontweight="bold", color="#FF6F00")
    ax.text(0.10, 0.45, "• Mode: -drive format=raw,file=fat:rw:disk_img\n• On-the-fly FAT32 Partition & BPB Generation\n• Simulated LBA Sectors (512 bytes / sector)\n• Maps host disk_img/ to EFI System Partition (ESP)\n• Exposes standard IDE / AHCI block interface to OVMF", fontsize=8.5, color="#E65100")

    # Virtual Hardware Box 4: I/O Devices & Monitor
    ax.add_patch(Rectangle((0.52, 0.35), 0.40, 0.22, facecolor="#F3E5F5", edgecolor="#8E24AA", lw=1.6, zorder=2))
    ax.text(0.54, 0.53, "Virtual Devices & Debug Monitors", fontsize=11, fontweight="bold", color="#4A148C")
    ax.text(0.54, 0.45, "• xHCI Controller: NEC uPD720200 USB 3.0 (-device nec-usb-xhci)\n• GOP Display: Linear FrameBuffer (1024x768 32bpp BGR/RGB)\n• QEMU Monitor: -monitor stdio (info registers, xp, info pci)\n• Serial Port: UART 16550A (COM1 at I/O port 0x3F8)\n• Interrupts: Local APIC & IO-APIC Simulation", fontsize=8.5, color="#4A148C")

    # Connecting Arrows
    # Host Dir -> Virtual FAT
    ax.annotate(
        "On-the-fly FAT32 Synthesis\n(Directory-to-Blocks Mapping)",
        xy=(0.28, 0.35), xytext=(0.21, 0.22),
        arrowprops=dict(arrowstyle="->", color="#E65100", lw=2),
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#BF360C"
    )

    # Host Firmware -> OVMF ROM
    ax.annotate(
        "Mapped as Parallel Flash (-drive if=pflash)",
        xy=(0.72, 0.60), xytext=(0.51, 0.22),
        arrowprops=dict(arrowstyle="->", color="#0288D1", lw=2),
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#01579B"
    )

    # QEMU Monitor -> Host Terminal
    ax.annotate(
        "Bidirectional stdio Monitor Stream",
        xy=(0.80, 0.22), xytext=(0.72, 0.35),
        arrowprops=dict(arrowstyle="<->", color="#8E24AA", lw=2),
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#4A148C"
    )

    # Virtual FAT -> OVMF BDS
    ax.annotate(
        "ESP Scan & \\EFI\\BOOT\\BOOTX64.EFI Load",
        xy=(0.52, 0.70), xytext=(0.40, 0.53),
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.2),
        ha="center", va="center", fontsize=9, fontweight="bold", color="#1B5E20"
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.02, 0.96)
    ax.axis("off")
    ax.set_title("QEMU Virtual PC & OVMF UEFI Firmware Architecture\n(Hardware Emulation, Virtual VFAT Storage, and Debugging Infrastructure)", fontsize=13.5, fontweight="bold", color="#0D47A1", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_virtual_fat_and_boot_flow(output_path: str = "1/result/fig1_8_virtual_fat_and_boot_flow.png") -> str:
    """Generate high-resolution visual comparison of Virtual FAT synthesis and UEFI boot sequence."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_fat, ax_boot) = plt.subplots(1, 2, figsize=(16.0, 10.0), dpi=300)

    # -------------------------------------------------------------
    # Left Subplot: Virtual FAT32 On-the-fly Synthesis
    # -------------------------------------------------------------
    ax_fat.set_title("(A) QEMU Virtual FAT (VFAT) On-the-Fly Structure\n(fat:rw:disk_img Directory to FAT32 Sectors)", fontsize=12, fontweight="bold", color="#E65100", pad=12)
    ax_fat.set_xlim(0, 1.0)
    ax_fat.set_ylim(0, 1.0)
    ax_fat.axis("off")

    # Host Directory Box (Top Left)
    ax_fat.add_patch(Rectangle((0.05, 0.72), 0.90, 0.24, facecolor="#FFF3E0", edgecolor="#FF9800", lw=1.8))
    ax_fat.text(0.08, 0.91, "Host Filesystem Directory Hierarchy", fontsize=10.5, fontweight="bold", color="#E65100")
    tree_text = (
        "disk_img/\n"
        " └── EFI/\n"
        "      └── BOOT/\n"
        "           └── BOOTX64.EFI   (PE32+ Executable, ~1.5 KiB)"
    )
    ax_fat.text(0.12, 0.78, tree_text, family="monospace", fontsize=9.2, color="#3E2723")

    # Arrow Down
    ax_fat.annotate(
        "QEMU block/vvfat.c synthesizes FAT32 sectors in RAM",
        xy=(0.50, 0.65), xytext=(0.50, 0.72),
        arrowprops=dict(arrowstyle="->", color="#D84315", lw=2.2),
        ha="center", va="center", fontsize=8.8, fontweight="bold", color="#BF360C"
    )

    # FAT32 Sector Layout (Bottom Left)
    ax_fat.add_patch(Rectangle((0.05, 0.05), 0.90, 0.58, facecolor="#FBE9E7", edgecolor="#D84315", lw=1.8))
    ax_fat.text(0.08, 0.59, "Synthesized Virtual FAT32 Block Device (LBA Sectors)", fontsize=10.5, fontweight="bold", color="#BF360C")

    # Sector Slices
    slices = [
        ("Sector 0", "MBR / Partition Table (ESP Type 0xEF)", "#FFCCBC", 0.51),
        ("Sectors 1–31", "FAT32 Reserved Sectors (BPB, FSInfo)", "#FFE0B2", 0.42),
        ("Sectors 32–N", "FAT #1 (Cluster Allocation Table: 2→3→EOF)", "#FFF9C4", 0.33),
        ("Sectors N–2N", "FAT #2 (Mirrored Backup FAT Table)", "#FFF9C4", 0.24),
        ("Data Area", "Cluster 2: Root Dir ('EFI' dir entry 0x10)\nCluster 3: EFI Dir ('BOOT' dir entry 0x10)\nCluster 4: BOOT Dir ('BOOTX64.EFI' 0x20)\nClusters 5–6: Raw Binary Data of BOOTX64.EFI", "#C8E6C9", 0.10),
    ]

    for label, desc, color, y_pos in slices:
        h = 0.07 if "Data Area" not in label else 0.12
        ax_fat.add_patch(Rectangle((0.08, y_pos), 0.84, h, facecolor=color, edgecolor="#5D4037", lw=1.2))
        ax_fat.text(0.10, y_pos + h - 0.025, label, fontsize=8.8, fontweight="bold", color="#212121")
        ax_fat.text(0.35, y_pos + h/2, desc, fontsize=7.8, family="monospace", color="#37474F", va="center")

    # -------------------------------------------------------------
    # Right Subplot: Step-by-Step UEFI Boot Sequence Flow
    # -------------------------------------------------------------
    ax_boot.set_title("(B) OVMF UEFI Virtual Boot Sequence Flow\n(From Power-On Reset Vector to MikanOS EfiMain)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_boot.set_xlim(0, 1.0)
    ax_boot.set_ylim(0, 1.0)
    ax_boot.axis("off")

    stages = [
        ("1. Reset Vector (0xFFFFFFF0)", "CPU starts execution in Real Mode at top of 4GB.\nOVMF SEC core sets up Cache-As-RAM (CAR).", "#BBDEFB", 0.84),
        ("2. PEI (Pre-EFI Initialization)", "Initializes guest RAM (1024M), detects chipset,\nand transitions CPU to 32-bit Protected Mode.", "#B3E5FC", 0.68),
        ("3. DXE (Driver Execution Env)", "Switches to 64-bit Long Mode (PML4 paging).\nDispatches GOP, PCI, xHCI, and FAT filesystem drivers.", "#B2EBF2", 0.52),
        ("4. BDS (Boot Device Selection)", "Scans virtual devices. Detects Virtual FAT device\nas valid EFI System Partition (ESP).", "#C8E6C9", 0.36),
        ("5. Path Resolution & Relocation", "Finds '\\EFI\\BOOT\\BOOTX64.EFI'. Reads PE32+ header,\nallocates memory pages, and maps sections.", "#FFF59D", 0.20),
        ("6. Jump to EfiMain (MikanOS)", "Calls EfiMain(ImageHandle, SystemTable).\nOutputs 'Hello, world!' to UEFI ConOut console!", "#FFCC80", 0.04),
    ]

    for title, desc, color, y_pos in stages:
        h = 0.12
        ax_boot.add_patch(Rectangle((0.05, y_pos), 0.90, h, facecolor=color, edgecolor="#1565C0", lw=1.4))
        ax_boot.text(0.08, y_pos + 0.085, title, fontsize=9.2, fontweight="bold", color="#0D47A1")
        ax_boot.text(0.08, y_pos + 0.035, desc, fontsize=8.0, color="#263238")

        if y_pos > 0.05:
            # Flow Arrow down
            ax_boot.annotate(
                "",
                xy=(0.50, y_pos), xytext=(0.50, y_pos - 0.035),
                arrowprops=dict(arrowstyle="<-", color="#1565C0", lw=1.8),
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# =============================================================================
# Chapter 1.5: Behind the Scenes Boot Sequence, Reset Vector, and UEFI Phases
# =============================================================================

class CpuOperatingMode(Enum):
    """x86 CPU Operating Modes during System Startup."""
    REAL_MODE_16 = "16-bit Real Mode (with 4GiB CS Base)"
    PROTECTED_MODE_32 = "32-bit Protected Mode (Flat Segmentation)"
    COMPATIBILITY_MODE_64 = "IA-32e Compatibility Mode (32-bit App in Long Mode)"
    LONG_MODE_64 = "64-bit Long Mode (4-Level Paging, PML4)"


class TargetArchitecture(Enum):
    """CPU Target Architectures and UEFI Fallback Executable Specs."""
    X86_64 = ("x86-64 (AMD64 / Intel 64)", "BOOTX64.EFI")
    AARCH64 = ("AArch64 (ARM 64-bit)", "BOOTAA64.EFI")
    IA32 = ("IA-32 (x86 32-bit)", "BOOTIA32.EFI")
    ARM32 = ("ARM 32-bit (Thumb-2)", "BOOTARM.EFI")
    RISCV64 = ("RISC-V 64-bit", "BOOTRISCV64.EFI")

    def __init__(self, description: str, fallback_filename: str):
        self.description = description
        self.fallback_filename = fallback_filename

    @property
    def fallback_path(self) -> str:
        return f"\\EFI\\BOOT\\{self.fallback_filename}"


@dataclass
class CpuResetVector:
    """Hardware Reset State immediately following CPU Power-On (Intel SDM Vol 3A)."""
    rip: int = 0xFFF0
    cs_selector: int = 0xF000
    cs_base: int = 0xFFFF_0000
    cr0: int = 0x6000_0010
    cr4: int = 0x0000_0000
    eflags: int = 0x0000_0002
    edx_bist: int = 0x0000_0000

    @property
    def physical_address(self) -> int:
        """Physical reset vector address: CS_base + RIP = 0xFFFFFFF0 (4GiB - 16 bytes)."""
        return self.cs_base + self.rip

    @property
    def distance_from_4gib(self) -> int:
        """Bytes below top of 32-bit physical address space."""
        return 0x1_0000_0000 - self.physical_address

    def summary(self) -> Dict[str, Any]:
        return {
            "physical_reset_vector": f"0x{self.physical_address:08X}",
            "distance_from_4gib_bytes": self.distance_from_4gib,
            "cs_ip": f"0x{self.cs_selector:04X}:0x{self.rip:04X}",
            "cs_base": f"0x{self.cs_base:08X}",
            "cr0": f"0x{self.cr0:08X}",
            "pe_bit": self.cr0 & 1,
            "cd_bit": (self.cr0 >> 30) & 1,
            "eflags": f"0x{self.eflags:08X}",
            "bist_passed": self.edx_bist == 0,
        }


class UefiPlatformPhase(Enum):
    """UEFI Platform Initialization (PI) Specification Phases."""
    SEC = "SEC (Security Phase)"
    PEI = "PEI (Pre-EFI Initialization Phase)"
    DXE = "DXE (Driver Execution Environment Phase)"
    BDS = "BDS (Boot Device Selection Phase)"
    TSL = "TSL (Transient System Load Phase)"
    RT = "RT (Run Time Phase)"
    AL = "AL (After Life Phase)"


@dataclass
class UefiPhaseDetail:
    """Detailed architectural information for a UEFI boot phase."""
    phase: UefiPlatformPhase
    cpu_mode: CpuOperatingMode
    memory_type: str
    primary_role: str
    key_events: List[str]


def get_uefi_platform_phases() -> List[UefiPhaseDetail]:
    """Returns the architectural details of the 5 primary UEFI boot phases."""
    return [
        UefiPhaseDetail(
            phase=UefiPlatformPhase.SEC,
            cpu_mode=CpuOperatingMode.PROTECTED_MODE_32,
            memory_type="Cache-as-RAM (CAR / L1/L2 temporary cache)",
            primary_role="Root of Trust, CAR setup, transition away from 16-bit Reset Vector",
            key_events=[
                "Fetch first instruction at Reset Vector 0xFFFFFFF0 (4GiB - 16B)",
                "Switch CPU to 32-bit Protected Mode",
                "Configure MTRRs to enable Cache-as-RAM (CAR) before DRAM exists",
                "Hand off control and PPI pointer to PEI Foundation",
            ],
        ),
        UefiPhaseDetail(
            phase=UefiPlatformPhase.PEI,
            cpu_mode=CpuOperatingMode.PROTECTED_MODE_32,
            memory_type="CAR -> Physical DRAM (Memory Controller initialized)",
            primary_role="CPU/Chipset initialization and main DRAM discovery",
            key_events=[
                "Dispatch PEI Modules (PEIMs)",
                "Initialize Memory Controller (DDR timings, SPD, electrical calibration)",
                "Evacuate CAR stack/heap to discovered physical DRAM",
                "Build HOB (Hand-Off Block) list describing system memory and hardware state",
                "Pass HOB list to DXE Foundation",
            ],
        ),
        UefiPhaseDetail(
            phase=UefiPlatformPhase.DXE,
            cpu_mode=CpuOperatingMode.LONG_MODE_64,
            memory_type="Physical DRAM (4-Level Identity Paging enabled, CR3 loaded)",
            primary_role="Switch to 64-bit Long Mode, load device drivers, produce Protocols",
            key_events=[
                "Construct 4-level page tables (PML4 -> PDP -> PD -> PT)",
                "Enable IA32_EFER.LME, CR4.PAE, and CR0.PG to enter 64-bit Long Mode",
                "Initialize UEFI Boot Services and Runtime Services",
                "Dispatch device drivers: PCI Root, xHCI/USB, GOP (Graphics), FAT/Disk",
                "Construct Protocol Database and Device Path tree",
            ],
        ),
        UefiPhaseDetail(
            phase=UefiPlatformPhase.BDS,
            cpu_mode=CpuOperatingMode.LONG_MODE_64,
            memory_type="Physical DRAM (UEFI Boot Services available)",
            primary_role="Enforce boot policy, resolve boot target from NVRAM or Removable Media",
            key_events=[
                "Connect console devices (ConIn, ConOut, ConErr)",
                "Parse NVRAM variables: BootOrder, Boot0000, Boot0001, etc.",
                "Execute Boot Order search; if none match, trigger Removable Media Fallback",
                "Scan FAT volumes for \\EFI\\BOOT\\BOOT{ARCH}.EFI (e.g. BOOTX64.EFI)",
                "Load binary into memory via gBS->LoadImage()",
                "Transfer control via gBS->StartImage()",
            ],
        ),
        UefiPhaseDetail(
            phase=UefiPlatformPhase.TSL,
            cpu_mode=CpuOperatingMode.LONG_MODE_64,
            memory_type="Physical DRAM (EFI System Table & BootServices active)",
            primary_role="Execute OS Loader / UEFI Application (MikanOS BOOTX64.EFI)",
            key_events=[
                "Pass control to EfiMain(ImageHandle, SystemTable)",
                "ABI register setup: RCX=ImageHandle, RDX=SystemTable (Microsoft x64 ABI)",
                "Application draws to screen via ConOut->OutputString() or GOP frame buffer",
                "Application prepares kernel and invokes ExitBootServices()",
            ],
        ),
    ]


@dataclass
class StorageVolume:
    """Mock storage partition representation for boot discovery."""
    label: str
    partition_type: str  # "GPT ESP", "MBR FAT", "Raw"
    fs_type: str         # "FAT32", "FAT16", "FAT12", "NTFS", "EXT4"
    is_removable: bool   # USB drive or virtual FAT
    files: Dict[str, int] = field(default_factory=dict)  # Normalized uppercase path -> size

    def has_file(self, normalized_path: str) -> bool:
        norm = normalized_path.replace("\\", "/").strip("/").upper()
        return norm in self.files


@dataclass
class NvramBootEntry:
    """UEFI NVRAM Boot#### Variable Entry."""
    boot_id: int  # e.g. 0x0001 for Boot0001
    description: str
    target_volume_label: str
    target_file_path: str  # e.g. "\\EFI\\fedora\\grubx64.efi"
    is_enabled: bool = True


@dataclass
class BootResolutionReport:
    """Report generated by UEFI Boot Manager resolving boot targets."""
    success: bool
    selected_volume: Optional[str]
    resolved_file_path: Optional[str]
    resolution_method: str
    log_trace: List[str]


class UefiBootManagerSimulator:
    """Simulates UEFI BDS (Boot Device Selection) decision tree and fallback resolution."""

    def __init__(self, architecture: TargetArchitecture = TargetArchitecture.X86_64):
        self.arch = architecture
        self.volumes: List[StorageVolume] = []
        self.nvram_entries: List[NvramBootEntry] = []

    def add_volume(self, volume: StorageVolume) -> None:
        self.volumes.append(volume)

    def add_nvram_entry(self, entry: NvramBootEntry) -> None:
        self.nvram_entries.append(entry)

    def resolve(self) -> BootResolutionReport:
        logs: List[str] = [
            f"[BDS Phase] Initiating boot resolution for architecture: {self.arch.description}",
            f"[BDS Phase] Standard fallback path: {self.arch.fallback_path}",
        ]

        # 1. NVRAM BootOrder evaluation
        logs.append(f"Step 1: Inspecting NVRAM BootOrder ({len(self.nvram_entries)} configured entries)")
        for entry in self.nvram_entries:
            if not entry.is_enabled:
                logs.append(f"  Skipping disabled entry Boot{entry.boot_id:04X} ({entry.description})")
                continue

            logs.append(f"  Evaluating Boot{entry.boot_id:04X}: {entry.description} -> [{entry.target_volume_label}] {entry.target_file_path}")
            vol = next((v for v in self.volumes if v.label == entry.target_volume_label), None)
            if vol:
                if self._is_uefi_compliant_fs(vol.fs_type) and vol.has_file(entry.target_file_path):
                    logs.append(f"  --> MATCH FOUND on volume [{vol.label}]: {entry.target_file_path}")
                    return BootResolutionReport(
                        success=True,
                        selected_volume=vol.label,
                        resolved_file_path=entry.target_file_path,
                        resolution_method=f"NVRAM BootOrder (Boot{entry.boot_id:04X}: {entry.description})",
                        log_trace=logs,
                    )
                else:
                    logs.append(f"  Target file not found or incompatible FS on [{vol.label}]")
            else:
                logs.append(f"  Target volume [{entry.target_volume_label}] is offline or detached")

        # 2. Removable Media Fallback (UEFI Spec Section 3.5.1.1)
        logs.append("Step 2: No bootable NVRAM target succeeded. Initiating Removable Media Fallback...")
        fallback_path = self.arch.fallback_path

        for vol in self.volumes:
            logs.append(f"  Probing volume [{vol.label}] (Type: {vol.partition_type}, FS: {vol.fs_type}, Removable: {vol.is_removable})")
            if not self._is_uefi_compliant_fs(vol.fs_type):
                logs.append(f"    Ignored: File system {vol.fs_type} is not UEFI-compliant (FAT12/16/32 required)")
                continue

            if vol.has_file(fallback_path):
                logs.append(f"    --> SUCCESS: Located canonical fallback binary '{fallback_path}' on [{vol.label}]")
                return BootResolutionReport(
                    success=True,
                    selected_volume=vol.label,
                    resolved_file_path=fallback_path,
                    resolution_method=f"Removable Media Fallback ({fallback_path})",
                    log_trace=logs,
                )
            else:
                logs.append(f"    Fallback binary '{fallback_path}' not present on [{vol.label}]")

        logs.append("[BDS Error] No valid boot device or fallback binary could be resolved.")
        return BootResolutionReport(
            success=False,
            selected_volume=None,
            resolved_file_path=None,
            resolution_method="None (Failed)",
            log_trace=logs,
        )

    @staticmethod
    def _is_uefi_compliant_fs(fs_type: str) -> bool:
        return fs_type.upper() in {"FAT32", "FAT16", "FAT12"}


# =============================================================================
# Visualizations for Chapter 1.5
# =============================================================================

def plot_cpu_reset_to_uefi_phases(output_path: str = "1/result/fig1_9_cpu_reset_to_uefi_phases.png") -> str:
    """
    Renders high-resolution architectural diagram:
    (A) x86-64 CPU Reset State & Reset Vector at 0xFFFFFFF0 (4GiB - 16B)
    (B) UEFI PI 5-Phase Transition Flow (SEC -> PEI -> DXE -> BDS -> TSL)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_plot_style()

    fig, (ax_reset, ax_phases) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor("#FAFAFA")

    # -------------------------------------------------------------
    # Left Subplot: CPU Power-On Reset & Memory Map (Top of 4GiB)
    # -------------------------------------------------------------
    ax_reset.set_title("(A) x86-64 CPU Power-On Reset Vector\n(4GiB Address Space & SPI Flash ROM Mapping)", fontsize=12, fontweight="bold", color="#B71C1C", pad=12)
    ax_reset.set_xlim(0, 1.0)
    ax_reset.set_ylim(0, 1.0)
    ax_reset.axis("off")

    # 4GiB Physical Address Column
    ax_reset.add_patch(Rectangle((0.10, 0.08), 0.35, 0.82, facecolor="#ECEFF1", edgecolor="#37474F", lw=1.8))
    ax_reset.text(0.275, 0.87, "32-bit Physical Address Space\n(0x00000000 - 0xFFFFFFFF: 4 GiB)", fontsize=8.5, fontweight="bold", ha="center", color="#263238")

    # Flash ROM Area (Top 16MB)
    ax_reset.add_patch(Rectangle((0.10, 0.72), 0.35, 0.14, facecolor="#FFCDD2", edgecolor="#C62828", lw=1.5))
    ax_reset.text(0.275, 0.79, "SPI Flash ROM (OVMF BIOS)\n(e.g., 0xFF000000 - 0xFFFFFFFF)", fontsize=7.8, fontweight="bold", ha="center", color="#B71C1C")

    # The 16-byte Reset Vector Block
    ax_reset.add_patch(Rectangle((0.10, 0.82), 0.35, 0.035, facecolor="#D50000", edgecolor="#000000", lw=2.0))
    ax_reset.text(0.275, 0.835, "★ Reset Vector (16B)\n0xFFFFFFF0", fontsize=7.5, fontweight="bold", ha="center", color="#FFFFFF")

    # Normal RAM
    ax_reset.add_patch(Rectangle((0.10, 0.12), 0.35, 0.58, facecolor="#C8E6C9", edgecolor="#2E7D32", lw=1.2))
    ax_reset.text(0.275, 0.40, "System DRAM (Main Memory)\n(Uninitialized at Power-On)", fontsize=8.0, fontweight="bold", ha="center", color="#1B5E20")

    # Address labels
    ax_reset.text(0.08, 0.86, "0xFFFFFFFF (4GiB)", fontsize=7.5, family="monospace", ha="right", va="center")
    ax_reset.text(0.08, 0.825, "0xFFFFFFF0", fontsize=7.5, family="monospace", ha="right", va="center", color="#D50000", fontweight="bold")
    ax_reset.text(0.08, 0.72, "0xFF000000", fontsize=7.5, family="monospace", ha="right", va="center")
    ax_reset.text(0.08, 0.12, "0x00000000 (0B)", fontsize=7.5, family="monospace", ha="right", va="center")

    # Register State Box on the Right of Left Subplot
    reg_y = 0.52
    ax_reset.add_patch(FancyBboxPatch((0.50, reg_y), 0.46, 0.38, boxstyle="round,pad=0.02", facecolor="#FFF9C4", edgecolor="#F57F17", lw=1.6))
    ax_reset.text(0.52, reg_y + 0.35, "CPU Initial Register State (Reset):", fontsize=8.8, fontweight="bold", color="#E65100")
    regs = [
        ("RIP (Instruction Pointer)", "0x0000FFF0 (16-bit)"),
        ("CS Selector / Base", "0xF000 / 0xFFFF0000"),
        ("Effective Reset Address", "0xFFFFFFF0 (Base + RIP)"),
        ("CR0 Control Register", "0x60000010 (PE=0 RealMode)"),
        ("EFLAGS", "0x00000002 (Reserved bit 1)"),
        ("EDX (BIST Result)", "0x00000000 (0 = Success)"),
    ]
    for idx, (lbl, val) in enumerate(regs):
        y_line = reg_y + 0.30 - (idx * 0.052)
        ax_reset.text(0.52, y_line, f"• {lbl}:", fontsize=7.5, fontweight="bold", color="#37474F")
        ax_reset.text(0.94, y_line, val, fontsize=7.5, family="monospace", ha="right", color="#B71C1C" if "Effective" in lbl else "#0D47A1")

    # Callout Arrow pointing to Reset Vector
    ax_reset.annotate(
        "First instruction executed by CPU\n(JMP to 32-bit SEC entry point)",
        xy=(0.45, 0.835), xytext=(0.52, 0.44),
        arrowprops=dict(arrowstyle="->", color="#D50000", lw=1.8),
        fontsize=7.8, fontweight="bold", color="#B71C1C",
    )

    # Key Architectural Takeaway
    ax_reset.add_patch(FancyBboxPatch((0.10, 0.02), 0.86, 0.08, boxstyle="square,pad=0.01", facecolor="#E1F5FE", edgecolor="#0288D1", lw=1.2))
    ax_reset.text(0.53, 0.06, "[POINT] なぜメモリの最上端 (4GiB - 16B) なのか？\nx86 CPUは互換性のため 16-bit リアルモードで起動するが、CSベースに 0xFFFF0000 が隠し設定され、\nFlash ROM の最上部に配置されたリセットコードに確実にジャンプできるように設計されている。", fontsize=7.4, ha="center", va="center", color="#01579B")


    # -------------------------------------------------------------
    # Right Subplot: UEFI Platform Initialization (PI) 5-Phase Flow
    # -------------------------------------------------------------
    ax_phases.set_title("(B) UEFI Platform Initialization (PI) 5-Phase Architecture\n(From Hardware Root of Trust to MikanOS EfiMain)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_phases.set_xlim(0, 1.0)
    ax_phases.set_ylim(0, 1.0)
    ax_phases.axis("off")

    phases_data = [
        ("1. SEC (Security Phase)", "32-bit Protected Mode", "Cache-as-RAM (CAR)", "#FFCDD2", "#B71C1C", 0.82,
         "• First instruction at 0xFFFFFFF0\n• Hardware root of trust & CAR init (No DRAM yet)\n• Passes temporary stack pointer & PPIs to PEI"),
        ("2. PEI (Pre-EFI Initialization)", "32-bit Protected Mode", "CAR -> Physical DRAM", "#FFE0B2", "#E65100", 0.63,
         "• Dispatches PEI Modules (PEIMs)\n• Initializes Memory Controller & DRAM hardware\n• Builds HOB (Hand-Off Block) list describing system memory"),
        ("3. DXE (Driver Execution Environment)", "64-bit Long Mode (PML4)", "Main DRAM (Identity Paging)", "#C8E6C9", "#1B5E20", 0.44,
         "• Switches CPU to 64-bit Long Mode (CR3 / Paging)\n• Loads core protocols: GOP (Graphics), PCI, USB xHCI, FAT\n• Constructs UEFI Protocol Database"),
        ("4. BDS (Boot Device Selection)", "64-bit Long Mode", "Main DRAM (Boot Services)", "#BBDEFB", "#0D47A1", 0.25,
         "• Reads NVRAM BootOrder variables\n• Scans FAT partitions for \\EFI\\BOOT\\BOOTX64.EFI fallback\n• Loads PE32+ image into memory via LoadImage()"),
        ("5. TSL (Transient System Load)", "64-bit Long Mode", "Main DRAM (Application Space)", "#E1BEE7", "#4A148C", 0.06,
         "• Calls EfiMain(ImageHandle, SystemTable) via Microsoft x64 ABI\n• MikanOS outputs 'Hello, world!' via ConOut->OutputString\n• Prepares for ExitBootServices() transfer of control"),
    ]

    for title, cpu_mode, mem_type, bg_color, border_color, y_pos, desc in phases_data:
        h = 0.155
        ax_phases.add_patch(FancyBboxPatch((0.04, y_pos), 0.92, h, boxstyle="round,pad=0.015", facecolor=bg_color, edgecolor=border_color, lw=1.6))
        ax_phases.text(0.07, y_pos + h - 0.03, title, fontsize=9.2, fontweight="bold", color=border_color)
        ax_phases.text(0.93, y_pos + h - 0.03, f"[{cpu_mode} | {mem_type}]", fontsize=7.4, family="monospace", ha="right", color="#37474F")
        ax_phases.text(0.07, y_pos + 0.02, desc, fontsize=7.6, color="#212121", va="bottom")

        if y_pos > 0.08:
            ax_phases.annotate(
                "",
                xy=(0.50, y_pos), xytext=(0.50, y_pos - 0.025),
                arrowprops=dict(arrowstyle="<-", color=border_color, lw=2.0),
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_uefi_boot_device_and_file_resolution(output_path: str = "1/result/fig1_10_uefi_boot_device_and_file_resolution.png") -> str:
    """
    Renders high-resolution diagram:
    (A) UEFI Boot Manager Decision Tree (NVRAM BootOrder vs Removable Media Fallback)
    (B) Standard Architecture Fallback Naming & Folder Structure per UEFI Spec 3.5
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_plot_style()

    fig, (ax_tree, ax_arch) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#FAFAFA")

    # -------------------------------------------------------------
    # Left Subplot: BDS Decision Tree Flowchart
    # -------------------------------------------------------------
    ax_tree.set_title("(A) UEFI BDS Boot Target Resolution Decision Tree\n(How UEFI Discovers \\EFI\\BOOT\\BOOTX64.EFI)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax_tree.set_xlim(0, 1.0)
    ax_tree.set_ylim(0, 1.0)
    ax_tree.axis("off")

    nodes = [
        ("1. BDS Phase Starts", "Read NVRAM BootOrder variable (e.g. Boot0001, Boot0002)", "#E3F2FD", "#1565C0", 0.88, 0.10),
        ("2. Iterate BootOrder Entries", "Check each configured OS device path & executable", "#BBDEFB", "#0D47A1", 0.73, 0.10),
        ("Condition: Bootable NVRAM target found?", "YES: LoadImage & StartImage() -> Boot installed OS\nNO: Move to Removable Media Fallback", "#FFF9C4", "#F57F17", 0.56, 0.12),
        ("3. Removable Media Scan", "Enumerate block devices: USB sticks, CD/DVD, Virtual FAT", "#C8E6C9", "#2E7D32", 0.39, 0.10),
        ("4. Partition & FAT32 Verification", "Verify GPT ESP (EFI System Partition) or MBR with FAT12/16/32", "#E8F5E9", "#1B5E20", 0.24, 0.10),
        ("5. Standard Path Fallback", "Locate '\\EFI\\BOOT\\BOOTX64.EFI' -> Load into memory & call EfiMain", "#FFE0B2", "#E65100", 0.08, 0.11),
    ]

    for title, desc, bg, border, y_pos, h in nodes:
        ax_tree.add_patch(FancyBboxPatch((0.08, y_pos), 0.84, h, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=border, lw=1.6))
        ax_tree.text(0.11, y_pos + h - 0.028, title, fontsize=9.2, fontweight="bold", color=border)
        ax_tree.text(0.11, y_pos + 0.02, desc, fontsize=7.8, color="#212121", va="bottom")

        if y_pos > 0.10:
            ax_tree.annotate(
                "",
                xy=(0.50, y_pos), xytext=(0.50, y_pos - 0.04),
                arrowprops=dict(arrowstyle="<-", color=border, lw=2.0),
            )

    # -------------------------------------------------------------
    # Right Subplot: Architecture-Specific Fallback Matrix & FAT Folder Tree
    # -------------------------------------------------------------
    ax_arch.set_title("(B) UEFI Specification Section 3.5 Fallback Spec\n(Architecture Naming Matrix & FAT Directory Tree)", fontsize=12, fontweight="bold", color="#B71C1C", pad=12)
    ax_arch.set_xlim(0, 1.0)
    ax_arch.set_ylim(0, 1.0)
    ax_arch.axis("off")

    # Table of Architectures and Fallback Names
    ax_arch.add_patch(Rectangle((0.04, 0.54), 0.92, 0.42, facecolor="#F5F5F5", edgecolor="#616161", lw=1.4))
    ax_arch.text(0.06, 0.92, "UEFI Default Fallback Executable Names (UEFI Spec 3.5.1):", fontsize=8.8, fontweight="bold", color="#212121")

    headers = ("Target CPU Architecture", "Default Fallback Filename", "Status")
    ax_arch.text(0.06, 0.86, headers[0], fontsize=8.0, fontweight="bold", color="#424242")
    ax_arch.text(0.55, 0.86, headers[1], fontsize=8.0, fontweight="bold", color="#424242")
    ax_arch.text(0.85, 0.86, headers[2], fontsize=8.0, fontweight="bold", color="#424242")
    ax_arch.axhline(0.845, xmin=0.06, xmax=0.94, color="#9E9E9E", lw=1.0)

    rows = [
        ("x86-64 / AMD64 (Intel 64)", "\\EFI\\BOOT\\BOOTX64.EFI", "MikanOS Target", "#D50000", True),
        ("AArch64 (ARM 64-bit)", "\\EFI\\BOOT\\BOOTAA64.EFI", "Raspberry Pi / Mac", "#0D47A1", False),
        ("IA-32 (x86 32-bit Legacy)", "\\EFI\\BOOT\\BOOTIA32.EFI", "Early Intel Tablets", "#455A64", False),
        ("ARM 32-bit (Thumb-2)", "\\EFI\\BOOT\\BOOTARM.EFI", "Embedded ARM", "#455A64", False),
        ("RISC-V 64-bit", "\\EFI\\BOOT\\BOOTRISCV64.EFI", "Next-Gen Open ISA", "#2E7D32", False),
    ]

    for idx, (arch_str, path_str, note, col, highlight) in enumerate(rows):
        y_r = 0.79 - (idx * 0.055)
        if highlight:
            ax_arch.add_patch(Rectangle((0.05, y_r - 0.015), 0.90, 0.048, facecolor="#FFEBEE", edgecolor="#EF5350", lw=1.0))
        ax_arch.text(0.06, y_r, arch_str, fontsize=7.8, fontweight="bold" if highlight else "normal", color=col)
        ax_arch.text(0.55, y_r, path_str, fontsize=7.6, family="monospace", fontweight="bold" if highlight else "normal", color=col)
        ax_arch.text(0.85, y_r, note, fontsize=7.4, color=col)

    # FAT Folder Tree Diagram
    ax_arch.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.46, boxstyle="round,pad=0.015", facecolor="#E8EAF6", edgecolor="#3949AB", lw=1.6))
    ax_arch.text(0.07, 0.46, "FAT32 USB / ESP Directory Hierarchy:", fontsize=9.0, fontweight="bold", color="#1A237E")

    tree_text = (
        "/ (FAT32 Volume Root)\n"
        " └── EFI/                             (Directory entry: ATTR_DIRECTORY 0x10)\n"
        "      └── BOOT/                       (Standard UEFI bootloader container)\n"
        "           ├── BOOTX64.EFI            ★ [MikanOS Boot Application] (PE32+ Executable)\n"
        "           │    ├── PE32+ Header       (Machine: 0x8664 AMD64, Subsystem: 0x0A UEFI App)\n"
        "           │    ├── Section .text      (x86-64 machine instructions: EfiMain)\n"
        "           │    └── Section .data      (UTF-16 'Hello, world!' string & literals)\n"
        "           └── (BOOTAA64.EFI)         (Optional for multi-arch bootable media)"
    )
    ax_arch.text(0.07, 0.42, tree_text, fontsize=7.6, family="monospace", color="#263238", va="top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path










class MutationOutcome(Enum):
    """Result status of a binary mutation experiment in UEFI execution."""
    SUCCESS = "SUCCESS"                               # Expected display, clean execution
    TRUNCATED_VALID = "TRUNCATED_VALID"               # Properly shortened string with UTF-16 null termination
    NULL_TERMINATOR_MISSING = "NULL_TERMINATOR_MISSING" # Remnant memory displayed due to missing null terminator
    SECTION_OVERFLOW = "SECTION_OVERFLOW"             # Exceeded raw section bounds or damaged headers
    EXECUTION_CRASH = "EXECUTION_CRASH"               # CPU exception / crash from broken loop or stack corruption

@dataclass
class BinaryPatch:
    """Represents a single binary patch applied to an executable."""
    offset: int
    original_bytes: bytes
    patched_bytes: bytes
    description: str

    @property
    def size(self) -> int:
        return len(self.patched_bytes)

@dataclass
class BinaryDiffEntry:
    """Represents a byte-level difference between original and mutated binaries."""
    offset: int
    original_byte: int
    patched_byte: int

    def __str__(self) -> str:
        return f"0x{self.offset:04X}: 0x{self.original_byte:02X} -> 0x{self.patched_byte:02X}"

class BinaryMutator:
    """
    Simulates and manages byte-level mutations and patches on UEFI PE32+ executables (e.g. BOOTX64.EFI).
    Provides analysis of string modifications, opcode changes, null termination, and section boundaries.
    """
    def __init__(self, data: bytes | bytearray):
        self.original_data: bytes = bytes(data)
        self.data: bytearray = bytearray(data)
        self.patches: List[BinaryPatch] = []

    @property
    def current_bytes(self) -> bytes:
        return bytes(self.data)

    def find_bytes(self, pattern: bytes) -> List[int]:
        """Locates all byte offsets matching the given pattern."""
        offsets = []
        pos = 0
        while True:
            idx = self.data.find(pattern, pos)
            if idx == -1:
                break
            offsets.append(idx)
            pos = idx + 1
        return offsets

    def find_utf16le(self, text: str) -> List[int]:
        """Locates all byte offsets containing the UTF-16LE encoded string."""
        pattern = text.encode("utf-16le")
        return self.find_bytes(pattern)

    @staticmethod
    def encode_utf16le(text: str, null_terminate: bool = True) -> bytes:
        """Encodes text into UTF-16LE, optionally adding a 2-byte null terminator (0x00, 0x00)."""
        raw = text.encode("utf-16le")
        if null_terminate:
            raw += b"\x00\x00"
        return raw

    def decode_utf16le(self, offset: int, max_chars: int = 128) -> Tuple[str, bool]:
        """
        Decodes UTF-16LE string from offset until a null terminator (0x0000) or max_chars.
        Returns (decoded_string, is_null_terminated).
        """
        chars = []
        is_terminated = False
        limit = min(offset + max_chars * 2, len(self.data) - 1)
        for ptr in range(offset, limit, 2):
            if ptr + 1 >= len(self.data):
                break
            code_unit = self.data[ptr] | (self.data[ptr + 1] << 8)
            if code_unit == 0:
                is_terminated = True
                break
            chars.append(chr(code_unit))
        return "".join(chars), is_terminated

    def apply_patch(self, offset: int, patch_bytes: bytes, description: str = "") -> BinaryPatch:
        """Applies a patch at the specified byte offset."""
        if offset + len(patch_bytes) > len(self.data):
            # If extending beyond file length, extend bytearray
            extension = (offset + len(patch_bytes)) - len(self.data)
            self.data.extend(b"\x00" * extension)

        orig = bytes(self.data[offset : offset + len(patch_bytes)])
        self.data[offset : offset + len(patch_bytes)] = patch_bytes
        patch = BinaryPatch(
            offset=offset,
            original_bytes=orig,
            patched_bytes=patch_bytes,
            description=description,
        )
        self.patches.append(patch)
        return patch

    def diff(self) -> List[BinaryDiffEntry]:
        """Computes list of differences between the original binary and the currently mutated binary."""
        diffs = []
        max_len = max(len(self.original_data), len(self.data))
        for i in range(max_len):
            orig = self.original_data[i] if i < len(self.original_data) else 0x00
            cur = self.data[i] if i < len(self.data) else 0x00
            if orig != cur:
                diffs.append(BinaryDiffEntry(offset=i, original_byte=orig, patched_byte=cur))
        return diffs

    def format_diff_table(self, max_entries: int = 25) -> str:
        """Formats the diff as an ASCII comparison table."""
        diffs = self.diff()
        if not diffs:
            return "No binary modifications (Binaries are identical)."
        lines = [
            "Offset (Hex) | Original Byte | Patched Byte | Description",
            "-------------+---------------+--------------+----------------------"
        ]
        for d in diffs[:max_entries]:
            orig_char = chr(d.original_byte) if 32 <= d.original_byte <= 126 else "."
            patch_char = chr(d.patched_byte) if 32 <= d.patched_byte <= 126 else "."
            lines.append(f"0x{d.offset:06X}   | 0x{d.original_byte:02X} ('{orig_char}')   | 0x{d.patched_byte:02X} ('{patch_char}')   | diff")
        if len(diffs) > max_entries:
            lines.append(f"... and {len(diffs) - max_entries} more byte differences.")
        return "\n".join(lines)

    def simulate_execution(self) -> Dict[str, Any]:
        """
        Simulates UEFI firmware execution of this binary, evaluating:
        1. String modification at .data/.rdata (0x400)
        2. Null termination integrity
        3. Machine code infinite loop at .text (0x212: EB FE)
        4. Section bounds and file size limits
        """
        result: Dict[str, Any] = {
            "outcome": MutationOutcome.SUCCESS,
            "screen_output": "",
            "halted": True,
            "cpu_state": "HALTED (Looping at 0x212)",
            "message": "",
        }

        # Check section boundary (original size 1536 bytes / 0x600)
        if len(self.data) > 1536:
            result["outcome"] = MutationOutcome.SECTION_OVERFLOW
            result["halted"] = False
            result["cpu_state"] = "CRASH / UEFI_LOAD_ERROR"
            result["message"] = (
                f"Binary size {len(self.data)} bytes exceeds raw section/file capacity (1536 bytes). "
                f"UEFI BDS loader rejects the malformed PE32+ image or corrupts adjacent memory."
            )
            return result

        # Check .text loop integrity: hello.efi original has 'jmp $' (0xEB, 0xFE) at offset 0x212
        text_loop_bytes = bytes(self.data[0x212:0x214]) if len(self.data) >= 0x214 else b""
        if text_loop_bytes != b"\xeb\xfe":
            # Infinite loop modified (e.g. replaced with NOP 0x90 0x90 or RET 0xC3)
            result["outcome"] = MutationOutcome.EXECUTION_CRASH
            result["halted"] = False
            result["cpu_state"] = "CPU_FAULT (Stack imbalance / GP exception / Triple Fault)"
            result["message"] = (
                f"Infinite loop 'jmp $' (EB FE) at offset 0x212 was overwritten with {text_loop_bytes.hex()}. "
                f"The CPU executed past the loop. Because EfiMain does not properly balance the stack "
                f"(missing 'add rsp, 0x28' before 'ret'), returning crashes or falls into unmapped padding!"
            )
            decoded, terminated = self.decode_utf16le(0x400, max_chars=32)
            result["screen_output"] = decoded
            return result

        # Analyze string at .data/.rdata offset 0x400
        str_offset = 0x400
        decoded, terminated = self.decode_utf16le(str_offset, max_chars=64)
        result["screen_output"] = decoded

        # Check if null terminator is present within the original string footprint (0x400..0x41C)
        if not terminated:
            result["outcome"] = MutationOutcome.NULL_TERMINATOR_MISSING
            result["message"] = (
                f"String at 0x{str_offset:X} lacks a 2-byte null terminator (0x0000). "
                f"UEFI OutputString continues reading memory beyond the intended string until it encounters 0x0000, "
                f"causing memory artifacts/garbage to be displayed on screen!"
            )
        else:
            orig_decoded, _ = BinaryMutator(self.original_data).decode_utf16le(str_offset)
            if decoded == orig_decoded:
                result["outcome"] = MutationOutcome.SUCCESS
                result["message"] = "Original string displayed cleanly."
            elif len(decoded) < len(orig_decoded):
                result["outcome"] = MutationOutcome.TRUNCATED_VALID
                result["message"] = f"String shortened to '{decoded}' and properly terminated with 0x0000."
            else:
                result["outcome"] = MutationOutcome.SUCCESS
                result["message"] = f"String successfully mutated to '{decoded}'."

        return result


def plot_hands_on_experiment_cycle(output_path: str = "1/result/fig1_11_hands_on_experiment_cycle.png") -> str:
    """
    Renders high-resolution architectural diagram:
    (A) Low-Level Development OODA Experiment Loop (Observe -> Orient -> Decide -> Act)
    (B) Top-Down Theory vs Bottom-Up Empirical Learning Comparison
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_plot_style()

    fig, (ax_loop, ax_comp) = plt.subplots(1, 2, figsize=(16, 8.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor("#FAFAFA")

    # -------------------------------------------------------------
    # Left Subplot: OODA Loop in Low-Level / OS Development
    # -------------------------------------------------------------
    ax_loop.set_title("(A) Low-Level Systems Learning: Empirical OODA Loop\n(Hands-On Hypothesis-Testing vs Black-Box Fear)",
                     fontsize=11.5, fontweight="bold", pad=14, color="#1A237E")
    ax_loop.set_xlim(0, 10)
    ax_loop.set_ylim(0, 10)
    ax_loop.axis("off")

    # Loop center box
    ax_loop.add_patch(FancyBboxPatch((3.4, 3.6), 3.2, 2.4, boxstyle="round,pad=0.15",
                                     facecolor="#E8EAF6", edgecolor="#3949AB", lw=2.0))
    ax_loop.text(5.0, 5.1, "Core Mindset:\n'Run It First, Observe,\nThen Deconstruct'",
                 fontsize=9.2, fontweight="bold", color="#1A237E", ha="center", va="center")
    ax_loop.text(5.0, 4.0, "(スモールステップ実験の原則)", fontsize=8.0, color="#5C6BC0", ha="center")

    # 4 Nodes: Observe, Orient, Decide, Act
    nodes = [
        ("1. 観察 (Observe)", "Screen Output, Registers,\nHex Dump, QEMU Monitor", 5.0, 8.6, "#E3F2FD", "#1565C0"),
        ("2. 方向付け (Orient)", "Form Hypothesis:\nUTF-16, PE Layout, Calling Conv.", 8.4, 4.8, "#E8F5E9", "#2E7D32"),
        ("3. 意思決定 (Decide)", "Design Mutation Patch:\nOffset, Bytes, Null Terminator", 5.0, 1.0, "#FFF3E0", "#E65100"),
        ("4. 行動 (Act / Patch)", "Binary Hex Edit, Compile,\nQEMU Virtual Boot Execution", 1.6, 4.8, "#F3E5F5", "#6A1B9A"),
    ]

    for title, desc, x, y, fc, ec in nodes:
        ax_loop.add_patch(FancyBboxPatch((x - 1.45, y - 0.75), 2.9, 1.5, boxstyle="round,pad=0.1",
                                         facecolor=fc, edgecolor=ec, lw=1.8))
        ax_loop.text(x, y + 0.35, title, fontsize=9.2, fontweight="bold", color=ec, ha="center", va="center")
        ax_loop.text(x, y - 0.25, desc, fontsize=7.6, color="#37474F", ha="center", va="center")

    # Directed curved arrows forming circular loop
    arrow_style = ArrowStyle("->", head_length=5, head_width=3)
    ax_loop.annotate("", xy=(7.2, 5.5), xytext=(6.2, 8.0),
                     arrowprops=dict(arrowstyle=arrow_style, color="#1E88E5", lw=2.2, connectionstyle="arc3,rad=-0.2"))
    ax_loop.annotate("", xy=(6.2, 1.8), xytext=(7.4, 4.0),
                     arrowprops=dict(arrowstyle=arrow_style, color="#43A047", lw=2.2, connectionstyle="arc3,rad=-0.2"))
    ax_loop.annotate("", xy=(2.8, 4.0), xytext=(3.8, 1.8),
                     arrowprops=dict(arrowstyle=arrow_style, color="#FB8C00", lw=2.2, connectionstyle="arc3,rad=-0.2"))
    ax_loop.annotate("", xy=(3.8, 8.0), xytext=(2.6, 5.5),
                     arrowprops=dict(arrowstyle=arrow_style, color="#8E24AA", lw=2.2, connectionstyle="arc3,rad=-0.2"))

    # -------------------------------------------------------------
    # Right Subplot: Learning Methodology Comparison
    # -------------------------------------------------------------
    ax_comp.set_title("(B) Learning Philosophy Comparison:\nTop-Down Theory vs Hands-On Empirical Loop",
                     fontsize=11.5, fontweight="bold", pad=14, color="#1A237E")
    ax_comp.axis("off")

    ax_comp.add_patch(FancyBboxPatch((0.05, 0.54), 0.90, 0.40, boxstyle="round,pad=0.02",
                                     facecolor="#FFEBEE", edgecolor="#C62828", lw=1.6))
    ax_comp.text(0.08, 0.89, "[×] 理論完全読破型 (Top-Down Theory-First Approach)",
                 fontsize=9.8, fontweight="bold", color="#B71C1C")
    comp_theory = (
        "• 特徴: Intel SDM や UEFI 仕様書（数千ページ）をすべて理解してから実装しようとする\n"
        "• 課題: 抽象的な仕様と実機の挙動が結びつかず、途中で挫折しやすい（情報過多）\n"
        "• フィードバック遅延: 初めてコードを動かすまでに膨大な時間がかかり、疑問が未解決のまま蓄積\n"
        "• デバッグ耐性: 想定外のエラー（ハング・黒画面）が発生した際に原因の切り分けが極めて困難"
    )
    ax_comp.text(0.08, 0.85, comp_theory, fontsize=8.2, color="#37474F", va="top", linespacing=1.4)

    ax_comp.add_patch(FancyBboxPatch((0.05, 0.06), 0.90, 0.44, boxstyle="round,pad=0.02",
                                     facecolor="#E8F5E9", edgecolor="#2E7D32", lw=2.0))
    ax_comp.text(0.08, 0.45, "[✓] 実験・観察駆動型 (Hands-On Empirical OODA Approach)",
                 fontsize=9.8, fontweight="bold", color="#1B5E20")
    comp_hands_on = (
        "• 特徴: 最小のバイナリを手動で動かし、一部のバイトを書き換えて挙動の変化を観察する\n"
        "• 強み: 文字列長やヌル終端、機械語ループを壊すことで『なぜ動いていたのか』を身体感覚で体得\n"
        "• 高速フィードバック: 数秒でパッチ→QEMU起動→結果確認のサイクルを回し、仮説を検証\n"
        "• デバッグ耐性: 意図的に壊して壊れ方を観察しているため、本番でのハングや例外にも冷静に対処可能"
    )
    ax_comp.text(0.08, 0.41, comp_hands_on, fontsize=8.2, color="#37474F", va="top", linespacing=1.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_binary_mutation_anatomy(output_path: str = "1/result/fig1_12_binary_mutation_anatomy.png") -> str:
    """
    Renders high-resolution architectural diagram:
    (A) PE32+ Binary Layout & Mutation Injection Targets (DOS, PE, .text, .rdata)
    (B) 4 Key Mutation Experiment Scenarios & Memory Corruption Manifestations
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_plot_style()

    fig, (ax_mem, ax_cases) = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor("#FAFAFA")

    # -------------------------------------------------------------
    # Left Subplot: PE32+ Binary Structure & Mutation Target Offsets
    # -------------------------------------------------------------
    ax_mem.set_title("(A) PE32+ (hello.efi) Binary Layout\n& Mutation Target Offsets",
                     fontsize=11.5, fontweight="bold", pad=14, color="#1A237E")
    ax_mem.set_xlim(0, 10)
    ax_mem.set_ylim(0, 10)
    ax_mem.axis("off")

    sections = [
        ("0x000 - 0x1FF (512B)", "DOS & PE32+ Headers", "Magic 'MZ', 'PE\\0\\0', Optional Header,\nSection Table (.text, .data)", "#ECEFF1", "#455A64"),
        ("0x200 - 0x3FF (512B)", "Section .text (Code)", "x86-64 Machine Instructions\nEfiMain Entrypoint at 0x200\n★ Loop: 0x212 (EB FE) 'jmp $'", "#E1F5FE", "#0288D1"),
        ("0x400 - 0x5FF (512B)", "Section .data / .rdata", "Read-only Literals & Strings\n★ String: 0x400 ('Hello, world!\\0')\nPadding: 0x41C..0x5FF (Zeroes)", "#E8F5E9", "#388E3C"),
    ]

    y_start = 8.8
    for addr_range, title, desc, fc, ec in sections:
        ax_mem.add_patch(FancyBboxPatch((0.8, y_start - 2.1), 8.4, 2.0, boxstyle="round,pad=0.02",
                                        facecolor=fc, edgecolor=ec, lw=1.8))
        ax_mem.text(1.1, y_start - 0.4, f"{title}  [{addr_range}]", fontsize=9.2, fontweight="bold", color=ec)
        ax_mem.text(1.1, y_start - 1.1, desc, fontsize=7.8, color="#263238", va="top", linespacing=1.3)
        y_start -= 2.6

    ax_mem.annotate("Target 1: Infinite Loop (0x212)\nOpcode: EB FE (jmp $)",
                    xy=(9.2, 5.0), xytext=(5.5, 2.7),
                    arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.8),
                    fontsize=8.0, fontweight="bold", color="#B71C1C",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#D32F2F"))

    ax_mem.annotate("Target 2: UTF-16 String (0x400)\n'Hello, world!' (28 bytes)",
                    xy=(9.2, 2.3), xytext=(5.5, 0.7),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.8),
                    fontsize=8.0, fontweight="bold", color="#0D47A1",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0"))

    # -------------------------------------------------------------
    # Right Subplot: 4 Key Mutation Experiment Scenarios
    # -------------------------------------------------------------
    ax_cases.set_title("(B) 4 Key Mutation Experiment Scenarios\n& Low-Level Physical Manifestations",
                      fontsize=11.5, fontweight="bold", pad=14, color="#1A237E")
    ax_cases.axis("off")

    cases = [
        ("Case 1: 同一長文字列置換 (Same-Length Mutation)",
         "• 変更内容: 'Hello, world!' (13文字) -> 'Hello, Mikan!' (13文字) at 0x400\n"
         "• メモリ状態: 28バイト領域に完全に適合。ヌル終端 (0x0000) も温存\n"
         "• 結果: [SUCCESS] 画面に 'Hello, Mikan!' が正常表示。副作用なし",
         "#E8F5E9", "#2E7D32", "#1B5E20"),

        ("Case 2: ヌル文字欠落短縮 (Missing Null Terminator)",
         "• 変更内容: 'Hi!' (3文字) を 0x400 に上書きするが、終端 0x0000 を書かない\n"
         "• メモリ状態: 0x400..0x405 に 'Hi!' が入るが、0x406 以降に元の 'llo, world!' が残留\n"
         "• 結果: [CORRUPTED] 画面に 'Hillo, world!' または後続ゴミデータが露出！",
         "#FFF8E1", "#F57F17", "#E65100"),

        ("Case 3: セクション超過オーバーフロー (Section Overflow)",
         "• 変更内容: 300文字超（600B以上）の巨大文字列を無理やりバイナリへ注入\n"
         "• メモリ状態: ファイルサイズが 1536B を突破、または PE ヘッダの SizeOfRawData と矛盾\n"
         "• 結果: [OVERFLOW / REJECT] UEFI BDS ローダがロード拒否、または隣接メモリ破損",
         "#FFEBEE", "#C62828", "#B71C1C"),

        ("Case 4: 機械語命令パッチ (Machine Code Opcode Patch)",
         "• 変更内容: 0x212 の 'jmp $' (EB FE) を 'nop nop' (90 90) に書き換え\n"
         "• メモリ状態: CPU が無限ループを脱出。しかしスタック復帰処理（add rsp, 0x28）がない\n"
         "• 結果: [CRASH / CPU EXCEPTION] 不正アドレスへ ret、未定義動作 / フリーズ発生",
         "#EDE7F6", "#512DA8", "#311B92"),
    ]

    y_pos = 0.96
    for title, desc, fc, ec, tc in cases:
        h = 0.21
        ax_cases.add_patch(FancyBboxPatch((0.04, y_pos - h), 0.92, h, boxstyle="round,pad=0.015",
                                         facecolor=fc, edgecolor=ec, lw=1.6))
        ax_cases.text(0.07, y_pos - 0.04, title, fontsize=8.8, fontweight="bold", color=tc)
        ax_cases.text(0.07, y_pos - 0.08, desc, fontsize=7.6, color="#263238", va="top", linespacing=1.35)
        y_pos -= (h + 0.035)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 1 Section 1.7: Booting with UEFI BIOS (Legacy BIOS vs UEFI, MBR vs GPT)
# ==============================================================================

class FirmwareType(Enum):
    """Firmware system architecture type."""
    LEGACY_BIOS = "Legacy BIOS"
    UEFI = "UEFI"


class CpuMode(Enum):
    """x86 CPU operating modes during boot."""
    REAL_MODE_16 = "16-bit Real Mode"
    PROTECTED_MODE_32 = "32-bit Protected Mode"
    LONG_MODE_64 = "64-bit Long Mode (IA-32e)"


class PartitionScheme(Enum):
    """Disk partition table scheme."""
    MBR = "MBR (Master Boot Record)"
    GPT = "GPT (GUID Partition Table)"


@dataclass
class BootComparisonFeature:
    """Comparison metric between Legacy BIOS and UEFI."""
    feature_name: str
    legacy_bios_value: str
    uefi_value: str
    impact_explanation: str


@dataclass
class RealModeAddress:
    """Real mode segment:offset calculation and A20 gate simulation."""
    segment: int
    offset: int
    a20_enabled: bool = True

    @property
    def linear_address_raw(self) -> int:
        """Calculate raw 21-bit linear address: (segment * 16) + offset."""
        return (self.segment << 4) + self.offset

    @property
    def effective_address(self) -> int:
        """Effective address taking A20 gate into account.

        When A20 is disabled, bit 20 (1 << 20 = 0x100000) is masked out,
        causing 1MB wraparound (0x100000 -> 0x000000).
        """
        raw = self.linear_address_raw
        if self.a20_enabled:
            return raw
        return raw & ~(1 << 20)

    @property
    def is_wrapped(self) -> bool:
        """True if address exceeds 1MB and wraps due to disabled A20."""
        return (not self.a20_enabled) and (self.linear_address_raw >= 0x100000)


@dataclass
class MbrPartitionEntry:
    """16-byte MBR partition table entry."""
    boot_indicator: int          # 0x80: active/bootable, 0x00: inactive
    starting_chs: bytes          # 3 bytes Cylinder-Head-Sector
    partition_type: int          # e.g., 0x07 (NTFS/exFAT), 0x0B (FAT32), 0xEF (ESP)
    ending_chs: bytes            # 3 bytes CHS
    starting_lba: int            # 32-bit starting LBA
    size_in_lba: int             # 32-bit sector count

    @property
    def is_bootable(self) -> bool:
        return self.boot_indicator == 0x80

    def size_bytes(self, sector_size: int = 512) -> int:
        return self.size_in_lba * sector_size

    def size_gib(self, sector_size: int = 512) -> float:
        return self.size_bytes(sector_size) / (1024 ** 3)

    @staticmethod
    def max_addressable_capacity_tib(sector_size: int = 512) -> float:
        """MBR uses 32-bit LBA: 2^32 * 512B = 2.199 TB = 2.0 TiB."""
        max_sectors = (1 << 32)
        return (max_sectors * sector_size) / (1024 ** 4)


@dataclass
class GptHeader:
    """92-byte EFI Partition Table Header (LBA 1)."""
    signature: bytes = b"EFI PART"      # 8 bytes: 'EFI PART'
    revision: int = 0x00010000           # 4 bytes: 1.0 (0x00010000)
    header_size: int = 92                # 4 bytes: usually 92
    header_crc32: int = 0                # 4 bytes: CRC32 of header with this field zeroed
    current_lba: int = 1                 # 8 bytes: LBA 1 for primary
    backup_lba: int = 0                  # 8 bytes: last LBA of disk for backup
    first_usable_lba: int = 34           # 8 bytes: usually 34 (after protective MBR + header + 32 entry sectors)
    last_usable_lba: int = 0             # 8 bytes: backup LBA - 33
    disk_guid: str = ""                  # 16 bytes: unique disk GUID
    partition_entry_lba: int = 2         # 8 bytes: LBA 2
    num_partition_entries: int = 128     # 4 bytes: usually 128
    size_partition_entry: int = 128      # 4 bytes: usually 128 bytes
    partition_array_crc32: int = 0       # 4 bytes: CRC32 of entry array

    @staticmethod
    def max_addressable_capacity_zb(sector_size: int = 512) -> float:
        """GPT uses 64-bit LBA: 2^64 * 512B = 9.44 ZB (Zettabytes)."""
        max_sectors = (1 << 64)
        return (max_sectors * sector_size) / (10 ** 21)


@dataclass
class GptPartitionEntry:
    """128-byte GPT partition entry."""
    type_guid: str                       # 16-byte partition type GUID
    unique_guid: str                     # 16-byte partition unique GUID
    starting_lba: int                    # 8 bytes: uint64
    ending_lba: int                      # 8 bytes: uint64
    attributes: int                      # 8 bytes: bit flags
    name: str                            # 72 bytes: UTF-16LE partition name (36 chars)

    ESP_TYPE_GUID: str = "C12A7328-F81F-11D2-BA4B-00A0C93EC93B"
    LINUX_DATA_GUID: str = "0FC63DAF-8483-4772-8E79-3D69D8477DE4"
    MS_BASIC_DATA_GUID: str = "EBD0A0A2-B9E5-4433-87C0-68B6B72699C7"

    @property
    def is_esp(self) -> bool:
        """True if partition is EFI System Partition."""
        return self.type_guid.upper() == self.ESP_TYPE_GUID.upper()

    @property
    def sector_count(self) -> int:
        return (self.ending_lba - self.starting_lba) + 1

    def size_bytes(self, sector_size: int = 512) -> int:
        return self.sector_count * sector_size

    def size_mib(self, sector_size: int = 512) -> float:
        return self.size_bytes(sector_size) / (1024 ** 2)


class LegacyBiosVsUefiComparator:
    """Quantitative and qualitative architectural comparator."""

    @staticmethod
    def get_comparison_features() -> List[BootComparisonFeature]:
        return [
            BootComparisonFeature(
                feature_name="CPU起動時動作モード",
                legacy_bios_value="16-bit リアルモード (Real Mode)",
                uefi_value="64-bit ロングモード (Long Mode / IA-32e)",
                impact_explanation="BIOSは8086互換リアルモードで起動するため、OS側でA20ゲート解除、GDT初期化、保護モード、ページング、ロングモードへの過酷な多段モード昇格が必要。UEFIは起動直後から64bit全命令が使用可能。"
            ),
            BootComparisonFeature(
                feature_name="アクセス可能メモリ空間",
                legacy_bios_value="1 MB (実質 640 KB: コンベンショナルメモリ)",
                uefi_value="全物理メモリ空間 (64-bit フラット / 恒等マッピング)",
                impact_explanation="BIOSは1MB空間内のセグメント境界やVRAM予約領域を避ける細工が必須。UEFIはGB/TB単位の物理メモリに直接アクセス可能。"
            ),
            BootComparisonFeature(
                feature_name="ブートローダサイズ制限",
                legacy_bios_value="わずか 512 バイト (MBR ブートセクタ)",
                uefi_value="実質無制限 (ファイルシステム上の実行可能バイナリ)",
                impact_explanation="MBRは446バイトのコード内にファイルシステムドライバすら入らず、Stage 1 -> Stage 2の多段ローダが必須。UEFIはMB単位のC++バイナリを一発ロード可能。"
            ),
            BootComparisonFeature(
                feature_name="ストレージ管理 & 容量限界",
                legacy_bios_value="MBR (32-bit LBA: 最大 2.0 TiB, 最大4パーティション)",
                uefi_value="GPT (64-bit LBA: 最大 9.4 ZB, 標準128パーティション)",
                impact_explanation="現代の4TB以上のHDD/NVMe SSDはMBRではアドレッシング不可能。GPTはバックアップヘッダとCRC32チェックサムで高い堅牢性を誇る。"
            ),
            BootComparisonFeature(
                feature_name="ファイルシステムの直接認識",
                legacy_bios_value="非対応 (生セクタ LBA / CHS 読み込みのみ)",
                uefi_value="FAT12/16/32 ネイティブ認識 (/EFI/BOOT/BOOTX64.EFI)",
                impact_explanation="BIOSは特定セクタに直接機械語を書き込む必要があったが、UEFIは一般的なFAT32ストレージにファイルをコピーするだけで起動可能。"
            ),
            BootComparisonFeature(
                feature_name="ファームウェアサービス",
                legacy_bios_value="INT 10h / INT 13h 等の生16bit BIOS割り込み",
                uefi_value="Boot Services (メモリ・GOP等) + Runtime Services (NVRAM・時刻)",
                impact_explanation="C言語の関数ポインタ呼び出しで高機能API（解像度変更、メモリマップ取得、ファイルI/O）を型安全に利用可能。"
            ),
        ]


def plot_legacy_bios_vs_uefi_boot_flow(output_path: str = "1/result/fig1_13_legacy_bios_vs_uefi_boot_flow.png") -> str:
    """Generate high-resolution architecture diagram comparing Legacy BIOS vs UEFI boot flow."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(15, 9.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.18, left=0.05, right=0.95, top=0.90, bottom=0.04)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.set_title("レガシー BIOS ブートシーケンス (Legacy BIOS Boot Flow)\n【16ビットリアルモードからの過酷な多段モード昇格】",
                     fontsize=11.5, fontweight="bold", pad=12, color="#B71C1C")
    ax_right.set_title("現代 UEFI ブートシーケンス (UEFI Boot Flow)\n【64ビットロングモード直接起動・FAT32ファイル実行】",
                      fontsize=11.5, fontweight="bold", pad=12, color="#0D47A1")

    for ax in [ax_left, ax_right]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Left: Legacy BIOS Flow
    bios_steps = [
        ("1. 電源投入 & CPU リセット (Power-on Reset)",
         "• CPU は 16-bit リアルモード (Real Mode) で起動\n• CS:IP = 0xFFFF:0x0000 (0xFFFF0) のリセットベクタへジャンプ\n• アドレッシング可能空間: わずか 1MB (20-bit)",
         "#FFEBEE", "#C62828", "#B71C1C"),

        ("2. MBR ブートセクタ読み込み (512 Bytes Limit)",
         "• ストレージ LBA 0 (512B) を物理アドレス 0x7C00 にロード\n• コード領域は実質 446 バイトのみ！ (末尾はPTと0xAA55)\n• ファイルシステムを理解できず、セクタ番号直打ちで次段をロード",
         "#FFCDD2", "#D32F2F", "#B71C1C"),

        ("3. Stage 1 / Stage 2 多段ブートローダ",
         "• 446Bの制約を突破するため、次段ローダをディスク生セクタからロード\n• BIOS INT 13h 割り込みを用いた低速CHS/LBAセクタ読み込み",
         "#EF9A9A", "#E53935", "#B71C1C"),

        ("4. A20 ゲート解除 & 32-bit 保護モード移行",
         "• 80286互換ラップアラウンド防止回路 (A20 Gate) を開放\n• GDT (Global Descriptor Table) をメモリに自前構築\n• CR0.PE = 1 にセットし、32-bit 保護モード (Protected Mode) へ昇格",
         "#FFE0B2", "#FB8C00", "#E65100"),

        ("5. ページング構築 & 64-bit ロングモード移行",
         "• PAE (CR4.PAE = 1) 有効化 & 4階層ページテーブル (PML4) 構築\n• IA32_EFER.LME = 1 & CR0.PG = 1 でようやく 64-bit Long Mode へ到達！\n• ※OSブートローダ開発の最大の挫折ポイント",
         "#FFF3E0", "#FF9800", "#EF6C00"),

        ("6. 64-bit カーネルエントリポイント呼び出し",
         "• ようやく 64-bit カーネルコードを実行可能\n• しかし画面描画はVGAテキスト(80x25)か低解像度VESAに制限",
         "#E8F5E9", "#43A047", "#1B5E20")
    ]

    y_pos = 0.95
    step_h = 0.125
    gap = 0.032
    for i, (title, desc, fc, ec, tc) in enumerate(bios_steps):
        ax_left.add_patch(FancyBboxPatch((0.05, y_pos - step_h), 0.90, step_h,
                                        boxstyle="round,pad=0.015", facecolor=fc, edgecolor=ec, lw=1.8))
        ax_left.text(0.08, y_pos - 0.032, title, fontsize=9.2, fontweight="bold", color=tc)
        ax_left.text(0.08, y_pos - 0.062, desc, fontsize=7.8, color="#212121", va="top", linespacing=1.35)
        if i < len(bios_steps) - 1:
            ax_left.annotate("", xy=(0.5, y_pos - step_h - gap + 0.005), xytext=(0.5, y_pos - step_h),
                            arrowprops=dict(arrowstyle="->", color="#C62828", lw=2.0))
        y_pos -= (step_h + gap)

    # Right: UEFI Flow
    uefi_steps = [
        ("1. 電源投入 & SEC フェーズ (Security Phase)",
         "• CPU リセットベクタ 0xFFFFFFF0 (4GB直下) から直ちに 64-bit 初期化開始\n• キャッシュをRAMとして使用 (CAR: Cache-as-RAM) し、早期 C 言語実行",
         "#E3F2FD", "#1976D2", "#0D47A1"),

        ("2. PEI / DXE フェーズ (メモリ & ハードウェア初期化)",
         "• 本格的な DRAM コントローラ初期化、PCI バス探索、UEFI ドライバ展開\n• 4階層ページングが自動構築され、CPU は完全な 64-bit Long Mode へ移行",
         "#BBDEFB", "#1E88E5", "#0D47A1"),

        ("3. BDS フェーズ: ESP & FAT32 ファイル認識",
         "• GPT (GUID Partition Table) を解析し、EFI システムパーティションを特定\n• 内蔵 FAT32 ドライバにより /EFI/BOOT/BOOTX64.EFI を通常ファイルとして検出",
         "#90CAF9", "#2196F3", "#0D47A1"),

        ("4. BOOTX64.EFI ロード & PE32+ 再配置",
         "• PE32+ 実行ファイルを物理メモリへロードし、ベース再配置 (Relocation) 適用\n• サイズ制限なし！ 数十MBの巨大ブートローダも一括ロード可能",
         "#E1F5FE", "#0288D1", "#01579B"),

        ("5. 64-bit Long Mode で EfiMain() を直接実行",
         "• RCX=ImageHandle, RDX=SystemTable で C/C++ 関数ポインタ呼び出し\n• GOP による高解像度 32-bit フルカラー描画、メモリマップ取得が即座に使用可能！",
         "#E8F5E9", "#2E7D32", "#1B5E20"),

        ("6. ExitBootServices() による OS カーネルへの引き渡し",
         "• ブートサービスを解放し、OS カーネルが全物理メモリとハードウェアを完全掌握\n• 煩わしい CPU モード昇格・セグメント計算がゼロ！",
         "#C8E6C9", "#388E3C", "#1B5E20")
    ]

    y_pos = 0.95
    for i, (title, desc, fc, ec, tc) in enumerate(uefi_steps):
        ax_right.add_patch(FancyBboxPatch((0.05, y_pos - step_h), 0.90, step_h,
                                         boxstyle="round,pad=0.015", facecolor=fc, edgecolor=ec, lw=1.8))
        ax_right.text(0.08, y_pos - 0.032, title, fontsize=9.2, fontweight="bold", color=tc)
        ax_right.text(0.08, y_pos - 0.062, desc, fontsize=7.8, color="#212121", va="top", linespacing=1.35)
        if i < len(uefi_steps) - 1:
            ax_right.annotate("", xy=(0.5, y_pos - step_h - gap + 0.005), xytext=(0.5, y_pos - step_h),
                             arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2.0))
        y_pos -= (step_h + gap)

    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_mbr_vs_gpt_disk_layout(output_path: str = "1/result/fig1_14_mbr_vs_gpt_disk_layout.png") -> str:
    """Generate high-resolution architecture diagram comparing MBR vs GPT disk partitioning layouts."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax_mbr, ax_gpt) = plt.subplots(2, 1, figsize=(15, 9.2), dpi=300,
                                         gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.35, "left": 0.05, "right": 0.95, "top": 0.92, "bottom": 0.05})

    # Top: MBR Layout
    ax_mbr.set_title("MBR (Master Boot Record) ディスクレイアウト — わずか 512 バイトの単一セクタ構造 (2.0 TiB 制限)",
                     fontsize=11.5, fontweight="bold", pad=10, color="#C62828")
    ax_mbr.set_xlim(0, 512)
    ax_mbr.set_ylim(0, 1)
    ax_mbr.axis("off")

    # MBR segments
    mbr_segments = [
        (0, 446, "ブートストラップ機械語コード (Bootstrap Code)",
         "446 Bytes\n• リアルモード機械語\n• ファイルシステム解釈不可\n• 次セクタ読み込みのみ", "#FFCDD2", "#C62828"),
        (446, 16, "PT 1", "16B\n#1", "#FFE0B2", "#E65100"),
        (462, 16, "PT 2", "16B\n#2", "#FFE0B2", "#E65100"),
        (478, 16, "PT 3", "16B\n#3", "#FFE0B2", "#E65100"),
        (494, 16, "PT 4", "16B\n#4", "#FFE0B2", "#E65100"),
        (510, 2, "シグネチャ\n0x55, 0xAA", "2B\nValid", "#C8E6C9", "#2E7D32")
    ]

    for start, size, title, label, fc, ec in mbr_segments:
        ax_mbr.add_patch(FancyBboxPatch((start, 0.25), size, 0.55, boxstyle="square,pad=0",
                                        facecolor=fc, edgecolor=ec, lw=1.5))
        mid = start + size / 2
        ax_mbr.text(mid, 0.55, title, fontsize=8.5, fontweight="bold", ha="center", va="center", color=ec)
        ax_mbr.text(mid, 0.38, label, fontsize=7.2, ha="center", va="center", color="#263238")

    ax_mbr.text(256, 0.08,
                "[!] MBR の致命的制約: 32-bit LBA (2^32 * 512B = 2.0 TiB限界) / パーティション最大4つ / 冗長性ゼロ (破損時復旧不能)",
                fontsize=8.5, fontweight="bold", color="#B71C1C", ha="center")

    # Bottom: GPT Layout
    ax_gpt.set_title("GPT (GUID Partition Table) ディスクレイアウト — 64-bit LBA (9.4 ZB) と完全な冗長性 (Backup Header/Array)",
                     fontsize=11.5, fontweight="bold", pad=10, color="#0D47A1")
    ax_gpt.set_xlim(0, 100)
    ax_gpt.set_ylim(0, 1)
    ax_gpt.axis("off")

    gpt_blocks = [
        (1, 7, "Protective\nMBR", "LBA 0\n(512B)\n保護用MBR", "#ECEFF1", "#607D8B"),
        (9, 10, "Primary\nGPT Header", "LBA 1\n(92B+0Pad)\nGUID/CRC32", "#BBDEFB", "#1976D2"),
        (20, 14, "Partition\nEntry Array", "LBA 2..33\n128エントリ\n各128バイト", "#FFE0B2", "#F57C00"),
        (35, 20, "EFI System Partition\n(ESP: FAT32)", "LBA 34..K\n/EFI/BOOT/\nBOOTX64.EFI", "#C8E6C9", "#388E3C"),
        (56, 22, "OS Data Partitions\n(MikanOS / Linux / Windows)", "LBA K+1..M\nファイルシステム\nユーザーデータ領域", "#E1BEE7", "#7B1FA2"),
        (79, 12, "Backup Entry\nArray", "LBA N-33..N-1\n冗長エントリ\n(CRC32照合)", "#FFE0B2", "#F57C00"),
        (92, 7, "Backup\nHeader", "LBA N\n末尾冗長\nヘッダ", "#BBDEFB", "#1976D2")
    ]

    for start, size, title, label, fc, ec in gpt_blocks:
        ax_gpt.add_patch(FancyBboxPatch((start, 0.30), size, 0.52, boxstyle="square,pad=0",
                                        facecolor=fc, edgecolor=ec, lw=1.5))
        mid = start + size / 2
        ax_gpt.text(mid, 0.62, title, fontsize=8.0, fontweight="bold", ha="center", va="center", color=ec)
        ax_gpt.text(mid, 0.42, label, fontsize=7.0, ha="center", va="center", color="#37474F")

    ax_gpt.text(50, 0.12,
                "[OK] GPT の革命的進歩: 64-bit LBA (9.4 Zettabytes) / 標準128パーティション / 128-bit GUID識別 / 末尾バックアップと双方向CRC32自己検証",
                fontsize=8.5, fontweight="bold", color="#0D47A1", ha="center")

    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 1.8: OSを作る道具 (LLVM/Clang Toolchain & Freestanding Environment)
# ==============================================================================

@dataclass(frozen=True)
class TargetTriple:
    """Represents a compiler target triple (arch-vendor-os-environment)."""

    arch: str
    vendor: str
    os_name: str
    environment: Optional[str] = None

    @classmethod
    def parse(cls, triple_str: str) -> TargetTriple:
        """Parse a target triple string such as 'x86_64-pc-win32-coff', 'x86_64-unknown-linux-gnu', or 'x86_64-elf'."""
        parts = [p for p in triple_str.strip().split("-") if p]
        if not parts:
            raise ValueError(f"Invalid empty target triple string")
        if len(parts) == 1:
            return cls(arch=parts[0], vendor="unknown", os_name="none", environment=None)
        if len(parts) == 2:
            return cls(arch=parts[0], vendor="unknown", os_name=parts[1], environment=None)
        if len(parts) == 3:
            return cls(arch=parts[0], vendor=parts[1], os_name=parts[2], environment=None)
        arch = parts[0]
        vendor = parts[1]
        os_name = parts[2]
        env = "-".join(parts[3:])
        return cls(arch=arch, vendor=vendor, os_name=os_name, environment=env)

    @property
    def canonical_string(self) -> str:
        """Return the normalized target triple string."""
        if self.environment:
            return f"{self.arch}-{self.vendor}-{self.os_name}-{self.environment}"
        return f"{self.arch}-{self.vendor}-{self.os_name}"

    def is_cross_compile(self, host: Union[str, TargetTriple]) -> bool:
        """Determine whether compiling for this target from host is a cross-compilation."""
        if isinstance(host, str):
            host = TargetTriple.parse(host)
        return (
            self.arch != host.arch
            or self.os_name != host.os_name
            or self.environment != host.environment
        )

    def is_uefi_compatible(self) -> bool:
        """Check if this target produces PE/COFF binaries suitable for UEFI x86-64."""
        if self.arch != "x86_64":
            return False
        # Windows / COFF targets generate PE32+ for UEFI
        return (
            "win32" in self.os_name.lower()
            or "windows" in self.os_name.lower()
            or (self.environment and "coff" in self.environment.lower())
        )

    def object_format(self) -> str:
        """Return expected relocatable object file format."""
        if self.is_uefi_compatible() or (self.environment and "coff" in self.environment.lower()):
            return "COFF"
        if "linux" in self.os_name.lower() or "elf" in self.os_name.lower() or "elf" in (self.environment or "").lower() or "none" in self.os_name.lower():
            return "ELF"
        if "darwin" in self.os_name.lower() or "macos" in self.os_name.lower():
            return "Mach-O"
        return "Unknown"

    def executable_format(self) -> str:
        """Return final executable binary format."""
        if self.is_uefi_compatible():
            return "PE32+"
        if "linux" in self.os_name.lower() or "elf" in self.os_name.lower() or "elf" in (self.environment or "").lower():
            return "ELF64"
        if "darwin" in self.os_name.lower() or "macos" in self.os_name.lower():
            return "Mach-O 64"
        return "Unknown"


class FreestandingEnvironmentModel:
    """Models ISO C/C++ Freestanding Environment constraints for OS development."""

    STANDARD_FREESTANDING_HEADERS: set[str] = {
        "float.h",
        "iso646.h",
        "limits.h",
        "stdalign.h",
        "stdarg.h",
        "stdbool.h",
        "stddef.h",
        "stdint.h",
        "stdnoreturn.h",
    }

    HOSTED_ONLY_HEADERS: set[str] = {
        "stdio.h",
        "stdlib.h",
        "string.h",
        "math.h",
        "time.h",
        "assert.h",
        "signal.h",
        "setjmp.h",
        "errno.h",
        "ctype.h",
        "locale.h",
        "wchar.h",
        "wctype.h",
        "unistd.h",
        "fcntl.h",
        "sys/types.h",
        "pthread.h",
    }

    REQUIRED_BUILTIN_MEM_FUNCTIONS: list[str] = [
        "memcpy",
        "memmove",
        "memset",
        "memcmp",
    ]

    @classmethod
    def is_header_allowed(cls, header: str) -> bool:
        """Check whether a given header file is permitted in a freestanding environment."""
        clean_header = header.strip("<> \"'")
        return clean_header in cls.STANDARD_FREESTANDING_HEADERS

    @classmethod
    def analyze_includes(cls, source_code: str) -> Dict[str, Any]:
        """Analyze #include directives in C/C++ source code for freestanding compliance."""
        import re

        include_pattern = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
        found_headers = include_pattern.findall(source_code)

        allowed = []
        forbidden = []
        unknown = []

        for h in found_headers:
            if h in cls.STANDARD_FREESTANDING_HEADERS:
                allowed.append(h)
            elif h in cls.HOSTED_ONLY_HEADERS:
                forbidden.append(h)
            else:
                unknown.append(h)

        is_compliant = len(forbidden) == 0
        return {
            "total_includes": len(found_headers),
            "allowed_freestanding": allowed,
            "forbidden_hosted": forbidden,
            "custom_or_unknown": unknown,
            "is_compliant": is_compliant,
            "status_message": (
                "PASS: 全てのヘッダがフリースタンディング環境に適合しています。"
                if is_compliant
                else f"VIOLATION: ホスト環境依存ヘッダ ({', '.join(forbidden)}) が含まれています。OSカーネル/UEFIアプリでは利用できません。"
            ),
        }


class CompilerToolchainPipeline:
    """Simulates the 4 stages of C/C++ compilation pipeline to PE32+ UEFI / ELF."""

    ESSENTIAL_OS_FLAGS: dict[str, str] = {
        "-ffreestanding": "ホスト libc への依存を遮断し、フリースタンディング環境としてコンパイル",
        "-mno-red-zone": "割り込みやファームウェア呼び出し時のスタック128バイトRed Zone破壊を防止",
        "-fno-stack-protector": "スタックカナリア未定義参照 (__stack_chk_fail) の発生を抑制",
        "-fshort-wchar": "UEFI規格の16-bit Unicode (CHAR16) に wchar_t のサイズを合致",
    }

    @classmethod
    def validate_compiler_flags(cls, flags: List[str]) -> Dict[str, Any]:
        """Inspect flags for critical OS bare-metal / UEFI requirements."""
        flag_set = set(flags)
        present = {}
        missing = {}

        for opt, desc in cls.ESSENTIAL_OS_FLAGS.items():
            if opt in flag_set:
                present[opt] = desc
            else:
                missing[opt] = desc

        score = (len(present) / len(cls.ESSENTIAL_OS_FLAGS)) * 100.0
        return {
            "score_percent": score,
            "present_flags": present,
            "missing_flags": missing,
            "is_safe_for_os_dev": len(missing) == 0,
        }

    @classmethod
    def trace_stages(cls, source_name: str, target_triple: str) -> List[Dict[str, str]]:
        """Trace the 4 stages of compilation for a given source and target."""
        triple = TargetTriple.parse(target_triple)
        base = Path(source_name).stem

        return [
            {
                "stage": "1. プリプロセッサ (Preprocessing)",
                "tool": "clang -E",
                "input": f"{base}.c / {base}.cpp",
                "output": f"{base}.i / {base}.ii",
                "description": "#include ヘッダ展開、#define マクロ置換、条件付きコンパイル (#ifdef) の評価",
            },
            {
                "stage": "2. コンパイラ (Compilation)",
                "tool": "clang -S (or -emit-llvm)",
                "input": f"{base}.i",
                "output": f"{base}.s (Assembly) / {base}.ll (LLVM IR)",
                "description": "構文解析、中間表現 (LLVM IR) 最適化、ターゲットアーキテクチャ依存のアセンブリ言語生成",
            },
            {
                "stage": "3. アセンブラ (Assembler)",
                "tool": "clang -c (llvm-mc)",
                "input": f"{base}.s",
                "output": f"{base}.o ({triple.object_format()} Relocatable Object)",
                "description": f"アセンブリから機械語バイナリへの変換、未解決外部シンボルと再配置情報の生成 ({triple.object_format()}形式)",
            },
            {
                "stage": "4. リンカ (Linker)",
                "tool": "lld-link / ld.lld",
                "input": f"{base}.o + Libs",
                "output": f"{base}.efi ({triple.executable_format()})",
                "description": f"複数オブジェクトと静的ライブラリの結合、シンボルアドレス解決、最終実行可能バイナリ ({triple.executable_format()}) の出力",
            },
        ]


def plot_llvm_toolchain_and_build_pipeline(
    output_path: str = "1/result/fig1_15_llvm_toolchain_and_build_pipeline.png"
) -> str:
    """Generate high-DPI architecture diagram of LLVM Toolchain & 4-Stage Build Pipeline."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax_llvm, ax_pipe) = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [1.1, 1.3]})
    fig.patch.set_facecolor("#FAFAFA")

    # Top: LLVM Modular Architecture
    ax_llvm.set_title("LLVM モジュラーアーキテクチャ — 中間表現 (LLVM IR) によるクロスコンパイルの超汎用化",
                     fontsize=12, fontweight="bold", pad=12, color="#1A237E")
    ax_llvm.set_xlim(0, 100)
    ax_llvm.set_ylim(0, 100)
    ax_llvm.axis("off")

    # Frontends
    frontends = [
        (4, 70, 18, 16, "Clang\n(C / C++)", "#E3F2FD", "#1565C0"),
        (4, 48, 18, 16, "rustc\n(Rust)", "#FFE0B2", "#E65100"),
        (4, 26, 18, 16, "Swift / Go\n(Other)", "#EDE7F6", "#512DA8"),
    ]
    for x, y, w, h, label, fc, ec in frontends:
        ax_llvm.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", facecolor=fc, edgecolor=ec, lw=1.5))
        ax_llvm.text(x + w / 2, y + h / 2, label, fontsize=9.5, fontweight="bold", ha="center", va="center", color=ec)

    # Arrows to IR
    for _, y, _, h, _, _, _ in frontends:
        ax_llvm.annotate("", xy=(30, 56), xytext=(22, y + h / 2),
                         arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5))

    # Core: LLVM IR & Optimizer
    ax_llvm.add_patch(FancyBboxPatch((30, 26), 36, 60, boxstyle="round,pad=0.8", facecolor="#E8F5E9", edgecolor="#2E7D32", lw=2))
    ax_llvm.text(48, 77, "LLVM 共通中間表現\n(LLVM IR: .ll / .bc)", fontsize=11, fontweight="bold", ha="center", va="center", color="#1B5E20")
    ax_llvm.text(48, 62, "• 静的単一代入形式 (SSA Form)\n• 無限個の仮想レジスタ\n• ハードウェア非依存の共通コードモデル",
                 fontsize=8.5, ha="center", va="center", color="#2E7D32")
    
    ax_llvm.add_patch(FancyBboxPatch((34, 30), 28, 22, boxstyle="round,pad=0.4", facecolor="#C8E6C9", edgecolor="#388E3C", lw=1.5))
    ax_llvm.text(48, 41, "LLVM オプティマイザ\n(Target-Independent Passes)\nインライン展開 / 定数畳み込み / 死滅コード除去",
                 fontsize=8.5, fontweight="bold", ha="center", va="center", color="#1B5E20")

    # Arrows to Backends
    backends = [
        (74, 70, 22, 16, "x86-64 Backend\n(PE32+ / ELF64)\n-> MikanOS / UEFI", "#FFEBEE", "#C62828"),
        (74, 48, 22, 16, "AArch64 Backend\n(ARM64 ELF)\n-> Apple M / RPi", "#F3E5F5", "#6A1B9A"),
        (74, 26, 22, 16, "RISC-V Backend\n(RV64GC ELF)\n-> 組み込みオープンOS", "#FFF8E1", "#F57F17"),
    ]
    for x, y, w, h, label, fc, ec in backends:
        ax_llvm.annotate("", xy=(x, y + h / 2), xytext=(66, 56),
                         arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5))
        ax_llvm.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", facecolor=fc, edgecolor=ec, lw=1.5))
        ax_llvm.text(x + w / 2, y + h / 2, label, fontsize=8.5, fontweight="bold", ha="center", va="center", color=ec)

    ax_llvm.text(50, 6,
                 "★ 単一コンパイラバイナリ (clang) で、-target を指定するだけで任意のOS・CPUアーキテクチャのバイナリをクロス生成可能",
                 fontsize=9.5, fontweight="bold", ha="center", color="#0D47A1",
                 bbox=dict(boxstyle="square,pad=0.4", facecolor="#E3F2FD", edgecolor="#1976D2", alpha=0.9))

    # Bottom: 4-Stage Build Pipeline
    ax_pipe.set_title("C/C++ ビルドパイプラインの4段階 — ソースコードから UEFI 実行バイナリ (.efi) までの完全フロー",
                      fontsize=12, fontweight="bold", pad=12, color="#004D40")
    ax_pipe.set_xlim(0, 100)
    ax_pipe.set_ylim(0, 100)
    ax_pipe.axis("off")

    stages = [
        (3, 30, 20, 52, "1. プリプロセッサ\n(Preprocessor)", "clang -E", "source.c\nsource.h", "source.i\n(プリプロセス済み)",
         "#E0F2F1", "#00695C", "# include 展開\n# define 置換\nコメント除去"),
        (27, 30, 20, 52, "2. コンパイラ\n(Compiler)", "clang -S\n(-ffreestanding)", "source.i", "source.s\n(アセンブリ)",
         "#FFF3E0", "#E65100", "構文解析 (AST)\nLLVM IR生成\nx86-64 機械語変換"),
        (51, 30, 20, 52, "3. アセンブラ\n(Assembler)", "clang -c\n(llvm-mc)", "source.s", "source.o\n(COFF / ELF)",
         "#EDE7F6", "#4527A0", "機械語バイナリ化\nシンボルテーブル\n再配置情報生成"),
        (75, 30, 22, 52, "4. リンカ\n(Linker)", "lld-link\n(/subsystem:efi_app)", "source.o\n+ Libs", "BOOTX64.EFI\n(PE32+ 実行形式)",
         "#E8F5E9", "#2E7D32", "複数オブジェクト結合\nアドレス・シンボル解決\nPE32+/COFF出力"),
    ]

    for i, (x, y, w, h, title, tool, in_file, out_file, fc, ec, detail) in enumerate(stages):
        ax_pipe.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", facecolor=fc, edgecolor=ec, lw=1.8))
        ax_pipe.text(x + w / 2, y + 44, title, fontsize=9.5, fontweight="bold", ha="center", va="center", color=ec)
        ax_pipe.text(x + w / 2, y + 33, f"ツール: {tool}", fontsize=7.8, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor=ec, lw=0.8))
        ax_pipe.text(x + w / 2, y + 21, f"入力: {in_file}\n出力: {out_file}", fontsize=7.8, ha="center", va="center", color="#263238")
        ax_pipe.text(x + w / 2, y + 8, detail, fontsize=7.2, ha="center", va="center", color="#37474F")

        if i < len(stages) - 1:
            ax_pipe.annotate("", xy=(stages[i + 1][0], y + 26), xytext=(x + w, y + 26),
                             arrowprops=dict(arrowstyle="-|>", color="#00695C", lw=2.2))

    ax_pipe.text(50, 10,
                 "★ OSカーネル自作では、第4段階のリンカにホスト標準ライブラリ (libc.so / crt0.o) を渡さず、自作エントリポイント (EfiMain / KernelMain) を直接指定する",
                 fontsize=9.0, fontweight="bold", ha="center", color="#B71C1C",
                 bbox=dict(boxstyle="square,pad=0.4", facecolor="#FFEBEE", edgecolor="#C62828", alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_freestanding_vs_hosted_environment(
    output_path: str = "1/result/fig1_16_freestanding_vs_hosted_environment.png"
) -> str:
    """Generate high-DPI architecture diagram comparing Hosted vs Freestanding Environments."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax_host, ax_free) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("#FAFAFA")

    # Left: Hosted Environment
    ax_host.set_title("ホスト環境 (Hosted Environment)\n— 一般アプリケーション開発の世界",
                      fontsize=12, fontweight="bold", pad=12, color="#1565C0")
    ax_host.set_xlim(0, 100)
    ax_host.set_ylim(0, 100)
    ax_host.axis("off")

    host_layers = [
        (10, 80, 80, 13, "ユーザーアプリケーション\n(int main(int argc, char** argv))", "#E3F2FD", "#1565C0", "printf(\"Hello, World\\n\");"),
        (10, 61, 80, 14, "C/C++ 標準ライブラリ (libc / glibc / MSVCRT)\nstdio.h / stdlib.h / string.h / malloc() / exit()", "#BBDEFB", "#0D47A1", "バッファリング / ヒープ管理"),
        (10, 42, 80, 14, "システムコール境界 (Ring 3 -> Ring 0 特権移行)\nwrite(1, buf, len) / mmap() / exit_group()", "#FFE0B2", "#E65100", "SYSCALL / INT 0x80 命令"),
        (10, 23, 80, 14, "既存 OS カーネル (Linux / Windows NT)\nプロセス管理 / 仮想メモリ / デバイスドライバ", "#FFCDD2", "#C62828", "完全な抽象化と保護"),
        (10, 4, 80, 14, "物理ハードウェア (CPU / RAM / ディスプレイ)", "#ECEFF1", "#37474F", "MMIO / ポートI/O / 割り込み"),
    ]

    for x, y, w, h, label, fc, ec, note in host_layers:
        ax_host.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", facecolor=fc, edgecolor=ec, lw=1.8))
        ax_host.text(x + w / 2, y + h / 2 + 1.5, label, fontsize=9.0, fontweight="bold", ha="center", va="center", color=ec)
        ax_host.text(x + w / 2, y + 2.5, f"[{note}]", fontsize=7.5, ha="center", va="center", color="#37474F")

    for i in range(len(host_layers) - 1):
        ax_host.annotate("", xy=(50, host_layers[i + 1][1] + host_layers[i + 1][3]),
                         xytext=(50, host_layers[i][1]),
                         arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2.0))

    # Right: Freestanding Environment
    ax_free.set_title("フリースタンディング環境 (Freestanding Environment)\n— OS カーネル & UEFI アプリ自作の世界 (-ffreestanding)",
                      fontsize=12, fontweight="bold", pad=12, color="#2E7D32")
    ax_free.set_xlim(0, 100)
    ax_free.set_ylim(0, 100)
    ax_free.axis("off")

    free_layers = [
        (10, 78, 80, 16, "MikanOS カーネル / UEFI アプリケーション\n(EfiMain(ImageHandle, SystemTable) / KernelMain)",
         "#E8F5E9", "#2E7D32", "標準libcなし！printf/malloc使用不能！"),
        (10, 52, 80, 21, "フリースタンディング標準ヘッダ (言語仕様のみ)\n<stdint.h> (uint64_t) / <stddef.h> (size_t, NULL)\n<stdarg.h> (va_list) / <limits.h> / <stdbool.h>",
         "#C8E6C9", "#1B5E20", "★ OS非依存の型定義・マクロ・組み込み関数のみ提供\n(stdio.h や stdlib.h は存在しない)"),
        (10, 28, 80, 18, "UEFI ファームウェア / ベアメタルインターフェース\nSystemTable->ConOut->OutputString() / GOP VRAM直書き",
         "#FFF9C4", "#F57F17", "ファームウェアサービスを直接呼び出し / レジスタ直制御"),
        (10, 4, 80, 18, "物理ハードウェア (x86-64 CPU / 物理RAM / VRAM)\nCR0〜CR4 / ページテーブル / PCI / APIC",
         "#ECEFF1", "#37474F", "メモリマップ・特権命令・スタックを自力制御"),
    ]

    for x, y, w, h, label, fc, ec, note in free_layers:
        ax_free.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", facecolor=fc, edgecolor=ec, lw=1.8))
        ax_free.text(x + w / 2, y + h / 2 + 2.5, label, fontsize=9.0, fontweight="bold", ha="center", va="center", color=ec)
        ax_free.text(x + w / 2, y + 3.5, f"[{note}]", fontsize=7.5, fontweight="bold", ha="center", va="center", color="#C62828")

    for i in range(len(free_layers) - 1):
        ax_free.annotate("", xy=(50, free_layers[i + 1][1] + free_layers[i + 1][3]),
                         xytext=(50, free_layers[i][1]),
                         arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path



# ==============================================================================
# Chapter 1.9: C言語でハローワールド & コラム1.1: PEとCOFFとELF
# ==============================================================================

@dataclass
class UefiStructField:
    """UEFI 構造体フィールドの定義とメモリレイアウト"""
    name: str
    offset: int
    size: int
    type_name: str
    description: str


class UefiSystemTableModel:
    """UEFI システムテーブル (EFI_SYSTEM_TABLE) およびコンソール出力プロトコルのメモリモデル

    『ゼロからのOS自作入門』第1章 1.9節の hello.c で定義された最小構造体と、
    UEFI仕様書における完全な定義との対応関係、およびアロー演算子 (->) による
    ポインタ間接参照チェーンをシミュレートする。
    """

    # hello.c における EFI_SYSTEM_TABLE のレイアウト
    MINIMAL_SYSTEM_TABLE_FIELDS = [
        UefiStructField("dummy", 0, 52, "char[52]", "EFI_TABLE_HEADER(24B) + Vendor/Revision + ConIn(28B) の集約ダミー"),
        UefiStructField("ConsoleOutHandle", 52, 8, "EFI_HANDLE (void*)", "標準出力デバイスのUEFIハンドル"),
        UefiStructField("ConOut", 60, 8, "EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL*", "標準テキスト出力プロトコルへのポインタ"),
    ]

    # hello.c における EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL のレイアウト
    MINIMAL_TEXT_OUTPUT_FIELDS = [
        UefiStructField("dummy", 0, 8, "void*", "Reset() 関数ポインタ（未使用のためダミーポインタ）"),
        UefiStructField("OutputString", 8, 8, "EFI_TEXT_STRING", "文字列描画関数ポインタ OutputString(This, String)"),
    ]

    def __init__(self, system_table_base: int = 0x3F80_0000, con_out_base: int = 0x3F80_1000):
        self.system_table_base = system_table_base
        self.con_out_base = con_out_base
        self.output_buffer: list[str] = []
        self.call_history: list[dict[str, Any]] = []

    def inspect_system_table_layout(self) -> list[dict[str, Any]]:
        """EFI_SYSTEM_TABLE のメモリ配置情報を返す"""
        return [
            {
                "field": f.name,
                "offset": f.offset,
                "offset_hex": f"0x{f.offset:02X}",
                "size": f.size,
                "type": f.type_name,
                "description": f.description,
            }
            for f in self.MINIMAL_SYSTEM_TABLE_FIELDS
        ]

    def inspect_text_output_protocol_layout(self) -> list[dict[str, Any]]:
        """EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL のメモリ配置情報を返す"""
        return [
            {
                "field": f.name,
                "offset": f.offset,
                "offset_hex": f"0x{f.offset:02X}",
                "size": f.size,
                "type": f.type_name,
                "description": f.description,
            }
            for f in self.MINIMAL_TEXT_OUTPUT_FIELDS
        ]

    def encode_utf16le(self, text: str) -> bytes:
        """C言語のワイド文字列リテラル L"..." を UTF-16LE バイト列（ヌル終端含む）に変換"""
        return text.encode("utf-16le") + b"\x00\x00"

    def decode_utf16le(self, raw_bytes: bytes) -> str:
        """UTF-16LE バイト列からヌル終端直前までの文字列をデコード"""
        # 2バイトアライメントでヌル終端 (0x0000) を探す
        end = len(raw_bytes)
        for i in range(0, len(raw_bytes) - 1, 2):
            if raw_bytes[i] == 0 and raw_bytes[i + 1] == 0:
                end = i
                break
        return raw_bytes[:end].decode("utf-16le", errors="replace")

    def simulate_call_output_string(self, this_ptr: int, string_utf16_bytes: bytes) -> tuple[int, str]:
        """SystemTable->ConOut->OutputString(SystemTable->ConOut, L"...") の実行を模倣する

        Returns:
            (status_code, rendered_text): (0=EFI_SUCCESS, 表示されたテキスト)
        """
        # 第一引数 This ポインタの正当性検査 (オブジェクト指向 C言語の約束事)
        if this_ptr != self.con_out_base:
            error_status = 0x8000_0000_0000_0002  # EFI_INVALID_PARAMETER
            self.call_history.append({
                "status": "EFI_INVALID_PARAMETER",
                "this_ptr": hex(this_ptr),
                "expected_this": hex(self.con_out_base),
                "rendered": "",
            })
            return error_status, ""

        rendered = self.decode_utf16le(string_utf16_bytes)
        self.output_buffer.append(rendered)
        self.call_history.append({
            "status": "EFI_SUCCESS",
            "this_ptr": hex(this_ptr),
            "expected_this": hex(self.con_out_base),
            "rendered": rendered,
        })
        return 0, rendered

    def trace_pointer_dereference(self) -> list[str]:
        """SystemTable->ConOut->OutputString(...) のアロー演算子の評価ステップをトレース"""
        st_base = hex(self.system_table_base)
        co_offset = 60
        co_ptr_addr = hex(self.system_table_base + co_offset)
        co_base = hex(self.con_out_base)
        out_str_offset = 8
        func_ptr_addr = hex(self.con_out_base + out_str_offset)

        return [
            f"1. SystemTable ポインタ値 ({st_base}) を基底アドレスとして読み出し",
            f"2. ConOut メンバのオフセット (+60 = 0x3C バイト) を加算 -> アドレス {co_ptr_addr}",
            f"3. アドレス {co_ptr_addr} から 64ビットポインタ値を間接参照 (Dereference) -> ConOut={co_base}",
            f"4. ConOut ({co_base}) を基底として OutputString オフセット (+8 = 0x08 バイト) を加算 -> アドレス {func_ptr_addr}",
            f"5. アドレス {func_ptr_addr} から関数ポインタを間接参照し、呼び出し規約 (Microsoft x64) に従い RCX={co_base}, RDX=&String で CALL",
        ]


class BinaryFormatKind(Enum):
    """実行可能バイナリ形式の分類"""
    PE32_PLUS = "PE32+ (PE/COFF 64-bit)"
    ELF64 = "ELF64 (Executable and Linkable Format 64-bit)"
    COFF_OBJ = "COFF Object (.o)"


@dataclass
class BinaryFormatFeature:
    """バイナリ形式の比較項目"""
    category: str
    pe32_plus: str
    elf64: str
    historical_reason: str


class BinaryFormatComparator:
    """コラム1.1: PEとCOFFとELF のバイナリ構造・歴史的選定比較シミュレータ"""

    COMPARISONS: list[BinaryFormatFeature] = [
        BinaryFormatFeature(
            category="マジックナンバー (Signature)",
            pe32_plus="MS-DOSヘッダ 'MZ' (0x5A4D) + PEシグネチャ 'PE\\0\\0' (0x00004550)",
            elf64="ELFマジック 0x7F 'ELF' (0x464C457F)",
            historical_reason="PEはMS-DOS互換ヘッダを先頭に残す設計。ELFはUnix System V由来で最初から純粋な32/64bit形式。"
        ),
        BinaryFormatFeature(
            category="ヘッダ構造と階層",
            pe32_plus="DOS Header (64B) -> DOS Stub -> COFF Header (20B) -> Optional Header (112B) -> Data Directories (128B) -> Section Headers (40B×N)",
            elf64="ELF Header (64B) -> Program Header Table (56B×M, ローダ用) -> Section Header Table (64B×N, リンカ用)",
            historical_reason="PEはCOFFの拡張としてOptional Headerを追加。ELFは実行用のセグメント(Program Header)とリンク用のセクションを明確に二層分離。"
        ),
        BinaryFormatFeature(
            category="エントリポイント指定",
            pe32_plus="AddressOfEntryPoint (ImageBaseからの相対仮想アドレス: RVA)",
            elf64="e_entry (絶対仮想アドレス、またはPIE時はベースからのオフセット)",
            historical_reason="PEは任意のアドレスにリロケーション可能なRVAが基本。UEFIファームウェアもメモリの空きスロットにRVAで配置する。"
        ),
        BinaryFormatFeature(
            category="サブシステム識別",
            pe32_plus="Subsystem フィールド (IMAGE_SUBSYSTEM_EFI_APPLICATION = 10, EFI_BOOT_SERVICE_DRIVER = 11 等)",
            elf64="EI_OSABI フィールド (System V, Linux等) + PT_INTERP (動的リンカパス)",
            historical_reason="PEはOSがGUI/CUI/EFIを1つのフィールドで直接識別。UEFI仕様書はPEのSubsystem=10をそのまま規格として採用。"
        ),
        BinaryFormatFeature(
            category="主な採用環境",
            pe32_plus="UEFI ファームウェア規格、Microsoft Windows (64-bit)",
            elf64="Linux, FreeBSD, Android, 組込みLinux, POSIX準拠OS全般",
            historical_reason="1990年代後半、IntelがItanium (IA-64) 向けにEFI仕様を策定した際、Microsoftと密接に協業していたためPE32+が規格として採用された。"
        ),
    ]

    @classmethod
    def get_comparison_table(cls) -> list[dict[str, str]]:
        return [
            {
                "category": f.category,
                "pe32_plus": f.pe32_plus,
                "elf64": f.elf64,
                "historical_reason": f.historical_reason,
            }
            for f in cls.COMPARISONS
        ]

    @classmethod
    def explain_uefi_pe_adoption(cls) -> str:
        """なぜ UEFI は ELF ではなく PE32+ を採用したのかの技術的・歴史的背景"""
        return (
            "【なぜUEFIはELFではなくPE32+を採用したのか？】\n"
            "1. 歴史的経緯 (IntelとMicrosoftの協調):\n"
            "   UEFIの前身である EFI (Extensible Firmware Interface) は、1990年代後半に Intel が\n"
            "   次世代64ビットプロセッサ Itanium (IA-64) 向けに開発しました。当時、Itanium向けの\n"
            "   OS・開発ツールチェーンで最も強力なパートナーが Microsoft であり、Windows NT が全面採用\n"
            "   していた PE/COFF 形式をそのままファームウェアのバイナリ標準として採用しました。\n"
            "\n"
            "2. RVA (相対仮想アドレス) による位置独立性 (Position Independence):\n"
            "   PE32+ はすべてのシンボルやエントリポイントを ImageBase からの相対オフセット (RVA)\n"
            "   で表現し、`.reloc` セクション（ベース再配置テーブル）を持つため、UEFI BIOS が起動時に\n"
            "   メモリ上の任意の空きアドレスにブートローダをロードして即座に再配置・実行できます。\n"
            "\n"
            "3. サブシステム (Subsystem) の標準化:\n"
            "   PEヘッダの `Subsystem` フィールドに `IMAGE_SUBSYSTEM_EFI_APPLICATION (10)` や\n"
            "   `IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER (11)` を定義するだけで、ファームウェアローダが\n"
            "   バイナリの実行タイプを瞬時に判別できる統一仕様が完成しました。"
        )


class MinimalUefiCCompilerModel:
    """Clang / LLD による hello.c ビルドパイプラインのシミュレータ"""

    def __init__(self, source_code: str):
        self.source_code = source_code

    def validate_freestanding_compliance(self) -> dict[str, Any]:
        """ソースコードが Freestanding 環境 (stdio.h なし、自前型定義) に適合しているか検証"""
        has_stdio = "#include <stdio.h>" in self.source_code
        has_stdlib = "#include <stdlib.h>" in self.source_code
        has_efi_main = "EfiMain" in self.source_code
        has_system_table = "EFI_SYSTEM_TABLE" in self.source_code
        has_output_string = "OutputString" in self.source_code

        is_compliant = (not has_stdio) and (not has_stdlib) and has_efi_main and has_system_table and has_output_string
        return {
            "is_compliant": is_compliant,
            "has_forbidden_hosted_headers": has_stdio or has_stdlib,
            "has_efi_main_entry": has_efi_main,
            "has_system_table_def": has_system_table,
            "has_output_string_call": has_output_string,
        }

    def simulate_build_commands(self) -> list[dict[str, str]]:
        """Makefile のコンパイル・リンクコマンドをシミュレート"""
        return [
            {
                "stage": "Compilation (Clang)",
                "command": "clang -target x86_64-pc-win32-coff -o hello.o -c hello.c",
                "target_triple": "x86_64-pc-win32-coff",
                "output": "hello.o (COFF Object File)",
                "purpose": "x86-64 向け COFF 形式オブジェクトファイルへのコンパイル",
            },
            {
                "stage": "Linking (LLD-Link)",
                "command": "lld-link /subsystem:efi_application /entry:EfiMain /out:hello.efi hello.o",
                "subsystem": "efi_application (10)",
                "entry_point": "EfiMain",
                "output": "hello.efi (PE32+ Executable)",
                "purpose": "UEFI アプリケーション形式の PE32+ バイナリ合成",
            },
        ]


def plot_uefi_c_system_table_layout(output_path: str = "1/result/fig1_17_uefi_c_data_structures_and_system_table.png") -> str:
    """Figure 1.17: C言語での最小UEFIアプリケーション構造と EFI_SYSTEM_TABLE のメモリ配置図 (DPI 300)"""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.set_title("C言語による最小 UEFI アプリケーションのデータ構造と EFI_SYSTEM_TABLE メモリマップ\n"
                 "— SystemTable->ConOut->OutputString(SystemTable->ConOut, L\"...\") の完全解剖",
                 fontsize=13, fontweight="bold", pad=15, color="#1A237E")

    # Block 1: hello.c C言語コードスニペット
    c_box = FancyBboxPatch((3, 48), 30, 44, boxstyle="round,pad=0.6", facecolor="#EDE7F6", edgecolor="#4A148C", lw=2)
    ax.add_patch(c_box)
    ax.text(18, 89, "hello.c 最小コード", fontsize=11, fontweight="bold", ha="center", color="#4A148C")

    code_text = (
        "typedef struct {\n"
        "  char dummy[52];\n"
        "  EFI_HANDLE ConsoleOutHandle;\n"
        "  EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL\n"
        "      *ConOut;\n"
        "} EFI_SYSTEM_TABLE;\n"
        "\n"
        "EFI_STATUS EfiMain(\n"
        "    EFI_HANDLE ImageHandle,\n"
        "    EFI_SYSTEM_TABLE *SystemTable) {\n"
        "  SystemTable->ConOut->OutputString(\n"
        "      SystemTable->ConOut,\n"
        "      L\"Hello, world!\\n\");\n"
        "  while (1);\n"
        "  return 0;\n"
        "}"
    )
    ax.text(5, 68, code_text, fontsize=7.8, family="monospace", va="center", color="#1A237E")

    # Block 2: EFI_SYSTEM_TABLE メモリレイアウト
    st_box = FancyBboxPatch((37, 35), 28, 57, boxstyle="round,pad=0.6", facecolor="#E3F2FD", edgecolor="#1565C0", lw=2)
    ax.add_patch(st_box)
    ax.text(51, 89, "EFI_SYSTEM_TABLE\n(メモリ配置: 基底 0x3F80_0000)", fontsize=10.5, fontweight="bold", ha="center", color="#1565C0")

    st_cells = [
        (40, 75, 22, 9, "dummy[52] (52 Bytes)\nオフセット 0x00 〜 0x33\n[Header/Vendor/ConIn 等]", "#BBDEFB", "#0D47A1"),
        (40, 64, 22, 8, "ConsoleOutHandle (8 Bytes)\nオフセット 0x34 〜 0x3B (+52)\n[コンソールハンドル]", "#90CAF9", "#0D47A1"),
        (40, 48, 22, 13, "ConOut* (8 Bytes ポインタ)\nオフセット 0x3C 〜 0x43 (+60)\n値: 0x3F80_1000\n[TextOutputProtocol を指す]", "#FFE082", "#E65100"),
    ]
    for x, y, w, h, lbl, fc, ec in st_cells:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.3", facecolor=fc, edgecolor=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2, lbl, fontsize=8.0, fontweight="bold", ha="center", va="center", color=ec)

    # Block 3: EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL メモリレイアウト
    proto_box = FancyBboxPatch((69, 38), 28, 54, boxstyle="round,pad=0.6", facecolor="#FFF8E1", edgecolor="#F57F17", lw=2)
    ax.add_patch(proto_box)
    ax.text(83, 89, "EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL\n(メモリ配置: 基底 0x3F80_1000)", fontsize=10, fontweight="bold", ha="center", color="#F57F17")

    proto_cells = [
        (72, 73, 22, 9, "dummy (8 Bytes, 0x00〜0x07)\nReset() 関数ポインタ\n(未呼出のためダミー)", "#FFE082", "#E65100"),
        (72, 54, 22, 16, "OutputString (8 Bytes, 0x08〜0x0F)\nEFI_TEXT_STRING 関数ポインタ\n値: 0x3FE5_22A0\nOutputString(This, String)\n[UEFI ファームウェア内部コード]", "#FFCC80", "#BF360C"),
    ]
    for x, y, w, h, lbl, fc, ec in proto_cells:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.3", facecolor=fc, edgecolor=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2, lbl, fontsize=8.0, fontweight="bold", ha="center", va="center", color=ec)

    # Arrows between structures
    # From SystemTable ConOut pointer to TextOutputProtocol
    ax.annotate("", xy=(72, 62), xytext=(62, 54),
                arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2.5, connectionstyle="arc3,rad=-0.15"))
    ax.text(66, 61, "*ConOut 参照", fontsize=8.5, fontweight="bold", color="#E65100", ha="center")

    # Block 4: Lower explanation area (Wide string & Calling convention)
    call_box = FancyBboxPatch((3, 4), 94, 28, boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#2E7D32", lw=2)
    ax.add_patch(call_box)
    ax.text(50, 28, "Microsoft x64 呼び出し規約と OutputString の実行メカニズム", fontsize=11, fontweight="bold", ha="center", color="#2E7D32")

    explanation_items = [
        "① アロー演算子の評価: SystemTable (0x3F80_0000) + 60バイト -> ConOutポインタ (0x3F80_1000) を逆参照",
        "② メンバ関数ポインタ取得: ConOut (0x3F80_1000) + 8バイト -> OutputString 実装先アドレス (0x3FE5_22A0) を逆参照",
        "③ 第1引数 (RCX) = ConOut (0x3F80_1000): C言語における明示的な 'this' ポインタ渡し (オブジェクト指向設計)",
        "④ 第2引数 (RDX) = L\"Hello, world!\\n\": 16ビット UTF-16LE 文字列へのポインタ (各文字2バイト: 'H'=0x0048, 'e'=0x0065... 終端=0x0000)",
        "⑤ 画面描画と復帰: UEFI ファームウェアが GOP フレームバッファへ文字フォントを描画し、EFI_SUCCESS (0) を RAX で返却",
    ]
    for idx, text in enumerate(explanation_items):
        ax.text(6, 23.5 - idx * 4.3, text, fontsize=8.4, color="#1B5E20")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_pe32plus_coff_elf_comparison(output_path: str = "1/result/fig1_18_pe32plus_coff_elf_binary_format_comparison.png") -> str:
    """Figure 1.18: コラム1.1: PE32+ (PE/COFF) vs ELF64 バイナリ構造比較図 (DPI 300)"""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax_pe, ax_elf) = plt.subplots(1, 2, figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    # Left: PE32+ (PE/COFF)
    ax_pe.set_title("PE32+ (PE/COFF 64-bit)\n— UEFI ファームウェア & Windows の実行ファイル形式",
                    fontsize=12, fontweight="bold", pad=12, color="#0D47A1")
    ax_pe.set_xlim(0, 100)
    ax_pe.set_ylim(0, 100)
    ax_pe.axis("off")

    pe_layers = [
        (10, 88, 80, 8, "MS-DOS Header (64 Bytes)\nMagic: 'MZ' (0x5A4D) / e_lfanew -> PE Header Offset", "#E3F2FD", "#0D47A1"),
        (10, 78, 80, 8, "MS-DOS Stub Program\n'This program cannot be run in DOS mode'", "#BBDEFB", "#1565C0"),
        (10, 68, 80, 8, "PE Signature (4 Bytes)\n'PE\\0\\0' (0x00004550)", "#90CAF9", "#1565C0"),
        (10, 56, 80, 10, "COFF File Header (20 Bytes)\nMachine: x86-64 (0x8664) / NumberOfSections / TimeDateStamp", "#64B5F6", "#0D47A1"),
        (10, 36, 80, 18, "Optional Header (PE32+ 112B + Data Directories 128B)\nMagic: 0x020B (PE32+) / AddressOfEntryPoint (RVA)\nImageBase / SectionAlignment / Subsystem=10 (EFI_APP)", "#FFE082", "#E65100"),
        (10, 22, 80, 12, "Section Table (40 Bytes × N)\n.text (コード) / .data (データ) / .rdata (定数/文字列) / .reloc (再配置)", "#C8E6C9", "#2E7D32"),
        (10, 4, 80, 16, "Section Data (実際の機械語・データ本体)\n.text: EfiMain の x86-64 機械語\n.rdata: L\"Hello, world!\\n\" (UTF-16LE 文字列)", "#E8F5E9", "#1B5E20"),
    ]

    for x, y, w, h, lbl, fc, ec in pe_layers:
        ax_pe.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", facecolor=fc, edgecolor=ec, lw=1.6))
        ax_pe.text(x + w / 2, y + h / 2, lbl, fontsize=7.8, fontweight="bold", ha="center", va="center", color=ec)

    # Right: ELF64
    ax_elf.set_title("ELF64 (Executable and Linkable Format)\n— Linux / BSD / POSIX 標準の実行ファイル形式",
                     fontsize=12, fontweight="bold", pad=12, color="#B71C1C")
    ax_elf.set_xlim(0, 100)
    ax_elf.set_ylim(0, 100)
    ax_elf.axis("off")

    elf_layers = [
        (10, 82, 80, 14, "ELF Header (64 Bytes)\nMagic: 0x7F 'ELF' (0x464C457F) / 64-bit / Little Endian\ne_entry (エントリ仮想アドレス) / e_phoff / e_shoff", "#FFCDD2", "#B71C1C"),
        (10, 60, 80, 20, "Program Header Table (56 Bytes × M) [実行・ローダ用]\nPT_LOAD: 仮想アドレスへのメモリマッピング指示 (コード/データ)\nPT_DYNAMIC: 動的リンカ情報 / PT_INTERP: /lib64/ld-linux-x86-64.so.2", "#FFE0B2", "#E65100"),
        (10, 36, 80, 22, "Segments & Sections (セグメントとセクション本体)\nSegment 1 (R-X): .text, .rodata (文字列等)\nSegment 2 (RW-): .data, .bss\n(PT_LOAD セグメントの中に各セクションが束ねられる)", "#E8F5E9", "#1B5E20"),
        (10, 16, 80, 18, "Section Header Table (64 Bytes × N) [リンク用・strip可能]\n.symtab (シンボル表) / .strtab (文字列表)\n.rela.text (再配置情報) / デバッグ情報 (.debug_*)", "#D1C4E9", "#4A148C"),
        (10, 4, 80, 10, "特長: 実行ビュー(Segments) と リンクビュー(Sections) を二重構造で分離", "#ECEFF1", "#37474F"),
    ]

    for x, y, w, h, lbl, fc, ec in elf_layers:
        ax_elf.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", facecolor=fc, edgecolor=ec, lw=1.6))
        ax_elf.text(x + w / 2, y + h / 2, lbl, fontsize=8.0, fontweight="bold", ha="center", va="center", color=ec)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# =============================================================================
# Chapter 1 Comprehensive Exercises Simulation & Verification
# =============================================================================

class Chapter1Quiz:
    """Comprehensive multiple-choice quiz engine covering Sections 1.1 to 1.9."""

    def __init__(self) -> None:
        self.questions: list[QuizQuestion] = [
            QuizQuestion(
                question_id=1,
                section="1.1",
                topic="PE32+ バイナリ構造とマジックナンバー",
                question="64ビットUEFIアプリケーション（PE32+ / x86-64）のバイナリにおいて、DOSヘッダのe_magic、PEシグネチャ、Optional Header Magicの組み合わせとして正しいものはどれか？",
                options=[
                    "e_magic: 0x7F45 ('\\x7fE'), PEシグネチャ: 0x464C457F ('ELF'), Magic: 0x010B (PE32)",
                    "e_magic: 0x5A4D ('MZ'), PEシグネチャ: 0x00004550 ('PE\\0\\0'), Magic: 0x020B (PE32+)",
                    "e_magic: 0xEB58 ('\\xebX'), PEシグネチャ: 0xAA55, Magic: 0x8664 (x86-64)",
                    "e_magic: 0x5A4D ('MZ'), PEシグネチャ: 0x00004550 ('PE\\0\\0'), Magic: 0x010B (PE32)",
                ],
                correct_index=1,
                explanation="PE32+バイナリは先頭2バイトがDOSヘッダマジック 'MZ' (0x5A4D)、オフセット0x3Cが指す位置に 'PE\\0\\0' (0x00004550)、COFFヘッダ直後のOptional HeaderのMagicが0x020B（64-bit PE32+）となります（32-bit PE32は0x010B）。",
            ),
            QuizQuestion(
                question_id=2,
                section="1.1",
                topic="Microsoft x64 呼出規約とスタック整列",
                question="UEFIファームウェアが採用する Microsoft x64 呼出規約において、整数第1〜第4引数を渡すレジスタとスタック整列の要件として正しいものはどれか？",
                options=[
                    "引数: RDI, RSI, RDX, RCX / スタックは 8バイト境界で整列",
                    "引数: RCX, RDX, R8, R9 / 32バイトのシャドウスペース確保 & call直前にRSPを16バイト境界に整列",
                    "引数: RAX, RBX, RCX, RDX / スタック渡しのみ使用しレジスタは使わない",
                    "引数: RCX, RDX, RSI, RDI / 64バイトのシャドウスペース確保 & 32バイト整列",
                ],
                correct_index=1,
                explanation="Microsoft x64 ABIでは、先頭4引数をRCX, RDX, R8, R9で渡し、呼び出し側は関数呼び出し直前に32バイト（4引数分）のシャドウスペースをスタックに確保し、かつCALL命令発行時点でRSPが16バイト境界（整列）になっている必要があります。",
            ),
            QuizQuestion(
                question_id=3,
                section="1.2",
                topic="Linux ブロックデバイス管理",
                question="Linuxカーネルにおけるブロックデバイス（/dev/sdX 等）の管理において、メジャー番号（Major）とマイナー番号（Minor）の役割として正しいものはどれか？",
                options=[
                    "メジャー番号はファイルシステム種別（ext4/FAT）を表し、マイナー番号はマウントポイントIDを表す",
                    "メジャー番号はブロックサイズを表し、マイナー番号はセクタ総数を表す",
                    "メジャー番号は対応するデバイスドライバ（SCSIディスク等）を識別し、マイナー番号はそのドライバ配下の特定ドライブやパーティションを識別する",
                    "メジャー番号はUSBバスポート番号を表し、マイナー番号は転送速度（MB/s）を表す",
                ],
                correct_index=2,
                explanation="Linuxのdev_tはメジャー番号（ドライバ識別子、例: 8はSCSIディスクドライバ）とマイナー番号（ドライブsda, sdbやパーティションsda1等を区別するID）で構成されています。",
            ),
            QuizQuestion(
                question_id=4,
                section="1.3",
                topic="WSL1 vs WSL2 アーキテクチャ",
                question="WSL2上で自作OSやUEFI開発を行う際、WSL1と比較した内部アーキテクチャ上の決定的な特徴として最も適切なものはどれか？",
                options=[
                    "WSL2はWindows NTカーネルがLinuxシステムコールを直接トランスレーションしている",
                    "WSL2はHyper-Vベースの軽量仮想マシン上で完全なLinuxカーネルが動作しており、USBデバイスの直結にはUSB/IPが必要である",
                    "WSL2ではWindowsホストのCドライブへのアクセス速度がWSL1よりも常に高速である",
                    "WSL2は64ビットLong Modeをサポートしておらず32ビット専用である",
                ],
                correct_index=1,
                explanation="WSL1はNTカーネル上のLXCoreトランスレーション層でしたが、WSL2はHyper-V Utility VM上で本物のLinuxカーネルが動作しています。そのためUSB物理ドライブのパススルーにはusbipd-win等のUSB/IPツールが必要です。",
            ),
            QuizQuestion(
                question_id=5,
                section="1.4",
                topic="QEMU と OVMF UEFI ブート",
                question="QEMU仮想マシン上でMikanOSのブートローダをテストする際、ファームウェアとして指定する OVMF (Open Virtual Machine Firmware) の役割は何か？",
                options=[
                    "LinuxカーネルのbzImageを直接メモリに展開するGRUB互換ローダ",
                    "EDK IIベースのオープンソースUEFIファームウェアであり、仮想PCハードウェアの初期化、GOP画面提供、FAT32からのBOOTX64.EFI探索を実行する",
                    "CPU命令を1命令ずつインタープリタ実行するソフトウェアJITコンパイラ",
                    "FATファイルシステムをNTFSにリアルタイム変換するファイルシステムドライバ",
                ],
                correct_index=1,
                explanation="OVMFはTianoCore EDK IIの仮想マシン向けUEFIファームウェア実装です。QEMUに`-bios OVMF.fd`またはpflashで与えることで、実機UEFI PCと全く同じブートシーケンスを仮想環境上で再現できます。",
            ),
            QuizQuestion(
                question_id=6,
                section="1.5",
                topic="CPUリセットベクタとUEFI PIフェーズ",
                question="x86-64 CPUが電源投入された直後に最初に命令フェッチを開始する物理アドレス（リセットベクタ）と、UEFI Platform Initialization (PI) の実行順序として正しいものはどれか？",
                options=[
                    "リセットベクタ: 0x00000000 / フェーズ: DXE → PEI → SEC → BDS",
                    "リセットベクタ: 0xFFFFFFF0 / フェーズ: SEC (Security) → PEI (Pre-EFI) → DXE (Driver Execution) → BDS (Boot Device Select)",
                    "リセットベクタ: 0x00007C00 / フェーズ: MBR → VBR → IPL → Kernel",
                    "リセットベクタ: 0x00100000 / フェーズ: TSL → RT → AL → OS",
                ],
                correct_index=1,
                explanation="CPUリセットベクタは物理アドレス 0xFFFFFFF0（4GB直下の16バイト前）です。UEFIファームウェアはSEC（セキュリティ・CPUキャッシュのRAM化）→ PEI（主メモリ初期化）→ DXE（各種ドライバロード）→ BDS（起動デバイス探索とBOOTX64.EFI実行）の順で進行します。",
            ),
            QuizQuestion(
                question_id=7,
                section="1.6",
                topic="UTF-16LE 文字コードとバイナリ改変",
                question="UEFIのConOut->OutputString関数が受け取る文字列ポインタの仕様として正しいものはどれか？",
                options=[
                    "ASCIIコード（1文字1バイト）で表現され、0x00で終端する",
                    "UTF-8コード（1〜4バイト可変長）で表現され、改行はLF (0x0A) 単独である",
                    "UTF-16LE（1文字2バイトの小端）で表現され、2バイトの0x0000で終端し、改行はCR+LF (0x000D, 0x000A) である",
                    "Shift-JISコード（2バイト）で表現され、EOT (0x04) で終端する",
                ],
                correct_index=2,
                explanation="UEFI仕様書において、画面出力用文字列（CHAR16*）はUCS-2/UTF-16LEで規定されています。1文字2バイト（リトルエンディアン）であり、終端は2バイトのヌル文字（0x0000）、改行はCR+LF (L\"\\r\\n\") となります。",
            ),
            QuizQuestion(
                question_id=8,
                section="1.7",
                topic="MBR vs GPT ディスクパーティション数理",
                question="レガシーMBRの最大ディスク容量制限が約2TiBであるのに対し、UEFIで標準採用されるGPT（GUID Partition Table）の最大容量制限の理論値はどれか？",
                options=[
                    "MBR: 16ビットセクタ（32MiB） / GPT: 32ビットセクタ（2TiB）",
                    "MBR: 32ビットLBA × 512B（2TiB） / GPT: 64ビットLBA × 512B（8ZiB = 約 9.44 × 10^21 バイト）",
                    "MBR: 48ビットLBA × 512B（128PiB） / GPT: 128ビットLBA（無限）",
                    "MBR: 32ビットLBA × 4KiB（16TiB） / GPT: 64ビットLBA × 4KiB（64ZiB）",
                ],
                correct_index=1,
                explanation="MBRはLBAセクタ番号が32ビット（2^32 × 512B = 2TiB）に制限されていました。一方GPTは64ビットLBAを採用しており、2^64 × 512B = 8 ZiB（ゼビバイト = 9.44 × 10^21 バイト）という膨大な容量を扱えます。",
            ),
            QuizQuestion(
                question_id=9,
                section="1.8",
                topic="Clang フリースタンディング 4大フラグ",
                question="ClangでベアメタルUEFIバイナリをビルドする際、-mno-red-zone を指定する決定的な理由はどれか？",
                options=[
                    "Red Zoneを使用するとコンパイル後のバイナリサイズが128KB増大してしまうため",
                    "System V AMD64 ABIで許容されているRSP下方の128バイト保護領域（Red Zone）を、非同期割り込みやUEFIファームウェア呼出によるスタック破壊から防ぐため",
                    "Red Zoneは32ビット保護モード専用のレガシー機能でありx86-64では未定義命令例外を発生させるため",
                    "Red Zoneを有効にするとCPUのキャッシュミス率が跳ね上がるため",
                ],
                correct_index=1,
                explanation="x86-64 ABIのRed Zone（スタックポインタRSPより下方の128バイト）は、割り込みハンドラが同一スタックを使用する場合にスタック破壊を引き起こす致命的リスクがあります。ベアメタルOS開発では `-mno-red-zone` が不可欠です。",
            ),
            QuizQuestion(
                question_id=10,
                section="1.9",
                topic="最小C言語UEFIアプリとEFI_SYSTEM_TABLE",
                question="64ビットUEFIにおける EFI_SYSTEM_TABLE 構造体から ConOut ポインタを取得し、OutputString を呼び出す際のポインタ演算・呼び出し規約として正しいものはどれか？",
                options=[
                    "ConOut は先頭オフセット 0バイトにあり、第一引数には出力文字列のみを渡す",
                    "EFI_TABLE_HEADER (24B) とファームウェア情報の後、オフセット 64バイト (0x40) に ConOut があり、OutputString の第一引数には This ポインタ (ConOut 自身) を渡す",
                    "ConOut はオフセット 52バイトにあり、32ビットポインタであるため64ビットレジスタの上位半分をクリアして呼ぶ",
                    "EFI_SYSTEM_TABLE はスタックに値渡しされるため、レジスタ RAX のオフセット 16バイトを参照する",
                ],
                correct_index=1,
                explanation="64ビットUEFIではポインタが8バイト整列されます。EFI_TABLE_HEADERが24バイト、FirmwareVendor(8B)＋FirmwareRevision(4B+4Bパディング)＋ConsoleInHandle(8B)＋ConIn(8B)＋ConsoleOutHandle(8B)の計64バイト位置にConOutが存在します。またインターフェース関数第1引数には必ずThisポインタを渡します。",
            ),
        ]

    def evaluate(self, answers: dict[int, int]) -> dict[str, any]:
        """Evaluate user answers and return comprehensive scores, explanations, and mastery breakdown."""
        details = []
        correct_count = 0
        section_scores: dict[str, dict[str, int]] = {
            "1.1": {"correct": 0, "total": 0},
            "1.2": {"correct": 0, "total": 0},
            "1.3": {"correct": 0, "total": 0},
            "1.4": {"correct": 0, "total": 0},
            "1.5": {"correct": 0, "total": 0},
            "1.6": {"correct": 0, "total": 0},
            "1.7": {"correct": 0, "total": 0},
            "1.8": {"correct": 0, "total": 0},
            "1.9": {"correct": 0, "total": 0},
        }

        for q in self.questions:
            user_ans = answers.get(q.question_id, -1)
            is_correct = (user_ans == q.correct_index)
            if is_correct:
                correct_count += 1
                section_scores[q.section]["correct"] += 1
            section_scores[q.section]["total"] += 1

            details.append({
                "question_id": q.question_id,
                "section": q.section,
                "topic": q.topic,
                "question": q.question,
                "user_answer": user_ans,
                "correct_answer": q.correct_index,
                "is_correct": is_correct,
                "explanation": q.explanation,
            })

        total = len(self.questions)
        percentage = (correct_count / total) * 100.0 if total > 0 else 0.0

        mastery_rates: dict[str, float] = {}
        for sec, stat in section_scores.items():
            rate = (stat["correct"] / stat["total"]) * 100.0 if stat["total"] > 0 else 0.0
            mastery_rates[sec] = rate

        return {
            "total_questions": total,
            "correct_count": correct_count,
            "percentage": percentage,
            "details": details,
            "mastery_rates": mastery_rates,
            "passed": percentage >= 80.0,
        }


class Chapter1TheoryValidator:
    """Validator for Chapter 1 fill-in-the-blank theory and math exercises."""

    @staticmethod
    def validate_pe_headers(magic_dict: dict[str, any]) -> tuple[bool, str]:
        """Validate PE32+ DOS, PE, and Optional header magic constants."""
        expected = {
            "dos_magic": 0x5A4D,          # 'MZ'
            "pe_signature": 0x00004550,    # 'PE\0\0'
            "optional_magic": 0x020B,      # PE32+ (64-bit)
        }
        errors = []
        for key, exp_val in expected.items():
            val = magic_dict.get(key)
            if isinstance(val, str):
                try:
                    val = int(val, 0)
                except ValueError:
                    errors.append(f"{key}: 数値変換不能 '{val}'")
                    continue
            if val != exp_val:
                errors.append(f"{key}: 期待値 0x{exp_val:X}, 入力値 {hex(val) if isinstance(val, int) else val}")

        if not errors:
            return True, "正解！PE32+ (64-bit UEFI) のヘッダマジック（MZ: 0x5A4D, PE: 0x4550, Optional: 0x020B）を正しく同定しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_uefi_calling_convention(conv_dict: dict[str, any]) -> tuple[bool, str]:
        """Validate Microsoft x64 Calling Convention register parameters and stack alignment."""
        expected_regs = {
            "arg1": "rcx",
            "arg2": "rdx",
            "arg3": "r8",
            "arg4": "r9",
        }
        errors = []
        for k, exp in expected_regs.items():
            u_val = str(conv_dict.get(k, "")).strip().lower()
            if u_val != exp:
                errors.append(f"{k}: 期待値 '{exp}', 入力値 '{u_val}'")

        shadow = conv_dict.get("shadow_space_bytes")
        if shadow != 32:
            errors.append(f"shadow_space_bytes: 期待値 32, 入力値 {shadow}")

        align = conv_dict.get("stack_alignment_bytes")
        if align != 16:
            errors.append(f"stack_alignment_bytes: 期待値 16, 入力値 {align}")

        if not errors:
            return True, "正解！Microsoft x64 呼出規約（RCX, RDX, R8, R9, 32Bシャドウスペース, 16Bスタック整列）に完全準拠しています。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_clang_uefi_flags(flags: list[str]) -> tuple[bool, str]:
        """Validate essential Clang / LLD flags for freestanding UEFI executable generation."""
        flag_set = {f.strip() for f in flags}
        required = {"-target x86_64-unknown-windows", "-ffreestanding", "-mno-red-zone", "-fno-stack-protector", "-fshort-wchar"}
        missing = required - flag_set
        if not missing:
            return True, "正解！Clang UEFIコンパイルの5大必須フラグが完璧に指定されています。"
        return False, f"不正解です。不足している必須フラグ: {missing}"

    @staticmethod
    def validate_mbr_vs_gpt_math(math_dict: dict[str, any]) -> tuple[bool, str]:
        """Validate MBR vs GPT addressing and theoretical capacity limits."""
        mbr_bits = math_dict.get("mbr_lba_bits")
        gpt_bits = math_dict.get("gpt_lba_bits")
        errors = []
        if mbr_bits != 32:
            errors.append(f"mbr_lba_bits: 期待値 32, 入力値 {mbr_bits}")
        if gpt_bits != 64:
            errors.append(f"gpt_lba_bits: 期待値 64, 入力値 {gpt_bits}")

        mbr_tib = math_dict.get("mbr_max_tib")
        if mbr_tib not in (2, 2.0):
            errors.append(f"mbr_max_tib: 期待値 2.0 TiB, 入力値 {mbr_tib}")

        if not errors:
            return True, "正解！MBR (32-bit LBA -> 2 TiB) と GPT (64-bit LBA -> 8 ZiB) のアドレッシング数理を正確に算出しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_system_table_layout(offsets: dict[str, int]) -> tuple[bool, str]:
        """Validate 64-bit UEFI EFI_SYSTEM_TABLE and SIMPLE_TEXT_OUTPUT_INTERFACE offsets."""
        con_out = offsets.get("con_out_offset")
        out_str = offsets.get("output_string_offset")
        errors = []
        if con_out != 64:
            errors.append(f"con_out_offset: 期待値 64 (0x40), 入力値 {con_out}")
        if out_str != 8:
            errors.append(f"output_string_offset: 期待値 8 (0x08), 入力値 {out_str}")

        if not errors:
            return True, "正解！64ビットUEFI EFI_SYSTEM_TABLE (ConOut: +64B) と OutputString (+8B) のオフセット構造に完全適合しています。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"


def plot_chapter1_exercises_summary(output_path: Path | str) -> Path:
    """Generate high-DPI summary diagram showing Chapter 1 exercise structure and learning taxonomy."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_facecolor("#FAFAFA")

    exercises = [
        {"id": "Ex 1", "sec": "1.1 / 1.7", "title": "PE32+ / GPT 理論検証", "desc": "DOS/PE/Optionalマジック同定、MBR 2TiB vs GPT 8ZiB LBA数理導出", "color": "#1E88E5", "y": 0.84},
        {"id": "Ex 2", "sec": "1.1 / 1.8", "title": "ABI & Clang フラグ検証", "desc": "Microsoft x64呼出規約（RCX/RDX/R8/R9/32B/16B）、4大必須フラグ判定", "color": "#43A047", "y": 0.67},
        {"id": "Ex 3", "sec": "1.2 / 1.6", "title": "Linuxデバイス & 文字列", "desc": "dev_tメジャー・マイナー数理、UTF-16LEエンコード & ヌル終端検証", "color": "#FB8C00", "y": 0.50},
        {"id": "Ex 4", "sec": "1.9", "title": "UEFI SystemTable 構造", "desc": "64-bit EFI_SYSTEM_TABLE (ConOut: +64B, OutputString: +8B) 構造体計算", "color": "#8E24AA", "y": 0.33},
        {"id": "Ex 5", "sec": "1.1-1.9", "title": "C++20 コア検証 & 総合テスト", "desc": "x64 ABI/PEヘッダ C++シミュレータ、全節網羅10問自己採点クイズ & レーダー", "color": "#D81B60", "y": 0.16},
    ]

    import matplotlib.patches as patches

    for ex in exercises:
        y = ex["y"]
        rect = patches.FancyBboxPatch(
            (0.05, y - 0.06), 0.90, 0.12,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="white", edgecolor=ex["color"], linewidth=2, zorder=2
        )
        ax.add_patch(rect)

        badge = patches.FancyBboxPatch(
            (0.07, y - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=ex["color"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(0.13, y, ex["id"], color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

        ax.text(0.22, y + 0.02, f"[{ex['sec']}] {ex['title']}", color="#212121", fontsize=12, fontweight="bold", va="center")
        ax.text(0.22, y - 0.025, ex["desc"], color="#555555", fontsize=9.5, va="center")

        circle = plt.Circle((0.91, y), 0.025, facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(0.91, y, "✓", color="#2E7D32", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.05, 0.97)
    ax.axis("off")
    ax.set_title("MikanOS Study — Chapter 1 Exercises Curriculum & Assessment Map",
                 fontsize=14, fontweight="bold", pad=20, color="#1A237E")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_chapter1_mastery_radar(output_path: Path | str, mastery_rates: dict[str, float] | None = None) -> Path:
    """Generate high-DPI 9-axis radar chart showing Chapter 1 concept mastery across all 9 sections."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        "1.1 PE32+ バイナリ\n(PE32+ & x64 ABI)",
        "1.2 ブロックデバイス\n(Block Devices & dev_t)",
        "1.3 WSL2 環境\n(WSL2 & USB/IP)",
        "1.4 QEMU & OVMF\n(QEMU & OVMF Boot)",
        "1.5 リセット & PI\n(Reset & PI Phases)",
        "1.6 UTF-16LE 観察\n(UTF-16LE & Mutation)",
        "1.7 UEFI vs BIOS\n(MBR vs GPT 8ZiB)",
        "1.8 Clang ツール\n(Freestanding Toolchain)",
        "1.9 最小 C 実装\n(EFI_SYSTEM_TABLE)",
    ]
    num_vars = len(categories)

    if mastery_rates is None:
        mastery_rates = {f"1.{i}": 100.0 for i in range(1, 10)}

    values = [mastery_rates.get(f"1.{i}", 100.0) for i in range(1, 10)]
    values += values[:1]  # Close loop

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FCFCFD")

    # Grid lines and levels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#757575", size=9)
    plt.ylim(0, 105)

    plt.xticks(angles[:-1], categories, color="#212121", size=9.5, fontweight="bold")

    # Plot data
    ax.plot(angles, values, color="#1976D2", linewidth=2.5, linestyle="solid", marker="o", markersize=6)
    ax.fill(angles, values, color="#2196F3", alpha=0.25)

    # Threshold circle (80%)
    threshold_vals = [80.0] * (num_vars + 1)
    ax.plot(angles, threshold_vals, color="#E53935", linewidth=1.2, linestyle="--", label="Pass Threshold (80%)")

    avg_score = float(np.mean([mastery_rates.get(f"1.{i}", 100.0) for i in range(1, 10)]))
    plt.title(f"Chapter 1 Concept Mastery Radar (Average: {avg_score:.1f}%)\n[x86-64 / UEFI / PE32+ / Freestanding Toolchain]",
              size=13, fontweight="bold", pad=25, color="#0D47A1")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 2.1: EDK II Architecture & Build System Models
# ==============================================================================

class Edk2ModuleType(str, Enum):
    """EDK II Module Types as specified by UEFI Platform Initialization (PI) and EDK II specifications."""
    BASE = "BASE"
    SEC = "SEC"
    PEI_CORE = "PEI_CORE"
    PEIM = "PEIM"
    DXE_CORE = "DXE_CORE"
    DXE_DRIVER = "DXE_DRIVER"
    DXE_RUNTIME_DRIVER = "DXE_RUNTIME_DRIVER"
    DXE_SMM_DRIVER = "DXE_SMM_DRIVER"
    UEFI_DRIVER = "UEFI_DRIVER"
    UEFI_APPLICATION = "UEFI_APPLICATION"
    USER_DEFINED = "USER_DEFINED"


class Edk2Architecture(str, Enum):
    """Target CPU Architectures supported by EDK II."""
    IA32 = "IA32"
    X64 = "X64"
    ARM = "ARM"
    AARCH64 = "AARCH64"
    RISCV64 = "RISCV64"


@dataclass
class Edk2DecModel:
    """EDK II Package Declaration File (.dec) model."""
    package_name: str
    package_guid: str
    package_version: str = "0.1"
    dec_specification: str = "0x00010005"
    includes: list[str] = field(default_factory=list)
    guids: dict[str, str] = field(default_factory=dict)
    protocols: dict[str, str] = field(default_factory=dict)
    pcds: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse_dec_text(cls, text: str) -> Edk2DecModel:
        """Parse raw .dec file text into an Edk2DecModel instance."""
        defines: dict[str, str] = {}
        includes: list[str] = []
        guids: dict[str, str] = {}
        protocols: dict[str, str] = {}
        pcds: dict[str, str] = {}

        current_section = ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                continue

            if current_section == "defines":
                if "=" in line:
                    k, v = line.split("=", 1)
                    defines[k.strip()] = v.strip()
            elif current_section.startswith("includes"):
                includes.append(line)
            elif current_section.startswith("guids"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    guids[k.strip()] = v.strip()
            elif current_section.startswith("protocols"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    protocols[k.strip()] = v.strip()
            elif "pcd" in current_section:
                if "=" in line:
                    k, v = line.split("=", 1)
                    pcds[k.strip()] = v.strip()

        return cls(
            package_name=defines.get("PACKAGE_NAME", "UnknownPackage"),
            package_guid=defines.get("PACKAGE_GUID", ""),
            package_version=defines.get("PACKAGE_VERSION", "0.1"),
            dec_specification=defines.get("DEC_SPECIFICATION", "0x00010005"),
            includes=includes,
            guids=guids,
            protocols=protocols,
            pcds=pcds,
        )


@dataclass
class Edk2InfModel:
    """EDK II Module Information File (.inf) model."""
    base_name: str
    file_guid: str
    module_type: str = "UEFI_APPLICATION"
    version_string: str = "0.1"
    entry_point: str = "UefiMain"
    inf_version: str = "0x00010006"
    sources: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    library_classes: list[str] = field(default_factory=list)
    protocols: list[str] = field(default_factory=list)
    guids: list[str] = field(default_factory=list)

    @classmethod
    def parse_inf_text(cls, text: str) -> Edk2InfModel:
        """Parse raw .inf file text into an Edk2InfModel instance."""
        defines: dict[str, str] = {}
        sources: list[str] = []
        packages: list[str] = []
        library_classes: list[str] = []
        protocols: list[str] = []
        guids: list[str] = []

        current_section = ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                continue

            if current_section == "defines":
                if "=" in line:
                    k, v = line.split("=", 1)
                    defines[k.strip()] = v.strip()
            elif current_section.startswith("sources"):
                sources.append(line)
            elif current_section.startswith("packages"):
                packages.append(line)
            elif current_section.startswith("libraryclasses"):
                library_classes.append(line)
            elif current_section.startswith("protocols"):
                protocols.append(line)
            elif current_section.startswith("guids"):
                guids.append(line)

        return cls(
            base_name=defines.get("BASE_NAME", "UnknownModule"),
            file_guid=defines.get("FILE_GUID", ""),
            module_type=defines.get("MODULE_TYPE", "UEFI_APPLICATION"),
            version_string=defines.get("VERSION_STRING", "0.1"),
            entry_point=defines.get("ENTRY_POINT", "UefiMain"),
            inf_version=defines.get("INF_VERSION", "0x00010006"),
            sources=sources,
            packages=packages,
            library_classes=library_classes,
            protocols=protocols,
            guids=guids,
        )


@dataclass
class Edk2DscModel:
    """EDK II Platform Description File (.dsc) model."""
    platform_name: str
    platform_guid: str
    platform_version: str = "0.1"
    dsc_specification: str = "0x00010005"
    output_directory: str = "Build/MikanLoader$(ARCH)"
    supported_architectures: list[str] = field(default_factory=lambda: ["X64"])
    build_targets: list[str] = field(default_factory=lambda: ["DEBUG", "RELEASE", "NOOPT"])
    library_classes: dict[str, str] = field(default_factory=dict)
    components: list[str] = field(default_factory=list)

    @classmethod
    def parse_dsc_text(cls, text: str) -> Edk2DscModel:
        """Parse raw .dsc file text into an Edk2DscModel instance."""
        defines: dict[str, str] = {}
        library_classes: dict[str, str] = {}
        components: list[str] = []

        current_section = ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                continue

            if current_section == "defines":
                if "=" in line:
                    k, v = line.split("=", 1)
                    defines[k.strip()] = v.strip()
            elif current_section.startswith("libraryclasses"):
                if "|" in line:
                    cls_name, inf_path = line.split("|", 1)
                    library_classes[cls_name.strip()] = inf_path.strip()
            elif current_section.startswith("components"):
                components.append(line)

        archs = [a.strip() for a in defines.get("SUPPORTED_ARCHITECTURES", "X64").split("|")]
        targets = [t.strip() for t in defines.get("BUILD_TARGETS", "DEBUG|RELEASE|NOOPT").split("|")]

        return cls(
            platform_name=defines.get("PLATFORM_NAME", "UnknownPlatform"),
            platform_guid=defines.get("PLATFORM_GUID", ""),
            platform_version=defines.get("PLATFORM_VERSION", "0.1"),
            dsc_specification=defines.get("DSC_SPECIFICATION", "0x00010005"),
            output_directory=defines.get("OUTPUT_DIRECTORY", "Build/MikanLoader$(ARCH)"),
            supported_architectures=archs,
            build_targets=targets,
            library_classes=library_classes,
            components=components,
        )


@dataclass
class DscDependencyResolver:
    """Dependency resolver connecting Module INF requirements with Platform DSC bindings."""
    dsc: Edk2DscModel
    modules: dict[str, Edk2InfModel] = field(default_factory=dict)

    def register_module(self, path: str, module: Edk2InfModel) -> None:
        """Register a module INF model under a specific path."""
        self.modules[path] = module

    def resolve_dependencies(self, component_path: str) -> dict[str, Any]:
        """Resolve all library classes required by a component against the DSC library bindings."""
        if component_path not in self.modules:
            return {
                "component": component_path,
                "is_valid": False,
                "error": f"Component {component_path} not found in registered modules",
                "resolved": {},
                "missing": [],
            }

        module = self.modules[component_path]
        resolved: dict[str, str] = {}
        missing: list[str] = []

        for req_class in module.library_classes:
            if req_class in self.dsc.library_classes:
                resolved[req_class] = self.dsc.library_classes[req_class]
            else:
                missing.append(req_class)

        return {
            "component": component_path,
            "base_name": module.base_name,
            "is_valid": len(missing) == 0,
            "resolved": resolved,
            "missing": missing,
            "total_required": len(module.library_classes),
            "total_resolved": len(resolved),
        }

    def get_full_build_tree(self) -> dict[str, Any]:
        """Get the full dependency resolution tree for all components listed in DSC."""
        tree: dict[str, Any] = {}
        all_valid = True
        for comp in self.dsc.components:
            res = self.resolve_dependencies(comp)
            tree[comp] = res
            if not res.get("is_valid", False):
                all_valid = False
        return {
            "platform_name": self.dsc.platform_name,
            "all_valid": all_valid,
            "components": tree,
        }


@dataclass
class Edk2BuildPipelineSimulator:
    """Simulates the 6-step EDK II BaseTools build pipeline."""
    workspace: str = "/home/student/edk2"
    edk_tools_path: str = "/home/student/edk2/BaseTools"
    target_arch: str = "X64"
    target_toolchain: str = "CLANG38"
    build_target: str = "DEBUG"

    def simulate_build(self, dsc: Edk2DscModel, inf: Edk2InfModel) -> dict[str, Any]:
        """Simulate execution of the 'build' command for the specified DSC and main INF."""
        steps: list[dict[str, Any]] = []

        # Step 1: Environment Inspection
        env_valid = bool(self.workspace and self.edk_tools_path)
        steps.append({
            "step": 1,
            "name": "Environment Validation",
            "command": "edksetup.sh",
            "detail": f"WORKSPACE={self.workspace}, EDK_TOOLS_PATH={self.edk_tools_path}",
            "status": "PASS" if env_valid else "FAIL",
        })

        # Step 2: DSC Parsing & Target Resolution
        arch_supported = self.target_arch in dsc.supported_architectures
        target_supported = self.build_target in dsc.build_targets
        steps.append({
            "step": 2,
            "name": "Platform & Target Resolution",
            "command": f"build -p {dsc.platform_name}.dsc -a {self.target_arch} -b {self.build_target} -t {self.target_toolchain}",
            "detail": f"Target Arch: {self.target_arch} (Supported: {arch_supported}), Mode: {self.build_target} (Supported: {target_supported})",
            "status": "PASS" if (arch_supported and target_supported) else "FAIL",
        })

        # Step 3: Library Class Dependency Injection
        resolver = DscDependencyResolver(dsc=dsc)
        resolver.register_module(f"{dsc.platform_name}/{inf.base_name}.inf", inf)
        res = resolver.resolve_dependencies(f"{dsc.platform_name}/{inf.base_name}.inf")
        steps.append({
            "step": 3,
            "name": "Library Dependency Resolution",
            "command": "Dependency Injection (LibraryClasses -> Instances)",
            "detail": f"Required {res['total_required']} classes, Resolved {res['total_resolved']} instances (Missing: {res['missing']})",
            "status": "PASS" if res["is_valid"] else "FAIL",
        })

        # Step 4: AutoGen Code Generation
        out_dir = dsc.output_directory.replace("$(ARCH)", self.target_arch)
        module_build_dir = f"{out_dir}/{self.build_target}_{self.target_toolchain}/{self.target_arch}/{dsc.platform_name}/{inf.base_name}"
        autogen_files = [f"{module_build_dir}/AutoGen.c", f"{module_build_dir}/AutoGen.h", f"{module_build_dir}/Makefile"]
        steps.append({
            "step": 4,
            "name": "AutoGen Synthesizer",
            "command": "BaseTools/Bin/build.py (AutoGen phase)",
            "detail": f"Generated {len(autogen_files)} AutoGen artifacts (Entry vector, GUID descriptors, dependency Makefile)",
            "generated_files": autogen_files,
            "status": "PASS",
        })

        # Step 5: Compilation & Native LLD Linking
        compile_cmd = f"clang -target x86_64-pc-win32-coff -c {inf.sources[0]} -o {module_build_dir}/{inf.base_name}.obj"
        link_cmd = f"lld-link /out:{module_build_dir}/{inf.base_name}.dll /subsystem:efi_application {module_build_dir}/{inf.base_name}.obj"
        steps.append({
            "step": 5,
            "name": "Compile & Link Execution",
            "command": f"{compile_cmd} && {link_cmd}",
            "detail": f"Compiled {len(inf.sources)} source files into COFF object and linked into PE/COFF image",
            "status": "PASS",
        })

        # Step 6: GenFw Post-Processing & Final EFI Image Creation
        final_efi = f"{module_build_dir}/{inf.base_name}.efi"
        genfw_cmd = f"GenFw -e EFI_APPLICATION -o {final_efi} {module_build_dir}/{inf.base_name}.dll"
        steps.append({
            "step": 6,
            "name": "GenFw Image Packaging",
            "command": genfw_cmd,
            "detail": f"Packaged final executable: {final_efi}",
            "output_binary": final_efi,
            "status": "PASS",
        })

        success = all(s["status"] == "PASS" for s in steps)
        return {
            "success": success,
            "final_efi": final_efi if success else None,
            "steps": steps,
        }


@dataclass
class BuildApproachComparator:
    """Comprehensive comparison between Chapter 1 manual Clang build and Chapter 2 EDK II build."""
    criteria: list[dict[str, str]] = field(default_factory=lambda: [
        {
            "category": "ツールチェーン & 駆動方式",
            "ch1_manual": "Clang + LLD をシェルから直接ワンライナー実行",
            "ch2_edk2": "Python/C による BaseTools (`build`) 駆動のメタビルド",
            "advantage": "EDK II は数千ファイルの巨大ファームウェアでも破綻しない",
        },
        {
            "category": "UEFI 仕様準拠と標準ヘッダ",
            "ch1_manual": "必要な構造体（`EFI_SYSTEM_TABLE` 等）と型を手書き定義",
            "ch2_edk2": "`MdePkg` (Module Dev Env) による全仕様ヘッダの完全提供",
            "advantage": "EDK II は仕様変更や拡張プロトコル（GOP, FS, USB）に即応可能",
        },
        {
            "category": "ライブラリ抽象化 (DI)",
            "ch1_manual": "なし（関数ポインタ直接呼び出し）",
            "ch2_edk2": "LibraryClass 抽象インターフェースと DSC での実装注入",
            "advantage": "実機デバッグ用とNULLスタブ用ライブラリをDSC一行で切替可能",
        },
        {
            "category": "マルチアーキテクチャ移植性",
            "ch1_manual": "x86-64 専用フラグをハードコード",
            "ch2_edk2": "DSC/INF で X64, IA32, ARM, AARCH64, RISCV64 を透過サポート",
            "advantage": "CPUアーキテクチャ非依存の共通ローダ設計が可能",
        },
        {
            "category": "学習コストと初期設定",
            "ch1_manual": "極めて低い（LLVMさえあれば1秒で実行可能）",
            "ch2_edk2": "高い（BaseToolsビルド、環境変数、DSC/INF構文の習得が必要）",
            "advantage": "第1章は入門に最適、第2章以降の実践OS開発にはEDK IIが必須",
        },
    ])


def plot_edk2_architecture_and_packages(output_path: Path | str) -> Path:
    """Generate high-DPI architecture diagram illustrating EDK II Package hierarchy (DEC/INF/DSC) and DI."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#F8F9FA")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    ax.text(50, 96, "EDK II Package Architecture & Dependency Injection Model",
            fontsize=15, fontweight="bold", ha="center", va="center", color="#0D47A1")
    ax.text(50, 92, "DEC (Interface Declaration) ↔ INF (Module Spec) ↔ DSC (Platform Description & Library Binding)",
            fontsize=10.5, ha="center", va="center", color="#424242", style="italic")

    # Left Container: MdePkg (Standard Foundation)
    p1 = FancyBboxPatch((4, 25), 42, 62, boxstyle="round,pad=1.5",
                        facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2.0)
    ax.add_patch(p1)
    ax.text(25, 83, "MdePkg (Module Development Environment Package)",
            fontsize=11.5, fontweight="bold", ha="center", color="#0D47A1")
    ax.text(25, 79, "UEFI Specification Standard Headers & Base Libraries",
            fontsize=9.5, ha="center", color="#1976D2")

    # MdePkg sub-components
    # DEC Box
    b_dec = FancyBboxPatch((6, 62), 38, 13, boxstyle="round,pad=0.8",
                           facecolor="#FFFFFF", edgecolor="#1976D2", linewidth=1.5)
    ax.add_patch(b_dec)
    ax.text(8, 71, "[DEC] MdePkg.dec (Package Declaration)", fontsize=10.5, fontweight="bold", color="#0D47A1")
    ax.text(8, 66.5, "• [Includes]: Include/ (Uefi.h, PiPei.h, PiDxe.h)\n• [Guids], [Protocols]: gEfiLoadedImageProtocolGuid ...",
            fontsize=8.5, color="#37474F")

    # Library Instances Box
    b_libs = FancyBboxPatch((6, 29), 38, 30, boxstyle="round,pad=0.8",
                            facecolor="#FFFFFF", edgecolor="#1976D2", linewidth=1.5)
    ax.add_patch(b_libs)
    ax.text(8, 55, "[LIBS] Library Implementations (Instances)", fontsize=10.5, fontweight="bold", color="#0D47A1")
    ax.text(8, 50.5, "• UefiApplicationEntryPoint.inf (main wrapper)\n• UefiLib.inf (Print, AsciiPrint, StatusToString)\n• BaseLib.inf (CPU intrinsic: AsmCli, AsmHlt, AsmReadCr3)\n• BaseMemoryLib.inf (CopyMem, SetMem, CompareMem)\n• UefiBootServicesTableLib.inf (gBS global pointer)\n• BasePrintLib.inf, DevicePathLib.inf ...",
            fontsize=8.5, color="#37474F")

    # Right Container: MikanLoaderPkg (User Custom OS Loader)
    p2 = FancyBboxPatch((54, 25), 42, 62, boxstyle="round,pad=1.5",
                        facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2.0)
    ax.add_patch(p2)
    ax.text(75, 83, "MikanLoaderPkg (Custom Loader Package)",
            fontsize=11.5, fontweight="bold", ha="center", color="#E65100")
    ax.text(75, 79, "OS Loader Application & Platform Description",
            fontsize=9.5, ha="center", color="#EF6C00")

    # MikanLoaderPkg sub-components
    # DEC Box
    b_mdec = FancyBboxPatch((56, 68), 38, 7.5, boxstyle="round,pad=0.8",
                            facecolor="#FFFFFF", edgecolor="#F57C00", linewidth=1.2)
    ax.add_patch(b_mdec)
    ax.text(58, 72.5, "[DEC] MikanLoaderPkg.dec", fontsize=10, fontweight="bold", color="#E65100")
    ax.text(58, 69.5, "Defines PACKAGE_NAME, PACKAGE_GUID, PACKAGE_VERSION", fontsize=8.5, color="#4E342E")

    # INF Box
    b_inf = FancyBboxPatch((56, 45), 38, 20.5, boxstyle="round,pad=0.8",
                           facecolor="#FFFFFF", edgecolor="#F57C00", linewidth=1.5)
    ax.add_patch(b_inf)
    ax.text(58, 62, "[INF] Loader.inf (Module Information)", fontsize=10.5, fontweight="bold", color="#E65100")
    ax.text(58, 58, "• MODULE_TYPE = UEFI_APPLICATION, ENTRY_POINT = UefiMain\n• [Sources]: Main.c\n• [Packages]: MdePkg/MdePkg.dec\n• [LibraryClasses]: (Abstract Requirements)\n    - UefiLib\n    - UefiApplicationEntryPoint",
            fontsize=8.5, color="#4E342E")

    # DSC Box
    b_dsc = FancyBboxPatch((56, 28), 38, 14.5, boxstyle="round,pad=0.8",
                           facecolor="#FFFDE7", edgecolor="#F57F17", linewidth=1.8)
    ax.add_patch(b_dsc)
    ax.text(58, 39, "[DSC] MikanLoaderPkg.dsc (Dependency Injection)", fontsize=10.5, fontweight="bold", color="#F57F17")
    ax.text(58, 35, "• [Components]: MikanLoaderPkg/Loader.inf\n• [LibraryClasses]: (Binding Abstract → Concrete)\n    UefiLib | MdePkg/.../UefiLib.inf\n    UefiApplicationEntryPoint | MdePkg/.../EntryPoint.inf",
            fontsize=8.5, color="#5D4037")

    # Arrows representing Dependency Injection / Resolution
    ax.annotate("", xy=(44.5, 43), xytext=(55.5, 33),
                arrowprops=dict(arrowstyle="->", color="#C2185B", lw=2.5, ls="--"))
    ax.text(48, 39, "DSC Binds\nAbstract → Concrete", fontsize=9, fontweight="bold",
            color="#C2185B", ha="center", bbox=dict(boxstyle="square,pad=0.2", facecolor="#FCE4EC", edgecolor="#C2185B"))

    ax.annotate("", xy=(55.5, 52), xytext=(44.5, 68),
                arrowprops=dict(arrowstyle="->", color="#1976D2", lw=2.0))
    ax.text(51, 62, "INF Includes\nStandard DEC", fontsize=8.5, fontweight="bold",
            color="#1976D2", ha="center")

    # Bottom Legend / Summary
    b_sum = FancyBboxPatch((4, 4), 92, 16.5, boxstyle="round,pad=1.0",
                           facecolor="#ECEFF1", edgecolor="#607D8B", linewidth=1.5)
    ax.add_patch(b_sum)
    ax.text(6, 17, "[SUMMARY] 3大メタ設定ファイルの役割と依存性逆転原則 (Dependency Inversion Principle)",
            fontsize=10.5, fontweight="bold", color="#263238")
    ax.text(6, 13, "1. DEC (.dec): パッケージが「何を提供できるか」を世界に宣言する（公開ヘッダ、GUID、プロトコル）。", fontsize=9, color="#37474F")
    ax.text(6, 9.5, "2. INF (.inf): 個別モジュール（Loader）が「何を必要とするか」を抽象ライブラリ名で要求する（UefiLib 等）。", fontsize=9, color="#37474F")
    ax.text(6, 6, "3. DSC (.dsc): プラットフォーム全体が「抽象ライブラリ要求をどの具象実装に結びつけるか」を最終決定する（依存性注入）。", fontsize=9, color="#37474F")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_edk2_build_pipeline_comparison(output_path: Path | str) -> Path:
    """Generate high-DPI diagram comparing Chapter 1 manual build vs Chapter 2 EDK II build pipeline."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.patch.set_facecolor("#FCFCFD")

    # Chapter 1 Panel
    ax1.set_facecolor("#F5F5F5")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis("off")
    ax1.text(50, 96, "【第1章】手動 Clang/LLD 直接ビルド", fontsize=12, fontweight="bold", ha="center", color="#424242")
    ax1.text(50, 92, "Minimal Freestanding One-Liner Pipeline", fontsize=9.5, ha="center", color="#757575", style="italic")

    ch1_steps = [
        ("1. 手書きソースコード", "hello.c\n(自前 struct EFI_SYSTEM_TABLE)", "#E0E0E0", "#616161"),
        ("2. Clang 直接コンパイル", "clang -target x86_64-pc-win32-coff\n-ffreestanding -mno-red-zone -c", "#FFE0B2", "#E65100"),
        ("3. COFF オブジェクト生成", "hello.o (中間バイナリ)", "#E0E0E0", "#616161"),
        ("4. LLD 直接リンク", "lld-link /subsystem:efi_application\n/entry:EfiMain hello.o", "#C8E6C9", "#2E7D32"),
        ("5. 最終 UEFI アプリケーション", "hello.efi (PE32+ 実行ファイル)", "#BBDEFB", "#1565C0"),
    ]

    y_pos = 78
    for idx, (title, desc, bg, fg) in enumerate(ch1_steps):
        p = FancyBboxPatch((10, y_pos - 8), 80, 11, boxstyle="round,pad=0.8",
                           facecolor=bg, edgecolor=fg, linewidth=1.5)
        ax1.add_patch(p)
        ax1.text(14, y_pos + 0.5, title, fontsize=10, fontweight="bold", color=fg)
        ax1.text(14, y_pos - 5.5, desc, fontsize=8.5, color="#212121")
        if idx < len(ch1_steps) - 1:
            ax1.annotate("", xy=(50, y_pos - 10), xytext=(50, y_pos - 8),
                         arrowprops=dict(arrowstyle="->", color="#616161", lw=2.0))
        y_pos -= 15

    ax1.text(50, 5, "特徴: 高速・シンプルだが、APIが増えると手書きヘッダが破綻",
             fontsize=9, fontweight="bold", ha="center", color="#D32F2F",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#EF5350"))

    # Chapter 2 Panel
    ax2.set_facecolor("#F0F4C3")
    ax2.patch.set_alpha(0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis("off")
    ax2.text(50, 96, "【第2章】EDK II BaseTools メタビルド", fontsize=12, fontweight="bold", ha="center", color="#1B5E20")
    ax2.text(50, 92, "Industrial Standard Package & AutoGen Pipeline", fontsize=9.5, ha="center", color="#33691E", style="italic")

    ch2_steps = [
        ("1. メタ定義ファイル", "MikanLoaderPkg.dsc, Loader.inf, Main.c\nMdePkg/MdePkg.dec (標準仕様ヘッダ)", "#E8F5E9", "#2E7D32"),
        ("2. AutoGen (BaseTools build.py)", "依存解決 → AutoGen.c, AutoGen.h, Makefile 生成\n(Entry Point ラッパー合成, GUID テーブル生成)", "#FFF9C4", "#F57F17"),
        ("3. Make & Clang コンパイル", "Makefile 駆動により Main.c + 依存ライブラリ群を\n-target x86_64-pc-win32-coff で並列コンパイル", "#FFE0B2", "#E65100"),
        ("4. LLD-Link リンク", "各モジュール .obj とライブラリ群を合流\n/subsystem:efi_application で DLL/PE 出力", "#C8E6C9", "#2E7D32"),
        ("5. GenFw 変換 & 検証", "GenFw により UEFI 仕様厳密チェックと\nLoader.efi への最終パッケージング", "#BBDEFB", "#0D47A1"),
    ]

    y_pos = 78
    for idx, (title, desc, bg, fg) in enumerate(ch2_steps):
        p = FancyBboxPatch((10, y_pos - 8), 80, 11, boxstyle="round,pad=0.8",
                           facecolor=bg, edgecolor=fg, linewidth=1.5)
        ax2.add_patch(p)
        ax2.text(14, y_pos + 0.5, title, fontsize=10, fontweight="bold", color=fg)
        ax2.text(14, y_pos - 5.5, desc, fontsize=8.5, color="#212121")
        if idx < len(ch2_steps) - 1:
            ax2.annotate("", xy=(50, y_pos - 10), xytext=(50, y_pos - 8),
                         arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.0))
        y_pos -= 15

    ax2.text(50, 5, "特徴: 初期構築は重厚だが、仕様準拠・依存解決・多言語移植性が完全保証",
             fontsize=9, fontweight="bold", ha="center", color="#2E7D32",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#66BB6A"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path



# ==============================================================================
# Chapter 2.2: EDK II Hello World, UEFI Status, UTF-16, and Preprocessor Simulator
# ==============================================================================

class UefiStatus:
    """UEFI Status Code representation and bit-manipulation utilities (64-bit integer)."""

    EFI_ERROR_MASK: int = 0x8000_0000_0000_0000

    # Common UEFI Status Codes (UEFI Specification 2.8+ Table 238)
    SUCCESS: int = 0
    LOAD_ERROR: int = EFI_ERROR_MASK | 1
    INVALID_PARAMETER: int = EFI_ERROR_MASK | 2
    UNSUPPORTED: int = EFI_ERROR_MASK | 3
    BAD_BUFFER_SIZE: int = EFI_ERROR_MASK | 4
    BUFFER_TOO_SMALL: int = EFI_ERROR_MASK | 5
    NOT_READY: int = EFI_ERROR_MASK | 6
    DEVICE_ERROR: int = EFI_ERROR_MASK | 7
    WRITE_PROTECTED: int = EFI_ERROR_MASK | 8
    OUT_OF_RESOURCES: int = EFI_ERROR_MASK | 9
    VOLUME_CORRUPTED: int = EFI_ERROR_MASK | 10
    VOLUME_FULL: int = EFI_ERROR_MASK | 11
    NO_MEDIA: int = EFI_ERROR_MASK | 12
    MEDIA_CHANGED: int = EFI_ERROR_MASK | 13
    NOT_FOUND: int = EFI_ERROR_MASK | 14
    ACCESS_DENIED: int = EFI_ERROR_MASK | 15

    _NAMES: Dict[int, str] = {
        0: "EFI_SUCCESS",
        EFI_ERROR_MASK | 1: "EFI_LOAD_ERROR",
        EFI_ERROR_MASK | 2: "EFI_INVALID_PARAMETER",
        EFI_ERROR_MASK | 3: "EFI_UNSUPPORTED",
        EFI_ERROR_MASK | 4: "EFI_BAD_BUFFER_SIZE",
        EFI_ERROR_MASK | 5: "EFI_BUFFER_TOO_SMALL",
        EFI_ERROR_MASK | 6: "EFI_NOT_READY",
        EFI_ERROR_MASK | 7: "EFI_DEVICE_ERROR",
        EFI_ERROR_MASK | 8: "EFI_WRITE_PROTECTED",
        EFI_ERROR_MASK | 9: "EFI_OUT_OF_RESOURCES",
        EFI_ERROR_MASK | 10: "EFI_VOLUME_CORRUPTED",
        EFI_ERROR_MASK | 11: "EFI_VOLUME_FULL",
        EFI_ERROR_MASK | 12: "EFI_NO_MEDIA",
        EFI_ERROR_MASK | 13: "EFI_MEDIA_CHANGED",
        EFI_ERROR_MASK | 14: "EFI_NOT_FOUND",
        EFI_ERROR_MASK | 15: "EFI_ACCESS_DENIED",
    }

    @classmethod
    def is_error(cls, status: int) -> bool:
        """Return True if high bit 63 is set (UEFI Error flag)."""
        return bool(status & cls.EFI_ERROR_MASK)

    @classmethod
    def error_code(cls, status: int) -> int:
        """Extract the numeric error code without the error bit."""
        return status & ~cls.EFI_ERROR_MASK

    @classmethod
    def name(cls, status: int) -> str:
        """Return human-readable EFI_STATUS identifier."""
        return cls._NAMES.get(status, f"EFI_CUSTOM_STATUS(0x{status:016X})")

    @classmethod
    def to_hex(cls, status: int) -> str:
        """Return formatted 16-character hex representation."""
        return f"0x{status:016X}"


class UefiStringEncoder:
    """UTF-16LE string encoder and memory layout analyzer for UEFI CHAR16 wide strings."""

    @staticmethod
    def encode(text: str) -> bytes:
        """Encode Python string into null-terminated UTF-16LE byte sequence."""
        return text.encode("utf-16-le") + b"\x00\x00"

    @staticmethod
    def decode(data: bytes) -> str:
        """Decode null-terminated UTF-16LE bytes into Python string."""
        if data.endswith(b"\x00\x00"):
            data = data[:-2]
        return data.decode("utf-16-le")

    @staticmethod
    def to_char16_list(text: str) -> List[int]:
        """Convert string to list of 16-bit unsigned integer values (including null terminator)."""
        values = [ord(c) for c in text]
        values.append(0)
        return values

    @classmethod
    def analyze_memory_layout(cls, text: str) -> List[Dict[str, Any]]:
        """Analyze memory bytes, offset, character representation, and word values."""
        raw_bytes = cls.encode(text)
        records = []
        for i in range(0, len(raw_bytes), 2):
            low = raw_bytes[i]
            high = raw_bytes[i + 1] if i + 1 < len(raw_bytes) else 0
            val16 = low | (high << 8)
            char_rep = chr(val16) if 32 <= val16 <= 126 else (r"\n" if val16 == 10 else (r"\0" if val16 == 0 else f"U+{val16:04X}"))
            records.append({
                "offset": i,
                "hex_bytes": f"{low:02X} {high:02X}",
                "uint16_val": f"0x{val16:04X}",
                "char": char_rep,
            })
        return records


@dataclass
class PreprocessResult:
    """Output of CPreprocessorSimulator."""

    entry_path: str
    expanded_code: str
    source_line_count: int
    expanded_line_count: int
    include_tree: Dict[str, Any]
    included_files_in_order: List[str]
    guard_symbols_defined: List[str]


class CPreprocessorSimulator:
    """Deterministic C preprocessor simulator modeling #include, search paths, and include guards."""

    def __init__(self) -> None:
        self.files: Dict[str, str] = {}
        self.system_include_dirs: List[str] = []
        self.defined_macros: set[str] = set()

    def add_file(self, path: str, content: str) -> None:
        """Register a file in virtual file system with normalized POSIX path."""
        norm_path = os.path.normpath(path)
        self.files[norm_path] = content

    def add_include_dir(self, dir_path: str) -> None:
        """Add a directory to the system include path list (-I flags)."""
        norm_dir = os.path.normpath(dir_path)
        if norm_dir not in self.system_include_dirs:
            self.system_include_dirs.append(norm_dir)

    def resolve_include(self, include_target: str, current_dir: str, is_system: bool) -> Optional[str]:
        """Resolve #include <target> or #include "target" according to C standard rules."""
        # Rule 1: Quoted include searches current directory first
        if not is_system:
            local_candidate = os.path.normpath(os.path.join(current_dir, include_target))
            if local_candidate in self.files:
                return local_candidate

        # Rule 2: Search system include directories in order
        for inc_dir in self.system_include_dirs:
            candidate = os.path.normpath(os.path.join(inc_dir, include_target))
            if candidate in self.files:
                return candidate

        # Rule 3: Exact path fallback in virtual root
        norm_target = os.path.normpath(include_target)
        if norm_target in self.files:
            return norm_target

        return None

    def preprocess(self, entry_file: str) -> PreprocessResult:
        """Preprocess entry file, expanding #include directives while tracking guards and tree."""
        norm_entry = os.path.normpath(entry_file)
        if norm_entry not in self.files:
            raise FileNotFoundError(f"Entry file '{entry_file}' not found in virtual filesystem.")

        included_files_in_order: List[str] = []
        guard_symbols_defined: List[str] = []
        active_guards: set[str] = set()
        seen_pragma_once: set[str] = set()

        def parse_file(current_path: str, visited_stack: List[str]) -> Tuple[List[str], Dict[str, Any]]:
            if current_path in visited_stack:
                raise RuntimeError(f"Recursive circular include detected: {' -> '.join(visited_stack + [current_path])}")

            content = self.files[current_path]
            lines = content.splitlines()
            current_dir = os.path.dirname(current_path)

            tree_node: Dict[str, Any] = {
                "file": current_path,
                "children": [],
                "skipped_by_guard": False,
            }

            # Check pragma once
            if current_path in seen_pragma_once:
                tree_node["skipped_by_guard"] = True
                return [], tree_node

            out_lines: List[str] = []
            guard_stack: List[Tuple[str, bool]] = []  # (symbol, skipping)

            for line in lines:
                sline = line.strip()

                if sline.startswith("#pragma once"):
                    seen_pragma_once.add(current_path)
                    continue

                if sline.startswith("#ifndef"):
                    parts = sline.split()
                    if len(parts) >= 2:
                        sym = parts[1]
                        skipping = (sym in active_guards)
                        guard_stack.append((sym, skipping))
                        if skipping:
                            continue

                elif sline.startswith("#define"):
                    parts = sline.split()
                    if len(parts) >= 2:
                        sym = parts[1]
                        if not any(sk for _, sk in guard_stack):
                            active_guards.add(sym)
                            guard_symbols_defined.append(sym)

                elif sline.startswith("#endif"):
                    if guard_stack:
                        guard_stack.pop()
                        continue

                # If currently skipped by guard, do not emit
                if any(sk for _, sk in guard_stack):
                    continue

                if sline.startswith("#include"):
                    # Parse <header.h> or "header.h"
                    is_system = False
                    target = ""
                    if "<" in sline and ">" in sline:
                        is_system = True
                        target = sline[sline.find("<") + 1 : sline.find(">")]
                    elif '"' in sline:
                        first_q = sline.find('"')
                        second_q = sline.find('"', first_q + 1)
                        if second_q != -1:
                            target = sline[first_q + 1 : second_q]

                    if target:
                        resolved = self.resolve_include(target, current_dir, is_system)
                        if not resolved:
                            raise FileNotFoundError(f"Header '{target}' could not be resolved from '{current_path}' (system={is_system})")

                        included_files_in_order.append(resolved)
                        child_lines, child_tree = parse_file(resolved, visited_stack + [current_path])
                        tree_node["children"].append(child_tree)
                        out_lines.append(f"/* >>> BEGIN EXPAND: {target} ({resolved}) */")
                        out_lines.extend(child_lines)
                        out_lines.append(f"/* <<< END EXPAND: {target} */")
                        continue

                out_lines.append(line)

            return out_lines, tree_node

        expanded_lines, tree = parse_file(norm_entry, [])
        expanded_code = "\n".join(expanded_lines)
        source_line_count = len(self.files[norm_entry].splitlines())
        expanded_line_count = len(expanded_lines)

        return PreprocessResult(
            entry_path=norm_entry,
            expanded_code=expanded_code,
            source_line_count=source_line_count,
            expanded_line_count=expanded_line_count,
            include_tree=tree,
            included_files_in_order=included_files_in_order,
            guard_symbols_defined=guard_symbols_defined,
        )


def plot_uefi_hello_world_execution_flow(output_path: Path | str) -> Path:
    """Generate high-DPI architecture diagram of UEFI Hello World execution flow."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor("#FCFCFD")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97, "MikanOS 第2章 2.2: EDK II Hello World (`osbook_day02a`) 実行制御フロー & メモリ構造",
            fontsize=13, fontweight="bold", ha="center", color="#0D47A1")
    ax.text(50, 93.5, "UEFI BIOS ブートマネージャから UefiMain 呼び出し、Print(L\"...\") 出力、CPU 静止までの完全トレース",
            fontsize=9.5, ha="center", color="#455A64", style="italic")

    # Column 1: Firmware & Boot Manager Phase
    box1 = FancyBboxPatch((4, 10), 28, 80, boxstyle="round,pad=1.0", facecolor="#E3F2FD", edgecolor="#1976D2", lw=1.8)
    ax.add_patch(box1)
    ax.text(18, 86, "【Phase 1】ファームウェア初期化\n& ブートマネージャ", fontsize=10.5, fontweight="bold", ha="center", color="#0D47A1")

    steps1 = [
        ("1. CPUリセット / SEC・PEI", "SEC (Security) → PEI (Pre-EFI)\nメモリ初期化 & CPU 64bit Long Mode 移行", "#BBDEFB"),
        ("2. DXE (Driver Execution)", "UEFI システムテーブル (EFI_SYSTEM_TABLE)\nブートサービス (gBS) テーブル構築", "#BBDEFB"),
        ("3. BDS (Boot Device Select)", "FAT32 USB / ディスクの探索\n`\\EFI\\BOOT\\BOOTX64.EFI` 検出", "#90CAF9"),
        ("4. PE32+ イメージロード", "Relocation & セグメント配置\nImageHandle 生成 & メモリ割当", "#90CAF9"),
        ("5. エントリポイント呼び出し", "call *EntryPoint(ImageHandle, SystemTable)\nMicrosoft x64 ABI (RCX, RDX)", "#64B5F6"),
    ]
    y = 75
    for title, desc, bg in steps1:
        p = FancyBboxPatch((6, y - 7.5), 24, 9, boxstyle="round,pad=0.5", facecolor=bg, edgecolor="#1565C0", lw=1.0)
        ax.add_patch(p)
        ax.text(7.5, y - 1.5, title, fontsize=9, fontweight="bold", color="#0D47A1")
        ax.text(7.5, y - 6.5, desc, fontsize=7.8, color="#212121")
        y -= 13

    # Column 2: Loader.efi & UefiMain
    box2 = FancyBboxPatch((36, 10), 28, 80, boxstyle="round,pad=1.0", facecolor="#FFF8E1", edgecolor="#FFA000", lw=1.8)
    ax.add_patch(box2)
    ax.text(50, 86, "【Phase 2】Loader.efi 内部実行\n`UefiMain` (`Main.c`)", fontsize=10.5, fontweight="bold", ha="center", color="#E65100")

    steps2 = [
        ("1. AutoGen エントリラッパー", "_ModuleEntryPoint(ImageHandle, SystemTable)\ngST, gBS, gImageHandle グローバル初期化", "#FFE082"),
        ("2. UefiMain 呼び出し", "EFI_STATUS EFIAPI UefiMain(\n  EFI_HANDLE ImageHandle,\n  EFI_SYSTEM_TABLE *SystemTable)", "#FFD54F"),
        ("3. Print(L\"Hello, Mikan World!\\n\")", "UefiLib 高水準関数呼び出し\nUTF-16LE CHAR16 文字列フォーマット", "#FFCA28"),
        ("4. ConOut->OutputString()", "SystemTable->ConOut->OutputString(\n  SystemTable->ConOut, L\"...\") 呼び出し", "#FFB300"),
        ("5. while (1); ループ静止", "ブートマネージャへの早期リターンを防ぎ\n画面出力を維持したまま CPU を静止", "#FF8F00"),
    ]
    y = 75
    for title, desc, bg in steps2:
        p = FancyBboxPatch((38, y - 7.5), 24, 9, boxstyle="round,pad=0.5", facecolor=bg, edgecolor="#BF360C", lw=1.0)
        ax.add_patch(p)
        ax.text(39.5, y - 1.5, title, fontsize=9, fontweight="bold", color="#B71C1C")
        ax.text(39.5, y - 6.5, desc, fontsize=7.8, color="#212121")
        y -= 13

    # Column 3: Output & Hardware State
    box3 = FancyBboxPatch((68, 10), 28, 80, boxstyle="round,pad=1.0", facecolor="#E8F5E9", edgecolor="#388E3C", lw=1.8)
    ax.add_patch(box3)
    ax.text(82, 86, "【Phase 3】画面描画 & ハードウェア状態\n(Console & GOP)", fontsize=10.5, fontweight="bold", ha="center", color="#1B5E20")

    steps3 = [
        ("1. テキスト出力プロトコル", "EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL\n行・桁カーソル管理、属性（白文字/黒背景）", "#C8E6C9"),
        ("2. GOP フレームバッファ転送", "文字フォントグリフを VRAM (GOP)\nピクセルバッファへラスタライズ転送", "#A5D6A7"),
        ("3. QEMU / 実機モニタ表示", "「Hello, Mikan World!」\nターミナルコンソール画面に明瞭に表示", "#81C784"),
        ("4. CPU レジスタ状態", "RIP: Main.c while(1) 命令アドレス\n無限ループで静止（ファームウェア復帰防止）", "#66BB6A"),
        ("5. 正常終了戻り値 (未達)", "return EFI_SUCCESS; (0x00000000_00000000)\n※次節以降でカーネルロードへ進展", "#4CAF50"),
    ]
    y = 75
    for title, desc, bg in steps3:
        p = FancyBboxPatch((70, y - 7.5), 24, 9, boxstyle="round,pad=0.5", facecolor=bg, edgecolor="#1B5E20", lw=1.0)
        ax.add_patch(p)
        ax.text(71.5, y - 1.5, title, fontsize=9, fontweight="bold", color="#004D40")
        ax.text(71.5, y - 6.5, desc, fontsize=7.8, color="#212121")
        y -= 13

    # Connectors between columns
    ax.annotate("", xy=(36, 50), xytext=(32, 50), arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=2.5))
    ax.annotate("", xy=(68, 50), xytext=(64, 50), arrowprops=dict(arrowstyle="->", color="#E65100", lw=2.5))

    # Bottom summary banner
    summary_box = FancyBboxPatch((4, 2), 92, 6, boxstyle="round,pad=0.4", facecolor="#ECEFF1", edgecolor="#78909C", lw=1.2)
    ax.add_patch(summary_box)
    ax.text(50, 4.8, "EDK II の UefiLib が SystemTable->ConOut を抽象化し、Print() で UTF-16LE 文字列描画を1行で実現",
            fontsize=9.2, fontweight="bold", ha="center", color="#263238")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_preprocessor_include_expansion_and_tree(output_path: Path | str) -> Path:
    """Generate high-DPI diagram detailing C Preprocessor #include expansion and include guard logic."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.5))
    fig.patch.set_facecolor("#FCFCFD")

    # Left Panel: Text Expansion and Search Path Rules
    ax1.set_facecolor("#F9FBE7")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis("off")
    ax1.text(50, 96, "【コラム2.1】プリプロセッサ #include の展開メカニズム", fontsize=11.5, fontweight="bold", ha="center", color="#33691E")
    ax1.text(50, 92, "単なるテキストの丸ごとコピペ（テキスト置換）と探索パスの違い", fontsize=9, ha="center", color="#558B2F", style="italic")

    # Code Box Before Expansion
    box_before = FancyBboxPatch((8, 52), 36, 36, boxstyle="round,pad=0.6", facecolor="#FFFFFF", edgecolor="#827717", lw=1.5)
    ax1.add_patch(box_before)
    ax1.text(26, 84, "Main.c (展開前: 約10行)", fontsize=9.5, fontweight="bold", ha="center", color="#827717")
    code_before = (
        '#include <Uefi.h>\n'
        '#include <Library/UefiLib.h>\n'
        '\n'
        'EFI_STATUS EFIAPI UefiMain(\n'
        '    EFI_HANDLE ImageHandle,\n'
        '    EFI_SYSTEM_TABLE *SystemTable)\n'
        '{\n'
        '  Print(L"Hello, Mikan World!\\n");\n'
        '  while (1);\n'
        '  return EFI_SUCCESS;\n'
        '}'
    )
    ax1.text(10, 80, code_before, fontsize=7.2, family="monospace", va="top", color="#212121")

    # Arrow for expansion
    ax1.annotate("", xy=(56, 70), xytext=(46, 70),
                 arrowprops=dict(arrowstyle="->", color="#E65100", lw=3.0))
    ax1.text(51, 72, "CPP 展開\n(-E オプション)", fontsize=8.5, fontweight="bold", ha="center", color="#E65100")

    # Code Box After Expansion
    box_after = FancyBboxPatch((58, 52), 36, 36, boxstyle="round,pad=0.6", facecolor="#FFF3E0", edgecolor="#E65100", lw=1.5)
    ax1.add_patch(box_after)
    ax1.text(76, 84, "Main.i (展開後: 数万行！)", fontsize=9.5, fontweight="bold", ha="center", color="#BF360C")
    code_after = (
        '// typedef unsigned long long UINT64;\n'
        '// typedef UINT64 EFI_STATUS;\n'
        '// struct _EFI_SYSTEM_TABLE {\n'
        '//   EFI_TABLE_HEADER Hdr;\n'
        '//   ... (thousands of lines of types)\n'
        '// };\n'
        '// EFI_STATUS Print(IN CONST CHAR16*);\n'
        '// ... all headers expanded inline\n'
        '\n'
        'EFI_STATUS EFIAPI UefiMain(...) {\n'
        '  Print(L"Hello, Mikan World!\\n");\n'
        '  while (1); return 0;\n'
        '}'
    )
    ax1.text(60, 80, code_after, fontsize=7.2, family="monospace", va="top", color="#3E2723")

    # Search Path Rules Card
    box_rules = FancyBboxPatch((8, 6), 86, 40, boxstyle="round,pad=0.6", facecolor="#FFFFFF", edgecolor="#558B2F", lw=1.5)
    ax1.add_patch(box_rules)
    ax1.text(51, 42, "探索規則: #include <...> vs #include \"...\"", fontsize=10, fontweight="bold", ha="center", color="#1B5E20")

    rule_text = (
        "1. #include <Header.h> (システムヘッダ探索):\n"
        "   - カレントディレクトリを無視し、コンパイラの -I インクルードパスのみを順次探索。\n"
        "   - EDK II では Loader.inf の [Packages] MdePkg.dec により、MdePkg/Include が自動注入。\n\n"
        "2. #include \"Header.h\" (ローカルヘッダ探索):\n"
        "   - まず『インクルードを指示したソースが存在するカレントディレクトリ』を最優先で探索。\n"
        "   - カレントに存在しない場合、システムインクルードパス (-I) を順次フォールバック探索。\n\n"
        "3. EDK II におけるインクルードパス解決:\n"
        "   - MdePkg.dec の [Includes] Include, Include/X64 が BaseTools により -I フラグに変換される。"
    )
    ax1.text(12, 38, rule_text, fontsize=8.0, va="top", color="#263238")

    # Right Panel: Include Tree & Guards
    ax2.set_facecolor("#EDE7F6")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis("off")
    ax2.text(50, 96, "多重インクルード防止（インクルードガード）の数理構造", fontsize=11.5, fontweight="bold", ha="center", color="#4A148C")
    ax2.text(50, 92, "木構造展開（Include Tree）と重複定義エラー (Redefinition) の遮断", fontsize=9, ha="center", color="#6A1B9A", style="italic")

    # Tree visualization
    tree_box = FancyBboxPatch((8, 48), 84, 40, boxstyle="round,pad=0.6", facecolor="#FFFFFF", edgecolor="#7B1FA2", lw=1.5)
    ax2.add_patch(tree_box)
    ax2.text(50, 84, "MikanLoaderPkg の Include 依存関係ツリー", fontsize=10, fontweight="bold", ha="center", color="#4A148C")

    tree_art = (
        "Main.c\n"
        " |-- <Uefi.h>\n"
        " |    |-- <Uefi/UefiBaseType.h>\n"
        " |    |    +-- <Base.h> (UINTN, CHAR16, EFI_STATUS types)\n"
        " |    +-- <Uefi/UefiSpec.h> (EFI_SYSTEM_TABLE, EFI_BOOT_SERVICES)\n"
        " +-- <Library/UefiLib.h>\n"
        "      |-- <Uefi.h>  ---> [Guard Active: 2nd expansion skipped]\n"
        "      +-- Print(), Unicode string functions prototype"
    )
    ax2.text(12, 80, tree_art, fontsize=8.2, family="monospace", va="top", color="#212121")

    # Guard Mechanism Comparison Card
    guard_box = FancyBboxPatch((8, 6), 84, 38, boxstyle="round,pad=0.6", facecolor="#F3E5F5", edgecolor="#8E24AA", lw=1.5)
    ax2.add_patch(guard_box)
    ax2.text(50, 40, "インクルードガード方式の比較", fontsize=10, fontweight="bold", ha="center", color="#4A148C")

    guard_desc = (
        "【伝統的マクロガード (#ifndef / #define / #endif)】\n"
        "#ifndef __UEFI_H__\n"
        "#define __UEFI_H__\n"
        "// ヘッダ本体\n"
        "#endif\n"
        "・長所: C89/C99/C11/C++ 全規格で 100% 準拠・ポータブル。\n"
        "・動作: 2回目以降は #ifndef が偽になり、ファイル末尾まで全行読み飛ばされる。\n\n"
        "【現代的ディレクティブ (#pragma once)】\n"
        "・動作: コンパイラが inode/ファイルパスを記憶し、2回目のファイルオープン自体を省略。\n"
        "・長所: 大規模プロジェクトでプリプロセスが高速化。EDK II 標準では双方を併用。"
    )
    ax2.text(12, 36, guard_desc, fontsize=8.0, va="top", color="#212121")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 2.3: Main Memory Architecture & MMIO Simulation Models
# ==============================================================================

class MemoryRegionType(Enum):
    """Classification of physical memory address space regions."""
    USABLE_RAM = "Usable RAM (DRAM)"
    RESERVED = "Reserved / Platform Hardware"
    ACPI_RECLAIM = "ACPI Reclaim Memory"
    ACPI_NVS = "ACPI Non-Volatile Storage (NVS)"
    MMIO = "Memory-Mapped I/O (MMIO)"
    MMIO_PORT_SPACE = "MMIO Port Space"
    FRAMEBUFFER = "Graphics Framebuffer (VRAM)"
    BOOT_SERVICES_CODE = "UEFI Boot Services Code"
    BOOT_SERVICES_DATA = "UEFI Boot Services Data"
    RUNTIME_SERVICES_CODE = "UEFI Runtime Services Code"
    RUNTIME_SERVICES_DATA = "UEFI Runtime Services Data"
    FIRMWARE_ROM = "Firmware / Option ROM"
    MEMORY_HOLE = "Unmapped Memory Hole (Bus Error on Access)"


@dataclass(frozen=True)
class PhysicalMemoryRegion:
    """Represents a contiguous range in the physical address space."""
    base_addr: int
    length_bytes: int
    region_type: MemoryRegionType
    description: str
    is_cacheable: bool = True

    @property
    def end_addr(self) -> int:
        """Return the exclusive end address (base + length)."""
        return self.base_addr + self.length_bytes

    @property
    def page_count(self) -> int:
        """Return the number of 4KiB pages required to cover this region."""
        return (self.length_bytes + 4095) // 4096

    @property
    def is_page_aligned(self) -> bool:
        """Check if base address and length are aligned to 4KiB (0x1000) boundaries."""
        return (self.base_addr & 0xFFF) == 0 and (self.length_bytes & 0xFFF) == 0

    @property
    def is_usable_ram(self) -> bool:
        """Check if this region can be freely allocated for OS code/data after boot."""
        return self.region_type in (
            MemoryRegionType.USABLE_RAM,
            MemoryRegionType.BOOT_SERVICES_CODE,
            MemoryRegionType.BOOT_SERVICES_DATA,
        )

    def contains_address(self, addr: int) -> bool:
        """Check whether a given physical address falls within this region."""
        return self.base_addr <= addr < self.end_addr

    def overlaps_with(self, other: "PhysicalMemoryRegion") -> bool:
        """Check if this region overlaps with another region."""
        return max(self.base_addr, other.base_addr) < min(self.end_addr, other.end_addr)


class PhysicalAddressSpace:
    """Simulates the 64-bit physical address space, address decoding, and memory holes."""

    def __init__(self, address_width_bits: int = 48):
        if address_width_bits not in (32, 36, 40, 48, 52):
            raise ValueError(f"Unsupported physical address width: {address_width_bits} bits")
        self.address_width_bits = address_width_bits
        self.max_address = (1 << address_width_bits) - 1
        self._regions: list[PhysicalMemoryRegion] = []

    @property
    def regions(self) -> list[PhysicalMemoryRegion]:
        """Return regions sorted by base address."""
        return sorted(self._regions, key=lambda r: r.base_addr)

    def add_region(self, region: PhysicalMemoryRegion) -> None:
        """Add a memory region to the physical address space."""
        if region.end_addr > self.max_address + 1:
            raise ValueError(
                f"Region end 0x{region.end_addr:X} exceeds max physical address 0x{self.max_address:X}"
            )
        # Check overlaps with existing regions
        for existing in self._regions:
            if region.overlaps_with(existing):
                raise ValueError(
                    f"Address collision: '{region.description}' [0x{region.base_addr:08X}-0x{region.end_addr:08X}) "
                    f"conflicts with '{existing.description}' [0x{existing.base_addr:08X}-0x{existing.end_addr:08X})"
                )
        self._regions.append(region)

    def resolve_address(self, addr: int) -> Optional[PhysicalMemoryRegion]:
        """Decode a physical address and return the destination memory or MMIO region."""
        if addr > self.max_address:
            return None
        for r in self._regions:
            if r.contains_address(addr):
                return r
        return None

    def detect_memory_holes(self) -> list[tuple[int, int, int]]:
        """Identify unmapped gaps (memory holes) between registered regions.
        
        Returns a list of tuples: (start_addr, end_addr, hole_size_bytes)
        """
        sorted_regs = self.regions
        holes = []
        if not sorted_regs:
            return [(0, self.max_address + 1, self.max_address + 1)]

        if sorted_regs[0].base_addr > 0:
            holes.append((0, sorted_regs[0].base_addr, sorted_regs[0].base_addr))

        for i in range(len(sorted_regs) - 1):
            curr_end = sorted_regs[i].end_addr
            next_start = sorted_regs[i + 1].base_addr
            if curr_end < next_start:
                holes.append((curr_end, next_start, next_start - curr_end))

        return holes

    def find_free_ram_chunk(self, size_bytes: int, alignment: int = 4096) -> Optional[int]:
        """Search for a continuous usable RAM chunk of given size satisfying alignment.
        
        Returns the aligned starting physical address or None if not available.
        """
        for r in self.regions:
            if r.region_type == MemoryRegionType.USABLE_RAM:
                # Calculate aligned start
                aligned_start = (r.base_addr + alignment - 1) & ~(alignment - 1)
                if aligned_start + size_bytes <= r.end_addr:
                    return aligned_start
        return None

    def get_memory_summary(self) -> dict[str, int]:
        """Aggregate total bytes for each region type."""
        summary: dict[str, int] = {
            "total_usable_ram_bytes": 0,
            "total_mmio_bytes": 0,
            "total_reserved_bytes": 0,
            "total_acpi_bytes": 0,
            "total_firmware_bytes": 0,
            "total_uefi_services_bytes": 0,
            "total_mapped_bytes": 0,
            "total_hole_bytes": 0,
        }
        for r in self._regions:
            summary["total_mapped_bytes"] += r.length_bytes
            if r.region_type == MemoryRegionType.USABLE_RAM:
                summary["total_usable_ram_bytes"] += r.length_bytes
            elif r.region_type in (MemoryRegionType.MMIO, MemoryRegionType.MMIO_PORT_SPACE, MemoryRegionType.FRAMEBUFFER):
                summary["total_mmio_bytes"] += r.length_bytes
            elif r.region_type == MemoryRegionType.RESERVED:
                summary["total_reserved_bytes"] += r.length_bytes
            elif r.region_type in (MemoryRegionType.ACPI_RECLAIM, MemoryRegionType.ACPI_NVS):
                summary["total_acpi_bytes"] += r.length_bytes
            elif r.region_type == MemoryRegionType.FIRMWARE_ROM:
                summary["total_firmware_bytes"] += r.length_bytes
            elif r.region_type in (
                MemoryRegionType.BOOT_SERVICES_CODE,
                MemoryRegionType.BOOT_SERVICES_DATA,
                MemoryRegionType.RUNTIME_SERVICES_CODE,
                MemoryRegionType.RUNTIME_SERVICES_DATA,
            ):
                summary["total_uefi_services_bytes"] += r.length_bytes

        for _, _, hole_len in self.detect_memory_holes():
            summary["total_hole_bytes"] += hole_len

        return summary

    @classmethod
    def simulate_standard_pc_4gb(cls) -> "PhysicalAddressSpace":
        """Construct a realistic 4GB+ modern x86-64 PC physical memory map (QEMU + OVMF style)."""
        pas = cls(address_width_bits=48)
        # 1. Real Mode IVT & BDA (0x0000_0000 - 0x0000_0FFF)
        pas.add_region(PhysicalMemoryRegion(0x0000_0000, 0x1000, MemoryRegionType.RESERVED, "Real-Mode IVT & BDA", is_cacheable=True))
        # 2. Low Usable Conventional RAM (0x0000_1000 - 0x0007_FFFF: 508 KiB)
        pas.add_region(PhysicalMemoryRegion(0x0000_1000, 0x7F000, MemoryRegionType.USABLE_RAM, "Low Conventional RAM (<640KB)", is_cacheable=True))
        # 3. EBDA (Extended BIOS Data Area) (0x0008_0000 - 0x0009_FFFF: 128 KiB)
        pas.add_region(PhysicalMemoryRegion(0x0008_0000, 0x20000, MemoryRegionType.RESERVED, "Extended BIOS Data Area (EBDA)", is_cacheable=True))
        # 4. VGA VRAM / MMIO (0x000A_0000 - 0x000B_FFFF: 128 KiB)
        pas.add_region(PhysicalMemoryRegion(0x000A_0000, 0x20000, MemoryRegionType.FRAMEBUFFER, "Legacy VGA Framebuffer MMIO", is_cacheable=False))
        # 5. Video BIOS / Option ROM (0x000C_0000 - 0x000D_FFFF: 128 KiB)
        pas.add_region(PhysicalMemoryRegion(0x000C_0000, 0x20000, MemoryRegionType.FIRMWARE_ROM, "Video BIOS & Option ROMs", is_cacheable=True))
        # 6. Motherboard BIOS / UEFI Low ROM (0x000E_0000 - 0x000F_FFFF: 128 KiB)
        pas.add_region(PhysicalMemoryRegion(0x000E_0000, 0x20000, MemoryRegionType.FIRMWARE_ROM, "Motherboard UEFI / BIOS ROM", is_cacheable=True))
        # 7. Main Usable RAM below 2GB (0x0010_0000 - 0x7FFF_FFFF: ~2047 MiB)
        pas.add_region(PhysicalMemoryRegion(0x0010_0000, 0x7FE0_0000, MemoryRegionType.USABLE_RAM, "Main Conventional RAM (1MB-2GB)", is_cacheable=True))
        # 8. UEFI Boot Services Code/Data (0x7FF0_0000 - 0x7FFF_FFFF: 1 MiB)
        pas.add_region(PhysicalMemoryRegion(0x7FF0_0000, 0x10_0000, MemoryRegionType.BOOT_SERVICES_DATA, "UEFI Boot Services Pool", is_cacheable=True))
        # 9. ACPI Reclaim & NVS (0x8000_0000 - 0x800F_FFFF: 1 MiB)
        pas.add_region(PhysicalMemoryRegion(0x8000_0000, 0x10_0000, MemoryRegionType.ACPI_RECLAIM, "ACPI Tables & Reclaim Memory", is_cacheable=True))
        # 10. High GOP Framebuffer MMIO (0xC000_0000 - 0xC0FF_FFFF: 16 MiB)
        pas.add_region(PhysicalMemoryRegion(0xC000_0000, 0x0100_0000, MemoryRegionType.FRAMEBUFFER, "GOP High-Res Framebuffer MMIO", is_cacheable=False))
        # 11. PCIe MMIO Bar Space (0xC100_0000 - 0xFEB0_0000: ~987 MiB)
        pas.add_region(PhysicalMemoryRegion(0xC100_0000, 0x3DB0_0000, MemoryRegionType.MMIO, "PCI Express Device MMIO Space (xHCI/NVMe/NIC)", is_cacheable=False))
        # 12. I/O APIC MMIO (0xFEC0_0000 - 0xFEC0_0FFF: 4 KiB)
        pas.add_region(PhysicalMemoryRegion(0xFEC0_0000, 0x1000, MemoryRegionType.MMIO, "I/O APIC MMIO Register Window", is_cacheable=False))
        # 13. Local APIC MMIO (0xFEE0_0000 - 0xFEE0_0FFF: 4 KiB)
        pas.add_region(PhysicalMemoryRegion(0xFEE0_0000, 0x1000, MemoryRegionType.MMIO, "Local APIC MMIO Register Window", is_cacheable=False))
        # 14. High OVMF Firmware Flash ROM (0xFFC0_0000 - 0xFFFF_FFFF: 4 MiB)
        pas.add_region(PhysicalMemoryRegion(0xFFC0_0000, 0x40_0000, MemoryRegionType.FIRMWARE_ROM, "OVMF High Firmware Flash ROM (Reset Vector)", is_cacheable=True))
        # 15. Extended RAM above 4GB (0x1_0000_0000 - 0x1_7FFF_FFFF: 2 GiB)
        pas.add_region(PhysicalMemoryRegion(0x1_0000_0000, 0x8000_0000, MemoryRegionType.USABLE_RAM, "High RAM above 4GB (>4GB Conventional RAM)", is_cacheable=True))
        return pas

    @classmethod
    def simulate_legacy_1mb_layout(cls) -> "PhysicalAddressSpace":
        """Construct the historical 1MB (20-bit address bus) PC/XT memory map."""
        pas = cls(address_width_bits=32)
        pas.add_region(PhysicalMemoryRegion(0x00000, 0x00400, MemoryRegionType.RESERVED, "Interrupt Vector Table (IVT: 256 x 4B)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0x00400, 0x00100, MemoryRegionType.RESERVED, "BIOS Data Area (BDA: 256B)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0x00500, 0x07700, MemoryRegionType.USABLE_RAM, "Low RAM (DOS Data / Free: ~29.75KB)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0x07C00, 0x00200, MemoryRegionType.USABLE_RAM, "MBR Bootsector Load Location (512B)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0x07E00, 0x78200, MemoryRegionType.USABLE_RAM, "Conventional DOS Memory Free RAM (~480.5KB)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0x80000, 0x20000, MemoryRegionType.RESERVED, "Extended BIOS Data Area (EBDA: 128KB)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0xA0000, 0x10000, MemoryRegionType.FRAMEBUFFER, "EGA/VGA Graphics VRAM MMIO (64KB)", is_cacheable=False))
        pas.add_region(PhysicalMemoryRegion(0xB0000, 0x08000, MemoryRegionType.FRAMEBUFFER, "MDA Monochrome Text RAM (32KB)", is_cacheable=False))
        pas.add_region(PhysicalMemoryRegion(0xB8000, 0x08000, MemoryRegionType.FRAMEBUFFER, "CGA Color Text RAM (32KB)", is_cacheable=False))
        pas.add_region(PhysicalMemoryRegion(0xC0000, 0x08000, MemoryRegionType.FIRMWARE_ROM, "Video BIOS ROM (32KB)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0xC8000, 0x18000, MemoryRegionType.FIRMWARE_ROM, "Option ROMs / SCSI / Network (96KB)", is_cacheable=True))
        pas.add_region(PhysicalMemoryRegion(0xE0000, 0x20000, MemoryRegionType.FIRMWARE_ROM, "Motherboard System BIOS ROM (128KB)", is_cacheable=True))
        return pas


def plot_physical_address_space_and_mmio(
    output_path: str = "2/result/fig2_05_physical_address_space_and_mmio_architecture.png",
) -> str:
    """Generate high-resolution architecture diagram of modern 64-bit physical address space and MMIO routing."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.24)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_mmio = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[1, :])

    # -------------------------------------------------------------
    # Panel 1: Physical Memory Address Layout (0 to 6GB)
    # -------------------------------------------------------------
    ax_map.set_title("x86-64 物理アドレス空間（Physical Address Space）の配置", fontsize=11, fontweight="bold", pad=12)
    ax_map.set_xlim(0, 100)
    ax_map.set_ylim(0, 100)
    ax_map.axis("off")

    # Map blocks: bottom=0 to top=6GB (normalized)
    regions_viz = [
        ("0x1_0000_0000 - 0x1_7FFF_FFFF\n(4GB - 6GB: 2GB High RAM)", 79, 17, "#81C784", "#2E7D32", "High Usable RAM (DRAM)\n4GB以上拡張メモリ"),
        ("0xFFC0_0000 - 0xFFFF_FFFF (4MB)", 73, 5, "#BA68C8", "#6A1B9A", "High OVMF Flash ROM\nReset Vector (0xFFFFFFF0)"),
        ("0xFEE0_0000 (4KB) & 0xFEC0_0000 (4KB)", 67, 5, "#FFB74D", "#E65100", "Local APIC & I/O APIC MMIO"),
        ("0xC100_0000 - 0xFEB0_0000 (~1GB)", 53, 13, "#FF8A65", "#BF360C", "PCIe MMIO Hole (BAR空間)\nxHCI/NVMe/NIC レジスタ"),
        ("0xC000_0000 - 0xC0FF_FFFF (16MB)", 46, 6, "#FFD54F", "#F57F17", "GOP Framebuffer (VRAM MMIO)\n画面描画ピクセル領域"),
        ("0x8000_0000 - 0x800F_FFFF (1MB)", 40, 5, "#4DD0E1", "#006064", "ACPI Reclaim / NVS テーブル"),
        ("0x0010_0000 - 0x7FFF_FFFF (~2GB)", 19, 20, "#A5D6A7", "#1B5E20", "Main Usable RAM (DRAM)\nカーネル・スタック・ヒープ配置可能"),
        ("0x000A_0000 - 0x000F_FFFF (384KB)", 11, 7, "#E0E0E0", "#424242", "Legacy VGA, Option ROM, BIOS"),
        ("0x0000_0000 - 0x0009_FFFF (640KB)", 2, 8, "#C8E6C9", "#2E7D32", "Low Conventional RAM & BDA/EBDA"),
    ]

    for addr_label, y_pos, height, color, border_color, desc in regions_viz:
        rect = FancyBboxPatch(
            (2, y_pos), 46, height,
            boxstyle="round,pad=0.4", facecolor=color, edgecolor=border_color, lw=1.3, alpha=0.92
        )
        ax_map.add_patch(rect)
        ax_map.text(25, y_pos + height / 2, desc, fontsize=7.6, ha="center", va="center", fontweight="bold", color="#1A237E" if "RAM" not in desc else "#003300")
        ax_map.text(50, y_pos + height / 2, addr_label, fontsize=7.2, va="center", family="monospace", color="#37474F")

    # Legend at right bottom of panel 1
    ax_map.text(2, 0.5, "※ 物理アドレス空間には DRAM だけでなく、デバイスレジスタ (MMIO) や ROM がマッピングされる", fontsize=7.3, color="#D32F2F", fontstyle="italic")

    # -------------------------------------------------------------
    # Panel 2: Memory Controller & MMIO Address Routing Mechanism
    # -------------------------------------------------------------
    ax_mmio.set_title("CPU とメモリコントローラによる MMIO ルーティング機構", fontsize=11, fontweight="bold", pad=12)
    ax_mmio.set_xlim(0, 100)
    ax_mmio.set_ylim(0, 100)
    ax_mmio.axis("off")

    # CPU Core
    cpu_box = FancyBboxPatch((30, 82), 40, 14, boxstyle="round,pad=0.5", facecolor="#E1F5FE", edgecolor="#0288D1", lw=1.8)
    ax_mmio.add_patch(cpu_box)
    ax_mmio.text(50, 91, "x86-64 CPU Core", fontsize=9.5, fontweight="bold", ha="center", color="#01579B")
    ax_mmio.text(50, 85, "mov [0x00100000], eax  vs  mov [0xFEE00020], eax", fontsize=7.2, family="monospace", ha="center", color="#0277BD")

    # Arrow Down to System Agent / Root Complex
    ax_mmio.annotate("", xy=(50, 68), xytext=(50, 82), arrowprops=dict(facecolor="#37474F", edgecolor="#37474F", width=2, headwidth=8))
    ax_mmio.text(52, 75, "Physical Address Bus (48/52-bit)", fontsize=7.2, va="center", color="#37474F", family="monospace")

    # Host Bridge / Memory Controller / PCIe Root Complex
    mc_box = FancyBboxPatch((15, 46), 70, 22, boxstyle="round,pad=0.6", facecolor="#FFF9C4", edgecolor="#FBC02D", lw=1.8)
    ax_mmio.add_patch(mc_box)
    ax_mmio.text(50, 61, "Memory Controller / PCIe Root Complex", fontsize=9.2, fontweight="bold", ha="center", color="#F57F17")
    ax_mmio.text(50, 53, "アドレスデコーダ（Address Decoder）が番地を判定:\n・DRAM範囲ならメモリチャネルへルーティング\n・BAR/MMIO範囲なら PCIe / APIC 内部バスへ転送", fontsize=7.5, ha="center", color="#E65100")

    # Branch 1: DRAM Bus (Left Arrow)
    ax_mmio.annotate("", xy=(25, 26), xytext=(35, 46), arrowprops=dict(facecolor="#2E7D32", edgecolor="#2E7D32", width=2, headwidth=8))
    dram_box = FancyBboxPatch((5, 10), 40, 16, boxstyle="round,pad=0.5", facecolor="#C8E6C9", edgecolor="#388E3C", lw=1.5)
    ax_mmio.add_patch(dram_box)
    ax_mmio.text(25, 21, "DDR4 / DDR5 DRAM (DIMM)", fontsize=8.5, fontweight="bold", ha="center", color="#1B5E20")
    ax_mmio.text(25, 14, "通常のRAM (読み書き高速・キャッシュ有効)\nOSカーネル・ヒープ・スタック配置領域", fontsize=6.8, ha="center", color="#2E7D32")

    # Branch 2: PCIe / MMIO Bus (Right Arrow)
    ax_mmio.annotate("", xy=(75, 26), xytext=(65, 46), arrowprops=dict(facecolor="#C2185B", edgecolor="#C2185B", width=2, headwidth=8))
    mmio_box = FancyBboxPatch((55, 10), 42, 16, boxstyle="round,pad=0.5", facecolor="#FFCDD2", edgecolor="#D32F2F", lw=1.5)
    ax_mmio.add_patch(mmio_box)
    ax_mmio.text(76, 21, "PCIe MMIO & Local APIC", fontsize=8.5, fontweight="bold", ha="center", color="#B71C1C")
    ax_mmio.text(76, 14, "デバイス制御レジスタ (キャッシュ無効)\n書き込むと機器制御・描画が発生", fontsize=6.8, ha="center", color="#C2185B")

    # -------------------------------------------------------------
    # Panel 3: Technical Summary & Operating System Implications
    # -------------------------------------------------------------
    ax_summary.set_title("OS開発における物理メモリアーキテクチャの必須知識と落とし穴", fontsize=10.5, fontweight="bold", pad=8)
    ax_summary.set_xlim(0, 100)
    ax_summary.set_ylim(0, 100)
    ax_summary.axis("off")

    cards = [
        (
            2, 94, 30, 86, "#E8F5E9", "#4CAF50",
            "1. 物理RAMの不連続性 (Sparsity)",
            "・搭載RAMが8GBや16GBあっても、0番地から連続して配置されているわけではない。\n"
            "・32bit互換の歴史的都合により、3.5GB〜4GB付近に『PCI MMIO Hole』が存在する。\n"
            "・ホールに押し出された物理RAMは4GB以上の領域（High Memory）に再配置される。"
        ),
        (
            35, 94, 30, 86, "#FFF3E0", "#FF9800",
            "2. MMIO (Memory-Mapped I/O) の危険性",
            "・MMIO領域はRAMではない。読み書きするとハードウェアの状態が変化する（Side Effect）。\n"
            "・OSが空きメモリと誤認してページテーブルやスタックを配置すると暴走・クラッシュする。\n"
            "・CPUキャッシュ（PAT/MTRR）を無効（Uncacheable）にする必要がある。"
        ),
        (
            68, 94, 30, 86, "#EDE7F6", "#7E57C2",
            "3. なぜブートローダがメモリマップを取得するのか？",
            "・CPU起動直後はどこがRAMでどこがMMIO/予約領域かOSには分からない。\n"
            "・UEFIファームウェアだけがマザーボードのハードウェア構成を把握している。\n"
            "・次節以降の GetMemoryMap() で正確な『利用可能RAMリスト』を取得することが不可欠！"
        ),
    ]

    for x, y_top, w, h, bg, border, title, body in cards:
        card = FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0.5", facecolor=bg, edgecolor=border, lw=1.5)
        ax_summary.add_patch(card)
        ax_summary.text(x + w / 2, y_top - 12, title, fontsize=8.2, fontweight="bold", ha="center", color="#212121")
        ax_summary.text(x + 2, y_top - 20, body, fontsize=6.8, va="top", color="#424242", linespacing=1.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_x86_legacy_and_modern_memory_layout_comparison(
    output_path: str = "2/result/fig2_06_x86_legacy_and_modern_memory_layout_comparison.png",
) -> str:
    """Generate high-resolution comparison plot between 1MB Legacy PC layout and 64-bit Modern UEFI layout."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_leg, ax_mod) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"wspace": 0.28})

    # Left: 1MB Legacy PC Layout
    ax_leg.set_title("レガシーPC (1MB / 20-bit Real Mode) メモリマップ", fontsize=11, fontweight="bold", pad=12)
    ax_leg.set_xlim(0, 100)
    ax_leg.set_ylim(0, 100)
    ax_leg.axis("off")

    leg_blocks = [
        ("0xE0000 - 0xFFFFF (128KB)", 86, 12, "#CE93D8", "#6A1B9A", "System BIOS ROM (Reset Vector 0xFFFF0)"),
        ("0xC8000 - 0xDFFFF (96KB)", 74, 11, "#D1C4E9", "#4527A0", "Option ROMs (SCSI / Network / Storage)"),
        ("0xC0000 - 0xC7FFF (32KB)", 64, 9, "#B39DDB", "#311B92", "Video BIOS ROM (VGA INT 10h)"),
        ("0xB8000 - 0xBFFFF (32KB)", 56, 7, "#FFE082", "#FF8F00", "CGA Color Text Buffer (0xB8000)"),
        ("0xB0000 - 0xB7FFF (32KB)", 50, 5, "#FFECB3", "#FFA000", "MDA Monochrome Text Buffer"),
        ("0xA0000 - 0xAFFFF (64KB)", 40, 9, "#FFCC80", "#E65100", "EGA/VGA Graphics Framebuffer (MMIO)"),
        ("0x80000 - 0x9FFFF (128KB)", 28, 11, "#CFD8DC", "#37474F", "Extended BIOS Data Area (EBDA)"),
        ("0x07E00 - 0x7FFFF (~480KB)", 14, 13, "#A5D6A7", "#1B5E20", "Conventional Free RAM (DOS Program Area)"),
        ("0x07C00 - 0x07DFF (512B)", 9, 4, "#EF9A9A", "#B71C1C", "MBR Bootsector (0x7C00 ジャンプ位置)"),
        ("0x00500 - 0x07BFF (~29KB)", 5, 3.5, "#C8E6C9", "#2E7D32", "DOS Low System RAM"),
        ("0x00400 - 0x004FF (256B)", 2.5, 2, "#FFF59D", "#F57F17", "BIOS Data Area (BDA)"),
        ("0x00000 - 0x003FF (1KB)", 0.2, 1.8, "#FFAB91", "#BF360C", "Interrupt Vector Table (IVT: 256エントリ)"),
    ]

    for addr, y, h, c, bc, label in leg_blocks:
        box = FancyBboxPatch((2, y), 50, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_leg.add_patch(box)
        ax_leg.text(27, y + h / 2, label, fontsize=6.8, ha="center", va="center", fontweight="bold", color="#212121")
        ax_leg.text(54, y + h / 2, addr, fontsize=6.8, va="center", family="monospace", color="#37474F")

    # Right: Modern 64-bit UEFI Layout
    ax_mod.set_title("現代の 64-bit UEFI システム (x86-64) メモリマップ", fontsize=11, fontweight="bold", pad=12)
    ax_mod.set_xlim(0, 100)
    ax_mod.set_ylim(0, 100)
    ax_mod.axis("off")

    mod_blocks = [
        ("0x1_0000_0000+ (High RAM)", 82, 16, "#81C784", "#1B5E20", "High Conventional RAM\n(4GB以上の広大なメイン物理RAM)"),
        ("0xFFC0_0000 - 0xFFFF_FFFF (4MB)", 73, 8, "#BA68C8", "#4A148C", "OVMF / UEFI Firmware Flash ROM\n(64-bit Reset Vector)"),
        ("0xFEC0_0000 - 0xFEE0_0FFF (8KB)", 64, 8, "#FFB74D", "#E65100", "Local APIC & I/O APIC MMIO Windows"),
        ("0xC100_0000 - 0xFEB0_0000 (~1GB)", 47, 16, "#FF8A65", "#BF360C", "PCIe MMIO Hole (BAR空間)\nグラフィックボード・NVMe・NICの制御空間"),
        ("0xC000_0000 - 0xC0FF_FFFF (16MB)", 38, 8, "#FFE082", "#FF8F00", "GOP (Graphics Output Protocol) フレームバッファ"),
        ("0x8000_0000 - 0x800F_FFFF (1MB)", 30, 7, "#80DEEA", "#006064", "ACPI Reclaim / NVS テーブル領域"),
        ("0x7FF0_0000 - 0x7FFF_FFFF (1MB)", 22, 7, "#B2DFDB", "#004D40", "UEFI Boot Services Code & Data"),
        ("0x0010_0000 - 0x7FEF_FFFF (~2GB)", 6, 15, "#A5D6A7", "#2E7D32", "Main Conventional RAM (1MB - 2GB)\nブートローダ・OSカーネル読み込み先"),
        ("0x0000_0000 - 0x000F_FFFF (1MB)", 0.5, 5, "#EEEEEE", "#616161", "1MB以下 レガシー互換領域 (互換性維持用)"),
    ]

    for addr, y, h, c, bc, label in mod_blocks:
        box = FancyBboxPatch((2, y), 50, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_mod.add_patch(box)
        ax_mod.text(27, y + h / 2, label, fontsize=7.1, ha="center", va="center", fontweight="bold", color="#212121")
        ax_mod.text(54, y + h / 2, addr, fontsize=7.0, va="center", family="monospace", color="#37474F")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 2.4: UEFI Memory Map & EFI_MEMORY_DESCRIPTOR Classes & Visualization
# ==============================================================================

class UefiMemoryType(Enum):
    """UEFI memory types defined in UEFI Specification Section 7.2 (EFI_MEMORY_TYPE)."""

    EfiReservedMemoryType = 0
    EfiLoaderCode = 1
    EfiLoaderData = 2
    EfiBootServicesCode = 3
    EfiBootServicesData = 4
    EfiRuntimeServicesCode = 5
    EfiRuntimeServicesData = 6
    EfiConventionalMemory = 7
    EfiUnusableMemory = 8
    EfiACPIReclaimMemory = 9
    EfiACPIMemoryNVS = 10
    EfiMemoryMappedIO = 11
    EfiMemoryMappedIOPortSpace = 12
    EfiPalCode = 13
    EfiPersistentMemory = 14
    EfiMaxMemoryType = 15

    @property
    def is_usable_by_os_initially(self) -> bool:
        """Memory that is free and directly usable as RAM by the OS immediately."""
        return self == UefiMemoryType.EfiConventionalMemory

    @property
    def is_reclaimable_after_exit_boot_services(self) -> bool:
        """Memory that can be safely reclaimed and used as free RAM once ExitBootServices() is called."""
        return self in (
            UefiMemoryType.EfiBootServicesCode,
            UefiMemoryType.EfiBootServicesData,
            UefiMemoryType.EfiLoaderCode,
            UefiMemoryType.EfiLoaderData,
        )

    @property
    def is_runtime_required(self) -> bool:
        """Memory that the OS kernel must preserve and map into virtual address space for UEFI Runtime Services."""
        return self in (
            UefiMemoryType.EfiRuntimeServicesCode,
            UefiMemoryType.EfiRuntimeServicesData,
            UefiMemoryType.EfiMemoryMappedIO,
            UefiMemoryType.EfiACPIMemoryNVS,
            UefiMemoryType.EfiPalCode,
        )

    @property
    def category_name(self) -> str:
        if self == UefiMemoryType.EfiConventionalMemory:
            return "OS利用可能RAM (Conventional)"
        elif self in (UefiMemoryType.EfiBootServicesCode, UefiMemoryType.EfiBootServicesData):
            return "ブートサービス領域 (ExitBootServices後解放可能)"
        elif self in (UefiMemoryType.EfiRuntimeServicesCode, UefiMemoryType.EfiRuntimeServicesData):
            return "ランタイムサービス領域 (OS実行中保持必須)"
        elif self in (UefiMemoryType.EfiLoaderCode, UefiMemoryType.EfiLoaderData):
            return "ローダ領域 (ブートローダ自身)"
        elif self in (UefiMemoryType.EfiACPIReclaimMemory, UefiMemoryType.EfiACPIMemoryNVS):
            return "ACPIテーブル領域"
        elif self in (UefiMemoryType.EfiMemoryMappedIO, UefiMemoryType.EfiMemoryMappedIOPortSpace):
            return "MMIOデバイス領域"
        elif self == UefiMemoryType.EfiPersistentMemory:
            return "不揮発性メモリ (NVDIMM/PMEM)"
        elif self == UefiMemoryType.EfiPalCode:
            return "プロセッサ抽象化レイヤ (PAL)"
        else:
            return "予約・使用不可領域"


# UEFI Memory Attributes (UEFI 2.9 / 2.10 Spec Section 7.2 Table 7.3)
EFI_MEMORY_UC = 0x0000000000000001
EFI_MEMORY_WC = 0x0000000000000002
EFI_MEMORY_WT = 0x0000000000000004
EFI_MEMORY_WB = 0x0000000000000008
EFI_MEMORY_UCE = 0x0000000000000010
EFI_MEMORY_WP = 0x0000000000001000
EFI_MEMORY_RP = 0x0000000000002000
EFI_MEMORY_XP = 0x0000000000004000
EFI_MEMORY_NV = 0x0000000000008000
EFI_MEMORY_MORE_RELIABLE = 0x0000000000010000
EFI_MEMORY_RO = 0x0000000000020000
EFI_MEMORY_SP = 0x0000000000040000
EFI_MEMORY_CPU_CRYPTO = 0x0000000000080000
EFI_MEMORY_RUNTIME = 0x8000000000000000


@dataclass
class UefiMemoryDescriptor:
    """Represents a single UEFI memory descriptor (EFI_MEMORY_DESCRIPTOR).

    C layout in x86-64 EDK II:
    typedef struct {
        UINT32                Type;           // 0x00 - 0x03 (4 bytes)
        UINT32                Pad;            // 0x04 - 0x07 (4 bytes alignment pad)
        EFI_PHYSICAL_ADDRESS  PhysicalStart;  // 0x08 - 0x0F (8 bytes)
        EFI_VIRTUAL_ADDRESS   VirtualStart;   // 0x10 - 0x17 (8 bytes)
        UINT64                NumberOfPages;  // 0x18 - 0x1F (8 bytes, 4KiB pages)
        UINT64                Attribute;      // 0x20 - 0x27 (8 bytes)
    } EFI_MEMORY_DESCRIPTOR;                  // Total size: 48 bytes (0x30)
    """

    type: UefiMemoryType
    physical_start: int
    virtual_start: int
    number_of_pages: int
    attribute: int
    pad: int = 0

    PAGE_SIZE: int = 4096

    @property
    def size_bytes(self) -> int:
        return self.number_of_pages * self.PAGE_SIZE

    @property
    def physical_end(self) -> int:
        return self.physical_start + self.size_bytes

    @property
    def is_conventional(self) -> bool:
        return self.type == UefiMemoryType.EfiConventionalMemory

    @property
    def is_reclaimable_after_exit_boot_services(self) -> bool:
        return self.type in (
            UefiMemoryType.EfiBootServicesCode,
            UefiMemoryType.EfiBootServicesData,
            UefiMemoryType.EfiLoaderCode,
            UefiMemoryType.EfiLoaderData,
        )

    @property
    def is_runtime_service(self) -> bool:
        return self.type in (
            UefiMemoryType.EfiRuntimeServicesCode,
            UefiMemoryType.EfiRuntimeServicesData,
        )

    @property
    def is_runtime_mapped_required(self) -> bool:
        return bool(self.attribute & EFI_MEMORY_RUNTIME)

    @property
    def is_page_aligned(self) -> bool:
        return (self.physical_start % self.PAGE_SIZE) == 0

    def attribute_names(self) -> List[str]:
        names = []
        attrs = [
            (EFI_MEMORY_UC, "UC"),
            (EFI_MEMORY_WC, "WC"),
            (EFI_MEMORY_WT, "WT"),
            (EFI_MEMORY_WB, "WB"),
            (EFI_MEMORY_UCE, "UCE"),
            (EFI_MEMORY_WP, "WP"),
            (EFI_MEMORY_RP, "RP"),
            (EFI_MEMORY_XP, "XP"),
            (EFI_MEMORY_NV, "NV"),
            (EFI_MEMORY_MORE_RELIABLE, "MORE_RELIABLE"),
            (EFI_MEMORY_RO, "RO"),
            (EFI_MEMORY_SP, "SP"),
            (EFI_MEMORY_CPU_CRYPTO, "CPU_CRYPTO"),
            (EFI_MEMORY_RUNTIME, "RUNTIME"),
        ]
        for flag, name in attrs:
            if self.attribute & flag:
                names.append(name)
        return names

    def to_bytes(self, descriptor_size: int = 48) -> bytes:
        if descriptor_size < 40:
            raise ValueError(f"descriptor_size must be at least 40 bytes, got {descriptor_size}")
        raw = struct.pack(
            "<IIQQQQ",
            self.type.value,
            self.pad,
            self.physical_start,
            self.virtual_start,
            self.number_of_pages,
            self.attribute,
        )
        if descriptor_size > 40:
            raw += b"\x00" * (descriptor_size - 40)
        return raw

    @classmethod
    def from_bytes(cls, data: bytes, descriptor_size: int = 48) -> "UefiMemoryDescriptor":
        if len(data) < 40:
            raise ValueError(f"Data length must be at least 40 bytes, got {len(data)}")
        mem_type_val, pad, phys, virt, pages, attr = struct.unpack("<IIQQQQ", data[:40])
        try:
            mtype = UefiMemoryType(mem_type_val)
        except ValueError:
            mtype = UefiMemoryType.EfiReservedMemoryType
        return cls(
            type=mtype,
            pad=pad,
            physical_start=phys,
            virtual_start=virt,
            number_of_pages=pages,
            attribute=attr,
        )


class UefiMemoryMapSimulator:
    """Simulates a complete UEFI memory map (array of EFI_MEMORY_DESCRIPTOR)
    and demonstrates descriptor pointer stepping and memory lifecycle management.
    """

    def __init__(self, descriptor_size: int = 48, descriptor_version: int = 1):
        self.descriptors: List[UefiMemoryDescriptor] = []
        self.descriptor_size = descriptor_size
        self.descriptor_version = descriptor_version

    def add_descriptor(self, desc: UefiMemoryDescriptor) -> None:
        self.descriptors.append(desc)

    @property
    def total_entries(self) -> int:
        return len(self.descriptors)

    @property
    def map_size_bytes(self) -> int:
        return len(self.descriptors) * self.descriptor_size

    def total_bytes_by_type(self) -> Dict[UefiMemoryType, int]:
        result = {t: 0 for t in UefiMemoryType}
        for d in self.descriptors:
            result[d.type] += d.size_bytes
        return result

    def total_pages_by_type(self) -> Dict[UefiMemoryType, int]:
        result = {t: 0 for t in UefiMemoryType}
        for d in self.descriptors:
            result[d.type] += d.number_of_pages
        return result

    def total_conventional_bytes(self) -> int:
        return sum(d.size_bytes for d in self.descriptors if d.is_conventional)

    def total_boot_services_reclaimable_bytes(self) -> int:
        return sum(d.size_bytes for d in self.descriptors if d.is_reclaimable_after_exit_boot_services)

    def total_runtime_bytes(self) -> int:
        return sum(d.size_bytes for d in self.descriptors if d.is_runtime_service)

    def serialize(self) -> bytes:
        return b"".join(d.to_bytes(self.descriptor_size) for d in self.descriptors)

    @classmethod
    def deserialize(cls, buffer: bytes, descriptor_size: int = 48, descriptor_version: int = 1) -> "UefiMemoryMapSimulator":
        instance = cls(descriptor_size=descriptor_size, descriptor_version=descriptor_version)
        offset = 0
        while offset + descriptor_size <= len(buffer):
            chunk = buffer[offset : offset + descriptor_size]
            instance.add_descriptor(UefiMemoryDescriptor.from_bytes(chunk, descriptor_size))
            offset += descriptor_size
        return instance

    @classmethod
    def create_standard_uefi_map(cls) -> "UefiMemoryMapSimulator":
        """Generates a realistic UEFI memory map reflecting QEMU OVMF execution with 4GB RAM."""
        sim = cls(descriptor_size=48, descriptor_version=1)
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiBootServicesCode,
            physical_start=0x00000000,
            virtual_start=0,
            number_of_pages=159,  # 0x00000 to 0x9F000 (636KB)
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiReservedMemoryType,
            physical_start=0x0009F000,
            virtual_start=0,
            number_of_pages=1,    # EBDA
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiConventionalMemory,
            physical_start=0x00100000,
            virtual_start=0,
            number_of_pages=32512,  # 1MB to ~128MB
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiLoaderCode,
            physical_start=0x06000000,
            virtual_start=0,
            number_of_pages=128,   # 512KB for bootloader binary
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiLoaderData,
            physical_start=0x06080000,
            virtual_start=0,
            number_of_pages=256,   # 1MB for loader stack/heap
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiConventionalMemory,
            physical_start=0x06180000,
            virtual_start=0,
            number_of_pages=498816, # ~1.9GB main conventional RAM
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiACPIReclaimMemory,
            physical_start=0x7FE00000,
            virtual_start=0,
            number_of_pages=128,   # 512KB ACPI tables
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiACPIMemoryNVS,
            physical_start=0x7FE80000,
            virtual_start=0,
            number_of_pages=64,    # 256KB ACPI NVS
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiRuntimeServicesCode,
            physical_start=0x7FEC0000,
            virtual_start=0,
            number_of_pages=128,   # 512KB Runtime Code
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC | EFI_MEMORY_RUNTIME,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiRuntimeServicesData,
            physical_start=0x7FF40000,
            virtual_start=0,
            number_of_pages=192,   # 768KB Runtime Data
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC | EFI_MEMORY_RUNTIME,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiMemoryMappedIO,
            physical_start=0xC0000000,
            virtual_start=0,
            number_of_pages=4096,  # 16MB GOP Framebuffer
            attribute=EFI_MEMORY_UC | EFI_MEMORY_WC | EFI_MEMORY_RUNTIME,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiMemoryMappedIO,
            physical_start=0xFEE00000,
            virtual_start=0,
            number_of_pages=1,     # Local APIC MMIO
            attribute=EFI_MEMORY_UC | EFI_MEMORY_RUNTIME,
        ))
        sim.add_descriptor(UefiMemoryDescriptor(
            type=UefiMemoryType.EfiConventionalMemory,
            physical_start=0x100000000,  # 4GB+ High RAM
            virtual_start=0,
            number_of_pages=524288,      # 2GB High RAM
            attribute=EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC,
        ))
        return sim


def plot_efi_memory_descriptor_layout_and_attributes(
    output_path: str = "2/result/fig2_07_efi_memory_descriptor_layout_and_attributes.png",
) -> str:
    """Renders a high-DPI explanatory diagram of EFI_MEMORY_DESCRIPTOR struct memory layout and attribute bits."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_layout, ax_attr) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"wspace": 0.28})

    # Left: EFI_MEMORY_DESCRIPTOR Struct Layout (40 bytes total)
    ax_layout.set_title("EFI_MEMORY_DESCRIPTOR 構造体のメモリ配置 (40 bytes / x86-64 EDK II)", fontsize=11, fontweight="bold", pad=12)
    ax_layout.set_xlim(0, 100)
    ax_layout.set_ylim(0, 100)
    ax_layout.axis("off")

    desc_fields = [
        ("0x00 - 0x03 (4B)", 82, 13, "#BBDEFB", "#1565C0", "Type (UINT32)\nメモリの種類 (EFI_MEMORY_TYPE: 0〜15)"),
        ("0x04 - 0x07 (4B)", 67, 13, "#EEEEEE", "#757575", "Pad (UINT32 パディング)\n64bit変数アライメントのための4バイト空白"),
        ("0x08 - 0x0F (8B)", 52, 13, "#C8E6C9", "#2E7D32", "PhysicalStart (EFI_PHYSICAL_ADDRESS / 64bit)\n領域の先頭物理アドレス (4KiB境界整列必須)"),
        ("0x10 - 0x17 (8B)", 37, 13, "#E1BEE7", "#6A1B9A", "VirtualStart (EFI_VIRTUAL_ADDRESS / 64bit)\n領域の仮想アドレス (ブートローダ実行時は0)"),
        ("0x18 - 0x1F (8B)", 22, 13, "#FFE0B2", "#E65100", "NumberOfPages (UINT64 / 64bit)\n領域に含まれる4KiBページ数 (サイズ = Pages × 4096)"),
        ("0x20 - 0x27 (8B)", 7, 13, "#FFCDD2", "#C62828", "Attribute (UINT64 / 64bit)\nキャッシュ性・アクセス保護・ランタイム保持フラグ"),
    ]

    for addr, y, h, c, bc, label in desc_fields:
        box = FancyBboxPatch((2, y), 50, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_layout.add_patch(box)
        ax_layout.text(27, y + h / 2, label, fontsize=7.2, ha="center", va="center", fontweight="bold", color="#212121")
        ax_layout.text(54, y + h / 2, addr, fontsize=7.2, va="center", family="monospace", color="#37474F")

    # Header / total size indicator
    ax_layout.text(50, 97, "sizeof(EFI_MEMORY_DESCRIPTOR) = 40 バイト (0x28) / descriptor_size >= 40", fontsize=8.8, ha="center", fontweight="bold", color="#0D47A1")

    # Right: Memory Attributes Bitmask (Attribute 64-bit breakdown)
    ax_attr.set_title("Attribute 属性フラグのビット定義と役割 (UEFI 2.10 Spec Table 7.3)", fontsize=11, fontweight="bold", pad=12)
    ax_attr.set_xlim(0, 100)
    ax_attr.set_ylim(0, 100)
    ax_attr.axis("off")

    attr_items = [
        ("0x8000_0000_0000_0000", 82, 13, "#EF9A9A", "#B71C1C", "EFI_MEMORY_RUNTIME (ビット63)\nOSカーネル起動後も仮想アドレス空間へのマッピング必須"),
        ("0x0000_0000_0002_0000", 67, 13, "#CE93D8", "#4A148C", "EFI_MEMORY_RO (Read-Only / ビット17)\n書き込み保護領域 (コードセクションやファームウェアROM)"),
        ("0x0000_0000_0000_4000", 52, 13, "#B39DDB", "#311B92", "EFI_MEMORY_XP (Execute-Protect / ビット14)\nNX (No-Execute) 実行禁止領域 (データ領域・スタック)"),
        ("0x0000_0000_0000_1000", 37, 13, "#90CAF9", "#0D47A1", "EFI_MEMORY_WP (Write-Protect / ビット12)\n書き込み保護可能属性"),
        ("0x0000_0000_0000_0008", 22, 13, "#A5D6A7", "#1B5E20", "EFI_MEMORY_WB (Write-Back キャッシュ / ビット3)\n通常の物理RAMに対する標準キャッシュ動作 (最速)"),
        ("0x0000_0000_0000_0002", 12, 8, "#FFE082", "#FF8F00", "EFI_MEMORY_WC (Write-Combining / ビット1)\nVRAM/フレームバッファ描画用のバースト書き込み"),
        ("0x0000_0000_0000_0001", 2, 8, "#FFAB91", "#BF360C", "EFI_MEMORY_UC (Uncacheable / ビット0)\nMMIOデバイス制御レジスタ用のキャッシュ無効領域"),
    ]

    for bitmask, y, h, c, bc, label in attr_items:
        box = FancyBboxPatch((2, y), 50, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_attr.add_patch(box)
        ax_attr.text(27, y + h / 2, label, fontsize=6.8, ha="center", va="center", fontweight="bold", color="#212121")
        ax_attr.text(54, y + h / 2, bitmask, fontsize=6.5, va="center", family="monospace", color="#37474F")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_uefi_memory_types_classification_and_lifecycle(
    output_path: str = "2/result/fig2_08_uefi_memory_types_classification_and_lifecycle.png",
) -> str:
    """Renders a high-DPI technical figure detailing the 16 EFI_MEMORY_TYPE classifications and OS lifecycle transitions."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_cat, ax_life) = plt.subplots(1, 2, figsize=(15, 9.0), gridspec_kw={"wspace": 0.28})

    # Left: Classification of 16 EFI_MEMORY_TYPEs
    ax_cat.set_title("EFI_MEMORY_TYPE 全16種と OS における分類", fontsize=11, fontweight="bold", pad=12)
    ax_cat.set_xlim(0, 100)
    ax_cat.set_ylim(0, 100)
    ax_cat.axis("off")

    categories = [
        ("0x7: EfiConventionalMemory", 83, 13, "#A5D6A7", "#1B5E20", "① OS完全自由利用可能 RAM\nOSがカーネル・ヒープ・ページテーブル・アプリに直ちに割り当て可能"),
        ("0x3: EfiBootServicesCode\n0x4: EfiBootServicesData\n0x1: EfiLoaderCode, 0x2: EfiLoaderData", 59, 21, "#BBDEFB", "#0D47A1", "② ブートローダ・ブートサービス領域\nブート中はUEFI/ローダが使用。\nExitBootServices() 呼び出し後にOSが完全回収・再利用可能！"),
        ("0x5: EfiRuntimeServicesCode\n0x6: EfiRuntimeServicesData\n0x9: EfiACPIReclaimMemory\n0xA: EfiACPIMemoryNVS", 33, 23, "#FFE082", "#E65100", "③ ファームウェア実行・保持領域\nOS起動後もUEFIランタイム関数 (時刻・電源) や\nACPIテーブル解析のために保持・仮想マッピングが必要"),
        ("0xB: EfiMemoryMappedIO\n0xC: EfiMemoryMappedIOPortSpace\n0x0: EfiReservedMemoryType\n0x8: EfiUnusableMemory, 0xE: EfiPersistentMemory", 3, 27, "#FFCDD2", "#B71C1C", "④ MMIOデバイス・予約・使用不可領域\nハードウェア制御レジスタ (GOP, APIC, PCIe BAR) や\n故障RAM・ファームウェア予約空間。RAMとして割り当て厳禁！"),
    ]

    for name, y, h, c, bc, label in categories:
        box = FancyBboxPatch((2, y), 50, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_cat.add_patch(box)
        ax_cat.text(27, y + h / 2, label, fontsize=6.8, ha="center", va="center", fontweight="bold", color="#212121")
        ax_cat.text(54, y + h / 2, name, fontsize=6.8, va="center", family="monospace", color="#37474F")

    # Right: OS Lifecycle Timeline before and after ExitBootServices()
    ax_life.set_title("ExitBootServices() 前後のメモリ再利用ライフサイクル推移", fontsize=11, fontweight="bold", pad=12)
    ax_life.set_xlim(0, 100)
    ax_life.set_ylim(0, 100)
    ax_life.axis("off")

    phases = [
        ("フェーズ 1: UEFI ブートローダ実行中", 78, 17, "#E3F2FD", "#1565C0",
         "【gBS (Boot Services) 生存】\n・gBS->AllocatePages() / GetMemoryMap() 等のUEFI機能が全て使用可能\n・ブートローダ自身 (LoaderCode/Data) と UEFIがメモリの多くを専有\n・ConventionalMemory のみ空きRAMとして認識"),
        ("=== gBS->ExitBootServices(image_handle, map_key) 実行 ===", 68, 6, "#FFD54F", "#F57F17",
         "⚠️ 境界線: UEFIブートサービス消滅・ハードウェアタイマ等の停止"),
        ("フェーズ 2: OS カーネル初期化 & メモリ回収", 45, 19, "#C8E6C9", "#2E7D32",
         "【ブートサービス領域の回収 (Reclaim)】\n・BootServicesCode / BootServicesData (~数MB〜数十MB) を全回収\n・LoaderCode / LoaderData もカーネルジャンプ完了後に回収可能\n・OSのビットマップメモリマネージャ (BitmapMemoryManager) にマージ"),
        ("フェーズ 3: OS 永続稼働 (ランタイムサービス連携)", 18, 23, "#FFF9C4", "#F57F17",
         "【ランタイム領域の永続保護】\n・RuntimeServicesCode / RuntimeServicesData は破壊厳禁！\n・gRT->SetVirtualAddressMap() でカーネル仮想アドレス空間にマップ\n・gRT->ResetSystem() (シャットダウン) や gRT->GetTime() で利用"),
    ]

    for title, y, h, c, bc, desc in phases:
        box = FancyBboxPatch((2, y), 96, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_life.add_patch(box)
        ax_life.text(50, y + h * 0.75, title, fontsize=7.8, ha="center", va="center", fontweight="bold", color="#0D47A1")
        ax_life.text(50, y + h * 0.35, desc, fontsize=6.8, ha="center", va="center", color="#212121")

    # Arrow from Phase 1 to Phase 2 to Phase 3
    ax_life.annotate("", xy=(50, 68), xytext=(50, 78),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#1565C0"))
    ax_life.annotate("", xy=(50, 45), xytext=(50, 68),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#2E7D32"))
    ax_life.annotate("", xy=(50, 18), xytext=(50, 45),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#E65100"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


@dataclass
class MemoryMap:
    """Python representation of MikanOS 'struct MemoryMap'.

    struct MemoryMap {
      UINTN buffer_size;
      VOID* buffer;
      UINTN map_size;
      UINTN map_key;
      UINTN descriptor_size;
      UINT32 descriptor_version;
    };
    """
    buffer_size: int
    buffer: Optional[bytearray] = None
    map_size: int = 0
    map_key: int = 0
    descriptor_size: int = 48
    descriptor_version: int = 1

    @classmethod
    def allocate(cls, size_in_bytes: int = 16384) -> "MemoryMap":
        """Allocates a buffer of given size (default: 16KiB = 4096 * 4)."""
        return cls(buffer_size=size_in_bytes, buffer=bytearray(size_in_bytes))

    def get_descriptor(self, index: int) -> UefiMemoryDescriptor:
        """Returns the descriptor at the given index using descriptor_size pointer stepping."""
        if self.buffer is None:
            raise ValueError("Buffer is null")
        offset = index * self.descriptor_size
        if offset + self.descriptor_size > self.map_size:
            raise IndexError(f"Descriptor index {index} out of range (offset {offset} exceeds map_size {self.map_size})")
        return UefiMemoryDescriptor.from_bytes(
            bytes(self.buffer[offset : offset + self.descriptor_size]),
            descriptor_size=self.descriptor_size,
        )

    def __iter__(self):
        """Iterates over all descriptors in the buffer with descriptor_size stepping."""
        if self.buffer is None or self.descriptor_size == 0:
            return
        offset = 0
        while offset + self.descriptor_size <= self.map_size:
            chunk = bytes(self.buffer[offset : offset + self.descriptor_size])
            yield UefiMemoryDescriptor.from_bytes(chunk, descriptor_size=self.descriptor_size)
            offset += self.descriptor_size

    @property
    def entry_count(self) -> int:
        if self.descriptor_size == 0:
            return 0
        return self.map_size // self.descriptor_size


class UefiBootServicesSimulator:
    """Simulates UEFI Boot Services (gBS) table operations related to memory management,
    specifically GetMemoryMap(), AllocatePages(), and ExitBootServices().
    Tracks MapKey state and validates memory map consistency across lifecycle changes.
    """

    def __init__(
        self,
        memory_map_sim: Optional[UefiMemoryMapSimulator] = None,
        initial_map_key: int = 100,
    ) -> None:
        if memory_map_sim is None:
            self.memory_map = UefiMemoryMapSimulator.create_standard_uefi_map()
        else:
            self.memory_map = memory_map_sim
        self.map_key = initial_map_key
        self.boot_services_exited = False
        self.allocation_history: List[str] = []

    def get_memory_map(self, map_struct: MemoryMap) -> int:
        """Implements gBS->GetMemoryMap() behavior.

        EFI_STATUS GetMemoryMap(
            IN OUT UINTN                  *MemoryMapSize,
            OUT    EFI_MEMORY_DESCRIPTOR  *MemoryMap,
            OUT    UINTN                  *MapKey,
            OUT    UINTN                  *DescriptorSize,
            OUT    UINT32                 *DescriptorVersion
        );
        """
        if self.boot_services_exited:
            return UefiStatus.UNSUPPORTED

        needed_size = self.memory_map.map_size_bytes

        # If buffer is None or too small, UEFI returns EFI_BUFFER_TOO_SMALL
        # and writes the required size into MemoryMapSize.
        if map_struct.buffer is None or map_struct.buffer_size < needed_size:
            map_struct.map_size = needed_size
            return UefiStatus.BUFFER_TOO_SMALL

        # Successful copy
        raw_bytes = self.memory_map.serialize()
        map_struct.buffer[:len(raw_bytes)] = raw_bytes
        map_struct.map_size = needed_size
        map_struct.map_key = self.map_key
        map_struct.descriptor_size = self.memory_map.descriptor_size
        map_struct.descriptor_version = self.memory_map.descriptor_version
        return UefiStatus.SUCCESS

    def allocate_pages(self, memory_type: UefiMemoryType, pages: int) -> int:
        """Simulates gBS->AllocatePages(). Alters memory map and increments MapKey."""
        if self.boot_services_exited:
            raise RuntimeError("Cannot allocate pages after ExitBootServices")

        # Find first conventional memory chunk with enough pages
        for idx, desc in enumerate(self.memory_map.descriptors):
            if desc.type == UefiMemoryType.EfiConventionalMemory and desc.number_of_pages >= pages:
                allocated_start = desc.physical_start
                new_allocated = UefiMemoryDescriptor(
                    type=memory_type,
                    physical_start=allocated_start,
                    virtual_start=0,
                    number_of_pages=pages,
                    attribute=desc.attribute,
                )
                if desc.number_of_pages == pages:
                    self.memory_map.descriptors[idx] = new_allocated
                else:
                    remaining = UefiMemoryDescriptor(
                        type=UefiMemoryType.EfiConventionalMemory,
                        physical_start=allocated_start + pages * 4096,
                        virtual_start=0,
                        number_of_pages=desc.number_of_pages - pages,
                        attribute=desc.attribute,
                    )
                    self.memory_map.descriptors[idx] = new_allocated
                    self.memory_map.descriptors.insert(idx + 1, remaining)

                self.map_key += 1
                self.allocation_history.append(
                    f"AllocatePages: {pages} pages ({pages * 4096} B) at 0x{allocated_start:08X}, new MapKey={self.map_key}"
                )
                return allocated_start

        raise RuntimeError("Out of conventional memory in simulator")

    def exit_boot_services(self, image_handle: int, map_key: int) -> int:
        """Simulates gBS->ExitBootServices(ImageHandle, MapKey).
        Returns EFI_SUCCESS if MapKey matches; otherwise EFI_INVALID_PARAMETER.
        """
        if self.boot_services_exited:
            return UefiStatus.INVALID_PARAMETER
        if map_key != self.map_key:
            return UefiStatus.INVALID_PARAMETER
        self.boot_services_exited = True
        return UefiStatus.SUCCESS


def plot_uefi_get_memory_map_protocol_and_buffer_flow(
    output_path: str = "2/result/fig2_09_uefi_get_memory_map_protocol_and_buffer_flow.png",
) -> str:
    """Renders high-DPI diagram of GetMemoryMap() argument protocol, buffer sizing, and pointer iteration."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_proto, ax_flow, ax_ptr) = plt.subplots(1, 3, figsize=(18, 8.5), gridspec_kw={"width_ratios": [1.1, 1.0, 1.1], "wspace": 0.26})

    # Subplot 1: Protocol Signature & Parameter Semantics
    ax_proto.set_title("1. gBS->GetMemoryMap() 引数仕様と IN/OUT 変換", fontsize=11, fontweight="bold", pad=12)
    ax_proto.set_xlim(0, 100)
    ax_proto.set_ylim(0, 100)
    ax_proto.axis("off")

    params = [
        ("IN OUT UINTN *MemoryMapSize", 80, 16, "#BBDEFB", "#0D47A1",
         "【IN】 呼び出し元が確保したバッファのバイトサイズ\n【OUT 成功時】 実際に書き込まれたメモリマップサイズ\n【OUT 容量不足】 必要な最小バイトサイズ (EFI_BUFFER_TOO_SMALL)"),
        ("OUT EFI_MEMORY_DESCRIPTOR *MemoryMap", 60, 16, "#C8E6C9", "#2E7D32",
         "【OUT】 ディスクリプタ配列を書き込むバッファの先頭アドレス\nNULL またはサイズ不足時は何も書き込まれない"),
        ("OUT UINTN *MapKey", 42, 14, "#FFE082", "#E65100",
         "【OUT】 現在のメモリマップ識別キー (Revision Key)\nExitBootServices() 呼び出し時の整合性検証に必須！"),
        ("OUT UINTN *DescriptorSize", 24, 14, "#E1BEE7", "#4A148C",
         "【OUT】 1つのディスクリプタのバイトサイズ (>= sizeof(DESC))\nパディングや将来の拡張に対応するため、ポインタ加算で使用！"),
        ("OUT UINT32 *DescriptorVersion", 6, 14, "#CFD8DC", "#37474F",
         "【OUT】 ディスクリプタ構造体のバージョン (UEFI Spec では 1)"),
    ]

    for name, y, h, c, bc, desc in params:
        box = FancyBboxPatch((2, y), 96, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_proto.add_patch(box)
        ax_proto.text(5, y + h * 0.75, name, fontsize=8.0, va="center", family="monospace", fontweight="bold", color=bc)
        ax_proto.text(5, y + h * 0.35, desc, fontsize=6.8, va="center", color="#212121")

    # Subplot 2: Buffer Allocation Strategies: Fixed vs Dynamic
    ax_flow.set_title("2. バッファ確保方針の比較 (固定 vs 動的)", fontsize=11, fontweight="bold", pad=12)
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0, 100)
    ax_flow.axis("off")

    # MikanOS Fixed Buffer
    box_fixed = FancyBboxPatch((2, 53), 96, 43, boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32", lw=1.5)
    ax_flow.add_patch(box_fixed)
    ax_flow.text(50, 91, "【MikanOS 採用】 固定バッファ (16 KiB)", fontsize=9.0, ha="center", fontweight="bold", color="#1B5E20")
    ax_flow.text(50, 77, "CHAR8 memmap_buf[4096 * 4];\nstruct MemoryMap memmap = {sizeof(memmap_buf), ...};", fontsize=7.2, ha="center", family="monospace", color="#0D47A1")
    ax_flow.text(50, 62, "[OK] 利点: 事前静的確保のため UEFIメモリ確保API 不要\n[OK] メモリマップを自ら変質させない (ゼロ副作用)\n[OK] 一般的なPCで数十〜百数十エントリ (~8KiB) に対し\n16KiB (約341エントリ分) で十分な安全域を確保", fontsize=6.8, ha="center", color="#212121")

    # Dynamic Allocation Pitfall
    box_dyn = FancyBboxPatch((2, 4), 96, 44, boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#E65100", lw=1.5)
    ax_flow.add_patch(box_dyn)
    ax_flow.text(50, 43, "【動的確保方式】 二段階問い合わせの落とし穴", fontsize=9.0, ha="center", fontweight="bold", color="#BF360C")
    ax_flow.text(50, 31, "1. GetMemoryMap(size=0) -> EFI_BUFFER_TOO_SMALL\n2. gBS->AllocatePool(needed_size)\n3. GetMemoryMap(allocated_buf) -> 再び失敗！？", fontsize=7.0, ha="center", color="#37474F")
    ax_flow.text(50, 13, "[!] 動く標的問題 (Moving Target Problem):\nAllocatePool 自体がメモリマップを分割・変更し、\nエントリ数が増加してバッファが再オーバーフロー！\n→ 余裕を持たせた再試行ループ (+マージン) が必須", fontsize=6.6, ha="center", color="#B71C1C")

    # Subplot 3: Variable-size Descriptor Iteration & Pointer Math
    ax_ptr.set_title("3. 可変長ディスクリプタ走査とポインタ演算", fontsize=11, fontweight="bold", pad=12)
    ax_ptr.set_xlim(0, 100)
    ax_ptr.set_ylim(0, 100)
    ax_ptr.axis("off")

    # Buffer layout visualization
    buf_box = FancyBboxPatch((4, 60), 92, 35, boxstyle="round,pad=0.2", facecolor="#F5F5F5", edgecolor="#616161", lw=1.2)
    ax_ptr.add_patch(buf_box)
    ax_ptr.text(50, 90, "MemoryMap.buffer (バイト列メモリ配置)", fontsize=8.2, ha="center", fontweight="bold", color="#37474F")

    entries = [
        ("Desc [0]\n(48B+pad)", 8, 26, "#90CAF9"),
        ("Desc [1]\n(48B+pad)", 38, 26, "#A5D6A7"),
        ("Desc [2]\n(48B+pad)", 68, 26, "#FFE082"),
    ]
    for label, x, w, c in entries:
        box_e = FancyBboxPatch((x, 65), w, 18, boxstyle="square,pad=0", facecolor=c, edgecolor="#424242", lw=1.0)
        ax_ptr.add_patch(box_e)
        ax_ptr.text(x + w / 2, 74, label, fontsize=6.8, ha="center", va="center", fontweight="bold")

    # Pointer stepping arrows
    ax_ptr.annotate("iter", xy=(8, 62), xytext=(8, 52),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1565C0"))
    ax_ptr.annotate("+ descriptor_size", xy=(38, 62), xytext=(20, 52),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#2E7D32"))
    ax_ptr.annotate("+ descriptor_size", xy=(68, 62), xytext=(50, 52),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#E65100"))

    # C code correct vs incorrect comparison
    code_box = FancyBboxPatch((4, 4), 92, 42, boxstyle="round,pad=0.3", facecolor="#EDE7F6", edgecolor="#512DA8", lw=1.2)
    ax_ptr.add_patch(code_box)
    ax_ptr.text(50, 41, "C言語における正しいポインタ加算 (MikanOS実装)", fontsize=8.0, ha="center", fontweight="bold", color="#311B92")
    ax_ptr.text(6, 27, "[OK] 正解 (バイトポインタ加算):\nEFI_PHYSICAL_ADDRESS iter;\nfor (iter = (EFI_PHYSICAL_ADDRESS)map->buffer;\n     iter < (EFI_PHYSICAL_ADDRESS)map->buffer + map->map_size;\n     iter += map->descriptor_size) { ... }",
                fontsize=6.5, color="#1B5E20")
    ax_ptr.text(6, 9, "[NG] 誤り: desc++ (C言語の型ポインタ加算)\nsizeof(EFI_MEMORY_DESCRIPTOR) 分しか進まず、\nファームウェアがパディングを持つ環境で全データがズレて破壊！",
                fontsize=6.5, color="#B71C1C")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_uefi_mapkey_lifecycle_and_exit_boot_services_consistency(
    output_path: str = "2/result/fig2_10_uefi_mapkey_lifecycle_and_exit_boot_services_consistency.png",
) -> str:
    """Renders high-DPI diagram of MapKey state machine and ExitBootServices consistency verification."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_seq, ax_why) = plt.subplots(2, 1, figsize=(15, 10.0), gridspec_kw={"height_ratios": [1.2, 0.9], "hspace": 0.28})

    # Upper Subplot: MapKey Lifecycle Sequence
    ax_seq.set_title("1. MapKey ライフサイクル推移と ExitBootServices() 整合性検証シーケンス", fontsize=11, fontweight="bold", pad=12)
    ax_seq.set_xlim(0, 100)
    ax_seq.set_ylim(0, 100)
    ax_seq.axis("off")

    steps = [
        ("① 初期状態 (ファームウェア稼働)", 81, 14, "#E3F2FD", "#1565C0",
         "UEFI ファームウェア内部の MapKey カウンタ = 42\nシステムメモリ状態: 安定"),
        ("② OSブートローダ: GetMemoryMap() 呼び出し", 63, 14, "#C8E6C9", "#2E7D32",
         "gBS->GetMemoryMap(&map) 成功！\nOSローダが取得した map.map_key = 42 (現時点の最新状態と一致)"),
        ("③ 【割り込み/イベント】 バックグラウンドでのメモリ確保発生！", 45, 14, "#FFCDD2", "#C62828",
         "タイマ・USBドライバ等が gBS->AllocatePool() を実行\nファームウェアが内部の MapKey を 43 にインクリメント！ ⚠️ OSの持つキー(42)と乖離！"),
        ("④ OSブートローダ: ExitBootServices(handle, 42) 実行", 27, 14, "#FFE082", "#E65100",
         "ファームウェア判定: 入力キー(42) != 内部キー(43)\n→ EFI_INVALID_PARAMETER で拒否！ OSへの制御移行を緊急停止"),
        ("⑤ 再試行: GetMemoryMap() 再取得 & ExitBootServices(handle, 43)", 7, 16, "#D1C4E9", "#4A148C",
         "1. GetMemoryMap() を即座に再呼び出し -> 最新 map_key=43 を取得\n2. メモリ確保を行わずに直ちに ExitBootServices(handle, 43) を実行\n→ EFI_SUCCESS！ ブートサービス完全停止・OSカーネルへ制御移行！"),
    ]

    for title, y, h, c, bc, desc in steps:
        box = FancyBboxPatch((2, y), 96, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_seq.add_patch(box)
        ax_seq.text(5, y + h * 0.75, title, fontsize=7.8, fontweight="bold", color=bc)
        ax_seq.text(5, y + h * 0.35, desc, fontsize=6.8, color="#212121")

    # Downward sequence arrows
    for y_arrow in [81, 63, 45, 27]:
        ax_seq.annotate("", xy=(50, y_arrow), xytext=(50, y_arrow + 2),
                        arrowprops=dict(arrowstyle="->", lw=2.0, color="#37474F"))

    # Lower Subplot: Why UEFI enforces MapKey Consistency
    ax_why.set_title("2. なぜ MapKey による整合性保証がカーネル保護に必須なのか？", fontsize=11, fontweight="bold", pad=12)
    ax_why.set_xlim(0, 100)
    ax_why.set_ylim(0, 100)
    ax_why.axis("off")

    reasons = [
        ("A. カーネル展開領域の誤破壊防止", 2, 46, 50, 42, "#E8F5E9", "#1B5E20",
         "もし MapKey 検証がなければ:\nOS は『ここは空きメモリ (ConventionalMemory) だ』\nと思い込んでカーネルコードを展開する。\nしかし直前にファームウェアが ACPI や Runtime 領域として\n再割り当てしていた場合、OS がファームウェアの生きた\nデータを上書き破壊し、起動不能 (Triple Fault / ハング) に陥る。"),
        ("B. 仮想メモリ (ページング) 構築の完全性", 52, 46, 50, 42, "#EDE7F6", "#311B92",
         "OS カーネルはメモリマップを元にページテーブルを構築する:\nRuntimeServices や MMIO は 1:1 アイデンティティマップまたは\nカーネル専用仮想アドレスに正しくマップされなければならない。\nマップ取得後に境界が変化していると、間違った属性 (キャッシュ無効等)\nが適用され、MMIO や ACPI アクセス時に CPU 例外が発生する。"),
    ]

    for title, x, w, y, h, c, bc, desc in reasons:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_why.add_patch(box)
        ax_why.text(x + w / 2, y + h * 0.85, title, fontsize=8.2, ha="center", fontweight="bold", color=bc)
        ax_why.text(x + 2, y + h * 0.45, desc, fontsize=6.8, va="center", color="#212121")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Section 2.6: Saving Memory Map to File Simulations (EFI_FILE_PROTOCOL & CSV)
# ==============================================================================

# UEFI File Open Mode Flags
EFI_FILE_MODE_READ: int = 0x0000000000000001
EFI_FILE_MODE_WRITE: int = 0x0000000000000002
EFI_FILE_MODE_CREATE: int = 0x8000000000000000

# Standard UEFI Protocol GUID constants
EFI_LOADED_IMAGE_PROTOCOL_GUID: str = "5B1B31A1-9562-11D2-8E3F-00A0C969723B"
EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID: str = "964E5B22-6459-11D2-8E39-00A0C969723B"


class MockFileProtocol:
    """Mock implementation of EFI_FILE_PROTOCOL representing a UEFI file or directory."""

    def __init__(
        self,
        name: str = "",
        mode: int = EFI_FILE_MODE_READ,
        attributes: int = 0,
        is_directory: bool = False,
    ) -> None:
        self.name: str = name
        self.mode: int = mode
        self.attributes: int = attributes
        self.is_directory: bool = is_directory
        self.is_closed: bool = False
        self.buffer: bytearray = bytearray()
        self.children: dict[str, MockFileProtocol] = {} if is_directory else {}
        self.cursor: int = 0

    def write(self, data: Union[str, bytes]) -> int:
        """Writes string or byte data to the file, returning bytes written."""
        if self.is_closed:
            raise RuntimeError(f"Cannot write to closed file '{self.name}' (EFI_DEVICE_ERROR)")
        if self.is_directory:
            raise RuntimeError(f"Cannot write data directly to directory '{self.name}'")
        if not (self.mode & EFI_FILE_MODE_WRITE):
            raise PermissionError(f"File '{self.name}' was not opened with EFI_FILE_MODE_WRITE")

        raw_bytes = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self.buffer.extend(raw_bytes)
        self.cursor += len(raw_bytes)
        return len(raw_bytes)

    def read(self, size: int) -> bytes:
        """Reads up to `size` bytes from the file."""
        if self.is_closed:
            raise RuntimeError(f"Cannot read from closed file '{self.name}'")
        chunk = self.buffer[self.cursor : self.cursor + size]
        self.cursor += len(chunk)
        return bytes(chunk)

    def set_position(self, position: int) -> None:
        """Sets file cursor position (simulating EFI_FILE_PROTOCOL.SetPosition)."""
        if self.is_closed:
            raise RuntimeError(f"Cannot set position on closed file '{self.name}'")
        if position == 0xFFFF_FFFF_FFFF_FFFF:  # Special UEFI constant for EOF
            self.cursor = len(self.buffer)
        else:
            self.cursor = max(0, min(position, len(self.buffer)))

    def open(
        self,
        relative_path: str,
        mode: int = EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE,
        attributes: int = 0,
    ) -> MockFileProtocol:
        """Opens or creates a file or sub-directory within this directory."""
        if not self.is_directory:
            raise RuntimeError(f"Cannot open sub-file in non-directory '{self.name}'")
        if self.is_closed:
            raise RuntimeError(f"Directory '{self.name}' is closed")

        clean_path = relative_path.lstrip("\\/").replace("\\", "/")
        if not clean_path:
            raise ValueError("Path must not be empty")

        if clean_path in self.children:
            existing = self.children[clean_path]
            existing.mode = mode
            existing.is_closed = False
            existing.cursor = 0
            return existing

        if mode & EFI_FILE_MODE_CREATE:
            new_file = MockFileProtocol(name=clean_path, mode=mode, attributes=attributes, is_directory=False)
            self.children[clean_path] = new_file
            return new_file

        raise FileNotFoundError(f"File '{clean_path}' not found in directory '{self.name}' (EFI_NOT_FOUND)")

    def close(self) -> None:
        """Closes the file and flushes unwritten buffers."""
        self.is_closed = True

    def get_content_as_str(self, encoding: str = "utf-8") -> str:
        """Decodes the current file buffer as string."""
        return self.buffer.decode(encoding, errors="replace")

    def get_content_bytes(self) -> bytes:
        """Returns a copy of the raw bytes."""
        return bytes(self.buffer)


class MockSimpleFileSystemProtocol:
    """Mock implementation of EFI_SIMPLE_FILE_SYSTEM_PROTOCOL."""

    def __init__(self, volume_label: str = "MIKANOS_BOOT") -> None:
        self.volume_label: str = volume_label
        self.root_dir: MockFileProtocol = MockFileProtocol(name="\\", is_directory=True)

    def open_volume(self) -> MockFileProtocol:
        """Returns the root directory EFI_FILE_PROTOCOL."""
        return self.root_dir


@dataclass
class MockLoadedImageProtocol:
    """Mock implementation of EFI_LOADED_IMAGE_PROTOCOL."""

    image_base: int = 0x4000_0000
    image_size: int = 1024 * 1024
    device_handle: int = 0x1001


class MockUefiEnvironment:
    """Full mock UEFI environment simulating handle database and protocols."""

    def __init__(self) -> None:
        self.image_handle: int = 0x0001
        self.device_handle: int = 0x1001
        self.loaded_image: MockLoadedImageProtocol = MockLoadedImageProtocol(device_handle=self.device_handle)
        self.file_system: MockSimpleFileSystemProtocol = MockSimpleFileSystemProtocol()

        self.protocols: dict[int, dict[str, Any]] = {
            self.image_handle: {
                EFI_LOADED_IMAGE_PROTOCOL_GUID: self.loaded_image,
            },
            self.device_handle: {
                EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID: self.file_system,
            },
        }

    def open_protocol(self, handle: int, protocol_guid: str) -> Any:
        """Simulates gBS->OpenProtocol()."""
        handle_map = self.protocols.get(handle)
        if not handle_map or protocol_guid not in handle_map:
            raise KeyError(f"Protocol '{protocol_guid}' not installed on handle 0x{handle:04x} (EFI_UNSUPPORTED)")
        return handle_map[protocol_guid]

    def open_root_dir(self, image_handle: Optional[int] = None) -> MockFileProtocol:
        """Implements OpenRootDir logic from MikanOS LoaderPkg/Main.c."""
        if image_handle is None:
            image_handle = self.image_handle

        loaded_image = self.open_protocol(image_handle, EFI_LOADED_IMAGE_PROTOCOL_GUID)
        fs = self.open_protocol(loaded_image.device_handle, EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID)
        return fs.open_volume()


class MemoryMapCsvExporter:
    """Simulates SaveMemoryMap() function from MikanOS day02b."""

    CSV_HEADER: str = "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n"
    ATTRIBUTE_MASK: int = 0xFFFFF

    @classmethod
    def save_memory_map(
        cls,
        memory_map: MemoryMapStruct,
        file: MockFileProtocol,
    ) -> str:
        """Exports UEFI MemoryMapStruct to a CSV formatted file protocol."""
        file.write(cls.CSV_HEADER)

        for i, desc in enumerate(memory_map.descriptors):
            line = (
                f"{i}, {desc.type.value:x}, {desc.type.name}, "
                f"{desc.physical_start:08x}, {desc.number_of_pages:x}, "
                f"{desc.attribute & cls.ATTRIBUTE_MASK:x}\n"
            )
            file.write(line)

        return file.get_content_as_str()


def parse_memory_map_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parses exported MikanOS memory map CSV text into structured records."""
    lines = [line.strip() for line in csv_text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0]
    expected_header_prefix = "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute"
    if not header.startswith(expected_header_prefix):
        raise ValueError(f"Invalid memory map CSV header: '{header}'")

    records: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx = int(parts[0], 10)
        type_val = int(parts[1], 16)
        type_name = parts[2]
        phys_start = int(parts[3], 16)
        num_pages = int(parts[4], 16)
        attribute = int(parts[5], 16)
        size_bytes = num_pages * 4096
        phys_end = phys_start + size_bytes

        records.append({
            "Index": idx,
            "Type": type_val,
            "TypeName": type_name,
            "PhysicalStart": phys_start,
            "PhysicalEnd": phys_end,
            "NumberOfPages": num_pages,
            "Attribute": attribute,
            "SizeBytes": size_bytes,
            "SizeMiB": size_bytes / (1024 * 1024),
        })

    return records


def plot_uefi_file_system_protocol_hierarchy(
    output_path: str = "2/result/fig2_11_uefi_file_system_protocol_hierarchy.png",
) -> str:
    """Renders high-DPI diagram of UEFI Protocol hierarchy, OpenRootDir, and file lifecycle."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_arch, ax_life) = plt.subplots(
        2, 1, figsize=(15, 11.0), gridspec_kw={"height_ratios": [1.1, 0.9], "hspace": 0.28}
    )

    # Upper subplot: UEFI Protocol Object Model Architecture
    ax_arch.set_title("1. UEFI プロトコル指向アーキテクチャとファイルシステム探索階層 (OpenRootDir)", fontsize=11, fontweight="bold", pad=12)
    ax_arch.set_xlim(0, 100)
    ax_arch.set_ylim(0, 100)
    ax_arch.axis("off")

    nodes = [
        ("ImageHandle (EFI_HANDLE)\nブートローダ自身の実行モジュール", 3, 62, 28, 28, "#E3F2FD", "#1565C0",
         "エントリポイント引数\nimage_handle"),
        ("EFI_LOADED_IMAGE_PROTOCOL\nロード元デバイスの特定", 36, 62, 28, 28, "#EDE7F6", "#4527A0",
         "gEfiLoadedImageProtocolGuid\n-> DeviceHandle を取得"),
        ("DeviceHandle & EFI_SIMPLE_FILE_SYSTEM_PROTOCOL\nFATファイルシステムの抽象化", 69, 62, 28, 28, "#E8F5E9", "#2E7D32",
         "gEfiSimpleFileSystemProtocolGuid\n-> OpenVolume(&root_dir)"),
        ("root_dir (EFI_FILE_PROTOCOL*)\nルートディレクトリ ('\\')", 36, 12, 28, 28, "#FFF3E0", "#E65100",
         "root_dir->Open(..., L\"\\\\memmap\",\n  READ|WRITE|CREATE, 0)"),
        ("memmap_file (EFI_FILE_PROTOCOL*)\n新規作成された \\memmap ファイル", 69, 12, 28, 28, "#FFEBEE", "#C62828",
         "SaveMemoryMap(&memmap, file)\nfile->Write(len, buf)\nfile->Close()"),
    ]

    for title, x, y, w, h, c, bc, sub in nodes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", facecolor=c, edgecolor=bc, lw=1.3)
        ax_arch.add_patch(box)
        ax_arch.text(x + w / 2, y + h * 0.72, title, fontsize=8.0, ha="center", fontweight="bold", color=bc)
        ax_arch.text(x + w / 2, y + h * 0.28, sub, fontsize=7.0, ha="center", color="#263238")

    # Arrows in Upper Subplot
    # 1 -> 2
    ax_arch.annotate("", xy=(36, 76), xytext=(31, 76),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color="#1565C0"))
    ax_arch.text(33.5, 78, "① OpenProtocol", fontsize=7.2, ha="center", color="#1565C0", fontweight="bold")

    # 2 -> 3
    ax_arch.annotate("", xy=(69, 76), xytext=(64, 76),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color="#4527A0"))
    ax_arch.text(66.5, 78, "② OpenProtocol", fontsize=7.2, ha="center", color="#4527A0", fontweight="bold")

    # 3 -> 4
    ax_arch.annotate("", xy=(50, 40), xytext=(83, 62),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color="#2E7D32", connectionstyle="arc3,rad=-0.15"))
    ax_arch.text(70, 48, "③ OpenVolume", fontsize=7.2, ha="center", color="#2E7D32", fontweight="bold")

    # 4 -> 5
    ax_arch.annotate("", xy=(69, 26), xytext=(64, 26),
                     arrowprops=dict(arrowstyle="->", lw=2.0, color="#E65100"))
    ax_arch.text(66.5, 29, "④ root_dir->Open", fontsize=7.2, ha="center", color="#E65100", fontweight="bold")

    # Lower subplot: File Protocol Lifecycle & Safety
    ax_life.set_title("2. EFI_FILE_PROTOCOL のライフサイクル遷移とクローズフラッシュの重要性", fontsize=11, fontweight="bold", pad=12)
    ax_life.set_xlim(0, 100)
    ax_life.set_ylim(0, 100)
    ax_life.axis("off")

    life_steps = [
        ("1. Open (ファイル作成・開く)", 3, 20, 21, 65, "#E1F5FE", "#0288D1",
         "root_dir->Open(\n  ..., L\"\\\\memmap\",\n  READ|WRITE|CREATE, 0)\n\n• UTF-16LE パス指定\n• FAT32 ディレクトリ内に\n  新規エントリ確保"),
        ("2. Header Write (CSVヘッダ書き込み)", 27, 20, 21, 65, "#E8F5E9", "#388E3C",
         "len = AsciiStrLen(header);\nfile->Write(file, &len, header);\n\n• len は IN OUT 引数\n• 要求バイト数渡し\n• 実書込バイト数返却"),
        ("3. Row Loop (メモリディスクリプタ書込)", 51, 20, 21, 65, "#FFF8E1", "#FFA000",
         "for (iter... iter += desc_size)\n  AsciiSPrint(buf, ...);\n  file->Write(file, &len, buf);\n\n• 可変長バイトポインタ走査\n• 属性下位20bitマスク"),
        ("4. Close (バッファフラッシュ・完了)", 75, 20, 22, 65, "#FFEBEE", "#D32F2F",
         "file->Close(memmap_file);\n\n⚠️ クローズの絶対遵守:\n未書込キャッシュをFATへ書き出し、\nディレクトリエントリの\nファイルサイズを確定更新！"),
    ]

    for title, x, y, w, h, c, bc, desc in life_steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", facecolor=c, edgecolor=bc, lw=1.2)
        ax_life.add_patch(box)
        ax_life.text(x + w / 2, y + h * 0.88, title, fontsize=7.6, ha="center", fontweight="bold", color=bc)
        ax_life.text(x + 1.5, y + h * 0.45, desc, fontsize=6.8, va="center", color="#212121")

    for x_arr in [24, 48, 72]:
        ax_life.annotate("", xy=(x_arr + 3, 52), xytext=(x_arr, 52),
                         arrowprops=dict(arrowstyle="->", lw=2.0, color="#455A64"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_memory_map_csv_export_and_layout(
    output_path: str = "2/result/fig2_12_memory_map_csv_export_and_layout.png",
) -> str:
    """Renders high-DPI diagram of Memory Map CSV layout, attribute bitmasking, and parsed distribution."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_fmt, ax_chart) = plt.subplots(
        2, 1, figsize=(15, 10.5), gridspec_kw={"height_ratios": [1.1, 0.9], "hspace": 0.32}
    )

    # Upper Subplot: CSV Format & Attribute Bitmasking
    ax_fmt.set_title("1. メモリディスクリプタ構造体から CSV 行へのマッピングと属性ビットマスク数理", fontsize=11, fontweight="bold", pad=12)
    ax_fmt.set_xlim(0, 100)
    ax_fmt.set_ylim(0, 100)
    ax_fmt.axis("off")

    # CSV Format Table Visual
    csv_box = FancyBboxPatch((2, 58), 96, 38, boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#37474F", lw=1.2)
    ax_fmt.add_patch(csv_box)
    ax_fmt.text(50, 91, '出力CSV行フォーマット: AsciiSPrint(buf, sizeof(buf), "%u, %x, %-ls, %08lx, %lx, %lx\\n", ...)',
                fontsize=8.0, ha="center", fontweight="bold", color="#263238")

    cols = [
        ("Index (%u)", 4, 13, "0, 1, 2...", "ディスクリプタ通し番号 (i++)\n0 から昇順", "#E3F2FD", "#1565C0"),
        ("Type (%x)", 18, 12, "7 (Conventional)", "EFI_MEMORY_TYPE 16進値\n例: 7 = 0x7", "#EDE7F6", "#512DA8"),
        ("Type(name) (%-ls)", 31, 20, "EfiConventionalMemory", "UTF-16 文字列名称 (左詰)\nGetMemoryTypeUnicode()", "#E8F5E9", "#2E7D32"),
        ("PhysicalStart (%08lx)", 52, 19, "00100000", "物理開始アドレス (16進数)\n最低8桁ゼロ埋めフォーマット", "#FFF3E0", "#E65100"),
        ("NumberOfPages (%lx)", 72, 13, "6f7f", "4KiBページ数 (16進数)\nバイト数 = pages * 4096", "#FCE4EC", "#C2185B"),
        ("Attribute (%lx)", 86, 11, "f", "desc->Attribute & 0xffffflu\n下位20bit属性マスク", "#FFEBEE", "#C62828"),
    ]

    for title, x, w, ex, desc, c, bc in cols:
        b = FancyBboxPatch((x, 62), w, 24, boxstyle="round,pad=0.2", facecolor=c, edgecolor=bc, lw=1.0)
        ax_fmt.add_patch(b)
        ax_fmt.text(x + w / 2, 81, title, fontsize=6.8, ha="center", fontweight="bold", color=bc)
        ax_fmt.text(x + w / 2, 73, f"例: {ex}", fontsize=6.3, ha="center", color="#212121")
        ax_fmt.text(x + w / 2, 65, desc, fontsize=5.8, ha="center", color="#546E7A")

    # Bitmask Explanation Box
    mask_box = FancyBboxPatch((2, 4), 96, 48, boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#FBC02D", lw=1.2)
    ax_fmt.add_patch(mask_box)
    ax_fmt.text(50, 46, "【重要数理】なぜ desc->Attribute & 0xffffflu (下位20ビットマスク) を行うのか？",
                fontsize=8.0, ha="center", fontweight="bold", color="#F57F17")
    ax_fmt.text(5, 33, "• UEFI 仕様のメモリキャッシュ属性ビット定義 (UEFI Spec 7.2.1):\n"
                       "  - Bit 0: UC (Uncacheable, 0x1)          - Bit 1: WC (Write-Coalescing, 0x2)    - Bit 2: WT (Write-Through, 0x4)\n"
                       "  - Bit 3: WB (Write-Back, 0x8)           - Bit 4: UCE (Uncacheable Exported)    - Bit 12: WP (Write-Protected)\n"
                       "  - Bit 13: RP (Read-Protected)           - Bit 14: XP (Execute-Protected)       - Bit 15: NV (Non-Volatile)\n"
                       "• 0xFFFFF (20ビットマスク) の意義:\n"
                       "  上位ビット (Bit 20〜63) はプラットフォーム・ベンダー固有予約属性や実行時属性 (EFI_MEMORY_RUNTIME = 0x8000000000000000) であり、\n"
                       "  CSV出力時に上位の巨大なフラグを切り離して、メモリ本来のキャッシュ・保護属性 (下位20bit) のみを簡潔に比較観察するため！",
                fontsize=6.8, color="#212121")

    # Lower Subplot: Parsed Memory Distribution from simulated CSV
    ax_chart.set_title("2. 保存された \\memmap CSV のパース集計によるメモリタイプ別容量分布 (QEMU OVMF 標準 2GiB)", fontsize=11, fontweight="bold", pad=12)

    # Sample distribution simulating QEMU standard 2GB memory map
    categories = [
        "Conventional\n(空きRAM)",
        "BootServices\n(Data/Code)",
        "RuntimeServices\n(Data/Code)",
        "ACPI\n(Reclaim/NVS)",
        "Reserved\n(ファームウェア)",
        "MMIO\n(デバイスI/O)",
    ]
    sizes_mib = [1912.4, 68.5, 12.8, 4.2, 10.1, 32.0]
    colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#607D8B", "#F44336"]

    bars = ax_chart.bar(categories, sizes_mib, color=colors, edgecolor="#37474F", width=0.55, lw=1.0)
    ax_chart.set_ylabel("容量 (MiB)", fontsize=9.0, fontweight="bold")
    ax_chart.set_ylim(0, 2200)

    for bar, val in zip(bars, sizes_mib):
        y = bar.get_height()
        pct = (val / sum(sizes_mib)) * 100
        ax_chart.text(
            bar.get_x() + bar.get_width() / 2, y + 35,
            f"{val:.1f} MiB\n({pct:.1f}%)", ha="center", fontsize=7.2, fontweight="bold", color="#263238"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 2.7: Analyzing Memory Map Output (メモリマップの確認)
# ==============================================================================

class MemoryMapAnalyzer:
    """Parses and computes statistical and architectural metrics on UEFI memory maps.

    Handles CSV-formatted memory map dumps output by MikanOS SaveMemoryMap()
    and evaluates free conventional memory, boot services reclaimability,
    fragmentation, and physical address space layout.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        """Initializes analyzer with records parsed from memory map CSV."""
        self.records: list[dict[str, Any]] = sorted(records, key=lambda r: r["PhysicalStart"])
        self._total_pages: int = sum(r["NumberOfPages"] for r in self.records)
        self._total_bytes: int = self._total_pages * 4096

    @classmethod
    def from_csv(cls, csv_text: str) -> "MemoryMapAnalyzer":
        """Factory method to construct analyzer directly from CSV string."""
        records = parse_memory_map_csv(csv_text)
        return cls(records)

    @property
    def total_records(self) -> int:
        """Returns the number of memory descriptor records."""
        return len(self.records)

    @property
    def total_pages(self) -> int:
        """Returns the total number of 4KiB pages across all descriptors."""
        return self._total_pages

    @property
    def total_bytes(self) -> int:
        """Returns the total size in bytes across all descriptors."""
        return self._total_bytes

    @property
    def total_mib(self) -> float:
        """Returns the total size in mebibytes (MiB)."""
        return self._total_bytes / (1024 * 1024)

    @property
    def total_gib(self) -> float:
        """Returns the total size in gibibytes (GiB)."""
        return self._total_bytes / (1024 * 1024 * 1024)

    def type_breakdown(self) -> dict[str, dict[str, Any]]:
        """Computes page count, byte size, MiB, and percentage per memory type."""
        breakdown: dict[str, dict[str, Any]] = {}

        for r in self.records:
            type_name = r["TypeName"]
            if type_name not in breakdown:
                breakdown[type_name] = {
                    "type_code": r["Type"],
                    "count": 0,
                    "pages": 0,
                    "bytes": 0,
                    "mib": 0.0,
                    "percentage": 0.0,
                }
            breakdown[type_name]["count"] += 1
            breakdown[type_name]["pages"] += r["NumberOfPages"]
            breakdown[type_name]["bytes"] += r["SizeBytes"]

        for info in breakdown.values():
            info["mib"] = info["bytes"] / (1024 * 1024)
            info["percentage"] = (info["bytes"] / self._total_bytes * 100) if self._total_bytes > 0 else 0.0

        return breakdown

    def conventional_memory_summary(self) -> dict[str, Any]:
        """Analyzes EfiConventionalMemory (Type 7), representing immediately usable RAM."""
        conv_records = [r for r in self.records if r["Type"] == 7 or r["TypeName"] == "EfiConventionalMemory"]
        pages = sum(r["NumberOfPages"] for r in conv_records)
        bytes_val = pages * 4096
        mib_val = bytes_val / (1024 * 1024)
        gib_val = bytes_val / (1024 * 1024 * 1024)
        pct = (bytes_val / self._total_bytes * 100) if self._total_bytes > 0 else 0.0

        max_block = max(conv_records, key=lambda r: r["NumberOfPages"]) if conv_records else None
        min_block = min(conv_records, key=lambda r: r["NumberOfPages"]) if conv_records else None

        return {
            "record_count": len(conv_records),
            "pages": pages,
            "bytes": bytes_val,
            "mib": mib_val,
            "gib": gib_val,
            "percentage": pct,
            "regions": conv_records,
            "max_block": max_block,
            "min_block": min_block,
        }

    def reclaimable_boot_services_summary(self) -> dict[str, Any]:
        """Analyzes EfiBootServicesCode and EfiBootServicesData reclaimable after ExitBootServices()."""
        bs_records = [
            r for r in self.records
            if r["Type"] in (3, 4) or r["TypeName"] in ("EfiBootServicesCode", "EfiBootServicesData")
        ]
        pages = sum(r["NumberOfPages"] for r in bs_records)
        bytes_val = pages * 4096
        mib_val = bytes_val / (1024 * 1024)
        pct = (bytes_val / self._total_bytes * 100) if self._total_bytes > 0 else 0.0

        return {
            "record_count": len(bs_records),
            "pages": pages,
            "bytes": bytes_val,
            "mib": mib_val,
            "percentage": pct,
            "regions": bs_records,
        }

    def available_memory_after_boot(self) -> dict[str, Any]:
        """Computes total usable memory once ExitBootServices() reclaims boot services memory."""
        conv = self.conventional_memory_summary()
        bs = self.reclaimable_boot_services_summary()

        combined_pages = conv["pages"] + bs["pages"]
        combined_bytes = combined_pages * 4096
        combined_mib = combined_bytes / (1024 * 1024)
        combined_gib = combined_bytes / (1024 * 1024 * 1024)
        combined_pct = (combined_bytes / self._total_bytes * 100) if self._total_bytes > 0 else 0.0

        return {
            "pages": combined_pages,
            "bytes": combined_bytes,
            "mib": combined_mib,
            "gib": combined_gib,
            "percentage": combined_pct,
            "conventional_mib": conv["mib"],
            "reclaimable_bs_mib": bs["mib"],
        }

    def fragmentation_metrics(self) -> dict[str, Any]:
        """Evaluates memory fragmentation metrics for EfiConventionalMemory."""
        conv = self.conventional_memory_summary()
        count = conv["record_count"]
        if count == 0:
            return {
                "num_blocks": 0,
                "avg_block_mib": 0.0,
                "largest_block_ratio": 0.0,
                "fragmentation_index": 0.0,
            }

        avg_mib = conv["mib"] / count
        max_block_mib = (conv["max_block"]["SizeBytes"] / (1024 * 1024)) if conv["max_block"] else 0.0
        largest_ratio = (max_block_mib / conv["mib"]) if conv["mib"] > 0 else 1.0
        # Fragmentation index: 1.0 - largest_block_ratio (0.0 = completely contiguous, 1.0 = highly fragmented)
        frag_index = 1.0 - largest_ratio

        return {
            "num_blocks": count,
            "avg_block_mib": avg_mib,
            "largest_block_ratio": largest_ratio,
            "fragmentation_index": frag_index,
        }

    def find_free_regions_above(self, min_bytes: int) -> list[dict[str, Any]]:
        """Filters conventional memory regions large enough to satisfy a minimum byte size."""
        conv = self.conventional_memory_summary()
        return [r for r in conv["regions"] if r["SizeBytes"] >= min_bytes]

    def gap_analysis(self) -> list[dict[str, Any]]:
        """Identifies address gaps between consecutive descriptors in physical memory."""
        gaps: list[dict[str, Any]] = []
        for i in range(len(self.records) - 1):
            curr_end = self.records[i]["PhysicalEnd"]
            next_start = self.records[i + 1]["PhysicalStart"]
            if next_start > curr_end:
                gap_size = next_start - curr_end
                gaps.append({
                    "gap_index": len(gaps),
                    "after_index": self.records[i]["Index"],
                    "before_index": self.records[i + 1]["Index"],
                    "gap_start": curr_end,
                    "gap_end": next_start,
                    "gap_bytes": gap_size,
                    "gap_mib": gap_size / (1024 * 1024),
                })
        return gaps

    def summary_table(self) -> str:
        """Formats a human-readable text summary table of the memory breakdown."""
        lines = [
            f"{'Type (Name)':<28} | {'Count':>5} | {'Pages':>10} | {'Size (MiB)':>10} | {'Ratio (%)':>9}",
            "-" * 72,
        ]
        breakdown = self.type_breakdown()
        for name, data in sorted(breakdown.items(), key=lambda item: item[1]["bytes"], reverse=True):
            lines.append(
                f"{name:<28} | {data['count']:>5} | {data['pages']:>10} | {data['mib']:>10.2f} | {data['percentage']:>8.2f}%"
            )
        lines.append("-" * 72)
        lines.append(
            f"{'TOTAL':<28} | {self.total_records:>5} | {self.total_pages:>10} | {self.total_mib:>10.2f} | 100.00%"
        )
        return "\n".join(lines)

    @classmethod
    def create_sample_qemu_memmap_csv(cls, total_ram_gib: float = 2.0) -> str:
        """Generates realistic QEMU OVMF CSV memory map text for simulation and testing."""
        # Scale conventional memory based on total_ram_gib (default 2.0 GiB)
        # 1 page = 4096 bytes, 2GiB = 524288 pages
        total_pages_approx = int(total_ram_gib * 1024 * 1024 * 1024 // 4096)
        main_ram_pages = total_pages_approx - 0x10000  # reserve ~256MiB for firmware and MMIO

        csv_rows = [
            "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute",
            "0, 0, EfiReservedMemoryType, 00000000, 1, f",
            "1, 7, EfiConventionalMemory, 00001000, 9f, f",
            "2, 0, EfiReservedMemoryType, 000a0000, 60, f",
            f"3, 7, EfiConventionalMemory, 00100000, {main_ram_pages - 0x6000:x}, f",
            f"4, 3, EfiBootServicesCode, {(main_ram_pages - 0x5f00) * 4096:08x}, 1200, f",
            f"5, 4, EfiBootServicesData, {(main_ram_pages - 0x4d00) * 4096:08x}, 2400, f",
            f"6, 5, EfiRuntimeServicesCode, {(main_ram_pages - 0x2900) * 4096:08x}, 400, f",
            f"7, 6, EfiRuntimeServicesData, {(main_ram_pages - 0x2500) * 4096:08x}, 800, f",
            f"8, 9, EfiACPIReclaimMemory, {(main_ram_pages - 0x1d00) * 4096:08x}, 80, f",
            f"9, a, EfiACPIMemoryNVS, {(main_ram_pages - 0x1c80) * 4096:08x}, 80, f",
            f"10, 7, EfiConventionalMemory, {(main_ram_pages - 0x1c00) * 4096:08x}, 1c00, f",
            "11, b, EfiMemoryMappedIO, 80000000, 10000, 1",
            "12, b, EfiMemoryMappedIO, fed00000, 1, 1",
            "13, b, EfiMemoryMappedIO, fee00000, 1, 1",
            "14, b, EfiMemoryMappedIO, ffe00000, 200, 1",
        ]
        return "\n".join(csv_rows) + "\n"


def plot_memory_map_analysis(
    analyzer: MemoryMapAnalyzer,
    output_path: str = "2/result/fig2_13_memory_map_analysis_breakdown.png",
) -> str:
    """Renders high-DPI figure of memory map breakdown, conventional RAM, and reclaimable pool."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.22)

    ax_pie = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_avail = fig.add_subplot(gs[1, 0])
    ax_frag = fig.add_subplot(gs[1, 1])

    # 1. Type Breakdown Pie Chart
    breakdown = analyzer.type_breakdown()
    labels: list[str] = []
    sizes: list[float] = []
    color_map = {
        "EfiConventionalMemory": "#4CAF50",
        "EfiBootServicesCode": "#2196F3",
        "EfiBootServicesData": "#03A9F4",
        "EfiRuntimeServicesCode": "#9C27B0",
        "EfiRuntimeServicesData": "#BA68C8",
        "EfiACPIReclaimMemory": "#FF9800",
        "EfiACPIMemoryNVS": "#FFB74D",
        "EfiReservedMemoryType": "#78909C",
        "EfiMemoryMappedIO": "#E53935",
        "EfiLoaderCode": "#009688",
        "EfiLoaderData": "#4DB6AC",
    }
    colors: list[str] = []

    for name, data in sorted(breakdown.items(), key=lambda item: item[1]["bytes"], reverse=True):
        if data["mib"] > 0.1:  # Filter out trivial micro-regions for clean pie chart
            labels.append(f"{name}\n({data['mib']:.1f} MiB)")
            sizes.append(data["mib"])
            colors.append(color_map.get(name, "#B0BEC5"))

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
        textprops={"fontsize": 7.5, "color": "#263238"},
        wedgeprops={"edgecolor": "#37474F", "linewidth": 1.0, "antialiased": True}
    )
    for at in autotexts:
        at.set_fontsize(7.0)
        at.set_fontweight("bold")
    ax_pie.set_title("1. メモリタイプ別容量構成比 (Type Breakdown Pie Chart)", fontsize=10.5, fontweight="bold", pad=10)

    # 2. Type Breakdown Horizontal Bar
    names = [name.replace("Efi", "") for name in breakdown.keys()]
    mibs = [data["mib"] for data in breakdown.values()]
    bar_colors = [color_map.get(name, "#B0BEC5") for name in breakdown.keys()]

    y_pos = range(len(names))
    ax_bar.barh(y_pos, mibs, color=bar_colors, edgecolor="#37474F", height=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=7.5)
    ax_bar.set_xlabel("容量 (MiB)", fontsize=8.5, fontweight="bold")
    ax_bar.set_title("2. メモリタイプ別実容量 (MiB) 比較", fontsize=10.5, fontweight="bold", pad=10)
    for i, v in enumerate(mibs):
        ax_bar.text(v + (max(mibs) * 0.01), i, f"{v:.1f} MiB", va="center", fontsize=7.0, color="#263238")

    # 3. Memory Usability Stages (Bootloader vs Post-ExitBootServices)
    conv = analyzer.conventional_memory_summary()
    bs = analyzer.reclaimable_boot_services_summary()
    post_boot = analyzer.available_memory_after_boot()

    stages = ["ブートローダ実行中\n(Conventional RAMのみ)", "OS起動完了後\n(ExitBootServices 後)"]
    conv_parts = [conv["mib"], conv["mib"]]
    bs_parts = [0.0, bs["mib"]]

    b1 = ax_avail.bar(stages, conv_parts, label="EfiConventionalMemory (即時空きRAM)", color="#4CAF50", edgecolor="#2E7D32", width=0.45)
    b2 = ax_avail.bar(stages, bs_parts, bottom=conv_parts, label="BootServices (回収可能RAM)", color="#2196F3", edgecolor="#1565C0", width=0.45)

    ax_avail.set_ylabel("利用可能 RAM 容量 (MiB)", fontsize=8.5, fontweight="bold")
    ax_avail.set_ylim(0, post_boot["mib"] * 1.25)
    ax_avail.set_title("3. OS起動ステージ別 利用可能RAM容量推移", fontsize=10.5, fontweight="bold", pad=10)
    ax_avail.legend(loc="upper left", fontsize=7.5)

    ax_avail.text(0, conv["mib"] + 30, f"{conv['mib']:.1f} MiB\n({conv['percentage']:.1f}%)", ha="center", fontsize=7.8, fontweight="bold", color="#1B5E20")
    ax_avail.text(1, post_boot["mib"] + 30, f"{post_boot['mib']:.1f} MiB\n(+{bs['mib']:.1f} MiB 回収)", ha="center", fontsize=7.8, fontweight="bold", color="#0D47A1")

    # 4. Fragmentation & Region Distribution
    frag = analyzer.fragmentation_metrics()
    conv_regions = conv["regions"]
    region_labels = [f"Region #{r['Index']}\n({r['PhysicalStart']:08x})" for r in conv_regions[:8]]
    region_sizes = [r["SizeMiB"] for r in conv_regions[:8]]

    ax_frag.bar(range(len(region_sizes)), region_sizes, color="#81C784", edgecolor="#2E7D32", width=0.5)
    ax_frag.set_xticks(range(len(region_labels)))
    ax_frag.set_xticklabels(region_labels, fontsize=6.8, rotation=25, ha="right")
    ax_frag.set_ylabel("ブロックサイズ (MiB)", fontsize=8.5, fontweight="bold")
    ax_frag.set_title(
        f"4. Conventional RAM 断片化分析 (総ブロック数: {frag['num_blocks']}, 断片化度: {frag['fragmentation_index']:.3f})",
        fontsize=10.5, fontweight="bold", pad=10
    )

    for i, v in enumerate(region_sizes):
        ax_frag.text(i, v + (max(region_sizes) * 0.02), f"{v:.1f} MiB", ha="center", fontsize=7.0, color="#1B5E20")

    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.96, hspace=0.32, wspace=0.25)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_physical_memory_layout_map(
    analyzer: MemoryMapAnalyzer,
    output_path: str = "2/result/fig2_14_physical_memory_layout_map.png",
) -> str:
    """Renders high-DPI horizontal allocation bar of 0-4GiB physical memory layout."""
    setup_plot_style()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_main, ax_low) = plt.subplots(
        2, 1, figsize=(15, 8.5), gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.40}
    )

    color_map = {
        "EfiConventionalMemory": "#4CAF50",
        "EfiBootServicesCode": "#2196F3",
        "EfiBootServicesData": "#03A9F4",
        "EfiRuntimeServicesCode": "#9C27B0",
        "EfiRuntimeServicesData": "#BA68C8",
        "EfiACPIReclaimMemory": "#FF9800",
        "EfiACPIMemoryNVS": "#FFB74D",
        "EfiReservedMemoryType": "#78909C",
        "EfiMemoryMappedIO": "#E53935",
        "EfiLoaderCode": "#009688",
        "EfiLoaderData": "#4DB6AC",
    }

    # 1. Main Memory Layout (0 to 4GiB)
    max_addr = max(r["PhysicalEnd"] for r in analyzer.records)
    plot_limit = max(max_addr, 0x100000000)  # At least 4GiB (0x100000000)
    limit_gib = plot_limit / (1024 * 1024 * 1024)

    ax_main.set_title(
        f"1. 物理アドレス空間全体 (0 〜 {limit_gib:.1f}GiB) のメモリマップ配置 (Physical Memory Allocation Map)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax_main.set_xlim(0, plot_limit)
    ax_main.set_ylim(-0.5, 1.5)
    ax_main.set_yticks([])

    # Plot memory segments as horizontal bar strips
    for r in analyzer.records:
        start = r["PhysicalStart"]
        width = r["SizeBytes"]
        color = color_map.get(r["TypeName"], "#B0BEC5")
        ax_main.barh(0.5, width, left=start, height=0.6, color=color, edgecolor="#263238", linewidth=0.5)

    # Key markers
    markers = [
        (0x0, "0x0 (Reset/IVT)"),
        (0x100000, "1MiB (High RAM)"),
        (0x80000000, "2GiB"),
        (0xE0000000, "3.5GiB (PCI MMIO Hole)"),
        (0x100000000, "4GiB (32-bit Limit)"),
    ]
    for addr, label in markers:
        if addr <= plot_limit:
            ax_main.axvline(x=addr, color="#B71C1C", linestyle="--", alpha=0.6, linewidth=1.0)
            ax_main.text(addr, 1.15, label, rotation=35, ha="left", fontsize=7.2, color="#B71C1C", fontweight="bold")

    ax_main.set_xlabel("物理アドレス (Bytes / Hex)", fontsize=8.5, fontweight="bold")
    # Format x-axis in Hex/GiB
    ticks = [0, 0x40000000, 0x80000000, 0xC0000000, 0x100000000]
    ax_main.set_xticks(ticks)
    ax_main.set_xticklabels(["0x0 (0GiB)", "0x40000000 (1GiB)", "0x80000000 (2GiB)", "0xC0000000 (3GiB)", "0x100000000 (4GiB)"], fontsize=7.5)

    # 2. Detailed Low Memory Layout (0 to 16MiB Zoom)
    low_limit = 16 * 1024 * 1024  # 16 MiB
    ax_low.set_title(
        "2. 低位メモリ領域詳細 (0 〜 16MiB 拡大表示: レガシー予約・低位Conventional・カーネル配置候補)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax_low.set_xlim(0, low_limit)
    ax_low.set_ylim(-0.5, 1.5)
    ax_low.set_yticks([])

    low_records = [r for r in analyzer.records if r["PhysicalStart"] < low_limit]
    for r in low_records:
        start = r["PhysicalStart"]
        width = min(r["SizeBytes"], low_limit - start)
        color = color_map.get(r["TypeName"], "#B0BEC5")
        ax_low.barh(0.5, width, left=start, height=0.6, color=color, edgecolor="#263238", linewidth=0.6)
        if width > 1024 * 512:  # Label regions larger than 512KiB
            ax_low.text(
                start + width / 2, 0.5, f"{r['TypeName'].replace('Efi', '')}\n({r['SizeBytes']/1024:.0f}KiB)",
                ha="center", va="center", fontsize=6.8, fontweight="bold", color="#FFFFFF" if color in ("#E53935", "#2196F3", "#9C27B0") else "#212121"
            )

    low_markers = [
        (0x0, "0x0 (IVT/BDA)"),
        (0x9F000, "0x9F000 (EBDA)"),
        (0xA0000, "0xA0000 (VRAM 128KB)"),
        (0x100000, "0x100000 (1MiB カーネル標準配置アドレス)"),
    ]
    for addr, label in low_markers:
        ax_low.axvline(x=addr, color="#0D47A1", linestyle=":", alpha=0.8, linewidth=1.2)
        ax_low.text(addr, 1.15, label, rotation=30, ha="left", fontsize=7.2, color="#0D47A1", fontweight="bold")

    ax_low.set_xlabel("物理アドレス (Bytes)", fontsize=8.5, fontweight="bold")
    low_ticks = [0, 0x100000, 0x400000, 0x800000, 0xC00000, 0x1000000]
    ax_low.set_xticks(low_ticks)
    ax_low.set_xticklabels(["0x0", "1MiB (0x100000)", "4MiB (0x400000)", "8MiB (0x800000)", "12MiB (0xC00000)", "16MiB (0x1000000)"], fontsize=7.5)

    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.96, hspace=0.45)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# =============================================================================
# Chapter 2.8: Pointers, Memory Addressing, and Pointer Arithmetic
# =============================================================================

@dataclass
class MemoryCellModel:
    """Byte-addressable memory space simulation for low-level pointer modeling."""
    base_address: int = 0x1000
    size: int = 4096
    memory: bytearray = field(default_factory=bytearray)

    def __post_init__(self):
        if not self.memory:
            self.memory = bytearray(self.size)

    def _check_bounds(self, addr: int, length: int) -> None:
        if addr < self.base_address or (addr + length) > (self.base_address + self.size):
            raise IndexError(
                f"Address out of bounds: [0x{addr:X}..0x{addr+length:X}) not in "
                f"[0x{self.base_address:X}..0x{self.base_address+self.size:X})"
            )

    def write_byte(self, addr: int, val: int) -> None:
        self._check_bounds(addr, 1)
        self.memory[addr - self.base_address] = val & 0xFF

    def read_byte(self, addr: int) -> int:
        self._check_bounds(addr, 1)
        return self.memory[addr - self.base_address]

    def write_bytes(self, addr: int, data: bytes) -> None:
        self._check_bounds(addr, len(data))
        offset = addr - self.base_address
        self.memory[offset:offset + len(data)] = data

    def read_bytes(self, addr: int, count: int) -> bytes:
        self._check_bounds(addr, count)
        offset = addr - self.base_address
        return bytes(self.memory[offset:offset + count])

    def write_uint16(self, addr: int, val: int) -> None:
        self.write_bytes(addr, struct.pack("<H", val & 0xFFFF))

    def read_uint16(self, addr: int) -> int:
        return struct.unpack("<H", self.read_bytes(addr, 2))[0]

    def write_uint32(self, addr: int, val: int) -> None:
        self.write_bytes(addr, struct.pack("<I", val & 0xFFFFFFFF))

    def read_uint32(self, addr: int) -> int:
        return struct.unpack("<I", self.read_bytes(addr, 4))[0]

    def write_int32(self, addr: int, val: int) -> None:
        self.write_bytes(addr, struct.pack("<i", val))

    def read_int32(self, addr: int) -> int:
        return struct.unpack("<i", self.read_bytes(addr, 4))[0]

    def write_uint64(self, addr: int, val: int) -> None:
        self.write_bytes(addr, struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))

    def read_uint64(self, addr: int) -> int:
        return struct.unpack("<Q", self.read_bytes(addr, 8))[0]

    def hex_dump(self, start_addr: int, count: int) -> list[str]:
        lines = []
        for a in range(start_addr, start_addr + count, 16):
            chunk_len = min(16, start_addr + count - a)
            b = self.read_bytes(a, chunk_len)
            hex_part = " ".join(f"{x:02X}" for x in b)
            ascii_part = "".join(chr(x) if 32 <= x < 127 else "." for x in b)
            lines.append(f"0x{a:08X}: {hex_part:<48}  |{ascii_part}|")
        return lines


@dataclass
class VariableEntry:
    """Represents a named variable in memory."""
    name: str
    type_name: str
    size: int
    address: int
    value: Any


@dataclass
class PointerVariable:
    """Represents a pointer variable in 64-bit architecture."""
    name: str
    target_type: str
    target_type_size: int
    pointer_addr: int
    target_addr: int
    pointer_size: int = 8  # 64-bit architecture pointers are always 8 bytes

    def dereference(self, memory: MemoryCellModel) -> Any:
        """Dereferences the pointer and returns the value at the target address."""
        if self.target_addr == 0:
            raise ValueError(f"Null pointer dereference on pointer '{self.name}' (0x0)")
        if self.target_type in ("char", "int8", "int8_t"):
            return memory.read_byte(self.target_addr)
        elif self.target_type in ("uint8", "uint8_t", "unsigned char"):
            return memory.read_byte(self.target_addr)
        elif self.target_type in ("short", "int16", "int16_t"):
            b = memory.read_bytes(self.target_addr, 2)
            return struct.unpack("<h", b)[0]
        elif self.target_type in ("uint16", "uint16_t", "unsigned short"):
            return memory.read_uint16(self.target_addr)
        elif self.target_type in ("int", "int32", "int32_t"):
            return memory.read_int32(self.target_addr)
        elif self.target_type in ("uint32", "uint32_t", "unsigned int"):
            return memory.read_uint32(self.target_addr)
        elif self.target_type in ("uint64", "uint64_t", "unsigned long long", "uintptr_t"):
            return memory.read_uint64(self.target_addr)
        else:
            return memory.read_bytes(self.target_addr, self.target_type_size)

    def store(self, memory: MemoryCellModel, value: Any) -> None:
        """Stores a value at the target address through dereferencing (*p = value)."""
        if self.target_addr == 0:
            raise ValueError(f"Null pointer write on pointer '{self.name}' (0x0)")
        if self.target_type in ("char", "int8", "int8_t", "uint8", "uint8_t"):
            memory.write_byte(self.target_addr, value)
        elif self.target_type in ("short", "int16", "int16_t"):
            memory.write_bytes(self.target_addr, struct.pack("<h", value))
        elif self.target_type in ("uint16", "uint16_t"):
            memory.write_uint16(self.target_addr, value)
        elif self.target_type in ("int", "int32", "int32_t"):
            memory.write_int32(self.target_addr, value)
        elif self.target_type in ("uint32", "uint32_t"):
            memory.write_uint32(self.target_addr, value)
        elif self.target_type in ("uint64", "uint64_t", "uintptr_t"):
            memory.write_uint64(self.target_addr, value)
        elif isinstance(value, (bytes, bytearray)):
            memory.write_bytes(self.target_addr, value)
        else:
            raise TypeError(f"Unsupported store type: {type(value)}")

    def advance(self, steps: int) -> int:
        """Computes pointer arithmetic: target_addr + steps * sizeof(T)."""
        return self.target_addr + steps * self.target_type_size


class PointerModelSimulator:
    """Simulates pointer variables, memory layout, and pointer arithmetic in 64-bit systems."""

    def __init__(self, base_address: int = 0x1000, size: int = 4096):
        self.memory = MemoryCellModel(base_address=base_address, size=size)
        self.variables: dict[str, VariableEntry] = {}
        self.pointers: dict[str, PointerVariable] = {}
        self.next_var_addr = base_address
        self.next_ptr_addr = base_address + 0x800  # Separate pointer stack region

    def allocate_variable(
        self,
        name: str,
        type_name: str,
        value: Any,
        address: Optional[int] = None,
    ) -> VariableEntry:
        type_sizes = {
            "char": 1, "uint8": 1, "uint8_t": 1, "int8_t": 1,
            "short": 2, "uint16": 2, "uint16_t": 2, "int16_t": 2,
            "int": 4, "uint32": 4, "uint32_t": 4, "int32_t": 4,
            "uint64": 8, "uint64_t": 8, "int64_t": 8, "uintptr_t": 8,
            "double": 8, "EFI_MEMORY_DESCRIPTOR": 48,
        }
        size = type_sizes.get(type_name, 4)
        if address is None:
            # 4 or 8-byte alignment
            align = min(size, 8)
            aligned_addr = (self.next_var_addr + (align - 1)) & ~(align - 1)
            address = aligned_addr
            self.next_var_addr = address + size

        entry = VariableEntry(name=name, type_name=type_name, size=size, address=address, value=value)
        self.variables[name] = entry

        # Write initial value into memory
        if type_name in ("int", "int32", "int32_t"):
            self.memory.write_int32(address, value)
        elif type_name in ("uint32", "uint32_t"):
            self.memory.write_uint32(address, value)
        elif type_name in ("uint64", "uint64_t", "uintptr_t"):
            self.memory.write_uint64(address, value)
        elif type_name in ("char", "uint8", "uint8_t"):
            self.memory.write_byte(address, value if isinstance(value, int) else ord(value))
        elif isinstance(value, bytes):
            self.memory.write_bytes(address, value)
        return entry

    def create_pointer(
        self,
        ptr_name: str,
        target_name: str,
        ptr_address: Optional[int] = None,
    ) -> PointerVariable:
        if target_name not in self.variables:
            raise KeyError(f"Variable '{target_name}' not found")
        target_var = self.variables[target_name]
        if ptr_address is None:
            # 8-byte alignment for 64-bit pointer
            aligned_addr = (self.next_ptr_addr + 7) & ~7
            ptr_address = aligned_addr
            self.next_ptr_addr = ptr_address + 8

        ptr = PointerVariable(
            name=ptr_name,
            target_type=target_var.type_name,
            target_type_size=target_var.size,
            pointer_addr=ptr_address,
            target_addr=target_var.address,
            pointer_size=8,
        )
        self.pointers[ptr_name] = ptr
        # Write target address into pointer memory (8 bytes little-endian)
        self.memory.write_uint64(ptr_address, target_var.address)
        return ptr

    def dereference(self, ptr_name: str) -> Any:
        if ptr_name not in self.pointers:
            raise KeyError(f"Pointer '{ptr_name}' not found")
        return self.pointers[ptr_name].dereference(self.memory)

    def assign_via_pointer(self, ptr_name: str, value: Any) -> None:
        if ptr_name not in self.pointers:
            raise KeyError(f"Pointer '{ptr_name}' not found")
        ptr = self.pointers[ptr_name]
        ptr.store(self.memory, value)
        # Update variable entry if pointing to tracked variable
        for var in self.variables.values():
            if var.address == ptr.target_addr:
                var.value = value

    def pointer_arithmetic(self, ptr_name: str, delta: int) -> int:
        if ptr_name not in self.pointers:
            raise KeyError(f"Pointer '{ptr_name}' not found")
        return self.pointers[ptr_name].advance(delta)

    def cast_pointer(self, ptr_name: str, new_type_name: str, new_type_size: int) -> PointerVariable:
        if ptr_name not in self.pointers:
            raise KeyError(f"Pointer '{ptr_name}' not found")
        old_ptr = self.pointers[ptr_name]
        new_name = f"{ptr_name}_cast_{new_type_name}"
        casted = PointerVariable(
            name=new_name,
            target_type=new_type_name,
            target_type_size=new_type_size,
            pointer_addr=old_ptr.pointer_addr,
            target_addr=old_ptr.target_addr,
            pointer_size=8,
        )
        self.pointers[new_name] = casted
        return casted


def plot_figure_2_15_pointer_memory_model(output_path: str = "2/result/fig2_15_pointer_memory_model_and_address.png") -> str:
    """Generates Figure 2.15: Memory byte array, variables, addresses, and pointer memory model."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(15, 9), facecolor="#FAFAFA")
    fig.suptitle(
        "Figure 2.15: メモリバイト配列、変数とアドレス、およびポインタ変数の 64-bit メモリモデル\n"
        "(Memory Byte Array, Variable Addressing, and 64-bit Pointer Architecture Model)",
        fontsize=13, fontweight="bold", y=0.98, color="#1A237E"
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.1, 1.0])
    ax_top = fig.add_subplot(gs[0, :])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------------------
    # Panel 1: Top - Linear Byte Array & Memory Cells (0x1000 to 0x100F, 0x2000 to 0x2007)
    # -------------------------------------------------------------------------
    ax_top.set_title("【1】物理/論理メモリ上の連続バイト配置とポインタ参照の相互関係 (Pointer-to-Pointee Mapping)", fontsize=11, fontweight="bold", pad=10)
    ax_top.set_xlim(-0.5, 17.5)
    ax_top.set_ylim(-0.5, 4.0)
    ax_top.axis("off")

    # Draw variable 'val' (int, 4 bytes at 0x1000)
    val_addrs = [0x1000, 0x1001, 0x1002, 0x1003]
    val_bytes = [0x78, 0x56, 0x34, 0x12]  # Little-endian for 0x12345678
    val_labels = ["LSB (+0)", "(+1)", "(+2)", "MSB (+3)"]

    ax_top.text(0.0, 3.4, "変数 int val = 0x12345678; (4バイト, 先頭アドレス &val == 0x1000)", fontsize=10.5, fontweight="bold", color="#B71C1C")
    for i, (addr, b, lbl) in enumerate(zip(val_addrs, val_bytes, val_labels)):
        rect = plt.Rectangle((i * 1.8 + 0.2, 2.0), 1.6, 1.1, facecolor="#FFCDD2", edgecolor="#B71C1C", linewidth=1.8)
        ax_top.add_patch(rect)
        ax_top.text(i * 1.8 + 1.0, 2.7, f"0x{b:02X}", ha="center", va="center", fontsize=11, fontweight="bold", color="#B71C1C")
        ax_top.text(i * 1.8 + 1.0, 2.3, f"{lbl}", ha="center", va="center", fontsize=8.0, color="#5D4037")
        ax_top.text(i * 1.8 + 1.0, 1.6, f"0x{addr:04X}", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#263238")

    # Draw pointer 'ptr' (int*, 8 bytes at 0x2000, holds 0x0000000000001000)
    ptr_addrs = [0x2000 + i for i in range(8)]
    ptr_bytes = [0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  # Little-endian for 0x1000

    ax_top.text(7.8, 3.4, "ポインタ変数 int* ptr = &val; (8バイト, 保持値 ptr == 0x0000000000001000)", fontsize=10.5, fontweight="bold", color="#0D47A1")
    for i, (addr, b) in enumerate(zip(ptr_addrs, ptr_bytes)):
        rect = plt.Rectangle((7.8 + i * 1.15, 2.0), 1.05, 1.1, facecolor="#BBDEFB", edgecolor="#0D47A1", linewidth=1.6)
        ax_top.add_patch(rect)
        ax_top.text(7.8 + i * 1.15 + 0.525, 2.7, f"0x{b:02X}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0D47A1")
        lbl = "LSB" if i == 0 else ("MSB" if i == 7 else f"+{i}")
        ax_top.text(7.8 + i * 1.15 + 0.525, 2.3, lbl, ha="center", va="center", fontsize=7.5, color="#37474F")
        ax_top.text(7.8 + i * 1.15 + 0.525, 1.6, f"..{addr & 0xF:X}", ha="center", va="center", fontsize=8.0, color="#455A64")

    ax_top.text(7.8 + 4.6, 1.25, "先頭アドレス &ptr == 0x2000 (x86-64 ではポインタは常に 8 バイト)", ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")

    # Arrows for & and *
    # Arrow 1: &val operation (from val to ptr)
    ax_top.annotate(
        "アドレス演算子 &val (0x1000 を取得)",
        xy=(7.7, 2.5), xycoords="data",
        xytext=(4.0, 0.7), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.2, connectionstyle="arc3,rad=-0.2"),
        fontsize=9.5, fontweight="bold", color="#2E7D32", ha="center"
    )

    # Arrow 2: *ptr dereference (from ptr pointing back to val)
    ax_top.annotate(
        "間接参照演算子 *ptr (0x1000 番地へアクセスして 4 バイト読み書き)",
        xy=(3.8, 3.15), xycoords="data",
        xytext=(8.0, 3.85), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#C2185B", lw=2.2, connectionstyle="arc3,rad=-0.15"),
        fontsize=9.5, fontweight="bold", color="#C2185B", ha="center"
    )

    # -------------------------------------------------------------------------
    # Panel 2: Bottom-Left - Little Endian vs Pointer Width
    # -------------------------------------------------------------------------
    ax_bottom_left.set_title("【2】データ型別メモリ占有幅とリトルエンディアン (Little-Endian Byte Layout)", fontsize=10.5, fontweight="bold", pad=8)
    ax_bottom_left.set_xlim(-0.5, 9.0)
    ax_bottom_left.set_ylim(-0.5, 4.5)
    ax_bottom_left.axis("off")

    types_info = [
        ("char c = 'A' (0x41)", 1, ["0x41"], "#C8E6C9", "#2E7D32"),
        ("short s = 0x1234", 2, ["0x34", "0x12"], "#FFE0B2", "#E65100"),
        ("int n = 0x12345678", 4, ["0x78", "0x56", "0x34", "0x12"], "#FFCDD2", "#C62828"),
        ("void* p / uint64_t (64-bit)", 8, ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"], "#BBDEFB", "#1565C0"),
    ]

    for y_idx, (name, size, byte_vals, fc, ec) in enumerate(reversed(types_info)):
        y = y_idx * 1.1 + 0.2
        ax_bottom_left.text(0.0, y + 0.45, f"{name} : {size} バイト", fontsize=8.8, fontweight="bold", color=ec)
        for b_idx, val_str in enumerate(byte_vals):
            rect = plt.Rectangle((b_idx * 0.95, y - 0.25), 0.88, 0.55, facecolor=fc, edgecolor=ec, linewidth=1.2)
            ax_bottom_left.add_patch(rect)
            ax_bottom_left.text(b_idx * 0.95 + 0.44, y + 0.02, val_str, ha="center", va="center", fontsize=8.0, fontweight="bold", color=ec)

    ax_bottom_left.text(
        4.0, -0.3,
        "※ x86-64 はリトルエンディアン: 最下位バイト (LSB) が最も若いアドレス (+0) に格納される",
        ha="center", fontsize=8.0, style="italic", color="#424242"
    )

    # -------------------------------------------------------------------------
    # Panel 3: Bottom-Right - The Core OS Truth about Pointers
    # -------------------------------------------------------------------------
    ax_bottom_right.set_title("【3】OS開発におけるポインタの本質と 3 大原則", fontsize=10.5, fontweight="bold", pad=8)
    ax_bottom_right.axis("off")

    truth_box = (
        "【OS自作におけるポインタの 3 大原則】\n\n"
        "(1) アドレスは単なる 64-bit の数値 (uint64_t / uintptr_t)\n"
        "   - メモリは 1 バイトごとに番号が付いた巨大なバイト配列。\n"
        "   - ポインタ自身も「アドレスという数値を記憶する 8 バイトの変数」に過ぎない。\n\n"
        "(2) 型 (Type) は「読み書き幅」と「加減算の歩進幅」を決める\n"
        "   - int* なのか char* なのかは、メモリ上の番地そのものではなく、\n"
        "     コンパイラに『何バイトを 1 単位としてアクセスするか』を教える属性。\n\n"
        "(3) OS カーネルは『固定された物理アドレス』を直接ポインタ化する\n"
        "   - 通常アプリ: OS にメモリを割り当ててもらい、そのポインタを受け取る。\n"
        "   - OS カーネル: VRAM (0xE0000000) や MMIO、カーネル配置 (0x100000) など\n"
        "     ハードウェア規定の数値を reinterpret_cast してポインタを能動生成する！"
    )
    ax_bottom_right.text(
        0.05, 0.92, truth_box, transform=ax_bottom_right.transAxes,
        fontsize=8.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#E8EAF6", edgecolor="#3949AB", linewidth=1.5),
        color="#1A237E"
    )

    plt.subplots_adjust(top=0.90, bottom=0.06, left=0.04, right=0.96, hspace=0.35, wspace=0.25)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_figure_2_16_pointer_arithmetic_scaling(output_path: str = "2/result/fig2_16_pointer_arithmetic_and_type_scaling.png") -> str:
    """Generates Figure 2.16: Pointer arithmetic, type-based stride scaling, and the byte-offset trap."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(15, 9), facecolor="#FAFAFA")
    fig.suptitle(
        "Figure 2.16: ポインタの型による算術演算スケーリングとバイト単位走査の数理モデル\n"
        "(Pointer Arithmetic Scaling by Type Size and the MikanOS Byte-Offset Casting Rule)",
        fontsize=13, fontweight="bold", y=0.98, color="#1A237E"
    )

    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
    ax_scaling = fig.add_subplot(gs[0])
    ax_uefi_trap = fig.add_subplot(gs[1])

    # -------------------------------------------------------------------------
    # Panel 1: Scaling Comparison across Types
    # -------------------------------------------------------------------------
    ax_scaling.set_title(
        "【1】型別ポインタ加算 p + n のアドレス変位: Address(p + n) = Address(p) + n × sizeof(*p)",
        fontsize=11, fontweight="bold", pad=10
    )
    ax_scaling.set_xlim(-1, 33)
    ax_scaling.set_ylim(-0.5, 5.2)
    ax_scaling.set_xlabel("先頭アドレスからの変位バイト数 (Bytes Offset from Base 0x1000)", fontsize=9.0, fontweight="bold")
    ax_scaling.set_xticks(range(0, 33, 4))
    ax_scaling.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax_scaling.set_yticks([])

    stride_demos = [
        ("char* (sizeof=1)", 1, "#4CAF50", "#1B5E20"),
        ("short* (sizeof=2)", 2, "#FF9800", "#E65100"),
        ("int* (sizeof=4)", 4, "#E91E63", "#880E4F"),
        ("uint64_t* (sizeof=8)", 8, "#2196F3", "#0D47A1"),
        ("struct EFI_DESCRIPTOR* (sizeof=48)", 16, "#9C27B0", "#4A148C"),  # Drawn scaled down for visual
    ]

    for row_idx, (t_name, size, fc, ec) in enumerate(stride_demos):
        y = row_idx * 1.0 + 0.4
        ax_scaling.text(-0.8, y, t_name, ha="right", va="center", fontsize=8.8, fontweight="bold", color=ec)

        # Plot base pointer p
        rect0 = plt.Rectangle((0, y - 0.3), max(0.8, size * 0.9), 0.6, facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax_scaling.add_patch(rect0)
        ax_scaling.text(max(0.4, size * 0.45), y, "p (+0)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#FFFFFF")

        # Plot p + 1, p + 2, p + 3
        max_steps = 1 if size == 16 else (3 if size <= 8 else 2)
        for step in range(1, max_steps + 1):
            offset = step * size
            if offset + size <= 32:
                rect = plt.Rectangle((offset, y - 0.3), max(0.8, size * 0.9), 0.6, facecolor=fc, edgecolor=ec, linewidth=1.4, alpha=0.85)
                ax_scaling.add_patch(rect)
                label = f"p+{step} (+{offset}B)" if size != 16 else f"p+1 (+48B ※縮尺)"
                ax_scaling.text(offset + max(0.4, size * 0.45), y, label, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#FFFFFF")

                # Arrow connecting steps
                prev_offset = (step - 1) * size
                ax_scaling.annotate(
                    "", xy=(offset, y), xytext=(prev_offset + max(0.8, size * 0.9), y),
                    arrowprops=dict(arrowstyle="->", color=ec, lw=1.2)
                )

    # -------------------------------------------------------------------------
    # Panel 2: The MikanOS MemoryMap Descriptor Traversal Trap
    # -------------------------------------------------------------------------
    ax_uefi_trap.set_title(
        "【2】MikanOS 第2章の核心: メモリマップ可変長ディスクリプタ走査における『型キャスト』の必然性",
        fontsize=11, fontweight="bold", pad=10
    )
    ax_uefi_trap.set_xlim(-0.5, 10.5)
    ax_uefi_trap.set_ylim(-0.5, 4.0)
    ax_uefi_trap.axis("off")

    # Wrong case
    ax_uefi_trap.text(
        0.0, 3.3,
        "[NG] 【落とし穴】ディスクリプタ型ポインタへの直接加算 (desc += descriptor_size)",
        fontsize=9.8, fontweight="bold", color="#D32F2F"
    )
    wrong_box = (
        "EFI_MEMORY_DESCRIPTOR* desc = buffer;\n"
        "desc += descriptor_size;  // 大惨事！\n"
        "// 理由: desc + 48 は 48 バイト進むのではなく、\n"
        "// 48 * sizeof(EFI_MEMORY_DESCRIPTOR) = 48 * 48 = 2,304 バイト先に飛んでしまう！"
    )
    ax_uefi_trap.text(
        0.0, 2.3, wrong_box, fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFEBEE", edgecolor="#EF5350", linewidth=1.4),
        color="#B71C1C"
    )

    # Correct case
    ax_uefi_trap.text(
        5.5, 3.3,
        "[OK] 【正解】uintptr_t または uint8_t* へのキャストによるバイト単位加算",
        fontsize=9.8, fontweight="bold", color="#2E7D32"
    )
    correct_box = (
        "// MikanOS Day02b 実装パターン (uintptr_t 整数加算)\n"
        "auto desc = reinterpret_cast<EFI_MEMORY_DESCRIPTOR*>(\n"
        "    reinterpret_cast<uintptr_t>(buffer) + iter\n"
        ");\n"
        "// またはバイトポインタ加算: (reinterpret_cast<uint8_t*>(desc) + descriptor_size)\n"
        "// -> 正確に 48 バイト (1 ディスクリプタ分) だけ次のエントリへ進む！"
    )
    ax_uefi_trap.text(
        5.5, 2.2, correct_box, fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#66BB6A", linewidth=1.4),
        color="#1B5E20"
    )

    summary_note = (
        "[Key Point] 教訓: C/C++ のポインタ算術は常に要素サイズ単位でスケーリングされる。\n"
        "バイト単位の任意オフセットを進める必要がある OS 低レイヤ処理では、必ず uintptr_t または uint8_t* / char* にキャストする。"
    )
    ax_uefi_trap.text(
        0.0, 0.4, summary_note, fontsize=8.8, fontweight="bold", color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E3F2FD", edgecolor="#2196F3", linewidth=1.2)
    )

    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.96, hspace=0.35)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path






# =============================================================================
# Chapter 2.9: Struct Memory Layout, Arrow Operator, and Double Pointers
# =============================================================================

@dataclass
class StructMember:
    """Represents a member field within a struct with offset and alignment math."""
    name: str
    type_name: str
    size: int
    alignment: int
    offset: int = 0
    padding_before: int = 0


class StructMemoryLayout:
    """Simulates struct memory layout, alignment, padding, and arrow operator dereferencing."""

    TYPE_SIZES_AND_ALIGN = {
        "char": (1, 1),
        "uint8_t": (1, 1),
        "int8_t": (1, 1),
        "bool": (1, 1),
        "short": (2, 2),
        "uint16_t": (2, 2),
        "int16_t": (2, 2),
        "CHAR16": (2, 2),
        "int": (4, 4),
        "uint32_t": (4, 4),
        "int32_t": (4, 4),
        "UINT32": (4, 4),
        "float": (4, 4),
        "long long": (8, 8),
        "uint64_t": (8, 8),
        "int64_t": (8, 8),
        "UINT64": (8, 8),
        "UINTN": (8, 8),
        "uintptr_t": (8, 8),
        "void*": (8, 8),
        "pointer": (8, 8),
        "double": (8, 8),
    }

    def __init__(self, struct_name: str, members_def: list[tuple[str, str]]):
        """
        Initializes struct layout from list of (member_name, type_name).
        Calculates offsets, alignment padding, and total struct size.
        """
        self.struct_name = struct_name
        self.members: list[StructMember] = []
        self.total_size = 0
        self.struct_alignment = 1
        self._calculate_layout(members_def)

    def _calculate_layout(self, members_def: list[tuple[str, str]]) -> None:
        current_offset = 0
        max_align = 1

        for name, type_name in members_def:
            if type_name.endswith("*"):
                size, align = 8, 8
            else:
                size, align = self.TYPE_SIZES_AND_ALIGN.get(type_name, (4, 4))

            max_align = max(max_align, align)

            # Padding before this member to satisfy alignment
            padding = 0
            if current_offset % align != 0:
                padding = align - (current_offset % align)
                current_offset += padding

            member = StructMember(
                name=name,
                type_name=type_name,
                size=size,
                alignment=align,
                offset=current_offset,
                padding_before=padding,
            )
            self.members.append(member)
            current_offset += size

        # Structure tail padding to align to max_align
        tail_padding = 0
        if current_offset % max_align != 0:
            tail_padding = max_align - (current_offset % max_align)
            current_offset += tail_padding

        self.struct_alignment = max_align
        self.total_size = current_offset

    def get_member(self, name: str) -> StructMember:
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(f"Member '{name}' not found in struct '{self.struct_name}'")

    def simulate_arrow_operator(self, base_address: int, member_name: str) -> tuple[int, int, str]:
        """
        Simulates p->member access.
        Returns (effective_address, offset, assembly_instruction).
        """
        member = self.get_member(member_name)
        effective_address = base_address + member.offset
        asm = f"MOV RAX, [RDI + {member.offset}]  ; RDI holds base_address 0x{base_address:X}"
        return effective_address, member.offset, asm

    def get_summary_dict(self) -> dict[str, Any]:
        return {
            "struct_name": self.struct_name,
            "total_size": self.total_size,
            "struct_alignment": self.struct_alignment,
            "members": [
                {
                    "name": m.name,
                    "type": m.type_name,
                    "size": m.size,
                    "alignment": m.alignment,
                    "offset": m.offset,
                    "padding_before": m.padding_before,
                }
                for m in self.members
            ],
        }


class DoublePointerSimulator:
    """
    Simulates pointer-to-pointer semantics, memory layout, stack frames,
    and UEFI output parameter patterns (e.g. OpenRootDir / AllocatePool).
    Uses a sparse virtual memory model supporting 64-bit address spaces.
    """

    def __init__(self, stack_base: int = 0x7FFF_0020, heap_base: int = 0x1000_5000):
        self.stack_base = stack_base
        self.heap_base = heap_base
        self._memory_bytes: dict[int, int] = {}  # Sparse byte-level memory
        self.next_stack_addr = stack_base
        self.next_heap_addr = heap_base
        self.entities: dict[str, dict[str, Any]] = {}

    def write_bytes(self, addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self._memory_bytes[addr + i] = b

    def read_bytes(self, addr: int, size: int) -> bytes:
        return bytes(self._memory_bytes.get(addr + i, 0) for i in range(size))

    def write_uint64(self, addr: int, val: int) -> None:
        self.write_bytes(addr, struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))

    def read_uint64(self, addr: int) -> int:
        data = self.read_bytes(addr, 8)
        return struct.unpack("<Q", data)[0]

    def allocate_target_object(self, name: str, size: int, initial_data: bytes = b"\x00") -> int:
        """Allocates an object on heap (e.g., EFI_FILE_PROTOCOL instance)."""
        addr = self.next_heap_addr
        self.next_heap_addr += ((size + 15) // 16) * 16  # 16-byte aligned

        data = initial_data.ljust(size, b"\x00")
        self.write_bytes(addr, data)
        self.entities[name] = {
            "name": name,
            "category": "object",
            "address": addr,
            "size": size,
            "type": "struct",
        }
        return addr

    def allocate_local_pointer(self, name: str, target_address: int = 0) -> int:
        """
        Allocates a local pointer variable on stack (e.g., EFI_FILE_PROTOCOL* root_dir = nullptr).
        Address on stack is 8-byte aligned.
        """
        addr = self.next_stack_addr
        self.next_stack_addr += 8
        self.write_uint64(addr, target_address)
        self.entities[name] = {
            "name": name,
            "category": "pointer",
            "address": addr,
            "size": 8,
            "target_address": target_address,
            "type": "T*",
        }
        return addr

    def allocate_double_pointer(self, name: str, target_pointer_name: str) -> int:
        """
        Allocates a pointer-to-pointer variable (e.g., EFI_FILE_PROTOCOL** new_handle = &root_dir).
        Holds the address of the pointer variable on stack.
        """
        if target_pointer_name not in self.entities:
            raise KeyError(f"Target pointer '{target_pointer_name}' does not exist")
        ptr_entry = self.entities[target_pointer_name]
        target_ptr_addr = ptr_entry["address"]

        addr = self.next_stack_addr
        self.next_stack_addr += 8
        self.write_uint64(addr, target_ptr_addr)
        self.entities[name] = {
            "name": name,
            "category": "double_pointer",
            "address": addr,
            "size": 8,
            "target_address": target_ptr_addr,
            "target_pointer_name": target_pointer_name,
            "type": "T**",
        }
        return addr

    def dereference_pointer(self, pointer_name: str) -> int:
        """Returns the target address stored in the pointer (*p)."""
        ptr = self.entities[pointer_name]
        return self.read_uint64(ptr["address"])

    def dereference_double_pointer_level1(self, double_pointer_name: str) -> int:
        """Dereferences once: (*pp) -> address of the target pointer."""
        pp = self.entities[double_pointer_name]
        target_ptr_addr = self.read_uint64(pp["address"])
        return target_ptr_addr

    def dereference_double_pointer_level2(self, double_pointer_name: str) -> int:
        """Dereferences twice: (**pp) -> value pointed to by the underlying pointer."""
        target_ptr_addr = self.dereference_double_pointer_level1(double_pointer_name)
        target_obj_addr = self.read_uint64(target_ptr_addr)
        return target_obj_addr

    def simulate_pass_by_value_failure(self, caller_ptr_name: str, allocated_obj_addr: int) -> dict[str, Any]:
        """
        Simulates what happens if a caller passes a pointer by value to an initialization function:
            void FailAlloc(EFI_FILE_PROTOCOL* p) { p = &new_object; }
        The callee updates its local copy, leaving caller_ptr unchanged!
        """
        caller_ptr = self.entities[caller_ptr_name]
        initial_val = self.read_uint64(caller_ptr["address"])

        # Callee receives a copy of the pointer on its own stack
        callee_copy_addr = self.next_stack_addr
        self.next_stack_addr += 8
        self.write_uint64(callee_copy_addr, initial_val)

        # Callee overwrites its local copy
        self.write_uint64(callee_copy_addr, allocated_obj_addr)

        # Value in caller remains initial_val!
        caller_val_after = self.read_uint64(caller_ptr["address"])

        return {
            "success": False,
            "caller_ptr_address": caller_ptr["address"],
            "caller_ptr_value_before": initial_val,
            "caller_ptr_value_after": caller_val_after,
            "callee_copy_address": callee_copy_addr,
            "callee_copy_value": allocated_obj_addr,
            "is_caller_updated": (caller_val_after == allocated_obj_addr),
            "explanation": "Callee modified its local copy on stack; caller's pointer remains unchanged (0x0 nullptr)",
        }

    def simulate_pass_by_double_pointer_success(
        self,
        caller_ptr_name: str,
        allocated_obj_addr: int,
    ) -> dict[str, Any]:
        """
        Simulates pass-by-double-pointer (UEFI protocol output pattern):
            EFI_STATUS OpenRootDir(EFI_SIMPLE_FILE_SYSTEM_PROTOCOL* This, EFI_FILE_PROTOCOL** Root) {
                *Root = &new_root_protocol;
                return EFI_SUCCESS;
            }
        """
        caller_ptr = self.entities[caller_ptr_name]
        initial_val = self.read_uint64(caller_ptr["address"])

        # Callee receives the address of caller's pointer: &root_dir
        callee_arg_pp = caller_ptr["address"]

        # Callee writes to *Root:
        # *Root = allocated_obj_addr
        self.write_uint64(callee_arg_pp, allocated_obj_addr)
        caller_val_after = self.read_uint64(caller_ptr["address"])

        return {
            "success": True,
            "caller_ptr_address": caller_ptr["address"],
            "caller_ptr_value_before": initial_val,
            "caller_ptr_value_after": caller_val_after,
            "callee_arg_pp": callee_arg_pp,
            "is_caller_updated": (caller_val_after == allocated_obj_addr),
            "explanation": "Callee dereferenced *pp and wrote allocated_obj_addr directly into caller's stack variable",
        }


# =============================================================================
# Plotting Functions for Chapter 2.9 (High-Resolution DPI 300)
# =============================================================================

def plot_figure_2_17_struct_layout_and_arrow_operator(output_path: str = "2/result/fig2_17_struct_memory_layout_and_arrow_operator.png") -> str:
    """
    Generates Figure 2.17: Struct Memory Layout, Member Offsets, Alignment Padding,
    and Arrow Operator (->) vs (*p).member Equivalence.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_plot_style()

    fig = plt.figure(figsize=(15, 12), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "図 2.17: 構造体のメモリレイアウト・アライメント数理とアロー演算子 (->) の機械語メカニズム",
        fontsize=15,
        fontweight="bold",
        color="#0D47A1",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.1], hspace=0.38)

    # -------------------------------------------------------------------------
    # Panel 1: Struct Memory Layout, Offsets, and Padding Math
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(
        "【1】構造体のメモリ配置数理: メンバのアライメント要件とパディング（埋め草）の発生原理",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlim(-0.5, 24.5)
    ax1.set_ylim(-0.8, 3.2)
    ax1.axis("off")

    members = [
        ("uint32_t a", 0, 4, "#1976D2", "0x00: a (4B)"),
        ("uint8_t b", 4, 1, "#388E3C", "0x04: b (1B)"),
        ("padding", 5, 1, "#BDBDBD", "0x05: [pad 1B]"),
        ("uint16_t c", 6, 2, "#F57C00", "0x06: c (2B)"),
        ("uint64_t d", 8, 8, "#7B1FA2", "0x08: d (8B, 64-bit int / pointer)"),
    ]

    for byte_idx in range(16):
        ax1.text(byte_idx + 0.5, 2.5, f"+{byte_idx}B", ha="center", va="center", fontsize=7.5, color="#555555")

    for name, offset, size, color, label in members:
        rect = plt.Rectangle(
            (offset, 0.8), size, 1.4,
            facecolor=color, edgecolor="#212121", linewidth=1.5, alpha=0.88
        )
        ax1.add_patch(rect)
        ax1.text(
            offset + size / 2.0, 1.5, label,
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#FFFFFF" if name != "padding" else "#212121"
        )
        ax1.annotate(
            f"Offset {offset}",
            xy=(offset, 0.8), xytext=(offset, 0.2),
            ha="center", fontsize=7.5, fontweight="bold", color=color if name != "padding" else "#757575",
            arrowprops=dict(arrowstyle="->", color=color if name != "padding" else "#757575", lw=1.2)
        )

    rule_text = (
        "【アライメント数理】\n"
        "• 各メンバのアドレスは自身のサイズ（またはアライメント要件）の倍数でなければならない:\n"
        "    Address(member) % Alignment(member) == 0\n"
        "• uint16_t c は 2の倍数番地に配置される必要があるため、b (1B) の後ろに 1バイトのパディングが自動挿入される。\n"
        "• uint64_t d は 8の倍数番地 (offset 8) に配置されるため、パディングなしで直結する。\n"
        "• 構造体全体のサイズは、最大メンバアライメント (8B) の倍数 (16B) に切り上げられる。"
    )
    ax1.text(
        16.8, 1.5, rule_text, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor="#9E9E9E", lw=1.0)
    )

    # -------------------------------------------------------------------------
    # Panel 2: Arrow Operator (->) vs (*p).member Equivalence & Assembly
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title(
        "【2】アロー演算子 (->) と (*p).member の完全等価性と x86-64 レジスタオフセット解決",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlim(-0.5, 24.5)
    ax2.set_ylim(-0.5, 3.2)
    ax2.axis("off")

    box_syntax = (
        "【C/C++ 構文の等価性 (Syntax Sugar)】\n\n"
        "  SampleStruct obj;\n"
        "  SampleStruct* p = &obj;\n\n"
        "  (*p).d = 42;    // [1] 間接参照 (*) してからドット (.) アクセス\n"
        "  p->d   = 42;    // [2] アロー演算子 (->) によるアクセス\n\n"
        "  ★ [1] と [2] はコンパイラにとって完全に同一！\n"
        "  ※ なぜアロー演算子があるのか？\n"
        "     演算子優先順位で . (1位) が * (2位) より高いため、\n"
        "     *p.d と書くと *(p.d) と解釈されてコンパイルエラーになる！\n"
        "     毎回 (*p).d と括弧を書く煩わしさを解消したのが p->d。"
    )
    ax2.text(
        0.0, 1.4, box_syntax, fontsize=8.2, va="center",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#E8EAF6", edgecolor="#3F51B5", lw=1.3)
    )

    box_asm = (
        "【x86-64 機械語・アセンブリレベルの動作】\n\n"
        "  // 引数 p が RDI レジスタで渡された場合:\n"
        "  MOV RAX, [RDI + 8]   ; RDI (ベースアドレス) に offset 8 を加算し、\n"
        "                       ; そのメモリ番地から 8 バイトを RAX にロード！\n\n"
        "  数式モデル:\n"
        "    Address(p->member) = (uintptr_t)p + offsetof(Struct, member)\n"
        "    Value              = *((MemberType*)(Address(p->member)))\n\n"
        "  ★ CPU には「アロー演算子」という命令は存在せず、\n"
        "     すべて『ベースレジスタ + 即値オフセット』のアドレス修飾命令に変換される。"
    )
    ax2.text(
        12.5, 1.4, box_asm, fontsize=8.2, va="center",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#E0F2F1", edgecolor="#00897B", lw=1.3)
    )

    # -------------------------------------------------------------------------
    # Panel 3: MikanOS Chapter 2 Use Cases (MemoryMap, gBS, SystemTable)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title(
        "【3】MikanOS 第2章におけるアロー演算子の実用例（gBS, gST, root_dir, struct MemoryMap）",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax3.set_xlim(-0.5, 24.5)
    ax3.set_ylim(-0.5, 3.2)
    ax3.axis("off")

    case1 = (
        "【ケース A: UEFI サービス呼び出し】\n"
        "gBS->GetMemoryMap(...) \n"
        "• gBS は EFI_BOOT_SERVICES* 構造体ポインタ\n"
        "• GetMemoryMap は関数ポインタメンバ\n"
        "• 動作: (*gBS).GetMemoryMap(...) と等価\n"
        "• 関数テーブル (vtable 類似) へのオフセット間接ジャンプ"
    )
    ax3.text(
        0.0, 1.4, case1, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF3E0", edgecolor="#FB8C00", lw=1.2)
    )

    case2 = (
        "【ケース B: プロトコル多重参照】\n"
        "SystemTable->ConOut->OutputString(ConOut, ...)\n"
        "• 1段階目: SystemTable->ConOut\n"
        "    EFI_SYSTEM_TABLE のオフセット 64B から\n"
        "    SIMPLE_TEXT_OUTPUT_INTERFACE* を取得\n"
        "• 2段階目: ->OutputString\n"
        "    そのインターフェース内の関数ポインタを呼び出し"
    )
    ax3.text(
        8.2, 1.4, case2, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FCE4EC", edgecolor="#E91E63", lw=1.2)
    )

    case3 = (
        "【ケース C: MemoryMap ドット vs アロー】\n"
        "struct MemoryMap memmap;\n"
        "memmap.buffer_size = sizeof(buffer);  // 実体はドット .\n"
        "GetMemoryMap(&memmap);               // ポインタを渡す\n\n"
        "EFI_STATUS GetMemoryMap(struct MemoryMap* map) {\n"
        "    map->buffer_size = ...;          // 関数内はアロー ->\n"
        "}"
    )
    ax3.text(
        16.4, 1.4, case3, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EDE7F6", edgecolor="#673AB7", lw=1.2)
    )

    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.04, right=0.96)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_figure_2_18_double_pointer_and_uefi_interface(output_path: str = "2/result/fig2_18_double_pointer_and_uefi_interface.png") -> str:
    """
    Generates Figure 2.18: Double Pointers (Pointer-to-Pointer T**),
    Stack vs Heap Memory Model, and UEFI Output Parameter Pattern.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_plot_style()

    fig = plt.figure(figsize=(15, 12), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "図 2.18: ポインタのポインタ (T**) のメモリ構造と UEFI 出力引数パターンの完全図解",
        fontsize=15,
        fontweight="bold",
        color="#0D47A1",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.2, 1.0], hspace=0.38)

    # -------------------------------------------------------------------------
    # Panel 1: Three Passing Modes (Pass by Value, Pointer, Double Pointer)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(
        "【1】引数渡しの3形態の比較: なぜポインタ自身を書き換えるには『二重ポインタ』が必要なのか",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlim(-0.5, 24.5)
    ax1.set_ylim(-0.5, 3.2)
    ax1.axis("off")

    mode1 = (
        "【A. 値渡し (Pass by Value)】\n"
        "void f(int x) { x = 99; }\n\n"
        "• 呼び出し元の変数: a = 10\n"
        "• スタックに a の値 10 がコピーされる\n"
        "• 関数内で x を変更しても a は 10 のまま！\n"
        "• [結果] 呼び出し元の値は不変"
    )
    ax1.text(
        0.0, 1.4, mode1, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFEBEE", edgecolor="#EF5350", lw=1.2)
    )

    mode2 = (
        "【B. ポインタ渡し (Pass by Pointer)】\n"
        "void f(int* p) { *p = 99; }\n\n"
        "• 呼び出し元の変数: a = 10\n"
        "• スタックに a のアドレス &a が渡される\n"
        "• *p = 99 により、a の中身が 99 に書き換わる！\n"
        "• [結果] 指している先の値 (*p) を更新可能"
    )
    ax1.text(
        8.2, 1.4, mode2, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8E1", edgecolor="#FFB300", lw=1.2)
    )

    mode3 = (
        "【C. 二重ポインタ渡し (Pass by Pointer to Pointer)】\n"
        "void f(int** pp) { *pp = &new_var; }\n\n"
        "• 呼び出し元のポインタ変数: int* ptr = nullptr\n"
        "• スタックに ptr 自身のアドレス &ptr が渡される\n"
        "• *pp = &new_var により、ptr の中身 (指す先のアドレス) が書き換わる！\n"
        "• [結果] ポインタ変数そのもの (ptr) を更新可能"
    )
    ax1.text(
        16.4, 1.4, mode3, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#4CAF50", lw=1.2)
    )

    # -------------------------------------------------------------------------
    # Panel 2: Stack & Heap Layout for UEFI OpenRootDir Pattern
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title(
        "【2】UEFI OpenRootDir における二重ポインタ (&root_dir) のメモリチェーン完全トレース",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlim(-0.5, 24.5)
    ax2.set_ylim(-0.8, 3.4)
    ax2.axis("off")

    rect_ptr = plt.Rectangle((1.0, 0.8), 5.5, 1.6, facecolor="#E3F2FD", edgecolor="#1565C0", lw=1.5)
    ax2.add_patch(rect_ptr)
    ax2.text(3.75, 2.1, "呼び出し元 (UefiMain) スタック", ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")
    ax2.text(3.75, 1.6, "EFI_FILE_PROTOCOL* root_dir", ha="center", fontsize=8.0, fontweight="bold")
    ax2.text(3.75, 1.1, "番地: 0x7FFF0020\n値: 0x10005000 (更新後)", ha="center", fontsize=7.8, color="#1B5E20")

    rect_arg = plt.Rectangle((9.0, 0.8), 5.5, 1.6, facecolor="#F3E5F5", edgecolor="#7B1FA2", lw=1.5)
    ax2.add_patch(rect_arg)
    ax2.text(11.75, 2.1, "関数 (OpenRootDir) 引数レジスタ", ha="center", fontsize=8.5, fontweight="bold", color="#4A148C")
    ax2.text(11.75, 1.6, "EFI_FILE_PROTOCOL** Root (RDX)", ha="center", fontsize=8.0, fontweight="bold")
    ax2.text(11.75, 1.1, "格納値: 0x7FFF0020 (&root_dir)", ha="center", fontsize=7.8, color="#311B92")

    rect_obj = plt.Rectangle((17.0, 0.8), 6.5, 1.6, facecolor="#E8F5E9", edgecolor="#2E7D32", lw=1.5)
    ax2.add_patch(rect_obj)
    ax2.text(20.25, 2.1, "UEFI ファームウェア内部 / ヒープ", ha="center", fontsize=8.5, fontweight="bold", color="#1B5E20")
    ax2.text(20.25, 1.6, "EFI_FILE_PROTOCOL 実体", ha="center", fontsize=8.0, fontweight="bold")
    ax2.text(20.25, 1.1, "番地: 0x10005000\nメンバ: Open, Read, Write...", ha="center", fontsize=7.8, color="#1B5E20")

    ax2.annotate(
        "&root_dir を渡す",
        xy=(9.0, 1.6), xytext=(6.5, 1.6),
        ha="center", fontsize=8.0, fontweight="bold", color="#7B1FA2",
        arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.5)
    )

    ax2.annotate(
        "*Root = 0x10005000\n(アドレスを書き戻す)",
        xy=(6.5, 1.2), xytext=(17.0, 1.2),
        ha="center", fontsize=8.0, fontweight="bold", color="#C2185B",
        arrowprops=dict(arrowstyle="->", color="#C2185B", lw=1.8, connectionstyle="arc3,rad=-0.25")
    )

    ax2.annotate(
        "生成オブジェクトのアドレス",
        xy=(17.0, 1.8), xytext=(14.5, 1.8),
        ha="center", fontsize=7.5, color="#2E7D32",
        arrowprops=dict(arrowstyle="<-", color="#2E7D32", lw=1.2)
    )

    # -------------------------------------------------------------------------
    # Panel 3: Why C Needs Double Pointers vs C++ References
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title(
        "【3】C言語 (UEFI ABI) と C++ の対比: なぜ EFI_FILE_PROTOCOL*& ではなく EFI_FILE_PROTOCOL** なのか",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax3.set_xlim(-0.5, 24.5)
    ax3.set_ylim(-0.5, 3.2)
    ax3.axis("off")

    c_box = (
        "【UEFI / C言語 ABI (C89/C99)】\n"
        "• C言語には「参照 (Reference)」機能が存在しない。\n"
        "• 関数の戻り値はステータスコード EFI_STATUS (64ビット整数) に固定されている。\n"
        "• 生成されたプロトコルやバッファのアドレスを返すには、\n"
        "  「呼び出し元のポインタのアドレス (&ptr: 二重ポインタ)」を出力引数として受け取るしかない。\n"
        "• UEFI 仕様書における汎用インターフェース: VOID** Interface"
    )
    ax3.text(
        0.0, 1.4, c_box, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFFDE7", edgecolor="#FBC02D", lw=1.2)
    )

    cpp_box = (
        "【現代 C++ との比較】\n"
        "• C++ では参照渡し void Open(EFI_FILE_PROTOCOL*& root) が可能だが、\n"
        "  コンパイルされたバイナリ (機械語) の実態は『アドレス渡し (二重ポインタ)』と全く同一！\n"
        "• C++ の std::unique_ptr や戻り値タプル std::tuple<Status, Ptr> は\n"
        "  C言語 ABI 境界 (UEFI ファームウェアとブートローダ間) では使えない。\n"
        "• したがって、低レイヤ OS 開発では二重ポインタの完全な理解が不可欠となる。"
    )
    ax3.text(
        12.5, 1.4, cpp_box, fontsize=8.0, va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EDE7F6", edgecolor="#512DA8", lw=1.2)
    )

    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.04, right=0.96)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# =============================================================================
# Chapter 2 Exercises: Comprehensive Verification & Quiz Engine
# =============================================================================

class Chapter2Quiz:
    """Comprehensive multiple-choice quiz engine covering Sections 2.1 to 2.9."""

    def __init__(self) -> None:
        self.questions: list[QuizQuestion] = [
            QuizQuestion(
                question_id=1,
                section="2.1",
                topic="EDK II パッケージ設定ファイル (.dec, .dsc, .inf)",
                question="EDK II パッケージにおける .dec, .inf, .dsc の3つの設定ファイルの役割の組み合わせとして正しいものはどれか？",
                options=[
                    ".dec: バイナリ実行ファイル, .inf: パッケージ全体ビルド構成, .dsc: 公開インターフェース宣言",
                    ".dec: 公開プロトコル/GUID等の宣言 (Declaration), .inf: 単一モジュールのソース・依存関係定義 (Information), .dsc: ビルド対象パッケージ全体の構成・ライブラリ解決 (Description)",
                    ".dec: リンカスクリプト, .inf: Makefile生成ルール, .dsc: ドキュメント定義",
                    ".dec: 単一コンポーネント定義, .inf: パッケージ全体記述, .dsc: 公開ヘッダ宣言",
                ],
                correct_index=1,
                explanation="EDK IIでは、.dec (Package Declaration) が公開プロトコルやGUID・ヘッダを宣言し、.inf (Module Information) が各モジュールのソースコードや依存ライブラリクラスを定義し、.dsc (Platform Description) が実際にビルドするモジュール群と各ライブラリクラスの実体インスタンスのマッピングを統合管理します。",
            ),
            QuizQuestion(
                question_id=2,
                section="2.2",
                topic="UefiMain シグネチャとプリプロセッサ",
                question="EDK IIブートローダのエントリポイント UefiMain のシグネチャと、プリプロセッサ #include の動作として正しいものはどれか？",
                options=[
                    "int main(int argc, char** argv) であり、#include は実行時に共有ライブラリを動的ロードする",
                    "EFI_STATUS EFIAPI UefiMain(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) であり、#include は指定されたヘッダファイルの内容をコンパイル前にその位置へテキスト展開する",
                    "void UefiMain(void) であり、#include はバイナリオブジェクトをリンクする",
                    "EFI_STATUS UefiMain(EFI_BOOT_SERVICES *BS) であり、#include は仮想関数テーブルを自動生成する",
                ],
                correct_index=1,
                explanation="EDK IIアプリケーションのエントリポイントは UEFI呼び出し規約修飾子 EFIAPI を持つ EFI_STATUS EFIAPI UefiMain(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) です。またC言語のプリプロセッサ #include は、指定ヘッダファイルをその場にそのままテキスト展開する単純な文字列置換機構です。",
            ),
            QuizQuestion(
                question_id=3,
                section="2.3",
                topic="メインメモリ物理アドレス空間とMMIO",
                question="現代の x86-64 アーキテクチャにおける物理アドレス空間と MMIO（Memory-Mapped I/O）の特徴として正しいものはどれか？",
                options=[
                    "物理アドレス空間の全領域には必ず物理RAM（DRAM）が連続して隙間なく敷き詰められている",
                    "物理アドレス空間はRAMだけでなく、デバイス制御用レジスタ（MMIO）やPCIホール、ファームウェアROM領域などが割り当てられており、アドレス空間はスパース（非連続）である",
                    "MMIOレジスタへのアクセスにはIN/OUT命令のみが使用可能で、MOV等のメモリアクセス命令は使用できない",
                    "32ビット物理アドレス空間では最大16TiBのRAMを直接マッピングできる",
                ],
                correct_index=1,
                explanation="物理アドレス空間は物理RAMだけでなく、GPUやxHCI等のハードウェア制御用レジスタ（MMIO）やPCIデバイス用予約空間（PCIホール）、ROM領域が混在して配置されており、物理RAMは飛び飛び（スパース）に配置されます。MMIO領域はMOV等の通常メモリアクセス命令で直接制御できます。",
            ),
            QuizQuestion(
                question_id=4,
                section="2.4",
                topic="EFI_MEMORY_DESCRIPTOR 構造体とアライメント",
                question="UEFI規格における EFI_MEMORY_DESCRIPTOR 構造体（自然アライメント適用時）のサイズとフィールド構成として正しいものはどれか？",
                options=[
                    "サイズは 32バイト（Type: 4B, PhysicalStart: 8B, VirtualStart: 8B, NumberOfPages: 8B, Attribute: 4B）",
                    "サイズは 40バイト（Type: 4B + Pad: 4B, PhysicalStart: 8B, VirtualStart: 8B, NumberOfPages: 8B, Attribute: 8B）",
                    "サイズは 48バイト（Type: 8B, PhysicalStart: 8B, VirtualStart: 8B, NumberOfPages: 8B, Attribute: 8B, Reserved: 8B）",
                    "サイズは 24バイト（Type: 4B, PhysicalStart: 4B, VirtualStart: 4B, NumberOfPages: 4B, Attribute: 8B）",
                ],
                correct_index=1,
                explanation="64ビット環境において EFI_MEMORY_DESCRIPTOR は Type (UINT32: 4バイト) の直後に 8バイトアライメントのための 4バイトパディングが挿入され、続く PhysicalStart (8B), VirtualStart (8B), NumberOfPages (8B), Attribute (8B) と合わせて合計 40バイト（0x28バイト）となります。",
            ),
            QuizQuestion(
                question_id=5,
                section="2.5",
                topic="GetMemoryMap() の引数仕様と MapKey",
                question="gBS->GetMemoryMap() API の呼び出し規則および戻り値・MapKey の役割として正しいものはどれか？",
                options=[
                    "バッファサイズが不足している場合は EFI_NOT_FOUND が返り、バッファサイズの再取得はできない",
                    "バッファサイズが不足している場合は EFI_BUFFER_TOO_SMALL が返り、必要なバイト数が MemoryMapSize に書き戻される。また取得した MapKey は ExitBootServices() に渡してメモリ状態の整合性検証に用いられる",
                    "MapKey は暗号化キーであり、メモリマップの内容を復号するために使用する",
                    "GetMemoryMap() は常に静的に確保された固定アドレスのメモリマップを返し、引数でバッファを渡す必要はない",
                ],
                correct_index=1,
                explanation="GetMemoryMap() は提供バッファが不足すると EFI_BUFFER_TOO_SMALL を返し、MemoryMapSize 引数に必要サイズを書き戻します。MapKey はメモリマップ取得時点のファームウェアメモリ状態のスナップショット識別子であり、ExitBootServices() 呼び出し時にメモリマップが変化していないことを検証するために必須です。",
            ),
            QuizQuestion(
                question_id=6,
                section="2.5",
                topic="メモリディスクリプタのポインタ走査と DescriptorSize",
                question="メモリマップのバッファ内にある各ディスクリプタを反復走査する際、(EFI_MEMORY_DESCRIPTOR*)p + 1 ではなく (uintptr_t)p += DescriptorSize のように走査すべき理由はどれか？",
                options=[
                    "C言語のポインタ加算はバイト単位で計算されるため、型キャストを省略するため",
                    "UEFI仕様書において DescriptorSize は将来の拡張のため sizeof(EFI_MEMORY_DESCRIPTOR) より大きい値（例: 48バイト等）が返される可能性があり、固定サイズポインタ加算ではアライメントずれ・不正アクセスが生じるため",
                    "DescriptorSize は各ディスクリプタごとに異なる値をとる可変長フィールドだから",
                    "CPUが64ビットポインタのインクリメント命令をサポートしていないため",
                ],
                correct_index=1,
                explanation="UEFI仕様では、ファームウェア実装や将来の規格拡張によって各ディスクリプタに追加フィールドが付与されることを想定し、GetMemoryMap() が実際のディスクリプタサイズ（DescriptorSize）を返します。そのため、sizeof(EFI_MEMORY_DESCRIPTOR) に依存したポインタ加算ではなく、バイトポインタに DescriptorSize を加算して次の要素へ進める必要があります。",
            ),
            QuizQuestion(
                question_id=7,
                section="2.6",
                topic="ファイル保存と属性マスク (EFI_FILE_PROTOCOL)",
                question="メモリマップを FAT ボリュームの memmap ファイルへ CSV 形式で保存する際、Attribute フィールドの出力で Attribute & 0xffffflu のようにビットマスクをかける理由として最も適切なものはどれか？",
                options=[
                    "属性値が負数になるのを防ぐため",
                    "UEFI仕様で定義されているメモリキャッシュ・アクセス属性フラグは下位ビット群（UC, WC, WT, WB, XP, RO, RUNTIME 等）に集約されており、フォーマット文字列 %08lx で下位5桁（または主要フラグ）を簡潔・明瞭に出力するため",
                    "上位ビットに格納されている暗号化キーをマスクして隠蔽するため",
                    "ファイルシステムの最大ファイルサイズ制限に引っかからないようにするため",
                ],
                correct_index=1,
                explanation="EFI_MEMORY_DESCRIPTOR の Attribute は 64ビット整数ですが、一般的なキャッシュ属性（UC: 0x1, WC: 0x2, WT: 0x4, WB: 0x8）や実行保護（XP: 0x8000...）、ランタイム属性（RUNTIME: 0x8000...）など、主要な基本属性ビットを簡潔に CSV に 16進数出力するために & 0xffffflu のマスクが書籍のサンプルコードで適用されています。",
            ),
            QuizQuestion(
                question_id=8,
                section="2.7",
                topic="ExitBootServices() 後の再利用可能メモリ種別",
                question="OS カーネルがブートサービスを終了（ExitBootServices 完了）した後に、OS が自由な物理メモリ（ヒープやページフレーム）として再利用できるメモリ種別の組み合わせはどれか？",
                options=[
                    "EfiReservedMemoryType, EfiMemoryMappedIO, EfiMemoryMappedIOPortSpace",
                    "EfiRuntimeServicesCode, EfiRuntimeServicesData, EfiACPIReclaimMemory",
                    "EfiConventionalMemory（元の空きRAM）に加え、EfiBootServicesCode および EfiBootServicesData",
                    "EfiUnusableMemory, EfiPalCode, EfiACPIMemoryNVS",
                ],
                correct_index=2,
                explanation="ExitBootServices() が完了すると UEFI ブートローダやブートサービス機能は不要となるため、ブートサービス用に割り当てられていた EfiBootServicesCode と EfiBootServicesData は解放され、通常の空きRAMである EfiConventionalMemory と同様に OS カーネルが完全に再利用可能となります（RuntimeServices 領域や ACPI 領域は解放してはいけません）。",
            ),
            QuizQuestion(
                question_id=9,
                section="2.8",
                topic="64-bit ポインタのサイズとポインタ算術スケーリング",
                question="64ビット（x86-64）環境におけるポインタ変数およびポインタ演算の性質として正しいものはどれか？",
                options=[
                    "char* のサイズは 1バイト、int* のサイズは 4バイト、void* のサイズは 8バイトである",
                    "指す先の型が何であれ、すべてのポインタ変数のサイズは一貫して 8バイト（64ビット）であり、T* 型のポインタ ptr に対する ptr + 1 はアドレス値として sizeof(T) バイト進む",
                    "ptr + 1 は常にアドレス値が 1バイトだけインクリメントされる",
                    "ポインタ変数にはメモリのアドレスではなく、変数の値そのもののコピーが格納されている",
                ],
                correct_index=1,
                explanation="64ビットアーキテクチャでは、アドレス空間全体（0〜2^64-1）を指し示すためにすべてのポインタ変数は一律 8バイト（64ビット）のサイズを持ちます。またポインタ算術において ptr + n は、メモリアドレス値として n × sizeof(T) バイト加算されます。",
            ),
            QuizQuestion(
                question_id=10,
                section="2.9",
                topic="アロー演算子と二重ポインタ (void** 出力引数)",
                question="構造体ポインタに対するアロー演算子 -> と、UEFI API で頻出する二重ポインタ（void** や EFI_FILE_PROTOCOL**）の役割として正しいものはどれか？",
                options=[
                    "p->member は (&p).member と等価であり、二重ポインタは配列の二次元アクセス専用である",
                    "p->member は (*p).member と完全等価であり、呼び出し先関数が生成・確保したポインタ変数のアドレス（指す先）を呼び出し元のポインタ変数に書き戻すために、呼び出し元ポインタのアドレス (&ptr) を二重ポインタとして渡す",
                    "アロー演算子はC++専用の演算子であり、C言語では (*p).member しか使えない",
                    "二重ポインタを渡すと、関数の引数は値渡し（Pass by value）となり呼び出し元のポインタは変更されない",
                ],
                correct_index=1,
                explanation="p->member は (*p).member（デリファレンスしてメンバ参照）の糖衣構文（完全等価）です。また C言語には参照渡し（Reference）がないため、関数内部で呼び出し元のポインタ変数自体を書き換えて新しいアドレスを返す（出力引数とする）には、ポインタのアドレスを二重ポインタ（T**）として渡す必要があります。",
            ),
        ]

    def evaluate(self, answers: dict[int, int]) -> dict[str, any]:
        """Evaluate user answers and return comprehensive scores, explanations, and mastery breakdown."""
        details = []
        correct_count = 0
        section_scores: dict[str, dict[str, int]] = {
            "2.1": {"correct": 0, "total": 0},
            "2.2": {"correct": 0, "total": 0},
            "2.3": {"correct": 0, "total": 0},
            "2.4": {"correct": 0, "total": 0},
            "2.5": {"correct": 0, "total": 0},
            "2.6": {"correct": 0, "total": 0},
            "2.7": {"correct": 0, "total": 0},
            "2.8": {"correct": 0, "total": 0},
            "2.9": {"correct": 0, "total": 0},
        }

        for q in self.questions:
            user_ans = answers.get(q.question_id, -1)
            is_correct = (user_ans == q.correct_index)
            if is_correct:
                correct_count += 1
                section_scores[q.section]["correct"] += 1
            section_scores[q.section]["total"] += 1

            details.append({
                "question_id": q.question_id,
                "section": q.section,
                "topic": q.topic,
                "question": q.question,
                "user_answer": user_ans,
                "correct_answer": q.correct_index,
                "is_correct": is_correct,
                "explanation": q.explanation,
            })

        total = len(self.questions)
        percentage = (correct_count / total) * 100.0 if total > 0 else 0.0

        mastery_rates: dict[str, float] = {}
        for sec, stat in section_scores.items():
            rate = (stat["correct"] / stat["total"]) * 100.0 if stat["total"] > 0 else 0.0
            mastery_rates[sec] = rate

        return {
            "total_questions": total,
            "correct_count": correct_count,
            "percentage": percentage,
            "details": details,
            "mastery_rates": mastery_rates,
            "passed": percentage >= 80.0,
        }


class Chapter2TheoryValidator:
    """Validation helper for Chapter 2 systems theory, EDK II architecture, and memory math."""

    @staticmethod
    def validate_edk2_files(file_roles: dict[str, str]) -> tuple[bool, str]:
        """Validate EDK II package configuration files and their roles."""
        dec_role = file_roles.get(".dec", "").lower()
        inf_role = file_roles.get(".inf", "").lower()
        dsc_role = file_roles.get(".dsc", "").lower()

        errors = []
        if not ("declaration" in dec_role or "宣言" in dec_role or "プロトコル" in dec_role or "guid" in dec_role):
            errors.append(".dec: 公開プロトコル/GUID宣言 (Declaration) であるべきです")
        if not ("module" in inf_role or "information" in inf_role or "モジュール" in inf_role or "情報" in inf_role):
            errors.append(".inf: モジュール情報・ビルドソース定義 (Module Information) であるべきです")
        if not ("description" in dsc_role or "パッケージ" in dsc_role or "全体" in dsc_role or "記述" in dsc_role):
            errors.append(".dsc: パッケージ全体ビルド構成・ライブラリ解決 (Platform Description) であるべきです")

        if not errors:
            return True, "正解！EDK IIの3大設定ファイル（.dec, .inf, .dsc）の役割を正確に同定しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_memory_descriptor_layout(layout: dict[str, int]) -> tuple[bool, str]:
        """Validate 64-bit EFI_MEMORY_DESCRIPTOR field offsets and sizes."""
        struct_size = layout.get("struct_size")
        type_size = layout.get("type_size")
        pad_size = layout.get("pad_size")
        phys_start_size = layout.get("physical_start_size")
        num_pages_size = layout.get("number_of_pages_size")
        attr_size = layout.get("attribute_size")

        errors = []
        if struct_size != 40:
            errors.append(f"struct_size: 期待値 40バイト (0x28), 入力値 {struct_size}")
        if type_size != 4:
            errors.append(f"type_size: 期待値 4バイト (UINT32), 入力値 {type_size}")
        if pad_size != 4:
            errors.append(f"pad_size: 期待値 4バイト (アライメントパディング), 入力値 {pad_size}")
        if phys_start_size != 8:
            errors.append(f"physical_start_size: 期待値 8バイト (EFI_PHYSICAL_ADDRESS), 入力値 {phys_start_size}")
        if num_pages_size != 8:
            errors.append(f"number_of_pages_size: 期待値 8バイト (UINT64), 入力値 {num_pages_size}")
        if attr_size != 8:
            errors.append(f"attribute_size: 期待値 8バイト (UINT64), 入力値 {attr_size}")

        if not errors:
            return True, "正解！EFI_MEMORY_DESCRIPTOR 構造体の40バイトレイアウトおよび自然アライメントパディングを正確に定義しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_exit_boot_services_reclaimable(types: list[str]) -> tuple[bool, str]:
        """Validate memory types that become usable RAM after ExitBootServices()."""
        type_set = {t.strip() for t in types}
        expected_reclaimable = {"EfiConventionalMemory", "EfiBootServicesCode", "EfiBootServicesData"}
        if type_set == expected_reclaimable:
            return True, "正解！ExitBootServices完了後にOSが自由に使用できる3大メモリ種別（Conventional, BootServicesCode, BootServicesData）を完璧に特定しました。"
        missing = expected_reclaimable - type_set
        extra = type_set - expected_reclaimable
        msg = []
        if missing:
            msg.append(f"不足している種別: {missing}")
        if extra:
            msg.append(f"再利用不可（解放禁止）が含まれています: {extra}")
        return False, f"不正解です。{'; '.join(msg)}"

    @staticmethod
    def validate_pointer_arithmetic(math_dict: dict[str, any]) -> tuple[bool, str]:
        """Validate 64-bit pointer sizes and pointer arithmetic scaling."""
        ptr_size = math_dict.get("pointer_size_bytes")
        scaling_uint32 = math_dict.get("uint32_ptr_add_1_bytes")
        scaling_uint64 = math_dict.get("uint64_ptr_add_1_bytes")

        errors = []
        if ptr_size != 8:
            errors.append(f"pointer_size_bytes: 期待値 8 (64-bit), 入力値 {ptr_size}")
        if scaling_uint32 != 4:
            errors.append(f"uint32_ptr_add_1_bytes: 期待値 4 (sizeof(uint32_t)), 入力値 {scaling_uint32}")
        if scaling_uint64 != 8:
            errors.append(f"uint64_ptr_add_1_bytes: 期待値 8 (sizeof(uint64_t)), 入力値 {scaling_uint64}")

        if not errors:
            return True, "正解！64ビットポインタサイズ（8B）と型スケーリング加算の数理を正確に算出しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_struct_alignment(alignment_dict: dict[str, int]) -> tuple[bool, str]:
        """Validate struct natural alignment and padding computation."""
        total_size = alignment_dict.get("total_size")
        pad_after_a = alignment_dict.get("pad_after_a")
        pad_after_c = alignment_dict.get("pad_after_c")

        errors = []
        if total_size != 12:
            errors.append(f"total_size: 期待値 12バイト, 入力値 {total_size}")
        if pad_after_a != 3:
            errors.append(f"pad_after_a: 期待値 3バイト, 入力値 {pad_after_a}")
        if pad_after_c != 2:
            errors.append(f"pad_after_c: 期待値 2バイト, 入力値 {pad_after_c}")

        if not errors:
            return True, "正解！自然アライメント規則に基づくパディングと構造体総サイズ（12B）を正確に計算しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"


def plot_chapter2_exercises_summary(output_path: Path | str) -> Path:
    """Generate high-DPI summary diagram showing Chapter 2 exercise structure and learning taxonomy."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_facecolor("#FAFAFA")

    exercises = [
        {"id": "Ex 1", "sec": "2.1 / 2.2", "title": "EDK II & メインメモリ理論検証", "desc": ".dec/.inf/.dsc の3層ビルド体系、UefiMain シグネチャ、物理空間 RAM vs MMIO", "color": "#1E88E5", "y": 0.84},
        {"id": "Ex 2", "sec": "2.4 / 2.5", "title": "メモリマップ & ディスクリプタ走査", "desc": "EFI_MEMORY_DESCRIPTOR 40B構造体数理、DescriptorSize による安全走査", "color": "#43A047", "y": 0.67},
        {"id": "Ex 3", "sec": "2.6 / 2.7", "title": "ファイル保存 & 利用可能RAM集計", "desc": "CSV出力・属性マスク、ExitBootServices 後の再利用可能メモリ (Conventional + BS)", "color": "#FB8C00", "y": 0.50},
        {"id": "Ex 4", "sec": "2.8 / 2.9", "title": "ポインタ算術 & 構造体アライメント", "desc": "64-bit ポインタスケーリング、アロー演算子、二重ポインタ (void** 出力引数)", "color": "#8E24AA", "y": 0.33},
        {"id": "Ex 5", "sec": "2.1-2.9", "title": "C++20 コア検証 & 総合テスト", "desc": "可変長走査・集計・二重ポインタ C++シミュレータ、全節網羅10問クイズ & レーダー", "color": "#D81B60", "y": 0.16},
    ]

    import matplotlib.patches as patches

    for ex in exercises:
        y = ex["y"]
        rect = patches.FancyBboxPatch(
            (0.05, y - 0.06), 0.90, 0.12,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="white", edgecolor=ex["color"], linewidth=2, zorder=2
        )
        ax.add_patch(rect)

        badge = patches.FancyBboxPatch(
            (0.07, y - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=ex["color"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(0.13, y, ex["id"], color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

        ax.text(0.22, y + 0.02, f"[{ex['sec']}] {ex['title']}", color="#212121", fontsize=12, fontweight="bold", va="center")
        ax.text(0.22, y - 0.025, ex["desc"], color="#555555", fontsize=9.5, va="center")

        circle = plt.Circle((0.91, y), 0.025, facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(0.91, y, "✓", color="#2E7D32", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.05, 0.97)
    ax.axis("off")
    ax.set_title("MikanOS Study — Chapter 2 Exercises Curriculum & Assessment Map",
                 fontsize=14, fontweight="bold", pad=20, color="#1A237E")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_chapter2_mastery_radar(output_path: Path | str, mastery_rates: dict[str, float] | None = None) -> Path:
    """Generate high-DPI 9-axis radar chart showing Chapter 2 concept mastery across all 9 sections."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        "2.1 EDK II 構成\n(DEC/INF/DSC & Build)",
        "2.2 ハローワールド\n(UefiMain & Include)",
        "2.3 メインメモリ\n(RAM vs MMIO & Holes)",
        "2.4 メモリマップ\n(Descriptor & Types)",
        "2.5 マップ取得\n(GetMemoryMap & MapKey)",
        "2.6 ファイル保存\n(CSV Export & Mask)",
        "2.7 マップ確認\n(Analysis & Usable RAM)",
        "2.8 ポインタ基礎\n(64bit Address & Scaling)",
        "2.9 アロー & 二重P\n(Alignment & Double Ptr)",
    ]
    num_vars = len(categories)

    if mastery_rates is None:
        mastery_rates = {f"2.{i}": 100.0 for i in range(1, 10)}

    values = [mastery_rates.get(f"2.{i}", 100.0) for i in range(1, 10)]
    values += values[:1]  # Close loop

    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FCFDFD")

    plt.xticks(angles[:-1], categories, color="#1A237E", size=10, weight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#757575", size=8.5)
    plt.ylim(0, 105)

    ax.plot(angles, values, color="#3949AB", linewidth=2.5, linestyle="solid", label="Chapter 2 Concept Mastery")
    ax.fill(angles, values, color="#5C6BC0", alpha=0.35)

    perfect_values = [100.0] * (num_vars + 1)
    ax.plot(angles, perfect_values, color="#43A047", linewidth=1.2, linestyle="dashed", label="Target (100% Mastery)")

    plt.title("Chapter 2: EDK II & Memory Map — Comprehensive Mastery Radar",
              size=13, weight="bold", pad=25, color="#0D47A1")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 3: QEMU Monitor & x86-64 Debugging Infrastructure
# ==============================================================================

@dataclass
class CpuRegisterState:
    """Represents x86-64 CPU registers dumped by QEMU Monitor `info registers`."""
    rax: int = 0
    rbx: int = 0
    rcx: int = 0
    rdx: int = 0
    rsi: int = 0
    rdi: int = 0
    rbp: int = 0
    rsp: int = 0x7ffffff0
    r8: int = 0
    r9: int = 0
    r10: int = 0
    r11: int = 0
    r12: int = 0
    r13: int = 0
    r14: int = 0
    r15: int = 0
    rip: int = 0x100000
    rflags: int = 0x00000202  # IF (bit 9), Reserved bit 1
    cr0: int = 0x80010033     # PG, PE, WP, ET
    cr2: int = 0
    cr3: int = 0x00100000     # Page Table root (PML4) physical address
    cr4: int = 0x000006f0     # PAE, OSFXSR, OSXMMEXCPT
    cs: int = 0x0038
    ds: int = 0x0030
    es: int = 0x0030
    fs: int = 0x0030
    gs: int = 0x0030
    ss: int = 0x0030

    def format_qemu_info_registers(self) -> str:
        """Format registers into realistic QEMU Monitor `info registers` string."""
        lines = [
            f"RAX={self.rax:016x} RBX={self.rbx:016x} RCX={self.rcx:016x} RDX={self.rdx:016x}",
            f"RSI={self.rsi:016x} RDI={self.rdi:016x} RBP={self.rbp:016x} RSP={self.rsp:016x}",
            f"R8 ={self.r8:016x} R9 ={self.r9:016x} R10={self.r10:016x} R11={self.r11:016x}",
            f"R12={self.r12:016x} R13={self.r13:016x} R14={self.r14:016x} R15={self.r15:016x}",
            f"RIP={self.rip:016x} RFL={self.rflags:08x} [----{ 'I' if self.rflags & 0x200 else '-' }---]",
            f"CS={self.cs:04x} DS={self.ds:04x} ES={self.es:04x} FS={self.fs:04x} GS={self.gs:04x} SS={self.ss:04x}",
            f"CR0={self.cr0:08x} CR2={self.cr2:016x} CR3={self.cr3:016x} CR4={self.cr4:08x}",
        ]
        return "\n".join(lines)

    @classmethod
    def parse_qemu_info_registers(cls, dump_str: str) -> "CpuRegisterState":
        """Parse register values from a QEMU Monitor dump string."""
        import re
        state = cls()
        reg_pattern = re.compile(r"([A-Z0-9]+)\s*=\s*([0-9a-fA-F]+)")
        matches = reg_pattern.findall(dump_str)
        for name, val_hex in matches:
            name_lower = name.lower()
            val = int(val_hex, 16)
            if hasattr(state, name_lower):
                setattr(state, name_lower, val)
            elif name_lower == "rfl":
                state.rflags = val
        return state

    @property
    def is_paging_enabled(self) -> bool:
        """Check if Paging (PG bit 31) is enabled in CR0."""
        return bool(self.cr0 & (1 << 31))

    @property
    def is_protected_mode(self) -> bool:
        """Check if Protection Enable (PE bit 0) is set in CR0."""
        return bool(self.cr0 & 1)

    @property
    def is_interrupt_enabled(self) -> bool:
        """Check if Interrupt Flag (IF bit 9) is set in RFLAGS."""
        return bool(self.rflags & (1 << 9))


class PhysicalMemoryInspector:
    """Simulates physical RAM and QEMU monitor `xp` (examine physical memory) command."""
    def __init__(self, size_bytes: int = 16 * 1024 * 1024):
        self.size = size_bytes
        self.ram = bytearray(size_bytes)

    def write_bytes(self, physical_addr: int, data: bytes | bytearray | list[int]):
        """Write raw bytes into simulated physical memory."""
        if physical_addr + len(data) > self.size:
            raise ValueError(f"Address 0x{physical_addr + len(data):x} exceeds RAM size 0x{self.size:x}")
        self.ram[physical_addr:physical_addr + len(data)] = bytes(data)

    def read_bytes(self, physical_addr: int, count: int) -> bytes:
        """Read count bytes from simulated physical memory."""
        if physical_addr + count > self.size:
            raise ValueError(f"Read out of bounds: 0x{physical_addr + count:x} > 0x{self.size:x}")
        return bytes(self.ram[physical_addr:physical_addr + count])

    def execute_xp(self, format_spec: str, physical_addr: int) -> list[str]:
        """
        Execute QEMU `xp /nuf addr` command.
        format_spec: e.g., '/16bx', '/4gx', '/8wd', '/10i'
        - n: count (default 1)
        - u: unit (b: 1 byte, h: 2 bytes, w: 4 bytes, g: 8 bytes giant word)
        - f: format (x: hex, d: signed decimal, u: unsigned dec, c: char, i: instruction asm stub)
        """
        import re
        import struct

        spec = format_spec.lstrip("/")
        match = re.match(r"^(\d*)([bhwg])?([xducis])?$", spec)
        if not match:
            count = 1
            unit = "w"
            fmt = "x"
        else:
            cnt_str, unit_str, fmt_str = match.groups()
            count = int(cnt_str) if cnt_str else 1
            unit = unit_str if unit_str else "w"
            fmt = fmt_str if fmt_str else "x"

        unit_sizes = {"b": 1, "h": 2, "w": 4, "g": 8}
        unit_bytes = unit_sizes.get(unit, 4)

        results = []
        curr_addr = physical_addr

        # Format elements per line
        items_per_line = {1: 16, 2: 8, 4: 4, 8: 2}.get(unit_bytes, 4)

        for line_start in range(0, count, items_per_line):
            line_count = min(items_per_line, count - line_start)
            line_addr = curr_addr + line_start * unit_bytes
            tokens = [f"0x{line_addr:016x}:"]

            for i in range(line_count):
                item_addr = line_addr + i * unit_bytes
                raw = self.read_bytes(item_addr, unit_bytes)

                if unit == "b":
                    val = raw[0]
                    if fmt == "x":
                        tokens.append(f"0x{val:02x}")
                    elif fmt == "d":
                        val_s = struct.unpack("b", raw)[0]
                        tokens.append(f"{val_s}")
                    elif fmt == "c":
                        ch = chr(val) if 32 <= val <= 126 else "."
                        tokens.append(f"'{ch}'")
                    else:
                        tokens.append(f"{val}")
                elif unit == "h":
                    val = struct.unpack("<H", raw)[0]
                    if fmt == "x":
                        tokens.append(f"0x{val:04x}")
                    elif fmt == "d":
                        val_s = struct.unpack("<h", raw)[0]
                        tokens.append(f"{val_s}")
                    else:
                        tokens.append(f"{val}")
                elif unit == "w":
                    val = struct.unpack("<I", raw)[0]
                    if fmt == "x":
                        tokens.append(f"0x{val:08x}")
                    elif fmt == "d":
                        val_s = struct.unpack("<i", raw)[0]
                        tokens.append(f"{val_s}")
                    else:
                        tokens.append(f"{val}")
                elif unit == "g":
                    val = struct.unpack("<Q", raw)[0]
                    if fmt == "x":
                        tokens.append(f"0x{val:016x}")
                    elif fmt == "d":
                        val_s = struct.unpack("<q", raw)[0]
                        tokens.append(f"{val_s}")
                    else:
                        tokens.append(f"{val}")

            results.append(" ".join(tokens))

        return results


class QemuMonitorEngine:
    """Interactive QEMU monitor simulator engine for bare-metal OS debugging."""
    def __init__(self, ram_size: int = 16 * 1024 * 1024):
        self.regs = CpuRegisterState()
        self.ram = PhysicalMemoryInspector(ram_size)
        self.is_running = True

    def execute_command(self, cmd_line: str) -> str:
        """Parse and execute a QEMU monitor CLI command."""
        tokens = cmd_line.strip().split()
        if not tokens:
            return ""

        cmd = tokens[0]

        if cmd in ("info", "i") and len(tokens) >= 2:
            sub = tokens[1]
            if sub in ("registers", "r"):
                return self.regs.format_qemu_info_registers()
            elif sub == "mem":
                return f"Paging enabled: {self.regs.is_paging_enabled}, PML4 root: 0x{self.regs.cr3:016x}"
            else:
                return f"unknown sub-command: {sub}"

        elif cmd == "xp":
            if len(tokens) < 2:
                return "Usage: xp [/nuf] addr"
            if len(tokens) == 2:
                format_spec = "/1xw"
                addr = int(tokens[1], 0)
            else:
                format_spec = tokens[1]
                addr = int(tokens[2], 0)
            lines = self.ram.execute_xp(format_spec, addr)
            return "\n".join(lines)

        elif cmd in ("x",):
            # Virtual memory dump (identity mapped in early boot)
            if len(tokens) < 2:
                return "Usage: x [/nuf] addr"
            if len(tokens) == 2:
                format_spec = "/1xw"
                addr = int(tokens[1], 0)
            else:
                format_spec = tokens[1]
                addr = int(tokens[2], 0)
            # In simulated identity mapping, virtual == physical
            lines = self.ram.execute_xp(format_spec, addr)
            return "\n".join(lines)

        elif cmd in ("p", "print"):
            expr = " ".join(tokens[1:])
            try:
                val = eval(expr, {}, {"rax": self.regs.rax, "rip": self.regs.rip, "rsp": self.regs.rsp})
                return f"${hex(val)}"
            except Exception as e:
                return f"Evaluation error: {e}"

        elif cmd in ("c", "cont"):
            self.is_running = True
            return "Execution continued."

        elif cmd in ("stop",):
            self.is_running = False
            return "CPU stopped."

        elif cmd in ("q", "quit"):
            return "QEMU exiting."

        elif cmd in ("help", "h"):
            return (
                "QEMU Monitor commands:\n"
                "  info registers  - Show CPU general/segment/control registers\n"
                "  xp /nuf addr    - Examine physical memory at addr\n"
                "  x /nuf addr     - Examine virtual memory at addr\n"
                "  print <expr>    - Evaluate expression\n"
                "  c               - Resume guest execution\n"
                "  stop            - Stop guest execution\n"
                "  quit            - Quit QEMU"
            )

        return f"Unknown command: {cmd}"


def plot_qemu_monitor_architecture(output_path: Path | str) -> Path:
    """Generate high-DPI architecture diagram of QEMU host process, Guest VM, and Monitor control plane."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#FAFCFF")

    import matplotlib.patches as patches

    # Outer QEMU Host Process
    host_box = patches.FancyBboxPatch(
        (0.04, 0.05), 0.92, 0.88,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#F0F4F8", edgecolor="#1565C0", linewidth=2.5, zorder=1
    )
    ax.add_patch(host_box)
    ax.text(0.06, 0.90, "QEMU Hypervisor / Emulator Process (Host OS Userspace)",
            fontsize=13, fontweight="bold", color="#0D47A1", va="center")

    # Guest VM Execution Domain (Left)
    guest_box = patches.FancyBboxPatch(
        (0.07, 0.10), 0.48, 0.74,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#FFFFFF", edgecolor="#2E7D32", linewidth=2, zorder=2
    )
    ax.add_patch(guest_box)
    ax.text(0.09, 0.80, "Guest VM Domain (MikanOS / Bare Metal)",
            fontsize=12, fontweight="bold", color="#1B5E20", va="center")

    # Guest components
    comp_defs = [
        {"y": 0.64, "h": 0.12, "title": "Virtual CPU (x86-64 Core)", "desc": "RAX..R15, RIP, RSP, RFLAGS\nControl Registers CR0/CR3/CR4, TSS", "color": "#E8F5E9", "border": "#4CAF50"},
        {"y": 0.44, "h": 0.16, "title": "Simulated Physical Memory (RAM)", "desc": "UEFI Bootloader (0x06000000)\nKernel Image (0x00100000)\nStack & Page Tables (PML4..PT)", "color": "#E1F5FE", "border": "#03A9F4"},
        {"y": 0.24, "h": 0.15, "title": "Virtual Peripherals & MMIO", "desc": "GOP VRAM FrameBuffer (0xE0000000)\nLocal APIC (0xFEE00000)\nxHCI USB Controller (BAR0 MMIO)", "color": "#FFF3E0", "border": "#FF9800"},
        {"y": 0.12, "h": 0.08, "title": "Guest Console Display (Ctrl+Alt+1)", "desc": "UEFI GOP Pixel output / MikanOS Desktop", "color": "#F3E5F5", "border": "#AB47BC"},
    ]

    for c in comp_defs:
        b = patches.FancyBboxPatch(
            (0.09, c["y"]), 0.44, c["h"],
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor=c["color"], edgecolor=c["border"], linewidth=1.5, zorder=3
        )
        ax.add_patch(b)
        ax.text(0.11, c["y"] + c["h"] - 0.03, c["title"], fontsize=10.5, fontweight="bold", color="#212121")
        lines = c["desc"].split("\n")
        for idx, line in enumerate(lines):
            ax.text(0.11, c["y"] + c["h"] - 0.065 - idx * 0.03, line, fontsize=9, color="#555555")

    # QEMU Monitor & Debugging Plane (Right)
    mon_box = patches.FancyBboxPatch(
        (0.60, 0.10), 0.34, 0.74,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#FFFFFF", edgecolor="#C2185B", linewidth=2, zorder=2
    )
    ax.add_patch(mon_box)
    ax.text(0.62, 0.80, "QEMU Monitor Interface (Ctrl+Alt+2)",
            fontsize=12, fontweight="bold", color="#880E4F", va="center")

    mon_cmds = [
        {"y": 0.63, "cmd": "info registers", "desc": "Inspect registers without intrusive probes\nReads vCPU structure in host memory", "color": "#FCE4EC"},
        {"y": 0.46, "cmd": "xp /nuf <phys_addr>", "desc": "Directly dumps physical RAM bytes\nBypasses guest page tables & MMU", "color": "#EDE7F6"},
        {"y": 0.32, "cmd": "x /nuf <virt_addr>", "desc": "Dumps virtual memory via CR3 paging\nTranslates 48-bit canonical addresses", "color": "#E0F2F1"},
        {"y": 0.18, "cmd": "screendump <file.ppm>", "desc": "Saves raw GOP framebuffer to disk\nVerify graphics without physical display", "color": "#FFF8E1"},
    ]

    for m in mon_cmds:
        b = patches.FancyBboxPatch(
            (0.62, m["y"]), 0.30, 0.12,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=m["color"], edgecolor="#7E57C2", linewidth=1.2, zorder=3
        )
        ax.add_patch(b)
        ax.text(0.635, m["y"] + 0.085, m["cmd"], fontsize=10.5, fontweight="bold", family="monospace", color="#311B92")
        for idx, line in enumerate(m["desc"].split("\n")):
            ax.text(0.635, m["y"] + 0.05 - idx * 0.028, line, fontsize=8.5, color="#424242")

    # Arrows connecting Monitor to Guest Components
    arrow1 = patches.FancyArrowPatch((0.62, 0.69), (0.53, 0.69),
                                      arrowstyle="<->", mutation_scale=15, color="#E91E63", linewidth=2, zorder=4)
    ax.add_patch(arrow1)
    ax.text(0.575, 0.71, "vCPU State", fontsize=8.5, fontweight="bold", color="#AD1457", ha="center")

    arrow2 = patches.FancyArrowPatch((0.62, 0.52), (0.53, 0.52),
                                      arrowstyle="<->", mutation_scale=15, color="#3F51B5", linewidth=2, zorder=4)
    ax.add_patch(arrow2)
    ax.text(0.575, 0.54, "RAM Read", fontsize=8.5, fontweight="bold", color="#283593", ha="center")

    arrow3 = patches.FancyArrowPatch((0.62, 0.24), (0.53, 0.28),
                                      arrowstyle="<->", mutation_scale=15, color="#F57C00", linewidth=2, zorder=4)
    ax.add_patch(arrow3)
    ax.text(0.575, 0.29, "VRAM Dump", fontsize=8.5, fontweight="bold", color="#E65100", ha="center")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.98)
    ax.axis("off")
    ax.set_title("MikanOS Chapter 3.1 — QEMU Hypervisor, Guest VM & Monitor Inspection Architecture",
                 fontsize=14, fontweight="bold", pad=15, color="#0D47A1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_qemu_registers_and_memory_dump(output_path: Path | str) -> Path:
    """Generate high-DPI visual decoding of `info registers` and `xp` physical memory dump."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_reg, ax_mem) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("#FCFDFD")

    import matplotlib.patches as patches

    # Subplot 1: Register Breakdown
    ax_reg.set_facecolor("#F9FAFB")
    ax_reg.set_title("1. QEMU `info registers` Output Decoding", fontsize=12.5, fontweight="bold", color="#1565C0", pad=12)

    regs_data = [
        {"cat": "General Purpose Registers (64-bit)", "items": [
            ("RAX", "0x0000000000000000", "Syscall number / Return value"),
            ("RBX", "0x00000000067a3018", "Callee-saved base register"),
            ("RCX", "0x0000000000000002", "Counter / Syscall arg 4"),
            ("RDX", "0x0000000000000000", "Data / Syscall arg 3"),
            ("RSI", "0x00000000067b4100", "Source Index / Syscall arg 2"),
            ("RDI", "0x00000000067a2f18", "Dest Index / Syscall arg 1"),
            ("RBP", "0x0000000007fe3e20", "Base Pointer (Stack frame)"),
            ("RSP", "0x0000000007fe3dc0", "Stack Pointer (Top of stack)"),
        ]},
        {"cat": "Instruction Pointer & Flags", "items": [
            ("RIP", "0x000000000678f142", "Next instruction to execute"),
            ("RFL", "0x00000206", "Flags: [IF=1, PF=1, Reserved=1]"),
        ]},
        {"cat": "Control Registers (CPU Operating Mode)", "items": [
            ("CR0", "0x80010033", "PG=1 (Paging), PE=1 (Protected)"),
            ("CR3", "0x0000000007fa0000", "Physical address of PML4 table"),
            ("CR4", "0x000006f0", "PAE=1, OSFXSR=1 (SSE enabled)"),
        ]},
    ]

    y = 0.94
    for group in regs_data:
        ax_reg.text(0.04, y, group["cat"], fontsize=10.5, fontweight="bold", color="#0D47A1")
        y -= 0.04
        for name, val, desc in group["items"]:
            # Register card
            rcard = patches.FancyBboxPatch(
                (0.04, y - 0.045), 0.92, 0.05,
                boxstyle="round,pad=0.008,rounding_size=0.01",
                facecolor="#FFFFFF", edgecolor="#B0BEC5", linewidth=1, zorder=2
            )
            ax_reg.add_patch(rcard)
            ax_reg.text(0.07, y - 0.02, name, fontsize=10, fontweight="bold", family="monospace", color="#D81B60")
            ax_reg.text(0.22, y - 0.02, val, fontsize=9.5, family="monospace", color="#1E88E5")
            ax_reg.text(0.60, y - 0.02, desc, fontsize=8.5, color="#424242")
            y -= 0.058
        y -= 0.02

    ax_reg.set_xlim(0, 1.0)
    ax_reg.set_ylim(0, 1.0)
    ax_reg.axis("off")

    # Subplot 2: Physical Memory Dump `xp /16bx` and `xp /4gx`
    ax_mem.set_facecolor("#F9FAFB")
    ax_mem.set_title("2. QEMU `xp /nuf addr` Physical Memory Examination", fontsize=12.5, fontweight="bold", color="#2E7D32", pad=12)

    # Explanation of syntax
    syntax_box = patches.FancyBboxPatch(
        (0.04, 0.81), 0.92, 0.15,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.5, zorder=2
    )
    ax_mem.add_patch(syntax_box)
    ax_mem.text(0.07, 0.92, "Syntax: xp /<count><unit><format> <physical_address>",
                fontsize=10.5, fontweight="bold", color="#1B5E20")
    ax_mem.text(0.07, 0.87, "• count (n): Number of units to dump (e.g., 4, 16)\n• unit (u):  b(1B), h(2B), w(4B), g(8B giant word)\n• format(f): x(hex), d(signed dec), u(unsigned dec), c(char), i(asm)",
                fontsize=8.8, color="#2E7D32")

    # Example 1: xp /16bx 0x100000
    y_m = 0.74
    ax_mem.text(0.04, y_m, "Example A: xp /16bx 0x00100000 (Byte Dump)", fontsize=10.5, fontweight="bold", color="#37474F")
    y_m -= 0.035

    hex_bytes = ["48", "83", "EC", "28", "E8", "12", "00", "00", "00", "48", "83", "C4", "28", "C3", "90", "90"]
    # Draw byte boxes
    for idx, b in enumerate(hex_bytes):
        col = idx % 8
        row = idx // 8
        bx = 0.04 + col * 0.115
        by = y_m - row * 0.065 - 0.045
        box = patches.FancyBboxPatch(
            (bx, by), 0.105, 0.055,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            facecolor="#FFF9C4" if idx < 4 else "#E1F5FE", edgecolor="#FFA000" if idx < 4 else "#0288D1", linewidth=1.2, zorder=2
        )
        ax_mem.add_patch(box)
        ax_mem.text(bx + 0.052, by + 0.032, b, fontsize=10, fontweight="bold", family="monospace", color="#212121", ha="center")
        ax_mem.text(bx + 0.052, by + 0.012, f"+{idx:x}", fontsize=7.5, family="monospace", color="#757575", ha="center")

    y_m -= 0.16
    ax_mem.text(0.04, y_m, "Example B: xp /2gx 0x00100000 (64-bit Little-Endian Dump)", fontsize=10.5, fontweight="bold", color="#37474F")
    y_m -= 0.04

    # 64-bit word cards
    words_64 = [
        ("0x00100000", "0x000012e828ec8348", "sub rsp, 28h; call +12h"),
        ("0x00100008", "0x9090c328c4834800", "add rsp, 28h; ret; nop; nop"),
    ]

    for addr_str, word_val, disasm in words_64:
        wcard = patches.FancyBboxPatch(
            (0.04, y_m - 0.065), 0.92, 0.06,
            boxstyle="round,pad=0.008,rounding_size=0.01",
            facecolor="#FFFFFF", edgecolor="#3949AB", linewidth=1.2, zorder=2
        )
        ax_mem.add_patch(wcard)
        ax_mem.text(0.07, y_m - 0.025, addr_str + ":", fontsize=9.5, fontweight="bold", family="monospace", color="#283593")
        ax_mem.text(0.32, y_m - 0.025, word_val, fontsize=9.5, fontweight="bold", family="monospace", color="#C2185B")
        ax_mem.text(0.07, y_m - 0.052, f"Disassembly interpretation: {disasm}", fontsize=8.5, color="#555555")
        y_m -= 0.08

    # Note on Little Endianness
    le_box = patches.FancyBboxPatch(
        (0.04, 0.10), 0.92, 0.15,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#FBE9E7", edgecolor="#FF5722", linewidth=1.2, zorder=2
    )
    ax_mem.add_patch(le_box)
    ax_mem.text(0.07, 0.21, "⚠️ Little-Endian Byte Order Alert", fontsize=10.5, fontweight="bold", color="#D84315")
    ax_mem.text(0.07, 0.14, "Bytes [48, 83, EC, 28] in memory appear as 0x28EC8348 when read as a\n32-bit word, or 0x000012E828EC8348 as a 64-bit quadword.\nLowest addressed byte holds least significant bits (LSB).",
                fontsize=8.5, color="#BF360C")

    ax_mem.set_xlim(0, 1.0)
    ax_mem.set_ylim(0, 1.0)
    ax_mem.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path



# =============================================================================
# Chapter 3.2: x86-64 Architecture Register Simulation & Visualizations
# =============================================================================

class X86_64RegisterFile:
    """Simulates 64-bit General-Purpose Registers, sub-register aliasing,

    the 32-bit zero-extension rule, and LIFO stack operations.
    """

    GPR_64_NAMES = [
        "rax", "rbx", "rcx", "rdx",
        "rsi", "rdi", "rbp", "rsp",
        "r8",  "r9",  "r10", "r11",
        "r12", "r13", "r14", "r15",
    ]

    SUB_32_TO_64 = {
        "eax": "rax", "ebx": "rbx", "ecx": "rcx", "edx": "rdx",
        "esi": "rsi", "edi": "rdi", "ebp": "rbp", "esp": "rsp",
        "r8d": "r8",  "r9d": "r9",  "r10d": "r10", "r11d": "r11",
        "r12d": "r12", "r13d": "r13", "r14d": "r14", "r15d": "r15",
    }

    SUB_16_TO_64 = {
        "ax": "rax", "bx": "rbx", "cx": "rcx", "dx": "rdx",
        "si": "rsi", "di": "rdi", "bp": "rbp", "sp": "rsp",
        "r8w": "r8", "r9w": "r9", "r10w": "r10", "r11w": "r11",
        "r12w": "r12", "r13w": "r13", "r14w": "r14", "r15w": "r15",
    }

    SUB_8L_TO_64 = {
        "al": "rax", "bl": "rbx", "cl": "rcx", "dl": "rdx",
        "sil": "rsi", "dil": "rdi", "bpl": "rbp", "spl": "rsp",
        "r8b": "r8", "r9b": "r9", "r10b": "r10", "r11b": "r11",
        "r12b": "r12", "r13b": "r13", "r14b": "r14", "r15b": "r15",
    }

    SUB_8H_TO_64 = {
        "ah": "rax", "bh": "rbx", "ch": "rcx", "dh": "rdx",
    }

    def __init__(self, initial_rsp: int = 0x0000000007FE3DC0, initial_rip: int = 0x00100000):
        self.gprs: dict[str, int] = {name: 0 for name in self.GPR_64_NAMES}
        self.gprs["rsp"] = initial_rsp & 0xFFFFFFFFFFFFFFFF
        self.rip: int = initial_rip & 0xFFFFFFFFFFFFFFFF
        self.stack_memory: dict[int, int] = {}  # 64-bit aligned virtual stack memory

    @property
    def rsp(self) -> int:
        return self.gprs["rsp"]

    @rsp.setter
    def rsp(self, val: int) -> None:
        self.gprs["rsp"] = val & 0xFFFFFFFFFFFFFFFF

    def read_64(self, name: str) -> int:
        name = name.lower()
        if name == "rip":
            return self.rip
        if name not in self.gprs:
            raise KeyError(f"Unknown 64-bit register: {name}")
        return self.gprs[name]

    def write_64(self, name: str, value: int) -> None:
        name = name.lower()
        if name == "rip":
            self.rip = value & 0xFFFFFFFFFFFFFFFF
            return
        if name not in self.gprs:
            raise KeyError(f"Unknown 64-bit register: {name}")
        self.gprs[name] = value & 0xFFFFFFFFFFFFFFFF

    def read_32(self, name: str) -> int:
        name = name.lower()
        reg64 = self.SUB_32_TO_64.get(name) or name
        return self.read_64(reg64) & 0xFFFFFFFF

    def write_32(self, name: str, value: int) -> None:
        """Writes to a 32-bit register.

        CRITICAL x86-64 RULE: Writing a 32-bit sub-register automatically
        ZERO-EXTENDS into the full 64-bit register, setting bits 32..63 to zero!
        """
        name = name.lower()
        reg64 = self.SUB_32_TO_64.get(name)
        if not reg64:
            if name in self.gprs:
                reg64 = name
            else:
                raise KeyError(f"Unknown 32-bit register: {name}")
        self.gprs[reg64] = value & 0xFFFFFFFF

    def read_16(self, name: str) -> int:
        name = name.lower()
        reg64 = self.SUB_16_TO_64.get(name) or name
        return self.read_64(reg64) & 0xFFFF

    def write_16(self, name: str, value: int) -> None:
        """Writes to a 16-bit register.

        PRESERVES upper 48 bits unchanged.
        """
        name = name.lower()
        reg64 = self.SUB_16_TO_64.get(name)
        if not reg64:
            raise KeyError(f"Unknown 16-bit register: {name}")
        curr = self.gprs[reg64]
        self.gprs[reg64] = (curr & ~0xFFFF) | (value & 0xFFFF)

    def read_8(self, name: str) -> int:
        name = name.lower()
        if name in self.SUB_8H_TO_64:
            reg64 = self.SUB_8H_TO_64[name]
            return (self.gprs[reg64] >> 8) & 0xFF
        if name in self.SUB_8L_TO_64:
            reg64 = self.SUB_8L_TO_64[name]
            return self.gprs[reg64] & 0xFF
        raise KeyError(f"Unknown 8-bit register: {name}")

    def write_8(self, name: str, value: int) -> None:
        """Writes to an 8-bit register (low byte or high byte AH/BH/CH/DH).

        PRESERVES unaffected bits.
        """
        name = name.lower()
        if name in self.SUB_8H_TO_64:
            reg64 = self.SUB_8H_TO_64[name]
            curr = self.gprs[reg64]
            self.gprs[reg64] = (curr & ~0xFF00) | ((value & 0xFF) << 8)
            return
        if name in self.SUB_8L_TO_64:
            reg64 = self.SUB_8L_TO_64[name]
            curr = self.gprs[reg64]
            self.gprs[reg64] = (curr & ~0xFF) | (value & 0xFF)
            return
        raise KeyError(f"Unknown 8-bit register: {name}")

    def push(self, value: int) -> int:
        """Simulates x86-64 `push value`: RSP -= 8, [RSP] = value."""
        self.rsp = (self.rsp - 8) & 0xFFFFFFFFFFFFFFFF
        self.stack_memory[self.rsp] = value & 0xFFFFFFFFFFFFFFFF
        return self.rsp

    def pop(self) -> int:
        """Simulates x86-64 `pop`: value = [RSP], RSP += 8."""
        if self.rsp not in self.stack_memory:
            raise RuntimeError(f"Stack underflow at RSP = 0x{self.rsp:016x}")
        val = self.stack_memory.pop(self.rsp)
        self.rsp = (self.rsp + 8) & 0xFFFFFFFFFFFFFFFF
        return val


class RFlagsRegister:
    """Simulates x86-64 RFLAGS (64-bit) / EFLAGS (32-bit) / FLAGS (16-bit).

    Computes arithmetic status flags (CF, PF, AF, ZF, SF, OF) and manages system/control flags.
    """

    CF_MASK = 1 << 0
    RES1_MASK = 1 << 1  # Always 1
    PF_MASK = 1 << 2
    AF_MASK = 1 << 4
    ZF_MASK = 1 << 6
    SF_MASK = 1 << 7
    TF_MASK = 1 << 8
    IF_MASK = 1 << 9
    DF_MASK = 1 << 10
    OF_MASK = 1 << 11
    IOPL_MASK = 3 << 12
    NT_MASK = 1 << 14
    RF_MASK = 1 << 16
    VM_MASK = 1 << 17
    AC_MASK = 1 << 18
    VIF_MASK = 1 << 19
    VIP_MASK = 1 << 20
    ID_MASK = 1 << 21

    def __init__(self, raw: int = 0x00000202):
        self.raw = raw | self.RES1_MASK

    @property
    def cf(self) -> bool:
        return bool(self.raw & self.CF_MASK)

    @property
    def pf(self) -> bool:
        return bool(self.raw & self.PF_MASK)

    @property
    def af(self) -> bool:
        return bool(self.raw & self.AF_MASK)

    @property
    def zf(self) -> bool:
        return bool(self.raw & self.ZF_MASK)

    @property
    def sf(self) -> bool:
        return bool(self.raw & self.SF_MASK)

    @property
    def tf(self) -> bool:
        return bool(self.raw & self.TF_MASK)

    @property
    def if_flag(self) -> bool:
        return bool(self.raw & self.IF_MASK)

    @property
    def df(self) -> bool:
        return bool(self.raw & self.DF_MASK)

    @property
    def of(self) -> bool:
        return bool(self.raw & self.OF_MASK)

    @property
    def iopl(self) -> int:
        return (self.raw >> 12) & 0x3

    def _set_bit(self, mask: int, val: bool) -> None:
        if val:
            self.raw |= mask
        else:
            self.raw &= ~mask
        self.raw |= self.RES1_MASK

    def update_arithmetic_add(self, a: int, b: int, width: int = 64) -> int:
        mask = (1 << width) - 1
        sign_bit = 1 << (width - 1)
        res_full = a + b
        res = res_full & mask

        # Carry Flag: unsigned overflow
        self._set_bit(self.CF_MASK, res_full > mask)

        # Zero Flag
        self._set_bit(self.ZF_MASK, res == 0)

        # Sign Flag
        self._set_bit(self.SF_MASK, bool(res & sign_bit))

        # Overflow Flag: signed overflow
        sa = (a & sign_bit) != 0
        sb = (b & sign_bit) != 0
        sres = (res & sign_bit) != 0
        self._set_bit(self.OF_MASK, (sa == sb) and (sres != sa))

        # Parity Flag: lowest byte has even number of 1-bits
        lowest_byte = res & 0xFF
        ones = bin(lowest_byte).count("1")
        self._set_bit(self.PF_MASK, (ones % 2) == 0)

        return res

    def update_arithmetic_sub(self, a: int, b: int, width: int = 64) -> int:
        mask = (1 << width) - 1
        sign_bit = 1 << (width - 1)
        res_full = a - b
        res = res_full & mask

        # Carry Flag: unsigned borrow (a < b)
        self._set_bit(self.CF_MASK, a < b)

        # Zero Flag
        self._set_bit(self.ZF_MASK, res == 0)

        # Sign Flag
        self._set_bit(self.SF_MASK, bool(res & sign_bit))

        # Overflow Flag: signed overflow (pos - neg = neg OR neg - pos = pos)
        sa = (a & sign_bit) != 0
        sb = (b & sign_bit) != 0
        sres = (res & sign_bit) != 0
        self._set_bit(self.OF_MASK, (sa != sb) and (sres != sa))

        # Parity Flag
        lowest_byte = res & 0xFF
        ones = bin(lowest_byte).count("1")
        self._set_bit(self.PF_MASK, (ones % 2) == 0)

        return res

    def format_flags_summary(self) -> str:
        flags = []
        if self.cf: flags.append("CF")
        if self.pf: flags.append("PF")
        if self.af: flags.append("AF")
        if self.zf: flags.append("ZF")
        if self.sf: flags.append("SF")
        if self.tf: flags.append("TF")
        if self.if_flag: flags.append("IF")
        if self.df: flags.append("DF")
        if self.of: flags.append("OF")
        flags.append(f"IOPL={self.iopl}")
        return f"[{' '.join(flags)}]"


@dataclass
class ControlRegisters:
    """Models x86-64 Control Registers (CR0..CR4, CR8) and MSR IA32_EFER."""
    cr0: int = 0x80010033  # PG=1, WP=1, ET=1, PE=1
    cr2: int = 0x0         # Page Fault Linear Address
    cr3: int = 0x07FA0000  # PML4 Root Base Address
    cr4: int = 0x000006F0  # PAE=1, OSFXSR=1, OSXMMEXCPT=1
    cr8: int = 0x0         # Task Priority Register (TPR)
    ia32_efer: int = 0x00000D01  # LME=1, LMA=1, NXE=1, SCE=1

    @property
    def is_protected_mode(self) -> bool:
        return bool(self.cr0 & (1 << 0))

    @property
    def is_write_protect(self) -> bool:
        return bool(self.cr0 & (1 << 16))

    @property
    def is_paging_enabled(self) -> bool:
        return bool(self.cr0 & (1 << 31))

    @property
    def pml4_base_address(self) -> int:
        return self.cr3 & 0x000FFFFFFFFFF000

    @property
    def is_pae_enabled(self) -> bool:
        return bool(self.cr4 & (1 << 5))

    @property
    def is_osfxsr_enabled(self) -> bool:
        return bool(self.cr4 & (1 << 9))

    @property
    def is_syscall_extension_enabled(self) -> bool:
        return bool(self.ia32_efer & (1 << 0))

    @property
    def is_long_mode_active(self) -> bool:
        return bool(self.ia32_efer & (1 << 10))

    @property
    def is_no_execute_enabled(self) -> bool:
        return bool(self.ia32_efer & (1 << 11))


class CallingConventionSimulator:
    """Simulates System V AMD64 ABI function calling conventions."""

    ARG_REGISTERS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
    RETURN_REGISTERS = ["rax", "rdx"]
    CALLEE_SAVED = ["rbx", "rsp", "rbp", "r12", "r13", "r14", "r15"]
    CALLER_SAVED = ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"]

    @classmethod
    def prepare_call(cls, rf: X86_64RegisterFile, args: list[int]) -> tuple[dict[str, int], list[int]]:
        """Maps arguments to registers RDI..R9, pushes excess arguments to stack."""
        reg_args = {}
        stack_args = []
        for i, val in enumerate(args):
            if i < len(cls.ARG_REGISTERS):
                reg_name = cls.ARG_REGISTERS[i]
                rf.write_64(reg_name, val)
                reg_args[reg_name] = val
            else:
                rf.push(val)
                stack_args.append(val)
        return reg_args, stack_args


def plot_x86_64_register_hierarchy(output_path: Path) -> Path:
    """Generates visualization for x86-64 register naming hierarchies,

    sub-register bit ranges, and the 32-bit zero-extension rule.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_hier, ax_zero) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#FFFFFF")

    # 1. Left Subplot: Register Sub-access Hierarchy
    ax_hier.set_facecolor("#F9FAFB")
    ax_hier.set_title("1. x86-64 Register Sub-Access Hierarchy & Aliasing", fontsize=12.5, fontweight="bold", color="#1565C0", pad=12)

    # Example A: RAX Hierarchy
    ax_hier.text(0.04, 0.94, "A. Legacy Register Group (RAX Example)", fontsize=11, fontweight="bold", color="#283593")

    # Full 64-bit RAX box
    rax_box = patches.FancyBboxPatch(
        (0.04, 0.83), 0.92, 0.08,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#E1F5FE", edgecolor="#0288D1", linewidth=1.5, zorder=2
    )
    ax_hier.add_patch(rax_box)
    ax_hier.text(0.06, 0.86, "RAX (64-bit: Bits 63 .. 0)", fontsize=10, fontweight="bold", color="#01579B")

    # 32-bit EAX box (bits 31..0)
    eax_box = patches.FancyBboxPatch(
        (0.50, 0.72), 0.46, 0.08,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.5, zorder=2
    )
    ax_hier.add_patch(eax_box)
    ax_hier.text(0.52, 0.75, "EAX (32-bit: Bits 31 .. 0)", fontsize=9.5, fontweight="bold", color="#1B5E20")

    # 16-bit AX box (bits 15..0)
    ax_box = patches.FancyBboxPatch(
        (0.73, 0.61), 0.23, 0.08,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#FFF9C4", edgecolor="#FBC02D", linewidth=1.5, zorder=2
    )
    ax_hier.add_patch(ax_box)
    ax_hier.text(0.75, 0.64, "AX (16-bit: 15..0)", fontsize=9, fontweight="bold", color="#F57F17")

    # 8-bit AH (bits 15..8) and AL (bits 7..0)
    ah_box = patches.FancyBboxPatch(
        (0.73, 0.50), 0.11, 0.08,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        facecolor="#FFE0B2", edgecolor="#FB8C00", linewidth=1.3, zorder=2
    )
    ax_hier.add_patch(ah_box)
    ax_hier.text(0.74, 0.53, "AH (15..8)", fontsize=8.5, fontweight="bold", color="#E65100")

    al_box = patches.FancyBboxPatch(
        (0.85, 0.50), 0.11, 0.08,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        facecolor="#FFCCBC", edgecolor="#FF5722", linewidth=1.3, zorder=2
    )
    ax_hier.add_patch(al_box)
    ax_hier.text(0.86, 0.53, "AL (7..0)", fontsize=8.5, fontweight="bold", color="#BF360C")

    # Example B: R8..R15 Hierarchy
    y_r8 = 0.42
    ax_hier.text(0.04, y_r8, "B. Extended Register Group (R8 Example: No High-Byte sub-register)", fontsize=11, fontweight="bold", color="#283593")
    y_r8 -= 0.10

    r8_box = patches.FancyBboxPatch(
        (0.04, y_r8), 0.92, 0.07,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#EDE7F6", edgecolor="#673AB7", linewidth=1.5, zorder=2
    )
    ax_hier.add_patch(r8_box)
    ax_hier.text(0.06, y_r8 + 0.025, "R8 (64-bit: Bits 63 .. 0)", fontsize=10, fontweight="bold", color="#311B92")
    y_r8 -= 0.09

    r8d_box = patches.FancyBboxPatch(
        (0.50, y_r8), 0.46, 0.07,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#E0F2F1", edgecolor="#00897B", linewidth=1.5, zorder=2
    )
    ax_hier.add_patch(r8d_box)
    ax_hier.text(0.52, y_r8 + 0.025, "R8D (32-bit: Bits 31 .. 0)", fontsize=9.5, fontweight="bold", color="#004D40")
    y_r8 -= 0.09

    r8w_box = patches.FancyBboxPatch(
        (0.73, y_r8), 0.23, 0.07,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#FCE4EC", edgecolor="#D81B60", linewidth=1.5, zorder=2
    )
    ax_hier.add_patch(r8w_box)
    ax_hier.text(0.75, y_r8 + 0.025, "R8W (16-bit: 15..0)", fontsize=9, fontweight="bold", color="#880E4F")
    y_r8 -= 0.09

    r8b_box = patches.FancyBboxPatch(
        (0.85, y_r8), 0.11, 0.07,
        boxstyle="round,pad=0.006,rounding_size=0.01",
        facecolor="#F3E5F5", edgecolor="#8E24AA", linewidth=1.3, zorder=2
    )
    ax_hier.add_patch(r8b_box)
    ax_hier.text(0.86, y_r8 + 0.025, "R8B (7..0)", fontsize=8.5, fontweight="bold", color="#4A148C")

    ax_hier.set_xlim(0, 1.0)
    ax_hier.set_ylim(0, 1.0)
    ax_hier.axis("off")

    # 2. Right Subplot: 32-bit Zero-Extension Rule vs 16/8-bit Preservation
    ax_zero.set_facecolor("#F9FAFB")
    ax_zero.set_title("2. Zero-Extension Rule vs Upper-Bit Preservation", fontsize=12.5, fontweight="bold", color="#C2185B", pad=12)

    # Warning Box
    rule_box = patches.FancyBboxPatch(
        (0.04, 0.82), 0.92, 0.14,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=1.5, zorder=2
    )
    ax_zero.add_patch(rule_box)
    ax_zero.text(0.07, 0.91, "[CRITICAL] The Golden Rule of x86-64 Sub-Registers", fontsize=10.5, fontweight="bold", color="#E65100")
    ax_zero.text(0.07, 0.85, "• 32-bit Write (MOV EAX, val): Upper 32 bits are AUTOMATICALLY ZEROED (Zero-Extended)\n• 16-bit Write (MOV AX, val):  Upper 48 bits are PRESERVED (Unchanged)\n• 8-bit Write  (MOV AL, val):  Upper 56 bits are PRESERVED (Unchanged)",
                 fontsize=8.5, color="#BF360C")

    # Step-by-step Visual Demonstration
    demos = [
        ("Step 0: Initial Full 64-bit RAX", "0xFFFFFFFFFFFFFFFF", "All 64 bits are 1"),
        ("Step 1: Execute `mov eax, 0x12345678`", "0x0000000012345678", "Upper 32 bits CLEARED TO ZERO!"),
        ("Step 2: Execute `mov ax, 0xAAAA`", "0x000000001234AAAA", "Bits 16..63 remain untouched"),
        ("Step 3: Execute `mov ah, 0x55`", "0x00000000123455AA", "Only bits 8..15 modified"),
        ("Step 4: Execute `mov al, 0x33`", "0x0000000012345533", "Only bits 0..7 modified"),
    ]

    y_d = 0.72
    for title, hex_val, desc in demos:
        ax_zero.text(0.04, y_d, title, fontsize=10, fontweight="bold", color="#37474F")
        y_d -= 0.045
        dcard = patches.FancyBboxPatch(
            (0.04, y_d - 0.045), 0.92, 0.05,
            boxstyle="round,pad=0.008,rounding_size=0.01",
            facecolor="#FFFFFF", edgecolor="#5C6BC0", linewidth=1.2, zorder=2
        )
        ax_zero.add_patch(dcard)
        ax_zero.text(0.07, y_d - 0.02, hex_val, fontsize=10, fontweight="bold", family="monospace", color="#C2185B")
        ax_zero.text(0.48, y_d - 0.02, desc, fontsize=8.8, color="#2E7D32" if "Upper 32" in desc else "#424242")
        y_d -= 0.075

    ax_zero.set_xlim(0, 1.0)
    ax_zero.set_ylim(0, 1.0)
    ax_zero.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_rflags_and_control_registers(output_path: Path) -> Path:
    """Generates visualization for RFLAGS status/system flags and CR0/CR3/CR4 control registers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_rfl, ax_cr) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#FFFFFF")

    # 1. Left Subplot: RFLAGS Bit Layout and Status Flags
    ax_rfl.set_facecolor("#F9FAFB")
    ax_rfl.set_title("1. x86-64 RFLAGS Register (64-bit) Architecture", fontsize=12.5, fontweight="bold", color="#00695C", pad=12)

    # Explanation card
    rfl_intro = patches.FancyBboxPatch(
        (0.04, 0.85), 0.92, 0.11,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#E0F2F1", edgecolor="#00897B", linewidth=1.5, zorder=2
    )
    ax_rfl.add_patch(rfl_intro)
    ax_rfl.text(0.07, 0.92, "RFLAGS: Status Flags (Math), Control Flags, System Flags", fontsize=10.5, fontweight="bold", color="#004D40")
    ax_rfl.text(0.07, 0.87, "• Bit 1: Reserved = 1 (Always set in x86 architecture)\n• Status Flags: CF (bit 0), PF (bit 2), AF (bit 4), ZF (bit 6), SF (bit 7), OF (bit 11)",
                fontsize=8.8, color="#00695C")

    # Table of Key Status Flags
    flags_table = [
        ("CF (Bit 0)", "Carry Flag", "Unsigned Overflow: 1 if unsigned carry/borrow occurs"),
        ("ZF (Bit 6)", "Zero Flag", "Calculation Result: 1 if arithmetic/logic result == 0"),
        ("SF (Bit 7)", "Sign Flag", "Signed Sign: 1 if Most Significant Bit (MSB) == 1"),
        ("OF (Bit 11)", "Overflow Flag", "Signed Overflow: 1 if signed range exceeded (- / +)"),
        ("IF (Bit 9)", "Interrupt Flag", "Hardware Interrupts: 1=Enabled (STI), 0=Disabled (CLI)"),
        ("DF (Bit 10)", "Direction Flag", "String Instructions: 0=Increment (CLD), 1=Decrement (STD)"),
    ]

    y_f = 0.77
    for bit_name, name, desc in flags_table:
        fcard = patches.FancyBboxPatch(
            (0.04, y_f - 0.05), 0.92, 0.052,
            boxstyle="round,pad=0.008,rounding_size=0.01",
            facecolor="#FFFFFF", edgecolor="#B2DFDB", linewidth=1.2, zorder=2
        )
        ax_rfl.add_patch(fcard)
        ax_rfl.text(0.07, y_f - 0.023, bit_name, fontsize=9.5, fontweight="bold", family="monospace", color="#00796B")
        ax_rfl.text(0.26, y_f - 0.023, name, fontsize=9, fontweight="bold", color="#37474F")
        ax_rfl.text(0.48, y_f - 0.023, desc, fontsize=8.2, color="#555555")
        y_f -= 0.068

    # Math Example Box
    math_box = patches.FancyBboxPatch(
        (0.04, 0.08), 0.92, 0.22,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#F3E5F5", edgecolor="#AB47BC", linewidth=1.5, zorder=2
    )
    ax_rfl.add_patch(math_box)
    ax_rfl.text(0.07, 0.25, "[CALCULATION] Mathematical Flag Updates in Assembly", fontsize=10.5, fontweight="bold", color="#6A1B9A")
    ax_rfl.text(0.07, 0.17, "Case 1: `sub rax, rax` (rax = rax)\n  -> Result = 0:  ZF = 1, SF = 0, CF = 0, OF = 0\nCase 2: `sub 3, 5`\n  -> Result = -2: CF = 1 (Borrow), SF = 1 (MSB=1), ZF = 0, OF = 0\nCase 3: `add 0x7FFFFFFF, 1` (32-bit signed max + 1)\n  -> Result = 0x80000000 (-2147483648): OF = 1 (Signed Overflow!), SF = 1, CF = 0",
                fontsize=8.5, family="monospace", color="#4A148C")

    ax_rfl.set_xlim(0, 1.0)
    ax_rfl.set_ylim(0, 1.0)
    ax_rfl.axis("off")

    # 2. Right Subplot: CPU Control Registers (CR0, CR3, CR4, IA32_EFER)
    ax_cr.set_facecolor("#F9FAFB")
    ax_cr.set_title("2. CPU Control Registers (CR0, CR3, CR4) & MSR (EFER)", fontsize=12.5, fontweight="bold", color="#283593", pad=12)

    cr_data = [
        ("CR0 (Control Register 0)", [
            ("PE (Bit 0)", "Protected Mode Enable (1 = 32/64-bit protected mode active)"),
            ("WP (Bit 16)", "Write Protect (Ring 0 cannot write to read-only user pages)"),
            ("PG (Bit 31)", "Paging Enable (1 = MMU translates virtual to physical addresses)"),
        ]),
        ("CR3 (Page Directory Base Register)", [
            ("PML4 Base", "Bits 12..51: Physical address of top-level page table (4KiB aligned)"),
            ("PCD / PWT", "Bits 4/3: Page-level Cache Disable & Write-Through caching control"),
        ]),
        ("CR4 (Architecture Extensions)", [
            ("PAE (Bit 5)", "Physical Address Extension (Required for x86-64 64-bit Long Mode)"),
            ("OSFXSR (Bit 9)", "Fast FPU/SSE Save & Restore (`fxsave` / `fxrstor`) Support"),
        ]),
        ("IA32_EFER (MSR 0xC0000080: Extended Features)", [
            ("LME (Bit 8) & LMA (Bit 10)", "Long Mode Enable & Long Mode Active (64-bit execution active)"),
            ("NXE (Bit 11)", "No-Execute Enable (Page fault on instruction fetch from data page)"),
        ]),
    ]

    y_c = 0.94
    for title, bits in cr_data:
        ax_cr.text(0.04, y_c, title, fontsize=10.5, fontweight="bold", color="#1A237E")
        y_c -= 0.04
        for b_name, b_desc in bits:
            ccard = patches.FancyBboxPatch(
                (0.04, y_c - 0.045), 0.92, 0.048,
                boxstyle="round,pad=0.008,rounding_size=0.01",
                facecolor="#FFFFFF", edgecolor="#9FA8DA", linewidth=1.1, zorder=2
            )
            ax_cr.add_patch(ccard)
            ax_cr.text(0.07, y_c - 0.02, b_name, fontsize=9.2, fontweight="bold", family="monospace", color="#C2185B")
            ax_cr.text(0.35, y_c - 0.02, b_desc, fontsize=8.2, color="#37474F")
            y_c -= 0.055
        y_c -= 0.015

    ax_cr.set_xlim(0, 1.0)
    ax_cr.set_ylim(0, 1.0)
    ax_cr.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 3.3: First Kernel Loading, ExitBootServices & Red Zone Architecture
# ==============================================================================

@dataclass
class Elf64Header:
    """Represents the 64-byte ELF64 Header (Elf64_Ehdr) for x86-64 executable binaries."""
    magic: bytes = b"\x7fELF"
    elf_class: int = 2          # 2 = 64-bit architecture (ELFCLASS64)
    endianness: int = 1         # 1 = 2's complement, little endian (ELFDATA2LSB)
    version: int = 1            # 1 = Original version of ELF (EV_CURRENT)
    os_abi: int = 0             # 0 = System V AMD64 ABI (ELFOSABI_SYSV)
    abi_version: int = 0        # 0
    e_type: int = 2             # 2 = Executable file (ET_EXEC)
    e_machine: int = 0x3E       # 0x3E (62) = AMD x86-64 architecture (EM_X86_64)
    e_version: int = 1          # 1 = Current version
    e_entry: int = 0x100000     # Virtual address to which the system first transfers control (e_entry)
    e_phoff: int = 64           # Program header table file offset (bytes)
    e_shoff: int = 0            # Section header table file offset (bytes)
    e_flags: int = 0            # Processor-specific flags
    e_ehsize: int = 64          # ELF header size in bytes (always 64 for ELF64)
    e_phentsize: int = 56       # Size of one program header table entry (56 bytes)
    e_phnum: int = 1            # Number of entries in program header table
    e_shentsize: int = 64       # Size of one section header table entry (64 bytes)
    e_shnum: int = 0            # Number of entries in section header table
    e_shstrndx: int = 0         # Section header table index of section name string table

    @classmethod
    def parse(cls, binary_data: bytes) -> "Elf64Header":
        """Parse ELF64 header from the first 64 bytes of an ELF binary."""
        if len(binary_data) < 64:
            raise ValueError(f"Binary data too short for ELF64 header: {len(binary_data)} bytes (minimum 64 bytes required)")

        magic = binary_data[0:4]
        if magic != b"\x7fELF":
            raise ValueError(f"Invalid ELF magic: {magic!r}, expected b'\\x7fELF'")

        elf_class = binary_data[4]
        endianness = binary_data[5]
        version = binary_data[6]
        os_abi = binary_data[7]
        abi_version = binary_data[8]

        fields = struct.unpack("<HHIQQQIHHHHHH", binary_data[16:64])

        return cls(
            magic=magic,
            elf_class=elf_class,
            endianness=endianness,
            version=version,
            os_abi=os_abi,
            abi_version=abi_version,
            e_type=fields[0],
            e_machine=fields[1],
            e_version=fields[2],
            e_entry=fields[3],
            e_phoff=fields[4],
            e_shoff=fields[5],
            e_flags=fields[6],
            e_ehsize=fields[7],
            e_phentsize=fields[8],
            e_phnum=fields[9],
            e_shentsize=fields[10],
            e_shnum=fields[11],
            e_shstrndx=fields[12]
        )

    def to_bytes(self) -> bytes:
        """Serialize Elf64Header to standard 64-byte binary layout."""
        e_ident = bytearray(16)
        e_ident[0:4] = self.magic
        e_ident[4] = self.elf_class
        e_ident[5] = self.endianness
        e_ident[6] = self.version
        e_ident[7] = self.os_abi
        e_ident[8] = self.abi_version

        body = struct.pack(
            "<HHIQQQIHHHHHH",
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.e_phoff,
            self.e_shoff,
            self.e_flags,
            self.e_ehsize,
            self.e_phentsize,
            self.e_phnum,
            self.e_shentsize,
            self.e_shnum,
            self.e_shstrndx
        )
        return bytes(e_ident) + body

    @staticmethod
    def get_entry_point_offset() -> int:
        """Return the exact byte offset of e_entry in ELF64 header: 24 (0x18)."""
        return 24

    def verify_entry_dereference(self, base_addr: int = 0x100000) -> dict[str, Any]:
        """Explain the exact C statement: UINT64 entry_addr = *(UINT64*)(kernel_base_addr + 24);"""
        entry_offset = self.get_entry_point_offset()
        deref_addr = base_addr + entry_offset
        return {
            "kernel_base_addr": hex(base_addr),
            "offset_bytes": entry_offset,
            "dereference_target_address": hex(deref_addr),
            "resolved_entry_point": hex(self.e_entry),
            "explanation": (
                f"In 64-bit ELF (Elf64_Ehdr), e_ident occupies 16 bytes (offsets 0-15), "
                f"e_type is 2 bytes (offsets 16-17), e_machine is 2 bytes (offsets 18-19), "
                f"and e_version is 4 bytes (offsets 20-23). Therefore, e_entry begins exactly "
                f"at offset 16 + 2 + 2 + 4 = 24 (0x18) as a 64-bit unsigned integer (UINT64). "
                f"Reading *(UINT64*)(kernel_base_addr + 24) retrieves {hex(self.e_entry)}."
            )
        }


class KernelLoaderStep(Enum):
    INIT = "INIT"
    OPEN_ROOT_DIR = "OPEN_ROOT_DIR"
    OPEN_KERNEL_FILE = "OPEN_KERNEL_FILE"
    GET_KERNEL_INFO = "GET_KERNEL_INFO"
    ALLOCATE_PAGES = "ALLOCATE_PAGES"
    READ_KERNEL_DATA = "READ_KERNEL_DATA"
    EXIT_BOOT_SERVICES = "EXIT_BOOT_SERVICES"
    JUMP_TO_ENTRY = "JUMP_TO_ENTRY"
    KERNEL_HALT_LOOP = "KERNEL_HALT_LOOP"


class KernelLoaderSimulator:
    """Simulates the complete MikanOS UEFI bootloader sequence for loading, preparing, and jumping to kernel.elf."""

    def __init__(self, target_base_addr: int = 0x100000) -> None:
        self.target_base_addr = target_base_addr
        self.current_step = KernelLoaderStep.INIT
        self.kernel_file_size: int = 0
        self.allocated_num_pages: int = 0
        self.allocated_base_addr: int = 0
        self.memory_buffer: bytearray = bytearray()
        self.map_key: int = 42
        self.boot_services_active: bool = True
        self.kernel_executed: bool = False
        self.execution_log: list[str] = []

    def open_kernel(self, filename: str = "\\kernel.elf", fake_binary_size: int = 4096) -> bool:
        """Simulate root_dir->Open(..., L'\\kernel.elf', EFI_FILE_MODE_READ, 0)."""
        self.current_step = KernelLoaderStep.OPEN_KERNEL_FILE
        self.kernel_file_size = fake_binary_size
        self.execution_log.append(f"Opened file '{filename}' via EFI_SIMPLE_FILE_SYSTEM_PROTOCOL.")
        return True

    def get_kernel_info(self) -> int:
        """Simulate kernel_file->GetInfo(&gEfiFileInfoGuid, ...)."""
        self.current_step = KernelLoaderStep.GET_KERNEL_INFO
        self.execution_log.append(f"Queried file metadata: kernel_file_size = {self.kernel_file_size} bytes.")
        return self.kernel_file_size

    def allocate_pages(self) -> tuple[int, int]:
        """Simulate gBS->AllocatePages(AllocateAddress, EfiLoaderData, num_pages, &kernel_base_addr)."""
        self.current_step = KernelLoaderStep.ALLOCATE_PAGES
        # Page calculation: (kernel_file_size + 0xfff) / 0x1000
        self.allocated_num_pages = (self.kernel_file_size + 0xFFF) // 0x1000
        self.allocated_base_addr = self.target_base_addr
        # Memory map key changes due to dynamic memory allocation
        self.map_key += 7
        self.execution_log.append(
            f"Allocated {self.allocated_num_pages} pages ({self.allocated_num_pages * 4096} bytes) "
            f"at physical address {hex(self.allocated_base_addr)} (MapKey updated to {self.map_key})."
        )
        return self.allocated_base_addr, self.allocated_num_pages

    def read_kernel_data(self, elf_binary: bytes | None = None) -> bytes:
        """Simulate kernel_file->Read(kernel_file, &kernel_file_size, (VOID*)kernel_base_addr)."""
        self.current_step = KernelLoaderStep.READ_KERNEL_DATA
        if elf_binary is None:
            # Generate minimal valid ELF64 header with entry point = 0x100000
            hdr = Elf64Header(e_entry=self.target_base_addr)
            elf_binary = hdr.to_bytes() + b"\x90" * (self.kernel_file_size - 64)
        
        self.memory_buffer = bytearray(elf_binary)
        self.execution_log.append(
            f"Read {len(self.memory_buffer)} bytes into buffer at {hex(self.allocated_base_addr)}."
        )
        return bytes(self.memory_buffer)

    def exit_boot_services(self, passed_map_key: int, retry_if_mismatch: bool = True) -> bool:
        """Simulate gBS->ExitBootServices(image_handle, memmap.map_key)."""
        self.current_step = KernelLoaderStep.EXIT_BOOT_SERVICES
        if passed_map_key != self.map_key:
            self.execution_log.append(
                f"ExitBootServices failed: MapKey mismatch (passed={passed_map_key}, current={self.map_key})."
            )
            if retry_if_mismatch:
                # Re-fetch memory map to get updated map key
                passed_map_key = self.map_key
                self.execution_log.append(f"Re-fetched memory map: renewed MapKey = {passed_map_key}.")
            else:
                return False

        self.boot_services_active = False
        self.execution_log.append(
            f"ExitBootServices SUCCEEDED with MapKey {passed_map_key}. UEFI Boot Services terminated. OS Kernel is now in full control."
        )
        return True

    def call_kernel_entry(self) -> dict[str, Any]:
        """Simulate UINT64 entry_addr = *(UINT64*)(kernel_base_addr + 24); entry_point();"""
        if self.boot_services_active:
            raise RuntimeError("Cannot execute kernel while UEFI Boot Services are still active! Call ExitBootServices first.")
        
        if len(self.memory_buffer) < 32:
            raise RuntimeError("Kernel buffer is empty or too small to extract ELF entry point.")

        entry_addr = struct.unpack("<Q", self.memory_buffer[24:32])[0]
        self.current_step = KernelLoaderStep.JUMP_TO_ENTRY
        self.kernel_executed = True
        self.current_step = KernelLoaderStep.KERNEL_HALT_LOOP
        self.execution_log.append(
            f"Jumped to kernel entry point at {hex(entry_addr)}. Entered KernelMain() infinite HLT loop."
        )

        return {
            "entry_point": hex(entry_addr),
            "executed": True,
            "final_cpu_state": "HLT (Halted / Interrupt Wait)",
            "log": list(self.execution_log)
        }


@dataclass
class StackFrameEntry:
    """Represents a slot on the x86-64 stack."""
    address: int
    offset_from_rsp: int
    name: str
    value: int
    is_corrupted: bool = False


class RedZoneSimulator:
    """Simulates the System V AMD64 ABI 128-byte Red Zone behavior and stack corruption upon asynchronous interrupts.

    In the System V AMD64 ABI:
    - The 128-byte area beyond the current RSP (%rsp - 1 to %rsp - 128) is designated as the 'Red Zone'.
    - Leaf functions or functions without downstream calls can utilize this zone without modifying RSP.
    - However, if a hardware interrupt or CPU exception occurs in kernel mode, the CPU hardware
      immediately pushes SS, RSP, RFLAGS, CS, RIP, (and error code) onto the current RSP,
      clobbering any data stored in the Red Zone!
    """

    def __init__(self, initial_rsp: int = 0x7FFFFFFF0000) -> None:
        self.initial_rsp = initial_rsp
        self.rsp = initial_rsp
        self.memory: dict[int, int] = {}
        self.stack_records: list[StackFrameEntry] = []
        self.mode_red_zone_enabled: bool = True  # True: compiler uses red zone; False: -mno-red-zone

    def configure_compiler_mode(self, use_red_zone: bool) -> None:
        """Switch between standard ABI (use_red_zone=True) and kernel ABI (use_red_zone=False, -mno-red-zone)."""
        self.mode_red_zone_enabled = use_red_zone

    def allocate_leaf_function_vars(self, vars_map: dict[str, int]) -> list[StackFrameEntry]:
        """Allocate local variables for a leaf function.

        - If red zone is enabled: writes directly to RSP - 8, RSP - 16, ..., WITHOUT adjusting RSP (sub rsp, N omitted).
        - If red zone is disabled (-mno-red-zone): explicitly executes `sub rsp, N`, shifting RSP below variables.
        """
        num_vars = len(vars_map)
        frame_size = ((num_vars * 8 + 15) // 16) * 16  # 16-byte alignment

        allocated = []
        if self.mode_red_zone_enabled:
            # RED ZONE: RSP remains unchanged! Data written into [RSP - 8, RSP - 128]
            offset = -8
            for name, val in vars_map.items():
                target_addr = self.rsp + offset
                self.memory[target_addr] = val
                entry = StackFrameEntry(target_addr, offset, name, val, is_corrupted=False)
                self.stack_records.append(entry)
                allocated.append(entry)
                offset -= 8
        else:
            # NO RED ZONE (-mno-red-zone): RSP is decremented first!
            self.rsp -= frame_size
            offset = 0
            for name, val in vars_map.items():
                target_addr = self.rsp + offset
                self.memory[target_addr] = val
                entry = StackFrameEntry(target_addr, offset, name, val, is_corrupted=False)
                self.stack_records.append(entry)
                allocated.append(entry)
                offset += 8

        return allocated

    def trigger_hardware_interrupt(self, vector: int = 32) -> list[str]:
        """Simulate CPU hardware behavior upon receiving an interrupt in Long Mode (x86-64).

        The CPU automatically pushes:
        1. SS (Stack Segment, 8 bytes)
        2. RSP (Old Stack Pointer, 8 bytes)
        3. RFLAGS (Flags register, 8 bytes)
        4. CS (Code Segment, 8 bytes)
        5. RIP (Instruction Pointer, 8 bytes)
        Total 40 bytes pushed directly below the CURRENT RSP!
        """
        interrupt_frame = [
            ("Interrupt_SS", 0x10),
            ("Interrupt_RSP", self.rsp),
            ("Interrupt_RFLAGS", 0x202),
            ("Interrupt_CS", 0x08),
            ("Interrupt_RIP", 0x100050),
        ]

        log = []
        push_addr = self.rsp
        for name, val in interrupt_frame:
            push_addr -= 8
            # Check if this address overwrites any existing stack entry
            for record in self.stack_records:
                if record.address == push_addr:
                    record.is_corrupted = True
                    log.append(
                        f"CRITICAL CORRUPTION: Variable '{record.name}' at {hex(push_addr)} "
                        f"(original value {hex(record.value)}) was OVERWRITTEN by CPU hardware pushing {name} ({hex(val)})!"
                    )
            self.memory[push_addr] = val

        self.rsp = push_addr
        return log

    # Alias for trigger_hardware_interrupt
    trigger_cpu_interrupt = trigger_hardware_interrupt

    def count_corrupted_variables(self) -> int:
        """Return the number of local variables destroyed by the interrupt."""
        return sum(1 for rec in self.stack_records if rec.is_corrupted)


def plot_kernel_loading_and_entry_flow(output_path: str = "3/result/fig3_05_kernel_loading_and_entry_flow.png") -> str:
    """Render high-resolution diagram of the MikanOS UEFI kernel loading and entry transition flow."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_facecolor("#FAFAFC")

    fig.suptitle(
        "MikanOS Bootloader to Kernel Transition Architecture (Chapter 3.3 / day03a)",
        fontsize=16,
        fontweight="bold",
        color="#1A237E",
        y=0.97
    )

    # 4 Major Sequential Phases
    phases = [
        {
            "step": "Phase 1: Kernel ELF Discovery & Sizing",
            "actor": "UEFI Bootloader (Main.c)",
            "api": "root_dir->Open(..., L\"\\\\kernel.elf\", ...)\nkernel_file->GetInfo(&gEfiFileInfoGuid, ...)",
            "desc": "Access FAT filesystem using EFI_SIMPLE_FILE_SYSTEM_PROTOCOL.\nRetrieve exact kernel binary size (kernel_file_size) in bytes.",
            "color": "#E3F2FD",
            "border": "#1976D2",
            "icon": "[DISK / FAT32]"
        },
        {
            "step": "Phase 2: 1MiB Physical Page Allocation & Read",
            "actor": "UEFI Boot Services (Memory Allocation)",
            "api": "gBS->AllocatePages(AllocateAddress, EfiLoaderData,\n    (size + 0xfff) / 0x1000, &kernel_base_addr);\nkernel_file->Read(..., (VOID*)kernel_base_addr);",
            "desc": "Fix base address at 0x100000 (1MiB, legacy real-mode ceiling).\nAllocate contiguous 4KiB pages and copy raw ELF binary into RAM.",
            "color": "#E8F5E9",
            "border": "#388E3C",
            "icon": "[RAM / 0x100000]"
        },
        {
            "step": "Phase 3: ExitBootServices Sovereignty Handoff",
            "actor": "UEFI Firmware -> OS Kernel Transition",
            "api": "status = gBS->ExitBootServices(image_handle, memmap.map_key);\nif (EFI_ERROR(status)) {\n    GetMemoryMap(&memmap); // Retry with updated MapKey\n    gBS->ExitBootServices(image_handle, memmap.map_key);\n}",
            "desc": "Terminate all UEFI Boot Services (no more Print, dynamic alloc).\nKernel acquires bare-metal CPU sovereignty. MapKey retry is mandatory.",
            "color": "#FFF3E0",
            "border": "#F57C00",
            "icon": "[CPU SOVEREIGNTY]"
        },
        {
            "step": "Phase 4: ELF64 Entry Dereference & Far Jump",
            "actor": "Kernel Launch (C++ KernelMain)",
            "api": "UINT64 entry_addr = *(UINT64*)(kernel_base_addr + 24);\ntypedef void EntryPointType(void);\n((EntryPointType*)entry_addr)();\n--> KernelMain(): while(1) __asm__(\"hlt\");",
            "desc": "Read e_entry from ELF64 header (offset 0x18 = 24 bytes).\nCast to function pointer and execute CALL. Enter infinite HLT loop.",
            "color": "#F3E5F5",
            "border": "#7B1FA2",
            "icon": "[KERNEL RUNTIME]"
        }
    ]

    card_width = 0.21
    gap = 0.035
    start_x = 0.04
    card_y = 0.45
    card_height = 0.44

    for i, p in enumerate(phases):
        x = start_x + i * (card_width + gap)

        card = patches.FancyBboxPatch(
            (x, card_y), card_width, card_height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=p["color"], edgecolor=p["border"], linewidth=2.0, zorder=2
        )
        ax.add_patch(card)

        # Header tag
        ax.text(x + 0.015, card_y + card_height - 0.035, p["icon"], fontsize=9.5, fontweight="bold", color=p["border"])
        ax.text(x + 0.015, card_y + card_height - 0.075, p["step"], fontsize=10.5, fontweight="bold", color="#212121")
        ax.text(x + 0.015, card_y + card_height - 0.105, f"Actor: {p['actor']}", fontsize=8.5, fontstyle="italic", color="#555555")

        # Divider
        ax.plot([x + 0.01, x + card_width - 0.01], [card_y + card_height - 0.12, card_y + card_height - 0.12], color=p["border"], alpha=0.4, linewidth=1.2)

        # Code API Box
        code_box = patches.FancyBboxPatch(
            (x + 0.01, card_y + 0.14), card_width - 0.02, 0.16,
            boxstyle="round,pad=0.008,rounding_size=0.01",
            facecolor="#FFFFFF", edgecolor="#B0BEC5", linewidth=1.0, zorder=3
        )
        ax.add_patch(code_box)
        ax.text(x + 0.015, card_y + 0.27, p["api"], fontsize=7.6, family="monospace", color="#1A237E", verticalalignment="top")

        # Description text
        ax.text(x + 0.012, card_y + 0.07, p["desc"], fontsize=8.2, color="#37474F", verticalalignment="top")

        # Arrow to next phase
        if i < len(phases) - 1:
            arrow_x = x + card_width + 0.005
            ax.annotate(
                "", xy=(arrow_x + gap - 0.01, card_y + card_height / 2),
                xytext=(arrow_x, card_y + card_height / 2),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#455A64", lw=2.5),
                zorder=4
            )

    # Bottom Callout: Detail on ELF64 Header Offset 24
    elf_detail_box = patches.FancyBboxPatch(
        (0.04, 0.08), 0.92, 0.30,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#ECEFF1", edgecolor="#607D8B", linewidth=1.8, zorder=2
    )
    ax.add_patch(elf_detail_box)

    ax.text(0.06, 0.33, "[DEEP-DIVE ARCHITECTURE] Why *(UINT64*)(kernel_base_addr + 24) Points to Entry Point",
            fontsize=12, fontweight="bold", color="#263238")

    elf_text = (
        "1. ELF64 Header (Elf64_Ehdr) Standard Memory Layout:\n"
        "   - Bytes 00..15 (16 Bytes): e_ident (ELF Magic 0x7F 'ELF', 64-bit class, 2's complement little-endian, System V ABI)\n"
        "   - Bytes 16..17 (02 Bytes): e_type (0x0002 = ET_EXEC: Executable binary)\n"
        "   - Bytes 18..19 (02 Bytes): e_machine (0x003E = 62 = EM_X86_64: AMD x86-64 Architecture)\n"
        "   - Bytes 20..23 (04 Bytes): e_version (0x00000001 = EV_CURRENT)\n"
        "   - Bytes 24..31 (08 Bytes): e_entry (64-bit Entry Point Virtual Address)  <--- EXACTLY AT OFFSET 24 (0x18)!\n"
        "2. The 1MiB Boundary (0x100000) Choice:\n"
        "   - In x86 architecture, the first 1MiB (0x00000000 - 0x000FFFFF) contains IVT, BDA, EBDA, Video RAM (VRAM 0xA0000),\n"
        "     and BIOS ROM. Loading the kernel at 1MiB (0x100000) prevents hardware memory collisions while staying within 32-bit RAM limits.\n"
        "3. MapKey Mismatch & ExitBootServices Consistency:\n"
        "   - Any memory allocation or device event modifies the UEFI internal MemoryMap, invalidating the previous MapKey.\n"
        "   - MikanOS implements an immediate re-fetch loop (`GetMemoryMap` -> `ExitBootServices`) ensuring infallible transition."
    )
    ax.text(0.06, 0.11, elf_text, fontsize=8.8, family="monospace", color="#37474F")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_red_zone_mechanism_and_trap(output_path: str = "3/result/fig3_06_red_zone_mechanism_and_trap.png") -> str:
    """Render high-resolution diagram explaining System V AMD64 ABI Red Zone and the kernel interrupt trap."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=300)
    fig.suptitle(
        "System V AMD64 ABI Red Zone (128-Byte) Mechanism & Kernel Crash Trap (Column 3.1)",
        fontsize=16,
        fontweight="bold",
        color="#B71C1C",
        y=0.97
    )

    # 1. Left Subplot: User Space (Standard ABI with Red Zone) vs Interrupt Destruction
    ax1.set_facecolor("#FFFDE7")
    ax1.set_title("1. Standard User Space ABI: Red Zone Optimization & Kernel Trap", fontsize=12.5, fontweight="bold", color="#D84315", pad=12)

    stack_cards = [
        (0.80, "Caller Frame (Saved RSP, Return RIP)", "#ECEFF1", "#90A4AE", "Normal Stack Area"),
        (0.68, "Current Function Stack Frame (%rsp points here)", "#C8E6C9", "#43A047", "RSP (Current Top of Stack)"),
        (0.53, "RED ZONE [RSP - 1 .. RSP - 128] (128 Bytes)\nCompiler stores local variables here WITHOUT moving RSP!\n`mov [rsp - 8], rax`  (Faster: No `sub rsp, 128` needed)", "#FFCDD2", "#E53935", "Scratch Space (128B)"),
        (0.33, "[!] TRAP IN OS KERNEL: Asynchronous Interrupt Arrives!\nCPU Hardware automatically pushes 40 bytes directly onto %rsp:\n[SS, RSP, RFLAGS, CS, RIP]\n---> OVERWRITES AND CLOBBERS RED ZONE LOCAL VARIABLES!", "#FF8A80", "#B71C1C", "STACK CORRUPTION!"),
        (0.12, "Result: Silent Data Corruption, Undefined Behavior, Kernel Crash\nUpon interrupt return (`iretq`), clobbered local variables cause invalid memory pointers.", "#FFEBEE", "#C62828", "FATAL CRASH"),
    ]

    for y, label, fc, ec, tag in stack_cards:
        box = patches.FancyBboxPatch(
            (0.05, y - 0.08), 0.90, 0.12 if "TRAP" in label else 0.09,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2
        )
        ax1.add_patch(box)
        ax1.text(0.08, y - 0.01, label, fontsize=9.2, fontweight="bold" if "TRAP" in label else "normal",
                 color="#880E4F" if "TRAP" in label else "#212121", family="monospace" if "mov" in label else "sans-serif")
        ax1.text(0.72, y - 0.065, f"[{tag}]", fontsize=8.0, fontweight="bold", color=ec)

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. Right Subplot: OS Kernel Solution (-mno-red-zone)
    ax2.set_facecolor("#E8F5E9")
    ax2.set_title("2. OS Kernel Solution: Compiling with `-mno-red-zone`", fontsize=12.5, fontweight="bold", color="#1B5E20", pad=12)

    safe_stack_cards = [
        (0.80, "Caller Frame (Saved RSP, Return RIP)", "#ECEFF1", "#90A4AE", "Normal Stack Area"),
        (0.66, "Pushed Stack Frame via `sub rsp, N`\nCompiler explicitly decrements RSP before saving local variables.\nLocal variables reside INSIDE valid stack space [RSP .. RSP + N].", "#C8E6C9", "#2E7D32", "Valid Stack Frame"),
        (0.50, "%rsp points strictly BELOW all active local variables", "#A5D6A7", "#1B5E20", "RSP (Safe Position)"),
        (0.33, "[OK] SAFE: Asynchronous Interrupt / Exception Occurs!\nCPU pushes [SS, RSP, RFLAGS, CS, RIP] below the updated %rsp.\nLocal variables above %rsp are completely untouched and safe!", "#B9F6CA", "#00C853", "ZERO CORRUPTION"),
        (0.12, "Why MikanOS Mandates `-mno-red-zone`:\nEvery kernel and freestanding bare-metal binary must disable Red Zone\nto guarantee infallible hardware interrupt handling.", "#E8F5E9", "#2E7D32", "KERNEL STABILITY"),
    ]

    for y, label, fc, ec, tag in safe_stack_cards:
        box = patches.FancyBboxPatch(
            (0.05, y - 0.08), 0.90, 0.12 if "SAFE" in label else 0.09,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2
        )
        ax2.add_patch(box)
        ax2.text(0.08, y - 0.01, label, fontsize=9.2, fontweight="bold" if "SAFE" in label else "normal",
                 color="#1B5E20" if "SAFE" in label else "#212121", family="monospace" if "sub" in label else "sans-serif")
        ax2.text(0.72, y - 0.065, f"[{tag}]", fontsize=8.0, fontweight="bold", color=ec)

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 3.4: UEFI GOP (Graphics Output Protocol) & Framebuffer Simulation
# ==============================================================================

class UefiGraphicsPixelFormat(Enum):
    """UEFI GOP Graphics Pixel Formats according to UEFI Specification 2.9+."""

    PixelRedGreenBlueReserved8BitPerColor = 0
    PixelBlueGreenRedReserved8BitPerColor = 1
    PixelBitMask = 2
    PixelBltOnly = 3
    PixelFormatMax = 4


@dataclass
class EfiPixelBitmask:
    """Bitmask for PixelBitMask format in UEFI GOP."""

    red_mask: int = 0x000000FF
    green_mask: int = 0x0000FF00
    blue_mask: int = 0x00FF0000
    reserved_mask: int = 0xFF000000


@dataclass
class EfiGraphicsOutputModeInformation:
    """EFI_GRAPHICS_OUTPUT_MODE_INFORMATION structure."""

    version: int = 0
    horizontal_resolution: int = 800
    vertical_resolution: int = 600
    pixel_format: UefiGraphicsPixelFormat = (
        UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor
    )
    pixel_information: EfiPixelBitmask = field(default_factory=EfiPixelBitmask)
    pixels_per_scan_line: int = 800


@dataclass
class EfiGraphicsOutputProtocolMode:
    """EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE structure."""

    max_mode: int = 1
    mode: int = 0
    info: EfiGraphicsOutputModeInformation = field(
        default_factory=EfiGraphicsOutputModeInformation
    )
    size_of_info: int = 36  # sizeof(EFI_GRAPHICS_OUTPUT_MODE_INFORMATION)
    frame_buffer_base: int = 0x80000000  # Default simulated VRAM base in QEMU MMIO
    frame_buffer_size: int = 0

    def __post_init__(self) -> None:
        if self.frame_buffer_size == 0:
            self.frame_buffer_size = (
                self.info.pixels_per_scan_line * self.info.vertical_resolution * 4
            )


EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID = "9042a9de-23dc-4a38-96fb-7aded080516a"


class UefiGopSimulator:
    """Simulates UEFI GOP protocol discovery, mode query, and VRAM memory buffer operations."""

    def __init__(
        self,
        horizontal_resolution: int = 800,
        vertical_resolution: int = 600,
        pixels_per_scan_line: int | None = None,
        pixel_format: UefiGraphicsPixelFormat = UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor,
        frame_buffer_base: int = 0x80000000,
    ) -> None:
        if horizontal_resolution <= 0 or vertical_resolution <= 0:
            raise ValueError("Resolution dimensions must be positive integers")
        stride = (
            pixels_per_scan_line
            if pixels_per_scan_line is not None
            else horizontal_resolution
        )
        if stride < horizontal_resolution:
            raise ValueError(
                "pixels_per_scan_line cannot be smaller than horizontal_resolution"
            )

        self.info = EfiGraphicsOutputModeInformation(
            version=0,
            horizontal_resolution=horizontal_resolution,
            vertical_resolution=vertical_resolution,
            pixel_format=pixel_format,
            pixels_per_scan_line=stride,
        )
        total_size = stride * vertical_resolution * 4
        self.mode = EfiGraphicsOutputProtocolMode(
            max_mode=1,
            mode=0,
            info=self.info,
            size_of_info=36,
            frame_buffer_base=frame_buffer_base,
            frame_buffer_size=total_size,
        )
        # Raw VRAM byte buffer emulating physical memory
        self.vram = bytearray(total_size)
        self.handles = ["GopHandle0"]
        self.opened = False

    def open_gop(self, handle: str = "GopHandle0") -> bool:
        """Simulate gBS->OpenProtocol on EFI_GRAPHICS_OUTPUT_PROTOCOL."""
        if handle in self.handles:
            self.opened = True
            return True
        return False

    def get_pixel_format_unicode(self) -> str:
        """Return Unicode format name string matching MikanOS GetPixelFormatUnicode."""
        fmt = self.info.pixel_format
        if fmt == UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor:
            return "PixelRedGreenBlueReserved8BitPerColor"
        elif fmt == UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor:
            return "PixelBlueGreenRedReserved8BitPerColor"
        elif fmt == UefiGraphicsPixelFormat.PixelBitMask:
            return "PixelBitMask"
        elif fmt == UefiGraphicsPixelFormat.PixelBltOnly:
            return "PixelBltOnly"
        elif fmt == UefiGraphicsPixelFormat.PixelFormatMax:
            return "PixelFormatMax"
        return "InvalidPixelFormat"

    def calculate_byte_offset(self, x: int, y: int) -> int:
        """Calculate physical VRAM byte offset using pixels_per_scan_line (stride)."""
        if not (
            0 <= x < self.info.pixels_per_scan_line
            and 0 <= y < self.info.vertical_resolution
        ):
            raise IndexError(f"Coordinates ({x}, {y}) out of buffer bounds")
        return (y * self.info.pixels_per_scan_line + x) * 4

    def fill_all(self, byte_val: int = 255) -> None:
        """Fill entire frame buffer with byte_val (MikanOS day03b white clear)."""
        val = byte_val & 0xFF
        for i in range(len(self.vram)):
            self.vram[i] = val

    def write_pixel(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """Write single pixel at (x, y) taking pixel_format into account."""
        if not (
            0 <= x < self.info.horizontal_resolution
            and 0 <= y < self.info.vertical_resolution
        ):
            return
        offset = self.calculate_byte_offset(x, y)
        fmt = self.info.pixel_format
        if fmt == UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor:
            self.vram[offset + 0] = r & 0xFF
            self.vram[offset + 1] = g & 0xFF
            self.vram[offset + 2] = b & 0xFF
            self.vram[offset + 3] = 0
        elif fmt == UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor:
            self.vram[offset + 0] = b & 0xFF
            self.vram[offset + 1] = g & 0xFF
            self.vram[offset + 2] = r & 0xFF
            self.vram[offset + 3] = 0

    def get_pixel_rgb(self, x: int, y: int) -> tuple[int, int, int]:
        """Read back pixel color as (R, G, B) tuple."""
        offset = self.calculate_byte_offset(x, y)
        fmt = self.info.pixel_format
        if fmt == UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor:
            return (self.vram[offset], self.vram[offset + 1], self.vram[offset + 2])
        elif fmt == UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor:
            return (self.vram[offset + 2], self.vram[offset + 1], self.vram[offset])
        return (0, 0, 0)

    def to_rgb_matrix(self, include_stride_padding: bool = False) -> np.ndarray:
        """Convert raw VRAM bytes to (H, W, 3) RGB numpy matrix for visualization."""
        w = (
            self.info.pixels_per_scan_line
            if include_stride_padding
            else self.info.horizontal_resolution
        )
        h = self.info.vertical_resolution
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                offset = (y * self.info.pixels_per_scan_line + x) * 4
                fmt = self.info.pixel_format
                if (
                    fmt
                    == UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor
                ):
                    r, g, b = (
                        self.vram[offset],
                        self.vram[offset + 1],
                        self.vram[offset + 2],
                    )
                elif (
                    fmt
                    == UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor
                ):
                    b, g, r = (
                        self.vram[offset],
                        self.vram[offset + 1],
                        self.vram[offset + 2],
                    )
                else:
                    r = g = b = self.vram[offset]
                img[y, x] = [r, g, b]
        return img


def plot_uefi_gop_framebuffer_architecture(output_path: Path | str) -> Path:
    """Generate high-resolution diagram of UEFI GOP interface and VRAM memory layout (fig3_07)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig = plt.figure(figsize=(15, 9.5), facecolor="#F8FAFC")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2], wspace=0.18)

    # Left: UEFI GOP Protocol Chain & Discovery
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title(
        "1. UEFI Graphics Output Protocol (GOP) Hierarchy & Pointer Chain",
        fontsize=12,
        fontweight="bold",
        color="#1E293B",
        pad=12,
    )

    proto_boxes = [
        (
            0.88,
            "EFI_BOOT_SERVICES::LocateHandleBuffer()",
            "Discovers all devices supporting gEfiGraphicsOutputProtocolGuid\nReturns handle buffer (e.g. gop_handles[0])",
            "#EFF6FF",
            "#3B82F6",
            "1. Discovery",
        ),
        (
            0.71,
            "EFI_BOOT_SERVICES::OpenProtocol()",
            "Opens EFI_GRAPHICS_OUTPUT_PROTOCOL interface pointer on handle\nReturns: EFI_GRAPHICS_OUTPUT_PROTOCOL* gop",
            "#EEF2FF",
            "#6366F1",
            "2. Open Protocol",
        ),
        (
            0.52,
            "EFI_GRAPHICS_OUTPUT_PROTOCOL",
            "├── QueryMode: EFI_GRAPHICS_OUTPUT_PROTOCOL_QUERY_MODE\n├── SetMode:   EFI_GRAPHICS_OUTPUT_PROTOCOL_SET_MODE\n├── Blt:       EFI_GRAPHICS_OUTPUT_PROTOCOL_BLT\n└── Mode:      EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE*",
            "#F0FDF4",
            "#22C55E",
            "Protocol Struct",
        ),
        (
            0.30,
            "EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE",
            "├── MaxMode:           UINT32\n├── Mode:              UINT32 (Current Mode Index)\n├── Info:              EFI_GRAPHICS_OUTPUT_MODE_INFORMATION*\n├── SizeOfInfo:        UINTN (36 Bytes)\n├── FrameBufferBase:   EFI_PHYSICAL_ADDRESS (e.g. 0x80000000)\n└── FrameBufferSize:   UINTN (e.g. 800 * 600 * 4 = 1,920,000)",
            "#FEF3C7",
            "#F59E0B",
            "Mode Struct",
        ),
        (
            0.08,
            "EFI_GRAPHICS_OUTPUT_MODE_INFORMATION",
            "├── HorizontalResolution: 800 pixels\n├── VerticalResolution:   600 pixels\n├── PixelFormat:          PixelRedGreenBlueReserved8BitPerColor\n└── PixelsPerScanLine:    800 (or >800 if stride padded)",
            "#FDF2F8",
            "#EC4899",
            "Info Struct",
        ),
    ]

    for y, title, desc, fc, ec, badge in proto_boxes:
        h = 0.14 if "\n" in desc and desc.count("\n") > 2 else 0.12
        box = patches.FancyBboxPatch(
            (0.04, y - h / 2),
            0.92,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
            zorder=2,
        )
        ax1.add_patch(box)
        ax1.text(
            0.07,
            y + h / 2 - 0.035,
            title,
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
            family="monospace",
        )
        ax1.text(
            0.07,
            y - 0.01,
            desc,
            fontsize=8.0,
            color="#334155",
            family="monospace",
            linespacing=1.3,
        )
        ax1.text(
            0.82,
            y + h / 2 - 0.035,
            f"[{badge}]",
            fontsize=7.8,
            fontweight="bold",
            color=ec,
        )

    # Arrows connecting the pointer chains
    arrow_props = dict(
        arrowstyle="-|>",
        color="#64748B",
        lw=2.0,
        mutation_scale=15,
        connectionstyle="arc3,rad=0",
    )
    ax1.annotate("", xy=(0.50, 0.77), xytext=(0.50, 0.81), arrowprops=arrow_props)
    ax1.annotate("", xy=(0.50, 0.59), xytext=(0.50, 0.65), arrowprops=arrow_props)
    ax1.annotate("", xy=(0.50, 0.37), xytext=(0.50, 0.45), arrowprops=arrow_props)
    ax1.annotate("", xy=(0.50, 0.15), xytext=(0.50, 0.23), arrowprops=arrow_props)

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # Right: Physical Memory Space & VRAM MMIO Mapping
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title(
        "2. Physical Memory Space & Direct FrameBuffer (VRAM) Access",
        fontsize=12,
        fontweight="bold",
        color="#1E293B",
        pad=12,
    )

    mem_regions = [
        (
            0.86,
            0.10,
            "0x00000000 - 0x0009FFFF",
            "Conventional Low Memory (RAM, 640 KiB)",
            "#E2E8F0",
            "#64748B",
        ),
        (
            0.74,
            0.09,
            "0x000A0000 - 0x000BFFFF",
            "Legacy VGA Framebuffer (128 KiB, Text/Mode13h)",
            "#FEE2E2",
            "#EF4444",
        ),
        (
            0.62,
            0.10,
            "0x00100000 (1 MiB) ~",
            "Extended RAM (Kernel loaded here at 1 MiB)",
            "#DCFCE7",
            "#10B981",
        ),
        (
            0.45,
            0.20,
            "0x80000000 (2 GiB) ~",
            "PCI MMIO Window / VRAM (FrameBufferBase)\nDirectly mapped to GPU Video RAM!\nSize = PixelsPerScanLine * H * 4 bytes (e.g. 1.92 MiB)\nUINT8* fb = (UINT8*)gop->Mode->FrameBufferBase;\nfor(i=0; i<Size; ++i) fb[i] = 255;  // Entire Screen -> White!",
            "#FEF9C3",
            "#CA8A04",
        ),
        (
            0.20,
            0.18,
            "Display Controller (GPU Scanning)",
            "Hardware CRT/LCD Controller reads VRAM cyclically\nRAMDAC / Digital Encoder converts byte stream to video signal\nPixel byte = 0xFF 0xFF 0xFF 0xFF\n---> Red=255, Green=255, Blue=255 (Pure White Screen!)",
            "#E0E7FF",
            "#4F46E5",
        ),
    ]

    for y, h, title, desc, fc, ec in mem_regions:
        box = patches.FancyBboxPatch(
            (0.04, y - h / 2),
            0.92,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
            zorder=2,
        )
        ax2.add_patch(box)
        ax2.text(
            0.07,
            y + h / 2 - 0.035,
            title,
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
            family="monospace",
        )
        ax2.text(
            0.07,
            y - 0.01,
            desc,
            fontsize=8.0,
            color="#334155",
            family="monospace",
            linespacing=1.3,
        )

    # Hardware connection arrow from VRAM to GPU Display Controller
    ax2.annotate(
        "",
        xy=(0.50, 0.29),
        xytext=(0.50, 0.35),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#CA8A04",
            lw=3.0,
            mutation_scale=20,
            connectionstyle="arc3,rad=0",
        ),
    )
    ax2.text(
        0.53,
        0.315,
        "Cyclic Video Scan (60Hz)",
        fontsize=8.5,
        fontweight="bold",
        color="#B45309",
    )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_pixel_format_and_scanline_stride(output_path: Path | str) -> Path:
    """Generate high-resolution diagram of PixelFormat bytes and PixelsPerScanLine stride padding (fig3_08)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig = plt.figure(figsize=(15, 9.5), facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3], hspace=0.28)

    # Top: Pixel Format Byte Order Comparison (RGBR vs BGRR)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title(
        "1. GOP Pixel Format Byte Order & Memory Representation (4 Bytes per Pixel)",
        fontsize=12,
        fontweight="bold",
        color="#1E293B",
        pad=10,
    )

    # Pixel format 1: PixelRedGreenBlueReserved8BitPerColor
    ax1.text(
        0.04,
        0.82,
        "Format A: PixelRedGreenBlueReserved8BitPerColor (Standard RGBA order)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
    )
    bytes_a = [
        (0.04, "Byte +0: Red (8-bit)", "#FEE2E2", "#EF4444"),
        (0.24, "Byte +1: Green (8-bit)", "#DCFCE7", "#10B981"),
        (0.44, "Byte +2: Blue (8-bit)", "#DBEAFE", "#3B82F6"),
        (0.64, "Byte +3: Reserved (8-bit)", "#F1F5F9", "#94A3B8"),
    ]
    for x, label, fc, ec in bytes_a:
        box = patches.FancyBboxPatch(
            (x, 0.62),
            0.18,
            0.16,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
        ax1.add_patch(box)
        ax1.text(
            x + 0.02,
            0.70,
            label,
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
            family="monospace",
        )
    ax1.text(
        0.84,
        0.70,
        "32-bit uint:\n0xXXBBGGRR\n(Little-Endian)",
        fontsize=8.5,
        fontweight="bold",
        color="#475569",
        family="monospace",
    )

    # Pixel format 2: PixelBlueGreenRedReserved8BitPerColor
    ax1.text(
        0.04,
        0.42,
        "Format B: PixelBlueGreenRedReserved8BitPerColor (Standard BGRA order)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
    )
    bytes_b = [
        (0.04, "Byte +0: Blue (8-bit)", "#DBEAFE", "#3B82F6"),
        (0.24, "Byte +1: Green (8-bit)", "#DCFCE7", "#10B981"),
        (0.44, "Byte +2: Red (8-bit)", "#FEE2E2", "#EF4444"),
        (0.64, "Byte +3: Reserved (8-bit)", "#F1F5F9", "#94A3B8"),
    ]
    for x, label, fc, ec in bytes_b:
        box = patches.FancyBboxPatch(
            (x, 0.22),
            0.18,
            0.16,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
        ax1.add_patch(box)
        ax1.text(
            x + 0.02,
            0.30,
            label,
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
            family="monospace",
        )
    ax1.text(
        0.84,
        0.30,
        "32-bit uint:\n0xXXRRGGBB\n(Little-Endian)",
        fontsize=8.5,
        fontweight="bold",
        color="#475569",
        family="monospace",
    )

    ax1.text(
        0.04,
        0.06,
        "[*] Note: When frame_buffer[i] = 255 (0xFF) is executed for all bytes, R=255, G=255, B=255 regardless of format,\n    rendering pure white (0xFFFFFF) across all GOP implementations!",
        fontsize=8.5,
        fontweight="bold",
        color="#B45309",
        family="monospace",
    )

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # Bottom: Stride (PixelsPerScanLine) vs HorizontalResolution
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title(
        "2. The Scanline Stride Trap: PixelsPerScanLine vs HorizontalResolution",
        fontsize=12,
        fontweight="bold",
        color="#1E293B",
        pad=10,
    )

    # Draw 3 scanlines illustrating stride padding
    scanlines = [
        (0.72, "Scanline Y = 0"),
        (0.50, "Scanline Y = 1"),
        (0.28, "Scanline Y = 2"),
    ]
    for y, label in scanlines:
        ax2.text(
            0.04,
            y + 0.09,
            label,
            fontsize=9.0,
            fontweight="bold",
            color="#334155",
            family="monospace",
        )
        # Visible width
        box_vis = patches.FancyBboxPatch(
            (0.04, y),
            0.62,
            0.07,
            boxstyle="square,pad=0.0",
            facecolor="#C7D2FE",
            edgecolor="#4F46E5",
            linewidth=1.5,
        )
        ax2.add_patch(box_vis)
        ax2.text(
            0.06,
            y + 0.02,
            "Visible Pixels [X = 0 .. HorizontalResolution - 1] (e.g. 800 pixels)",
            fontsize=8.2,
            fontweight="bold",
            color="#1E1B4B",
        )

        # Padding stride
        box_pad = patches.FancyBboxPatch(
            (0.66, y),
            0.20,
            0.07,
            boxstyle="square,pad=0.0",
            facecolor="#FEE2E2",
            edgecolor="#EF4444",
            linewidth=1.5,
            hatch="//",
        )
        ax2.add_patch(box_pad)
        ax2.text(
            0.67,
            y + 0.02,
            "Padding (Stride - HRes)",
            fontsize=8.0,
            fontweight="bold",
            color="#991B1B",
        )

    # Brackets & Annotations
    ax2.annotate(
        "",
        xy=(0.04, 0.88),
        xytext=(0.66, 0.88),
        arrowprops=dict(arrowstyle="<->", color="#4F46E5", lw=2.0),
    )
    ax2.text(
        0.24,
        0.91,
        "HorizontalResolution (Visible Width: W)",
        fontsize=9.0,
        fontweight="bold",
        color="#4F46E5",
    )

    ax2.annotate(
        "",
        xy=(0.04, 0.20),
        xytext=(0.86, 0.20),
        arrowprops=dict(arrowstyle="<->", color="#059669", lw=2.0),
    )
    ax2.text(
        0.32,
        0.14,
        "PixelsPerScanLine (Total Scanline Stride: S >= W)",
        fontsize=9.0,
        fontweight="bold",
        color="#059669",
    )

    formula_box = patches.FancyBboxPatch(
        (0.04, 0.01),
        0.92,
        0.09,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#F0FDF4",
        edgecolor="#22C55E",
        linewidth=1.5,
    )
    ax2.add_patch(formula_box)
    ax2.text(
        0.06,
        0.045,
        "[CORRECT FORMULA]: offset = (y * PixelsPerScanLine + x) * 4\n[FATAL TRAP]: If you use (y * HorizontalResolution + x) * 4, graphics will slant / tear diagonally on real hardware!",
        fontsize=8.5,
        fontweight="bold",
        color="#15803D",
        family="monospace",
    )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


# ==============================================================================
# Chapter 3 Section 3.5: Drawing Pixels from Kernel (osbook_day03c)
# ==============================================================================

class CallingConventionKind(Enum):
    """x86-64 呼出規約の種別 (Calling Convention Kind)"""
    MICROSOFT_X64 = "Microsoft x64 ABI (Windows / UEFI Default)"
    SYSTEM_V_AMD64 = "System V AMD64 ABI (Linux / ELF / MikanOS Kernel)"


@dataclass
class KernelArgumentBridge:
    """ブートローダからカーネルへの引数引き渡し・ABIブリッジシミュレータ

    EDK II 環境 (Microsoft x64 ABI) から ELF カーネル (System V AMD64 ABI) へ
    引数 (frame_buffer_base, frame_buffer_size) を渡す際のレジスタマッピングと
    EFIAPI 修飾子の有無がもたらすコンパイラ挙動をシミュレートする。
    """
    frame_buffer_base: int
    frame_buffer_size: int

    def get_ms_x64_registers(self) -> dict[str, int]:
        """Microsoft x64 ABI における第1〜第4引数レジスタマッピング

        1st: RCX, 2nd: RDX, 3rd: R8, 4th: R9
        """
        return {
            "RCX": self.frame_buffer_base,
            "RDX": self.frame_buffer_size,
            "R8": 0,
            "R9": 0,
        }

    def get_system_v_registers(self) -> dict[str, int]:
        """System V AMD64 ABI における第1〜第6引数レジスタマッピング

        1st: RDI, 2nd: RSI, 3rd: RDX, 4th: RCX, 5th: R8, 6th: R9
        """
        return {
            "RDI": self.frame_buffer_base,
            "RSI": self.frame_buffer_size,
            "RDX": 0,
            "RCX": 0,
            "R8": 0,
            "R9": 0,
        }

    def simulate_call(self, use_efiapi: bool = False) -> dict[str, Any]:
        """関数の呼び出し規約とレジスタ配置をシミュレートする

        Parameters
        ----------
        use_efiapi : bool
            関数ポインタ宣言に EFIAPI (__attribute__((ms_abi))) が付与されているか

        Returns
        -------
        dict[str, Any]
            呼び出し規約、渡されるレジスタ、カーネルが受け取るレジスタ、および成否
        """
        if use_efiapi:
            caller_abi = CallingConventionKind.MICROSOFT_X64
            caller_regs = self.get_ms_x64_registers()
        else:
            caller_abi = CallingConventionKind.SYSTEM_V_AMD64
            caller_regs = self.get_system_v_registers()

        # カーネル (System V ABI) が第1, 第2引数を期待するレジスタ
        callee_expected_1st = "RDI"
        callee_expected_2nd = "RSI"

        received_base = caller_regs.get(callee_expected_1st, 0)
        received_size = caller_regs.get(callee_expected_2nd, 0)

        success = (
            received_base == self.frame_buffer_base
            and received_size == self.frame_buffer_size
        )

        return {
            "use_efiapi": use_efiapi,
            "caller_abi": caller_abi,
            "caller_registers": caller_regs,
            "callee_expected": (callee_expected_1st, callee_expected_2nd),
            "received_base": received_base,
            "received_size": received_size,
            "success": success,
            "explanation": (
                "EFIAPI修飾子を省略したため、GCC/Clangはホスト(Linux ELF)の標準である"
                "System V AMD64 ABIコード (mov rdi, base; mov rsi, size) を生成。"
                "カーネル側の期待レジスタ (RDI, RSI) と完全に一致し、引数が正常に渡る。"
                if not use_efiapi else
                "EFIAPI修飾子を付与した場合、Microsoft x64 ABI (mov rcx, base; mov rdx, size) "
                "で呼び出されるため、カーネル (RDI, RSI) は未定義のゴミ値を受け取り描画が破綻する。"
            ),
        }


class MikanKernelDrawer:
    """MikanOS 第3章 3.5節 (day03c) カーネルピクセル描画シミュレータ

    KernelMain(uint64_t frame_buffer_base, uint64_t frame_buffer_size) による
    frame_buffer[i] = i % 256 のピクセル描画アルゴリズム、色パッキング、
    ストライドに応じたモアレ・縞模様パターンを完全再現する。
    """

    def __init__(
        self,
        frame_buffer_base: int = 0xFFFF_8000_0000,
        width: int = 800,
        height: int = 600,
        pixels_per_scan_line: Optional[int] = None,
        pixel_format: UefiGraphicsPixelFormat = UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor,
    ):
        self.frame_buffer_base = frame_buffer_base
        self.width = width
        self.height = height
        self.stride = pixels_per_scan_line if pixels_per_scan_line is not None else width
        self.pixel_format = pixel_format
        self.frame_buffer_size = self.stride * self.height * 4
        self.buffer = bytearray(self.frame_buffer_size)

    def execute_modulo256(self) -> None:
        """KernelMain の核となる描画ループを実行する: frame_buffer[i] = i % 256"""
        pattern = bytes(i % 256 for i in range(256))
        full_cycles = self.frame_buffer_size // 256
        remainder = self.frame_buffer_size % 256
        self.buffer = bytearray(pattern * full_cycles + pattern[:remainder])

    def get_byte_at(self, offset: int) -> int:
        """指定オフセットのバイト値を取得する"""
        if 0 <= offset < self.frame_buffer_size:
            return self.buffer[offset]
        raise IndexError(f"Offset 0x{offset:X} out of frame buffer range")

    def get_pixel_bytes(self, x: int, y: int) -> tuple[int, int, int, int]:
        """座標 (x, y) における 4 バイト (byte0, byte1, byte2, byte3) を取得する"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")
        offset = (y * self.stride + x) * 4
        return (
            self.buffer[offset],
            self.buffer[offset + 1],
            self.buffer[offset + 2],
            self.buffer[offset + 3],
        )

    def get_pixel_rgb(self, x: int, y: int) -> tuple[int, int, int]:
        """座標 (x, y) の描画色を (Red, Green, Blue) の 0〜255 タプルで取得する"""
        b0, b1, b2, b3 = self.get_pixel_bytes(x, y)
        if self.pixel_format == UefiGraphicsPixelFormat.PixelRedGreenBlueReserved8BitPerColor:
            return (b0, b1, b2)
        elif self.pixel_format == UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor:
            return (b2, b1, b0)
        else:
            return (b0, b1, b2)

    def analyze_stride_periodicity(self) -> dict[str, Any]:
        """ストライドと 256 バイト周期 (64 ピクセル周期) の幾何学的関係を解析する

        1 ピクセルは 4 バイトであるため、256 バイトはちょうど 64 ピクセルに相当する。
        offset(x, y) = 4 * (y * stride + x)
        offset(x, y) mod 256 = (4 * y * stride + 4 * x) mod 256
        """
        pixel_period = 64
        stride_pixel_mod = self.stride % pixel_period
        stride_byte_mod = (self.stride * 4) % 256

        line_shift_bytes = stride_byte_mod
        line_shift_pixels = line_shift_bytes // 4

        is_pure_vertical = (stride_byte_mod == 0)

        return {
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
            "bytes_per_pixel": 4,
            "cycle_bytes": 256,
            "cycle_pixels": pixel_period,
            "stride_pixel_mod": stride_pixel_mod,
            "stride_byte_mod": stride_byte_mod,
            "line_shift_bytes": line_shift_bytes,
            "line_shift_pixels": line_shift_pixels,
            "is_pure_vertical": is_pure_vertical,
            "pattern_type": (
                "完全な垂直縦縞 (Pure Vertical Stripes)"
                if is_pure_vertical else
                f"斜め縞 / モアレ模様 (Slanted Stripes / Moire, 行シフト: {line_shift_pixels} ピクセル/行)"
            ),
        }

    def render_to_numpy_array(self) -> np.ndarray:
        """画面表示用の RGB uint8 NumPy 配列 (Height, Width, 3) を生成する"""
        if len(self.buffer) == 0:
            self.execute_modulo256()

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        raw_arr = np.frombuffer(self.buffer, dtype=np.uint8)

        stride_bytes = self.stride * 4
        visible_bytes = self.width * 4

        for y in range(self.height):
            start = y * stride_bytes
            end = start + visible_bytes
            row_raw = raw_arr[start:end].reshape((self.width, 4))
            if self.pixel_format == UefiGraphicsPixelFormat.PixelBlueGreenRedReserved8BitPerColor:
                img[y, :, 0] = row_raw[:, 2]
                img[y, :, 1] = row_raw[:, 1]
                img[y, :, 2] = row_raw[:, 0]
            else:
                img[y, :, 0] = row_raw[:, 0]
                img[y, :, 1] = row_raw[:, 1]
                img[y, :, 2] = row_raw[:, 2]

        return img


def plot_bootloader_to_kernel_handoff(
    output_path: str = "3/result/fig3_09_bootloader_to_kernel_handoff.png",
) -> str:
    """図 3-9: ブートローダからカーネルへの制御移譲とABIレジスタ引き渡し構造図を生成する"""
    setup_plot_style()
    fig = plt.figure(figsize=(15, 9.5), facecolor="#F8FAFC")

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # 1. ブートローダからカーネルへの主権移譲シーケンス
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title(
        "Figure 3-9 (A): Bootloader to Kernel Transition & ExitBootServices Architecture",
        fontsize=12,
        fontweight="bold",
        pad=15,
        color="#0F172A",
    )

    stages = [
        ("1. EDK II Bootloader\n(MikanLoaderPkg)", "UEFI GOP Discovery\n& Resolution Acquisition", "#DBEAFE", "#2563EB"),
        ("2. Save Framebuffer Info", "Base: 0x80000000\nSize: Stride * Height * 4", "#FEF3C7", "#D97706"),
        ("3. ExitBootServices()", "UEFI API Shutdown\nMapKey Verification", "#FEE2E2", "#DC2626"),
        ("4. Function Pointer Call", "entry_point(base, size)\nABI Bridge without EFIAPI", "#E0E7FF", "#4F46E5"),
        ("5. MikanOS Kernel\n(KernelMain)", "Direct VRAM Access\nframe_buffer[i] = i % 256", "#DCFCE7", "#16A34A"),
    ]

    for i, (title, desc, bg, edge) in enumerate(stages):
        x = 0.03 + i * 0.195
        box = patches.FancyBboxPatch(
            (x, 0.25),
            0.16,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=bg,
            edgecolor=edge,
            linewidth=2.0,
        )
        ax1.add_patch(box)
        ax1.text(
            x + 0.08,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=edge,
        )
        ax1.text(
            x + 0.08,
            0.38,
            desc,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#334155",
            family="monospace",
        )

        if i < len(stages) - 1:
            ax1.annotate(
                "",
                xy=(x + 0.19, 0.525),
                xytext=(x + 0.165, 0.525),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=2.5),
            )

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. ABI 呼び出し規約比較 (Microsoft x64 vs System V AMD64)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title(
        "Figure 3-9 (B): ABI Calling Convention Mismatch & Bridge",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#0F172A",
    )

    abi_table = [
        ["Argument", "Microsoft x64 (UEFI)", "System V AMD64 (ELF)", "Match?"],
        ["1st Arg (Base)", "RCX", "RDI", "MISMATCH!"],
        ["2nd Arg (Size)", "RDX", "RSI", "MISMATCH!"],
        ["3rd Arg", "R8", "RDX", "MISMATCH!"],
        ["4th Arg", "R9", "RCX", "MISMATCH!"],
    ]
    table = ax2.table(
        cellText=abi_table[1:],
        colLabels=abi_table[0],
        loc="center",
        cellLoc="center",
        colColours=["#E2E8F0", "#DBEAFE", "#DCFCE7", "#FEE2E2"],
    )
    table.scale(1.0, 1.7)
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#94A3B8")
        if key[1] == 3 and key[0] > 0:
            cell.set_text_props(color="#DC2626", fontweight="bold")
    ax2.axis("off")

    # 3. EDK II の EFIAPI 省略による自然な解決メカニズム
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title(
        "Figure 3-9 (C): Why It Works — The EFIAPI Miracle",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#0F172A",
    )

    mechanism_box = patches.FancyBboxPatch(
        (0.04, 0.08),
        0.92,
        0.84,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        facecolor="#F0FDF4",
        edgecolor="#16A34A",
        linewidth=1.8,
    )
    ax3.add_patch(mechanism_box)

    code_text = (
        "// In MikanLoaderPkg/Main.c:\n"
        "typedef void EntryPointType(UINT64, UINT64);\n"
        "EntryPointType* entry_point = (EntryPointType*)entry_addr;\n"
        "entry_point(gop->Mode->FrameBufferBase,\n"
        "            gop->Mode->FrameBufferSize);\n\n"
        "[*] Notice: NO 'EFIAPI' modifier!\n"
        "  - With EFIAPI: __attribute__((ms_abi)) -> passes in RCX, RDX (BROKEN!)\n"
        "  - WITHOUT EFIAPI: GCC/Clang emits host default (System V AMD64)!\n"
        "    -> Generates: 'mov rdi, base; mov rsi, size; call entry_addr'\n"
        "  - Kernel expects: RDI, RSI -> PERFECT HARMONY!"
    )
    ax3.text(
        0.08,
        0.50,
        code_text,
        ha="left",
        va="center",
        fontsize=9.0,
        color="#15803D",
        family="monospace",
        fontweight="bold",
    )
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_kernel_pixel_stripe_pattern(
    output_path: str = "3/result/fig3_10_kernel_pixel_stripe_pattern.png",
) -> str:
    """図 3-10: frame_buffer[i] = i % 256 のピクセルバイト配置とストライプ・モアレ描画再現図"""
    setup_plot_style()
    fig = plt.figure(figsize=(15, 10.0), facecolor="#F8FAFC")

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], hspace=0.35, wspace=0.25)

    # 1. 256バイト周期と4バイトピクセルパッキング
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title(
        "Figure 3-10 (A): 4-Byte Pixel Packing & 64-Pixel Modulo 256 Cycle",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#0F172A",
    )

    bytes_data = [
        ("Pixel 0", [("B: 0", "#93C5FD"), ("G: 1", "#86EFAC"), ("R: 2", "#FCA5A5"), ("Resv: 3", "#E2E8F0")]),
        ("Pixel 1", [("B: 4", "#60A5FA"), ("G: 5", "#4ADE80"), ("R: 6", "#F87171"), ("Resv: 7", "#CBD5E1")]),
        ("Pixel 2", [("B: 8", "#3B82F6"), ("G: 9", "#22C55E"), ("R: 10", "#EF4444"), ("Resv: 11", "#94A3B8")]),
        ("...", [("...", "#F1F5F9"), ("...", "#F1F5F9"), ("...", "#F1F5F9"), ("...", "#F1F5F9")]),
        ("Pixel 63", [("B: 252", "#1D4ED8"), ("G: 253", "#15803D"), ("R: 254", "#B91C1C"), ("Resv: 255", "#475569")]),
    ]

    for p_idx, (p_name, channels) in enumerate(bytes_data):
        y = 0.82 - p_idx * 0.17
        ax1.text(0.04, y + 0.05, p_name, fontsize=9.5, fontweight="bold", color="#1E293B")
        for c_idx, (label, color) in enumerate(channels):
            x = 0.22 + c_idx * 0.18
            box = patches.FancyBboxPatch(
                (x, y),
                0.16,
                0.10,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor="#475569",
                linewidth=1.2,
            )
            ax1.add_patch(box)
            ax1.text(
                x + 0.08,
                y + 0.05,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
                family="monospace",
            )

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. 幾何学的ストライドと縞模様の数理
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title(
        "Figure 3-10 (B): Mathematical Slant Analysis: Stride vs Periodicity",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#0F172A",
    )

    math_box = patches.FancyBboxPatch(
        (0.04, 0.06),
        0.92,
        0.88,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#EFF6FF",
        edgecolor="#3B82F6",
        linewidth=1.5,
    )
    ax2.add_patch(math_box)

    math_text = (
        "[Mathematical Formulation of Pixel Color]\n"
        "  offset(x, y) = 4 * (y * PixelsPerScanLine + x)\n"
        "  byte_val(x, y) = offset(x, y) mod 256\n"
        "                 = [4 * y * PixelsPerScanLine + 4 * x] mod 256\n\n"
        "[*] Case 1: Stride is a multiple of 64 (e.g. 1024)\n"
        "  - 4 * 1024 = 4096 = 256 * 16 (mod 256 == 0)\n"
        "  -> byte_val(x, y) = (4 * x) mod 256 (Independent of y!)\n"
        "  -> Result: PERFECT VERTICAL STRIPES\n\n"
        "[*] Case 2: Stride is NOT a multiple of 64 (e.g. 800)\n"
        "  - 4 * 800 = 3200 = 256 * 12 + 128 (mod 256 == 128)\n"
        "  -> byte_val(x, y) = (128 * y + 4 * x) mod 256\n"
        "  -> Result: 32-pixel horizontal phase shift per row\n"
        "  -> Generates SLANTED / MOIRE PATTERN"
    )
    ax2.text(
        0.08,
        0.50,
        math_text,
        ha="left",
        va="center",
        fontsize=8.5,
        color="#1E3A8A",
        family="monospace",
    )
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    # 3. 画面シミュレーション (Stride = 320 vs Stride = 350)
    ax3 = fig.add_subplot(gs[1, 0])
    drawer_vertical = MikanKernelDrawer(width=320, height=200, pixels_per_scan_line=320)
    drawer_vertical.execute_modulo256()
    img_v = drawer_vertical.render_to_numpy_array()
    ax3.imshow(img_v)
    ax3.set_title(
        "Figure 3-10 (C): Stride = 320 (Multiple of 64: Vertical Stripes)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax3.set_xlabel("Pixel X (Width = 320)")
    ax3.set_ylabel("Pixel Y (Height = 200)")

    ax4 = fig.add_subplot(gs[1, 1])
    drawer_slanted = MikanKernelDrawer(width=320, height=200, pixels_per_scan_line=350)
    drawer_slanted.execute_modulo256()
    img_s = drawer_slanted.render_to_numpy_array()
    ax4.imshow(img_s)
    ax4.set_title(
        "Figure 3-10 (D): Stride = 350 (Non-multiple: Slanted / Moire Pattern)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax4.set_xlabel("Pixel X (Width = 320)")
    ax4.set_ylabel("Pixel Y (Height = 200)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path



# ==============================================================================
# Chapter 3 Section 3.6: Error Handling and Pointer Casts (osbook_day03d)
# ==============================================================================

EFI_ERROR_MASK: int = 1 << 63

EFI_STATUS_NAMES: dict[int, str] = {
    0: "EFI_SUCCESS",
    1: "EFI_WARN_UNKNOWN_GLYPH",
    2: "EFI_WARN_DELETE_FAILURE",
    3: "EFI_WARN_WRITE_FAILURE",
    4: "EFI_WARN_BUFFER_TOO_SMALL",
    5: "EFI_WARN_STALE_DATA",
    6: "EFI_WARN_FILE_SYSTEM",
    7: "EFI_WARN_RESET_REQUIRED",
    1 | EFI_ERROR_MASK: "EFI_LOAD_ERROR",
    2 | EFI_ERROR_MASK: "EFI_INVALID_PARAMETER",
    3 | EFI_ERROR_MASK: "EFI_UNSUPPORTED",
    4 | EFI_ERROR_MASK: "EFI_BAD_BUFFER_SIZE",
    5 | EFI_ERROR_MASK: "EFI_BUFFER_TOO_SMALL",
    6 | EFI_ERROR_MASK: "EFI_NOT_READY",
    7 | EFI_ERROR_MASK: "EFI_DEVICE_ERROR",
    8 | EFI_ERROR_MASK: "EFI_WRITE_PROTECTED",
    9 | EFI_ERROR_MASK: "EFI_OUT_OF_RESOURCES",
    10 | EFI_ERROR_MASK: "EFI_VOLUME_CORRUPTED",
    11 | EFI_ERROR_MASK: "EFI_VOLUME_FULL",
    12 | EFI_ERROR_MASK: "EFI_NO_MEDIA",
    13 | EFI_ERROR_MASK: "EFI_MEDIA_CHANGED",
    14 | EFI_ERROR_MASK: "EFI_NOT_FOUND",
    15 | EFI_ERROR_MASK: "EFI_ACCESS_DENIED",
    16 | EFI_ERROR_MASK: "EFI_NO_RESPONSE",
    17 | EFI_ERROR_MASK: "EFI_NO_MAPPING",
    18 | EFI_ERROR_MASK: "EFI_TIMEOUT",
    19 | EFI_ERROR_MASK: "EFI_NOT_STARTED",
    20 | EFI_ERROR_MASK: "EFI_ALREADY_STARTED",
    21 | EFI_ERROR_MASK: "EFI_ABORTED",
    22 | EFI_ERROR_MASK: "EFI_ICMP_ERROR",
    23 | EFI_ERROR_MASK: "EFI_TFTP_ERROR",
    24 | EFI_ERROR_MASK: "EFI_PROTOCOL_ERROR",
    25 | EFI_ERROR_MASK: "EFI_INCOMPATIBLE_VERSION",
    26 | EFI_ERROR_MASK: "EFI_SECURITY_VIOLATION",
    27 | EFI_ERROR_MASK: "EFI_CRC_ERROR",
    28 | EFI_ERROR_MASK: "EFI_END_OF_MEDIA",
    31 | EFI_ERROR_MASK: "EFI_END_OF_FILE",
}


@dataclass(frozen=True)
class EFIStatus:
    """UEFI Specification 準拠の EFI_STATUS 表現クラス

    64-bit 整数値 (UINTN) で表現され、最上位ビット (Bit 63) が 1 の場合に
    エラー (EFI_ERROR) と判定される。
    """
    raw_value: int

    @property
    def is_error(self) -> bool:
        """EFI_ERROR マクロの数理再現: 最上位ビット (MSB, Bit 63) が 1 か判定"""
        return bool(self.raw_value & EFI_ERROR_MASK)

    @property
    def is_success(self) -> bool:
        """正常終了 (EFI_SUCCESS == 0) か判定"""
        return self.raw_value == 0

    @property
    def is_warning(self) -> bool:
        """警告ステータス (最上位ビットが 0 かつ 0 以外) か判定"""
        return not self.is_error and self.raw_value != 0

    @property
    def code_number(self) -> int:
        """エラー番号 / 状態番号 (下位 63 ビット) を抽出"""
        return self.raw_value & (~EFI_ERROR_MASK)

    @property
    def name(self) -> str:
        """EDK II Print(L"%r") に相当する文字列名を取得"""
        if self.raw_value in EFI_STATUS_NAMES:
            return EFI_STATUS_NAMES[self.raw_value]
        if self.is_error:
            return f"EFI_ERROR_{self.code_number}"
        return f"EFI_WARNING_{self.code_number}"

    def __repr__(self) -> str:
        return f"EFIStatus({self.name}, raw=0x{self.raw_value:016X})"

    @classmethod
    def success(cls) -> "EFIStatus":
        return cls(0)

    @classmethod
    def error(cls, code: int) -> "EFIStatus":
        return cls(code | EFI_ERROR_MASK)


class BootStepAction(Enum):
    """ブートシーケンスにおけるエラー発生時の対処方針"""
    HALT_FATAL = "Fatal Error: Print error and Halt CPU"
    IGNORE_GRACEFUL = "Non-Fatal: Print warning and continue boot"
    RETRY_THEN_HALT = "Retry: Attempt recovery (e.g. GetMemoryMap) then Halt if failed"


@dataclass
class BootStepConfig:
    name: str
    description: str
    action_on_error: BootStepAction
    uefi_function: str


@dataclass
class BootloaderStepResult:
    step_name: str
    status: EFIStatus
    passed: bool
    action_taken: str
    output_message: str


class BootloaderErrorSimulator:
    """MikanLoaderPkg (day03d) のエラーハンドリングシミュレータ

    各ブートステップにおける EFI_STATUS を追跡し、
    致命的エラーによる Halt() や、\\memmap オープン失敗時の無視継続
    (Graceful Degradation) をシミュレートする。
    """

    STEPS: list[BootStepConfig] = [
        BootStepConfig("GetMemoryMap", "メモリマップの取得", BootStepAction.HALT_FATAL, "gBS->GetMemoryMap"),
        BootStepConfig("OpenRootDir", "ルートディレクトリのオープン", BootStepAction.HALT_FATAL, "gBS->OpenProtocol(LoadedImage & SimpleFileSystem)"),
        BootStepConfig("OpenMemmapFile", "デバッグ用 \\memmap ファイルのオープン", BootStepAction.IGNORE_GRACEFUL, "root_dir->Open"),
        BootStepConfig("SaveMemoryMap", "メモリマップのファイル書き込み", BootStepAction.HALT_FATAL, "file->Write"),
        BootStepConfig("CloseMemmapFile", "\\memmap ファイルのクローズ", BootStepAction.HALT_FATAL, "file->Close"),
        BootStepConfig("OpenGOP", "UEFI Graphics Output Protocol の取得", BootStepAction.HALT_FATAL, "gBS->LocateHandleBuffer & OpenProtocol"),
        BootStepConfig("OpenKernelFile", "カーネル ELF (\\kernel.elf) のオープン", BootStepAction.HALT_FATAL, "root_dir->Open"),
        BootStepConfig("GetKernelFileInfo", "カーネルファイルサイズ情報の取得", BootStepAction.HALT_FATAL, "kernel_file->GetInfo"),
        BootStepConfig("AllocatePages", "カーネル配置用物理メモリ (0x100000) 確保", BootStepAction.HALT_FATAL, "gBS->AllocatePages"),
        BootStepConfig("ReadKernel", "カーネル ELF バイナリのメモリ読み込み", BootStepAction.HALT_FATAL, "kernel_file->Read"),
        BootStepConfig("ExitBootServices", "UEFI ブートサービス停止と制御移譲準備", BootStepAction.RETRY_THEN_HALT, "gBS->ExitBootServices"),
    ]

    def __init__(self) -> None:
        self.step_results: list[BootloaderStepResult] = []
        self.is_halted: bool = False
        self.boot_completed: bool = False

    def run_simulation(self, injected_errors: Optional[dict[str, EFIStatus]] = None) -> list[BootloaderStepResult]:
        """指定されたステップにエラーを注入してブートシーケンスを実行する"""
        if injected_errors is None:
            injected_errors = {}

        self.step_results.clear()
        self.is_halted = False
        self.boot_completed = False

        memmap_file_opened = True

        for cfg in self.STEPS:
            # \memmap がオープン失敗していた場合、Save/Close はスキップされる
            if cfg.name in ("SaveMemoryMap", "CloseMemmapFile") and not memmap_file_opened:
                self.step_results.append(
                    BootloaderStepResult(
                        step_name=cfg.name,
                        status=EFIStatus.success(),
                        passed=True,
                        action_taken="Skipped because \\memmap open was failed and ignored.",
                        output_message="[SKIPPED]",
                    )
                )
                continue

            status = injected_errors.get(cfg.name, EFIStatus.success())

            if status.is_error:
                if cfg.action_on_error == BootStepAction.IGNORE_GRACEFUL:
                    if cfg.name == "OpenMemmapFile":
                        memmap_file_opened = False
                    msg = f"failed to open file '\\memmap': {status.name}\nIgnored."
                    self.step_results.append(
                        BootloaderStepResult(
                            step_name=cfg.name,
                            status=status,
                            passed=False,
                            action_taken="Ignored and continued boot process (Graceful degradation)",
                            output_message=msg,
                        )
                    )
                elif cfg.action_on_error == BootStepAction.RETRY_THEN_HALT:
                    msg = f"failed to exit boot services: {status.name} -> Retrying GetMemoryMap..."
                    self.step_results.append(
                        BootloaderStepResult(
                            step_name=cfg.name,
                            status=status,
                            passed=False,
                            action_taken="Retried GetMemoryMap & ExitBootServices -> Failed -> Halt()",
                            output_message=msg,
                        )
                    )
                    self.is_halted = True
                    break
                else:
                    msg = f"failed at {cfg.name}: {status.name}\nHalt."
                    self.step_results.append(
                        BootloaderStepResult(
                            step_name=cfg.name,
                            status=status,
                            passed=False,
                            action_taken="Halt() executed (while(1) __asm__('hlt');)",
                            output_message=msg,
                        )
                    )
                    self.is_halted = True
                    break
            else:
                self.step_results.append(
                    BootloaderStepResult(
                        step_name=cfg.name,
                        status=status,
                        passed=True,
                        action_taken="Success: Proceeding to next step",
                        output_message="OK",
                    )
                )

        if not self.is_halted:
            self.boot_completed = True

        return self.step_results


class CppCastKind(Enum):
    """C++ のキャスト種別および C スタイルキャスト"""
    C_STYLE_CAST = "(Type)expression"
    STATIC_CAST = "static_cast<Type>(expression)"
    REINTERPRET_CAST = "reinterpret_cast<Type>(expression)"
    CONST_CAST = "const_cast<Type>(expression)"
    DYNAMIC_CAST = "dynamic_cast<Type>(expression)"


@dataclass
class CastAnalysisResult:
    cast_kind: CppCastKind
    source_type: str
    target_type: str
    is_legal: bool
    is_safe: bool
    compile_error_reason: Optional[str]
    low_level_usage: str


class PointerCastSimulator:
    """C++ における型変換・ポインタキャストの安全性と低レイヤ挙動の解析器

    コラム 3.2 で論じられる C スタイルキャストの危険性と、
    C++ 4大キャストの静的チェック機構・OS開発における使い分けを検証する。
    """

    @staticmethod
    def analyze(source_type: str, target_type: str, cast_kind: CppCastKind) -> CastAnalysisResult:
        s = source_type.strip()
        t = target_type.strip()

        # 1. 整数 (uint64_t / UINT64) から 関数ポインタ (void (*)(uint64_t, uint64_t) 等) への変換
        if ("uint64_t" in s or "UINT64" in s) and ("*" in t and "(" in t):
            if cast_kind == CppCastKind.STATIC_CAST:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=False,
                    is_safe=False,
                    compile_error_reason="error: invalid 'static_cast' from integer type to function pointer type",
                    low_level_usage="static_cast は整数とポインタ間の直接変換を禁止し、型の安全性を守る。",
                )
            elif cast_kind == CppCastKind.REINTERPRET_CAST:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=True,
                    is_safe=True,
                    compile_error_reason=None,
                    low_level_usage="カーネルエントリポイント呼び出し (entry_point = reinterpret_cast<EntryPointType*>(entry_addr)) の絶対標準。",
                )
            elif cast_kind == CppCastKind.C_STYLE_CAST:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=True,
                    is_safe=False,
                    compile_error_reason=None,
                    low_level_usage="コンパイルは通るが、const性の誤消失や危険な型変換を静的に警告しないため非推奨。",
                )
            elif cast_kind in (CppCastKind.CONST_CAST, CppCastKind.DYNAMIC_CAST):
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=False,
                    is_safe=False,
                    compile_error_reason="error: invalid cast kind for integer to function pointer conversion",
                    low_level_usage="const_cast/dynamic_cast では整数・ポインタ間の変換は文法上不可。",
                )

        # 2. void* / VOID* から 構造体ポインタ (EFI_FILE_INFO* 等)
        if "void*" in s.lower() and "*" in t and "(" not in t:
            if cast_kind in (CppCastKind.STATIC_CAST, CppCastKind.REINTERPRET_CAST, CppCastKind.C_STYLE_CAST):
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=True,
                    is_safe=True,
                    compile_error_reason=None,
                    low_level_usage="void* バッファから具象構造体ポインタへの復元 (static_cast 推奨)。",
                )
            else:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=False,
                    is_safe=False,
                    compile_error_reason="error: invalid cast kind for void* to object pointer",
                    low_level_usage="const_cast / dynamic_cast では void* からの具象型キャストは不可。",
                )

        # 3. const char* から char* へのキャスト
        if "const" in s and "const" not in t:
            if cast_kind == CppCastKind.CONST_CAST:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=True,
                    is_safe=False,
                    compile_error_reason=None,
                    low_level_usage="レガシーC API等に渡すために const 修飾を一時的に外す用途。",
                )
            elif cast_kind == CppCastKind.C_STYLE_CAST:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=True,
                    is_safe=False,
                    compile_error_reason=None,
                    low_level_usage="Cスタイルキャストは const を暗黙に剥奪してしまう危険性がある。",
                )
            else:
                return CastAnalysisResult(
                    cast_kind=cast_kind,
                    source_type=source_type,
                    target_type=target_type,
                    is_legal=False,
                    is_safe=False,
                    compile_error_reason="error: cast casts away const qualifier",
                    low_level_usage="static_cast / reinterpret_cast は const 剥奪を拒否して安全を守る。",
                )

        return CastAnalysisResult(
            cast_kind=cast_kind,
            source_type=source_type,
            target_type=target_type,
            is_legal=True,
            is_safe=False,
            compile_error_reason=None,
            low_level_usage="一般的な型変換",
        )


def plot_uefi_status_and_error_handling_flow(output_path: Path | str) -> Path:
    """Figure 3-11: EFI_STATUS 64-bit ビット構造と MikanLoaderPkg 多層エラー制御フローの描画"""
    fig = plt.figure(figsize=(15, 10), facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.4], hspace=0.35, wspace=0.25)

    # 1. EFI_STATUS 64ビットのビットレイアウト
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("Figure 3-11 (A): UEFI 64-bit EFI_STATUS Bit Layout & Severity Flag", fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    # MSB (Bit 63: Severity)
    box_msb = patches.FancyBboxPatch((0.05, 0.55), 0.22, 0.28, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=1.8)
    ax1.add_patch(box_msb)
    ax1.text(0.16, 0.73, "Bit 63 (MSB)\nERROR BIT", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#991B1B")
    ax1.text(0.16, 0.61, "1 = Error (EFI_ERROR=True)\n0 = Success / Warning", ha="center", va="center", fontsize=8, color="#B91C1C")

    # Bits 62..0 (Status Code)
    box_code = patches.FancyBboxPatch((0.31, 0.55), 0.64, 0.28, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#E0E7FF", edgecolor="#6366F1", linewidth=1.8)
    ax1.add_patch(box_code)
    ax1.text(0.63, 0.73, "Bits 62 .. 0\nStatus Code (0 .. 2^63 - 1)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#3730A3")
    ax1.text(0.63, 0.61, "0 = EFI_SUCCESS\n1 = EFI_LOAD_ERROR, 5 = BUFFER_TOO_SMALL, 14 = NOT_FOUND, etc.", ha="center", va="center", fontsize=8, color="#4338CA")

    # 数理・判定式解説
    formula_box = patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.38, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.2)
    ax1.add_patch(formula_box)
    formula_text = (
        "[Mathematical & C Implementation of EFI_ERROR(status)]\n"
        "  #define EFI_ERROR(Status)  (((INTN)(Status)) < 0)\n"
        "  - In 64-bit architecture (x86-64), INTN is signed 64-bit integer (int64_t).\n"
        "  - If Bit 63 (Sign Bit) is 1, the value is negative (< 0) in two's complement.\n"
        "  - Bitwise Mask Check:\n"
        "      is_error = (status & 0x8000000000000000ULL) != 0\n"
        "  - Extremely fast single-cycle instruction check: TEST RAX, RAX; JS error_handler"
    )
    ax1.text(0.08, 0.27, formula_text, ha="left", va="center", fontsize=8.5, family="monospace", color="#1E293B")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. 代表的 UEFI Status コード一覧表
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("Figure 3-11 (B): Typical UEFI Status Codes & Human-readable (%r)", fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    table_data = [
        ["EFI_STATUS Constant", "Hex Code (64-bit)", "Type", "EDK II %r Format"],
        ["EFI_SUCCESS", "0x0000000000000000", "Success", "Success"],
        ["EFI_LOAD_ERROR", "0x8000000000000001", "Error", "Load Error"],
        ["EFI_INVALID_PARAMETER", "0x8000000000000002", "Error", "Invalid Parameter"],
        ["EFI_BUFFER_TOO_SMALL", "0x8000000000000005", "Error", "Buffer Too Small"],
        ["EFI_OUT_OF_RESOURCES", "0x8000000000000009", "Error", "Out of Resources"],
        ["EFI_NOT_FOUND", "0x800000000000000E", "Error", "Not Found"],
        ["EFI_WARN_UNKNOWN_GLYPH", "0x0000000000000001", "Warning", "Unknown Glyph"],
    ]
    table = ax2.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#334155")
            cell.get_text().set_color("#FFFFFF")
            cell.get_text().set_fontweight("bold")
        elif "Error" in table_data[row][2]:
            cell.set_facecolor("#FEF2F2" if row % 2 == 1 else "#FFFFFF")
        elif "Success" in table_data[row][2]:
            cell.set_facecolor("#F0FDF4")
        else:
            cell.set_facecolor("#FFFBEB")
    ax2.axis("off")

    # 3. ブートローダ多層エラーハンドリング制御フロー
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("Figure 3-11 (C): Multi-Layer Defense Error Handling Flow in MikanLoaderPkg (day03d)", fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    flow_steps = [
        ("1. GetMemoryMap()", "Critical", 0.08, 0.65, "#EFF6FF", "#3B82F6"),
        ("2. OpenRootDir()", "Critical", 0.24, 0.65, "#EFF6FF", "#3B82F6"),
        ("3. Open(\\memmap)", "Non-Critical", 0.40, 0.65, "#FEF3C7", "#D97706"),
        ("4. OpenGOP()", "Critical", 0.56, 0.65, "#EFF6FF", "#3B82F6"),
        ("5. Load kernel.elf", "Critical", 0.72, 0.65, "#EFF6FF", "#3B82F6"),
        ("6. ExitBootServices()", "Retry & Exit", 0.88, 0.65, "#EFF6FF", "#3B82F6"),
    ]

    for title, role, x, y, bg, border in flow_steps:
        box = patches.FancyBboxPatch((x - 0.06, y - 0.12), 0.12, 0.24, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax3.add_patch(box)
        ax3.text(x, y + 0.04, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A")
        ax3.text(x, y - 0.04, f"[{role}]", ha="center", va="center", fontsize=7.5, color=border, fontweight="bold")

    # 矢印 (OK)
    for i in range(len(flow_steps) - 1):
        x1 = flow_steps[i][2] + 0.06
        x2 = flow_steps[i+1][2] - 0.06
        ax3.annotate("", xy=(x2, 0.65), xytext=(x1, 0.65), arrowprops=dict(arrowstyle="->", color="#10B981", lw=2))
        ax3.text((x1 + x2)/2, 0.68, "OK", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#10B981")

    # 致命的エラー -> Halt
    halt_box = patches.FancyBboxPatch((0.40, 0.12), 0.20, 0.18, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#FEE2E2", edgecolor="#EF4444", linewidth=2.0)
    ax3.add_patch(halt_box)
    ax3.text(0.50, 0.23, "Halt() Function", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#991B1B")
    ax3.text(0.50, 0.16, "while(1) __asm__(\"hlt\");\n(Low-power CPU pause)", ha="center", va="center", fontsize=7.5, color="#B91C1C", family="monospace")

    # エラー矢印
    for idx in (0, 1, 3, 4, 5):
        x_src = flow_steps[idx][2]
        ax3.annotate("", xy=(0.40 if x_src < 0.5 else 0.60, 0.25), xytext=(x_src, 0.53), arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5, ls="--"))
        ax3.text(x_src + 0.02, 0.44, "EFI_ERROR", fontsize=7, color="#EF4444", fontweight="bold")

    # 特例: \memmap のエラー無視矢印 (Graceful Continuation)
    ax3.annotate("", xy=(flow_steps[3][2] - 0.06, 0.60), xytext=(flow_steps[2][2] + 0.06, 0.60), arrowprops=dict(arrowstyle="->", color="#D97706", lw=2, ls=":"))
    ax3.text(0.48, 0.56, "Error Ignored\n(Log: 'Ignored.')", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#D97706")

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return Path(output_path)


def plot_pointer_casts_type_safety_memory(output_path: Path | str) -> Path:
    """Figure 3-12: C++ 4大キャストの型安全性マトリクスとメモリ再解釈モデルの描画"""
    fig = plt.figure(figsize=(15, 10), facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.3], hspace=0.35, wspace=0.25)

    # 1. キャストの型安全性比較マトリクス
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("Figure 3-12 (A): Type Safety Matrix: C-Style Cast vs Modern C++ Casts", fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    cast_matrix = [
        ["Cast Mechanism", "Static Check", "Memory Reinterpretation", "Const Stripping", "Low-Level / OS Usage", "Safety Level"],
        ["(Type)expr\n(C-style)", "None (Brute-force)", "Supported (Unchecked)", "Silently Drops const", "Permissive but highly bug-prone", "Low (Dangerous)"],
        ["static_cast<T>", "Strict Compile-time", "Prohibited for int<->ptr", "Prohibited (Compiler Error)", "Numeric conversion, void* to T*", "High (Type-Safe)"],
        ["reinterpret_cast<T>", "Size/Category only", "Supported (Bit reinterpretation)", "Prohibited (Compiler Error)", "int<->ptr, Function ptr, Hardware MMIO", "Medium (Explicit Hazard)"],
        ["const_cast<T>", "CV-qualifier only", "Prohibited (Compiler Error)", "Allowed (Exclusively)", "Legacy C API interop", "Low/Restricted"],
        ["dynamic_cast<T>", "Runtime (RTTI)", "Prohibited", "Prohibited", "Requires RTTI (Unavailable in Freestanding OS)", "High (Runtime cost)"],
    ]

    table = ax1.table(cellText=cast_matrix, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.get_text().set_color("#FFFFFF")
            cell.get_text().set_fontweight("bold")
        elif row == 1:
            cell.set_facecolor("#FEF2F2")
        elif row == 3:
            cell.set_facecolor("#EFF6FF")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 == 1 else "#FFFFFF")
    ax1.axis("off")

    # 2. メモリ空間における reinterpret_cast の再解釈モデル
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("Figure 3-12 (B): Memory Reinterpretation in Kernel Invocation", fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    # 物理アドレス 0x100000 のボックス
    mem_box = patches.FancyBboxPatch((0.08, 0.45), 0.35, 0.45, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#F1F5F9", edgecolor="#475569", linewidth=1.5)
    ax2.add_patch(mem_box)
    ax2.text(0.255, 0.83, "Physical Memory @ 0x100000\n(Loaded ELF Image)", ha="center", va="center", fontsize=9, fontweight="bold", color="#0F172A")
    ax2.text(0.255, 0.70, "Offset +0x00: e_ident (ELF Magic)\nOffset +0x18: e_entry (24 bytes)\n  [Value = 0x101000]", ha="center", va="center", fontsize=8, family="monospace", color="#334155")
    ax2.text(0.255, 0.52, "Raw 64-bit Address Value:\nuint64_t entry_addr = 0x101000", ha="center", va="center", fontsize=8, family="monospace", color="#1D4ED8", fontweight="bold")

    # キャスト矢印
    ax2.annotate("", xy=(0.60, 0.67), xytext=(0.45, 0.67), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=2))
    ax2.text(0.525, 0.72, "reinterpret_cast\n<EntryPointType*>", ha="center", va="center", fontsize=8, fontweight="bold", color="#3B82F6")

    # 関数ポインタとしての解釈
    func_box = patches.FancyBboxPatch((0.62, 0.45), 0.32, 0.45, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.8)
    ax2.add_patch(func_box)
    ax2.text(0.78, 0.83, "Callable Function Pointer\n(EntryPointType*)", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
    ax2.text(0.78, 0.68, "void (*entry_point)(\n  UINT64 fb_base,\n  UINT64 fb_size\n);", ha="center", va="center", fontsize=8, family="monospace", color="#1D4ED8")
    ax2.text(0.78, 0.52, "CPU: CALL 0x101000\n(RDI=fb_base, RSI=fb_size)", ha="center", va="center", fontsize=8, family="monospace", color="#0F172A")

    # 静的キャスト不可の注記
    warn_box = patches.FancyBboxPatch((0.08, 0.08), 0.86, 0.28, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=1.2)
    ax2.add_patch(warn_box)
    warn_text = (
        "[Why static_cast Fails Here]\n"
        "  - static_cast<EntryPointType*>(entry_addr) -> COMPILER ERROR!\n"
        "  - C++ standard forbids static_cast between integers and pointers.\n"
        "  - reinterpret_cast explicitly tells compiler: 'Treat this 64-bit integer\n"
        "    as a callable code address without generating conversion instructions'."
    )
    ax2.text(0.12, 0.22, warn_text, ha="left", va="center", fontsize=8, family="monospace", color="#991B1B")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    # 3. CPU Halt (HLT 命令) vs ビジーループの電力・パイプライン比較
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("Figure 3-12 (C): CPU Energy & Pipeline: HLT Instruction vs Busy-Loop", fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    # Busy Loop
    busy_box = patches.FancyBboxPatch((0.08, 0.55), 0.84, 0.38, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#FFF1F2", edgecolor="#F43F5E", linewidth=1.5)
    ax3.add_patch(busy_box)
    ax3.text(0.12, 0.86, "[Pattern A] Busy Loop: while(1);", fontsize=9, fontweight="bold", color="#BE123C")
    busy_desc = (
        "- Assembly: loop: JMP loop\n"
        "- CPU Core Utilization: 100% (Full load on one thread)\n"
        "- Instruction Pipeline: Decoder, ALU, Branch Predictor active continuously\n"
        "- Power Consumption: Max TDP / High heat generation"
    )
    ax3.text(0.12, 0.68, busy_desc, fontsize=8, family="monospace", color="#9F1239")

    # HLT Function
    hlt_box = patches.FancyBboxPatch((0.08, 0.08), 0.84, 0.40, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#F0FDF4", edgecolor="#22C55E", linewidth=1.8)
    ax3.add_patch(hlt_box)
    ax3.text(0.12, 0.41, "[Pattern B] Halt(): while(1) __asm__(\"hlt\");", fontsize=9, fontweight="bold", color="#15803D")
    hlt_desc = (
        "- Assembly: loop: HLT; JMP loop\n"
        "- CPU Core State: Enters HALT state (C1 power state)\n"
        "- Instruction Pipeline: Clock gating stops CPU clock until external interrupt\n"
        "- Power Consumption: Drastically reduced / Prevents thermal throttling"
    )
    ax3.text(0.12, 0.22, hlt_desc, fontsize=8, family="monospace", color="#166534")

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return Path(output_path)


# ============================================================================
# Chapter 3.7: Pointers & Assembly Level (SIB, MOV vs LEA, Disassembly & Volatile)
# ============================================================================


class AsmDataSize(Enum):
    """x86-64 memory access operand sizes."""

    BYTE = 1
    WORD = 2
    DWORD = 4
    QWORD = 8

    @property
    def specifier(self) -> str:
        """Return Intel assembly size specifier string."""
        mapping = {
            AsmDataSize.BYTE: "BYTE PTR",
            AsmDataSize.WORD: "WORD PTR",
            AsmDataSize.DWORD: "DWORD PTR",
            AsmDataSize.QWORD: "QWORD PTR",
        }
        return mapping[self]

    @classmethod
    def from_type_name(cls, name: str) -> AsmDataSize:
        """Map C/C++ type name to AsmDataSize."""
        cleaned = name.strip().lower()
        if cleaned in ("uint8_t", "int8_t", "char", "unsigned char", "bool"):
            return cls.BYTE
        if cleaned in ("uint16_t", "int16_t", "short", "unsigned short", "char16_t"):
            return cls.WORD
        if cleaned in ("uint32_t", "int32_t", "int", "unsigned int", "char32_t", "float"):
            return cls.DWORD
        if cleaned in ("uint64_t", "int64_t", "long", "unsigned long", "void*", "double") or "*" in cleaned:
            return cls.QWORD
        raise ValueError(f"Unknown type name: {name}")


@dataclass
class SIBAddress:
    """x86-64 Scale-Index-Base (SIB) memory addressing calculation.

    Effective Address = Base + (Index * Scale) + Displacement
    """

    base: int = 0
    index: int = 0
    scale: int = 1
    displacement: int = 0

    def __post_init__(self) -> None:
        if self.scale not in (1, 2, 4, 8):
            raise ValueError(f"Invalid SIB scale factor: {self.scale}. Must be 1, 2, 4, or 8.")

    def calculate_effective_address(self) -> int:
        """Calculate effective linear memory address."""
        return self.base + (self.index * self.scale) + self.displacement

    def scale_encoding(self) -> int:
        """Return 2-bit SIB scale encoding (00b=1, 01b=2, 10b=4, 11b=8)."""
        mapping = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}
        return mapping[self.scale]

    def to_assembly_string(self, base_reg: str = "rax", index_reg: str = "rcx") -> str:
        """Format as Intel assembly memory operand bracket."""
        parts = [base_reg]
        if index_reg:
            if self.scale > 1:
                parts.append(f"{index_reg}*{self.scale}")
            else:
                parts.append(index_reg)
        expr = " + ".join(parts)
        if self.displacement > 0:
            expr += f" + {self.displacement}"
        elif self.displacement < 0:
            expr += f" - {-self.displacement}"
        return f"[{expr}]"


class VirtualCpu:
    """Lightweight virtual x86-64 machine simulator for MOV vs LEA verification."""

    REGISTERS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                 "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")

    def __init__(self, memory_size: int = 65536) -> None:
        self.memory = bytearray(memory_size)
        self.registers = {r: 0 for r in self.REGISTERS}
        self.memory_write_count = 0
        self.memory_read_count = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.executed_instructions: list[str] = []

    def get_reg(self, name: str) -> int:
        return self.registers[name.lower()]

    def set_reg(self, name: str, val: int) -> None:
        self.registers[name.lower()] = val & 0xFFFF_FFFF_FFFF_FFFF

    def execute_lea(self, dest_reg: str, sib: SIBAddress) -> int:
        """Execute LEA dest, [mem].

        CRITICAL: LEA performs address calculation ONLY and NEVER touches the memory bus!
        """
        eff_addr = sib.calculate_effective_address()
        self.set_reg(dest_reg, eff_addr)
        asm_line = f"lea {dest_reg.lower()}, {sib.to_assembly_string('rax', 'rcx')}"
        self.executed_instructions.append(asm_line)
        return eff_addr

    def execute_mov_store(self, sib: SIBAddress, src_reg: str, size: AsmDataSize) -> None:
        """Execute MOV [mem], src_reg (Write register value to memory via pointer)."""
        eff_addr = sib.calculate_effective_address()
        val = self.get_reg(src_reg)
        num_bytes = size.value

        if eff_addr + num_bytes > len(self.memory):
            raise IndexError("Memory store out of range")

        val_bytes = val.to_bytes(8, byteorder="little")[:num_bytes]
        self.memory[eff_addr:eff_addr + num_bytes] = val_bytes
        self.memory_write_count += 1
        self.bytes_written += num_bytes

        asm_line = f"mov {size.specifier} {sib.to_assembly_string('rax', 'rcx')}, {src_reg.lower()}"
        self.executed_instructions.append(asm_line)

    def execute_mov_load(self, dest_reg: str, sib: SIBAddress, size: AsmDataSize) -> int:
        """Execute MOV dest_reg, [mem] (Read memory value into register via pointer)."""
        eff_addr = sib.calculate_effective_address()
        num_bytes = size.value

        if eff_addr + num_bytes > len(self.memory):
            raise IndexError("Memory load out of range")

        raw_bytes = self.memory[eff_addr:eff_addr + num_bytes]
        val = int.from_bytes(raw_bytes, byteorder="little")
        self.set_reg(dest_reg, val)
        self.memory_read_count += 1
        self.bytes_read += num_bytes

        asm_line = f"mov {dest_reg.lower()}, {size.specifier} {sib.to_assembly_string('rax', 'rcx')}"
        self.executed_instructions.append(asm_line)
        return val


@dataclass
class PixelOptimizationProfile:
    """Disassembly analysis profile for a pixel drawing loop."""

    opt_level: str
    is_volatile: bool
    instructions_executed: int
    memory_writes: int
    total_bytes_written: int
    asm_snippet: list[str]


class PixelDisassemblyAnalyzer:
    """Simulator analyzing compiler optimization behavior for VRAM/framebuffer loops."""

    @staticmethod
    def simulate_o0(pixel_count: int) -> PixelOptimizationProfile:
        """Simulate unoptimized -O0 compilation loop (naive stack and recalculation)."""
        instrs_per_iter = 8
        overhead = 2
        return PixelOptimizationProfile(
            opt_level="-O0 (Unoptimized)",
            is_volatile=False,
            instructions_executed=overhead + pixel_count * instrs_per_iter,
            memory_writes=pixel_count,
            total_bytes_written=pixel_count,
            asm_snippet=[
                "mov    DWORD PTR [rbp-4], 0",
                ".L2:",
                "mov    eax, DWORD PTR [rbp-4]",
                "cdqe",
                "mov    rdx, QWORD PTR [rbp-16]",
                "add    rax, rdx",
                "mov    BYTE PTR [rax], 255",
                "add    DWORD PTR [rbp-4], 1",
                "cmp    DWORD PTR [rbp-4], 100",
                "jl     .L2",
            ],
        )

    @staticmethod
    def simulate_o2(pixel_count: int) -> PixelOptimizationProfile:
        """Simulate optimized -O2 loop with SIMD (SSE/AVX 16-byte stores)."""
        vector_size = 16
        vector_iters = pixel_count // vector_size
        remainder = pixel_count % vector_size
        setup_instrs = 4
        vector_instrs = vector_iters * 3
        remainder_instrs = remainder * 3

        return PixelOptimizationProfile(
            opt_level="-O2 (SIMD Vectorized & Unrolled)",
            is_volatile=False,
            instructions_executed=setup_instrs + vector_instrs + remainder_instrs,
            memory_writes=vector_iters + remainder,
            total_bytes_written=pixel_count,
            asm_snippet=[
                "movdqa xmm0, XMMWORD PTR .LC0[rip]  ; 16 bytes broadcast 0xFF",
                "xor    eax, eax",
                ".L3:",
                "movups XMMWORD PTR [rdi+rax], xmm0  ; 16 pixels per instruction!",
                "add    rax, 16",
                "cmp    rax, 96",
                "jne    .L3",
                "mov    DWORD PTR [rdi+96], -1       ; Remainder scalar store",
            ],
        )

    @staticmethod
    def simulate_volatile_o2(pixel_count: int) -> PixelOptimizationProfile:
        """Simulate -O2 with volatile uint8_t* (Guarantees exact byte stores, no vector fusion)."""
        instrs_per_iter = 3
        setup = 2
        return PixelOptimizationProfile(
            opt_level="-O2 with volatile uint8_t*",
            is_volatile=True,
            instructions_executed=setup + pixel_count * instrs_per_iter,
            memory_writes=pixel_count,
            total_bytes_written=pixel_count,
            asm_snippet=[
                "xor    eax, eax",
                ".L4:",
                "mov    BYTE PTR [rdi+rax], 255      ; Guaranteed single-byte write!",
                "add    rax, 1",
                "cmp    rax, 100",
                "jne    .L4",
            ],
        )


def plot_fig3_13_pointer_to_assembly_mapping_and_sib(output_path: str | Path) -> Path:
    """Generate Figure 3-13: Pointer to Assembly Mapping & SIB Hardware Addressing.

    Panels:
    (A) SIB (Scale-Index-Base) Hardware Addressing Mode & 2-bit scale encoding
    (B) LEA vs MOV Hardware Execution Flow (Address Calculation vs Memory Bus Access)
    (C) C++ Data Types & x86-64 Memory Size Specifiers (BYTE/WORD/DWORD/QWORD PTR)
    """
    setup_plot_style()
    fig = plt.figure(figsize=(15, 11), facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # ----------------------------------------------------
    # Panel A: SIB Hardware Addressing
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("Figure 3-13 (A): x86-64 SIB (Scale-Index-Base) Hardware Addressing & C Pointer Arithmetic",
                  fontsize=12, fontweight="bold", pad=12, color="#0F172A")

    # Overall SIB formula
    formula_box = patches.FancyBboxPatch((0.05, 0.72), 0.90, 0.20,
                                         boxstyle="round,pad=0.02,rounding_size=0.03",
                                         facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.8)
    ax1.add_patch(formula_box)
    ax1.text(0.50, 0.85, "Effective Linear Address = Base + (Index × Scale) + Displacement",
             ha="center", va="center", fontsize=13, fontweight="bold", color="#1E3A8A")
    ax1.text(0.50, 0.76, "Intel Syntax: [BaseReg + IndexReg * Scale + Disp]  (e.g., [rdi + rsi*4 + 16])",
             ha="center", va="center", fontsize=10, family="monospace", color="#2563EB")

    # 4 Components
    components = [
        ("Base Register (64-bit)", "Pointer Base Address\n(e.g., RAX, RDI, RSP, RBP)\nPoints to buffer start", "#FEF3C7", "#D97706", 0.05, 0.20),
        ("Index Register (64-bit)", "Array Index Variable `i`\n(e.g., RCX, RSI, RDX)\nZero-extended integer", "#E0E7FF", "#4338CA", 0.28, 0.20),
        ("Scale Factor (2-bit SIB)", "Size of Array Element\n1 (00b), 2 (01b), 4 (10b), 8 (11b)\nHardware sizeof(T) multiplier", "#DCFCE7", "#15803D", 0.51, 0.20),
        ("Displacement (Offset)", "Immediate Offset (const)\nSigned 8-bit or 32-bit int\nStruct member offset & stack", "#FCE7F3", "#BE185D", 0.74, 0.20),
    ]

    for title, desc, bg, border, x, w in components:
        box = patches.FancyBboxPatch((x, 0.15), w, 0.48, boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor=bg, edgecolor=border, linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(x + w / 2, 0.55, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A")
        ax1.text(x + w / 2, 0.34, desc, ha="center", va="center", fontsize=8.5, family="monospace", color="#334155")

    # Connectors
    ax1.text(0.26, 0.39, "+", fontsize=16, fontweight="bold", ha="center", va="center", color="#64748B")
    ax1.text(0.495, 0.39, "×", fontsize=16, fontweight="bold", ha="center", va="center", color="#64748B")
    ax1.text(0.725, 0.39, "+", fontsize=16, fontweight="bold", ha="center", va="center", color="#64748B")

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # ----------------------------------------------------
    # Panel B: LEA vs MOV
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("Figure 3-13 (B): LEA vs MOV Hardware Execution Flow",
                  fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    # LEA Box
    lea_box = patches.FancyBboxPatch((0.05, 0.52), 0.90, 0.40,
                                     boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.6)
    ax2.add_patch(lea_box)
    ax2.text(0.08, 0.85, "LEA (Load Effective Address): `lea rdi, [rax + rcx*4]`",
             fontsize=10, fontweight="bold", color="#15803D")
    lea_desc = (
        "• Operation: Evaluates SIB address inside ALU arithmetic unit\n"
        "• Memory Bus Access: ZERO (Never accesses RAM or Cache!)\n"
        "• Result: Destination register receives the calculated pointer address\n"
        "• C Equivalent: Pointer calculation or address-of `&p[i]` / `p + i`"
    )
    ax2.text(0.08, 0.67, lea_desc, fontsize=8, family="monospace", color="#166534")

    # MOV Box
    mov_box = patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.40,
                                     boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.6)
    ax2.add_patch(mov_box)
    ax2.text(0.08, 0.39, "MOV (Memory Dereference): `mov DWORD PTR [rax + rcx*4], edx`",
             fontsize=10, fontweight="bold", color="#991B1B")
    mov_desc = (
        "• Operation: Evaluates address AND initiates Memory Bus Transfer\n"
        "• Memory Bus Access: YES (L1/L2/L3 Cache or VRAM/RAM Bus Transaction)\n"
        "• Result: Memory at target address is read from or modified\n"
        "• C Equivalent: Pointer dereference `p[i] = val` or `val = p[i]`"
    )
    ax2.text(0.08, 0.21, mov_desc, fontsize=8, family="monospace", color="#7F1D1D")

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    # ----------------------------------------------------
    # Panel C: Data Size Specifiers
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("Figure 3-13 (C): Type Sizes & x86-64 Memory Specifiers",
                  fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    spec_rows = [
        ("BYTE PTR", "1 byte (8-bit)", "uint8_t, char, bool", "Scale = 1 (00b)", "mov BYTE PTR [rax], dl", "#FEF3C7", "#92400E"),
        ("WORD PTR", "2 bytes (16-bit)", "uint16_t, short", "Scale = 2 (01b)", "mov WORD PTR [rax], dx", "#E0E7FF", "#3730A3"),
        ("DWORD PTR", "4 bytes (32-bit)", "uint32_t, int, float", "Scale = 4 (10b)", "mov DWORD PTR [rax], edx", "#DCFCE7", "#166534"),
        ("QWORD PTR", "8 bytes (64-bit)", "uint64_t, pointers (T*)", "Scale = 8 (11b)", "mov QWORD PTR [rax], rdx", "#FCE7F3", "#9D174D"),
    ]

    y_start = 0.82
    y_step = 0.22
    for spec, size_str, types, scale_info, asm_ex, bg, fg in spec_rows:
        row_box = patches.FancyBboxPatch((0.05, y_start - 0.17), 0.90, 0.19,
                                         boxstyle="round,pad=0.015,rounding_size=0.02",
                                         facecolor=bg, edgecolor=fg, linewidth=1.2)
        ax3.add_patch(row_box)
        ax3.text(0.08, y_start - 0.05, f"{spec} ({size_str})", fontsize=9.5, fontweight="bold", color=fg)
        ax3.text(0.08, y_start - 0.13, f"C Types: {types} | {scale_info}", fontsize=8, color="#334155")
        ax3.text(0.92, y_start - 0.09, asm_ex, ha="right", va="center", fontsize=8, family="monospace", color="#0F172A")
        y_start -= y_step

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return Path(output_path)


def plot_fig3_14_disassembly_optimization_and_volatile(output_path: str | Path) -> Path:
    """Generate Figure 3-14: Disassembly Optimization & Volatile Memory Semantics.

    Panels:
    (A) Disassembly comparison (-O0 vs -O2 SIMD vs -O2 volatile)
    (B) Quantitative benchmark: Instructions executed vs Memory writes (N=100 pixels)
    (C) Hardware VRAM / MMIO write mechanism and why volatile is mandatory
    """
    setup_plot_style()
    fig = plt.figure(figsize=(15, 11), facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)

    # ----------------------------------------------------
    # Panel A: Disassembly Comparison
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("Figure 3-14 (A): Framebuffer Loop Disassembly Comparison: -O0 vs -O2 (SIMD) vs -O2 with volatile",
                  fontsize=12, fontweight="bold", pad=12, color="#0F172A")

    columns = [
        ("-O0 (No Optimization)", "#FEF2F2", "#DC2626", [
            "; Naive stack recalculation each step",
            "mov    DWORD PTR [rbp-4], 0",
            ".L2:",
            "mov    eax, DWORD PTR [rbp-4]",
            "cdqe",
            "mov    rdx, QWORD PTR [rbp-16]",
            "add    rax, rdx",
            "mov    BYTE PTR [rax], 255",
            "add    DWORD PTR [rbp-4], 1",
            "cmp    DWORD PTR [rbp-4], 100",
            "jl     .L2",
        ], 0.03, 0.29),
        ("-O2 (SIMD Auto-Vectorized)", "#EFF6FF", "#2563EB", [
            "; 16 bytes (128-bit) per instruction!",
            "movdqa xmm0, XMMWORD PTR .LC0[rip]",
            "xor    eax, eax",
            ".L3:",
            "movups XMMWORD PTR [rdi+rax], xmm0",
            "add    rax, 16",
            "cmp    rax, 96",
            "jne    .L3",
            "; Scalar stores for remainder",
            "mov    DWORD PTR [rdi+96], -1",
            "; Total: ~34 instructions!",
        ], 0.355, 0.29),
        ("-O2 with volatile uint8_t*", "#F0FDF4", "#16A34A", [
            "; Dedicated register pointer step",
            "; Guaranteed 1-byte memory write",
            "xor    eax, eax",
            ".L4:",
            "mov    BYTE PTR [rdi+rax], 255",
            "add    rax, 1",
            "cmp    rax, 100",
            "jne    .L4",
            "; Never vectorized or eliminated!",
            "; Exact 100 bus writes guaranteed",
        ], 0.68, 0.29),
    ]

    for title, bg, border, asm_lines, x, w in columns:
        col_box = patches.FancyBboxPatch((x, 0.08), w, 0.82, boxstyle="round,pad=0.02,rounding_size=0.03",
                                         facecolor=bg, edgecolor=border, linewidth=1.5)
        ax1.add_patch(col_box)
        ax1.text(x + w / 2, 0.84, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A")
        y_text = 0.74
        for line in asm_lines:
            color = "#047857" if line.startswith(";") else "#1E293B"
            ax1.text(x + 0.015, y_text, line, fontsize=7.5, family="monospace", color=color)
            y_text -= 0.065

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # ----------------------------------------------------
    # Panel B: Quantitative Benchmark (N=100)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("Figure 3-14 (B): Instruction Count & Bus Writes (N=100 Pixels)",
                  fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    labels = ["-O0\n(Naive)", "-O2\n(SIMD)", "-O2 + volatile\n(Hardware Safe)"]
    instrs = [802, 34, 302]
    writes = [100, 7, 100]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax2.bar(x - width/2, instrs, width, label="Instructions Executed", color="#6366F1", alpha=0.85)
    rects2 = ax2.bar(x + width/2, writes, width, label="Memory Bus Writes", color="#10B981", alpha=0.85)

    ax2.set_ylabel("Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.set_ylim(0, 950)

    # Values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{h}",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4338CA")
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h}",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color="#065F46")

    # ----------------------------------------------------
    # Panel C: Volatile & MMIO / VRAM Hardware Necessity
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("Figure 3-14 (C): Why `volatile` is Mandatory for Hardware MMIO / VRAM",
                  fontsize=11, fontweight="bold", pad=12, color="#0F172A")

    # Dead Store Elimination Danger
    danger_box = patches.FancyBboxPatch((0.05, 0.52), 0.90, 0.40,
                                        boxstyle="round,pad=0.02,rounding_size=0.03",
                                        facecolor="#FFF1F2", edgecolor="#F43F5E", linewidth=1.5)
    ax3.add_patch(danger_box)
    ax3.text(0.08, 0.85, "[Danger] Dead Store Elimination (Without volatile)",
             fontsize=9.5, fontweight="bold", color="#BE123C")
    danger_desc = (
        "• Compiler sees: `*p = 10; *p = 20;`\n"
        "• Optimizer assumption: 'Memory at p is passive RAM; first write is DEAD!'\n"
        "• Result: Compiler deletes `*p = 10`! Only `*p = 20` is emitted.\n"
        "• Hardware catastrophe: Hardware FIFO or reset register misses trigger!"
    )
    ax3.text(0.08, 0.67, danger_desc, fontsize=8, family="monospace", color="#9F1239")

    # Volatile Guarantee
    safe_box = patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.40,
                                      boxstyle="round,pad=0.02,rounding_size=0.03",
                                      facecolor="#F0FDF4", edgecolor="#22C55E", linewidth=1.5)
    ax3.add_patch(safe_box)
    ax3.text(0.08, 0.39, "[Guarantee] Observable Side Effect (With volatile uint8_t*)",
             fontsize=9.5, fontweight="bold", color="#15803D")
    safe_desc = (
        "• Compiler semantics: 'Every read/write is an observable side effect!'\n"
        "• Optimization barrier: No reordering, no dead store elimination, no cache omission\n"
        "• Result: Exactly 1 CPU MOV instruction hits the memory bus per iteration\n"
        "• Golden rule: ALL MMIO, VRAM framebuffers, and interrupt FIFOs MUST be volatile!"
    )
    ax3.text(0.08, 0.21, safe_desc, fontsize=8, family="monospace", color="#166534")

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return Path(output_path)


# =============================================================================
# Chapter 3 Exercises: Comprehensive Verification & Quiz Engine
# =============================================================================

class Chapter3Quiz:
    """Comprehensive multiple-choice quiz engine covering Sections 3.1 to 3.7."""

    def __init__(self) -> None:
        self.questions: list[QuizQuestion] = [
            QuizQuestion(
                question_id=1,
                section="3.1",
                topic="QEMUモニタコマンドとデバッグ (info registers & xp)",
                question="QEMUモニタにおけるコマンドの役割として正しいものはどれか？",
                options=[
                    "info registers はOSのファイル一覧を表示し、xp は仮想記憶空間のみを検査する",
                    "info registers は全CPU汎用レジスタ・制御レジスタ・EFLAGSの状態をダンプし、xp は物理アドレスのメモリ内容を指定フォーマットでダンプする",
                    "xp はブレークポイントを設定し、c はプログラムを強制終了する",
                    "x コマンドは物理メモリダンプ専用で、xp コマンドは仮想メモリダンプ専用である",
                ],
                correct_index=1,
                explanation="QEMUモニタの info registers コマンドは RAX〜R15、RIP、RSP、RFLAGS、CR0〜CR4 などの CPU 内部レジスタ状態を一覧ダンプします。また xp (examine physical memory) は物理アドレスを指定して生メモリをダンプし、x は仮想アドレス（ページング有効時）をダンプします。",
            ),
            QuizQuestion(
                question_id=2,
                section="3.2",
                topic="x86-64 レジスタ階層と32ビットゼロ拡張則",
                question="x86-64 の 64ビット汎用レジスタ（RAX 等）に対する32ビットサブレジスタ（EAX）書き込み時の動作として正しいものはどれか？",
                options=[
                    "EAXに書き込むと、RAXの上位32ビットは変更されず以前の値がそのまま保持される",
                    "EAXに書き込むと、RAXの上位32ビットは自動的にすべてゼロクリア（ゼロ拡張）される",
                    "EAXに書き込むと、RAXの最上位ビットが符号拡張されて上位32ビットにコピーされる",
                    "EAXへの書き込みは64ビットモードでは禁止されており一般保護例外 (#GP) が発生する",
                ],
                correct_index=1,
                explanation="x86-64 アーキテクチャの基本仕様として、32ビットレジスタ操作（例: MOV EAX, 0x12345678）を実行すると、対応する64ビットレジスタ（RAX）の上位32ビットは常に自動的にゼロクリアされます。一方、16ビット（AX）や8ビット（AL/AH）への書き込み時は上位ビットは保持されます。",
            ),
            QuizQuestion(
                question_id=3,
                section="3.2",
                topic="x86-64 制御レジスタ (CR0, CR3, CR4)",
                question="x86-64 の制御レジスタ CR0 および CR3 の役割の組み合わせとして正しいものはどれか？",
                options=[
                    "CR0 は浮動小数点演算の結果を保持し、CR3 はタイマ割り込み周期を設定する",
                    "CR0 はプロテクトモード有効化 (PE) やページング有効化 (PG) などのCPU動作モードを制御し、CR3 は4階層ページテーブルの最上位 (PML4) 物理アドレスを保持する",
                    "CR0 はスタックポインタであり、CR3 は命令ポインタである",
                    "CR0 はカーネル引数レジスタであり、CR3 はシステムコール番号レジスタである",
                ],
                correct_index=1,
                explanation="CR0 (Control Register 0) はビット0の PE (Protection Enable) やビット31の PG (Paging Enable) などの基本動作モードを司ります。CR3 はページテーブルの基底物理アドレス（x86-64 Long Mode では PML4 テーブルの物理アドレス）を保持します。",
            ),
            QuizQuestion(
                question_id=4,
                section="3.3",
                topic="ブートローダからのカーネルロードとエントリポイントジャンプ",
                question="MikanOSブートローダ (day03a) がカーネルを実行する手順として正しいものはどれか？",
                options=[
                    "カーネルファイルをメモリ 0x100000 に読み込み、ExitBootServices() を呼んでブートサービスを終了させた後、エントリポイント関数ポインタへ引数なしでジャンプした",
                    "カーネルを読み込まずに UEFI ファームウェア内の組み込みカーネル関数を呼び出した",
                    "ExitBootServices() を呼ばずにカーネルへジャンプし、カーネル内から gBS->GetMemoryMap() を常時呼ぶ設計にした",
                    "カーネルは常に 0x7C00 の MBR 領域に配置してリアルモードでジャンプした",
                ],
                correct_index=0,
                explanation="MikanOSのブートローダはカーネルファイルを 0x100000 (1MiB) の物理メモリへ配置し、gBS->ExitBootServices() を呼び出してUEFI制御下から脱出した後、エントリポイントアドレスを関数ポインタ `typedef void (*KernelEntry)(void);` にキャストして実行権をカーネルへ渡しました。",
            ),
            QuizQuestion(
                question_id=5,
                section="3.3",
                topic="System V AMD64 ABI の Red Zone と -mno-red-zone",
                question="System V AMD64 ABI の Red Zone 仕様と、OSカーネル開発で -mno-red-zone が絶対必須である理由として正しいものはどれか？",
                options=[
                    "Red Zone は 512バイトのセクタであり、ディスク読み込みのバッファとして使われるためカーネルでは無効化する",
                    "Red Zone は RSP より下の 128バイト領域で、関数が RSP を下げずにスクラッチ領域として使ってよいが、割り込みや例外が発生するとCPUやハンドラによってこの領域が上書き破壊されてしまうため、カーネルコンパイル時は -mno-red-zone で無効化しなければならない",
                    "Red Zone はカーネルメモリのセキュリティ保護領域であり、有効化しないと Ring 0 特権が剥奪される",
                    "Red Zone はグラフィックス描画専用のVRAM予約領域であり、ブートローダでのみ使用できる",
                ],
                correct_index=1,
                explanation="System V AMD64 ABI ではスタックポインタ RSP の下 128バイトが Red Zone として定義され、リーフ関数などがスタック拡張命令を省略して一時変数置き場として使えます。しかし非同期割り込みが発生すると、CPUは現在の RSP の直下にレジスタやリターンアドレスを退避するため、Red Zone のデータが破壊されます。したがってカーネルコードは必ず -mno-red-zone でコンパイルする必要があります。",
            ),
            QuizQuestion(
                question_id=6,
                section="3.4",
                topic="UEFI GOP (Graphics Output Protocol) の構造とストライド",
                question="UEFI GOP (`EFI_GRAPHICS_OUTPUT_PROTOCOL`) における画面パラメータとストライド（PixelsPerScanLine）について正しいものはどれか？",
                options=[
                    "PixelsPerScanLine は常に HorizontalResolution と完全に同一であり、パディングは存在しない",
                    "ハードウェアの制約やメモリアライメント（例: 64/128バイト境界）により、走査線あたりのピクセル数 PixelsPerScanLine は HorizontalResolution 以上になることがあり、(x, y) のピクセルオフセットは `y * PixelsPerScanLine + x` で計算しなければならない",
                    "GOP フレームバッファはテキスト文字専用のバッファであり、RGBピクセル値の直接書き込みはできない",
                    "FrameBufferBase は必ず 0x000A0000 の VGA 領域に固定される",
                ],
                correct_index=1,
                explanation="GPUやディスプレイコントローラのアライメント要件（バス効率向上）により、1走査線のピクセル幅 PixelsPerScanLine (ppsl) は表示画面幅 HorizontalResolution (hr) よりも大きくなる（ppsl >= hr）場合があります。そのためピクセル (x, y) のバイトオフセットは `4 * (y * ppsl + x)` で算出する必要があります。",
            ),
            QuizQuestion(
                question_id=7,
                section="3.4",
                topic="GOP ピクセルフォーマット (RGB vs BGR)",
                question="MikanOS (day03b) におけるピクセルフォーマットと色値書き込みについて正しいものはどれか？",
                options=[
                    "ピクセルフォーマットに関わらず、メモリ順は常に Red, Green, Blue の順に 1バイトずつ配置される",
                    "PixelRedGreenBlueReserved8BitPerColor の場合、メモリの連続4バイトには低位アドレスから順に [R, G, B, Reserved] が並び、PixelBlueGreenRedReserved8BitPerColor の場合は [B, G, R, Reserved] の順に並ぶ",
                    "1ピクセルは 2バイト（16ビット）固定であり、各色5ビットで表現される",
                    "Reserved のバイトには常に 0xFF を書き込まなければハードウェアが画面描画を拒否する",
                ],
                correct_index=1,
                explanation="UEFI GOP の 32-bit ピクセルフォーマットでは、RGB形式の場合はバイト [0]=R, [1]=G, [2]=B, [3]=Reserved となり、BGR形式の場合はバイト [0]=B, [1]=G, [2]=R, [3]=Reserved となります。白を描くには R=G=B=255 (0xFF) を書き込みます。",
            ),
            QuizQuestion(
                question_id=8,
                section="3.5",
                topic="カーネルへの引数引き渡しと System V AMD64 ABI 呼び出し規約",
                question="ブートローダからカーネル Main(fb_base, fb_size) へ引数を引き渡す際、System V AMD64 ABI において第1引数・第2引数が格納されるレジスタの組み合わせはどれか？",
                options=[
                    "第1引数: RCX, 第2引数: RDX",
                    "第1引数: RDI, 第2引数: RSI",
                    "第1引数: RAX, 第2引数: RBX",
                    "第1引数: スタックトップ [RSP], 第2引数: [RSP+8]",
                ],
                correct_index=1,
                explanation="Linux や MikanOS カーネルが採用する System V AMD64 ABI では、整数・ポインタ引数は第1引数から順に RDI, RSI, RDX, RCX, R8, R9 のレジスタ経由で渡されます（Microsoft x64 ABI では RCX, RDX, R8, R9）。ブートローダからカーネルへの引数は RDI (fb_base) と RSI (fb_size) に格納されます。",
            ),
            QuizQuestion(
                question_id=9,
                section="3.6",
                topic="UEFI ステータスコードと C++ reinterpret_cast",
                question="UEFI API の戻り値 EFI_STATUS のエラー判定マクロと、ポインタを安全に整数アドレスに再解釈する C++ キャスト演算子として正しいものはどれか？",
                options=[
                    "EFI_STATUS は負の値が成功を示し、キャストには dynamic_cast を用いる",
                    "EFI_STATUS は最上位ビット（ビット63: 0x8000000000000000）が 1 の場合にエラーを表すため EFI_ERROR(status) マクロで判定し、ポインタと整数の相互変換には reinterpret_cast を用いる",
                    "EFI_STATUS のエラー判定には常に status == 0 を比較し、キャストには const_cast のみを用いる",
                    "EFI_ERROR(status) は status == EFI_NOT_READY の場合のみ真を返す",
                ],
                correct_index=1,
                explanation="EFI_STATUS は最上位ビットが 1 であればエラー、0 であれば成功または警告を表します。EFI_ERROR(status) マクロは最上位ビットが立っているかを判定します。また、型体系を超えたポインタと整数型（uintptr_t や uint64_t）のビットパターン再解釈には C++ の reinterpret_cast を使用します。",
            ),
            QuizQuestion(
                question_id=10,
                section="3.7",
                topic="ポインタのアセンブリ展開、SIBアドレッシング、volatile 修飾",
                question="ポインタ間接参照のアセンブリ言語展開およびハードウェア VRAM/MMIO 操作における volatile の役割として正しいものはどれか？",
                options=[
                    "SIB アドレッシングは [base + index] の2項目しか指定できず、volatile はコンパイル速度を上げるための修飾子である",
                    "ポインタ演算 `base[i]` は CPU の SIB (Scale-Index-Base) アドレッシング `[base + index * scale + disp]` に直接対応し、ハードウェア MMIO/VRAM へのポインタにはコンパイラによる最適化（不要な書き込みの削除やSIMD統合）を防ぐため volatile を付与する必要がある",
                    "volatile を付けると CPU キャッシュが無効化され、コンパイル時にすべて NOP 命令に置換される",
                    "最適化オプション -O2 を指定すると volatile の指定は完全に無視される",
                ],
                correct_index=1,
                explanation="C言語の配列/ポインタアクセスは x86-64 の SIB (Scale-Index-Base) アドレッシング命令（例: `MOV [RDI + RAX*4], EDX`）に1命令でマップされます。また MMIO や VRAM への書き込みは「外部ハードウェアへの副作用（Observable side effect）」であるため、volatile を指定することでコンパイラによる不要な書き込みの省略（デッドストア削除）やループの改変を防ぎます。",
            ),
        ]

    def evaluate(self, answers: dict[int, int]) -> dict[str, Any]:
        """Evaluate submitted quiz answers, compute score and mastery percentage."""
        total_questions = len(self.questions)
        correct_count = 0
        details = []

        section_scores: dict[str, list[bool]] = {}
        for q in self.questions:
            section_scores.setdefault(q.section, [])

        for q in self.questions:
            user_ans = answers.get(q.question_id)
            is_correct = (user_ans == q.correct_index)
            if is_correct:
                correct_count += 1
            section_scores[q.section].append(is_correct)

            details.append({
                "question_id": q.question_id,
                "section": q.section,
                "topic": q.topic,
                "user_answer": user_ans,
                "correct_index": q.correct_index,
                "is_correct": is_correct,
                "explanation": q.explanation,
            })

        percentage = (correct_count / total_questions) * 100.0 if total_questions > 0 else 0.0
        mastery_rates: dict[str, float] = {}
        for sec, results in sorted(section_scores.items()):
            rate = (sum(results) / len(results)) * 100.0 if results else 0.0
            mastery_rates[sec] = rate

        return {
            "total_questions": total_questions,
            "correct_count": correct_count,
            "percentage": percentage,
            "details": details,
            "mastery_rates": mastery_rates,
            "passed": percentage >= 80.0,
        }


class Chapter3TheoryValidator:
    """Validation helper for Chapter 3 systems theory, x86-64 architecture, GOP math, and ABI."""

    @staticmethod
    def validate_qemu_monitor_commands(commands: dict[str, str]) -> tuple[bool, str]:
        """Validate QEMU monitor command mappings."""
        info_reg = commands.get("info registers", "").lower()
        xp_cmd = commands.get("xp", "").lower()
        x_cmd = commands.get("x", "").lower()

        errors = []
        if not ("register" in info_reg or "レジスタ" in info_reg or "dump" in info_reg or "ダンプ" in info_reg):
            errors.append("info registers: レジスタ状態のダンプであるべきです")
        if not ("physical" in xp_cmd or "物理" in xp_cmd):
            errors.append("xp: 物理メモリダンプ (examine physical) であるべきです")
        if not ("virtual" in x_cmd or "仮想" in x_cmd):
            errors.append("x: 仮想メモリダンプ (examine virtual) であるべきです")

        if not errors:
            return True, "正解！QEMUモニタの主要検査コマンドの役割を正確に同定しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_register_zero_extension(sim_dict: dict[str, int]) -> tuple[bool, str]:
        """Validate 32-bit write zero-extension behavior into 64-bit RAX."""
        initial_rax = sim_dict.get("initial_rax")
        write_eax = sim_dict.get("write_eax")
        final_rax = sim_dict.get("final_rax")

        errors = []
        if initial_rax is None or write_eax is None or final_rax is None:
            return False, "引数辞書に必要なキー (initial_rax, write_eax, final_rax) が不足しています。"

        expected_final = write_eax & 0xFFFFFFFF
        if final_rax != expected_final:
            errors.append(f"final_rax: 期待値 {hex(expected_final)}, 入力値 {hex(final_rax)}")

        if not errors:
            return True, "正解！32ビットサブレジスタ書き込み時の上位32ビットゼロクリア（ゼロ拡張）挙動を正確に計算しました。"
        return False, f"不正解です。{'; '.join(errors)}"

    @staticmethod
    def validate_red_zone_rules(rules: dict[str, any]) -> tuple[bool, str]:
        """Validate System V AMD64 ABI Red Zone properties."""
        size_bytes = rules.get("size_bytes")
        direction = str(rules.get("direction", "")).lower()
        compiler_flag = rules.get("compiler_flag")
        safe_for_interrupts = rules.get("safe_for_interrupts")

        errors = []
        if size_bytes != 128:
            errors.append(f"size_bytes: 期待値 128バイト, 入力値 {size_bytes}")
        if not ("below" in direction or "下" in direction or "負" in direction or "-" in direction):
            errors.append("direction: RSP より下（負のオフセット）であるべきです")
        if compiler_flag != "-mno-red-zone":
            errors.append(f"compiler_flag: 期待値 -mno-red-zone, 入力値 {compiler_flag}")
        if safe_for_interrupts is not False:
            errors.append("safe_for_interrupts: 割り込みハンドラによる上書き破壊の危険があるため False であるべきです")

        if not errors:
            return True, "正解！Red Zone の仕様とカーネルにおける -mno-red-zone の絶対必要性を正確に理解しています。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_gop_framebuffer_math(math_dict: dict[str, int]) -> tuple[bool, str]:
        """Validate GOP frame buffer offset and total size calculations."""
        h_res = math_dict.get("h_res", 800)
        v_res = math_dict.get("v_res", 600)
        ppsl = math_dict.get("ppsl", 832)
        bytes_per_pixel = math_dict.get("bytes_per_pixel", 4)
        target_x = math_dict.get("target_x", 100)
        target_y = math_dict.get("target_y", 50)
        pixel_offset = math_dict.get("pixel_byte_offset")
        total_fb_size = math_dict.get("total_fb_size")

        expected_offset = bytes_per_pixel * (target_y * ppsl + target_x)
        expected_size = bytes_per_pixel * ppsl * v_res

        errors = []
        if pixel_offset != expected_offset:
            errors.append(f"pixel_byte_offset: 期待値 {expected_offset}バイト, 入力値 {pixel_offset}")
        if total_fb_size != expected_size:
            errors.append(f"total_fb_size: 期待値 {expected_size}バイト, 入力値 {total_fb_size}")

        if not errors:
            return True, "正解！PixelsPerScanLine（ストライド）を考慮したフレームバッファアドレス数理を完璧に計算しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_system_v_abi_args(arg_regs: list[str]) -> tuple[bool, str]:
        """Validate 64-bit integer/pointer register passing order under System V AMD64 ABI."""
        regs_clean = [r.strip().upper() for r in arg_regs]
        expected_regs = ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
        if regs_clean == expected_regs:
            return True, "正解！System V AMD64 ABI の第1〜第6引数レジスタ順序（RDI, RSI, RDX, RCX, R8, R9）を正確に同定しました。"
        return False, f"不正解です。期待値: {expected_regs}, 入力値: {regs_clean}"

    @staticmethod
    def validate_pointer_assembly_sib(sib_dict: dict[str, int]) -> tuple[bool, str]:
        """Validate x86-64 SIB (Scale-Index-Base) addressing arithmetic."""
        base = sib_dict.get("base")
        index = sib_dict.get("index")
        scale = sib_dict.get("scale")
        disp = sib_dict.get("disp", 0)
        effective_address = sib_dict.get("effective_address")

        if any(v is None for v in [base, index, scale, effective_address]):
            return False, "引数辞書に必要なキー (base, index, scale, effective_address) が不足しています。"

        expected_ea = base + (index * scale) + disp
        if effective_address != expected_ea:
            return False, f"不正解です。期待値 {hex(expected_ea)}, 入力値 {hex(effective_address)}"
        return True, "正解！SIB (Scale-Index-Base) アドレッシングの実効アドレス計算を正確に算出しました。"


def plot_chapter3_exercises_summary(output_path: Path | str) -> Path:
    """Generate high-DPI summary diagram showing Chapter 3 exercise structure and learning taxonomy."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_facecolor("#FAFAFA")

    exercises = [
        {"id": "Ex 1", "sec": "3.1 / 3.2", "title": "QEMUモニタ & x86-64 レジスタ階層", "desc": "info registers / xp による物理メモリ検査、32bitゼロ拡張則、CR0/CR3/CR4 制御レジスタ", "color": "#1E88E5", "y": 0.84},
        {"id": "Ex 2", "sec": "3.3", "title": "カーネルエントリ & Red Zone の罠", "desc": "0x100000 への配置、ExitBootServices、Red Zone (128B) と -mno-red-zone の絶対必要性", "color": "#43A047", "y": 0.67},
        {"id": "Ex 3", "sec": "3.4", "title": "UEFI GOP & フレームバッファ数理", "desc": "PixelsPerScanLine ストライド計算、RGB vs BGR 32bit ピクセルフォーマット展開", "color": "#FB8C00", "y": 0.50},
        {"id": "Ex 4", "sec": "3.5 / 3.6", "title": "引数引き渡し ABI & 型安全エラー処理", "desc": "System V AMD64 ABI (RDI, RSI)、EFI_STATUS 最上位ビット判定、reinterpret_cast", "color": "#8E24AA", "y": 0.33},
        {"id": "Ex 5", "sec": "3.7 / 総合", "title": "アセンブリ展開・volatile & 総合テスト", "desc": "SIB アドレッシング [base + index*scale + disp]、MMIO volatile、10問クイズ & レーダー", "color": "#D81B60", "y": 0.16},
    ]

    for ex in exercises:
        y = ex["y"]
        rect = patches.FancyBboxPatch(
            (0.05, y - 0.06), 0.90, 0.12,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="white", edgecolor=ex["color"], linewidth=2, zorder=2
        )
        ax.add_patch(rect)

        badge = patches.FancyBboxPatch(
            (0.07, y - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=ex["color"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(0.13, y, ex["id"], color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

        ax.text(0.22, y + 0.02, f"[{ex['sec']}] {ex['title']}", color="#212121", fontsize=12, fontweight="bold", va="center")
        ax.text(0.22, y - 0.025, ex["desc"], color="#555555", fontsize=9.5, va="center")

        circle = plt.Circle((0.91, y), 0.025, facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(0.91, y, "✓", color="#2E7D32", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.05, 0.97)
    ax.axis("off")
    ax.set_title("MikanOS Study — Chapter 3 Exercises Curriculum & Assessment Map",
                 fontsize=14, fontweight="bold", pad=20, color="#1A237E")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_chapter3_mastery_radar(output_path: Path | str, mastery_rates: dict[str, float] | None = None) -> Path:
    """Generate high-DPI 7-axis radar chart showing Chapter 3 concept mastery across all 7 sections."""
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        "3.1 QEMUモニタ\n(info reg & xp Dump)",
        "3.2 x86-64 レジスタ\n(Hierarchy & CR0-4)",
        "3.3 カーネル起動\n(Entry & Red Zone)",
        "3.4 GOP 描画\n(Stride & Pixel Format)",
        "3.5 カーネル引数\n(ABI: RDI/RSI & Stripe)",
        "3.6 エラー処理\n(EFI_STATUS & Casts)",
        "3.7 ポインタとASM\n(SIB & volatile MMIO)",
    ]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    if mastery_rates is None:
        values = [100.0] * num_vars
    else:
        values = [
            mastery_rates.get("3.1", 100.0),
            mastery_rates.get("3.2", 100.0),
            mastery_rates.get("3.3", 100.0),
            mastery_rates.get("3.4", 100.0),
            mastery_rates.get("3.5", 100.0),
            mastery_rates.get("3.6", 100.0),
            mastery_rates.get("3.7", 100.0),
        ]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FCFCFD")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color="#1A237E", size=9.5, fontweight="bold")

    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#757575", size=8)
    plt.ylim(0, 105)

    ax.plot(angles, values, color="#1E88E5", linewidth=2.5, linestyle="solid", zorder=3)
    ax.fill(angles, values, color="#90CAF9", alpha=0.35, zorder=2)

    for angle, val in zip(angles[:-1], values[:-1]):
        ax.scatter([angle], [val], color="#0D47A1", s=45, zorder=4)

    plt.title("Chapter 3 Systems Architecture Mastery Profile",
              size=14, fontweight="bold", color="#0D47A1", y=1.12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# Chapter 4 Section 4.1: Make & Bare-metal Build Automation Simulation
# ==============================================================================

@dataclass
class MakeRule:
    """Represents a single rule in a Makefile with target, prerequisites, and recipe."""
    target: str
    prerequisites: list[str] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)
    is_pattern: bool = False

    def expand_recipe(self, target: str, prerequisites: list[str],
                      newer_prerequisites: list[str] | None = None,
                      stem: str = "") -> list[str]:
        """Expand automatic variables ($@, $<, $^, $?, $*) in recipes."""
        newer = newer_prerequisites if newer_prerequisites is not None else prerequisites
        first_prereq = prerequisites[0] if prerequisites else ""
        # Deduplicate while preserving order for $^
        all_prereqs = " ".join(dict.fromkeys(prerequisites))
        newer_str = " ".join(dict.fromkeys(newer))

        expanded = []
        for line in self.recipes:
            cmd = line.replace("$@", target)
            cmd = cmd.replace("$<", first_prereq)
            cmd = cmd.replace("$^", all_prereqs)
            cmd = cmd.replace("$?", newer_str)
            cmd = cmd.replace("$*", stem)
            expanded.append(cmd)
        return expanded


class MakeDependencyGraph:
    """Simulates GNU Make dependency graph evaluation, topological sorting,

    timestamp-based rebuild determination, and recipe execution.
    """

    def __init__(self) -> None:
        self.rules: dict[str, MakeRule] = {}
        self.pattern_rules: list[MakeRule] = []
        self.phony_targets: set[str] = set()
        self.timestamps: dict[str, float] = {}
        self.virtual_files: dict[str, str] = {}
        self.execution_log: list[dict[str, Any]] = []

    def add_rule(self, target: str, prerequisites: list[str], recipes: list[str]) -> None:
        """Add an explicit rule to the Makefile."""
        rule = MakeRule(target=target, prerequisites=prerequisites, recipes=recipes, is_pattern=False)
        self.rules[target] = rule

    def add_pattern_rule(self, target_pattern: str, prereq_pattern: str, recipes: list[str]) -> None:
        """Add a pattern rule (e.g. %.o: %.cpp) to the Makefile."""
        rule = MakeRule(target=target_pattern, prerequisites=[prereq_pattern], recipes=recipes, is_pattern=True)
        self.pattern_rules.append(rule)

    def mark_phony(self, *targets: str) -> None:
        """Register phony targets (.PHONY)."""
        for t in targets:
            self.phony_targets.add(t)

    def set_file(self, name: str, mtime: float, content: str = "") -> None:
        """Set or update a virtual file's timestamp and content."""
        self.timestamps[name] = mtime
        self.virtual_files[name] = content

    def remove_file(self, name: str) -> None:
        """Remove a virtual file from the environment."""
        self.timestamps.pop(name, None)
        self.virtual_files.pop(name, None)

    def file_exists(self, name: str) -> bool:
        """Check if a file exists in the virtual filesystem."""
        return name in self.timestamps

    def get_mtime(self, name: str) -> float:
        """Get timestamp of a file; returns 0.0 if not exists."""
        return self.timestamps.get(name, 0.0)

    def _resolve_rule(self, target: str) -> tuple[MakeRule | None, list[str], str]:
        """Find matching rule for target (explicit or via pattern rule)."""
        if target in self.rules:
            return self.rules[target], self.rules[target].prerequisites, ""

        # Check pattern rules
        for prule in self.pattern_rules:
            # Pattern format e.g. %.o
            if prule.target.startswith("%"):
                suffix = prule.target[1:]
                if target.endswith(suffix):
                    stem = target[:-len(suffix)]
                    prereqs = [p.replace("%", stem) for p in prule.prerequisites]
                    return prule, prereqs, stem
        return None, [], ""

    def get_dependencies_dag(self, target: str) -> dict[str, list[str]]:
        """Construct the dependency adjacency list starting from target."""
        adj: dict[str, list[str]] = {}
        visited: set[str] = set()

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            rule, prereqs, _ = self._resolve_rule(node)
            adj[node] = prereqs
            for p in prereqs:
                dfs(p)

        dfs(target)
        return adj

    def topological_sort(self, target: str) -> list[str]:
        """Return topological order of build evaluation (leaves first, target last).

        Raises ValueError on cyclic dependencies.
        """
        adj = self.get_dependencies_dag(target)
        visited: dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited
        order: list[str] = []

        def visit(node: str) -> None:
            state = visited.get(node, 0)
            if state == 1:
                raise ValueError(f"Cyclic dependency detected involving target: {node}")
            if state == 2:
                return
            visited[node] = 1
            for child in adj.get(node, []):
                visit(child)
            visited[node] = 2
            order.append(node)

        visit(target)
        return order

    def needs_rebuild(self, target: str) -> tuple[bool, str, list[str]]:
        """Determine if target needs rebuilding based on timestamps and .PHONY.

        Returns (needs_rebuild, reason, newer_prerequisites).
        """
        if target in self.phony_targets:
            return True, "Phony target always runs", []

        if not self.file_exists(target):
            return True, f"Target '{target}' does not exist", []

        target_mtime = self.get_mtime(target)
        rule, prereqs, _ = self._resolve_rule(target)

        newer_prereqs = [p for p in prereqs if self.get_mtime(p) > target_mtime]
        if newer_prereqs:
            return True, f"Prerequisites newer than target: {newer_prereqs}", newer_prereqs

        return False, "Target is up to date", []

    def build(self, target: str, current_time: float | None = None) -> list[str]:
        """Execute build sequence for target, executing recipes when necessary.

        Returns list of executed commands.
        """
        order = self.topological_sort(target)
        executed_commands: list[str] = []
        clock = current_time if current_time is not None else 1000.0

        for node in order:
            rule, prereqs, stem = self._resolve_rule(node)
            if not rule:
                # Leaf source file (no rule needed to produce it)
                if not self.file_exists(node) and node not in self.phony_targets:
                    raise FileNotFoundError(f"No rule to make target '{node}'")
                continue

            rebuild, reason, newer = self.needs_rebuild(node)
            if rebuild:
                clock += 1.0
                expanded_cmds = rule.expand_recipe(node, prereqs, newer, stem)
                executed_commands.extend(expanded_cmds)
                # Update timestamp on successful recipe execution
                if node not in self.phony_targets:
                    self.set_file(node, clock, f"compiled binary of {node}")
                self.execution_log.append({
                    "target": node,
                    "reason": reason,
                    "commands": expanded_cmds,
                    "new_mtime": clock
                })

        return executed_commands


@dataclass(frozen=True)
class CompilerFlag:
    """Details of a C++ compiler / linker flag used in MikanOS kernel build."""
    flag: str
    category: str
    purpose: str
    baremetal_necessity: str
    omission_risk: str


class CompilerFlagAnalyzer:
    """Provides detailed architectural inspection of MikanOS Day 04a compiler/linker flags."""

    FLAGS: list[CompilerFlag] = [
        CompilerFlag(
            flag="-O2",
            category="Optimization",
            purpose="最適化レベル2（命令スケジューリング、レジスタ割り当て、インライン展開、SIMDベクトル化）",
            baremetal_necessity="フレームバッファ描画やメモリ転送の性能を向上させ、不要なスタック読み書きを削減",
            omission_risk="極端な描画遅延や実行バイナリサイズの肥大化を招く"
        ),
        CompilerFlag(
            flag="-Wall",
            category="Diagnostics",
            purpose="主要なコンパイラ警告（型不一致、未初期化変数、未使用シンボル等）の全網羅的有効化",
            baremetal_necessity="OSカーネルにはデバッガが初動で存在しないため、未然に潜在バグを検出する最重要砦",
            omission_risk="未初期化ポインタ参照などのサイレントクラッシュを招く"
        ),
        CompilerFlag(
            flag="-g",
            category="Diagnostics",
            purpose="DWARF形式のデバッグシンボル情報をバイナリ内に埋め込む",
            baremetal_necessity="QEMU + GDB によるリモートデバッグ時に関数名や行番号、変数構造を直接可視化",
            omission_risk="逆アセンブル時に生のアドレスしか見えずデバッグ効率が激減する"
        ),
        CompilerFlag(
            flag="--target=x86_64-elf",
            category="Target Architecture",
            purpose="ホストOS（Linux/macOS）非依存のベアメタル x86-64 ELF64 ABI をクロスコンパイルターゲットに指定",
            baremetal_necessity="ホスト環境固有の共有ライブラリ依存や動的ローダ参照（.interp）を排除",
            omission_risk="ホストOS向けのLinuxバイナリが生成され、自作ブートローダで起動不能となる"
        ),
        CompilerFlag(
            flag="-ffreestanding",
            category="Runtime Environment",
            purpose="OSや標準Cライブラリが存在しない自立（フリースタンディング）実行環境であることをコンパイラに宣言",
            baremetal_necessity="main() を特別な関数とみなさず、暗黙のスタートアップコード（crt0.o）や標準ライブラリ呼び出しを禁止",
            omission_risk="存在しない libc シンボル（memset, exit 等）の暗黙参照でリンクエラーが発生する"
        ),
        CompilerFlag(
            flag="-mno-red-zone",
            category="Target Architecture",
            purpose="System V AMD64 ABI 規定のスタックポインタ下 128 バイトの一時退避領域（Red Zone）の利用を禁止",
            baremetal_necessity="非同期割り込み（Interrupt）発生時にCPUハードウェアがRSP下にレジスタをプッシュしてRed Zoneを破壊する事故を防ぐ",
            omission_risk="第7章以降で割り込みを有効化した瞬間にスタック破壊でカーネルが原因不明のパニックを起こす"
        ),
        CompilerFlag(
            flag="-fno-exceptions",
            category="C++ Feature Suppression",
            purpose="C++ 例外処理（try, catch, throw）およびスタック巻き戻し（Stack Unwinding）ランタイムを無効化",
            baremetal_necessity="例外ハンドラテーブル（.eh_frame）や libsupc++ / libc++abi の複雑なランタイムがベアメタルに存在しない",
            omission_risk="リンク時に __gxx_personality_v0 等の巨大ランタイムシンボル未解決エラーが発生する"
        ),
        CompilerFlag(
            flag="-fno-rtti",
            category="C++ Feature Suppression",
            purpose="ランタイム型情報（RTTI: dynamic_cast, typeid）および型情報記述子構造体を無効化",
            baremetal_necessity="型情報メタデータや vtable 内の RTTI 記述ポインタを省き、純粋な仮想関数ポインタのみに絞る",
            omission_risk="vtable から不要なランタイム型メタデータへの参照が発生しリンク不能となる"
        ),
        CompilerFlag(
            flag="-std=c++17",
            category="Language Standard",
            purpose="ISO C++17 言語規格に準拠した最新機能（構造化束縛、constexpr if、std::byte 等）を有効化",
            baremetal_necessity="高機能かつ型安全なモダンC++を活用した堅牢なカーネル設計を可能にする",
            omission_risk="古いC++規格に縛られ、型の安全機構やコード記述の表現力が著しく制約される"
        ),
        CompilerFlag(
            flag="--entry KernelMain",
            category="Linker Specification",
            purpose="ELF 実行可能ファイルのエントリポイント（実行開始アドレス）シンボルを KernelMain に指定",
            baremetal_necessity="ブートローダ（MikanLoaderPkg）が ELF ヘッダ e_entry を読み取って直接ジャンプ可能にする",
            omission_risk="リンカデフォルトの _start が期待され、リンク失敗または不正アドレスへジャンプする"
        ),
        CompilerFlag(
            flag="-z norelro",
            category="Linker Specification",
            purpose="RELRO（Relocation Read-Only: 起動後再配置セクションの読み取り専用保護）セグメントの生成を無効化",
            baremetal_necessity="動的ローダのいないベアメタル環境で ELF PT_LOAD セグメント構造を極限までシンプルに保つ",
            omission_risk="GNU_RELRO ヘッダが生成され、自作ブートローダの ELF セグメントローダが複雑化・破損する"
        ),
        CompilerFlag(
            flag="--image-base 0x100000",
            category="Linker Specification",
            purpose="カーネルプログラム全体の仮想/物理配置基準アドレスを 1MiB (0x100000) に固定",
            baremetal_necessity="レガシーBIOS/UEFIの予約メモリ（0x0〜0xFFFFF）を避け、空きConventional RAMの先頭に正確に配置",
            omission_risk="デフォルトの0x200000や0x400000にリンクされ、ブートローダが配置した1MiBと不整合を起こして即死する"
        ),
        CompilerFlag(
            flag="--static",
            category="Linker Specification",
            purpose="すべての動的ライブラリ参照を排除し、完全な自己完結型静的バイナリを生成",
            baremetal_necessity="ベアメタル環境には共有ライブラリ（.so）をロードする動的リンカが存在しない",
            omission_risk="動的リンクシンボルが残り、ロード時に参照未解決で実行不能となる"
        ),
    ]

    @classmethod
    def get_flag(cls, flag_name: str) -> CompilerFlag:
        """Find flag by its exact name."""
        for f in cls.FLAGS:
            if f.flag == flag_name:
                return f
        raise KeyError(f"Compiler flag '{flag_name}' not found.")

    @classmethod
    def get_flags_by_category(cls) -> dict[str, list[CompilerFlag]]:
        """Group flags by category."""
        res: dict[str, list[CompilerFlag]] = {}
        for f in cls.FLAGS:
            res.setdefault(f.category, []).append(f)
        return res


def plot_fig4_01_make_dag_and_timestamp_evaluation(output_path: Path | str) -> Path:
    """Generate high-DPI architecture diagram showing Makefile Dependency DAG,

    timestamp-based incremental build evaluation logic, and automatic variables mapping.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 8.5), facecolor="#FAFAFA")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.2, 0.9],
                          wspace=0.25, hspace=0.30)

    # Subplot 1: Dependency DAG Graph
    ax_dag = fig.add_subplot(gs[0, 0])
    ax_dag.set_facecolor("#FFFFFF")
    ax_dag.set_title("(A) MikanOS kernel/Makefile Dependency DAG (有向非巡回グラフ)",
                     fontsize=12, fontweight="bold", color="#1565C0", pad=12)

    # Node definitions: (x, y, label, type, box_color, border_color)
    nodes = [
        ("all", 0.15, 0.85, ".PHONY: all\n(擬似ターゲット)", "#E8F5E9", "#2E7D32"),
        ("kernel.elf", 0.50, 0.85, "Target: kernel.elf\n[0x100000 静的ELF64]", "#E3F2FD", "#1565C0"),
        ("main.o", 0.35, 0.45, "Target: main.o\n[ELF64 Object]", "#FFF3E0", "#E65100"),
        ("Makefile", 0.70, 0.45, "Prereq: Makefile\n[フラグ変更追従]", "#EDE7F6", "#4A148C"),
        ("main.cpp", 0.20, 0.10, "Source: main.cpp\n[C++17 カーネル実装]", "#FCE4EC", "#C2185B"),
    ]

    for name, x, y, label, bg, border in nodes:
        box = FancyBboxPatch((x - 0.11, y - 0.08), 0.22, 0.16,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=bg, edgecolor=border, linewidth=2, zorder=3)
        ax_dag.add_patch(box)
        ax_dag.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold",
                    color=border, zorder=4)

    # Directed edges (dependencies): from parent to required child
    edges = [
        ((0.26, 0.85), (0.39, 0.85), "all -> kernel.elf"),
        ((0.46, 0.77), (0.38, 0.53), "ld.lld $^ -o $@"),
        ((0.54, 0.77), (0.66, 0.53), "rebuild if Makefile changes"),
        ((0.32, 0.37), (0.23, 0.18), "clang++ -c $<"),
        ((0.40, 0.37), (0.64, 0.43), "recompile if flags change"),
    ]

    for (x1, y1), (x2, y2), edge_label in edges:
        ax_dag.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                                        color="#37474F", lw=1.8, ls="solid"), zorder=2)

    ax_dag.set_xlim(0.0, 0.90)
    ax_dag.set_ylim(0.0, 1.0)
    ax_dag.axis("off")

    # Subplot 2: Timestamp Evaluation Decision Tree
    ax_tree = fig.add_subplot(gs[0, 1])
    ax_tree.set_facecolor("#FFFFFF")
    ax_tree.set_title("(B) タイムスタンプ比較によるインクリメンタルビルド判定数理",
                      fontsize=12, fontweight="bold", color="#2E7D32", pad=12)

    flow_boxes = [
        (0.5, 0.90, "ビルド対象ターゲット T の評価開始", "#E8EAF6", "#283593"),
        (0.5, 0.72, "ターゲット T は存在するか？\n(file_exists(T) == true)", "#FFF8E1", "#F57F17"),
        (0.18, 0.50, "【T 不存在】\n即座にレシピ実行\n(Full Build)", "#FFEBEE", "#C62828"),
        (0.72, 0.50, "依存ファイル P の最新時刻チェック\nmax(mtime(P)) > mtime(T) ?", "#E0F2F1", "#00695C"),
        (0.48, 0.22, "【P が新しい】\n依存先が更新されたため\nレシピ実行 (Incremental Build)", "#FFF3E0", "#E65100"),
        (0.85, 0.22, "【T は最新】\n'T is up to date'\nレシピ実行スキップ！", "#E8F5E9", "#2E7D32"),
    ]

    for x, y, text, bg, border in flow_boxes:
        box = FancyBboxPatch((x - 0.15, y - 0.07), 0.30, 0.14,
                             boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=bg, edgecolor=border, linewidth=1.8, zorder=3)
        ax_tree.add_patch(box)
        ax_tree.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color=border, zorder=4)

    # Decision tree connectors
    tree_arrows = [
        ((0.5, 0.83), (0.5, 0.79), ""),
        ((0.35, 0.72), (0.18, 0.57), "No (不存在)"),
        ((0.65, 0.72), (0.72, 0.57), "Yes (存在)"),
        ((0.65, 0.43), (0.52, 0.29), "Yes (更新あり)"),
        ((0.79, 0.43), (0.85, 0.29), "No (変化なし)"),
    ]
    for (x1, y1), (x2, y2), lbl in tree_arrows:
        ax_tree.annotate(lbl, xy=(x2, y2), xytext=(x1, y1),
                         ha="center", va="center", fontsize=8, fontweight="bold", color="#455A64",
                         arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5",
                                         color="#455A64", lw=1.5), zorder=2)

    ax_tree.set_xlim(0.0, 1.0)
    ax_tree.set_ylim(0.10, 1.0)
    ax_tree.axis("off")

    # Subplot 3: Automatic Variables Mapping Table
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.set_facecolor("#FFFFFF")
    ax_tbl.set_title("(C) Make 自動変数 (Automatic Variables) の展開仕様",
                     fontsize=12, fontweight="bold", color="#4A148C", pad=10)

    table_data = [
        ["変数", "名称", "Day 04a での展開例", "OS自作での実務的役割"],
        ["$@", "ターゲット名 (Target)", "kernel.elf / main.o", "出力ファイル名の明示 (-o $@)"],
        ["$<", "第1必須条件 (First Prereq)", "main.cpp", "単一ソースのコンパイル (-c $<)"],
        ["$^", "全必須条件 (All Prereqs)", "main.o Makefile", "全オブジェクトのリンカ渡し"],
        ["$?", "更新された必須条件", "main.o (修正時)", "差分再生成・アーカイブ更新"],
        ["$*", "パターンの語幹 (Stem)", "main (from %.o: %.cpp)", "同名ヘッダや中間ファイルの特定"],
    ]

    col_widths = [0.10, 0.28, 0.26, 0.36]
    y_pos = 0.88
    for row_idx, row in enumerate(table_data):
        x_curr = 0.02
        is_hdr = (row_idx == 0)
        bg = "#EDE7F6" if is_hdr else ("#F5F5F5" if row_idx % 2 == 1 else "#FFFFFF")
        fg = "#311B92" if is_hdr else "#212121"
        weight = "bold" if is_hdr else "normal"

        for col_idx, cell in enumerate(row):
            w = col_widths[col_idx]
            rect = Rectangle((x_curr, y_pos - 0.13), w - 0.01, 0.12,
                             facecolor=bg, edgecolor="#BDBDBD", linewidth=0.8, zorder=2)
            ax_tbl.add_patch(rect)
            ax_tbl.text(x_curr + (w - 0.01) / 2, y_pos - 0.07, cell,
                        ha="center", va="center", fontsize=8.5, fontweight=weight, color=fg, zorder=3)
            x_curr += w
        y_pos -= 0.14

    ax_tbl.set_xlim(0.0, 1.0)
    ax_tbl.set_ylim(0.0, 1.0)
    ax_tbl.axis("off")

    # Subplot 4: Phony Target & Pattern Rule Mechanism
    ax_mech = fig.add_subplot(gs[1, 1])
    ax_mech.set_facecolor("#FFFFFF")
    ax_mech.set_title("(D) 擬似ターゲット (.PHONY) とパターンルールの役割",
                      fontsize=12, fontweight="bold", color="#C2185B", pad=10)

    mech_text = (
        "【1. .PHONY: all clean の絶対的必要性】\n"
        "・もしカレントディレクトリに 'clean' という名前のファイルが誤って存在した場合、\n"
        "  必須条件のない clean は『最新』とみなされ、rm -rf *.o が二度と実行されなくなる！\n"
        "・.PHONY を宣言することで、ファイルシステムの同名実体ファイルを無視し、\n"
        "  『常に無条件でレシピを実行すべきタスクコマンド』として Make に保証させる。\n\n"
        "【2. パターンルール %.o: %.cpp Makefile の実務的意義】\n"
        "・今後カーネルに graphics.cpp, font.cpp, pci.cpp が増えても Makefile の追記は\n"
        "  OBJS += graphics.o の1行のみで完結する拡張性を確保。\n"
        "・必須条件に 'Makefile' を含めることで、コンパイルフラグ（CXXFLAGS）を変更した際、\n"
        "  全ソースが自動で再コンパイルされ、古いフラグのバイナリが残る致命的事故を防ぐ！"
    )

    box_mech = FancyBboxPatch((0.02, 0.05), 0.96, 0.88,
                              boxstyle="round,pad=0.03,rounding_size=0.03",
                              facecolor="#FCE4EC", edgecolor="#C2185B", linewidth=1.5, zorder=2)
    ax_mech.add_patch(box_mech)
    ax_mech.text(0.05, 0.49, mech_text, ha="left", va="center", fontsize=8.2,
                 color="#880E4F", linespacing=1.45, zorder=3)

    ax_mech.set_xlim(0.0, 1.0)
    ax_mech.set_ylim(0.0, 1.0)
    ax_mech.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_fig4_02_compiler_and_linker_flag_architecture(output_path: Path | str) -> Path:
    """Generate high-DPI technical architecture diagram showing Clang/LLD

    bare-metal OS kernel build pipeline and flag roles.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 8.5), facecolor="#FAFAFA")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#FFFFFF")
    ax.set_title("Figure 4-2: MikanOS ベアメタル C++ カーネル ビルドパイプライン & コンパイラ・リンカフラグ体系",
                 fontsize=13, fontweight="bold", color="#0D47A1", pad=15)

    # 3-Stage Pipeline: Source -> Clang++ (CXXFLAGS) -> main.o -> ld.lld (LDFLAGS) -> kernel.elf
    stages = [
        (0.08, 0.72, 0.16, 0.18, "ソースコード\nmain.cpp\n(C++17 カーネル)", "#E8EAF6", "#283593"),
        (0.38, 0.72, 0.22, 0.18, "中間オブジェクト\nmain.o\n(ELF64 Relocatable)", "#FFF3E0", "#E65100"),
        (0.76, 0.72, 0.18, 0.18, "カーネルバイナリ\nkernel.elf\n(ELF64 Executable\n@ 0x100000)", "#E8F5E9", "#2E7D32"),
    ]

    for x, y, w, h, text, bg, border in stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=bg, edgecolor=border, linewidth=2, zorder=3)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=border, zorder=4)

    # Connectors between stages with tool names
    ax.annotate("clang++ $(CXXFLAGS) -c $<", xy=(0.38, 0.81), xytext=(0.24, 0.81),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1565C0",
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2, color="#1565C0"), zorder=2)

    ax.annotate("ld.lld $(LDFLAGS) -o $@ $^", xy=(0.76, 0.81), xytext=(0.60, 0.81),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#D84315",
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2, color="#D84315"), zorder=2)

    # Flags Breakdown Panels (Left: CXXFLAGS, Right: LDFLAGS)
    cxx_text = (
        "【CXXFLAGS: ベアメタル C++ コンパイル制御】\n"
        "• -O2: 最適化Lv2 (インライン化・SIMDベクトル化・不要スタック退避削除)\n"
        "• -Wall -g: 全警告有効化 + DWARF デバッグ情報 (QEMU GDB連携用)\n"
        "• --target=x86_64-elf: ホストOS非依存の ELF64 ベアメタルABI指定\n"
        "• -ffreestanding: libc や暗黙のスタートアップ (crt0) を完全排除\n"
        "• -mno-red-zone: RSP下 128B 一時領域禁止 (非同期割り込み破損防止！)\n"
        "• -fno-exceptions: C++ 例外 (try/catch/unwind) ランタイムを無効化\n"
        "• -fno-rtti: dynamic_cast / typeid 関連の不要メタデータを排除\n"
        "• -std=c++17: 構造化束縛・constexpr if を含む最新モダン C++17 規格"
    )

    box_cxx = FancyBboxPatch((0.05, 0.08), 0.42, 0.55,
                             boxstyle="round,pad=0.03,rounding_size=0.025",
                             facecolor="#F3E5F5", edgecolor="#6A1B9A", linewidth=1.8, zorder=3)
    ax.add_patch(box_cxx)
    ax.text(0.07, 0.35, cxx_text, ha="left", va="center", fontsize=8.8,
            color="#4A148C", linespacing=1.4, zorder=4)

    ld_text = (
        "【LDFLAGS: ELF64 静的リンク & 物理配置制御】\n"
        "• --entry KernelMain:\n"
        "    ELF ヘッダ e_entry に登録する実行開始関数シンボルを指定。\n"
        "    ブートローダ（MikanLoaderPkg）がここへジャンプ！\n\n"
        "• -z norelro:\n"
        "    RELRO (Relocation Read-Only) セグメントの生成を無効化。\n"
        "    ページング未初期化のベアメタルでセグメント配置を極限単純化。\n\n"
        "• --image-base 0x100000:\n"
        "    カーネルの配置基準物理アドレスを 1MiB (0x100000) に固定。\n"
        "    UEFI / BIOS 予約メモリ（0x0〜0xFFFFF）を安全に回避！\n\n"
        "• --static:\n"
        "    共有ライブラリ（.so）参照を完全遮断した自己完結型静的バイナリ。"
    )

    box_ld = FancyBboxPatch((0.53, 0.08), 0.42, 0.55,
                            boxstyle="round,pad=0.03,rounding_size=0.025",
                            facecolor="#E0F2F1", edgecolor="#00695C", linewidth=1.8, zorder=3)
    ax.add_patch(box_ld)
    ax.text(0.55, 0.35, ld_text, ha="left", va="center", fontsize=8.8,
            color="#004D40", linespacing=1.35, zorder=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path



# ==============================================================================
# Chapter 4.2: FrameBufferConfig, WritePixel Mathematics & ABI Calling Convention
# ==============================================================================


@dataclass(frozen=True)
class FrameBufferConfigSim:
    """Simulates MikanOS FrameBufferConfig struct memory layout and coordinate mapping."""

    frame_buffer_base: int
    pixels_per_scan_line: int
    horizontal_resolution: int
    vertical_resolution: int
    pixel_format: PixelFormat

    # Struct layout specifications for x86-64
    STRUCT_SIZE: int = 24  # 8 + 4 + 4 + 4 + 4 = 24 bytes (multiple of 8)
    FIELD_OFFSETS: Dict[str, Tuple[int, int, str]] = field(
        default_factory=lambda: {
            "frame_buffer": (0, 8, "uint8_t*"),
            "pixels_per_scan_line": (8, 4, "uint32_t"),
            "horizontal_resolution": (12, 4, "uint32_t"),
            "vertical_resolution": (16, 4, "uint32_t"),
            "pixel_format": (20, 4, "enum PixelFormat"),
        }
    )

    def __post_init__(self) -> None:
        if self.horizontal_resolution <= 0 or self.vertical_resolution <= 0:
            raise ValueError("Resolutions must be strictly positive integers")
        if self.pixels_per_scan_line < self.horizontal_resolution:
            raise ValueError(
                f"pixels_per_scan_line ({self.pixels_per_scan_line}) cannot be less than "
                f"horizontal_resolution ({self.horizontal_resolution})"
            )

    def calc_pixel_index(self, x: int, y: int) -> int:
        """Calculate linear pixel index from 2D coordinates: y * pixels_per_scan_line + x."""
        return y * self.pixels_per_scan_line + x

    def calc_byte_offset(self, x: int, y: int) -> int:
        """Calculate byte offset from framebuffer base: 4 * (y * pixels_per_scan_line + x)."""
        return 4 * self.calc_pixel_index(x, y)

    def calc_physical_address(self, x: int, y: int) -> int:
        """Calculate physical VRAM address for pixel (x, y)."""
        return self.frame_buffer_base + self.calc_byte_offset(x, y)

    def is_in_bounds(self, x: int, y: int) -> bool:
        """Check if 2D coordinate is within active display resolution."""
        return 0 <= x < self.horizontal_resolution and 0 <= y < self.vertical_resolution

    def get_stride_padding_pixels(self) -> int:
        """Calculate invisible padding pixels per scanline: pixels_per_scan_line - horizontal_resolution."""
        return self.pixels_per_scan_line - self.horizontal_resolution

    def total_buffer_size_bytes(self) -> int:
        """Calculate total VRAM buffer size required in bytes: pixels_per_scan_line * vertical_resolution * 4."""
        return self.pixels_per_scan_line * self.vertical_resolution * 4


@dataclass(frozen=True)
class PixelColorSim:
    """Represents an 8-bit RGB color."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        for val, name in [(self.r, "r"), (self.g, "g"), (self.b, "b")]:
            if not (0 <= val <= 255):
                raise ValueError(f"Color component {name}={val} out of 8-bit range [0, 255]")

    def to_bytes(self, fmt: PixelFormat) -> bytes:
        """Convert color to 4-byte pixel quadword according to GOP PixelFormat."""
        if fmt == PixelFormat.RGB_RESV_8BIT_PER_COLOR:
            return bytes([self.r, self.g, self.b, 0])
        elif fmt == PixelFormat.BGR_RESV_8BIT_PER_COLOR:
            return bytes([self.b, self.g, self.r, 0])
        else:
            raise ValueError(f"Unsupported PixelFormat: {fmt}")


class WritePixelSimulator:
    """Simulates MikanOS WritePixel and frame buffer operations matching osbook_day04b."""

    def __init__(self, config: FrameBufferConfigSim) -> None:
        self.config = config
        self.buffer_size = config.total_buffer_size_bytes()
        self.raw_memory = bytearray(self.buffer_size)

    def write_pixel(self, x: int, y: int, color: PixelColorSim) -> int:
        """Write single pixel at (x, y). Returns 0 on success, -1 on out-of-bounds."""
        if not self.config.is_in_bounds(x, y):
            return -1
        offset = self.config.calc_byte_offset(x, y)
        pixel_bytes = color.to_bytes(self.config.pixel_format)
        self.raw_memory[offset : offset + 4] = pixel_bytes
        return 0

    def fill_rectangle(self, x: int, y: int, width: int, height: int, color: PixelColorSim) -> int:
        """Draw filled rectangle at (x, y) with size (width x height). Returns count of written pixels."""
        written = 0
        for dy in range(height):
            for dx in range(width):
                if self.write_pixel(x + dx, y + dy, color) == 0:
                    written += 1
        return written

    def read_pixel_color(self, x: int, y: int) -> Optional[PixelColorSim]:
        """Read back pixel color at (x, y). Returns None if out of bounds."""
        if not self.config.is_in_bounds(x, y):
            return None
        offset = self.config.calc_byte_offset(x, y)
        raw = self.raw_memory[offset : offset + 4]
        if self.config.pixel_format == PixelFormat.RGB_RESV_8BIT_PER_COLOR:
            return PixelColorSim(r=raw[0], g=raw[1], b=raw[2])
        elif self.config.pixel_format == PixelFormat.BGR_RESV_8BIT_PER_COLOR:
            return PixelColorSim(r=raw[2], g=raw[1], b=raw[0])
        return None

    def execute_day04b_demo(self) -> None:
        """Execute the exact drawing logic of MikanOS day04b main.cpp.

        1. Clear full screen white (255, 255, 255)
        2. Draw green rectangle (0, 255, 0) of size 200x100 at (100, 100).
        """
        # Full screen white
        for x in range(self.config.horizontal_resolution):
            for y in range(self.config.vertical_resolution):
                self.write_pixel(x, y, PixelColorSim(255, 255, 255))

        # Green rectangle at (100, 100) size 200x100
        for x in range(200):
            for y in range(100):
                self.write_pixel(100 + x, 100 + y, PixelColorSim(0, 255, 0))

    def to_rgb_numpy(self) -> np.ndarray:
        """Extract visible framebuffer region as (height, width, 3) RGB uint8 numpy array."""
        h = self.config.vertical_resolution
        w = self.config.horizontal_resolution
        arr = np.zeros((h, w, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                c = self.read_pixel_color(x, y)
                if c is not None:
                    arr[y, x] = [c.r, c.g, c.b]
        return arr


class AbiKind(Enum):
    """x86-64 Application Binary Interface Calling Conventions."""

    SYSTEM_V_AMD64 = "System V AMD64 ABI"
    MICROSOFT_X64 = "Microsoft x64 ABI"


@dataclass(frozen=True)
class AbiSpec:
    """Specification details for an x86-64 calling convention."""

    name: str
    kind: AbiKind
    integer_arg_registers: List[str]
    float_arg_registers: List[str]
    return_registers: List[str]
    callee_saved_registers: List[str]
    caller_saved_registers: List[str]
    shadow_space_bytes: int
    red_zone_bytes: int
    stack_alignment_bytes: int
    os_usage: str


class AbiComparisonAnalyzer:
    """Compares System V AMD64 ABI and Microsoft x64 ABI calling conventions."""

    SYSTEM_V_SPEC = AbiSpec(
        name="System V AMD64 ABI",
        kind=AbiKind.SYSTEM_V_AMD64,
        integer_arg_registers=["RDI", "RSI", "RDX", "RCX", "R8", "R9"],
        float_arg_registers=["XMM0", "XMM1", "XMM2", "XMM3", "XMM4", "XMM5", "XMM6", "XMM7"],
        return_registers=["RAX", "RDX"],
        callee_saved_registers=["RBX", "RSP", "RBP", "R12", "R13", "R14", "R15"],
        caller_saved_registers=["RAX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "R10", "R11"],
        shadow_space_bytes=0,
        red_zone_bytes=128,
        stack_alignment_bytes=16,
        os_usage="Linux, BSD, macOS, MikanOS Kernel",
    )

    MICROSOFT_X64_SPEC = AbiSpec(
        name="Microsoft x64 ABI",
        kind=AbiKind.MICROSOFT_X64,
        integer_arg_registers=["RCX", "RDX", "R8", "R9"],
        float_arg_registers=["XMM0", "XMM1", "XMM2", "XMM3"],
        return_registers=["RAX"],
        callee_saved_registers=["RBX", "RBP", "RDI", "RSI", "RSP", "R12", "R13", "R14", "R15"],
        caller_saved_registers=["RAX", "RCX", "RDX", "R8", "R9", "R10", "R11"],
        shadow_space_bytes=32,
        red_zone_bytes=0,
        stack_alignment_bytes=16,
        os_usage="Windows, UEFI Firmware (EDK II)",
    )

    @classmethod
    def get_spec(cls, abi: AbiKind) -> AbiSpec:
        if abi == AbiKind.SYSTEM_V_AMD64:
            return cls.SYSTEM_V_SPEC
        return cls.MICROSOFT_X64_SPEC

    @classmethod
    def map_arguments(cls, abi: AbiKind, num_args: int) -> List[Dict[str, Any]]:
        """Map argument indices to registers or stack locations."""
        spec = cls.get_spec(abi)
        mapping = []
        reg_count = len(spec.integer_arg_registers)
        for i in range(num_args):
            if i < reg_count:
                mapping.append({"arg_index": i, "location": "register", "register": spec.integer_arg_registers[i]})
            else:
                stack_idx = i - reg_count
                offset = (
                    spec.shadow_space_bytes + 8 + stack_idx * 8
                    if abi == AbiKind.MICROSOFT_X64
                    else 8 + stack_idx * 8
                )
                mapping.append({"arg_index": i, "location": "stack", "offset": offset})
        return mapping

    @classmethod
    def analyze_uefi_kernel_bridge(cls, use_efiapi: bool) -> Dict[str, Any]:
        """Analyze the ABI transition when UEFI bootloader calls KernelMain.

        If `use_efiapi` is False (MikanOS choice):
          Caller compiles call with System V ABI -> passes config pointer in RDI.
          Kernel receives config pointer in RDI -> MATCH! Success.
        If `use_efiapi` is True:
          Caller compiles call with MS x64 ABI -> passes config pointer in RCX.
          Kernel expects config pointer in RDI -> MISMATCH! Kernel reads uninitialized RDI -> Crash!
        """
        if not use_efiapi:
            return {
                "caller_convention": "System V AMD64 ABI (by omitting EFIAPI)",
                "callee_convention": "System V AMD64 ABI (Kernel default)",
                "passed_in_register": "RDI",
                "expected_in_register": "RDI",
                "is_compatible": True,
                "consequence": "KernelMain correctly reads FrameBufferConfig pointer from RDI. OS boots successfully.",
            }
        else:
            return {
                "caller_convention": "Microsoft x64 ABI (due to EFIAPI / ms_abi)",
                "callee_convention": "System V AMD64 ABI (Kernel default)",
                "passed_in_register": "RCX",
                "expected_in_register": "RDI",
                "is_compatible": False,
                "consequence": "KernelMain reads uninitialized RDI register, causing Page Fault (#PF) or General Protection (#GP) crash!",
            }


def plot_fig4_03_framebuffer_config_and_coordinate_addressing(output_path: Path | str) -> Path:
    """Generate high-DPI diagram illustrating FrameBufferConfig memory layout,

    stride padding math, RGB/BGR format, and day04b rendered screen.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor="#FAFAFA")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.25)

    # Subplot 1: FrameBufferConfig Struct Memory Layout
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. struct FrameBufferConfig メモリレイアウト (24 Bytes, 64-bit ABI)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    fields = [
        (0x00, 8, "uint8_t* frame_buffer", "VRAM先頭物理アドレスポインタ (64bit)", "#E1F5FE", "#0288D1"),
        (0x08, 4, "uint32_t pixels_per_scan_line", "1水平行の総ピクセル数 (Stride)", "#E8F5E9", "#388E3C"),
        (0x0C, 4, "uint32_t horizontal_resolution", "有効描画横解像度 (Width)", "#FFF3E0", "#F57C00"),
        (0x10, 4, "uint32_t vertical_resolution", "有効描画縦解像度 (Height)", "#F3E5F5", "#7B1FA2"),
        (0x14, 4, "enum PixelFormat pixel_format", "画素配列順 (RGB / BGR 判定用)", "#FCE4EC", "#C2185B"),
    ]

    y_pos = 0.85
    for offset, size, name, desc, bg, border in fields:
        # Offset box
        off_box = FancyBboxPatch((0.03, y_pos - 0.12), 0.16, 0.11,
                                 boxstyle="round,pad=0.015", facecolor="#ECEFF1", edgecolor="#607D8B", lw=1.2)
        ax1.add_patch(off_box)
        ax1.text(0.11, y_pos - 0.065, f"+0x{offset:02X} ({size}B)", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color="#37474F")

        # Field box
        field_box = FancyBboxPatch((0.21, y_pos - 0.12), 0.76, 0.11,
                                   boxstyle="round,pad=0.015", facecolor=bg, edgecolor=border, lw=1.5)
        ax1.add_patch(field_box)
        ax1.text(0.24, y_pos - 0.045, name, ha="left", va="center",
                 fontsize=9.2, fontweight="bold", color=border)
        ax1.text(0.24, y_pos - 0.088, desc, ha="left", va="center",
                 fontsize=7.8, color="#555555")

        y_pos -= 0.16

    ax1.text(0.5, 0.05, "★アライメント規則: 最長メンバ(ポインタ: 8B)に揃えられ、sizeof = 24B (24 % 8 == 0)",
             ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1B5E20")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # Subplot 2: 2D Coordinates to Linear VRAM Mapping & Stride Padding
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. 2次元座標 (x, y) から VRAM 物理アドレスへの写像数理",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Diagram showing visible screen and padding
    screen_rect = patches.Rectangle((0.1, 0.45), 0.55, 0.45, linewidth=2, edgecolor="#1976D2",
                                    facecolor="#E3F2FD", label="有効画面領域 (H_res × V_res)")
    padding_rect = patches.Rectangle((0.65, 0.45), 0.25, 0.45, linewidth=1.5, linestyle="--",
                                     edgecolor="#D32F2F", facecolor="#FFEBEE", label="不可視パディング領域")
    ax2.add_patch(screen_rect)
    ax2.add_patch(padding_rect)

    ax2.text(0.375, 0.675, "有効表示画面\n(horizontal_resolution)", ha="center", va="center",
             fontsize=9, fontweight="bold", color="#0D47A1")
    ax2.text(0.775, 0.675, "パディング\n(未表示VRAM)", ha="center", va="center",
             fontsize=8.5, color="#C62828")

    # Arrows for dimensions
    ax2.annotate("", xy=(0.90, 0.94), xytext=(0.10, 0.94),
                 arrowprops=dict(arrowstyle="<->", color="#388E3C", lw=2))
    ax2.text(0.50, 0.97, "pixels_per_scan_line (Stride, 例: 1024 px)", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color="#2E7D32")

    ax2.annotate("", xy=(0.65, 0.40), xytext=(0.10, 0.40),
                 arrowprops=dict(arrowstyle="<->", color="#1976D2", lw=1.5))
    ax2.text(0.375, 0.36, "horizontal_resolution (例: 800 px)", ha="center", va="top",
             fontsize=8, color="#1565C0")

    # Pixel coordinate formula
    formula_text = (
        "【座標変換数学公式】\n"
        "• ピクセルインデックス:  Index(x, y) = y × pixels_per_scan_line + x\n"
        "• バイトオフセット:      ByteOffset(x, y) = 4 × Index(x, y)\n"
        "• 物理アドレス:          Address(x, y) = frame_buffer + ByteOffset(x, y)\n\n"
        "※注意: y × horizontal_resolution で計算すると行末パディングを跨げず画面が斜めに歪む！"
    )
    fbox = FancyBboxPatch((0.05, 0.04), 0.90, 0.28, boxstyle="round,pad=0.02",
                          facecolor="#FFFDE7", edgecolor="#FBC02D", lw=1.5)
    ax2.add_patch(fbox)
    ax2.text(0.08, 0.18, formula_text, ha="left", va="center", fontsize=8.2,
             color="#5D4037", linespacing=1.3)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Subplot 3: RGB vs BGR Pixel Format 4-Byte Layout
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. 画素フォーマット (RGB vs BGR) 4バイトクアッドワード構成",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # RGB Format
    ax3.text(0.05, 0.88, "kPixelRGBResv8BitPerColor:", fontsize=9.5, fontweight="bold", color="#1565C0")
    rgb_boxes = [
        (0.05, "+0: Red", "#FFCDD2", "#C62828"),
        (0.27, "+1: Green", "#C8E6C9", "#2E7D32"),
        (0.49, "+2: Blue", "#BBDEFB", "#1565C0"),
        (0.71, "+3: Reserved (0)", "#CFD8DC", "#455A64"),
    ]
    for x, label, bg, border in rgb_boxes:
        b = FancyBboxPatch((x, 0.68), 0.20, 0.16, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=border, lw=1.5)
        ax3.add_patch(b)
        ax3.text(x + 0.10, 0.76, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=border)

    # BGR Format
    ax3.text(0.05, 0.52, "kPixelBGRResv8BitPerColor:", fontsize=9.5, fontweight="bold", color="#2E7D32")
    bgr_boxes = [
        (0.05, "+0: Blue", "#BBDEFB", "#1565C0"),
        (0.27, "+1: Green", "#C8E6C9", "#2E7D32"),
        (0.49, "+2: Red", "#FFCDD2", "#C62828"),
        (0.71, "+3: Reserved (0)", "#CFD8DC", "#455A64"),
    ]
    for x, label, bg, border in bgr_boxes:
        b = FancyBboxPatch((x, 0.32), 0.20, 0.16, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=border, lw=1.5)
        ax3.add_patch(b)
        ax3.text(x + 0.10, 0.40, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color=border)

    ax3.text(0.50, 0.12, "WritePixel() は config.pixel_format に応じて分岐し、\n適切なメモリ位置に R/G/B バイトを格納する",
             ha="center", va="center", fontsize=8.5, color="#37474F", linespacing=1.3)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # Subplot 4: day04b Screen Drawing Simulation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. osbook_day04b 画面描画結果シミュレーション",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Simulate drawing
    sim_config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=400,
        horizontal_resolution=400,
        vertical_resolution=300,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    sim = WritePixelSimulator(sim_config)
    sim.execute_day04b_demo()
    img_data = sim.to_rgb_numpy()

    ax4.imshow(img_data, origin="upper")
    ax4.set_xlim(0, 400)
    ax4.set_ylim(300, 0)
    ax4.set_xlabel("X coordinate (pixels)", fontsize=8)
    ax4.set_ylabel("Y coordinate (pixels)", fontsize=8)

    # Annotate rectangle
    ax4.annotate("長方形 (100, 100)\n200 × 100 px (Green)", xy=(200, 150), xytext=(220, 70),
                 arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.5),
                 fontsize=8.5, fontweight="bold", color="#1B5E20",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#388E3C"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_fig4_04_abi_calling_convention_comparison(output_path: Path | str) -> Path:
    """Generate high-DPI architecture diagram comparing System V AMD64 ABI

    vs Microsoft x64 ABI and the UEFI-to-Kernel handoff.
    """
    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9.5), facecolor="#FAFAFA")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.25)

    # Subplot 1: Argument Passing Registers Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. 引数引き渡しレジスタ順序 (Integer / Pointer Arguments)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    arg_rows = [
        ("第1引数", "RDI", "RCX (※食い違い！)", "#FFCDD2"),
        ("第2引数", "RSI", "RDX (※食い違い！)", "#FFCDD2"),
        ("第3引数", "RDX", "R8", "#E8F5E9"),
        ("第4引数", "RCX", "R9", "#E8F5E9"),
        ("第5引数", "R8", "スタック [RSP + 0x28]", "#FFF3E0"),
        ("第6引数", "R9", "スタック [RSP + 0x30]", "#FFF3E0"),
    ]

    y = 0.86
    ax1.text(0.08, y, "引数", fontweight="bold", fontsize=9, color="#37474F")
    ax1.text(0.32, y, "System V AMD64 (Kernel)", fontweight="bold", fontsize=9, color="#1565C0")
    ax1.text(0.70, y, "Microsoft x64 (UEFI)", fontweight="bold", fontsize=9, color="#D84315")

    for arg_name, sysv, ms, bg in arg_rows:
        y -= 0.115
        row_box = FancyBboxPatch((0.04, y - 0.035), 0.92, 0.095, boxstyle="round,pad=0.01",
                                 facecolor=bg, edgecolor="#B0BEC5", lw=0.8)
        ax1.add_patch(row_box)
        ax1.text(0.08, y + 0.012, arg_name, fontsize=8.5, color="#263238")
        ax1.text(0.35, y + 0.012, sysv, fontsize=8.8, fontweight="bold", color="#0D47A1")
        ax1.text(0.70, y + 0.012, ms, fontsize=8.8, fontweight="bold", color="#BF360C")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # Subplot 2: Stack Frame & Special Spaces (Shadow Space vs Red Zone)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. スタックフレーム構造 (シャドウスペース vs レッドゾーン)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # MS Shadow Space box
    ms_box = FancyBboxPatch((0.05, 0.52), 0.42, 0.42, boxstyle="round,pad=0.02",
                            facecolor="#FBE9E7", edgecolor="#FF5722", lw=1.5)
    ax2.add_patch(ms_box)
    ms_text = (
        "【Microsoft x64 ABI】\n\n"
        "• シャドウスペース (Home Space):\n"
        "  呼出側 (Caller) がスタック上に\n"
        "  必ず 32バイト (4×8B) を確保！\n"
        "  (引数レジスタ RCX, RDX, R8, R9 を\n"
        "   関数側が即座に退避できる領域)\n\n"
        "• レッドゾーン: なし (存在しない)"
    )
    ax2.text(0.08, 0.73, ms_text, fontsize=8.2, color="#BF360C", linespacing=1.25)

    # System V Red Zone box
    sysv_box = FancyBboxPatch((0.53, 0.52), 0.42, 0.42, boxstyle="round,pad=0.02",
                             facecolor="#E3F2FD", edgecolor="#2196F3", lw=1.5)
    ax2.add_patch(sysv_box)
    sysv_text = (
        "【System V AMD64 ABI】\n\n"
        "• シャドウスペース: なし\n\n"
        "• レッドゾーン (Red Zone):\n"
        "  RSP の下 128バイトの一時領域。\n"
        "  リーフ関数でスタック確保命令を\n"
        "  省略して超高速化を図る仕組み。\n"
        "  ★カーネルでは -mno-red-zone 必須"
    )
    ax2.text(0.56, 0.73, sysv_text, fontsize=8.2, color="#0D47A1", linespacing=1.25)

    # Shared Stack Alignment rule
    align_box = FancyBboxPatch((0.05, 0.08), 0.90, 0.36, boxstyle="round,pad=0.02",
                               facecolor="#E8F5E9", edgecolor="#4CAF50", lw=1.5)
    ax2.add_patch(align_box)
    align_text = (
        "【共通規則: 16バイト・スタックアライメント (16-Byte Stack Alignment)】\n"
        "• CALL 命令発行直前の RSP は必ず 16の倍数 (RSP % 16 == 0) でなければならない。\n"
        "• CALL 命令で 8バイトの戻りアドレスが PUSH されるため、関数エントリ時点では\n"
        "  必ず (RSP % 16 == 8) となる！\n"
        "• SSE / AVX 命令（MOVAPS 等）は 16バイト整列スタックを要求し、違反すると #GP 例外で爆死する。"
    )
    ax2.text(0.08, 0.26, align_text, fontsize=8.2, color="#1B5E20", linespacing=1.3)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Subplot 3: Register Preservation Rules (Callee-saved vs Scratch)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. レジスタ保護責任 (Callee-saved vs Caller-saved)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    reg_text = (
        "【保存義務の比較 (誰がレジスタの値を守るか？)】\n\n"
        "1. カリーセーブレジスタ (Callee-saved / Preserved / Non-volatile):\n"
        "   呼び出された関数が復帰前に値を元通り復元する責任を負う。\n"
        "   • System V ABI:  RBX, RSP, RBP, R12, R13, R14, R15 (計 7本)\n"
        "   • Microsoft ABI: RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15 (計 9本)\n"
        "   ★★★ 極めて重要な違い: RDI と RSI ★★★\n"
        "   - MS ABI では RDI, RSI は『保護必須 (Callee-saved)』！\n"
        "   - System V では RDI, RSI は『使い捨て引数 (Caller-saved)』！\n\n"
        "2. スクラッチレジスタ (Caller-saved / Volatile):\n"
        "   関数呼び出しによって値が自由に破壊されてよいレジスタ。\n"
        "   • System V: RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11\n"
        "   • MS ABI:   RAX, RCX, RDX, R8, R9, R10, R11"
    )
    rbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                          facecolor="#FFF8E1", edgecolor="#FFA000", lw=1.5)
    ax3.add_patch(rbox)
    ax3.text(0.07, 0.50, reg_text, fontsize=8.2, color="#4E342E", linespacing=1.3)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # Subplot 4: UEFI Bootloader -> KernelMain Transition Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. MikanLoader から KernelMain への移行と EFIAPI 欠落の真相",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    bridge_text = (
        "【なぜ Main.c で EFIAPI を付けないのか？】\n\n"
        "■ 正しい実装 (MikanOS 公式):\n"
        "  typedef void EntryPointType(const struct FrameBufferConfig*);\n"
        "  -> EFIAPI なし！\n"
        "  -> Linux 上の Clang はホスト規定の『System V AMD64 ABI』コードを生成！\n"
        "  -> 呼出元 (Main.c) は &config を 【RDI】 に格納して CALL 実行！\n"
        "  -> カーネル側 KernelMain も RDI から第1引数を読取！ -> 完全一致！起動成功！\n\n"
        "■ 誤った実装 (もし EFIAPI を付けたら？):\n"
        "  typedef void EFIAPI EntryPointType(const struct FrameBufferConfig*);\n"
        "  -> EFIAPI は __attribute__((ms_abi)) に展開される！\n"
        "  -> 呼出元 (Main.c) は &config を 【RCX】 に格納して CALL 実行！\n"
        "  -> カーネル側は System V なので 【RDI】 から第1引数を読取！\n"
        "  -> RDI は未初期化のゴミ値！ -> config.frame_buffer アクセスで即座に #PF クラッシュ！"
    )
    bbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                          facecolor="#EDE7F6", edgecolor="#7E57C2", lw=1.5)
    ax4.add_patch(bbox)
    ax4.text(0.07, 0.50, bridge_text, fontsize=8.0, color="#311B92", linespacing=1.28)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# Chapter 4.3: Refactoring with C++ OOP & Placement New Architecture (osbook_day04c)
# ==============================================================================

class PixelWriterSim(abc.ABC):
    """Abstract base class for pixel writers (corresponding to MikanOS PixelWriter).
    
    Encapsulates FrameBufferConfig reference and provides polymorphic pixel rendering.
    """
    def __init__(self, config: FrameBufferConfigSim, buffer: Optional[bytearray] = None) -> None:
        self.config = config
        self.buffer = buffer if buffer is not None else bytearray(config.total_buffer_size_bytes())

    def pixel_at(self, x: int, y: int) -> int:
        """Calculate byte offset within the linear frame buffer."""
        if not self.config.is_in_bounds(x, y):
            raise IndexError(f"Pixel coordinates ({x}, {y}) out of screen bounds ({self.config.horizontal_resolution}x{self.config.vertical_resolution})")
        return self.config.calc_byte_offset(x, y)

    @abc.abstractmethod
    def write(self, x: int, y: int, color: PixelColorSim) -> None:
        """Write single pixel at (x, y) with specified color. Pure virtual method."""
        pass


class RGBResv8BitPerColorPixelWriterSim(PixelWriterSim):
    """Concrete PixelWriter for kPixelRGBResv8BitPerColor format."""
    def write(self, x: int, y: int, color: PixelColorSim) -> None:
        offset = self.pixel_at(x, y)
        self.buffer[offset] = color.r
        self.buffer[offset + 1] = color.g
        self.buffer[offset + 2] = color.b
        self.buffer[offset + 3] = 0


class BGRResv8BitPerColorPixelWriterSim(PixelWriterSim):
    """Concrete PixelWriter for kPixelBGRResv8BitPerColor format."""
    def write(self, x: int, y: int, color: PixelColorSim) -> None:
        offset = self.pixel_at(x, y)
        self.buffer[offset] = color.b
        self.buffer[offset + 1] = color.g
        self.buffer[offset + 2] = color.r
        self.buffer[offset + 3] = 0


@dataclass
class PlacementNewSimulator:
    """Simulates placement new on pre-allocated static storage without dynamic heap."""
    buffer_capacity: int
    alignment: int = 8
    allocated_object: Optional[Any] = None
    object_type_name: str = ""
    is_active: bool = False
    write_count: int = 0

    def allocate_placement(self, target_cls: type, *args, **kwargs) -> Any:
        """Construct object directly in static memory without dynamic heap."""
        obj = target_cls(*args, **kwargs)
        estimated_size = 16  # 8 bytes vptr + 8 bytes reference pointer
        if estimated_size > self.buffer_capacity:
            raise MemoryError(
                f"Static buffer capacity {self.buffer_capacity} bytes exceeded by object size {estimated_size} bytes"
            )
        self.allocated_object = obj
        self.object_type_name = target_cls.__name__
        self.is_active = True
        return obj

    def destroy_placement(self) -> None:
        """Simulate destructor execution and cleanup of static storage."""
        self.allocated_object = None
        self.object_type_name = ""
        self.is_active = False


@dataclass
class PolymorphismCostMetrics:
    """Performance and architectural comparison metrics between if-else and vtable."""
    total_pixels: int
    branch_instructions_ifelse: int
    branch_mispredictions_ifelse: int
    vtable_indirect_calls: int
    vtable_cache_misses: int
    estimated_cycles_ifelse: int
    estimated_cycles_vtable: int
    code_modularity_score: float  # 0.0 to 10.0 scale


class BranchVsVtableBenchmark:
    """Quantitative performance and software engineering benchmark for pixel drawing."""
    @staticmethod
    def evaluate(width: int = 800, height: int = 600) -> PolymorphismCostMetrics:
        total_pixels = width * height
        # If-Else branch per pixel:
        # 1 CMP instruction + 1 JNE per pixel
        branch_instructions = total_pixels * 2
        # Once branch predictor warms up (after ~10-20 iterations), mispredict is very low (< 0.01%)
        branch_mispredictions = 20
        # Pipeline latency: ~4 cycles per pixel including branch check overhead
        cycles_ifelse = total_pixels * 4 + branch_mispredictions * 15

        # Vtable indirect call:
        # Indirect call through vtable pointer (1 vptr lookup + 1 call [rax+0])
        # Indirect branch target buffer (BTB) has 100% hit rate because target address never changes
        vtable_calls = total_pixels
        vtable_misses = 1
        # Virtual call dispatch latency: ~2-3 cycles with warm BTB and L1i cache
        cycles_vtable = total_pixels * 3 + vtable_misses * 15

        return PolymorphismCostMetrics(
            total_pixels=total_pixels,
            branch_instructions_ifelse=branch_instructions,
            branch_mispredictions_ifelse=branch_mispredictions,
            vtable_indirect_calls=vtable_calls,
            vtable_cache_misses=vtable_misses,
            estimated_cycles_ifelse=cycles_ifelse,
            estimated_cycles_vtable=cycles_vtable,
            code_modularity_score=9.8
        )


def plot_fig4_05_pixel_writer_class_hierarchy_and_memory(output_path: str) -> str:
    """Generate high-resolution architecture figure for Section 4.3 (Figure 4.5)."""
    setup_plot_style()
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # Subplot 1: PixelWriter Class Hierarchy & OOP Abstraction
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. PixelWriter クラス階層構造とポリモーフィズム設計",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Base class box
    base_box = FancyBboxPatch((0.18, 0.58), 0.64, 0.36, boxstyle="round,pad=0.03",
                              facecolor="#E3F2FD", edgecolor="#1976D2", lw=2)
    ax1.add_patch(base_box)
    base_text = (
        "【抽象基底クラス】 class PixelWriter (Abstract Base)\n"
        "───────────────────────────────────────\n"
        " - private: const FrameBufferConfig& config_ (8 bytes)\n"
        "───────────────────────────────────────\n"
        " + public:  PixelWriter(const FrameBufferConfig& config)\n"
        " + public:  virtual ~PixelWriter() = default;\n"
        " + public:  virtual void Write(int x, int y, const PixelColor& c) = 0;\n"
        " # protected: uint8_t* PixelAt(int x, int y) { ... }"
    )
    ax1.text(0.20, 0.74, base_text, fontsize=7.8, color="#0D47A1", linespacing=1.28)

    # Derived RGB box
    rgb_box = FancyBboxPatch((0.04, 0.08), 0.42, 0.34, boxstyle="round,pad=0.02",
                             facecolor="#E8F5E9", edgecolor="#388E3C", lw=1.8)
    ax1.add_patch(rgb_box)
    rgb_text = (
        "class RGBResv8BitPerColorPixelWriter\n"
        ": public PixelWriter\n"
        "─────────────────────────────\n"
        "using PixelWriter::PixelWriter;\n"
        "override void Write(x, y, c) {\n"
        "  auto p = PixelAt(x, y);\n"
        "  p[0]=c.r; p[1]=c.g; p[2]=c.b;\n"
        "}"
    )
    ax1.text(0.06, 0.23, rgb_text, fontsize=7.5, color="#1B5E20", linespacing=1.25)

    # Derived BGR box
    bgr_box = FancyBboxPatch((0.54, 0.08), 0.42, 0.34, boxstyle="round,pad=0.02",
                             facecolor="#FFF3E0", edgecolor="#F57C00", lw=1.8)
    ax1.add_patch(bgr_box)
    bgr_text = (
        "class BGRResv8BitPerColorPixelWriter\n"
        ": public PixelWriter\n"
        "─────────────────────────────\n"
        "using PixelWriter::PixelWriter;\n"
        "override void Write(x, y, c) {\n"
        "  auto p = PixelAt(x, y);\n"
        "  p[0]=c.b; p[1]=c.g; p[2]=c.r;\n"
        "}"
    )
    ax1.text(0.56, 0.23, bgr_text, fontsize=7.5, color="#E65100", linespacing=1.25)

    # Inheritance arrows
    ax1.annotate("", xy=(0.32, 0.58), xytext=(0.25, 0.42),
                 arrowprops=dict(arrowstyle="-|>", color="#1976D2", lw=2, mutation_scale=15))
    ax1.annotate("", xy=(0.68, 0.58), xytext=(0.75, 0.42),
                 arrowprops=dict(arrowstyle="-|>", color="#1976D2", lw=2, mutation_scale=15))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # Subplot 2: Object Memory Layout and vtable Linkage
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. インスタンスのメモリレイアウトと vptr / vtable 結合",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Object layout box
    obj_box = FancyBboxPatch((0.06, 0.48), 0.42, 0.44, boxstyle="round,pad=0.02",
                             facecolor="#F3E5F5", edgecolor="#8E24AA", lw=1.8)
    ax2.add_patch(obj_box)
    obj_text = (
        "【インスタンス (sizeof = 16 bytes)】\n"
        "offset 0x00:  [ vptr (8 bytes) ]\n"
        "  └──> 仮想関数テーブルへ\n"
        "offset 0x08:  [ config_ 参照 (8 bytes) ]\n"
        "  └──> FrameBufferConfig へのポインタ\n"
        "───────────────────────────\n"
        "※派生クラスはメンバを追加しないため\n"
        "  RGB / BGR ともにジャスト 16 バイト！"
    )
    ax2.text(0.08, 0.68, obj_text, fontsize=7.6, color="#4A148C", linespacing=1.3)

    # vtable box
    vt_box = FancyBboxPatch((0.56, 0.48), 0.38, 0.44, boxstyle="round,pad=0.02",
                            facecolor="#EDE7F6", edgecolor="#5E35B1", lw=1.8)
    ax2.add_patch(vt_box)
    vt_text = (
        "【vtable (仮想関数テーブル)】\n"
        "[0]: ~PixelWriter() (デストラクタ)\n"
        "[1]: Write(int, int, Color)\n"
        "     └──> &RGB...::Write\n"
        "───────────────────────\n"
        "純粋仮想関数のままではリンク\n"
        "エラー (__cxa_pure_virtual)！\n"
        "派生クラスで完全オーバーライド！"
    )
    ax2.text(0.58, 0.68, vt_text, fontsize=7.4, color="#311B92", linespacing=1.3)

    # Arrow from vptr to vtable
    ax2.annotate("", xy=(0.56, 0.82), xytext=(0.48, 0.82),
                 arrowprops=dict(arrowstyle="->", color="#8E24AA", lw=2.2))

    # Static buffer placement explanation
    stat_box = FancyBboxPatch((0.06, 0.08), 0.88, 0.32, boxstyle="round,pad=0.02",
                              facecolor="#FAFAFA", edgecolor="#757575", lw=1.5)
    ax2.add_patch(stat_box)
    stat_text = (
        "【静的メモリ上の配置】 char pixel_writer_buf[sizeof(RGBResv8BitPerColorPixelWriter)];\n"
        "• OSブート直後はヒープ（malloc / OSページアロケータ）が一切存在しないフリースタンディング環境。\n"
        "• 通常の `new Class()` は `malloc()` を要求するため未定義動作・リンクエラーとなる。\n"
        "• したがって BSS / データセグメントに 16 バイトの静的バッファを確保し、Placement new で構築する！"
    )
    ax2.text(0.08, 0.22, stat_text, fontsize=7.8, color="#212121", linespacing=1.32)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Subplot 3: Placement New Execution Sequence
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. 配置 new (Placement new) の動作機構とメモリ再利用",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    pnew_text = (
        "【Placement new の 3 段階実行プロセス】\n\n"
        "1. 自前 operator new 定義:\n"
        "   void* operator new(size_t size, void* buf) { return buf; }\n"
        "   -> メモリ確保は行わず、第2引数で渡されたバッファポインタをそのまま返す！\n\n"
        "2. 呼び出し側の構文:\n"
        "   pixel_writer = new(pixel_writer_buf)\n"
        "       RGBResv8BitPerColorPixelWriter{frame_buffer_config};\n"
        "   -> コンパイラは上記 operator new を呼んでアドレスを取得し、\n"
        "      そのメモリ番地に対して直ちにコンストラクタを実行！\n"
        "   -> vptr が vtable_RGB を指すように初期化され、config_ に参照が格納される！\n\n"
        "3. ダミー operator delete:\n"
        "   void operator delete(void* obj) noexcept { }\n"
        "   -> 静的バッファなのでメモリ解放は不要。リンクエラー回避のために空関数を定義。"
    )
    pnbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                           facecolor="#E0F2F1", edgecolor="#00897B", lw=1.5)
    ax3.add_patch(pnbox)
    ax3.text(0.07, 0.50, pnew_text, fontsize=8.0, color="#004D40", linespacing=1.3)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # Subplot 4: day04c FrameBuffer Output Simulation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. day04c カーネル画面描画シミュレーション (GOP FrameBuffer)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Simulate drawing
    cfg = FrameBufferConfigSim(
        frame_buffer_base=0xFFFF8000000A0000,
        pixels_per_scan_line=800,
        horizontal_resolution=800,
        vertical_resolution=600,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR
    )
    writer = RGBResv8BitPerColorPixelWriterSim(cfg)
    # Fill white
    for y in range(600):
        for x in range(800):
            writer.write(x, y, PixelColorSim(255, 255, 255))
    # Draw green rectangle @ (100, 100), 200x100
    for y in range(100):
        for x in range(200):
            writer.write(100 + x, 100 + y, PixelColorSim(0, 255, 0))

    img = np.zeros((600, 800, 3), dtype=np.uint8)
    for y in range(600):
        for x in range(800):
            off = cfg.calc_byte_offset(x, y)
            img[y, x, 0] = writer.buffer[off]
            img[y, x, 1] = writer.buffer[off + 1]
            img[y, x, 2] = writer.buffer[off + 2]

    ax4.imshow(img, aspect="auto")
    ax4.set_xlabel("X [pixels]", fontsize=8.5)
    ax4.set_ylabel("Y [pixels]", fontsize=8.5)
    ax4.annotate("Green Rect: (100, 100) 200x100", xy=(200, 150), xytext=(350, 150),
                 arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2),
                 fontsize=8.5, fontweight="bold", color="#D32F2F")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_fig4_06_placement_new_and_runtime_cost(output_path: str) -> str:
    """Generate high-resolution architecture figure for Section 4.3 (Figure 4.6)."""
    setup_plot_style()
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # Subplot 1: Regular new vs Placement new comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. 通常の new (Heap) vs 配置 new (Placement new) の相違",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    comp_text = (
        "【メモリ確保メカニズムの決定的違い】\n\n"
        "■ 通常の `new Type(args)`:\n"
        "  1. ランタイムの `malloc()` / `sbrk()` / `mmap()` を呼び出してヒープからメモリ確保\n"
        "  2. 確保失敗時は `std::bad_alloc` 例外を送出 (例外サポートが必要)\n"
        "  3. 確保されたアドレス上でコンストラクタを実行\n"
        "  ✕ OSカーネル起動直後はヒープ機構が未初期化のため利用不可！\n\n"
        "■ 配置 `new(buf) Type(args)`:\n"
        "  1. メモリ確保を行わず、呼び出し側が用意したバッファアドレスをそのまま使用\n"
        "  2. 動的アロケーション・システムコール・例外機構への依存度ゼロ！\n"
        "  3. 指定アドレス上でコンストラクタのみを即座に実行\n"
        "  ◎ ベアメタル・組み込み・OS自作の初期化フェーズに最適！"
    )
    cbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                          facecolor="#FFFDE7", edgecolor="#FBC02D", lw=1.5)
    ax1.add_patch(cbox)
    ax1.text(0.07, 0.50, comp_text, fontsize=8.0, color="#F57F17", linespacing=1.3)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # Subplot 2: Alignment, Sizeof, and Padding in C++20
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. C++20 におけるメモリ配置・sizeof と alignof",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Memory table
    categories = ["PixelWriter (基底)", "RGB...PixelWriter", "BGR...PixelWriter", "pixel_writer_buf"]
    sizes = [16, 16, 16, 16]
    aligns = [8, 8, 8, 1]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width/2, sizes, width, label="sizeof [Bytes]", color="#1976D2")
    ax2.bar(x + width/2, aligns, width, label="alignof [Bytes]", color="#388E3C")

    ax2.set_ylabel("Bytes", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=7.5, rotation=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i in range(len(categories)):
        ax2.text(x[i] - width/2, sizes[i] + 0.5, f"{sizes[i]}B", ha="center", fontsize=7.8, fontweight="bold")
        ax2.text(x[i] + width/2, aligns[i] + 0.5, f"{aligns[i]}B", ha="center", fontsize=7.8, fontweight="bold")

    align_note = (
        "※注意点: `char buf[]` は `alignof == 1` のため、本来 x86-64 ポインタの\n"
        "自然アライメント（8バイト）を満たすには `alignas(alignof(Writer)) char buf[...]`\n"
        "または `alignas(void*) char buf[...]` と指定するのが最も安全。"
    )
    ax2.text(0.05, 0.65, align_note, transform=ax2.transAxes, fontsize=7.2,
             color="#C62828", bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#EF5350"))

    # Subplot 3: If-Else vs Vtable Runtime Cost Benchmark
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. 描画ループにおける If-Else 分岐 vs vtable 間接呼出 コスト比較",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    resolutions = ["320x240\n(76.8k)", "640x480\n(307.2k)", "800x600\n(480k)", "1024x768\n(786.4k)"]
    pixels = [320*240, 640*480, 800*600, 1024*768]
    cycles_ifelse = [p * 4 / 1e6 for p in pixels]
    cycles_vtable = [p * 3 / 1e6 for p in pixels]

    rx = np.arange(len(resolutions))
    rwidth = 0.35

    ax3.bar(rx - rwidth/2, cycles_ifelse, rwidth, label="day04b: If-Else 分岐 (毎ピクセル判定)", color="#E53935")
    ax3.bar(rx + rwidth/2, cycles_vtable, rwidth, label="day04c: vtable 多態性 (起動時1回解決)", color="#1E88E5")

    ax3.set_ylabel("推定実行サイクル数 [Mega Cycles]", fontsize=8.5)
    ax3.set_xticks(rx)
    ax3.set_xticklabels(resolutions, fontsize=8)
    ax3.legend(loc="upper left", fontsize=8.2)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i in range(len(resolutions)):
        ax3.text(rx[i] - rwidth/2, cycles_ifelse[i] + 0.05, f"{cycles_ifelse[i]:.2f}M", ha="center", fontsize=7.5)
        ax3.text(rx[i] + rwidth/2, cycles_vtable[i] + 0.05, f"{cycles_vtable[i]:.2f}M", ha="center", fontsize=7.5)

    # Subplot 4: Column 4.2 Compile Errors Are Your Friends
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. コラム 4.2「コンパイルエラーはお友達」と型安全性の恩恵",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    column_text = (
        "【コンパイルエラーが開発者を救う 3 つの場面】\n\n"
        "1. 純粋仮想関数未実装の即時検知:\n"
        "   `virtual void Write(...) = 0;`\n"
        "   もし新しい派生クラスで `Write` の実装を忘れると、コンパイル時に\n"
        "   『cannot allocate an object of abstract type』と怒られる。\n"
        "   -> 実行時に未初期化コードを実行してクラッシュするのを完全に防止！\n\n"
        "2. 仮想デストラクタによる安全な破棄:\n"
        "   `virtual ~PixelWriter() = default;`\n"
        "   基底クラスポインタ `delete pixel_writer` された時、派生クラスの\n"
        "   デストラクタが確実に呼ばれることを保証。\n\n"
        "3. コンパイラの警告オプション (-Wall -Wextra):\n"
        "   コンパイラは敵ではなく、バグを未然に防ぐ最強のパートナー！"
    )
    colbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                            facecolor="#EDE7F6", edgecolor="#673AB7", lw=1.5)
    ax4.add_patch(colbox)
    ax4.text(0.07, 0.50, column_text, fontsize=8.0, color="#311B92", linespacing=1.3)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# Chapter 4.4: vtable (Virtual Method Table & __cxa_pure_virtual)
# ==============================================================================

class PureVirtualCallError(RuntimeError):
    """Exception raised when a pure virtual function slot is invoked (__cxa_pure_virtual trap)."""
    pass


@dataclass
class VTableEntry:
    """Represents a single slot in an Itanium C++ ABI virtual method table."""
    name: str
    offset_bytes: int  # Offset from vptr (e.g., -16, -8, 0, 8, 16)
    target_address: int  # Virtual address of function or metadata
    symbol_name: str  # Mangled or readable symbol name
    is_pure_virtual: bool = False
    description: str = ""


@dataclass
class VirtualMethodTable:
    """Simulates an Itanium C++ ABI virtual table for a C++ polymorphic class."""
    class_name: str
    vtable_base_address: int  # Address where the vptr points (slot 0)
    entries: list[VTableEntry] = field(default_factory=list)

    def get_entry_by_offset(self, offset: int) -> Optional[VTableEntry]:
        for entry in self.entries:
            if entry.offset_bytes == offset:
                return entry
        return None

    def lookup_function(self, func_name: str) -> Optional[VTableEntry]:
        for entry in self.entries:
            if entry.name == func_name:
                return entry
        return None


class PolymorphicObjectLayout:
    """Simulates the in-memory object layout of a C++ polymorphic class instance."""

    def __init__(self, class_name: str, instance_address: int = 0x100000) -> None:
        self.class_name = class_name
        self.instance_address = instance_address
        self.members: list[dict[str, Any]] = []
        self.total_size = 0
        self.alignment = 8
        self.vptr_value: int = 0

    def add_vptr(self, vtable_address: int) -> None:
        """Add the hidden _vptr at offset 0 (8 bytes on x86-64)."""
        self.vptr_value = vtable_address
        self.members.append({
            "name": "_vptr",
            "offset": 0,
            "size": 8,
            "type": "void** (vptr)",
            "value": f"0x{vtable_address:016x}",
            "description": f"Pointer to vtable for {self.class_name}",
        })
        self.total_size = max(self.total_size, 8)

    def add_member(self, name: str, size: int, type_str: str, value: Any, description: str = "") -> None:
        """Add a member variable with natural alignment."""
        align = min(size, self.alignment)
        offset = (self.total_size + align - 1) & ~(align - 1)
        self.members.append({
            "name": name,
            "offset": offset,
            "size": size,
            "type": type_str,
            "value": value,
            "description": description,
        })
        self.total_size = offset + size

    def format_memory_map(self) -> str:
        """Return an ASCII visualization of the object memory layout."""
        lines = [
            f"+-----------------------------------------------------------------------------+",
            f"| Object Memory Layout: {self.class_name:<46} |",
            f"| Base Address: 0x{self.instance_address:016x} | Size: {self.total_size:>2} Bytes | Align: {self.alignment}B  |",
            f"+--------+------+---------------------+--------------------+------------------+",
            f"| Offset | Size | Field Name          | Type               | Value / Target   |",
            f"+--------+------+---------------------+--------------------+------------------+",
        ]
        for m in self.members:
            lines.append(
                f"| +0x{m['offset']:02x}  | {m['size']:>2}B  | {m['name']:<19} | {m['type']:<18} | {str(m['value']):<16} |"
            )
        lines.append(f"+--------+------+---------------------+--------------------+------------------+")
        return "\n".join(lines)


class ItaniumAbiVTableSimulator:
    """Simulates the compiler's generation of vtables and dynamic dispatch under Itanium C++ ABI."""

    def __init__(self) -> None:
        self.vtables: dict[str, VirtualMethodTable] = {}
        self.objects: dict[str, PolymorphicObjectLayout] = {}
        self.pure_virtual_call_count = 0
        self.dispatch_log: list[str] = []
        self._setup_mikanos_classes()

    def _setup_mikanos_classes(self) -> None:
        # Vtable for PixelWriter (Abstract Base Class)
        # In Itanium ABI:
        # offset -16: offset-to-top (0 for single inheritance)
        # offset  -8: RTTI typeinfo pointer (&_ZTI11PixelWriter)
        # offset   0: complete object destructor (&PixelWriter::~PixelWriter)
        # offset  +8: deleting destructor (&PixelWriter::~PixelWriter)
        # offset +16: Write slot -> __cxa_pure_virtual!
        pw_vtable_addr = 0x403000
        pw_entries = [
            VTableEntry("offset_to_top", -16, 0x0, "offset-to-top", False, "Offset to complete object (0)"),
            VTableEntry("typeinfo", -8, 0x403F00, "_ZTI11PixelWriter", False, "std::type_info for PixelWriter"),
            VTableEntry("complete_dtor", 0, 0x401050, "_ZN11PixelWriterD1Ev", False, "Complete object destructor"),
            VTableEntry("deleting_dtor", 8, 0x401070, "_ZN11PixelWriterD0Ev", False, "Deleting destructor"),
            VTableEntry("Write", 16, 0x401FF0, "__cxa_pure_virtual", True, "Pure virtual function trap!"),
        ]
        self.vtables["PixelWriter"] = VirtualMethodTable("PixelWriter", pw_vtable_addr, pw_entries)

        # Vtable for RGBResv8BitPerColorPixelWriter (Concrete Derived Class)
        rgb_vtable_addr = 0x403100
        rgb_entries = [
            VTableEntry("offset_to_top", -16, 0x0, "offset-to-top", False, "Offset to complete object (0)"),
            VTableEntry("typeinfo", -8, 0x403F20, "_ZTI30RGBResv8BitPerColorPixelWriter", False, "typeinfo for RGB writer"),
            VTableEntry("complete_dtor", 0, 0x401100, "_ZN30RGBResv8BitPerColorPixelWriterD1Ev", False, "Complete object destructor"),
            VTableEntry("deleting_dtor", 8, 0x401120, "_ZN30RGBResv8BitPerColorPixelWriterD0Ev", False, "Deleting destructor"),
            VTableEntry("Write", 16, 0x401200, "_ZN30RGBResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor", False, "RGB pixel writer override"),
        ]
        self.vtables["RGBResv8BitPerColorPixelWriter"] = VirtualMethodTable(
            "RGBResv8BitPerColorPixelWriter", rgb_vtable_addr, rgb_entries
        )

        # Vtable for BGRResv8BitPerColorPixelWriter (Concrete Derived Class)
        bgr_vtable_addr = 0x403200
        bgr_entries = [
            VTableEntry("offset_to_top", -16, 0x0, "offset-to-top", False, "Offset to complete object (0)"),
            VTableEntry("typeinfo", -8, 0x403F40, "_ZTI30BGRResv8BitPerColorPixelWriter", False, "typeinfo for BGR writer"),
            VTableEntry("complete_dtor", 0, 0x401150, "_ZN30BGRResv8BitPerColorPixelWriterD1Ev", False, "Complete object destructor"),
            VTableEntry("deleting_dtor", 8, 0x401170, "_ZN30BGRResv8BitPerColorPixelWriterD0Ev", False, "Deleting destructor"),
            VTableEntry("Write", 16, 0x401300, "_ZN30BGRResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor", False, "BGR pixel writer override"),
        ]
        self.vtables["BGRResv8BitPerColorPixelWriter"] = VirtualMethodTable(
            "BGRResv8BitPerColorPixelWriter", bgr_vtable_addr, bgr_entries
        )

    def create_instance(self, class_name: str, instance_address: int = 0x100000) -> PolymorphicObjectLayout:
        """Create simulated object instance layout with vptr initialized."""
        if class_name not in self.vtables:
            raise ValueError(f"Unknown class '{class_name}'. Available: {list(self.vtables.keys())}")

        obj = PolymorphicObjectLayout(class_name, instance_address)
        vtable = self.vtables[class_name]
        obj.add_vptr(vtable.vtable_base_address)
        obj.add_member("config_", 8, "const FrameBufferConfig&", "0x00007fffffffe000", "Reference pointer to config")
        self.objects[class_name] = obj
        return obj

    def dispatch(self, obj: PolymorphicObjectLayout, method_name: str, *args) -> tuple[int, str]:
        """
        Simulate dynamic dispatch:
        1. Read vptr at obj.instance_address (+0)
        2. Resolve vtable
        3. Lookup method slot
        4. Check for pure virtual trap (__cxa_pure_virtual)
        5. Invoke target function
        """
        vptr = obj.vptr_value
        # Find vtable matching vptr
        matching_vtable = None
        for vt in self.vtables.values():
            if vt.vtable_base_address == vptr:
                matching_vtable = vt
                break

        if not matching_vtable:
            raise RuntimeError(f"Corrupted vptr 0x{vptr:016x}: no matching vtable found!")

        entry = matching_vtable.lookup_function(method_name)
        if not entry:
            raise AttributeError(f"Method '{method_name}' not found in vtable for {matching_vtable.class_name}")

        self.dispatch_log.append(
            f"Dispatch: obj({obj.class_name}) -> vptr(0x{vptr:x}) -> slot(+{entry.offset_bytes}) "
            f"-> target(0x{entry.target_address:x}: {entry.symbol_name})"
        )

        if entry.is_pure_virtual or entry.symbol_name == "__cxa_pure_virtual":
            self.pure_virtual_call_count += 1
            raise PureVirtualCallError(
                f"FATAL: Pure virtual function call trap! Invoked slot '+{entry.offset_bytes}' "
                f"({entry.name}) on abstract class '{matching_vtable.class_name}' -> __cxa_pure_virtual()"
            )

        return (entry.target_address, entry.symbol_name)


class VPtrInspector:
    """Extracts and verifies raw binary pointers and vtable entries from emulated memory."""

    @staticmethod
    def inspect_vptr(raw_bytes: bytes) -> int:
        """Read 8-byte little-endian vptr from the head of an object buffer."""
        if len(raw_bytes) < 8:
            raise ValueError("Buffer must be at least 8 bytes to contain a 64-bit vptr.")
        return struct.unpack("<Q", raw_bytes[:8])[0]

    @staticmethod
    def disassemble_indirect_call(method_offset: int = 16) -> list[str]:
        """Generate annotated x86-64 assembly instructions for dynamic dispatch."""
        return [
            f"mov   rax, [rdi]        ; Step 1: Dereference this pointer (RDI) -> load vptr into RAX",
            f"mov   rax, [rax + {method_offset}]   ; Step 2: Index vtable at offset +{method_offset} -> load function address",
            f"call  rax               ; Step 3: Indirect call to resolved virtual method",
        ]


# ==============================================================================
# Chapter 4.4 Figures Generation
# ==============================================================================

def plot_fig4_07_vtable_memory_layout_and_dispatch(output_path: str) -> str:
    """
    Generate Fig 4.7: Vtable memory layout, vptr embedding, and x86-64 indirect call pipeline.
    """
    setup_plot_style()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig = plt.figure(figsize=(15, 11), facecolor="#F8F9FA")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # 1. Object In-Memory Structure (vptr + config_)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. オブジェクトメモリ構造と不可視メンバ `_vptr` (16 Bytes)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Diagram of Object
    obj_box = FancyBboxPatch((0.15, 0.48), 0.70, 0.42, boxstyle="round,pad=0.03",
                             facecolor="#E3F2FD", edgecolor="#1976D2", lw=2)
    ax1.add_patch(obj_box)
    ax1.text(0.50, 0.84, "RGBResv8BitPerColorPixelWriter インスタンス (16 Bytes)",
             ha="center", fontsize=9.5, fontweight="bold", color="#0D47A1")

    # Slot 1: vptr
    vptr_box = FancyBboxPatch((0.18, 0.65), 0.64, 0.14, boxstyle="round,pad=0.01",
                              facecolor="#BBDEFB", edgecolor="#1565C0", lw=1.5)
    ax1.add_patch(vptr_box)
    ax1.text(0.50, 0.72, "[+0x00] `_vptr` (8 Bytes ポインタ)\n-> 指向先: vtable for RGBWriter (+0x00)",
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0D47A1")

    # Slot 2: config_
    cfg_box = FancyBboxPatch((0.18, 0.48), 0.64, 0.14, boxstyle="round,pad=0.01",
                              facecolor="#C8E6C9", edgecolor="#2E7D32", lw=1.5)
    ax1.add_patch(cfg_box)
    ax1.text(0.50, 0.55, "[+0x08] `config_` 参照ポインタ (8 Bytes)\n-> 指向先: FrameBufferConfig 実体",
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1B5E20")

    # Note
    note1 = (
        "【重要ポイント】\n"
        "・仮想関数を1つでも宣言すると、コンパイラが先頭8バイトに `_vptr` を自動挿入\n"
        "・`sizeof(PixelWriter)` は参照(8B)だけでなく `_vptr`(8B) を含めて 16B に拡張\n"
        "・コンストラクタ実行時に `_vptr` が対応クラスの `vtable` アドレスで自動初期化"
    )
    ax1.text(0.08, 0.10, note1, fontsize=8.2, color="#37474F", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#90A4AE"))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # 2. Itanium C++ ABI Vtable Structure
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. Itanium C++ ABI 準拠の vtable 内部構造 (.rodata)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # VTable Layout
    slots = [
        ("-0x10", "Offset-to-Top", "0x0000000000000000", "#FFF9C4", "#FBC02D"),
        ("-0x08", "RTTI Pointer", "&_ZTI...PixelWriter", "#FFE0B2", "#FB8C00"),
        ("+0x00 [vptr]", "Complete Destructor", "&RGBWriter::~RGBWriter()", "#D1C4E9", "#5E35B1"),
        ("+0x08", "Deleting Destructor", "&RGBWriter::~RGBWriter()", "#D1C4E9", "#5E35B1"),
        ("+0x10", "Write() エントリ", "&RGBWriter::Write(x, y, c)", "#C8E6C9", "#2E7D32"),
    ]

    y_pos = 0.85
    for offset_str, slot_name, symbol_str, fcolor, ecolor in slots:
        sbox = FancyBboxPatch((0.08, y_pos - 0.08), 0.84, 0.09, boxstyle="round,pad=0.01",
                              facecolor=fcolor, edgecolor=ecolor, lw=1.5)
        ax2.add_patch(sbox)
        ax2.text(0.12, y_pos - 0.035, offset_str, fontsize=8.0, fontweight="bold", color="#212121")
        ax2.text(0.35, y_pos - 0.035, slot_name, fontsize=8.5, fontweight="bold", color="#0D47A1")
        ax2.text(0.68, y_pos - 0.035, symbol_str, fontsize=7.8, family="monospace", color="#37474F")
        y_pos -= 0.12

    note2 = (
        "※ `vptr` は `Complete Destructor` (+0x00) を指し、\n"
        "   負のオフセット (-0x08, -0x10) にメタデータ（RTTI・オフセット）を配置する構造"
    )
    ax2.text(0.08, 0.12, note2, fontsize=8.2, color="#B71C1C",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#EF5350"))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # 3. x86-64 Indirect Call Disassembly Pipeline
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. `pixel_writer->Write()` の逆アセンブル命令列 (3段階)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    asm_steps = [
        ("Step 1: vptr 読出", "mov  rax, [rdi]", "RDI (thisポインタ) の先頭8バイト (*_vptr) を RAX にロード"),
        ("Step 2: 関数アドレス読出", "mov  rax, [rax + 16]", "vtable の +16 バイト目 (Writeスロット) の関数ポインタを RAX にロード"),
        ("Step 3: 間接関数呼出", "call rax", "RAX が指す具象クラスの Write() 実装へジャンプ (間接呼出)"),
    ]

    ay = 0.80
    for title, code, desc in asm_steps:
        abox = FancyBboxPatch((0.06, ay - 0.11), 0.88, 0.13, boxstyle="round,pad=0.01",
                              facecolor="#E8EAF6", edgecolor="#3F51B5", lw=1.5)
        ax3.add_patch(abox)
        ax3.text(0.10, ay - 0.03, title, fontsize=8.2, fontweight="bold", color="#1A237E")
        ax3.text(0.38, ay - 0.03, code, fontsize=8.8, fontweight="bold", family="monospace", color="#B71C1C")
        ax3.text(0.10, ay - 0.08, desc, fontsize=7.8, color="#37474F")
        ay -= 0.18

    note3 = (
        "【System V AMD64 ABI レジスタ引数】\n"
        "・RDI: this (オブジェクト先頭アドレス)\n"
        "・ESI: x (描画X座標) | EDX: y (描画Y座標) | RCX: c (PixelColor)"
    )
    ax3.text(0.06, 0.10, note3, fontsize=8.0, color="#1B5E20",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#81C784"))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # 4. Dynamic Dispatch Flow Chart
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. 動的ディスパッチ (Dynamic Dispatch) の実行時シーケンス",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    stages = [
        ("呼び出し側コード", "`pixel_writer->Write(x, y, c);`", "#E1F5FE", "#0288D1"),
        ("ポインタ解決", "`pixel_writer` の指すオブジェクトメモリを参照", "#E0F2F1", "#00897B"),
        ("vtable参照", "`_vptr` を経由して `vtable for RGBWriter` を参照", "#FFF8E1", "#FFA000"),
        ("関数実行", "`RGBResv8BitPerColorPixelWriter::Write()` 実行", "#F1F8E9", "#689F38"),
    ]

    sy = 0.84
    for stitle, sdesc, sfcol, secol in stages:
        sbox = FancyBboxPatch((0.10, sy - 0.08), 0.80, 0.09, boxstyle="round,pad=0.01",
                              facecolor=sfcol, edgecolor=secol, lw=1.5)
        ax4.add_patch(sbox)
        ax4.text(0.14, sy - 0.035, stitle, fontsize=8.2, fontweight="bold", color="#212121")
        ax4.text(0.40, sy - 0.035, sdesc, fontsize=8.0, color="#0D47A1")

        if sy > 0.35:
            ax4.annotate("", xy=(0.50, sy - 0.09), xytext=(0.50, sy - 0.13),
                         arrowprops=dict(arrowstyle="<-", lw=1.5, color="#546E7A"))
        sy -= 0.15

    note4 = (
        "◎ メリット: 呼び出し側コードは具象型（RGB/BGR）を一切意識せず共通APIで描画可能\n"
        "◎ コスト: メモリアクセス2回 + 間接ジャンプ (BTB予測ヒット時はほぼ1サイクル)"
    )
    ax4.text(0.06, 0.10, note4, fontsize=8.0, color="#01579B",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1F5FE", edgecolor="#81D4FA"))

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_fig4_08_cxa_pure_virtual_and_abi(output_path: str) -> str:
    """
    Generate Fig 4.8: Pure virtual function call trap, __cxa_pure_virtual linker resolution,
    bare-metal vs hosted runtime, and performance trade-offs.
    """
    setup_plot_style()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig = plt.figure(figsize=(15, 11), facecolor="#F8F9FA")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # 1. Pure Virtual Function Trap Flow
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. 純粋仮想関数トラップ (`__cxa_pure_virtual`) 発生フロー",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    flow_text = (
        "【なぜ純粋仮想関数のスロットに実体が必要なのか？】\n\n"
        "1. 抽象クラス `PixelWriter` の vtable 作成:\n"
        "   `virtual void Write(...) = 0;` (純粋仮想関数)\n"
        "   スロットに `nullptr` を入れると、万が一の呼び出し時に NULL ポインタ参照\n"
        "   （Page Fault / 未定義動作）が発生してクラッシュ原因の特定が困難になる。\n\n"
        "2. Itanium C++ ABI の安全設計:\n"
        "   コンパイラは純粋仮想スロットに `&__cxa_pure_virtual` を自動格納する。\n\n"
        "3. 誤呼び出しのシナリオ（Pure Virtual Function Call Trap）:\n"
        "   ・基底クラスのコンストラクタ/デストラクタ内から仮想関数を直接/間接呼出\n"
        "   ・未初期化または破棄途中の派生オブジェクトをポインタ経由で操作\n"
        "   -> 実行時に `__cxa_pure_virtual()` が発動して安全にプログラムを停止！"
    )
    fbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                          facecolor="#FFFDE7", edgecolor="#FBC02D", lw=1.5)
    ax1.add_patch(fbox)
    ax1.text(0.07, 0.50, flow_text, fontsize=8.0, color="#E65100", linespacing=1.35)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    # 2. Host OS vs Bare-Metal Linker Resolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. ホストOS環境 vs ベアメタル (MikanOS) のリンク解決比較",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    link_text = (
        "【ベアメタル環境でリンクエラーが出る根本原因】\n\n"
        "■ 一般の Linux / macOS アプリ (Hosted Environment):\n"
        "  ・コンパイラが `libc++abi.a` や `libsupc++.a` を自動リンク\n"
        "  ・ランタイム内に `__cxa_pure_virtual` (標準エラーにメッセージ出力後 abort)\n"
        "    が用意されているため、リンクは正常に通過する。\n\n"
        "■ MikanOS カーネル (Bare-Metal / Freestanding Environment):\n"
        "  ・`-nostdlib`, `--freestanding` を指定して標準C++ランタイムを排除\n"
        "  ・リンカ `ld.lld` が未解決シンボルエラーを発行:\n"
        "    `undefined reference to '__cxa_pure_virtual'`\n\n"
        "■ MikanOS での解決策:\n"
        "  カーネルコード内に自前でスタブ関数を定義する！\n"
        "  `extern \"C\" void __cxa_pure_virtual() { while (1) __asm__(\"hlt\"); }`\n"
        "  -> リンクが通り、万一の純粋仮想呼び出し時も安全に CPU が HLT 停止！"
    )
    lbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                          facecolor="#EDE7F6", edgecolor="#673AB7", lw=1.5)
    ax2.add_patch(lbox)
    ax2.text(0.07, 0.50, link_text, fontsize=8.0, color="#311B92", linespacing=1.35)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # 3. Direct Call vs Indirect Call Cost Benchmark
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. 直接関数呼出 vs 仮想関数間接呼出 (BTBヒット時) コスト",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    categories = [
        "直接呼び出し\n(Direct call)",
        "仮想関数 (初回)\n(BTBミス + キャッシュミス)",
        "仮想関数 (ループ内)\n(BTBヒット + L1Dヒット)",
        "If-Else 分岐\n(予測失敗時ペナルティ)"
    ]
    cycles = [1.0, 18.0, 1.8, 15.0]
    colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]

    x = np.arange(len(categories))
    bars = ax3.bar(x, cycles, color=colors, width=0.55, edgecolor="#424242", lw=1)
    ax3.set_ylabel("典型所要クロックサイクル [Cycles]", fontsize=8.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=7.8)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar, c in zip(bars, cycles):
        ax3.text(bar.get_x() + bar.get_width() / 2, c + 0.5, f"~{c}c",
                 ha="center", fontsize=8.2, fontweight="bold")

    note3 = (
        "現代の x86-64 CPU (Skylake/Zen) では Branch Target Buffer (BTB) が\n"
        "間接分岐先を高度に学習するため、同一型へのループ描画ではほぼ直接呼出同等！"
    )
    ax3.text(0.05, 0.70, note3, transform=ax3.transAxes, fontsize=7.5,
             color="#0D47A1", bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#90CAF9"))

    # 4. Summary of Chapter 4 Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. 第4章 C++ OOP & vtable 導入の総括",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    summary_text = (
        "【第4章で獲得した OS カーネル設計技術】\n\n"
        "① 抽象化とポリモーフィズム (Polymorphism):\n"
        "   `PixelWriter` 基底クラスにより、RGB/BGR/将来のフォーマット差分を\n"
        "   カーネル描画ロジックから完全隠蔽。\n\n"
        "② 動的メモリ確保不要のオブジェクト構築 (Placement new):\n"
        "   ヒープ未初期化のベアメタル環境で静的バッファ上に安全にインスタンス構築。\n\n"
        "③ vtable と ABI レベルの動作メカニズム理解:\n"
        "   `_vptr` ポインタ、Itanium C++ ABI の負のオフセット、間接関数呼出命令、\n"
        "   純粋仮想関数のトラップ `__cxa_pure_virtual` の本質を解明。\n\n"
        "④ 次なる課題 (4.5節):\n"
        "   カーネルコードが大きくなると、単一メモリ領域への直配置では限界に直面。\n"
        "   ELF セグメントの正しい配置とゼロ初期化（.bss）が不可欠となる！"
    )
    sbox = FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02",
                          facecolor="#E8F5E9", edgecolor="#4CAF50", lw=1.5)
    ax4.add_patch(sbox)
    ax4.text(0.07, 0.50, summary_text, fontsize=8.0, color="#1B5E20", linespacing=1.30)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# Chapter 4.5: ELF Loader, PT_LOAD Segments & .bss Zero-Clear Simulator
# ==============================================================================

# ELF Header Ident indices and constants
EI_NIDENT = 16
ELF_MAGIC = b"\x7fELF"

ELFCLASSNONE = 0
ELFCLASS32 = 1
ELFCLASS64 = 2

ELFDATANONE = 0
ELFDATA2LSB = 1  # Little endian
ELFDATA2MSB = 2  # Big endian

EV_CURRENT = 1

ET_NONE = 0
ET_REL = 1
ET_EXEC = 2
ET_DYN = 3
ET_CORE = 4

EM_NONE = 0
EM_386 = 3
EM_X86_64 = 62

# Segment types (p_type)
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_SHLIB = 5
PT_PHDR = 6
PT_TLS = 7

# Segment flags (p_flags)
PF_X = 0x1  # Execute
PF_W = 0x2  # Write
PF_R = 0x4  # Read


@dataclass
class Elf64Ehdr:
    """ELF-64 Header (64 bytes).
    
    Offset  Size  Field
    0       16    e_ident
    16      2     e_type
    18      2     e_machine
    20      4     e_version
    24      8     e_entry
    32      8     e_phoff
    40      8     e_shoff
    48      4     e_flags
    52      2     e_ehsize
    54      2     e_phentsize
    56      2     e_phnum
    58      2     e_shentsize
    60      2     e_shnum
    62      2     e_shstrndx
    """
    e_ident: bytes = field(default_factory=lambda: b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8)
    e_type: int = ET_EXEC
    e_machine: int = EM_X86_64
    e_version: int = EV_CURRENT
    e_entry: int = 0x100000
    e_phoff: int = 64
    e_shoff: int = 0
    e_flags: int = 0
    e_ehsize: int = 64
    e_phentsize: int = 56
    e_phnum: int = 0
    e_shentsize: int = 64
    e_shnum: int = 0
    e_shstrndx: int = 0

    @classmethod
    def from_bytes(cls, data: bytes) -> Elf64Ehdr:
        """Parse 64-byte ELF header from binary data."""
        if len(data) < 64:
            raise ValueError(f"Data too short for Elf64_Ehdr: {len(data)} bytes (expected >= 64)")
        
        ident = data[:16]
        fields = struct.unpack("<HHIQQQIHHHHHH", data[16:64])
        return cls(
            e_ident=ident,
            e_type=fields[0],
            e_machine=fields[1],
            e_version=fields[2],
            e_entry=fields[3],
            e_phoff=fields[4],
            e_shoff=fields[5],
            e_flags=fields[6],
            e_ehsize=fields[7],
            e_phentsize=fields[8],
            e_phnum=fields[9],
            e_shentsize=fields[10],
            e_shnum=fields[11],
            e_shstrndx=fields[12],
        )

    def to_bytes(self) -> bytes:
        """Serialize Elf64_Ehdr into 64 bytes."""
        ident_padded = (self.e_ident + b"\x00" * 16)[:16]
        return struct.pack(
            "<16sHHIQQQIHHHHHH",
            ident_padded,
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.e_phoff,
            self.e_shoff,
            self.e_flags,
            self.e_ehsize,
            self.e_phentsize,
            self.e_phnum,
            self.e_shentsize,
            self.e_shnum,
            self.e_shstrndx,
        )

    @property
    def is_valid_magic(self) -> bool:
        """Check if magic is 0x7f 'E' 'L' 'F'."""
        return self.e_ident[:4] == ELF_MAGIC

    @property
    def is_64bit(self) -> bool:
        """Check if ELF class is 64-bit (ELFCLASS64 == 2)."""
        return len(self.e_ident) > 4 and self.e_ident[4] == ELFCLASS64

    @property
    def is_little_endian(self) -> bool:
        """Check if ELF data encoding is 2's complement little endian (ELFDATA2LSB == 1)."""
        return len(self.e_ident) > 5 and self.e_ident[5] == ELFDATA2LSB

    @property
    def is_x86_64(self) -> bool:
        """Check if target machine is AMD x86-64 (EM_X86_64 == 62)."""
        return self.e_machine == EM_X86_64


@dataclass
class Elf64Phdr:
    """ELF-64 Program Header (56 bytes).
    
    Offset  Size  Field
    0       4     p_type
    4       4     p_flags
    8       8     p_offset
    16      8     p_vaddr
    24      8     p_paddr
    32      8     p_filesz
    40      8     p_memsz
    48      8     p_align
    """
    p_type: int = PT_LOAD
    p_flags: int = PF_R | PF_X
    p_offset: int = 0
    p_vaddr: int = 0x100000
    p_paddr: int = 0x100000
    p_filesz: int = 0
    p_memsz: int = 0
    p_align: int = 0x1000

    @classmethod
    def from_bytes(cls, data: bytes) -> Elf64Phdr:
        """Parse 56-byte ELF program header from binary data."""
        if len(data) < 56:
            raise ValueError(f"Data too short for Elf64_Phdr: {len(data)} bytes (expected >= 56)")
        fields = struct.unpack("<IIQQQQQQ", data[:56])
        return cls(
            p_type=fields[0],
            p_flags=fields[1],
            p_offset=fields[2],
            p_vaddr=fields[3],
            p_paddr=fields[4],
            p_filesz=fields[5],
            p_memsz=fields[6],
            p_align=fields[7],
        )

    def to_bytes(self) -> bytes:
        """Serialize Elf64_Phdr into 56 bytes."""
        return struct.pack(
            "<IIQQQQQQ",
            self.p_type,
            self.p_flags,
            self.p_offset,
            self.p_vaddr,
            self.p_paddr,
            self.p_filesz,
            self.p_memsz,
            self.p_align,
        )

    @property
    def is_load(self) -> bool:
        """Check if segment is PT_LOAD."""
        return self.p_type == PT_LOAD

    @property
    def flags_str(self) -> str:
        """Human-readable segment permissions (e.g. 'R-X', 'RW-')."""
        r = "R" if (self.p_flags & PF_R) else "-"
        w = "W" if (self.p_flags & PF_W) else "-"
        x = "X" if (self.p_flags & PF_X) else "-"
        return f"{r}{w}{x}"

    @property
    def bss_size(self) -> int:
        """Calculate BSS size (p_memsz - p_filesz)."""
        return max(0, self.p_memsz - self.p_filesz)


class ElfLoaderSimulator:
    """Simulator of MikanOS UEFI bootloader ELF segment loading logic."""

    @staticmethod
    def parse_elf(elf_data: bytes) -> Tuple[Elf64Ehdr, List[Elf64Phdr]]:
        """Parse ELF header and all program headers from raw binary."""
        ehdr = Elf64Ehdr.from_bytes(elf_data)
        if not ehdr.is_valid_magic:
            raise ValueError("Invalid ELF magic number (expected \\x7fELF)")
        if not ehdr.is_64bit:
            raise ValueError("Not a 64-bit ELF binary (ELFCLASS64 required)")
        if not ehdr.is_x86_64:
            raise ValueError(f"Machine architecture {ehdr.e_machine} is not x86-64 (expected {EM_X86_64})")

        phdrs: List[Elf64Phdr] = []
        for i in range(ehdr.e_phnum):
            ph_offset = ehdr.e_phoff + i * ehdr.e_phentsize
            ph_data = elf_data[ph_offset : ph_offset + ehdr.e_phentsize]
            phdrs.append(Elf64Phdr.from_bytes(ph_data))
        return ehdr, phdrs

    @staticmethod
    def calc_load_address_range(ehdr: Elf64Ehdr, phdrs: List[Elf64Phdr]) -> Tuple[int, int]:
        """Compute [first, last) virtual address range for PT_LOAD segments.
        
        Corresponds to CalcLoadAddressRange() in MikanLoaderPkg/Main.c:
            *first = MAX_UINT64;
            *last = 0;
            for (Elf64_Half i = 0; i < ehdr->e_phnum; ++i) {
                if (phdr[i].p_type != PT_LOAD) continue;
                *first = MIN(*first, phdr[i].p_vaddr);
                *last = MAX(*last, phdr[i].p_vaddr + phdr[i].p_memsz);
            }
        """
        first = 0xFFFFFFFFFFFFFFFF
        last = 0
        has_load = False

        for phdr in phdrs:
            if phdr.p_type != PT_LOAD:
                continue
            has_load = True
            first = min(first, phdr.p_vaddr)
            last = max(last, phdr.p_vaddr + phdr.p_memsz)

        if not has_load:
            return (0, 0)
        return (first, last)

    @staticmethod
    def calc_required_pages(first_addr: int, last_addr: int, page_size: int = 4096) -> int:
        """Calculate number of 4KiB pages required to cover the address range.
        
        Formula from Main.c:
            num_pages = (kernel_last_addr - kernel_first_addr + 0xfff) / 0x1000
        """
        if last_addr <= first_addr:
            return 0
        span = last_addr - first_addr
        return (span + page_size - 1) // page_size

    @staticmethod
    def copy_load_segments(
        ehdr: Elf64Ehdr,
        phdrs: List[Elf64Phdr],
        elf_data: bytes,
        memory: bytearray,
        base_vaddr: int = 0x100000,
    ) -> Dict[str, Any]:
        """Simulate CopyLoadSegments from MikanLoaderPkg/Main.c.
        
        Copies segment file data to destination vaddr, and zeroes out the remaining
        BSS region (p_memsz - p_filesz).
        """
        loaded_segments_info = []

        for idx, phdr in enumerate(phdrs):
            if phdr.p_type != PT_LOAD:
                continue

            file_start = phdr.p_offset
            file_end = file_start + phdr.p_filesz
            if file_end > len(elf_data):
                raise ValueError(
                    f"Segment {idx} file range [0x{file_start:x}, 0x{file_end:x}) exceeds file size {len(elf_data)}"
                )

            seg_data = elf_data[file_start:file_end]

            mem_offset = phdr.p_vaddr - base_vaddr
            if mem_offset < 0 or (mem_offset + phdr.p_memsz) > len(memory):
                raise ValueError(
                    f"Segment {idx} memory range [0x{phdr.p_vaddr:x}, 0x{phdr.p_vaddr + phdr.p_memsz:x}) "
                    f"out of allocated buffer bounds [0x{base_vaddr:x}, 0x{base_vaddr + len(memory):x})"
                )

            # CopyMem(vaddr, segm_in_file, filesz)
            memory[mem_offset : mem_offset + phdr.p_filesz] = seg_data

            # SetMem(vaddr + filesz, remain_bytes, 0)
            remain_bytes = phdr.p_memsz - phdr.p_filesz
            if remain_bytes > 0:
                bss_start = mem_offset + phdr.p_filesz
                bss_end = bss_start + remain_bytes
                memory[bss_start:bss_end] = b"\x00" * remain_bytes

            loaded_segments_info.append({
                "index": idx,
                "vaddr": phdr.p_vaddr,
                "filesz": phdr.p_filesz,
                "memsz": phdr.p_memsz,
                "bss_size": remain_bytes,
                "flags": phdr.flags_str,
            })

        return {
            "loaded_count": len(loaded_segments_info),
            "segments": loaded_segments_info,
        }

    @staticmethod
    def build_mock_elf(
        entry: int = 0x100000,
        segments: Optional[List[Dict[str, Any]]] = None,
        machine: int = EM_X86_64,
    ) -> bytes:
        """Construct synthetic ELF64 executable binary for tests and notebooks."""
        if segments is None:
            segments = [
                {
                    "type": PT_LOAD,
                    "flags": PF_R | PF_X,
                    "vaddr": 0x100000,
                    "data": b"\x48\x31\xc0\xc3" + b"\x90" * 252,  # 256 bytes code
                    "memsz": 0x200,  # 512 bytes total (256 bytes code + 256 bytes BSS)
                    "align": 0x1000,
                }
            ]

        ehdr_size = 64
        phdr_size = 56
        phnum = len(segments)
        phoff = ehdr_size

        curr_offset = ehdr_size + phnum * phdr_size
        align_padding = (16 - (curr_offset % 16)) % 16
        curr_offset += align_padding

        phdrs_bytes = bytearray()
        data_blocks = bytearray()

        for seg in segments:
            seg_type = seg.get("type", PT_LOAD)
            seg_flags = seg.get("flags", PF_R | PF_X)
            seg_vaddr = seg.get("vaddr", 0x100000)
            seg_data = seg.get("data", b"")
            filesz = len(seg_data)
            memsz = seg.get("memsz", filesz)
            align = seg.get("align", 0x1000)

            phdr = Elf64Phdr(
                p_type=seg_type,
                p_flags=seg_flags,
                p_offset=curr_offset,
                p_vaddr=seg_vaddr,
                p_paddr=seg_vaddr,
                p_filesz=filesz,
                p_memsz=memsz,
                p_align=align,
            )
            phdrs_bytes.extend(phdr.to_bytes())
            data_blocks.extend(seg_data)
            curr_offset += filesz

        ehdr = Elf64Ehdr(
            e_type=ET_EXEC,
            e_machine=machine,
            e_entry=entry,
            e_phoff=phoff,
            e_ehsize=ehdr_size,
            e_phentsize=phdr_size,
            e_phnum=phnum,
        )

        elf_bytes = bytearray()
        elf_bytes.extend(ehdr.to_bytes())
        elf_bytes.extend(phdrs_bytes)
        elf_bytes.extend(b"\x00" * align_padding)
        elf_bytes.extend(data_blocks)
        return bytes(elf_bytes)


def generate_fig4_09_elf_segment_structure(
    output_path: str = "4/result/fig4_09_elf_segment_structure.png"
) -> str:
    """Generate high-DPI architecture diagram showing ELF-64 header, Phdrs, and segment mapping."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(15, 9), facecolor="#FAFAFA")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.25)

    # 1. ELF File Binary Layout vs Memory Address Space Mapping
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. ELF-64 ファイルバイナリ構造と仮想メモリ空間へのセグメント写像 (ELF Layout & Segment Mapping)",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=12)

    # Drawing File Layout on Left
    ax1.text(0.04, 0.90, "【ELF ファイル上の構造 (オンディスク)】", fontsize=9.5, fontweight="bold", color="#37474F")
    
    file_blocks = [
        ("Elf64_Ehdr\n(64 Bytes)", 0.04, 0.50, 0.12, 0.32, "#E3F2FD", "#1565C0", "マジック数, e_entry,\ne_phoff (32B), e_phnum"),
        ("Elf64_Phdr[0]\n(56 Bytes)", 0.17, 0.50, 0.11, 0.32, "#FFF3E0", "#E65100", "p_type = PT_LOAD\np_flags = R-X (コード)"),
        ("Elf64_Phdr[1]\n(56 Bytes)", 0.29, 0.50, 0.11, 0.32, "#FFF3E0", "#E65100", "p_type = PT_LOAD\np_flags = RW- (データ)"),
        (".text / .rodata\n(Code & Readonly)", 0.41, 0.50, 0.14, 0.32, "#E8F5E9", "#2E7D32", "機械語命令, 文字列定数\n(p_filesz == p_memsz)"),
        (".data\n(Initialized)", 0.56, 0.50, 0.11, 0.32, "#F3E5F5", "#6A1B9A", "初期値付き大域変数\n(p_filesz < p_memsz)"),
    ]

    for title, x, y, w, h, bg, border, desc in file_blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                             facecolor=bg, edgecolor=border, lw=1.5)
        ax1.add_patch(box)
        ax1.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                 fontsize=8.2, fontweight="bold", color=border)
        ax1.text(x + w / 2, y + h * 0.25, desc, ha="center", va="center",
                 fontsize=6.8, color="#424242", linespacing=1.2)

    # Drawing Memory Space on Right
    ax1.text(0.72, 0.90, "【メモリ上の仮想アドレス空間 (RAM)】", fontsize=9.5, fontweight="bold", color="#37474F")
    
    mem_blocks = [
        ("Code Segment\n(0x100000)\n.text / .rodata", 0.72, 0.52, 0.11, 0.30, "#E8F5E9", "#2E7D32", "R-X (実行可能)\nfilesz = memsz"),
        ("Data Segment\n(0x101000)\n.data", 0.84, 0.52, 0.065, 0.30, "#F3E5F5", "#6A1B9A", "RW-\nfilesz"),
        (".bss 領域\n(Zero-Cleared)", 0.91, 0.52, 0.065, 0.30, "#FFEBEE", "#C62828", "RW-\nmemsz - filesz\nゼロ埋め！"),
    ]

    for title, x, y, w, h, bg, border, desc in mem_blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                             facecolor=bg, edgecolor=border, lw=1.5)
        ax1.add_patch(box)
        ax1.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                 fontsize=7.8, fontweight="bold", color=border)
        ax1.text(x + w / 2, y + h * 0.25, desc, ha="center", va="center",
                 fontsize=6.8, color="#424242", linespacing=1.2)

    # Arrows from file to memory
    arrow1 = patches.FancyArrowPatch((0.48, 0.50), (0.75, 0.40),
                                     arrowstyle="->", connectionstyle="arc3,rad=-0.15",
                                     color="#2E7D32", lw=2, mutation_scale=15)
    ax1.add_patch(arrow1)
    ax1.text(0.60, 0.40, "CopyMem: p_offset → p_vaddr", fontsize=7.8, color="#2E7D32", fontweight="bold")

    arrow2 = patches.FancyArrowPatch((0.61, 0.50), (0.86, 0.40),
                                     arrowstyle="->", connectionstyle="arc3,rad=-0.15",
                                     color="#6A1B9A", lw=2, mutation_scale=15)
    ax1.add_patch(arrow2)
    ax1.text(0.70, 0.32, "CopyMem: .data", fontsize=7.8, color="#6A1B9A", fontweight="bold")

    # Bottom notes
    note1 = (
        "[POINT] 重要: ELFファイル中のヘッダ (Elf64_Ehdr, Elf64_Phdr) 自体はカーネル実行には不要！\n"
        "セグメント (PT_LOAD) だけを p_vaddr へコピーし、未初期化の .bss (p_memsz - p_filesz) は 0 で埋めるのが正規のOSローダ！"
    )
    ax1.text(0.04, 0.10, note1, fontsize=8.2, color="#0D47A1",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#90CAF9"))

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. Elf64_Ehdr (64-byte Header Bit Layout)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. Elf64_Ehdr (64バイト) フィールド構造と役割",
                  fontsize=11, fontweight="bold", color="#0D47A1", pad=10)

    ehdr_fields = [
        ("0x00 (16B)", "e_ident[16]", "\\x7fELF, 64-bit(2), LittleEndian(1), SYSV(0)"),
        ("0x10 (2B)", "e_type", "ET_EXEC (2) または ET_DYN (3)"),
        ("0x12 (2B)", "e_machine", "EM_X86_64 (62 = 0x3E)"),
        ("0x14 (4B)", "e_version", "EV_CURRENT (1)"),
        ("0x18 (8B)", "e_entry", "エントリポイント仮想アドレス (KernelMain)"),
        ("0x20 (8B)", "e_phoff", "プログラムヘッダテーブルのファイルオフセット"),
        ("0x30 (2B)", "e_phentsize", "プログラムヘッダ1個のバイトサイズ (56B)"),
        ("0x38 (2B)", "e_phnum", "プログラムヘッダの個数"),
    ]

    y_pos = 0.88
    for off, name, val in ehdr_fields:
        ax2.text(0.04, y_pos, off, fontsize=7.8, family="monospace", color="#546E7A", fontweight="bold")
        ax2.text(0.26, y_pos, name, fontsize=8.2, family="monospace", color="#0D47A1", fontweight="bold")
        ax2.text(0.50, y_pos, val, fontsize=7.8, color="#263238")
        y_pos -= 0.11

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    # 3. Elf64_Phdr (56-byte Program Header Bit Layout)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. Elf64_Phdr (56バイト) フィールド構造とロード数理",
                  fontsize=11, fontweight="bold", color="#0D47A1", pad=10)

    phdr_fields = [
        ("0x00 (4B)", "p_type", "PT_LOAD (1: メモリ展開対象) / PT_DYNAMIC (2)"),
        ("0x04 (4B)", "p_flags", "PF_X(1), PF_W(2), PF_R(4) のビット論理和"),
        ("0x08 (8B)", "p_offset", "ファイル先頭からのオフセット (CopyMem 元)"),
        ("0x10 (8B)", "p_vaddr", "配置先仮想アドレス (CopyMem 先)"),
        ("0x20 (8B)", "p_filesz", "ファイル上の実データ長 (CopyMem 長さ)"),
        ("0x28 (8B)", "p_memsz", "メモリ上所要サイズ (memsz >= filesz)"),
        ("0x30 (8B)", "p_align", "アライメント境界 (例: 0x1000 = 4096B)"),
    ]

    y_pos = 0.88
    for off, name, val in phdr_fields:
        ax3.text(0.04, y_pos, off, fontsize=7.8, family="monospace", color="#546E7A", fontweight="bold")
        ax3.text(0.24, y_pos, name, fontsize=8.2, family="monospace", color="#E65100", fontweight="bold")
        ax3.text(0.44, y_pos, val, fontsize=7.8, color="#263238")
        y_pos -= 0.115

    note3 = "数理則: .bss サイズ = p_memsz - p_filesz\n(filesz 分をコピーした後、差分を SetMem で 0 クリア)"
    ax3.text(0.04, 0.08, note3, fontsize=7.8, color="#C2185B",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCE4EC", edgecolor="#F48FB1"))

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_fig4_10_pt_load_and_bss_clearing(
    output_path: str = "4/result/fig4_10_pt_load_and_bss_clearing.png"
) -> str:
    """Generate high-DPI architecture diagram showing 2-stage loading and .bss zero-clearing."""
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(15, 9), facecolor="#FAFAFA")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.1], hspace=0.30, wspace=0.25)

    # 1. Naive Direct Copy vs 2-Stage ELF Loading Architecture
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. 従来のベタコピー (day03a) の致命的欠陥",
                  fontsize=11, fontweight="bold", color="#C62828", pad=10)

    flaw_text = (
        "【従来の単純ファイル読み込み方式】\n"
        "gBS->AllocatePages(0x100000, FileSize);\n"
        "kernel_file->Read(kernel_file, FileSize, 0x100000);\n\n"
        "[NG] 致命的課題 1: ELFヘッダ混入\n"
        "   0x100000 先頭に 64B の Elf64_Ehdr やプログラムヘッダが直配置され、\n"
        "   リンカが想定したセグメント仮想アドレスと完全に乖離する。\n\n"
        "[NG] 致命的課題 2: .bss 領域が未初期化・未確保\n"
        "   ELFファイル上に実データを持たない未初期化大域変数の空間が確保されず、\n"
        "   後続メモリの破壊やゴミデータによるカーネル暴走を招く！\n\n"
        "[NG] 致命的課題 3: 複数セグメントのアライメント崩壊\n"
        "   ファイル上のオフセットとメモリ上のページ境界（4KiB）の不一致に対応不能。"
    )
    fbox = FancyBboxPatch((0.04, 0.05), 0.92, 0.90, boxstyle="round,pad=0.02",
                          facecolor="#FFEBEE", edgecolor="#EF5350", lw=1.5)
    ax1.add_patch(fbox)
    ax1.text(0.07, 0.50, flaw_text, fontsize=7.8, color="#B71C1C", linespacing=1.28)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. Correct 2-Stage ELF Loader (day04d) Sequence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. 改良型 2段階ローダ (day04d) の処理フロー",
                  fontsize=11, fontweight="bold", color="#2E7D32", pad=10)

    flow_steps = [
        ("Step 1", "AllocatePool で一時バッファ確保 & ELFファイル丸ごと読み込み", "#E8F5E9", "#2E7D32"),
        ("Step 2", "CalcLoadAddressRange で全 PT_LOAD の [first, last) を走査", "#E3F2FD", "#1565C0"),
        ("Step 3", "num_pages 計算 & AllocatePages(first) で正確なRAMページ確保", "#FFF3E0", "#E65100"),
        ("Step 4", "CopyLoadSegments: CopyMem(filesz) + SetMem(remain_bytes, 0)", "#F3E5F5", "#6A1B9A"),
        ("Step 5", "FreePool で一時バッファ解放 & ExitBootServices → entry_addr 実行", "#E0F2F1", "#00695C"),
    ]

    y_pos = 0.84
    for step_num, desc, bg, border in flow_steps:
        box = FancyBboxPatch((0.04, y_pos - 0.06), 0.92, 0.12, boxstyle="round,pad=0.015",
                             facecolor=bg, edgecolor=border, lw=1.2)
        ax2.add_patch(box)
        ax2.text(0.08, y_pos, step_num, fontsize=8.2, fontweight="bold", color=border)
        ax2.text(0.22, y_pos, desc, fontsize=7.6, color="#263238")
        y_pos -= 0.165

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")

    # 3. .bss Zero-Clear Math & Memory Visualization
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. PT_LOAD セグメントと .bss ゼロクリア (Zero-Clearing) の詳細メカニズム",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=12)

    # Segment visual bar
    ax3.text(0.05, 0.85, "メモリ配置アドレス: phdr->p_vaddr", fontsize=8.5, fontweight="bold", color="#37474F")
    
    # Block 1: File data (filesz)
    b1 = FancyBboxPatch((0.05, 0.45), 0.50, 0.32, boxstyle="square,pad=0.0",
                        facecolor="#C8E6C9", edgecolor="#2E7D32", lw=2)
    ax3.add_patch(b1)
    ax3.text(0.30, 0.65, "実データ領域: p_filesz バイト\n(CopyMem(vaddr, segm_in_file, p_filesz))",
             ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1B5E20")
    ax3.text(0.30, 0.52, "初期値ありデータ (.text, .rodata, .data の初期値)",
             ha="center", va="center", fontsize=7.8, color="#388E3C")

    # Block 2: BSS data (remain_bytes = memsz - filesz)
    b2 = FancyBboxPatch((0.55, 0.45), 0.35, 0.32, boxstyle="square,pad=0.0",
                        facecolor="#FFCDD2", edgecolor="#C62828", lw=2, hatch="//")
    ax3.add_patch(b2)
    ax3.text(0.725, 0.65, "BSS 未初期化領域: p_memsz - p_filesz バイト\n(SetMem(vaddr + p_filesz, remain_bytes, 0))",
             ha="center", va="center", fontsize=8.8, fontweight="bold", color="#B71C1C")
    ax3.text(0.725, 0.52, "0 クリア必須！(C/C++規格で未初期化大域変数は 0 保証)",
             ha="center", va="center", fontsize=7.8, color="#D32F2F")

    # Brackets & annotations
    ax3.annotate("", xy=(0.05, 0.40), xytext=(0.55, 0.40),
                 arrowprops=dict(arrowstyle="<->", lw=1.5, color="#2E7D32"))
    ax3.text(0.30, 0.33, "p_filesz", ha="center", fontsize=9, fontweight="bold", color="#2E7D32")

    ax3.annotate("", xy=(0.55, 0.40), xytext=(0.90, 0.40),
                 arrowprops=dict(arrowstyle="<->", lw=1.5, color="#C62828"))
    ax3.text(0.725, 0.33, "remain_bytes = p_memsz - p_filesz", ha="center", fontsize=9, fontweight="bold", color="#C62828")

    ax3.annotate("", xy=(0.05, 0.23), xytext=(0.90, 0.23),
                 arrowprops=dict(arrowstyle="<->", lw=1.8, color="#0D47A1"))
    ax3.text(0.475, 0.16, "p_memsz (セグメント全体のメモリ所要量)", ha="center", fontsize=9.5, fontweight="bold", color="#0D47A1")

    # Entry point pointer note
    note_entry = (
        "エントリポイント導出:\n"
        "UINT64 entry_addr = *(UINT64*)(kernel_first_addr + 24);\n"
        "※ Elf64_Ehdr の offset 24 は e_entry (KernelMain の仮想アドレス)"
    )
    ax3.text(0.05, 0.04, note_entry, fontsize=7.8, color="#01579B",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1F5FE", edgecolor="#81D4FA"))

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# =============================================================================
# Chapter 4 Exercises: Comprehensive Verification & Quiz Engine
# =============================================================================

class Chapter4Quiz:
    """Comprehensive multiple-choice quiz engine covering Sections 4.1 to 4.5."""

    def __init__(self) -> None:
        self.questions: list[QuizQuestion] = [
            QuizQuestion(
                question_id=1,
                section="4.1",
                topic="Makefileの自動変数と依存関係解決 ($@, $<, $^)",
                question="GNU Make における主要な自動変数の役割の組み合わせとして正しいものはどれか？",
                options=[
                    "$@ は最初の依存ファイル名、$< はターゲット名、$^ はMakefile自体のパスを表す",
                    "$@ はターゲット名、$< は最初の依存ファイル名、$^ はすべての依存ファイル名のリストを表す",
                    "$@ は全オブジェクトファイル名、$< はコンパイルオプション、$^ は出力先ディレクトリを表す",
                    "$@ はターゲット名、$< はリンカスクリプト名、$^ はC++標準規格バージョンを表す",
                ],
                correct_index=1,
                explanation="Makeの自動変数は、ターゲット名を $@、最初の前提条件（依存ファイル）を $<、すべての前提条件の空白区切りリストを $^ で表します。これらを用いることで汎用的なパターンルール（例: %.o: %.cpp）を簡潔に記述できます。",
            ),
            QuizQuestion(
                question_id=2,
                section="4.1",
                topic="ベアメタル C++ コンパイルフラグ (-ffreestanding, -mno-red-zone 等)",
                question="MikanOS カーネルビルドで指定されるコンパイラフラグとその理由の組み合わせとして正しいものはどれか？",
                options=[
                    "-ffreestanding は標準ライブラリの全関数を静的リンクし、-fexceptions はカーネルクラッシュを防ぐ",
                    "-ffreestanding は標準ライブラリ（main関数や標準C関数）を仮定せず、-mno-red-zone は割り込みによるスタック破壊を防ぎ、-fno-exceptions / -fno-rtti はOS未初期化時の例外機構や型情報によるオーバーヘッドを排除する",
                    "-mno-red-zone はグラフィックの赤色描画を高速化し、-std=c++11 はレガシーBIOSとの互換性を確保する",
                    "-fno-rtti はコンパイル速度を10倍にするためのデバッグ専用オプションである",
                ],
                correct_index=1,
                explanation="カーネルなどのベアメタル環境では、ホストOSの標準ランタイムが存在しないため -ffreestanding を指定し、非同期割り込みによるスタック直下（128バイト）の破壊を防ぐため -mno-red-zone が必須です。またランタイムサポートが必要な例外 (-fno-exceptions) や動的型情報 (-fno-rtti) を無効化してバイナリを軽量・自律化します。",
            ),
            QuizQuestion(
                question_id=3,
                section="4.2",
                topic="FrameBufferConfig と VRAM 2次元座標オフセット計算数理",
                question="FrameBufferConfig において、解像度 800x600、走査線幅 pixels_per_scan_line = 832 のとき、ピクセル座標 (x=100, y=50) のバイトオフセット計算式と結果として正しいものはどれか？ (1ピクセル=4バイト)",
                options=[
                    "オフセット = 4 * (50 * 800 + 100) = 160,400 バイト",
                    "オフセット = 4 * (50 * 832 + 100) = 166,800 バイト",
                    "オフセット = 50 * 832 + 100 = 41,700 バイト",
                    "オフセット = 4 * (100 * 832 + 50) = 333,000 バイト",
                ],
                correct_index=1,
                explanation="走査線パディングが存在する場合、1行あたりのピクセル数は horizontal_resolution ではなく pixels_per_scan_line (ppsl) を用いる必要があります。バイトオフセットは `4 * (y * ppsl + x)` となり、`4 * (50 * 832 + 100) = 4 * (41600 + 100) = 4 * 41700 = 166,800 バイト` です。",
            ),
            QuizQuestion(
                question_id=4,
                section="4.2",
                topic="System V AMD64 ABI と Microsoft x64 ABI の比較",
                question="カーネル（System V AMD64 ABI）と UEFI（Microsoft x64 ABI）における関数引数引き渡しレジスタの仕様として正しいものはどれか？",
                options=[
                    "System V は第1〜第6引数を RDI, RSI, RDX, RCX, R8, R9 で渡し、MS x64 は第1〜第4引数を RCX, RDX, R8, R9 で渡し32バイトのシャドウスペースを要求する",
                    "System V はすべての引数をスタック経由で渡し、MS x64 は RAX, RBX, RCX, RDX を使用する",
                    "System V も MS x64 も完全に同一の引数レジスタ順序を採用しており、ABI変換は不要である",
                    "MS x64 は第1引数を RDI、第2引数を RSI で渡し、シャドウスペースは存在しない",
                ],
                correct_index=0,
                explanation="System V AMD64 ABI では第1引数から順に RDI, RSI, RDX, RCX, R8, R9 を使用します。対して Microsoft x64 ABI（UEFI標準）では RCX, RDX, R8, R9 を使用し、呼び出し側がスタック上に 32バイトのシャドウスペース（Home Space）を確保する規約になっています。",
            ),
            QuizQuestion(
                question_id=5,
                section="4.2",
                topic="画面白クリアと矩形描画ループ",
                question="全画面白クリアおよび矩形描画のVRAM走査アルゴリズムについて正しいものはどれか？",
                options=[
                    "白クリアはすべてのピクセルバイトに 0x00 を書き込み、黒クリアは 0xFF を書き込む",
                    "白クリアは各ピクセルの RGB 各成分に 255 (0xFF) を書き込み、矩形描画は外側 y、内側 x の2重ループで (y * ppsl + x) のピクセルアドレスへ描画色を書き込む",
                    "矩形描画ではストライド (ppsl) を無視して y * horizontal_resolution でインデックス計算する",
                    "RGB形式とBGR形式で白 (255, 255, 255) を描く際のバイト列値は全く異なる値になる",
                ],
                correct_index=1,
                explanation="白クリアは光の3原色すべてが最大（R=255, G=255, B=255）となるため、RGB形式でもBGR形式でも色バイトは [255, 255, 255, 0] となります。矩形描画は外側 y ループ・内側 x ループで走査線ストライド ppsl を考慮してアドレスを算出します。",
            ),
            QuizQuestion(
                question_id=6,
                section="4.3",
                topic="抽象基底クラス PixelWriter と純粋仮想関数",
                question="C++における抽象基底クラス PixelWriter の宣言と設計意図として正しいものはどれか？",
                options=[
                    "すべてのメンバ変数を private にし、派生クラスからのアクセスを一切禁止する",
                    "純粋仮想関数 `virtual void Write(int x, int y, const PixelColor& c) = 0;` を持ち、画面描画の共通インターフェースを定義しつつ、直接のインスタンス化を禁止する",
                    "PixelWriter はテンプレートクラスであり、仮想関数テーブル (vtable) は一切生成されない",
                    "純粋仮想関数を持つクラスはコンパイルエラーになるため、すべての仮想関数に空の波括弧 `{}` を実装しなければならない",
                ],
                correct_index=1,
                explanation="`virtual void Write(...) = 0;` のように `= 0` を指定した仮想関数を純粋仮想関数（pure virtual function）と呼び、これを持つクラスは抽象クラス（abstract class）となります。具象派生クラス（RGB/BGRPixelWriter）がこれをオーバーライドすることで多態性（ポリモーフィズム）を実現します。",
            ),
            QuizQuestion(
                question_id=7,
                section="4.3",
                topic="ベアメタル環境における配置 new (Placement new)",
                question="MikanOS 第4章で `new (pixel_writer_buf) RGBResv8BitPerColorPixelWriter(config);` のように配置 new を使用した決定的な理由は何ですか？",
                options=[
                    "配置 new を使うと、コンパイラがマルチスレッドセーフなヒープメモリを自動生成してくれるから",
                    "カーネル初期化時点では動的メモリアロケータ（malloc や通常の new）が存在しないため、静的に確保したバッファのアドレス上に直接オブジェクトのコンストラクタを呼び出す必要があるから",
                    "通常 new は C++20 で非推奨となり、配置 new が言語標準の必須構文になったから",
                    "配置 new は UEFI BIOS のファームウェアヒープ領域を自動的に借りてくる命令だから",
                ],
                correct_index=1,
                explanation="ベアメタル・フリースタンディング環境では、OSのメモリ管理機能がまだ完成していないため、ヒープを消費する通常の new はリンクエラーまたはクラッシュを引き起こします。配置 new（Placement new）は追加の動的メモリを一切確保せず、指定した既存メモリ領域上に直接コンストラクタを実行するため、初期化フェーズに最適です。",
            ),
            QuizQuestion(
                question_id=8,
                section="4.4",
                topic="仮想関数テーブル (vtable) と Itanium C++ ABI メモリレイアウト",
                question="Itanium C++ ABI における仮想関数呼び出し（vtable と _vptr）のメモリ構造と機械語展開として正しいものはどれか？",
                options=[
                    "オブジェクトの先頭 8バイトに vtable へのポインタ (_vptr) が格納され、間接呼び出しは `mov rax, [rdi]; call [rax + offset]` のように展開される",
                    "vtable は各インスタンスの末尾に埋め込まれ、インスタンスごとに仮想関数ポインタ配列がコピーされる",
                    "_vptr は 32ビットレジスタ EAX に常駐し、RAM上には一切配置されない",
                    "仮想関数呼び出しはコンパイル時にすべてインライン展開されるため、vtable というテーブルはメモリ上に存在しない",
                ],
                correct_index=0,
                explanation="Itanium C++ ABI では、仮想関数を持つクラスのインスタンス先頭に暗黙のポインタ `_vptr`（仮想関数テーブルのアドレス）が配置されます。仮想関数呼び出し時は、第1引数 RDI（thisポインタ）から `_vptr` をロードし、テーブル内の特定スロット（関数ポインタ）を間接 call します。",
            ),
            QuizQuestion(
                question_id=9,
                section="4.4",
                topic="純粋仮想関数トラップ (__cxa_pure_virtual) の役割",
                question="`__cxa_pure_virtual` シンボルの役割と、ベアメタルOSにおける推奨実装について正しいものはどれか？",
                options=[
                    "未実装の純粋仮想関数が誤って呼び出された際に呼ばれるランタイム関数であり、ベアメタルでは標準出力や例外が使えないため `while (1) __asm__(\"hlt\");` などでCPUを安全に停止させる",
                    "C++のコンパイラ最適化フラグであり、Makefile 内で定義しなければならないマクロである",
                    "UEFI ブートローダがカーネルの仮想関数テーブルをリロケーションするためのAPIである",
                    "このシンボルを定義すると、純粋仮想関数が自動的にデフォルト実装に置き換わる",
                ],
                correct_index=0,
                explanation="C++ ABI では、基底クラスのコンストラクタ/デストラクタから純粋仮想関数が呼ばれた場合などの異常系ハンドラとして `__cxa_pure_virtual` が参照されます。標準 C++ ランタイムがないベアメタル環境ではリンクエラーになるため、自作カーネル側で `extern \"C\" void __cxa_pure_virtual() { while (1) __asm__(\"hlt\"); }` を定義して安全に停止させます。",
            ),
            QuizQuestion(
                question_id=10,
                section="4.5",
                topic="ELF64 の PT_LOAD セグメント配置と .bss ゼロクリア",
                question="ELF64 ローダ (day04d) において、プログラムヘッダの p_memsz と p_filesz の関係と .bss ゼロクリア処理について正しいものはどれか？",
                options=[
                    "p_filesz は常に p_memsz より大きく、余分なファイル部分はメモリから切り捨てられる",
                    "p_memsz > p_filesz のとき、差分 (p_memsz - p_filesz) バイトは未初期化変数領域 (.bss) であり、ファイル中には存在しないため、メモリ配置時に 0 で埋める（ゼロクリアする）必要がある",
                    ".bss セクションはゼロクリアしてはならず、ガベージデータのまま保持しなければならない",
                    "ELFファイル内の全セグメントは常に 0x100000 固定でロードされ、p_vaddr は無視される",
                ],
                correct_index=1,
                explanation="ELF実行可能ファイルでは、ファイルサイズ削減のため未初期化大域変数領域（.bss）はディスク上にはデータを持たず `p_memsz > p_filesz` として表現されます。C/C++言語仕様により未初期化変数はゼロ初期化が保証されているため、ローダは `[p_vaddr + p_filesz, p_vaddr + p_memsz)` の領域を明示的にゼロクリアする必要があります。",
            ),
        ]

    def evaluate(self, answers: dict[int, int]) -> dict[str, any]:
        """Evaluate user answers against correct solutions."""
        correct_count = 0
        details = []
        section_correct: dict[str, int] = {f"4.{i}": 0 for i in range(1, 6)}
        section_total: dict[str, int] = {f"4.{i}": 0 for i in range(1, 6)}

        for q in self.questions:
            section_total[q.section] += 1
            user_ans = answers.get(q.question_id)
            is_correct = (user_ans == q.correct_index)
            if is_correct:
                correct_count += 1
                section_correct[q.section] += 1

            details.append({
                "question_id": q.question_id,
                "section": q.section,
                "topic": q.topic,
                "is_correct": is_correct,
                "user_answer": user_ans,
                "correct_answer": q.correct_index,
                "explanation": q.explanation,
            })

        percentage = (correct_count / len(self.questions)) * 100.0
        passed = (percentage >= 80.0)

        mastery_rates = {}
        for sec in section_total:
            total = section_total[sec]
            corr = section_correct[sec]
            mastery_rates[sec] = (corr / total * 100.0) if total > 0 else 0.0

        return {
            "total_questions": len(self.questions),
            "correct_count": correct_count,
            "percentage": percentage,
            "passed": passed,
            "mastery_rates": mastery_rates,
            "details": details,
        }


class Chapter4TheoryValidator:
    """Validation helper for Chapter 4 systems theory, Makefile, ABI, OOP, and ELF loading."""

    @staticmethod
    def validate_makefile_rules(rules: dict[str, str]) -> tuple[bool, str]:
        """Validate GNU Makefile automatic variable mappings and .PHONY concept."""
        target_var = rules.get("target_var", "").strip()
        first_dep_var = rules.get("first_dep_var", "").strip()
        all_deps_var = rules.get("all_deps_var", "").strip()
        phony_purpose = rules.get("phony_purpose", "").lower()

        errors = []
        if target_var != "$@":
            errors.append(f"target_var: 期待値 '$@', 入力値 '{target_var}'")
        if first_dep_var != "$<":
            errors.append(f"first_dep_var: 期待値 '$<', 入力値 '{first_dep_var}'")
        if all_deps_var != "$^":
            errors.append(f"all_deps_var: 期待値 '$^', 入力値 '{all_deps_var}'")
        if not ("ファイル" in phony_purpose or "同名" in phony_purpose or "衝突" in phony_purpose or "file" in phony_purpose):
            errors.append("phony_purpose: 同名ファイルとの衝突防止または非ファイルターゲットの保護であるべきです")

        if not errors:
            return True, "正解！Makefileの自動変数 ($@, $<, $^) および .PHONY の役割を正確に同定しました。"
        return False, f"不正解です。不一致項目: {'; '.join(errors)}"

    @staticmethod
    def validate_baremetal_cxx_flags(flags: list[str]) -> tuple[bool, str]:
        """Validate essential bare-metal C++ compiler flags."""
        flag_set = set(flags)
        required = {"-ffreestanding", "-mno-red-zone", "-fno-exceptions", "-fno-rtti"}
        missing = required - flag_set

        if not missing:
            return True, "正解！ベアメタル環境で必須となる4大コンパイラフラグを完全に網羅しています。"
        return False, f"不正解です。不足しているフラグ: {', '.join(sorted(missing))}"

    @staticmethod
    def validate_framebuffer_pixel_math(math_dict: dict[str, int]) -> tuple[bool, str]:
        """Validate FrameBufferConfig coordinate-to-byte-offset math."""
        x = math_dict.get("x", 100)
        y = math_dict.get("y", 50)
        ppsl = math_dict.get("pixels_per_scan_line", 832)
        bpp = math_dict.get("bytes_per_pixel", 4)
        byte_offset = math_dict.get("byte_offset")

        expected = bpp * (y * ppsl + x)
        if byte_offset == expected:
            return True, "正解！走査線ストライドを考慮した正確なVRAMピクセルオフセットを算出しました。"
        return False, f"不正解です。期待値 {expected} バイト, 入力値 {byte_offset} バイト"

    @staticmethod
    def validate_abi_registers(abi_dict: dict[str, list[str]]) -> tuple[bool, str]:
        """Validate System V AMD64 vs Microsoft x64 ABI argument passing registers."""
        sys_v = [r.upper() for r in abi_dict.get("system_v", [])]
        ms_x64 = [r.upper() for r in abi_dict.get("ms_x64", [])]

        expected_sys_v = ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
        expected_ms_x64 = ["RCX", "RDX", "R8", "R9"]

        errors = []
        if sys_v != expected_sys_v:
            errors.append(f"system_v: 期待値 {expected_sys_v}, 入力値 {sys_v}")
        if ms_x64 != expected_ms_x64:
            errors.append(f"ms_x64: 期待値 {expected_ms_x64}, 入力値 {ms_x64}")

        if not errors:
            return True, "正解！System V AMD64 と Microsoft x64 ABI の引数レジスタ規約を正確に識別しました。"
        return False, f"不正解です。{'; '.join(errors)}"

    @staticmethod
    def validate_placement_new_buffer(layout: dict[str, any]) -> tuple[bool, str]:
        """Validate Placement new buffer alignment and memory safety."""
        class_size = layout.get("class_size", 16)
        buffer_size = layout.get("buffer_size", 64)
        alignment = layout.get("alignment", 8)
        requires_free = layout.get("requires_free", False)

        errors = []
        if buffer_size < class_size:
            errors.append(f"buffer_size: クラスサイズ {class_size} 以上の領域が必要です")
        if alignment < 8:
            errors.append("alignment: 64ビットポインタを保持するため最低8バイト整列が必要です")
        if requires_free is not False:
            errors.append("requires_free: 静的バッファ上の配置newのためfree/deleteは不要 (False) です")

        if not errors:
            return True, "正解！配置 new による静的メモリ構築のサイズ・アライメント・ライフサイクルを正しく理解しています。"
        return False, f"不正解です。{'; '.join(errors)}"

    @staticmethod
    def validate_vtable_layout(vtable_info: dict[str, any]) -> tuple[bool, str]:
        """Validate Itanium C++ ABI virtual method table memory layout."""
        vptr_offset = vtable_info.get("vptr_offset", 0)
        vptr_size = vtable_info.get("vptr_size", 8)
        pure_virtual_handler = vtable_info.get("pure_virtual_handler", "")

        errors = []
        if vptr_offset != 0:
            errors.append(f"vptr_offset: 単一継承ではインスタンス先頭オフセット 0 に配置されます (入力値: {vptr_offset})")
        if vptr_size != 8:
            errors.append(f"vptr_size: 64ビットポインタのため 8 バイトであるべきです (入力値: {vptr_size})")
        if "__cxa_pure_virtual" not in pure_virtual_handler:
            errors.append(f"pure_virtual_handler: 純粋仮想関数のトラップハンドラ名は '__cxa_pure_virtual' です")

        if not errors:
            return True, "正解！Itanium C++ ABI の vtable/_vptr 構造と純粋仮想関数ハンドラを正確に同定しました。"
        return False, f"不正解です。{'; '.join(errors)}"

    @staticmethod
    def validate_elf_bss_clear(bss_info: dict[str, int]) -> tuple[bool, str]:
        """Validate ELF64 PT_LOAD segment .bss zero-clearing math."""
        p_vaddr = bss_info.get("p_vaddr", 0x100000)
        p_filesz = bss_info.get("p_filesz", 0x2400)
        p_memsz = bss_info.get("p_memsz", 0x3000)
        bss_start = bss_info.get("bss_start")
        bss_bytes = bss_info.get("bss_bytes")

        expected_start = p_vaddr + p_filesz
        expected_bytes = p_memsz - p_filesz

        errors = []
        if bss_start != expected_start:
            errors.append(f"bss_start: 期待値 {hex(expected_start)}, 入力値 {hex(bss_start) if bss_start else 'None'}")
        if bss_bytes != expected_bytes:
            errors.append(f"bss_bytes: 期待値 {expected_bytes} バイト, 入力値 {bss_bytes}")

        if not errors:
            return True, "正解！ELF PT_LOAD セグメントにおける .bss 領域の開始アドレスとゼロクリアバイト数を正確に計算しました。"
        return False, f"不正解です。{'; '.join(errors)}"


def plot_chapter4_exercises_summary(output_path: Path | str) -> Path:
    """Generate Chapter 4 comprehensive exercises curriculum and evaluation summary chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "第4章 総合演習問題カリキュラム & 理論・実装評価マップ\n"
        "(Chapter 4 Comprehensive Exercises & Verification Architecture)",
        fontsize=14, fontweight="bold", color="#1A237E", y=0.98,
    )

    # 1. 5大トピック構成 (Top-Left)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#FFFFFF")
    ax1.set_title("1. 第4章の学習・演習 5大コアトピック", fontsize=11, fontweight="bold", color="#0D47A1", pad=8)
    
    topics = [
        ("4.1 Make入門", "DAGトポロジカルソート・自動変数 ($@, $<, $^)・ベアメタルフラグ", "#E3F2FD", "#1565C0"),
        ("4.2 ピクセル描画 & ABI", "FrameBufferConfig・走査線ストライド数理・System V vs MS x64 ABI", "#E8F5E9", "#2E7D32"),
        ("4.3 PixelWriter 多態性", "抽象基底クラス・純粋仮想関数・ヒープなし配置 new・型安全性", "#FFF3E0", "#E65100"),
        ("4.4 vtable & Itanium ABI", "_vptr メモリ配置・間接call機械語・__cxa_pure_virtual トラップ", "#F3E5F5", "#6A1B9A"),
        ("4.5 ELFローダ & BSS", "Elf64_Ehdr/Phdr・PT_LOAD セグメント走査・.bss ゼロクリア数理", "#E0F2F1", "#00695C"),
    ]

    y_pos = 0.85
    for title, desc, bg, border in topics:
        box = FancyBboxPatch((0.03, y_pos - 0.06), 0.94, 0.12, boxstyle="round,pad=0.015",
                             facecolor=bg, edgecolor=border, lw=1.2)
        ax1.add_patch(box)
        ax1.text(0.06, y_pos, title, fontsize=9.2, fontweight="bold", color=border)
        ax1.text(0.32, y_pos, desc, fontsize=7.6, color="#263238")
        y_pos -= 0.165

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis("off")

    # 2. 演習問題構成比率 (Top-Right)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. 演習問題の配点・カテゴリ比率 (10問)", fontsize=11, fontweight="bold", color="#0D47A1", pad=8)
    
    categories = ["ビルド & フラグ (4.1)", "描画数理 & ABI (4.2)", "C++多態性 & new (4.3)", "vtable & ランタイム (4.4)", "ELF & メモリ展開 (4.5)"]
    counts = [2, 3, 2, 2, 1]
    colors = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#26A69A"]

    wedges, texts, autotexts = ax2.pie(
        counts, labels=categories, autopct="%1.0f%%", startangle=140,
        colors=colors, textprops=dict(fontsize=8.2, color="#263238"),
        wedgeprops=dict(width=0.45, edgecolor="white", lw=2)
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
        at.set_color("#FFFFFF")

    # 3. 理論検証フロー (Bottom-Left)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#FFFFFF")
    ax3.set_title("3. 自動採点 & 理論検証エンジン (Chapter4TheoryValidator)", fontsize=11, fontweight="bold", color="#0D47A1", pad=8)

    val_steps = [
        ("Step 1", "Makefile 自動変数 & .PHONY 判定", "#E1F5FE", "#0288D1"),
        ("Step 2", "ベアメタル 4大必須 C++ フラグ整合性", "#E8F8F5", "#1ABC9C"),
        ("Step 3", "VRAM 2次元オフセット数理: 4 * (y * ppsl + x)", "#FEF9E7", "#F39C12"),
        ("Step 4", "System V vs MS x64 ABI 引数レジスタ照合", "#F4ECF7", "#8E44AD"),
        ("Step 5", "ELF PT_LOAD .bss ゼロクリア範囲計算", "#FDEDEC", "#E74C3C"),
    ]

    y_pos = 0.85
    for step_num, desc, bg, border in val_steps:
        box = FancyBboxPatch((0.04, y_pos - 0.06), 0.92, 0.12, boxstyle="round,pad=0.015",
                             facecolor=bg, edgecolor=border, lw=1.2)
        ax3.add_patch(box)
        ax3.text(0.08, y_pos, step_num, fontsize=8.2, fontweight="bold", color=border)
        ax3.text(0.24, y_pos, desc, fontsize=7.8, color="#263238")
        y_pos -= 0.165

    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.axis("off")

    # 4. C++20 / Python ハイブリッド検証基盤 (Bottom-Right)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#FFFFFF")
    ax4.set_title("4. C++20 スタンドアロン検証 & pytest 統合", fontsize=11, fontweight="bold", color="#0D47A1", pad=8)

    comp_items = [
        ("C++20 Standalone", "src/ch04/os_chapter4_exercises.cpp\n(CMake: ch04_os_chapter4_exercises)", "#2E7D32"),
        ("Python Module", "common/os_sim.py\n(Chapter4Quiz, Chapter4TheoryValidator)", "#1565C0"),
        ("Unit Testing", "tests/test_ch4_exercises.py\n(pytest tests/ -v)", "#C62828"),
        ("Visual Verification", "4/result/fig4_ex1_exercises_summary.png\n4/result/fig4_ex2_ch4_mastery_radar.png", "#6A1B9A"),
    ]

    y_pos = 0.82
    for label, desc, col in comp_items:
        box = FancyBboxPatch((0.05, y_pos - 0.08), 0.90, 0.15, boxstyle="round,pad=0.015",
                             facecolor="#F5F5F5", edgecolor=col, lw=1.5)
        ax4.add_patch(box)
        ax4.text(0.09, y_pos + 0.02, label, fontsize=8.8, fontweight="bold", color=col)
        ax4.text(0.09, y_pos - 0.05, desc, fontsize=7.4, color="#37474F")
        y_pos -= 0.21

    ax4.set_xlim(0, 1.0)
    ax4.set_ylim(0, 1.0)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_chapter4_mastery_radar(output_path: Path | str, mastery_rates: dict[str, float]) -> Path:
    """Generate radar chart showing mastery rates across all 5 sections of Chapter 4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    sections = ["4.1\nMake入門", "4.2\nピクセル描画&ABI", "4.3\nPixelWriter多態性", "4.4\nvtable&ABI", "4.5\nELFローダ&BSS"]
    sec_keys = ["4.1", "4.2", "4.3", "4.4", "4.5"]
    rates = [mastery_rates.get(k, 100.0) for k in sec_keys]

    # Close the radar circle
    num_vars = len(sections)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    rates_closed = rates + rates[:1]
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#FFFFFF")

    # Draw grid
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), sections, fontsize=9.5, fontweight="bold", color="#1A237E")

    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="#78909C")
    ax.grid(color="#CFD8DC", linestyle="--", linewidth=0.8)

    # Plot data
    ax.plot(angles_closed, rates_closed, color="#1E88E5", linewidth=2.5, linestyle="solid", label="達成度 (Mastery)")
    ax.fill(angles_closed, rates_closed, color="#42A5F5", alpha=0.35)

    # Scatter points
    ax.scatter(angles, rates, color="#0D47A1", s=60, zorder=10)
    for angle, rate in zip(angles, rates):
        ax.text(angle, rate + 6, f"{rate:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="#0D47A1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# Chapter 5.1: Drawing Characters & References (osbook_day05a)
# ==============================================================================

# 16-byte raw bitmap font data for character 'A' (8x16 pixels)
FONT_A_RAW: tuple[int, ...] = (
    0b00000000,  # Row 0
    0b00011000,  # Row 1:     **
    0b00011000,  # Row 2:     **
    0b00011000,  # Row 3:     **
    0b00011000,  # Row 4:     **
    0b00100100,  # Row 5:    *  *
    0b00100100,  # Row 6:    *  *
    0b00100100,  # Row 7:    *  *
    0b00100100,  # Row 8:    *  *
    0b01111110,  # Row 9:   ******
    0b01000010,  # Row 10:  *    *
    0b01000010,  # Row 11:  *    *
    0b01000010,  # Row 12:  *    *
    0b11100111,  # Row 13: ***  ***
    0b00000000,  # Row 14:
    0b00000000,  # Row 15:
)


@dataclass(frozen=True)
class BitmapFontGlyph:
    """Represents an 8x16 monochrome bitmap font glyph."""
    char: str
    data: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.data) != 16:
            raise ValueError(f"Glyph data must contain exactly 16 rows, got {len(self.data)}")
        for i, row in enumerate(self.data):
            if not (0 <= row <= 255):
                raise ValueError(f"Row {i} value out of 8-bit range: {row}")

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 16

    def is_pixel_lit(self, dx: int, dy: int) -> bool:
        """Check if pixel at relative coordinate (dx, dy) is lit.
        Matches MikanOS bit-test logic: ((data[dy] << dx) & 0x80u) != 0.
        """
        if not (0 <= dx < 8 and 0 <= dy < 16):
            return False
        return bool((self.data[dy] << dx) & 0x80)

    def is_pixel_lit_right_shift(self, dx: int, dy: int) -> bool:
        """Alternative equivalent bit-test logic: ((data[dy] >> (7 - dx)) & 1) != 0."""
        if not (0 <= dx < 8 and 0 <= dy < 16):
            return False
        return bool((self.data[dy] >> (7 - dx)) & 1)

    def count_lit_pixels(self) -> int:
        """Count total lit pixels in this glyph."""
        return sum(bin(row).count("1") for row in self.data)

    def to_matrix(self) -> np.ndarray:
        """Convert glyph to a 16x8 boolean matrix (True for lit, False for background)."""
        mat = np.zeros((16, 8), dtype=bool)
        for dy in range(16):
            for dx in range(8):
                mat[dy, dx] = self.is_pixel_lit(dx, dy)
        return mat

    def to_ascii_art(self, lit_char: str = "@", unlit_char: str = ".") -> str:
        """Convert glyph to a multi-line ASCII art string."""
        lines = []
        for dy in range(16):
            row_str = "".join(lit_char if self.is_pixel_lit(dx, dy) else unlit_char for dx in range(8))
            lines.append(row_str)
        return "\n".join(lines)


# Singleton glyph instance for 'A'
GLYPH_A = BitmapFontGlyph(char="A", data=FONT_A_RAW)


def write_ascii_sim(
    writer: PixelWriterSim,
    x: int,
    y: int,
    c: str,
    color: PixelColorSim,
    glyph: BitmapFontGlyph = GLYPH_A,
) -> int:
    """Simulate MikanOS WriteAscii function (osbook_day05a).
    
    Draws character glyph 'A' using PixelWriter. Returns number of pixels rendered.
    If c != 'A', does nothing and returns 0.
    """
    if c != glyph.char:
        return 0

    pixels_drawn = 0
    for dy in range(glyph.height):
        for dx in range(glyph.width):
            if glyph.is_pixel_lit(dx, dy):
                writer.write(x + dx, y + dy, color)
                pixels_drawn += 1
    return pixels_drawn


def write_string_sim(
    writer: PixelWriterSim,
    x: int,
    y: int,
    s: str,
    color: PixelColorSim,
    glyph: BitmapFontGlyph = GLYPH_A,
) -> int:
    """Draw a string of characters sequentially with 8-pixel horizontal advance."""
    total_pixels = 0
    for i, ch in enumerate(s):
        total_pixels += write_ascii_sim(writer, x + i * glyph.width, y, ch, color, glyph)
    return total_pixels


@dataclass
class ReferenceVsPointerComparison:
    """Detailed theoretical & assembly comparison between C++ Reference and Pointer (Column 5.1)."""
    feature: str
    cpp_reference: str
    cpp_pointer: str
    hardware_assembly_note: str


def get_reference_vs_pointer_comparisons() -> list[ReferenceVsPointerComparison]:
    """Retrieve systematic comparison items between Reference and Pointer."""
    return [
        ReferenceVsPointerComparison(
            feature="構文と記号 (Syntax & Symbol)",
            cpp_reference="Type& ref = var; (宣言時に & 付与、使用時は通常変数と同じドット演算子 .)",
            cpp_pointer="Type* ptr = &var; (型に * 付与、アドレス取得に &、参照先に -> または *ptr)",
            hardware_assembly_note="構文上の違いのみ。生成機械語は両者ともメモリアドレスをレジスタに格納して間接参照",
        ),
        ReferenceVsPointerComparison(
            feature="NULL / nullptr の許容 (Nullability)",
            cpp_reference="不可 (常に有効なオブジェクトに束縛され、nullptr 参照は未定義動作 / 原則禁止)",
            cpp_pointer="許容 (nullptr / NULL を保持可能、ポインタ走査やオプショナル値として活用可能)",
            hardware_assembly_note="参照渡しを用いることで関数の引数チェック (if (p == nullptr)) を省略でき分岐予測と安全性が向上",
        ),
        ReferenceVsPointerComparison(
            feature="再束縛 / 再代入 (Rebinding)",
            cpp_reference="不可 (初期化時に束縛されたオブジェクトを変更できない。ref = val は束縛先への値代入)",
            cpp_pointer="可能 (ptr = &another_var で指す先のアドレスを何度でも自由に変更可能)",
            hardware_assembly_note="参照は不変のアドレスエイリアスとして最適化され、レジスタ割り当てが容易",
        ),
        ReferenceVsPointerComparison(
            feature="自身のアドレス (& 演算子)",
            cpp_reference="&ref は束縛先オブジェクトのアドレスそのものを返す (参照自体の独立アドレスは不可視)",
            cpp_pointer="&ptr はポインタ変数自身のアドレス (スタック上のポインタ格納領域) を返す",
            hardware_assembly_note="二重ポインタ Type** は存在するが、二重参照 Type&& は右辺値参照であり意味が全く異なる",
        ),
        ReferenceVsPointerComparison(
            feature="アセンブリ実装 (Assembly Implementation)",
            cpp_reference="leaq -16(%rbp), %rax / movq %rax, %rdi / call WriteAscii (ポインタ渡しと100%同一)",
            cpp_pointer="leaq -16(%rbp), %rax / movq %rax, %rdi / call WriteAscii (参照渡しと100%同一)",
            hardware_assembly_note="System V AMD64 ABI において、参照渡しとポインタ渡しは全く同一のレジスタ渡し規約に従う",
        ),
        ReferenceVsPointerComparison(
            feature="OSカーネル設計指針 (OS Design Guidelines)",
            cpp_reference="WriteAscii(PixelWriter& writer, ...): 必須かつ変更不要なサブシステムを渡す際に推奨",
            cpp_pointer="PixelWriter* pixel_writer;: 配置new前の未初期化状態や動的切り替えが必要な場合に必須",
            hardware_assembly_note="MikanOS では WriteAscii に参照を採用し、呼び出し側の安全性とコード簡潔性を両立",
        ),
    ]


def plot_fig5_01_font_glyph_and_bit_expansion(output_path: Path | str) -> Path:
    """Generate high-resolution architecture figure for Section 5.1 (Figure 5.1).
    Visualizes 8x16 bitmap font 'A', bit expansion mathematics, and VRAM drawing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.25)

    # Subplot 1: Glyph 'A' Bit Matrix & Row Bit Patterns
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor("#FAFAFA")
    ax1.set_title("1. 8×16 ビットマップフォント 'A' の内部データ構造とビット展開",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=12)

    # Draw grid for 16 rows and 8 columns
    matrix = GLYPH_A.to_matrix()
    cmap = ListedColormap(["#ECEFF1", "#FF6F00"])
    ax1.imshow(matrix, cmap=cmap, aspect="auto", extent=[-0.5, 7.5, 15.5, -0.5])

    # Overlay grid lines
    ax1.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax1.grid(which="minor", color="#90A4AE", linestyle="-", linewidth=1.2)
    ax1.tick_params(which="minor", size=0)

    ax1.set_xticks(range(8))
    ax1.set_xticklabels([f"dx={i}\n(bit {7-i})" for i in range(8)], fontsize=8.5, fontweight="bold", color="#263238")
    ax1.set_yticks(range(16))
    ax1.set_yticklabels([f"dy={i:2d}" for i in range(16)], fontsize=8.5, fontweight="bold", color="#263238")

    # Add text for bits and hex values
    for dy in range(16):
        byte_val = GLYPH_A.data[dy]
        for dx in range(8):
            bit_lit = bool((byte_val << dx) & 0x80)
            bit_str = "1" if bit_lit else "0"
            text_color = "#FFFFFF" if bit_lit else "#78909C"
            ax1.text(dx, dy, bit_str, ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)
        # Right label with hex value
        ax1.text(8.2, dy, f"0x{byte_val:02X} (0b{byte_val:08b})",
                 ha="left", va="center", fontsize=8.2, family="monospace", color="#004D40", fontweight="bold")

    ax1.set_xlim(-0.7, 12.2)
    ax1.set_xlabel("ビット列走査方向: MSB (0x80, dx=0) ──> LSB (0x01, dx=7)", fontsize=9.5, fontweight="bold", color="#0D47A1")

    # Subplot 2: Bit Shift Mathematical Formula & Architecture
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. ビット判定アルゴリズムの数理 (Bit-Test Logic)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)
    ax2.axis("off")

    formula_box = FancyBboxPatch((0.03, 0.05), 0.94, 0.90, boxstyle="round,pad=0.03",
                                 facecolor="#E3F2FD", edgecolor="#1976D2", lw=2)
    ax2.add_patch(formula_box)

    formula_text = (
        "【MikanOS ビット判定アルゴリズム (左シフト方式)】\n"
        "  if ((kFontA[dy] << dx) & 0x80u) {\n"
        "      writer.Write(x + dx, y + dy, color);\n"
        "  }\n"
        "───────────────────────────────────────\n"
        "● 数理モデルの動作ステップ:\n"
        "  1. 対象行の1バイトを取り出す: byte = kFontA[dy]\n"
        "  2. dx ビット分だけ左シフト: shifted = byte << dx\n"
        "     - dx=0 のとき: bit 7 (MSB) が最上位に留まる\n"
        "     - dx=1 のとき: bit 6 が最上位 (bit 7) の位置にシフト\n"
        "     - dx=7 のとき: bit 0 (LSB) が最上位 (bit 7) に到達\n"
        "  3. マスク 0x80 (0b10000000) と AND 演算:\n"
        "     - 最上位ビットが 1 なら非ゼロ (描画対象)\n"
        "     - 最上位ビットが 0 なら 0 (描画スキップ)\n"
        "───────────────────────────────────────\n"
        "● 等価な右シフト方式 (Alternative Right Shift):\n"
        "  if ((kFontA[dy] >> (7 - dx)) & 1) { ... }\n"
        "  - どちらも完全に等価だが、左シフト方式は定数マスク 0x80 を\n"
        "    固定でき、x86-64 の testb 命令等への最適化が効きやすい。"
    )
    ax2.text(0.06, 0.50, formula_text, fontsize=8.5,
             va="center", color="#0D47A1", linespacing=1.35)

    # Subplot 3: VRAM FrameBuffer Rendering Simulation ("AA" at (50, 50))
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#212121")
    ax3.set_title("3. VRAM フレームバッファへの文字描画結果 ('AA' 描画)",
                  fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)

    # Simulate frame buffer slice around (40..75, 40..75)
    config = FrameBufferConfigSim(
        frame_buffer_base=0x8000_0000,
        pixels_per_scan_line=200,
        horizontal_resolution=200,
        vertical_resolution=150,
        pixel_format=PixelFormat.RGB_RESV_8BIT_PER_COLOR,
    )
    writer = RGBResv8BitPerColorPixelWriterSim(config)

    # Fill white background
    for y in range(150):
        for x in range(200):
            writer.write(x, y, PixelColorSim(255, 255, 255))

    # Draw green rectangle as in day04c
    for y in range(10, 40):
        for x in range(10, 100):
            writer.write(x, y, PixelColorSim(0, 200, 0))

    # Draw 'A' at (50, 50) and 'A' at (58, 50) in black (0, 0, 0)
    write_ascii_sim(writer, 50, 50, 'A', PixelColorSim(0, 0, 0))
    write_ascii_sim(writer, 58, 50, 'A', PixelColorSim(0, 0, 0))

    # Convert buffer slice [45..70, 45..70] to numpy image
    img_slice = np.zeros((26, 26, 3), dtype=np.uint8)
    for row_idx, cy in enumerate(range(45, 71)):
        for col_idx, cx in enumerate(range(45, 71)):
            offset = writer.pixel_at(cx, cy)
            img_slice[row_idx, col_idx] = [
                writer.buffer[offset],
                writer.buffer[offset + 1],
                writer.buffer[offset + 2],
            ]

    ax3.imshow(img_slice, extent=[45, 71, 71, 45], interpolation="nearest")
    ax3.set_xlabel("VRAM X 座標 (Pixels) -> 文字送り +8 ピクセル (50 -> 58)", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    ax3.set_ylabel("VRAM Y 座標 (Pixels)", fontsize=8.5, fontweight="bold", color="#FFFFFF")
    ax3.tick_params(colors="#FFFFFF", labelsize=8)

    # Annotations
    ax3.text(54, 48, "'A' (50, 50)", color="#D32F2F", fontsize=8.5, fontweight="bold", ha="center")
    ax3.text(62, 48, "'A' (58, 50)", color="#1976D2", fontsize=8.5, fontweight="bold", ha="center")

    fig.suptitle("ゼロからのOS自作入門 第5.1節: 8×16 ビットマップフォントと文字描画アルゴリズム\n"
                 "(Chapter 5.1: 8x16 Bitmap Font Glyph, Bit-Expansion Math & VRAM Output)",
                 fontsize=13.5, fontweight="bold", color="#0D47A1", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_fig5_02_reference_vs_pointer_memory_model(output_path: Path | str) -> Path:
    """Generate high-resolution architecture figure for Column 5.1 (Figure 5.2).
    Compares C++ Reference vs Pointer memory layout, ABI passing, and usage principles.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1.0], hspace=0.32)

    # Subplot 1: Stack & Memory Architecture (Pointer vs Reference)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FAFAFA")
    ax1.set_title("1. スタックフレームとメモリ空間における「ポインタ」と「参照」の物理配置とABI",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax1.axis("off")

    # Left box: Pointer memory model
    ptr_box = FancyBboxPatch((0.03, 0.08), 0.45, 0.84, boxstyle="round,pad=0.03",
                             facecolor="#FFF3E0", edgecolor="#F57C00", lw=2)
    ax1.add_patch(ptr_box)
    ptr_text = (
        "【ポインタ型変数: PixelWriter* ptr】\n"
        "────────────────────────────────────────\n"
        "● スタック上に「8バイトのアドレス格納領域」が独立して確保される\n"
        "  - &ptr : スタック上のポインタ変数自身のアドレス (例: 0x7FFF_0008)\n"
        "  - ptr  : 指している対象オブジェクトのアドレス (例: 0x8000_1000)\n"
        "  - *ptr : デリファレンスして対象オブジェクトにアクセス\n"
        "────────────────────────────────────────\n"
        "● 特性とリスク:\n"
        "  [+] nullptr 代入可能 (初期化前や存在しない状態を表現可能)\n"
        "  [+] 再代入可能 (ptr = &other_writer でいつでも別の対象を指せる)\n"
        "  [-] ヌルポインタ参照による #PF (Page Fault) の危険性あり\n"
        "────────────────────────────────────────\n"
        "● System V AMD64 ABI 関数呼び出し (機械語):\n"
        "  movq  -8(%rbp), %rax       # スタックからポインタ値を読み出す\n"
        "  movq  %rax, %rdi           # 第1引数レジスタ RDI にアドレスを設定\n"
        "  call  WriteAscii           # 関数呼び出し"
    )
    ax1.text(0.05, 0.50, ptr_text, fontsize=8.2,
             va="center", color="#E65100", linespacing=1.30)

    # Right box: Reference memory model
    ref_box = FancyBboxPatch((0.52, 0.08), 0.45, 0.84, boxstyle="round,pad=0.03",
                             facecolor="#E8F5E9", edgecolor="#388E3C", lw=2)
    ax1.add_patch(ref_box)
    ref_text = (
        "【参照型エイリアス: PixelWriter& ref】\n"
        "────────────────────────────────────────\n"
        "● 型システム上の「別名 (エイリアス)」。独立した変数領域は不可視\n"
        "  - &ref : 束縛先オブジェクトのアドレスそのものを返す (0x8000_1000)\n"
        "  - ref.Write(...) : 通常のオブジェクトと全く同じドット記法\n"
        "  - 内部実装としてはポインタと同一の「8バイトアドレス」を保持\n"
        "────────────────────────────────────────\n"
        "● 特性と安全性:\n"
        "  [+] nullptr 禁止 (常に有効なオブジェクトへの束縛が保証される)\n"
        "  [+] 再束縛不可 (一度束縛されたら別のオブジェクトを指し直せない)\n"
        "  [+] if (writer == nullptr) の無駄なチェックが不要で高速・堅牢\n"
        "────────────────────────────────────────\n"
        "● System V AMD64 ABI 関数呼び出し (機械語):\n"
        "  leaq  -40(%rbp), %rax      # 実体オブジェクトのアドレスを計算\n"
        "  movq  %rax, %rdi           # 第1引数レジスタ RDI にアドレスを設定\n"
        "  call  WriteAscii           # 関数呼び出し (ポインタ渡しと100%同一!)"
    )
    ax1.text(0.54, 0.50, ref_text, fontsize=8.2,
             va="center", color="#1B5E20", linespacing=1.30)

    # Subplot 2: Comparison Matrix Table
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_title("2. コラム5.1: 参照 (Reference) とポインタ (Pointer) の 6 大特性比較対照表",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=10)
    ax2.axis("off")

    comparisons = get_reference_vs_pointer_comparisons()
    table_data = []
    for c in comparisons:
        table_data.append([c.feature, c.cpp_reference, c.cpp_pointer, c.hardware_assembly_note])

    col_labels = ["評価項目", "C++ 参照 (Reference: Type&)", "C++ ポインタ (Pointer: Type*)", "ハードウェア / ABI / 設計指針"]
    col_widths = [0.18, 0.28, 0.28, 0.26]

    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.7)

    # Style header and rows
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1A237E")
            cell.set_text_props(color="#FFFFFF", fontweight="bold")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F5F5F5")
            else:
                cell.set_facecolor("#FFFFFF")
            if col == 0:
                cell.set_text_props(fontweight="bold", color="#0D47A1")

    fig.suptitle("ゼロからのOS自作入門 コラム5.1: C++ 参照 vs ポインタのメモリモデルとABI比較\n"
                 "(Column 5.1: Reference vs Pointer in Modern C++ and x86-64 Bare-Metal Systems)",
                 fontsize=13.5, fontweight="bold", color="#0D47A1", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================================================================
# Chapter 5.2: Separate Compilation & Linking Simulator (osbook_day05b)
# ==============================================================================

class SymbolBinding(Enum):
    """Symbol binding scope in ELF symbol tables."""
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"
    WEAK = "WEAK"


class SymbolSection(Enum):
    """Section where a symbol is located."""
    TEXT = "t"       # Code / functions
    DATA = "d"       # Initialized writable data
    RODATA = "r"     # Read-only constants
    BSS = "b"        # Uninitialized zeroed data
    UNDEF = "U"      # Undefined symbol (external reference)
    COMMON = "c"     # Common uninitialized data


@dataclass
class SymbolSim:
    """Simulation of an ELF symbol table entry."""
    name: str
    section: SymbolSection
    binding: SymbolBinding = SymbolBinding.GLOBAL
    size: int = 0
    address: Optional[int] = None
    demangled_name: Optional[str] = None

    @property
    def is_defined(self) -> bool:
        """Return True if symbol is defined in this translation unit."""
        return self.section != SymbolSection.UNDEF

    @property
    def nm_type_char(self) -> str:
        """Return nm-compatible type character (uppercase for GLOBAL, lowercase for LOCAL)."""
        char = self.section.value
        if self.section == SymbolSection.UNDEF:
            return "U"
        return char.upper() if self.binding == SymbolBinding.GLOBAL else char.lower()


@dataclass
class ObjectFileSim:
    """Simulation of an ELF relocatable object file (.o) produced from a translation unit."""
    filename: str
    source_cpp: str
    included_headers: List[str] = field(default_factory=list)
    defined_symbols: Dict[str, SymbolSim] = field(default_factory=dict)
    undefined_symbols: Dict[str, SymbolSim] = field(default_factory=dict)

    def add_defined_symbol(self, name: str, section: SymbolSection, size: int = 0,
                           binding: SymbolBinding = SymbolBinding.GLOBAL,
                           demangled: Optional[str] = None) -> SymbolSim:
        """Register a defined symbol."""
        sym = SymbolSim(name=name, section=section, binding=binding, size=size, demangled_name=demangled)
        self.defined_symbols[name] = sym
        return sym

    def add_undefined_symbol(self, name: str, demangled: Optional[str] = None) -> SymbolSim:
        """Register an undefined external symbol reference."""
        sym = SymbolSim(name=name, section=SymbolSection.UNDEF, binding=SymbolBinding.GLOBAL, demangled_name=demangled)
        self.undefined_symbols[name] = sym
        return sym

    def dump_nm_style(self) -> List[str]:
        """Format symbols similar to 'nm -C' output."""
        lines = []
        for name, sym in sorted(self.defined_symbols.items()):
            display_name = sym.demangled_name or name
            lines.append(f"{sym.nm_type_char} {display_name}")
        for name, sym in sorted(self.undefined_symbols.items()):
            display_name = sym.demangled_name or name
            lines.append(f"{sym.nm_type_char} {display_name}")
        return lines


class LinkerSim:
    """Simulation of an ELF linker (LLD) performing symbol resolution and relocation."""

    def __init__(self, base_address: int = 0x0010_0000):
        self.base_address = base_address
        self.object_files: List[ObjectFileSim] = []
        self.resolved_symbols: Dict[str, int] = {}
        self.symbol_origins: Dict[str, str] = {}
        self.unresolved_symbols: List[Tuple[str, str]] = []  # (symbol_name, obj_filename)
        self.duplicate_symbols: Dict[str, List[str]] = {}    # symbol_name -> list of obj_filenames

    def add_object_file(self, obj: ObjectFileSim):
        """Add an object file to the link input list."""
        self.object_files.append(obj)

    def link(self) -> Tuple[bool, str]:
        """Perform symbol resolution, detect ODR violations and undefined symbols."""
        self.resolved_symbols.clear()
        self.symbol_origins.clear()
        self.unresolved_symbols.clear()
        self.duplicate_symbols.clear()

        current_address = self.base_address

        # Step 1: Collect and check defined symbols for duplicates (ODR violation)
        for obj in self.object_files:
            for name, sym in obj.defined_symbols.items():
                if sym.binding == SymbolBinding.GLOBAL:
                    if name in self.symbol_origins:
                        if name not in self.duplicate_symbols:
                            self.duplicate_symbols[name] = [self.symbol_origins[name]]
                        self.duplicate_symbols[name].append(obj.filename)
                    else:
                        self.symbol_origins[name] = obj.filename
                        self.resolved_symbols[name] = current_address
                        current_address += max(sym.size, 16)

        # Step 2: Resolve undefined symbols against global defined symbols
        for obj in self.object_files:
            for name in obj.undefined_symbols:
                if name not in self.resolved_symbols:
                    self.unresolved_symbols.append((name, obj.filename))

        success = (len(self.unresolved_symbols) == 0) and (len(self.duplicate_symbols) == 0)
        report_lines = [
            f"=== ELF Linker Simulation Report (Base Address: 0x{self.base_address:08X}) ===",
            f"Input Objects: {', '.join(o.filename for o in self.object_files)}",
            f"Total Resolved Global Symbols: {len(self.resolved_symbols)}",
        ]

        if self.duplicate_symbols:
            report_lines.append("[!] ERROR: Multiple definition (ODR violation) detected:")
            for sym, objs in self.duplicate_symbols.items():
                report_lines.append(f"    - Symbol '{sym}' defined in: {', '.join(objs)}")

        if self.unresolved_symbols:
            report_lines.append("[!] ERROR: Undefined symbol references detected:")
            for sym, obj in self.unresolved_symbols:
                report_lines.append(f"    - Symbol '{sym}' referenced from '{obj}' is undefined")

        if success:
            report_lines.append("[+] Link Succeeded: kernel.elf generated without errors.")
            for sym, addr in sorted(self.resolved_symbols.items(), key=lambda x: x[1]):
                report_lines.append(f"    0x{addr:08X} T {sym} ({self.symbol_origins[sym]})")

        return success, "\n".join(report_lines)


class DependencyGraphSim:
    """Simulation of Makefile header dependency tracking (clang++ -MM)."""

    def __init__(self):
        self.dependencies: Dict[str, List[str]] = {}  # source_file -> list of headers

    def add_dependency(self, source_file: str, headers: List[str]):
        """Register header dependencies for a source file."""
        self.dependencies[source_file] = list(headers)

    def generate_d_file_content(self, cpp_file: str) -> str:
        """Generate GNU Makefile dependency content as produced by clang++ -MM."""
        obj_name = cpp_file.replace(".cpp", ".o")
        headers = self.dependencies.get(cpp_file, [])
        header_str = " \\\n  ".join(headers)
        if headers:
            return f"{obj_name}: {cpp_file} \\\n  {header_str}\n"
        return f"{obj_name}: {cpp_file}\n"

    def determine_rebuild_targets(self, modified_file: str) -> List[str]:
        """Determine which object files need rebuild when a file is modified."""
        rebuild_targets = []
        if modified_file.endswith(".cpp"):
            rebuild_targets.append(modified_file.replace(".cpp", ".o"))
        else:
            for cpp, headers in self.dependencies.items():
                if modified_file in headers:
                    rebuild_targets.append(cpp.replace(".cpp", ".o"))
        return sorted(rebuild_targets)


def create_day05b_object_files() -> Tuple[ObjectFileSim, ObjectFileSim, ObjectFileSim]:
    """Create the exact MikanOS day05b modular object file representations."""
    # graphics.o
    graphics_o = ObjectFileSim(
        filename="graphics.o",
        source_cpp="graphics.cpp",
        included_headers=["graphics.hpp", "frame_buffer_config.hpp"],
    )
    graphics_o.add_defined_symbol(
        "_ZN30RGBResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
        SymbolSection.TEXT,
        size=48,
        demangled="RGBResv8BitPerColorPixelWriter::Write(int, int, PixelColor const&)",
    )
    graphics_o.add_defined_symbol(
        "_ZN30BGRResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
        SymbolSection.TEXT,
        size=48,
        demangled="BGRResv8BitPerColorPixelWriter::Write(int, int, PixelColor const&)",
    )
    graphics_o.add_defined_symbol(
        "_ZTV30RGBResv8BitPerColorPixelWriter",
        SymbolSection.RODATA,
        size=32,
        demangled="vtable for RGBResv8BitPerColorPixelWriter",
    )
    graphics_o.add_defined_symbol(
        "_ZTV30BGRResv8BitPerColorPixelWriter",
        SymbolSection.RODATA,
        size=32,
        demangled="vtable for BGRResv8BitPerColorPixelWriter",
    )

    # font.o
    font_o = ObjectFileSim(
        filename="font.o",
        source_cpp="font.cpp",
        included_headers=["font.hpp", "graphics.hpp"],
    )
    font_o.add_defined_symbol(
        "_Z10WriteAsciiR11PixelWriteriicRK10PixelColor",
        SymbolSection.TEXT,
        size=96,
        demangled="WriteAscii(PixelWriter&, int, int, char, PixelColor const&)",
    )
    font_o.add_defined_symbol(
        "_ZL6kFontA",
        SymbolSection.RODATA,
        size=16,
        binding=SymbolBinding.LOCAL,
        demangled="kFontA (file-scope constant)",
    )

    # main.o
    main_o = ObjectFileSim(
        filename="main.o",
        source_cpp="main.cpp",
        included_headers=["frame_buffer_config.hpp", "graphics.hpp", "font.hpp"],
    )
    main_o.add_defined_symbol(
        "KernelMain",
        SymbolSection.TEXT,
        size=256,
        demangled="KernelMain(FrameBufferConfig const&)",
    )
    main_o.add_undefined_symbol(
        "_ZN30RGBResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
        demangled="RGBResv8BitPerColorPixelWriter::Write(int, int, PixelColor const&)",
    )
    main_o.add_undefined_symbol(
        "_ZN30BGRResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
        demangled="BGRResv8BitPerColorPixelWriter::Write(int, int, PixelColor const&)",
    )
    main_o.add_undefined_symbol(
        "_ZTV30RGBResv8BitPerColorPixelWriter",
        demangled="vtable for RGBResv8BitPerColorPixelWriter",
    )
    main_o.add_undefined_symbol(
        "_ZTV30BGRResv8BitPerColorPixelWriter",
        demangled="vtable for BGRResv8BitPerColorPixelWriter",
    )
    main_o.add_undefined_symbol(
        "_Z10WriteAsciiR11PixelWriteriicRK10PixelColor",
        demangled="WriteAscii(PixelWriter&, int, int, char, PixelColor const&)",
    )

    return graphics_o, font_o, main_o


def plot_fig5_03_separate_compilation_and_linking_pipeline(output_path: Union[str, Path]) -> str:
    """Generate Figure 5.3: Complete Separate Compilation & Linking Architecture (DPI 300)."""
    setup_plot_style()
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.28, 0.44, 0.28], hspace=0.35)

    # Subplot 1: Source Files & Modular Header Split
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#F8F9FA")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 30)
    ax1.axis("off")
    ax1.set_title("1. ソースコード分割とヘッダの役割分担 (MikanOS day05b: graphics, font, main)",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=8)

    boxes_top = [
        ("graphics.hpp\n(宣言 / Interface)\n・struct PixelColor\n・class PixelWriter\n・class RGB/BGRPixelWriter", 10, 15, "#E3F2FD", "#1565C0"),
        ("graphics.cpp\n(定義 / Implementation)\n・RGBPixelWriter::Write()\n・BGRPixelWriter::Write()\n#include \"graphics.hpp\"", 30, 15, "#BBDEFB", "#0D47A1"),
        ("font.hpp\n(宣言 / Interface)\n・WriteAscii() 宣言\n#include \"graphics.hpp\"", 50, 15, "#E8F5E9", "#2E7D32"),
        ("font.cpp\n(定義 / Implementation)\n・const uint8_t kFontA[16]\n・WriteAscii() 実装\n#include \"font.hpp\"", 70, 15, "#C8E6C9", "#1B5E20"),
        ("main.cpp\n(制御 / Application)\n・KernelMain()\n・placement new\n#include \"graphics.hpp\"\n#include \"font.hpp\"", 90, 15, "#FFF3E0", "#E65100"),
    ]

    for title, cx, cy, bg, border in boxes_top:
        rect = patches.FancyBboxPatch(
            (cx - 8.5, cy - 11), 17, 22,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg, edgecolor=border, linewidth=2.0
        )
        ax1.add_patch(rect)
        ax1.text(cx, cy, title, fontsize=8.2, va="center", ha="center",
                 fontweight="bold", color="#212121", linespacing=1.25)

    # Subplot 2: Compilation into Translation Units (.o) with Symbol Tables
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 40)
    ax2.axis("off")
    ax2.set_title("2. 独立コンパイル（clang++ -c）とリロケータブル・オブジェクトファイル (.o) のシンボルテーブル",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=8)

    obj_boxes = [
        ("graphics.o (ELF Relocatable)\n────────────────────────\n[T] RGB/BGR::Write() (定義)\n[R] vtable for RGB/BGR\n[U] (外部依存なし)\n\n※main.cppに依存せず単独ビルド可能", 18, 20, "#E1F5FE", "#0288D1"),
        ("font.o (ELF Relocatable)\n────────────────────────\n[T] WriteAscii() (定義)\n[r] kFontA (ファイルローカル)\n[U] (外部依存なし)\n\n※フォント変更時のみ再コンパイル", 50, 20, "#E8F8F5", "#00897B"),
        ("main.o (ELF Relocatable)\n────────────────────────\n[T] KernelMain() (定義)\n[U] WriteAscii() (未解決)\n[U] RGB/BGR::Write() (未解決)\n[U] vtable RGB/BGR (未解決)", 82, 20, "#FFF8E1", "#FFA000"),
    ]

    for title, cx, cy, bg, border in obj_boxes:
        rect = patches.FancyBboxPatch(
            (cx - 13, cy - 17), 26, 35,
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor=bg, edgecolor=border, linewidth=2.2
        )
        ax2.add_patch(rect)
        ax2.text(cx, cy, title, fontsize=8.0, va="center", ha="center",
                 color="#212121", linespacing=1.30)

    for x in [18, 50, 82]:
        arrow = patches.FancyArrowPatch(
            (x, 38), (x, 34),
            arrowstyle="-|>", mutation_scale=15,
            color="#546E7A", linewidth=2.5
        )
        ax2.add_patch(arrow)
        ax2.text(x, 38.5, "clang++ -c", fontsize=7.5, ha="center", va="bottom", color="#37474F", fontweight="bold")

    # Subplot 3: Linker (LLD) Resolution & kernel.elf Creation
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#F1F8E9")
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 30)
    ax3.axis("off")
    ax3.set_title("3. リンカ（LLD）によるシンボル解決（Symbol Resolution）と実行可能ファイル（kernel.elf）の生成",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=8)

    rect_link = patches.FancyBboxPatch(
        (10, 4), 80, 22,
        boxstyle="round,pad=1.0,rounding_size=2.0",
        facecolor="#FFFFFF", edgecolor="#2E7D32", linewidth=2.5
    )
    ax3.add_patch(rect_link)

    linker_text = (
        "【リンカ (ld.lld) の動作フロー】\n"
        "① 各 .o ファイルのグローバル定義シンボル ([T], [R]) を集約して統合アドレス空間 (0x00100000〜) に配置\n"
        "② main.o の未解決参照 [U] (WriteAscii, vtable等) を graphics.o と font.o の実アドレスへ書き換え (Relocation)\n"
        "③ すべての未解決参照が解消されれば 64-bit ELF Executable (kernel.elf) を出力 (未解決があれば ld: error: undefined symbol)"
    )
    ax3.text(50, 15, linker_text, fontsize=8.6, ha="center", va="center",
             color="#1B5E20", fontweight="bold", linespacing=1.45)

    fig.suptitle("ゼロからのOS自作入門 第5章 5.2: 分割コンパイルとシンボル解決アーキテクチャ\n"
                 "(Figure 5.3: Separate Compilation, Translation Units, Symbol Resolution and Linking)",
                 fontsize=14, fontweight="bold", color="#0D47A1", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_fig5_04_header_dependency_graph_and_odr(output_path: Union[str, Path]) -> str:
    """Generate Figure 5.4: Header Dependency DAG, Makefile -MM, and ODR Mechanics (DPI 300)."""
    setup_plot_style()
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.48, 0.52], hspace=0.32)

    # Subplot 1: Header Dependency DAG & Makefile Automatic Dependencies
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FAFAFA")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 50)
    ax1.axis("off")
    ax1.set_title("1. ヘッダ依存関係有向非巡回グラフ (DAG) と Makefile の自動依存関係生成 (clang++ -MM)",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=8)

    dag_nodes = [
        ("frame_buffer_config.hpp\n(ルート定義)", 20, 38, "#E0F2F1", "#00695C"),
        ("graphics.hpp\n(PixelWriter, PixelColor)", 50, 38, "#E1F5FE", "#0277BD"),
        ("font.hpp\n(WriteAscii 宣言)", 80, 38, "#E8F5E9", "#2E7D32"),
        ("graphics.cpp\n→ graphics.o", 35, 16, "#BBDEFB", "#1565C0"),
        ("font.cpp\n→ font.o", 65, 16, "#C8E6C9", "#2E7D32"),
        ("main.cpp\n→ main.o", 85, 16, "#FFE0B2", "#EF6C00"),
    ]

    for name, x, y, bg, border in dag_nodes:
        box = patches.FancyBboxPatch(
            (x - 9, y - 5), 18, 10,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            facecolor=bg, edgecolor=border, linewidth=2.0
        )
        ax1.add_patch(box)
        ax1.text(x, y, name, fontsize=8.0, ha="center", va="center",
                 fontweight="bold", color="#212121", linespacing=1.2)

    arrows = [
        ((41, 38), (29, 38), "includes"),        # graphics.hpp -> frame_buffer_config.hpp
        ((71, 38), (59, 38), "includes"),        # font.hpp -> graphics.hpp
        ((35, 21), (45, 33), "includes"),        # graphics.cpp -> graphics.hpp
        ((65, 21), (75, 33), "includes"),        # font.cpp -> font.hpp
        ((85, 21), (82, 33), "includes"),        # main.cpp -> font.hpp
        ((80, 21), (55, 33), "includes"),        # main.cpp -> graphics.hpp
    ]

    for start, end, label in arrows:
        arr = patches.FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12,
            color="#455A64", linewidth=1.8, linestyle="--"
        )
        ax1.add_patch(arr)

    mm_text = (
        "【Makefile における自動依存関係生成 (clang++ -MM)】\n"
        "・コマンド: clang++ $(CPPFLAGS) $(CXXFLAGS) -MM $< > $@\n"
        "・生成物: .main.d → 'main.o: main.cpp frame_buffer_config.hpp graphics.hpp font.hpp'\n"
        "・効果: graphics.hpp を編集しただけで、make が main.o と graphics.o のリビルドを自動検知！"
    )
    rect_mm = patches.FancyBboxPatch(
        (5, 1.5), 90, 8,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        facecolor="#FFFDE7", edgecolor="#FBC02D", linewidth=1.5
    )
    ax1.add_patch(rect_mm)
    ax1.text(50, 5.5, mm_text, fontsize=8.2, ha="center", va="center", color="#3E2723", linespacing=1.25)

    # Subplot 2: One Definition Rule (ODR) & Include Guards Mechanics
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 50)
    ax2.axis("off")
    ax2.set_title("2. 単一定義規則 (One Definition Rule: ODR) とインクルードガード (#pragma once) の本質",
                  fontsize=12, fontweight="bold", color="#0D47A1", pad=8)

    # Left: Correct Separation
    rect_left = patches.FancyBboxPatch(
        (3, 4), 45, 42,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2.2
    )
    ax2.add_patch(rect_left)
    ax2.text(25.5, 42, "【正】宣言と定義の分離 (Clean Build)", fontsize=10.5,
             fontweight="bold", color="#1B5E20", ha="center")

    good_text = (
        "● ヘッダ (font.hpp):\n"
        "   #pragma once\n"
        "   void WriteAscii(PixelWriter& writer, ...);\n"
        "   → 関数のシグネチャ（型・名前）のみ宣言。\n"
        "     機械語コードやメモリ領域は生成されない。\n"
        "────────────────────────\n"
        "● 実装 (font.cpp):\n"
        "   void WriteAscii(...) { /* 実装コード */ }\n"
        "   → font.o にのみ実体シンボル [T] が1個生成。\n"
        "────────────────────────\n"
        "● リンカ (Linker):\n"
        "   main.o の [U] を font.o の唯一の [T] にバインド。\n"
        "   → ビルド成功！(ODR を完全遵守)"
    )
    ax2.text(5, 23, good_text, fontsize=7.8, va="center", color="#1B5E20", linespacing=1.28)

    # Right: Incorrect
    rect_right = patches.FancyBboxPatch(
        (52, 4), 45, 42,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor="#FFEBEE", edgecolor="#C62828", linewidth=2.2
    )
    ax2.add_patch(rect_right)
    ax2.text(74.5, 42, "【誤】ヘッダに関数定義を書く罠 (ODR 違反)", fontsize=10.5,
             fontweight="bold", color="#B71C1C", ha="center")

    bad_text = (
        "● ヘッダ (font.hpp) に実装を書いてしまう:\n"
        "   #pragma once\n"
        "   void WriteAscii(...) { /* 実装コード */ }\n"
        "────────────────────────\n"
        "● 落とし穴: #pragma once は同一翻訳単位内での\n"
        "   重複展開を防ぐだけで、別々の .cpp からの\n"
        "   インクルードは防げない！\n"
        "   → font.o に WriteAscii の [T] が生成\n"
        "   → main.o にも WriteAscii の [T] が生成\n"
        "────────────────────────\n"
        "● リンカエラー (LLD):\n"
        "   ld.lld: error: duplicate symbol: WriteAscii\n"
        "   >>> defined in font.o\n"
        "   >>> defined in main.o  (リンク失敗!)"
    )
    ax2.text(54, 23, bad_text, fontsize=7.8, va="center", color="#B71C1C", linespacing=1.28)

    fig.suptitle("ゼロからのOS自作入門 第5章 5.2: ヘッダ依存関係DAGと単一定義規則 (ODR) の深層\n"
                 "(Figure 5.4: Dependency DAG, Header Guards, and the One Definition Rule)",
                 fontsize=14, fontweight="bold", color="#0D47A1", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


