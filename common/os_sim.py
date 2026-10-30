"""Core simulation and visualization utilities for MikanOS architecture study."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, ArrowStyle
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
