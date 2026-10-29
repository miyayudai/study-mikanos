"""Core simulation and visualization utilities for MikanOS architecture study."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
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








