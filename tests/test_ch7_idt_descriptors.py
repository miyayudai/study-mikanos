"""Tests for Chapter 7.4: IDT (Interrupt Descriptor Table) and 16-byte Gate Descriptors."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    DescriptorType,
    InterruptDescriptorAttribute,
    make_idt_attr,
    InterruptDescriptor,
    IDTR,
    IDTTable,
    plot_idt_descriptor_16byte_layout,
    plot_lidt_instruction_and_vector_dispatch,
)


def test_descriptor_type_enum():
    """Verify x86-64 descriptor type constants."""
    assert DescriptorType.kUpper8Bytes == 0
    assert DescriptorType.kLDT == 2
    assert DescriptorType.kTSSAvailable == 9
    assert DescriptorType.kTSSBusy == 11
    assert DescriptorType.kCallGate == 12
    assert DescriptorType.kInterruptGate == 14
    assert DescriptorType.kTrapGate == 15


def test_interrupt_descriptor_attribute_bitfield():
    """Verify 16-bit InterruptDescriptorAttribute serialization and deserialization."""
    # Standard kernel Interrupt Gate (Type=0xE, DPL=0, Present=1, IST=0)
    attr = make_idt_attr(DescriptorType.kInterruptGate, descriptor_privilege_level=0, present=True, interrupt_stack_table=0)
    assert attr.type == DescriptorType.kInterruptGate
    assert attr.descriptor_privilege_level == 0
    assert attr.present is True
    assert attr.interrupt_stack_table == 0
    # Expected raw value: (14 << 8) | (1 << 15) = 0x8E00
    assert attr.raw_value == 0x8E00

    # Round-trip deserialization
    unpacked = InterruptDescriptorAttribute.from_raw(0x8E00)
    assert unpacked.type == DescriptorType.kInterruptGate
    assert unpacked.descriptor_privilege_level == 0
    assert unpacked.present is True
    assert unpacked.interrupt_stack_table == 0

    # User-level Trap Gate with IST (Type=0xF, DPL=3, Present=1, IST=2)
    user_attr = make_idt_attr(DescriptorType.kTrapGate, descriptor_privilege_level=3, present=True, interrupt_stack_table=2)
    # Expected: 2 | (15 << 8) | (3 << 13) | (1 << 15) = 2 | 0x0F00 | 0x6000 | 0x8000 = 0xEF02
    assert user_attr.raw_value == 0xEF02
    unpacked_user = InterruptDescriptorAttribute.from_raw(0xEF02)
    assert unpacked_user.type == DescriptorType.kTrapGate
    assert unpacked_user.descriptor_privilege_level == 3
    assert unpacked_user.present is True
    assert unpacked_user.interrupt_stack_table == 2


def test_interrupt_descriptor_structure_and_bytes():
    """Verify 16-byte (128-bit) InterruptDescriptor layout and 64-bit address splitting."""
    desc = InterruptDescriptor()
    test_addr = 0x123456789ABCDEF0
    kernel_cs = 0x0008
    attr = make_idt_attr(DescriptorType.kInterruptGate, 0)

    desc.attr = attr
    desc.segment_selector = kernel_cs
    desc.set_offset(test_addr)

    assert desc.offset_low == 0xDEF0
    assert desc.offset_middle == 0x9ABC
    assert desc.offset_high == 0x12345678
    assert desc.offset == test_addr
    assert desc.reserved == 0

    # Binary packaging (must be 16 bytes)
    raw_bytes = desc.to_bytes()
    assert len(raw_bytes) == 16

    # Unpack from bytes
    restored = InterruptDescriptor.from_bytes(raw_bytes)
    assert restored.offset == test_addr
    assert restored.segment_selector == kernel_cs
    assert restored.attr.type == DescriptorType.kInterruptGate
    assert restored.attr.present is True
    assert restored.reserved == 0

    # Canonical Higher-Half address test
    higher_half = 0xFFFF800000102030
    desc.set_offset(higher_half)
    assert desc.offset == higher_half
    assert desc.offset_high == 0xFFFF8000
    assert desc.offset_middle == 0x0010
    assert desc.offset_low == 0x2030


def test_idtr_structure_and_limits():
    """Verify 10-byte (80-bit) IDTR structure for LIDT instruction."""
    idtr = IDTR(limit=4095, base=0x0000000000100000)
    raw = idtr.to_bytes()
    assert len(raw) == 10

    unpacked = IDTR.from_bytes(raw)
    assert unpacked.limit == 4095
    assert unpacked.base == 0x0000000000100000

    # Standard 256-entry IDT limit formula: sizeof(idt) - 1 = 256 * 16 - 1 = 4095
    assert 256 * 16 - 1 == 4095


def test_idt_table_and_dispatch_simulation():
    """Verify IDTTable 256 entries management and CPU dispatch simulation."""
    table = IDTTable(256)
    assert len(table.entries) == 256

    # IDTR generation
    idtr = table.make_idtr(0x200000)
    assert idtr.limit == 4095
    assert idtr.base == 0x200000

    # Dispatch to unconfigured vector (Not present)
    disp_uninit = table.dispatch(0x40)
    assert not disp_uninit["success"]
    assert disp_uninit["error"] == "NotPresent"

    # Configure Vector 0x40 (xHCI MSI interrupt)
    handler_addr = 0x105000
    cs = 0x0008
    attr = make_idt_attr(DescriptorType.kInterruptGate, 0)
    table.set_entry(0x40, attr, handler_addr, cs)

    disp_xhci = table.dispatch(0x40)
    assert disp_xhci["success"] is True
    assert disp_xhci["handler_address"] == handler_addr
    assert disp_xhci["segment_selector"] == cs
    assert disp_xhci["type"] == "kInterruptGate"
    assert disp_xhci["dpl"] == 0

    # Bounds check
    disp_oob = table.dispatch(256)
    assert not disp_oob["success"]
    assert disp_oob["error"] == "VectorOutOfRange"


def test_figure_generation_and_validity(tmp_path):
    """Verify high-DPI figure generation for Figures 7.07 and 7.08."""
    fig1 = tmp_path / "fig7_07_idt_descriptor_16byte_layout.png"
    fig2 = tmp_path / "fig7_08_lidt_instruction_and_vector_dispatch.png"

    plot_idt_descriptor_16byte_layout(fig1)
    plot_lidt_instruction_and_vector_dispatch(fig2)

    assert fig1.exists()
    assert fig2.exists()
    assert fig1.stat().st_size > 50_000
    assert fig2.stat().st_size > 50_000

    with Image.open(fig1) as img1:
        assert img1.size[0] >= 2000
        assert img1.size[1] >= 1500

    with Image.open(fig2) as img2:
        assert img2.size[0] >= 2000
        assert img2.size[1] >= 1200


def test_cpp20_idt_descriptor_sim_executable():
    """Verify C++20 standalone simulator compilation and successful execution."""
    repo_root = Path(__file__).resolve().parent.parent
    src_file = repo_root / "src/ch07/idt_descriptor_sim.cpp"
    bin_file = repo_root / "src/ch07/idt_descriptor_sim_test_bin"

    assert src_file.exists()

    compile_cmd = [
        "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic",
        str(src_file), "-o", str(bin_file)
    ]
    res_compile = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert res_compile.returncode == 0, f"Compilation failed: {res_compile.stderr}"

    res_run = subprocess.run([str(bin_file)], capture_output=True, text=True)
    assert res_run.returncode == 0
    assert "All Chapter 7.4 C++20 Verification Tests PASSED Successfully!" in res_run.stdout
    assert "0x8E00" in res_run.stdout
    assert "Vector 0x40" in res_run.stdout

    if bin_file.exists():
        bin_file.unlink()


def test_notebook_execution_and_integrity():
    """Verify Section 7.4 Jupyter Notebook structure, cell outputs, and Colab compatibility."""
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "7/7.4_Configuring_IDT_Descriptors.ipynb"

    assert nb_path.exists(), "Notebook 7/7.4_Configuring_IDT_Descriptors.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 10, "Notebook must contain at least 10 rich cells"

    # Cell 0 must be Colab auto-setup
    cell0 = cells[0]
    assert cell0["cell_type"] == "code"
    cell0_source = "".join(cell0["source"])
    assert "google.colab" in cell0_source
    assert "study-mikanos" in cell0_source

    # Check for error outputs across all code cells
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", f"Error output detected in cell {i}: {out.get('evalue')}"

    # Verify key topics in Markdown cells
    full_text = "\n".join("".join(c["source"]) for c in cells)
    assert "7.4 割り込み記述子の設定" in full_text
    assert "InterruptDescriptor" in full_text
    assert "MakeIDTAttr" in full_text
    assert "SetIDTEntry" in full_text
    assert "LoadIDT" in full_text
    assert "LIDT" in full_text
    assert "0x8E00" in full_text or "0x8e" in full_text.lower()
