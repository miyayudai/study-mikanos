"""Tests for Chapter 4.4: Virtual Method Table (vtable) and __cxa_pure_virtual."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    PureVirtualCallError,
    VTableEntry,
    VirtualMethodTable,
    PolymorphicObjectLayout,
    ItaniumAbiVTableSimulator,
    VPtrInspector,
    plot_fig4_07_vtable_memory_layout_and_dispatch,
    plot_fig4_08_cxa_pure_virtual_and_abi,
)


def test_itanium_abi_vtable_structure_and_entries():
    """Verify simulated Itanium C++ ABI vtables for base and derived classes."""
    sim = ItaniumAbiVTableSimulator()

    # Base class PixelWriter (Abstract)
    pw_vt = sim.vtables["PixelWriter"]
    assert pw_vt.class_name == "PixelWriter"
    assert pw_vt.vtable_base_address == 0x403000

    offset_to_top = pw_vt.get_entry_by_offset(-16)
    assert offset_to_top is not None
    assert offset_to_top.target_address == 0

    typeinfo = pw_vt.get_entry_by_offset(-8)
    assert typeinfo is not None
    assert "_ZTI11PixelWriter" in typeinfo.symbol_name

    write_entry = pw_vt.lookup_function("Write")
    assert write_entry is not None
    assert write_entry.offset_bytes == 16
    assert write_entry.is_pure_virtual is True
    assert write_entry.symbol_name == "__cxa_pure_virtual"

    # Derived class RGBResv8BitPerColorPixelWriter (Concrete)
    rgb_vt = sim.vtables["RGBResv8BitPerColorPixelWriter"]
    assert rgb_vt.vtable_base_address == 0x403100
    rgb_write = rgb_vt.lookup_function("Write")
    assert rgb_write is not None
    assert rgb_write.is_pure_virtual is False
    assert "RGBResv8BitPerColorPixelWriter" in rgb_write.symbol_name

    # Derived class BGRResv8BitPerColorPixelWriter (Concrete)
    bgr_vt = sim.vtables["BGRResv8BitPerColorPixelWriter"]
    assert bgr_vt.vtable_base_address == 0x403200
    bgr_write = bgr_vt.lookup_function("Write")
    assert bgr_write is not None
    assert bgr_write.is_pure_virtual is False
    assert "BGRResv8BitPerColorPixelWriter" in bgr_write.symbol_name


def test_polymorphic_object_layout_and_vptr():
    """Verify memory layout of polymorphic class instances containing _vptr and members."""
    sim = ItaniumAbiVTableSimulator()
    obj = sim.create_instance("RGBResv8BitPerColorPixelWriter", instance_address=0x200000)

    assert obj.total_size == 16
    assert obj.alignment == 8
    assert obj.vptr_value == 0x403100
    assert len(obj.members) == 2

    # Member 0: _vptr at offset 0
    m0 = obj.members[0]
    assert m0["name"] == "_vptr"
    assert m0["offset"] == 0
    assert m0["size"] == 8

    # Member 1: config_ reference pointer at offset 8
    m1 = obj.members[1]
    assert m1["name"] == "config_"
    assert m1["offset"] == 8
    assert m1["size"] == 8

    # Memory map ASCII text
    mem_map = obj.format_memory_map()
    assert "_vptr" in mem_map
    assert "config_" in mem_map
    assert "+0x00" in mem_map
    assert "+0x08" in mem_map


def test_dynamic_dispatch_rgb_and_bgr():
    """Verify that dynamic dispatch resolves to the correct concrete method."""
    sim = ItaniumAbiVTableSimulator()

    rgb_obj = sim.create_instance("RGBResv8BitPerColorPixelWriter")
    addr_rgb, sym_rgb = sim.dispatch(rgb_obj, "Write", 10, 20, (255, 0, 0))
    assert addr_rgb == 0x401200
    assert "RGB" in sym_rgb

    bgr_obj = sim.create_instance("BGRResv8BitPerColorPixelWriter")
    addr_bgr, sym_bgr = sim.dispatch(bgr_obj, "Write", 10, 20, (255, 0, 0))
    assert addr_bgr == 0x401300
    assert "BGR" in sym_bgr


def test_pure_virtual_call_trap_raises_exception():
    """Verify that invoking a pure virtual function slot triggers PureVirtualCallError."""
    sim = ItaniumAbiVTableSimulator()

    # Creating an object pointing to abstract base class vtable
    abstract_obj = sim.create_instance("PixelWriter")

    with pytest.raises(PureVirtualCallError) as exc_info:
        sim.dispatch(abstract_obj, "Write", 0, 0, (0, 0, 0))

    assert "__cxa_pure_virtual()" in str(exc_info.value)
    assert sim.pure_virtual_call_count == 1


def test_vptr_inspector_raw_bytes_and_disassembly():
    """Verify low-level binary inspection and disassembly generation."""
    # Little-endian 64-bit address 0x0000000000403100
    raw_bytes = bytes([0x00, 0x31, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00])
    vptr = VPtrInspector.inspect_vptr(raw_bytes)
    assert vptr == 0x403100

    with pytest.raises(ValueError):
        VPtrInspector.inspect_vptr(b"\x00\x00")

    asm_lines = VPtrInspector.disassemble_indirect_call(method_offset=16)
    assert len(asm_lines) == 3
    assert "mov   rax, [rdi]" in asm_lines[0]
    assert "mov   rax, [rax + 16]" in asm_lines[1]
    assert "call  rax" in asm_lines[2]


def test_figure_generation_outputs(tmp_path):
    """Verify generation, resolution, and valid image format for Figures 4.7 and 4.8."""
    fig1 = tmp_path / "fig4_07_test.png"
    fig2 = tmp_path / "fig4_08_test.png"

    path1 = plot_fig4_07_vtable_memory_layout_and_dispatch(str(fig1))
    path2 = plot_fig4_08_cxa_pure_virtual_and_abi(str(fig2))

    assert Path(path1).exists()
    assert Path(path2).exists()

    with Image.open(path1) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 2000
        assert img1.format == "PNG"

    with Image.open(path2) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 2000
        assert img2.format == "PNG"


def test_cpp20_vtable_sim_standalone():
    """Verify standalone C++20 simulation binary builds and passes all tests."""
    binary_path = Path("build/bin/ch04_vtable_sim")
    assert binary_path.exists(), "Binary build/bin/ch04_vtable_sim must exist. Run cmake --build build."

    result = subprocess.run(
        [str(binary_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "Extracting vptr and invoking method via raw vtable pointer arithmetic... PASSED" in result.stdout
    assert "Verifying Itanium C++ ABI negative offsets (offset-to-top & typeinfo)... PASSED" in result.stdout
    assert "Simulating x86-64 indirect call machine instructions... PASSED" in result.stdout
    assert "Verifying pure virtual function call trap (__cxa_pure_virtual)..." in result.stdout
    assert "Verifying vtable memory footprint and static alignments... PASSED" in result.stdout
    assert "All Section 4.4 C++20 Verification Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 4.4 Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("4/4.4_Virtual_Method_Table_vtable.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 12, f"Notebook should have at least 12 detailed cells, got {len(cells)}"

    # Check cell 0 has Colab badge / setup
    first_cell_src = "".join(cells[0].get("source", []))
    assert "colab" in first_cell_src.lower() or "google colab" in first_cell_src.lower()

    # Verify every code cell executed without errors
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} must be executed"
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"Code cell {idx} failed with error: {out.get('evalue')}"
