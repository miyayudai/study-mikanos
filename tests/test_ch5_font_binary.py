"""Tests for Chapter 5.3: Generating Hankaku Font Object (osbook_day05c)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    HankakuFontParser,
    ObjcopyBinarySimulator,
    create_day05c_object_files,
    LinkerSim,
    plot_fig5_05_objcopy_binary_to_elf_pipeline,
    plot_fig5_06_hankaku_font_memory_and_symbol_resolution,
)


def test_hankaku_font_parser_compile_and_decompile():
    """Verify HankakuFontParser compilation and glyph matrix extraction."""
    sample_glyph_txt = (
        "........\n"
        "...@@...\n"
        "..@..@..\n"
        "..@..@..\n"
        "..@@@@..\n"
        "..@..@..\n"
        "..@..@..\n"
        "........\n"
        "........\n"
        "........\n"
        "........\n"
        "........\n"
        "........\n"
        "........\n"
        "........\n"
        "........\n"
    )
    compiled = HankakuFontParser.compile_text(sample_glyph_txt)
    assert len(compiled) == 16

    # Verify second row: ...@@... -> 0b00011000 = 24
    assert compiled[1] == 24

    # Verify fifth row: ..@@@@.. -> 0b00111100 = 60
    assert compiled[4] == 60

    decompiled = HankakuFontParser.decompile_glyph(compiled)
    assert "...@@..." in decompiled
    assert "..@@@@.." in decompiled

    matrix = HankakuFontParser.get_glyph_matrix(compiled)
    assert len(matrix) == 16
    assert len(matrix[0]) == 8
    assert matrix[1] == [0, 0, 0, 1, 1, 0, 0, 0]


def test_objcopy_binary_simulator_symbols():
    """Verify objcopy symbol generation for raw binary objects."""
    sim = ObjcopyBinarySimulator(filename="hankaku.bin", base_address=0x0010_5000)
    assert sim.size == 4096

    assert "_binary_hankaku_bin_start" in sim.symbols
    assert "_binary_hankaku_bin_end" in sim.symbols
    assert "_binary_hankaku_bin_size" in sim.symbols

    start_sym = sim.symbols["_binary_hankaku_bin_start"]
    end_sym = sim.symbols["_binary_hankaku_bin_end"]
    size_sym = sim.symbols["_binary_hankaku_bin_size"]

    assert start_sym.address == 0x0010_5000
    assert start_sym.symbol_type == "D"
    assert end_sym.address == 0x0010_5000 + 4096
    assert size_sym.symbol_type == "A"
    assert size_sym.address == 4096  # Absolute symbol value equals address


def test_objcopy_c_symbol_resolution_simulation():
    """Verify simulation of C++ symbol dereference mechanics vs the &size trap."""
    sim = ObjcopyBinarySimulator(filename="hankaku.bin", base_address=0x0010_5000)
    resolution = sim.simulate_c_symbol_resolution()

    correct_size = resolution["correct_size_retrieval"]
    assert correct_size["valid"] is True
    assert correct_size["result_value"] == 4096

    buggy_size = resolution["buggy_size_retrieval"]
    assert buggy_size["valid"] is False
    assert buggy_size["result_value"] is None

    correct_ptr = resolution["correct_data_pointer"]
    assert correct_ptr["valid"] is True
    assert correct_ptr["result_address"] == "0x105000"


def test_get_font_address_arithmetic_and_bounds():
    """Verify GetFont address arithmetic and bounds checking."""
    # Create simulator with 4096 bytes
    dummy_data = bytearray(4096)
    # Set pattern for 'A' (ASCII 65, offset 1040)
    dummy_data[65 * 16 + 1] = 0x18
    dummy_data[65 * 16 + 9] = 0x7E

    sim = ObjcopyBinarySimulator(filename="hankaku.bin", data=bytes(dummy_data))

    font_a = sim.get_font('A')
    assert font_a is not None
    assert len(font_a) == 16
    assert font_a[1] == 0x18
    assert font_a[9] == 0x7E

    # Character 0xFF (255 * 16 + 16 = 4096) -> within bounds
    font_ff = sim.get_font(255)
    assert font_ff is not None

    # Out of bounds: 256
    font_oob = sim.get_font(256)
    assert font_oob is None


def test_day05c_modular_linking_with_hankaku_o():
    """Verify linker resolution for MikanOS day05c with hankaku.o."""
    graphics_o, font_o, main_o, hankaku_o = create_day05c_object_files()

    linker = LinkerSim(base_address=0x0010_0000)
    linker.add_object_file(graphics_o)
    linker.add_object_file(font_o)
    linker.add_object_file(main_o)
    linker.add_object_file(hankaku_o)

    success, report = linker.link()
    assert success, f"Link failed:\n{report}"
    assert len(linker.duplicate_symbols) == 0
    assert len(linker.unresolved_symbols) == 0

    assert "_binary_hankaku_bin_start" in linker.resolved_symbols
    assert "_binary_hankaku_bin_size" in linker.resolved_symbols
    assert "KernelMain" in linker.resolved_symbols


def test_figure_generation_outputs(tmp_path):
    """Verify high-resolution architecture diagrams Figure 5.5 and Figure 5.6."""
    fig5_05_path = tmp_path / "fig5_05.png"
    fig5_06_path = tmp_path / "fig5_06.png"

    p1 = plot_fig5_05_objcopy_binary_to_elf_pipeline(fig5_05_path)
    p2 = plot_fig5_06_hankaku_font_memory_and_symbol_resolution(fig5_06_path)

    assert Path(p1).exists()
    assert Path(p2).exists()

    with Image.open(p1) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 2000

    with Image.open(p2) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 2000


def test_cpp20_font_binary_sim_standalone():
    """Verify C++20 standalone binary execution."""
    build_bin = Path("build/bin/ch05_font_binary_sim")
    if not build_bin.exists():
        subprocess.run(["cmake", "--build", "build", "--target", "ch05_font_binary_sim"], check=True)

    result = subprocess.run([str(build_bin)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "All Section 5.3 C++20 Verification Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify 5.3 notebook existence, cell structure, and error-free execution."""
    nb_path = Path("5/5.3_Generating_Hankaku_Font_Object.ipynb")
    assert nb_path.exists(), "Notebook 5.3 must exist."

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 12, "Notebook must have comprehensive section structure."

    # Verify Cell 0 is Google Colab setup
    cell_0_src = "".join(cells[0].get("source", []))
    assert "Google Colab" in cell_0_src, "Cell 0 must be Colab setup."

    # Verify all code cells have executed without errors
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Cell {idx} has not been executed."
            outputs = cell.get("outputs", [])
            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"Cell {idx} execution failed with error: {out.get('ename')}: {out.get('evalue')}"
                )
