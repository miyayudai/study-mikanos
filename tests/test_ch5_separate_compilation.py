"""Tests for Chapter 5.2: Separate Compilation and Linking (osbook_day05b)."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    SymbolBinding,
    SymbolSection,
    SymbolSim,
    ObjectFileSim,
    LinkerSim,
    DependencyGraphSim,
    create_day05b_object_files,
    plot_fig5_03_separate_compilation_and_linking_pipeline,
    plot_fig5_04_header_dependency_graph_and_odr,
)


def test_symbol_entry_and_object_file_sim():
    """Verify symbol types, nm character conventions, and object file symbol dumping."""
    sym_t = SymbolSim("func", SymbolSection.TEXT, SymbolBinding.GLOBAL)
    assert sym_t.nm_type_char == "T"
    assert sym_t.is_defined

    sym_local_t = SymbolSim("local_func", SymbolSection.TEXT, SymbolBinding.LOCAL)
    assert sym_local_t.nm_type_char == "t"

    sym_u = SymbolSim("ext_func", SymbolSection.UNDEF, SymbolBinding.GLOBAL)
    assert sym_u.nm_type_char == "U"
    assert not sym_u.is_defined

    obj = ObjectFileSim(filename="test.o", source_cpp="test.cpp")
    obj.add_defined_symbol("foo", SymbolSection.TEXT, size=32)
    obj.add_undefined_symbol("bar")

    nm_dump = obj.dump_nm_style()
    assert len(nm_dump) == 2
    assert "T foo" in nm_dump
    assert "U bar" in nm_dump


def test_linker_simulation_successful_resolution():
    """Verify standard MikanOS day05b modular link: graphics.o + font.o + main.o."""
    graphics_o, font_o, main_o = create_day05b_object_files()

    linker = LinkerSim(base_address=0x0010_0000)
    linker.add_object_file(graphics_o)
    linker.add_object_file(font_o)
    linker.add_object_file(main_o)

    success, report = linker.link()
    assert success, f"Link failed:\n{report}"
    assert len(linker.duplicate_symbols) == 0
    assert len(linker.unresolved_symbols) == 0

    # Ensure key symbols are resolved to addresses >= base_address
    assert "KernelMain" in linker.resolved_symbols
    assert "_Z10WriteAsciiR11PixelWriteriicRK10PixelColor" in linker.resolved_symbols
    assert linker.resolved_symbols["KernelMain"] >= 0x0010_0000
    assert "Link Succeeded: kernel.elf generated" in report


def test_linker_simulation_odr_violation_detection():
    """Verify detection of One Definition Rule (ODR) violation (duplicate definitions)."""
    obj1 = ObjectFileSim(filename="a.o", source_cpp="a.cpp")
    obj1.add_defined_symbol("WriteAscii", SymbolSection.TEXT, size=64)

    obj2 = ObjectFileSim(filename="b.o", source_cpp="b.cpp")
    obj2.add_defined_symbol("WriteAscii", SymbolSection.TEXT, size=64)

    linker = LinkerSim()
    linker.add_object_file(obj1)
    linker.add_object_file(obj2)

    success, report = linker.link()
    assert not success
    assert "WriteAscii" in linker.duplicate_symbols
    assert "Multiple definition (ODR violation)" in report


def test_linker_simulation_undefined_symbol_detection():
    """Verify detection of unresolved external symbols."""
    obj = ObjectFileSim(filename="main.o", source_cpp="main.cpp")
    obj.add_undefined_symbol("NonExistentFunction")

    linker = LinkerSim()
    linker.add_object_file(obj)

    success, report = linker.link()
    assert not success
    assert any(sym == "NonExistentFunction" for sym, _ in linker.unresolved_symbols)
    assert "Undefined symbol references detected" in report


def test_make_dependency_dag_evaluation():
    """Verify Makefile -MM dependency generation and incremental rebuild tracking."""
    dag = DependencyGraphSim()
    dag.add_dependency("graphics.cpp", ["graphics.hpp", "frame_buffer_config.hpp"])
    dag.add_dependency("font.cpp", ["font.hpp", "graphics.hpp"])
    dag.add_dependency("main.cpp", ["frame_buffer_config.hpp", "graphics.hpp", "font.hpp"])

    # Test .d file content
    d_content = dag.generate_d_file_content("main.cpp")
    assert d_content.startswith("main.o: main.cpp")
    assert "graphics.hpp" in d_content
    assert "font.hpp" in d_content

    # Modifying graphics.hpp should trigger rebuild of graphics.o, font.o, and main.o
    targets_gh = dag.determine_rebuild_targets("graphics.hpp")
    assert targets_gh == ["font.o", "graphics.o", "main.o"]

    # Modifying font.cpp should only rebuild font.o
    targets_fc = dag.determine_rebuild_targets("font.cpp")
    assert targets_fc == ["font.o"]

    # Modifying font.hpp should trigger rebuild of font.o and main.o
    targets_fh = dag.determine_rebuild_targets("font.hpp")
    assert targets_fh == ["font.o", "main.o"]


def test_figure_generation_outputs(tmp_path: Path):
    """Verify high-resolution figure generation for Section 5.2."""
    fig1 = tmp_path / "fig5_03_test.png"
    fig2 = tmp_path / "fig5_04_test.png"

    path1 = plot_fig5_03_separate_compilation_and_linking_pipeline(fig1)
    path2 = plot_fig5_04_header_dependency_graph_and_odr(fig2)

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


def test_cpp20_separate_compilation_sim_standalone():
    """Verify standalone C++20 simulation binary builds and passes all tests."""
    binary_path = Path("build/bin/ch05_separate_compilation_sim")
    assert binary_path.exists(), "Binary build/bin/ch05_separate_compilation_sim must exist. Run cmake --build build."

    result = subprocess.run(
        [str(binary_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "[Test 1] Testing Day05b Modular Subsystems Execution..." in result.stdout
    assert "[Test 2] Testing Day05b Linker Symbol Resolution" in result.stdout
    assert "[Test 3] Testing ODR Violation Detection" in result.stdout
    assert "[Test 4] Testing Makefile Dependency DAG and Incremental Rebuild Tracking..." in result.stdout
    assert "All Section 5.2 C++20 Verification Tests Passed Successfully!" in result.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify that Chapter 5.2 Notebook exists, is valid JSON, and has zero execution errors."""
    nb_path = Path("5/5.2_Separate_Compilation_and_Linking.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) >= 12, f"Notebook should have at least 12 detailed cells, got {len(cells)}"

    first_cell_src = "".join(cells[0].get("source", []))
    assert "colab" in first_cell_src.lower() or "google colab" in first_cell_src.lower()

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is not None, f"Code cell {idx} must be executed"
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"Code cell {idx} failed with error: {out.get('evalue')}"
