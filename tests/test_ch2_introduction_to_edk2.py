"""Unit tests for Chapter 2.1: Introduction to EDK II (Edk2PackageModel, DscResolver, BuildPipeline)."""

import subprocess
from pathlib import Path
import pytest

from common.os_sim import (
    Edk2ModuleType,
    Edk2Architecture,
    Edk2DecModel,
    Edk2InfModel,
    Edk2DscModel,
    DscDependencyResolver,
    Edk2BuildPipelineSimulator,
    BuildApproachComparator,
    plot_edk2_architecture_and_packages,
    plot_edk2_build_pipeline_comparison,
)

SAMPLE_DEC = """
[Defines]
  DEC_SPECIFICATION              = 0x00010005
  PACKAGE_NAME                   = MikanLoaderPkg
  PACKAGE_GUID                   = 452eae8e-71e9-11e8-a243-df3f1ffdebe1
  PACKAGE_VERSION                = 0.1

[Includes]
  Include
"""

SAMPLE_INF = """
[Defines]
  INF_VERSION                    = 0x00010006
  BASE_NAME                      = Loader
  FILE_GUID                      = c9d0d202-71e9-11e8-9e52-cfbfd0063fbf
  MODULE_TYPE                    = UEFI_APPLICATION
  VERSION_STRING                 = 0.1
  ENTRY_POINT                    = UefiMain

[Sources]
  Main.c

[Packages]
  MdePkg/MdePkg.dec

[LibraryClasses]
  UefiLib
  UefiApplicationEntryPoint
"""

SAMPLE_DSC = """
[Defines]
  PLATFORM_NAME                  = MikanLoaderPkg
  PLATFORM_GUID                  = d3f11f4e-71e9-11e8-a7e1-33fd4f7d5a3e
  PLATFORM_VERSION               = 0.1
  DSC_SPECIFICATION              = 0x00010005
  OUTPUT_DIRECTORY               = Build/MikanLoader$(ARCH)
  SUPPORTED_ARCHITECTURES        = X64
  BUILD_TARGETS                  = DEBUG|RELEASE|NOOPT

[LibraryClasses]
  UefiApplicationEntryPoint|MdePkg/Library/UefiApplicationEntryPoint/UefiApplicationEntryPoint.inf
  UefiLib|MdePkg/Library/UefiLib/UefiLib.inf
  BaseLib|MdePkg/Library/BaseLib/BaseLib.inf
  BaseMemoryLib|MdePkg/Library/BaseMemoryLib/BaseMemoryLib.inf
  DebugLib|MdePkg/Library/BaseDebugLibNull/BaseDebugLibNull.inf
  DevicePathLib|MdePkg/Library/UefiDevicePathLib/UefiDevicePathLib.inf
  MemoryAllocationLib|MdePkg/Library/UefiMemoryAllocationLib/UefiMemoryAllocationLib.inf
  PcdLib|MdePkg/Library/BasePcdLibNull/BasePcdLibNull.inf
  PrintLib|MdePkg/Library/BasePrintLib/BasePrintLib.inf
  UefiBootServicesTableLib|MdePkg/Library/UefiBootServicesTableLib/UefiBootServicesTableLib.inf
  UefiRuntimeServicesTableLib|MdePkg/Library/UefiRuntimeServicesTableLib/UefiRuntimeServicesTableLib.inf

[Components]
  MikanLoaderPkg/Loader.inf
"""


def test_edk2_dec_parsing():
    """Verify parsing and fields of EDK II Package Declaration (.dec)."""
    dec = Edk2DecModel.parse_dec_text(SAMPLE_DEC)
    assert dec.package_name == "MikanLoaderPkg"
    assert dec.package_guid == "452eae8e-71e9-11e8-a243-df3f1ffdebe1"
    assert dec.package_version == "0.1"
    assert dec.dec_specification == "0x00010005"
    assert "Include" in dec.includes


def test_edk2_inf_parsing():
    """Verify parsing and fields of EDK II Module Information (.inf)."""
    inf = Edk2InfModel.parse_inf_text(SAMPLE_INF)
    assert inf.base_name == "Loader"
    assert inf.file_guid == "c9d0d202-71e9-11e8-9e52-cfbfd0063fbf"
    assert inf.module_type == "UEFI_APPLICATION"
    assert inf.entry_point == "UefiMain"
    assert inf.sources == ["Main.c"]
    assert inf.packages == ["MdePkg/MdePkg.dec"]
    assert "UefiLib" in inf.library_classes
    assert "UefiApplicationEntryPoint" in inf.library_classes


def test_edk2_dsc_parsing():
    """Verify parsing and fields of EDK II Platform Description (.dsc)."""
    dsc = Edk2DscModel.parse_dsc_text(SAMPLE_DSC)
    assert dsc.platform_name == "MikanLoaderPkg"
    assert dsc.supported_architectures == ["X64"]
    assert "DEBUG" in dsc.build_targets
    assert "RELEASE" in dsc.build_targets
    assert len(dsc.library_classes) == 11
    assert "UefiLib" in dsc.library_classes
    assert dsc.library_classes["UefiLib"] == "MdePkg/Library/UefiLib/UefiLib.inf"
    assert dsc.components == ["MikanLoaderPkg/Loader.inf"]


def test_dsc_dependency_resolver_success():
    """Verify that dependency injection binds all required library classes."""
    dsc = Edk2DscModel.parse_dsc_text(SAMPLE_DSC)
    inf = Edk2InfModel.parse_inf_text(SAMPLE_INF)

    resolver = DscDependencyResolver(dsc=dsc)
    resolver.register_module("MikanLoaderPkg/Loader.inf", inf)

    res = resolver.resolve_dependencies("MikanLoaderPkg/Loader.inf")
    assert res["is_valid"] is True
    assert len(res["missing"]) == 0
    assert res["total_required"] == 2
    assert res["total_resolved"] == 2
    assert res["resolved"]["UefiLib"] == "MdePkg/Library/UefiLib/UefiLib.inf"
    assert res["resolved"]["UefiApplicationEntryPoint"] == "MdePkg/Library/UefiApplicationEntryPoint/UefiApplicationEntryPoint.inf"

    tree = resolver.get_full_build_tree()
    assert tree["all_valid"] is True
    assert "MikanLoaderPkg/Loader.inf" in tree["components"]


def test_dsc_dependency_resolver_missing_dependency():
    """Verify error detection when a required library class is missing from DSC."""
    dsc = Edk2DscModel.parse_dsc_text(SAMPLE_DSC)
    inf = Edk2InfModel.parse_inf_text(SAMPLE_INF)
    inf.library_classes.append("CustomGraphicsAccelerationLib")

    resolver = DscDependencyResolver(dsc=dsc)
    resolver.register_module("MikanLoaderPkg/Loader.inf", inf)

    res = resolver.resolve_dependencies("MikanLoaderPkg/Loader.inf")
    assert res["is_valid"] is False
    assert "CustomGraphicsAccelerationLib" in res["missing"]
    assert res["total_resolved"] == 2


def test_edk2_build_pipeline_simulator():
    """Verify that the 6-step EDK II build simulation succeeds."""
    dsc = Edk2DscModel.parse_dsc_text(SAMPLE_DSC)
    inf = Edk2InfModel.parse_inf_text(SAMPLE_INF)

    sim = Edk2BuildPipelineSimulator()
    result = sim.simulate_build(dsc, inf)

    assert result["success"] is True
    assert len(result["steps"]) == 6
    assert all(s["status"] == "PASS" for s in result["steps"])
    assert result["final_efi"] == "Build/MikanLoaderX64/DEBUG_CLANG38/X64/MikanLoaderPkg/Loader/Loader.efi"

    # Verify AutoGen step generated artifacts
    autogen_step = result["steps"][3]
    assert "AutoGen" in autogen_step["name"]
    assert len(autogen_step["generated_files"]) == 3


def test_build_approach_comparator():
    """Verify criteria comparison between manual Clang build and EDK II."""
    comp = BuildApproachComparator()
    assert len(comp.criteria) >= 5
    categories = [c["category"] for c in comp.criteria]
    assert any("ツールチェーン" in cat for cat in categories)
    assert any("UEFI 仕様準拠" in cat for cat in categories)
    assert any("ライブラリ抽象化" in cat for cat in categories)


def test_cpp20_edk2_package_analyzer_executable():
    """Verify C++20 standalone binary execution exits with code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "bin" / "ch02_edk2_package_analyzer"
    if not bin_path.exists():
        bin_path = repo_root / "bin" / "ch02_edk2_package_analyzer"

    assert bin_path.exists(), f"Binary not found at {bin_path}"
    proc = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert "All 5 EDK II Architecture & Dependency Verification Tests PASSED!" in proc.stdout


def test_figure_generation_outputs(tmp_path):
    """Verify high-DPI figure generation functions produce valid PNG files."""
    f1 = tmp_path / "fig2_01.png"
    f2 = tmp_path / "fig2_02.png"

    plot_edk2_architecture_and_packages(f1)
    plot_edk2_build_pipeline_comparison(f2)

    assert f1.exists() and f1.stat().st_size > 10_000
    assert f2.exists() and f2.stat().st_size > 10_000
