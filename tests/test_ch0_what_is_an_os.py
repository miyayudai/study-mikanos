"""Unit tests for Chapter 0 Section 0.2: What is an OS and POSIX."""

from pathlib import Path
import subprocess
import pytest

from common import MIKANOS_SYSCALL_SPECS, MockKernelSyscallDispatcher, SyscallSpec


def test_mikanos_syscall_specs() -> None:
    """Verify MikanOS system call specifications match textbook and architecture specs."""
    assert len(MIKANOS_SYSCALL_SPECS) >= 10
    by_name = {spec.name: spec for spec in MIKANOS_SYSCALL_SPECS}

    assert "PutString" in by_name
    assert "Exit" in by_name
    assert "OpenWindow" in by_name
    assert "WinDrawLine" in by_name

    put_str = by_name["PutString"]
    assert put_str.number == 2
    assert put_str.is_posix_standard is True

    exit_spec = by_name["Exit"]
    assert exit_spec.number == 3
    assert exit_spec.is_posix_standard is True

    open_win = by_name["OpenWindow"]
    assert open_win.number == 4
    assert open_win.is_posix_standard is False


def test_mock_kernel_syscall_dispatcher_normal() -> None:
    """Test normal stdout and stderr handling in MockKernelSyscallDispatcher."""
    dispatcher = MockKernelSyscallDispatcher()

    # Write to stdout (fd=1)
    ret1 = dispatcher.dispatch(syscall_num=2, arg1=1, arg2="Hello, ")
    assert ret1 == 7

    # Write to stdout again
    ret2 = dispatcher.dispatch(syscall_num=2, arg1=1, arg2="MikanOS!\n")
    assert ret2 == 9

    assert dispatcher.get_stdout() == "Hello, MikanOS!\n"
    assert len(dispatcher.syscall_history) == 2

    # Exit application
    ret_exit = dispatcher.dispatch(syscall_num=3, arg1=0, arg2=0)
    assert ret_exit == 0
    assert dispatcher.syscall_history[-1] == (3, "Exit", 0)


def test_mock_kernel_syscall_dispatcher_errors() -> None:
    """Test error handling in MockKernelSyscallDispatcher."""
    dispatcher = MockKernelSyscallDispatcher()

    # Invalid file descriptor (EBADF = -9)
    ret_bad_fd = dispatcher.dispatch(syscall_num=2, arg1=99, arg2="Invalid")
    assert ret_bad_fd == -9

    # Unimplemented system call (ENOSYS = -38)
    ret_unimplemented = dispatcher.dispatch(syscall_num=999, arg1=0, arg2=0)
    assert ret_unimplemented == -38


def test_cpp20_os_posix_architecture_execution(tmp_path: Path) -> None:
    """Compile and execute standalone C++20 src/ch00/os_posix_architecture.cpp."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch00" / "os_posix_architecture.cpp"
    assert cpp_src.exists()

    bin_path = tmp_path / "os_posix_architecture"
    compile_res = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", str(cpp_src), "-o", str(bin_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0
    assert "Hardware Abstraction Test Passed" in run_res.stdout
    assert "Memory Safety Check Passed" in run_res.stdout
    assert "Process Control Passed" in run_res.stdout


def test_generated_figures_exist() -> None:
    """Ensure Section 0.2 high-resolution figures exist and are sufficiently large (> 100KB)."""
    repo_root = Path(__file__).resolve().parent.parent
    fig3 = repo_root / "0" / "result" / "fig0_3_os_architecture_layers.png"
    fig4 = repo_root / "0" / "result" / "fig0_4_posix_system_call_flow.png"

    assert fig3.exists() and fig3.stat().st_size > 100_000
    assert fig4.exists() and fig4.stat().st_size > 100_000
