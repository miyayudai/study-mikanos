// src/ch00/os_posix_architecture.cpp
// Section 0.2: OS Architecture, Privilege Rings, POSIX Abstraction & System Call Dispatching
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mikanos::ch00 {

// CPU Privilege Level (x86-64 Segment Descriptor DPL)
enum class PrivilegeLevel : uint8_t {
    kRing0Kernel = 0,      // Supervisor mode (Kernel, Device Drivers)
    kRing1DeviceDrivers = 1,
    kRing2DeviceDrivers = 2,
    kRing3User = 3,        // User mode (Applications, Shell)
};

// System Call Numbers in MikanOS ABI (corresponds to day20a-day22g)
enum class SyscallNumber : uint64_t {
    kLogString = 1,
    kPutString = 2,
    kExit = 3,
    kOpenWindow = 4,
    kWinWriteString = 5,
    kWinFillRectangle = 6,
    kGetCurrentTick = 7,
    kWinRedraw = 8,
    kWinDrawLine = 9,
    kCloseWindow = 10,
    kReadEvent = 11,
};

// System Call Result Structure (returned in RAX:RDX)
struct SyscallResult {
    int64_t value;
    int error_code;  // 0 on success, negative or errno on failure
};

// Simulated Hardware Device
struct MockHardwareConsole {
    std::vector<std::string> screen_lines;

    void RenderText(std::string_view text) {
        screen_lines.emplace_back(text);
    }
};

// Simulated Operating System Kernel (Ring 0)
class MockKernel {
public:
    static constexpr int kFdStdin = 0;
    static constexpr int kFdStdout = 1;
    static constexpr int kFdStderr = 2;

    MockKernel() = default;

    // The kernel dispatch entrypoint invoked by the x86-64 SYSCALL instruction.
    // Privileged transition: CPU switches CS/SS to Ring 0 and jumps to LSTAR address.
    SyscallResult HandleSyscall(SyscallNumber number,
                                uint64_t arg1,
                                uint64_t arg2,
                                uint64_t arg3) {
        switch (number) {
            case SyscallNumber::kPutString: {
                // arg1: file descriptor, arg2: const char* ptr, arg3: size_t len
                int fd = static_cast<int>(arg1);
                auto* str_ptr = reinterpret_cast<const char*>(arg2);
                size_t len = static_cast<size_t>(arg3);

                if (str_ptr == nullptr) {
                    return {-1, -14};  // -EFAULT: Bad address
                }
                if (fd != kFdStdout && fd != kFdStderr) {
                    return {-1, -9};   // -EBADF: Bad file descriptor
                }

                std::string msg(str_ptr, len);
                hardware_console_.RenderText(msg);
                return {static_cast<int64_t>(len), 0};
            }
            case SyscallNumber::kExit: {
                // arg1: exit code
                app_exit_code_ = static_cast<int>(arg1);
                is_app_running_ = false;
                return {0, 0};
            }
            default:
                return {-1, -38};  // -ENOSYS: Function not implemented
        }
    }

    [[nodiscard]] const std::vector<std::string>& GetScreenOutput() const noexcept {
        return hardware_console_.screen_lines;
    }

    [[nodiscard]] std::optional<int> GetAppExitCode() const noexcept {
        return app_exit_code_;
    }

    [[nodiscard]] bool IsAppRunning() const noexcept {
        return is_app_running_;
    }

private:
    MockHardwareConsole hardware_console_;
    std::optional<int> app_exit_code_{std::nullopt};
    bool is_app_running_{true};
};

// Standard C Library (Newlib) system call stubs linked into Userland Application (Ring 3)
class NewlibSupport {
public:
    explicit NewlibSupport(MockKernel& kernel) : kernel_(kernel) {}

    // POSIX write() stub implementation:
    // ssize_t write(int fd, const void *buf, size_t count);
    int64_t Write(int fd, const void* buf, size_t count) {
        // Emit CPU SYSCALL instruction:
        // Pass SyscallNumber::kPutString in RAX, fd in RDI, buf in RSI, count in RDX
        auto res = kernel_.HandleSyscall(
            SyscallNumber::kPutString,
            static_cast<uint64_t>(fd),
            reinterpret_cast<uintptr_t>(buf),
            count
        );
        return res.value;
    }

    // POSIX _exit() stub implementation:
    // void _exit(int status);
    void Exit(int status) {
        kernel_.HandleSyscall(
            SyscallNumber::kExit,
            static_cast<uint64_t>(status),
            0,
            0
        );
    }

private:
    MockKernel& kernel_;
};

}  // namespace mikanos::ch00

int main() {
    using namespace mikanos::ch00;

    std::cout << "=== Section 0.2: OS Architecture, POSIX & System Call Dispatching (C++20) ===\n";

    // 1. Static Verification of Architectural Constants
    static_assert(sizeof(SyscallNumber) == sizeof(uint64_t), "Syscall numbers must fit in 64-bit RAX register");
    static_assert(static_cast<uint8_t>(PrivilegeLevel::kRing0Kernel) == 0, "Ring 0 must be kernel mode");
    static_assert(static_cast<uint8_t>(PrivilegeLevel::kRing3User) == 3, "Ring 3 must be user mode");

    // 2. Instantiate Kernel (Ring 0) and C Library / Userland Glue (Ring 3)
    MockKernel kernel;
    NewlibSupport libc(kernel);

    // 3. Verify Hardware Abstraction via POSIX write()
    const char* greeting = "Hello, MikanOS World via POSIX abstraction!\n";
    const size_t greeting_len = std::strlen(greeting);

    int64_t written_bytes = libc.Write(MockKernel::kFdStdout, greeting, greeting_len);
    assert(written_bytes == static_cast<int64_t>(greeting_len));
    assert(kernel.GetScreenOutput().size() == 1);
    assert(kernel.GetScreenOutput()[0] == greeting);

    std::cout << "[+] Hardware Abstraction Test Passed: Wrote " << written_bytes << " bytes to stdout.\n";

    // 4. Verify Resource Protection & Error Handling (EBADF for invalid fd)
    int64_t bad_fd_res = libc.Write(999, greeting, greeting_len);
    assert(bad_fd_res == -1);
    std::cout << "[+] Protection Check Passed: Invalid file descriptor rejected by kernel.\n";

    // 5. Verify Fault Handling (EFAULT for null buffer pointer)
    auto null_ptr_res = kernel.HandleSyscall(SyscallNumber::kPutString, 1, 0, 10);
    assert(null_ptr_res.error_code == -14);
    std::cout << "[+] Memory Safety Check Passed: Null buffer pointer gracefully rejected (-EFAULT).\n";

    // 6. Verify Process Lifecycle Termination via POSIX _exit()
    assert(kernel.IsAppRunning());
    libc.Exit(42);
    assert(!kernel.IsAppRunning());
    assert(kernel.GetAppExitCode().value_or(-1) == 42);
    std::cout << "[+] Process Control Passed: App terminated with exit code "
              << *kernel.GetAppExitCode() << " via SyscallExit.\n";

    std::cout << "\n[Summary]\n"
              << "- Ring 3 (User App) communicates with Ring 0 (Kernel) strictly via system calls.\n"
              << "- Hardware I/O details are abstracted behind POSIX-compliant standard C library functions.\n"
              << "- CPU privilege rings guarantee that faulty user apps cannot corrupt OS internal state.\n";

    return 0;
}
