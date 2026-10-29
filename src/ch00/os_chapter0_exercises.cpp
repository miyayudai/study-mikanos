/**
 * @file os_chapter0_exercises.cpp
 * @brief Chapter 0 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 0:
 * 1. Privilege Separation & System V AMD64 ABI System Call Dispatcher
 * 2. Bare-Metal GOP Framebuffer Address Calculation & Clipping Protection
 * 3. Binary Search Debugging Algorithm for Hang-up Isolation
 * 4. MikanOS Architectural Dependency Graph & Topological Verification
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises {

// =============================================================================
// Exercise 1: System V AMD64 ABI System Call Dispatcher Simulation
// =============================================================================

enum class SyscallId : uint64_t {
    kExit = 1,
    kPutString = 2,
    kAllocatePage = 3,
    kUnknown = 999,
};

struct SyscallRegisters {
    uint64_t rax = 0;  // Syscall number
    uint64_t rdi = 0;  // Arg 1
    uint64_t rsi = 0;  // Arg 2
    uint64_t rdx = 0;  // Arg 3
    uint64_t r10 = 0;  // Arg 4
    uint64_t r8  = 0;  // Arg 5
    uint64_t r9  = 0;  // Arg 6
};

struct SyscallResult {
    int64_t return_value;
    bool success;
    std::string message;
};

class MockSyscallDispatcher {
public:
    static SyscallResult Dispatch(const SyscallRegisters& regs) {
        switch (static_cast<SyscallId>(regs.rax)) {
            case SyscallId::kExit: {
                int exit_code = static_cast<int>(regs.rdi);
                return {exit_code, true, "Process exit requested with code: " + std::to_string(exit_code)};
            }
            case SyscallId::kPutString: {
                size_t len = static_cast<size_t>(regs.rsi);
                if (regs.rdi == 0 || len == 0) {
                    return {-1, false, "Invalid string pointer or length"};
                }
                return {static_cast<int64_t>(len), true, "String printed successfully (" + std::to_string(len) + " bytes)"};
            }
            case SyscallId::kAllocatePage: {
                size_t num_pages = static_cast<size_t>(regs.rdi);
                if (num_pages == 0 || num_pages > 1024) {
                    return {-1, false, "Requested page count out of bounds"};
                }
                uint64_t mock_virtual_addr = 0x40000000 + (num_pages * 4096);
                return {static_cast<int64_t>(mock_virtual_addr), true, "Pages allocated at virtual address"};
            }
            default:
                return {-38, false, "ENOSYS: Function not implemented"}; // Standard POSIX ENOSYS
        }
    }
};

// =============================================================================
// Exercise 2: GOP Framebuffer Address Calculation & Clipping Protection
// =============================================================================

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    uint32_t pixels_per_scan_line;
};

class FrameBufferCalculator {
public:
    static uint64_t CalculatePixelOffset(const FrameBufferConfig& config, uint32_t x, uint32_t y) {
        assert(x < config.horizontal_resolution && "X coordinate out of bounds");
        assert(y < config.vertical_resolution && "Y coordinate out of bounds");
        // Each pixel in 32-bit GOP takes 4 bytes (RGBA / BGRA)
        return (static_cast<uint64_t>(y) * config.pixels_per_scan_line + x) * 4;
    }

    static bool DrawPixelSafe(const FrameBufferConfig& config, uint32_t x, uint32_t y, uint32_t argb_color) {
        if (x >= config.horizontal_resolution || y >= config.vertical_resolution) {
            return false; // Clipping protection triggered
        }
        uint64_t offset = CalculatePixelOffset(config, x, y);
        uint32_t* pixel_addr = reinterpret_cast<uint32_t*>(config.frame_buffer + offset);
        *pixel_addr = argb_color;
        return true;
    }
};

// =============================================================================
// Exercise 3: Bare-Metal Binary Search Debugging Algorithm
// =============================================================================

class BareMetalDebugger {
public:
    /**
     * @brief Finds the exact failing line within [1, total_lines] using binary search.
     * @param total_lines Total number of lines/instructions in kernel initialization.
     * @param probe A callable returning true if kernel survives up to line K, false if it crashed before or at K.
     * @return The first line number where the fault occurs.
     */
    static int FindFaultyLine(int total_lines, const std::function<bool(int)>& probe, int& out_probe_count) {
        int low = 1;
        int high = total_lines;
        int faulty_line = -1;
        out_probe_count = 0;

        while (low <= high) {
            out_probe_count++;
            int mid = low + (high - low) / 2;
            bool survived = probe(mid);

            if (survived) {
                // If it survived up to mid, the bug must be strictly after mid
                low = mid + 1;
            } else {
                // If it crashed at or before mid, mid could be the faulty line or after it
                faulty_line = mid;
                high = mid - 1;
            }
        }
        return faulty_line;
    }
};

// =============================================================================
// Exercise 4: MikanOS Architecture DAG & Topological Verification
// =============================================================================

struct ArchitectureDependencyChecker {
    std::map<std::string, std::vector<std::string>> adj_list;
    std::map<std::string, int> in_degree;

    void AddDependency(const std::string& prerequisite, const std::string& dependent) {
        adj_list[prerequisite].push_back(dependent);
        if (!in_degree.contains(prerequisite)) in_degree[prerequisite] = 0;
        in_degree[dependent]++;
    }

    bool VerifyAcyclicAndGetOrder(std::vector<std::string>& out_order) const {
        std::map<std::string, int> deg = in_degree;
        std::queue<std::string> q;

        for (const auto& [node, d] : deg) {
            if (d == 0) q.push(node);
        }

        out_order.clear();
        while (!q.empty()) {
            std::string curr = q.front();
            q.pop();
            out_order.push_back(curr);

            auto it = adj_list.find(curr);
            if (it != adj_list.end()) {
                for (const auto& next : it->second) {
                    if (--deg[next] == 0) {
                        q.push(next);
                    }
                }
            }
        }

        return out_order.size() == in_degree.size();
    }
};

} // namespace mikanos::exercises

// =============================================================================
// Self-Testing Runner
// =============================================================================

int main() {
    using namespace mikanos::exercises;

    std::cout << "========================================================\n";
    std::cout << " MikanOS Chapter 0 Comprehensive Exercises (C++20)\n";
    std::cout << "========================================================\n\n";

    // Test Exercise 1: System Call Dispatcher
    std::cout << "[Exercise 1] Verifying System V AMD64 ABI Syscall Dispatcher...\n";
    {
        SyscallRegisters exit_regs{.rax = 1, .rdi = 42};
        auto res_exit = MockSyscallDispatcher::Dispatch(exit_regs);
        assert(res_exit.success && res_exit.return_value == 42);

        SyscallRegisters print_regs{.rax = 2, .rdi = 0x1000, .rsi = 13};
        auto res_print = MockSyscallDispatcher::Dispatch(print_regs);
        assert(res_print.success && res_print.return_value == 13);

        SyscallRegisters invalid_print{.rax = 2, .rdi = 0, .rsi = 10};
        auto res_invalid_p = MockSyscallDispatcher::Dispatch(invalid_print);
        assert(!res_invalid_p.success && res_invalid_p.return_value == -1);

        SyscallRegisters alloc_regs{.rax = 3, .rdi = 4};
        auto res_alloc = MockSyscallDispatcher::Dispatch(alloc_regs);
        assert(res_alloc.success && res_alloc.return_value == 0x40004000);

        SyscallRegisters unknown_regs{.rax = 999};
        auto res_unknown = MockSyscallDispatcher::Dispatch(unknown_regs);
        assert(!res_unknown.success && res_unknown.return_value == -38);

        std::cout << "  -> PASSED: All syscall dispatch cases verified successfully.\n";
    }

    // Test Exercise 2: Framebuffer Address & Clipping
    std::cout << "\n[Exercise 2] Verifying Framebuffer Offset & Clipping Safety...\n";
    {
        std::vector<uint8_t> buffer(800 * 600 * 4, 0);
        FrameBufferConfig config{
            .frame_buffer = buffer.data(),
            .horizontal_resolution = 800,
            .vertical_resolution = 600,
            .pixels_per_scan_line = 800,
        };

        uint64_t offset_0_0 = FrameBufferCalculator::CalculatePixelOffset(config, 0, 0);
        assert(offset_0_0 == 0);

        uint64_t offset_100_50 = FrameBufferCalculator::CalculatePixelOffset(config, 100, 50);
        assert(offset_100_50 == (50 * 800 + 100) * 4); // 160400 bytes

        bool draw_ok = FrameBufferCalculator::DrawPixelSafe(config, 100, 50, 0xFF00FF00); // Green
        assert(draw_ok);
        uint32_t val = *reinterpret_cast<uint32_t*>(buffer.data() + offset_100_50);
        assert(val == 0xFF00FF00);

        // Out of bounds checks
        bool clip_x = FrameBufferCalculator::DrawPixelSafe(config, 800, 100, 0xFFFFFFFF);
        assert(!clip_x);
        bool clip_y = FrameBufferCalculator::DrawPixelSafe(config, 100, 600, 0xFFFFFFFF);
        assert(!clip_y);

        std::cout << "  -> PASSED: GOP offset calculation & boundary clipping verified.\n";
    }

    // Test Exercise 3: Binary Search Debugging
    std::cout << "\n[Exercise 3] Verifying Binary Search Bare-Metal Debugger...\n";
    {
        const int total_lines = 1000;
        const int secret_crash_line = 734;

        auto mock_kernel_run = [secret_crash_line](int line_reached) -> bool {
            // Kernel survives if line_reached < secret_crash_line; crashes at secret_crash_line
            return line_reached < secret_crash_line;
        };

        int probe_count = 0;
        int found_line = BareMetalDebugger::FindFaultyLine(total_lines, mock_kernel_run, probe_count);
        assert(found_line == secret_crash_line);
        assert(probe_count <= 10); // ceil(log2(1000)) = 10

        std::cout << "  -> PASSED: Crash line " << found_line << " isolated in "
                  << probe_count << " probes (O(log N) budget: 10).\n";
    }

    // Test Exercise 4: Architecture DAG Verification
    std::cout << "\n[Exercise 4] Verifying MikanOS Architecture Dependency DAG...\n";
    {
        ArchitectureDependencyChecker checker;
        checker.AddDependency("UEFI_BootLoader", "Kernel_Entry");
        checker.AddDependency("Kernel_Entry", "GOP_PixelWriter");
        checker.AddDependency("GOP_PixelWriter", "Console_Text");
        checker.AddDependency("Kernel_Entry", "GDT_Segmentation");
        checker.AddDependency("GDT_Segmentation", "IDT_Interrupts");
        checker.AddDependency("IDT_Interrupts", "Local_APIC_Timer");
        checker.AddDependency("Local_APIC_Timer", "Task_Scheduler");
        checker.AddDependency("Task_Scheduler", "User_Applications");

        std::vector<std::string> order;
        bool is_valid_dag = checker.VerifyAcyclicAndGetOrder(order);
        assert(is_valid_dag);
        assert(order.front() == "UEFI_BootLoader");
        assert(order.back() == "User_Applications");

        std::cout << "  -> PASSED: Architectural dependency DAG verified without cycles.\n";
        std::cout << "  -> Valid build sequence:\n     ";
        for (size_t i = 0; i < order.size(); ++i) {
            std::cout << order[i] << (i + 1 < order.size() ? " -> " : "\n");
        }
    }

    std::cout << "\n========================================================\n";
    std::cout << " All Chapter 0 C++20 Exercises Passed Successfully!\n";
    std::cout << "========================================================\n";
    return 0;
}
