// src/ch00/os_debug_strategies.cpp
// Section 0.4: Small-step OS Development, QEMU vs Real Hardware & Bare-Metal Debugging Strategies
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::ch00 {

// 1. Primitive RGB color representation for VRAM visual marker debugging
struct RgbColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    constexpr bool operator==(const RgbColor& other) const = default;
};

// Canonical debugging palette used before console printk is implemented
namespace debug_colors {
inline constexpr RgbColor kBlack{0, 0, 0};
inline constexpr RgbColor kRed{255, 0, 0};          // Checkpoint 1: Bootloader entered
inline constexpr RgbColor kGreen{0, 255, 0};        // Checkpoint 2: Memory map acquired
inline constexpr RgbColor kBlue{0, 0, 255};          // Checkpoint 3: Kernel jumped
inline constexpr RgbColor kYellow{255, 255, 0};      // Checkpoint 4: Paging initialized
inline constexpr RgbColor kCyan{0, 255, 255};        // Checkpoint 5: Interrupts enabled
inline constexpr RgbColor kWhite{255, 255, 255};      // Checkpoint 6: Main loop reached
}  // namespace debug_colors

// 2. Types of bare-metal debugging strategies
enum class DebugStrategyType {
    kFramebufferMarker,  // Solid VRAM color changes to signal checkpoint arrival
    kHaltLoop,           // 'while(1);' or 'cli; hlt;' to freeze CPU at checkpoint
    kSerialUart,         // Direct MMIO/I/O writes to 0x3F8 (COM1 serial port)
    kQemuMonitor,        // CPU register & physical memory inspection via QEMU monitor
    kBinarySearch,       // Systematic binary search halting across source lines
};

// 3. Simulated Bare-Metal Execution Environment
// Models an OS startup sequence with an injected fault at 'bug_line'
class BareMetalExecutionEnvironment {
public:
    explicit BareMetalExecutionEnvironment(uint32_t total_lines, uint32_t bug_line)
        : total_lines_(total_lines), bug_line_(bug_line) {
        assert(bug_line_ >= 1 && bug_line_ <= total_lines_);
    }

    // Attempts to execute up to checkpoint_line with a VRAM color marker and a serial character
    bool ExecuteCheckpoint(uint32_t checkpoint_line, RgbColor marker_color, char serial_char) {
        if (checkpoint_line < bug_line_) {
            current_vram_ = marker_color;
            serial_output_ += serial_char;
            executed_checkpoints_.push_back(checkpoint_line);
            return true;
        }
        // CPU crashed / hung before or at reaching this checkpoint
        crashed_ = true;
        return false;
    }

    [[nodiscard]] RgbColor CurrentVramColor() const noexcept { return current_vram_; }
    [[nodiscard]] std::string_view SerialOutput() const noexcept { return serial_output_; }
    [[nodiscard]] bool HasCrashed() const noexcept { return crashed_; }
    [[nodiscard]] uint32_t TotalLines() const noexcept { return total_lines_; }
    [[nodiscard]] uint32_t BugLine() const noexcept { return bug_line_; }
    [[nodiscard]] const std::vector<uint32_t>& ExecutedCheckpoints() const noexcept {
        return executed_checkpoints_;
    }

private:
    uint32_t total_lines_;
    uint32_t bug_line_;
    RgbColor current_vram_{debug_colors::kBlack};
    std::string serial_output_{};
    std::vector<uint32_t> executed_checkpoints_{};
    bool crashed_{false};
};

// 4. Binary Search Debugging Instrumentation
struct BinarySearchProbe {
    size_t iteration;
    uint32_t low;
    uint32_t high;
    uint32_t probe_line;
    bool passed;  // true if code up to probe_line executed without fault
};

class BinarySearchDebugger {
public:
    static std::vector<BinarySearchProbe> LocateBug(uint32_t total_lines, uint32_t bug_line) {
        assert(bug_line >= 1 && bug_line <= total_lines);
        std::vector<BinarySearchProbe> history;
        uint32_t low = 1;
        uint32_t high = total_lines;
        size_t iter = 1;

        while (low < high) {
            uint32_t mid = low + (high - low) / 2;
            bool passed = (mid < bug_line);
            history.push_back({iter++, low, high, mid, passed});
            if (passed) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return history;
    }
};

// 5. Small-Step Incremental Development Milestone Model
struct DevelopmentMilestone {
    std::string_view day_tag;
    uint32_t chapter;
    std::string_view title;
    size_t added_lines;
    std::string_view verification_cue;
};

inline constexpr std::array<DevelopmentMilestone, 8> kKeyMilestones{{
    {"day01a", 1, "Minimal UEFI App", 4, "Immediate exit code 0 to firmware"},
    {"day02a", 2, "EDK II Hello World", 18, "Text 'Hello, world!' rendered on UEFI console"},
    {"day03c", 3, "Kernel Pixel Drawing", 48, "Entire screen painted white/yellow via GOP"},
    {"day05e", 5, "Console Class & Font", 130, "Smooth text scrolling on graphical desktop"},
    {"day06c", 6, "xHCI Mouse Driver", 240, "Mouse cursor tracks physical mouse movement"},
    {"day07b", 7, "IDT MSI Interrupts", 190, "Zero-latency event delivery via ArrayQueue FIFO"},
    {"day08c", 8, "Bitmap Memory Manager", 160, "Dynamic allocation of 4KiB physical frames"},
    {"day13d", 13, "Preemptive Multitasking", 280, "Parallel counter tasks switching at 20ms"},
}};

class SmallStepAnalyzer {
public:
    static size_t CalculateTotalLines(std::span<const DevelopmentMilestone> milestones) {
        size_t total = 0;
        for (const auto& m : milestones) {
            total += m.added_lines;
        }
        return total;
    }

    // Big-Bang development inspection cost: O(N)
    static double BigBangWorstCaseCost(size_t total_lines) {
        return static_cast<double>(total_lines);
    }

    // Small-Step average inspection cost per iteration: O(N / K)
    static double SmallStepAverageCost(size_t total_lines, size_t milestones_count) {
        if (milestones_count == 0) return 0.0;
        return static_cast<double>(total_lines) / static_cast<double>(milestones_count);
    }
};

}  // namespace mikanos::ch00

int main() {
    using namespace mikanos::ch00;

    std::cout << "====================================================================\n";
    std::cout << " MikanOS Section 0.4: Small-step OS Dev & Debugging Verification\n";
    std::cout << "====================================================================\n\n";

    // 1. Verify VRAM Color Marker & Serial UART Debugging
    std::cout << "[1] Bare-Metal Visual VRAM & Serial UART Simulation:\n";
    const uint32_t kTotalLines = 100;
    const uint32_t kBugLine = 42;  // Injected crash at line 42

    BareMetalExecutionEnvironment env(kTotalLines, kBugLine);

    // Checkpoint 1: Line 10 (Bootloader setup) -> RED, 'B'
    bool cp1 = env.ExecuteCheckpoint(10, debug_colors::kRed, 'B');
    assert(cp1 == true);
    assert(env.CurrentVramColor() == debug_colors::kRed);

    // Checkpoint 2: Line 30 (Memory map ready) -> GREEN, 'M'
    bool cp2 = env.ExecuteCheckpoint(30, debug_colors::kGreen, 'M');
    assert(cp2 == true);
    assert(env.CurrentVramColor() == debug_colors::kGreen);

    // Checkpoint 3: Line 50 (After bug at 42!) -> BLUE, 'K'
    bool cp3 = env.ExecuteCheckpoint(50, debug_colors::kBlue, 'K');
    assert(cp3 == false);  // Should fail because line 42 crashed
    assert(env.HasCrashed() == true);
    // VRAM color should remain GREEN from checkpoint 2, serial output should be "BM"
    assert(env.CurrentVramColor() == debug_colors::kGreen);
    assert(env.SerialOutput() == "BM");

    std::cout << "  - Total source lines : " << kTotalLines << "\n";
    std::cout << "  - Injected bug line  : " << kBugLine << "\n";
    std::cout << "  - VRAM Color frozen  : GREEN (R=0, G=255, B=0) -> Crash happened between Line 30 and 50!\n";
    std::cout << "  - Serial Port buffer : \"" << env.SerialOutput() << "\" (UART 0x3F8 logged 'B', 'M')\n\n";

    // 2. Verify Binary Search Debugging Convergence
    std::cout << "[2] Binary Search Pinpointing Bug Location:\n";
    auto bsearch_steps = BinarySearchDebugger::LocateBug(kTotalLines, kBugLine);

    std::cout << "  Iter | Search Window [Low..High] | Probe Line | Outcome\n";
    std::cout << "  -----+---------------------------+------------+---------\n";
    for (const auto& step : bsearch_steps) {
        std::cout << "   " << std::setw(2) << step.iteration << "  | ["
                  << std::setw(3) << step.low << " .. " << std::setw(3) << step.high << "]         | "
                  << std::setw(5) << step.probe_line << "      | "
                  << (step.passed ? "PASS (continue)" : "CRASH (hung)") << "\n";
    }

    // The final narrowed window must pinpoint the exact bug line
    assert(!bsearch_steps.empty());
    const auto& final_step = bsearch_steps.back();
    uint32_t found_line = final_step.passed ? (final_step.probe_line + 1) : final_step.probe_line;
    assert(found_line == kBugLine);

    size_t theoretical_max_steps = static_cast<size_t>(std::ceil(std::log2(kTotalLines)));
    std::cout << "  -> Located exact bug at Line " << found_line << " in " << bsearch_steps.size()
              << " steps (Theoretical upper bound ceil(log2(" << kTotalLines << ")) = "
              << theoretical_max_steps << "). PASS!\n\n";

    // 3. Verify Small-Step Incremental Development vs Big-Bang
    std::cout << "[3] Small-Step Incremental Development Cost Model:\n";
    size_t cumulative_lines = SmallStepAnalyzer::CalculateTotalLines(kKeyMilestones);
    double big_bang_cost = SmallStepAnalyzer::BigBangWorstCaseCost(cumulative_lines);
    double small_step_cost = SmallStepAnalyzer::SmallStepAverageCost(cumulative_lines, kKeyMilestones.size());

    std::cout << "  - Tracked key milestones : " << kKeyMilestones.size() << " stages\n";
    std::cout << "  - Total cumulative code  : " << cumulative_lines << " lines\n";
    std::cout << "  - Big-Bang search space  : " << big_bang_cost << " lines (O(N) global audit)\n";
    std::cout << "  - Small-Step delta space : " << small_step_cost << " lines per step (O(N/K) delta diff)\n";
    std::cout << "  - Debugging cost ratio   : " << std::fixed << std::setprecision(1)
              << (big_bang_cost / small_step_cost) << "x search space reduction!\n\n";

    for (const auto& m : kKeyMilestones) {
        std::cout << "    [" << m.day_tag << "] Ch " << std::setw(2) << m.chapter << " "
                  << std::left << std::setw(26) << m.title
                  << " +" << std::right << std::setw(3) << m.added_lines << " lines | "
                  << m.verification_cue << "\n";
    }

    std::cout << "\n>>> All C++20 Section 0.4 verifications PASSED successfully! <<<\n";
    return 0;
}
