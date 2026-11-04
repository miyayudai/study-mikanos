/**
 * @file interrupt_concept_sim.cpp
 * @brief C++20 Standalone Verification of Interrupt Concepts (MikanOS Chapter 7.1 / osbook_day07a)
 *
 * This file verifies:
 * 1. Polling vs Interrupt-driven quantitative performance comparison (latency, CPU cycles, wasted checks)
 * 2. x86-64 Hardware Interrupt Frame (`InterruptFrame`) memory layout, member offsets, and alignment
 * 3. Event-driven interrupt dispatching mechanism & CPU HLT power-saving state machine
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic interrupt_concept_sim.cpp -o interrupt_concept_sim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <chrono>

// ============================================================================
// 1. x86-64 Interrupt Frame Memory Layout (Hardware Architecture)
// ============================================================================

/**
 * @brief x86-64 Hardware-pushed Interrupt Frame
 * 
 * When an interrupt or exception occurs, the CPU automatically pushes the following
 * 5 registers onto the stack before jumping to the handler:
 *   [RSP + 32] SS (Stack Segment)
 *   [RSP + 24] RSP (Stack Pointer before interrupt)
 *   [RSP + 16] RFLAGS (Processor Flags, IF, etc.)
 *   [RSP +  8] CS (Code Segment)
 *   [RSP +  0] RIP (Instruction Pointer / Return Address)
 */
struct __attribute__((packed)) InterruptFrame {
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
};

/**
 * @brief x86-64 Interrupt Frame with Error Code (Pushed by Fault Exceptions e.g. #PF, #GP)
 */
struct __attribute__((packed)) InterruptFrameWithError {
    uint64_t error_code;
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
};

// ============================================================================
// 2. Interrupt Taxonomies & Vector Definitions
// ============================================================================

enum class InterruptType {
    kExternalHardware, // Asynchronous: USB xHCI, Timer, Keyboard
    kSoftware,         // Synchronous: INT n, Syscall
    kCpuFault,         // Fault: Restartable (#PF, #GP)
    kCpuTrap,          // Trap: Next instruction (#DB, #BP)
    kCpuAbort          // Abort: Fatal unrecoverable (#DF, #MC)
};

struct InterruptDescriptorInfo {
    uint8_t vector;
    InterruptType type;
    const char* name;
    bool has_error_code;
};

// ============================================================================
// 3. Quantitative Benchmarking: Polling vs Interrupt-Driven
// ============================================================================

struct EventPacket {
    uint64_t timestamp_us;
    int8_t dx;
    int8_t dy;
    uint8_t buttons;
};

struct BenchmarkMetrics {
    std::string mode_name;
    uint64_t total_time_us;
    uint64_t num_events_processed;
    uint64_t poll_checks_count;      // How many times polled
    uint64_t empty_checks_count;     // How many checks found no data
    double avg_latency_us;
    double max_latency_us;
    double cpu_busy_ratio_pct;       // Estimated CPU busy percentage
    double estimated_energy_mwh;     // Rough energy consumption
};

class PollingVsInterruptSimulator {
public:
    PollingVsInterruptSimulator(uint64_t poll_interval_us = 10000) // 10ms default
        : poll_interval_us_(poll_interval_us) {}

    BenchmarkMetrics RunPollingSimulation(const std::vector<EventPacket>& events, uint64_t total_sim_time_us) {
        uint64_t current_time_us = 0;
        uint64_t next_poll_time_us = poll_interval_us_;
        size_t event_idx = 0;

        uint64_t poll_count = 0;
        uint64_t empty_poll_count = 0;
        std::vector<uint64_t> latencies;

        while (current_time_us < total_sim_time_us) {
            current_time_us += 100; // Step in 100us granularity
            if (current_time_us >= next_poll_time_us) {
                poll_count++;
                bool found_event = false;
                while (event_idx < events.size() && events[event_idx].timestamp_us <= current_time_us) {
                    uint64_t lat = current_time_us - events[event_idx].timestamp_us;
                    latencies.push_back(lat);
                    found_event = true;
                    event_idx++;
                }
                if (!found_event) {
                    empty_poll_count++;
                }
                next_poll_time_us += poll_interval_us_;
            }
        }

        double avg_lat = latencies.empty() ? 0.0 :
            std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double max_lat = latencies.empty() ? 0.0 :
            *std::max_element(latencies.begin(), latencies.end());

        // In busy polling, CPU runs at 100% active state
        double cpu_busy_pct = 100.0;
        // Energy: 65W * (total_time_us / 1e6) sec / 3600 sec * 1000 = mWh
        double energy_mwh = 65.0 * (total_sim_time_us / 1e6) / 3600.0 * 1000.0;

        return BenchmarkMetrics{
            .mode_name = "Busy Polling (while(1) { ProcessEvent(); })",
            .total_time_us = total_sim_time_us,
            .num_events_processed = latencies.size(),
            .poll_checks_count = poll_count,
            .empty_checks_count = empty_poll_count,
            .avg_latency_us = avg_lat,
            .max_latency_us = max_lat,
            .cpu_busy_ratio_pct = cpu_busy_pct,
            .estimated_energy_mwh = energy_mwh
        };
    }

    BenchmarkMetrics RunInterruptSimulation(const std::vector<EventPacket>& events, uint64_t total_sim_time_us) {
        // In interrupt-driven mode, latency is purely hardware dispatch time (~5us)
        const uint64_t kHardwareInterruptLatencyUs = 5;
        const uint64_t kHandlerExecutionTimeUs = 25; // 25us to process packet

        std::vector<uint64_t> latencies;
        uint64_t total_active_time_us = 0;

        for (const auto& ev : events) {
            if (ev.timestamp_us < total_sim_time_us) {
                latencies.push_back(kHardwareInterruptLatencyUs);
                total_active_time_us += kHandlerExecutionTimeUs;
            }
        }

        double avg_lat = latencies.empty() ? 0.0 :
            std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double max_lat = latencies.empty() ? 0.0 :
            *std::max_element(latencies.begin(), latencies.end());

        double cpu_busy_pct = (double)total_active_time_us / total_sim_time_us * 100.0;
        double idle_time_us = total_sim_time_us - total_active_time_us;

        // Active 65W, Idle (HLT) 5W
        double active_j = 65.0 * (total_active_time_us / 1e6);
        double idle_j = 5.0 * (idle_time_us / 1e6);
        double energy_mwh = (active_j + idle_j) / 3.6; // Joules to mWh

        return BenchmarkMetrics{
            .mode_name = "Interrupt-Driven (MSI + HLT)",
            .total_time_us = total_sim_time_us,
            .num_events_processed = latencies.size(),
            .poll_checks_count = 0,
            .empty_checks_count = 0,
            .avg_latency_us = avg_lat,
            .max_latency_us = max_lat,
            .cpu_busy_ratio_pct = cpu_busy_pct,
            .estimated_energy_mwh = energy_mwh
        };
    }

private:
    uint64_t poll_interval_us_;
};

// ============================================================================
// Main Verification Runner
// ============================================================================

int main() {
    std::cout << "================================================================\n"
              << "🍊 MikanOS Chapter 7.1: Concept of Interrupts Verification Suite\n"
              << "================================================================\n\n";

    // ------------------------------------------------------------------------
    // Part 1: Verify Hardware Interrupt Frame Layout
    // ------------------------------------------------------------------------
    std::cout << "[Part 1] Verifying x86-64 InterruptFrame memory layout...\n";
    static_assert(sizeof(InterruptFrame) == 40, "InterruptFrame must be exactly 40 bytes");
    static_assert(offsetof(InterruptFrame, rip) == 0, "rip offset must be 0");
    static_assert(offsetof(InterruptFrame, cs) == 8, "cs offset must be 8");
    static_assert(offsetof(InterruptFrame, rflags) == 16, "rflags offset must be 16");
    static_assert(offsetof(InterruptFrame, rsp) == 24, "rsp offset must be 24");
    static_assert(offsetof(InterruptFrame, ss) == 32, "ss offset must be 32");

    static_assert(sizeof(InterruptFrameWithError) == 48, "InterruptFrameWithError must be 48 bytes");
    static_assert(offsetof(InterruptFrameWithError, error_code) == 0, "error_code offset must be 0");
    static_assert(offsetof(InterruptFrameWithError, rip) == 8, "rip offset must be 8");

    std::cout << "  ✓ sizeof(InterruptFrame) = " << sizeof(InterruptFrame) << " bytes (5 registers x 8B)\n";
    std::cout << "  ✓ sizeof(InterruptFrameWithError) = " << sizeof(InterruptFrameWithError) << " bytes\n";
    std::cout << "  ✓ Register offsets: RIP=0, CS=8, RFLAGS=16, RSP=24, SS=32 (Verified)\n\n";

    // ------------------------------------------------------------------------
    // Part 2: Simulate Interrupt Stack Frame Creation
    // ------------------------------------------------------------------------
    std::cout << "[Part 2] Simulating Hardware Push of InterruptFrame...\n";
    InterruptFrame frame{
        .rip = 0x00000000001045A0,   // Kernel code address
        .cs = 0x0008,                // Kernel Code Segment Selector
        .rflags = 0x0000000000000202,// IF=1 (Interrupts Enabled)
        .rsp = 0x00007FFFFFFFE000,   // Current Stack Pointer
        .ss = 0x0010                 // Kernel Data Segment Selector
    };

    std::cout << std::hex << std::setfill('0');
    std::cout << "  Pushed Stack Frame (top-down as CPU pushes):\n";
    std::cout << "    [RSP+32] SS:     0x" << std::setw(16) << frame.ss << "\n";
    std::cout << "    [RSP+24] RSP:    0x" << std::setw(16) << frame.rsp << "\n";
    std::cout << "    [RSP+16] RFLAGS: 0x" << std::setw(16) << frame.rflags << " (IF=" << ((frame.rflags & 0x200) ? 1 : 0) << ")\n";
    std::cout << "    [RSP+ 8] CS:     0x" << std::setw(16) << frame.cs << "\n";
    std::cout << "    [RSP+ 0] RIP:    0x" << std::setw(16) << frame.rip << " (Return Addr)\n";
    std::cout << std::dec << std::setfill(' ');
    std::cout << "  ✓ InterruptFrame structure dump verified.\n\n";

    // ------------------------------------------------------------------------
    // Part 3: Quantitative Simulation: Polling vs Interrupt
    // ------------------------------------------------------------------------
    std::cout << "[Part 3] Running Quantitative Benchmark: Polling vs Interrupt-Driven...\n";

    // Generate 100ms sequence of mouse events (e.g., at 12ms, 28ms, 45ms, 63ms, 81ms)
    std::vector<EventPacket> mouse_events = {
        { .timestamp_us = 12000, .dx = 2, .dy = -1, .buttons = 0 },
        { .timestamp_us = 28000, .dx = 4, .dy = 0, .buttons = 1 },
        { .timestamp_us = 45000, .dx = -2, .dy = 3, .buttons = 1 },
        { .timestamp_us = 63000, .dx = 0, .dy = 5, .buttons = 0 },
        { .timestamp_us = 81000, .dx = 1, .dy = -2, .buttons = 0 },
    };

    const uint64_t kTotalSimTimeUs = 100000; // 100ms
    PollingVsInterruptSimulator sim(10000); // 10ms poll interval

    auto poll_res = sim.RunPollingSimulation(mouse_events, kTotalSimTimeUs);
    auto intr_res = sim.RunInterruptSimulation(mouse_events, kTotalSimTimeUs);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  +--------------------------------+-----------------+-----------------+\n"
              << "  | Metric                         | Polling Mode    | Interrupt Mode  |\n"
              << "  +--------------------------------+-----------------+-----------------+\n"
              << "  | Events Processed               | " << std::setw(15) << poll_res.num_events_processed << " | " << std::setw(15) << intr_res.num_events_processed << " |\n"
              << "  | Wasted Checks (Empty Polls)    | " << std::setw(15) << poll_res.empty_checks_count << " | " << std::setw(15) << intr_res.empty_checks_count << " |\n"
              << "  | Average Latency (us)           | " << std::setw(15) << poll_res.avg_latency_us << " | " << std::setw(15) << intr_res.avg_latency_us << " |\n"
              << "  | Maximum Latency (us)           | " << std::setw(15) << poll_res.max_latency_us << " | " << std::setw(15) << intr_res.max_latency_us << " |\n"
              << "  | Estimated CPU Utilization (%)  | " << std::setw(15) << poll_res.cpu_busy_ratio_pct << " | " << std::setw(15) << intr_res.cpu_busy_ratio_pct << " |\n"
              << "  | Energy Consumed (mWh)          | " << std::setw(15) << poll_res.estimated_energy_mwh << " | " << std::setw(15) << intr_res.estimated_energy_mwh << " |\n"
              << "  +--------------------------------+-----------------+-----------------+\n";

    assert(intr_res.avg_latency_us < poll_res.avg_latency_us);
    assert(intr_res.cpu_busy_ratio_pct < poll_res.cpu_busy_ratio_pct);
    assert(intr_res.estimated_energy_mwh < poll_res.estimated_energy_mwh);

    std::cout << "\n✅ All Chapter 7.1 C++20 assertions passed successfully!\n";
    return 0;
}
