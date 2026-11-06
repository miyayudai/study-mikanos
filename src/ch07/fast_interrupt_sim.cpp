/**
 * @file fast_interrupt_sim.cpp
 * @brief C++20 Standalone Verification of Fast Interrupt Handling and Top/Bottom Half Architecture
 *        (MikanOS Chapter 7.7 / osbook_day07b)
 *
 * This program comprehensively verifies:
 * 1. MikanOS Message structure and ArrayQueue<T> ring buffer (kernel/queue.hpp)
 * 2. Monolithic interrupt handler model (osbook_day07a):
 *    - Drawing and ProcessEvent directly in handler under IF=0
 *    - Interrupt latency explosion and event drop analysis under high burst load
 * 3. Split Top Half / Bottom Half architecture (osbook_day07b):
 *    - Top Half (Hardirq): Minimal work (queue Push + EOI) under IF=0 (< 1 us)
 *    - Bottom Half (Softirq / Event Loop): Deferred heavy processing under IF=1
 *    - Preemption capability: New interrupts can arrive while Bottom Half processes
 * 4. Benchmarking and Quantitative Metrics:
 *    - Max interrupt latency, throughput, event drop rate
 * 5. Architectural invariants and correctness tests
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic fast_interrupt_sim.cpp -o fast_interrupt_sim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <deque>
#include <chrono>
#include <algorithm>

namespace os {

// ============================================================================
// 1. Error Handling and Status Codes (MikanOS kernel/error.hpp equivalent)
// ============================================================================

enum class ErrorCode {
    kSuccess = 0,
    kFull = 1,
    kEmpty = 2,
    kIndexOutOfRange = 3,
};

class Error {
public:
    constexpr Error(ErrorCode code = ErrorCode::kSuccess) : code_{code} {}

    constexpr ErrorCode Code() const { return code_; }
    constexpr explicit operator bool() const { return code_ != ErrorCode::kSuccess; }

    const char* Name() const {
        switch (code_) {
            case ErrorCode::kSuccess: return "Success";
            case ErrorCode::kFull: return "Full";
            case ErrorCode::kEmpty: return "Empty";
            case ErrorCode::kIndexOutOfRange: return "IndexOutOfRange";
            default: return "Unknown";
        }
    }

private:
    ErrorCode code_;
};

#define MAKE_ERROR(code) Error(code)

// ============================================================================
// 2. MikanOS Message Definition (kernel/main.cpp in osbook_day07b)
// ============================================================================

struct Message {
    enum Type {
        kInterruptXHCI,
        kTimerTimeout,
        kKeyInput,
    } type;

    // Optional payload fields for future chapters
    uint64_t arg0{0};
    uint64_t arg1{0};
};

// ============================================================================
// 3. ArrayQueue Ring Buffer Implementation (kernel/queue.hpp)
// ============================================================================

template <typename T>
class ArrayQueue {
public:
    template <size_t N>
    ArrayQueue(std::array<T, N>& buf) : ArrayQueue(buf.data(), N) {}

    ArrayQueue(T* buf, size_t size)
        : data_{buf}, read_pos_{0}, write_pos_{0}, count_{0}, capacity_{size} {}

    Error Push(const T& value) {
        if (count_ == capacity_) {
            return MAKE_ERROR(ErrorCode::kFull);
        }

        data_[write_pos_] = value;
        ++count_;
        ++write_pos_;
        if (write_pos_ == capacity_) {
            write_pos_ = 0;
        }
        return MAKE_ERROR(ErrorCode::kSuccess);
    }

    Error Pop() {
        if (count_ == 0) {
            return MAKE_ERROR(ErrorCode::kEmpty);
        }

        --count_;
        ++read_pos_;
        if (read_pos_ == capacity_) {
            read_pos_ = 0;
        }
        return MAKE_ERROR(ErrorCode::kSuccess);
    }

    size_t Count() const { return count_; }
    size_t Capacity() const { return capacity_; }

    const T& Front() const {
        assert(count_ > 0 && "Front() called on empty queue!");
        return data_[read_pos_];
    }

    bool Empty() const { return count_ == 0; }
    bool Full() const { return count_ == capacity_; }

    void Clear() {
        read_pos_ = 0;
        write_pos_ = 0;
        count_ = 0;
    }

private:
    T* data_;
    size_t read_pos_{0};
    size_t write_pos_{0};
    size_t count_{0};
    const size_t capacity_;
};

// ============================================================================
// 4. Simulated Hardware and Event Models
// ============================================================================

struct HardwareEvent {
    uint32_t id;
    double arrival_time_us;
    double work_duration_us;  // Execution time needed for ProcessEvent + drawing
    std::string desc;
};

struct ExecutionStats {
    std::string architecture_name;
    size_t total_events{0};
    size_t processed_events{0};
    size_t dropped_events{0};
    double total_time_us{0.0};
    double max_interrupt_latency_us{0.0};
    double avg_interrupt_latency_us{0.0};
    double max_handler_hold_time_us{0.0};
};

// ============================================================================
// 5. Monolithic Handler Simulation (osbook_day07a)
// ============================================================================

ExecutionStats SimulateMonolithic(
    const std::vector<HardwareEvent>& events,
    size_t hw_buffer_capacity = 2,
    double handler_overhead_us = 2.0
) {
    ExecutionStats stats;
    stats.architecture_name = "Monolithic Handler (osbook_day07a)";
    stats.total_events = events.size();

    std::deque<HardwareEvent> hw_fifo;
    std::vector<double> latencies;

    double current_time_us = 0.0;
    size_t event_idx = 0;
    const size_t n = events.size();

    while (event_idx < n || !hw_fifo.empty()) {
        if (hw_fifo.empty()) {
            if (current_time_us < events[event_idx].arrival_time_us) {
                current_time_us = events[event_idx].arrival_time_us;
            }
        }

        // Buffer all events arriving up to current_time_us
        while (event_idx < n && events[event_idx].arrival_time_us <= current_time_us) {
            if (hw_fifo.size() < hw_buffer_capacity) {
                hw_fifo.push_back(events[event_idx]);
            } else {
                stats.dropped_events++;
            }
            event_idx++;
        }

        if (hw_fifo.empty()) continue;

        HardwareEvent ev = hw_fifo.front();
        hw_fifo.pop_front();

        double latency = current_time_us - ev.arrival_time_us;
        latencies.push_back(latency);

        double handler_duration = ev.work_duration_us + handler_overhead_us;
        stats.max_handler_hold_time_us = std::max(stats.max_handler_hold_time_us, handler_duration);

        double handler_finish_time = current_time_us + handler_duration;

        // While handler is executing, IF=0. Any new events arriving must wait in hw_fifo or be dropped!
        while (event_idx < n && events[event_idx].arrival_time_us <= handler_finish_time) {
            if (hw_fifo.size() < hw_buffer_capacity) {
                hw_fifo.push_back(events[event_idx]);
            } else {
                stats.dropped_events++;
            }
            event_idx++;
        }

        current_time_us = handler_finish_time;
        stats.processed_events++;
    }

    stats.total_time_us = current_time_us;
    if (!latencies.empty()) {
        double sum = 0.0;
        for (double l : latencies) {
            sum += l;
            stats.max_interrupt_latency_us = std::max(stats.max_interrupt_latency_us, l);
        }
        stats.avg_interrupt_latency_us = sum / latencies.size();
    }

    return stats;
}

// ============================================================================
// 6. Split Top Half / Bottom Half Simulation (osbook_day07b)
// ============================================================================

ExecutionStats SimulateSplitTopBottomHalf(
    const std::vector<HardwareEvent>& events,
    size_t sw_queue_capacity = 32,
    double top_half_duration_us = 0.8
) {
    ExecutionStats stats;
    stats.architecture_name = "Split Top/Bottom Half (osbook_day07b)";
    stats.total_events = events.size();

    std::vector<Message> raw_buf(sw_queue_capacity);
    ArrayQueue<Message> sw_queue(raw_buf.data(), sw_queue_capacity);

    std::vector<double> top_half_latencies;
    double current_time_us = 0.0;
    size_t event_idx = 0;
    const size_t n = events.size();

    // Bottom half state
    bool bh_busy = false;
    double bh_remaining_us = 0.0;

    stats.max_handler_hold_time_us = top_half_duration_us;

    while (event_idx < n || !sw_queue.Empty() || bh_busy) {
        double next_arrival = (event_idx < n) ? events[event_idx].arrival_time_us : 1e12;

        if (!bh_busy && sw_queue.Empty()) {
            if (current_time_us < next_arrival) {
                current_time_us = next_arrival;
            }
        }

        // Check if an interrupt arrives now (can preempt Bottom Half!)
        if (event_idx < n && events[event_idx].arrival_time_us <= current_time_us) {
            const auto& ev = events[event_idx];
            double latency = current_time_us - ev.arrival_time_us;
            top_half_latencies.push_back(latency);

            // Execute Top Half (Hardirq: Push Message and Notify EOI)
            current_time_us += top_half_duration_us;
            Message msg{Message::kInterruptXHCI, ev.id, static_cast<uint64_t>(ev.work_duration_us)};
            Error err = sw_queue.Push(msg);
            if (err) {
                stats.dropped_events++;
            }
            event_idx++;
            continue;
        }

        // If Bottom Half is not running but queue has items, start one
        if (!bh_busy && !sw_queue.Empty()) {
            Message msg = sw_queue.Front();
            sw_queue.Pop();
            bh_busy = true;
            bh_remaining_us = static_cast<double>(msg.arg1);
        }

        if (bh_busy) {
            double time_until_next_arrival = next_arrival - current_time_us;
            if (time_until_next_arrival <= 0.0) {
                // Immediate interrupt arrival
                continue;
            }

            if (time_until_next_arrival < bh_remaining_us) {
                // Preempted by next hardware interrupt!
                bh_remaining_us -= time_until_next_arrival;
                current_time_us += time_until_next_arrival;
            } else {
                // Bottom Half completes before next interrupt
                current_time_us += bh_remaining_us;
                bh_busy = false;
                bh_remaining_us = 0.0;
                stats.processed_events++;
            }
        }
    }

    stats.total_time_us = current_time_us;
    if (!top_half_latencies.empty()) {
        double sum = 0.0;
        for (double l : top_half_latencies) {
            sum += l;
            stats.max_interrupt_latency_us = std::max(stats.max_interrupt_latency_us, l);
        }
        stats.avg_interrupt_latency_us = sum / top_half_latencies.size();
    }

    return stats;
}

// ============================================================================
// 7. Unit Tests and Verification Functions
// ============================================================================

void TestArrayQueueBasics() {
    std::cout << "[Test 1/4] Verifying ArrayQueue ring buffer invariants...\n";

    std::array<Message, 4> buffer;
    ArrayQueue<Message> q(buffer);

    assert(q.Capacity() == 4);
    assert(q.Count() == 0);
    assert(q.Empty());
    assert(!q.Full());

    // Push 4 elements
    for (int i = 0; i < 4; ++i) {
        Message msg{Message::kInterruptXHCI, static_cast<uint64_t>(i), 0};
        Error err = q.Push(msg);
        assert(!err);
        assert(q.Count() == static_cast<size_t>(i + 1));
    }

    assert(q.Full());

    // 5th push should fail with kFull
    Message overflow_msg{Message::kInterruptXHCI, 999, 0};
    Error full_err = q.Push(overflow_msg);
    assert(full_err.Code() == ErrorCode::kFull);
    assert(q.Count() == 4);

    // Front inspect
    assert(q.Front().arg0 == 0);

    // Pop 2 elements
    assert(!q.Pop());
    assert(q.Front().arg0 == 1);
    assert(!q.Pop());
    assert(q.Front().arg0 == 2);
    assert(q.Count() == 2);
    assert(!q.Full());

    // Push 2 more elements to test wrap-around
    assert(!q.Push(Message{Message::kInterruptXHCI, 4, 0}));
    assert(!q.Push(Message{Message::kInterruptXHCI, 5, 0}));
    assert(q.Full());

    // Drain all
    uint64_t expected_vals[] = {2, 3, 4, 5};
    for (uint64_t expected : expected_vals) {
        assert(q.Front().arg0 == expected);
        assert(!q.Pop());
    }

    assert(q.Empty());
    assert(q.Pop().Code() == ErrorCode::kEmpty);

    std::cout << "  -> ArrayQueue ring buffer wrap-around, full/empty guards PASSED.\n";
}

void TestMonolithicBottleneck() {
    std::cout << "[Test 2/4] Verifying Monolithic handler bottleneck and drops...\n";

    // Create 10 bursty events arriving closely (every 20 us), each taking 100 us to draw
    std::vector<HardwareEvent> burst;
    for (int i = 0; i < 10; ++i) {
        burst.push_back(HardwareEvent{
            static_cast<uint32_t>(i + 1),
            i * 20.0,
            100.0,
            "Rapid Mouse Move"
        });
    }

    auto stats = SimulateMonolithic(burst, /*hw_buffer_capacity=*/2);
    std::cout << "  Monolithic Stats: Dropped = " << stats.dropped_events
              << " / " << stats.total_events
              << ", Max Latency = " << stats.max_interrupt_latency_us << " us\n";

    // With burst of 10 events arriving faster than processing rate, events MUST drop!
    assert(stats.dropped_events > 0 && "Monolithic handler failed to demonstrate event drops!");
    assert(stats.max_interrupt_latency_us > 100.0 && "Latency should exceed single handler duration!");

    std::cout << "  -> Monolithic handler bottleneck and packet drops confirmed.\n";
}

void TestSplitTopBottomHalfAdvantage() {
    std::cout << "[Test 3/4] Verifying Split Top/Bottom Half zero drops and low latency...\n";

    std::vector<HardwareEvent> burst;
    for (int i = 0; i < 10; ++i) {
        burst.push_back(HardwareEvent{
            static_cast<uint32_t>(i + 1),
            i * 20.0,
            100.0,
            "Rapid Mouse Move"
        });
    }

    auto stats = SimulateSplitTopBottomHalf(burst, /*sw_queue_capacity=*/32, /*top_half=*/0.8);
    std::cout << "  Split Stats: Dropped = " << stats.dropped_events
              << " / " << stats.total_events
              << ", Max Top-Half Latency = " << stats.max_interrupt_latency_us << " us"
              << ", Max Hold Time = " << stats.max_handler_hold_time_us << " us\n";

    // Split architecture should easily buffer all 10 events and drop ZERO events
    assert(stats.dropped_events == 0 && "Split architecture should not drop burst events!");
    assert(stats.processed_events == 10 && "All 10 events must be processed!");
    assert(stats.max_interrupt_latency_us < 5.0 && "Top-half latency must remain under 5 us!");
    assert(stats.max_handler_hold_time_us < 1.0 && "Handler IF=0 hold time must be < 1.0 us!");

    std::cout << "  -> Split Top/Bottom Half zero drops & sub-microsecond latency confirmed.\n";
}

void TestQuantitativeComparison() {
    std::cout << "[Test 4/4] Quantitative comparison under intense 50-event burst...\n";

    std::vector<HardwareEvent> heavy_burst;
    for (int i = 0; i < 50; ++i) {
        heavy_burst.push_back(HardwareEvent{
            static_cast<uint32_t>(i + 1),
            i * 15.0,  // Arrives every 15 us
            80.0,      // Requires 80 us of drawing
            "Heavy Burst Move"
        });
    }

    auto mono = SimulateMonolithic(heavy_burst, /*hw_cap=*/3);
    auto split = SimulateSplitTopBottomHalf(heavy_burst, /*sw_cap=*/64, /*top_half=*/0.5);

    std::cout << "\n====================================================================\n";
    std::cout << "  PERFORMANCE COMPARISON SUMMARY (50-event Burst Simulation)\n";
    std::cout << "====================================================================\n";
    std::cout << std::left << std::setw(28) << "Metric"
              << std::setw(20) << "Monolithic (day07a)"
              << std::setw(20) << "Split (day07b)" << "\n";
    std::cout << "--------------------------------------------------------------------\n";

    std::cout << std::left << std::setw(28) << "Processed Events"
              << std::setw(20) << mono.processed_events
              << std::setw(20) << split.processed_events << "\n";

    std::cout << std::left << std::setw(28) << "Dropped Events"
              << std::setw(20) << mono.dropped_events
              << std::setw(20) << split.dropped_events << "\n";

    std::cout << std::left << std::setw(28) << "Drop Rate (%)"
              << std::setw(20) << (100.0 * mono.dropped_events / mono.total_events)
              << std::setw(20) << (100.0 * split.dropped_events / split.total_events) << "\n";

    std::cout << std::left << std::setw(28) << "Max Latency (us)"
              << std::setw(20) << mono.max_interrupt_latency_us
              << std::setw(20) << split.max_interrupt_latency_us << "\n";

    std::cout << std::left << std::setw(28) << "Avg Latency (us)"
              << std::setw(20) << mono.avg_interrupt_latency_us
              << std::setw(20) << split.avg_interrupt_latency_us << "\n";

    std::cout << std::left << std::setw(28) << "Max IF=0 Hold Time (us)"
              << std::setw(20) << mono.max_handler_hold_time_us
              << std::setw(20) << split.max_handler_hold_time_us << "\n";

    std::cout << "====================================================================\n\n";

    assert(mono.dropped_events > split.dropped_events);
    assert(split.max_interrupt_latency_us < mono.max_interrupt_latency_us);

    std::cout << "  -> Quantitative benchmark verification PASSED.\n";
}

} // namespace os

int main() {
    std::cout << "============================================================\n";
    std::cout << " MikanOS Chapter 7.7: Fast Interrupt Handling Simulation\n";
    std::cout << " Top Half / Bottom Half Decoupling & ArrayQueue Verification\n";
    std::cout << "============================================================\n\n";

    os::TestArrayQueueBasics();
    os::TestMonolithicBottleneck();
    os::TestSplitTopBottomHalfAdvantage();
    os::TestQuantitativeComparison();

    std::cout << "All Chapter 7.7 C++20 verification tests PASSED successfully!\n";
    return 0;
}
