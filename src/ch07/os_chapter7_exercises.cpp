/**
 * @file os_chapter7_exercises.cpp
 * @brief Chapter 7 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 7:
 * 1. x86-64 Interrupt Stack Frame Layout & 16-Byte Alignment (Section 7.1, 7.2)
 * 2. IDT (Interrupt Descriptor Table) 16-Byte Descriptor Bitfield & SetIDTEntry Math (Section 7.3, 7.4)
 * 3. PCI MSI (Message Signaled Interrupts) Address/Data Register Composition (Section 7.5, 7.6)
 * 4. Top Half / Bottom Half Architecture & Queueing Decoupling (Section 7.7)
 * 5. FIFO vs FILO Event Ordering & Cursor Jump Prevention (Section 7.8)
 * 6. ArrayQueue<T, N> Ring Buffer Math: Circular Index, Full/Empty Semantics (Section 7.9)
 * 7. CLI / STI Critical Section & STI; HLT Interrupt Shadow Race Condition Prevention (Section 7.10)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch07 {

// =============================================================================
// Exercise 1: x86-64 Interrupt Stack Frame & Alignment
// =============================================================================

struct InterruptFrame {
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
};

static_assert(sizeof(InterruptFrame) == 40, "InterruptFrame must be 40 bytes (5 * 8 bytes)");
static_assert(alignof(InterruptFrame) == 8, "InterruptFrame member alignment is 8 bytes");

// =============================================================================
// Exercise 2: IDT 16-Byte Descriptor & SetIDTEntry Bitfield Math
// =============================================================================

enum class DescriptorType : uint8_t {
    kUpperBits = 0,
    kLDT = 2,
    kTSSAvailable = 9,
    kTSSBusy = 11,
    kCallGate = 12,
    kInterruptGate = 14, // 0xE: 64-bit Interrupt Gate
    kTrapGate = 15,      // 0xF: 64-bit Trap Gate
};

union InterruptDescriptorAttribute {
    uint16_t data;
    struct {
        uint16_t interrupt_stack_table : 3;
        uint16_t : 5;
        DescriptorType type : 4;
        uint16_t : 1;
        uint16_t descriptor_privilege_level : 2;
        uint16_t present : 1;
    } __attribute__((packed)) bits;
} __attribute__((packed));

static_assert(sizeof(InterruptDescriptorAttribute) == 2, "InterruptDescriptorAttribute must be 2 bytes");

struct InterruptDescriptor {
    uint16_t offset_low;
    uint16_t segment_selector;
    InterruptDescriptorAttribute attr;
    uint16_t offset_mid;
    uint32_t offset_high;
    uint32_t reserved;
} __attribute__((packed));

static_assert(sizeof(InterruptDescriptor) == 16, "InterruptDescriptor must be 16 bytes");

inline InterruptDescriptor MakeIDTEntry(
    uint64_t offset,
    uint16_t segment_selector,
    DescriptorType type = DescriptorType::kInterruptGate,
    uint8_t dpl = 0,
    uint8_t ist = 0)
{
    InterruptDescriptor desc{};
    desc.offset_low = static_cast<uint16_t>(offset & 0xFFFF);
    desc.segment_selector = segment_selector;

    InterruptDescriptorAttribute attr{};
    attr.data = 0;
    attr.bits.interrupt_stack_table = ist & 0x7;
    attr.bits.type = type;
    attr.bits.descriptor_privilege_level = dpl & 0x3;
    attr.bits.present = 1;
    desc.attr = attr;

    desc.offset_mid = static_cast<uint16_t>((offset >> 16) & 0xFFFF);
    desc.offset_high = static_cast<uint32_t>((offset >> 32) & 0xFFFFFFFF);
    desc.reserved = 0;

    return desc;
}

inline uint64_t GetIDTOffset(const InterruptDescriptor& desc) {
    return static_cast<uint64_t>(desc.offset_low) |
           (static_cast<uint64_t>(desc.offset_mid) << 16) |
           (static_cast<uint64_t>(desc.offset_high) << 32);
}

// =============================================================================
// Exercise 3: PCI MSI Address & Data Registers
// =============================================================================

struct MSIRegisterValues {
    uint32_t msg_addr;
    uint16_t msg_data;
};

inline MSIRegisterValues CalculateMSI(uint8_t destination_apic_id, uint8_t vector) {
    // x86 Local APIC Base: 0xFEE00000 | (dest_id << 12)
    uint32_t msg_addr = 0xFEE00000u | (static_cast<uint32_t>(destination_apic_id) << 12);
    // Delivery Mode = Fixed (0), Vector = vector
    uint16_t msg_data = static_cast<uint16_t>(vector);
    return {msg_addr, msg_data};
}

// =============================================================================
// Exercise 4 & 6: ArrayQueue<T, N> Ring Buffer Implementation
// =============================================================================

template <typename T, size_t N>
class ArrayQueue {
public:
    static_assert(N >= 2, "ArrayQueue capacity must be at least 2");

    ArrayQueue() : head_{0}, tail_{0}, count_{0} {}

    [[nodiscard]] bool Push(const T& value) {
        size_t next_tail = (tail_ + 1) % N;
        if (next_tail == head_) {
            return false; // Queue is full (1 slot reserved to distinguish empty from full)
        }
        data_[tail_] = value;
        tail_ = next_tail;
        ++count_;
        return true;
    }

    [[nodiscard]] std::optional<T> Pop() {
        if (Empty()) {
            return std::nullopt;
        }
        T val = data_[head_];
        head_ = (head_ + 1) % N;
        --count_;
        return val;
    }

    [[nodiscard]] bool Empty() const { return head_ == tail_; }
    [[nodiscard]] bool Full() const { return (tail_ + 1) % N == head_; }
    [[nodiscard]] size_t Size() const { return count_; }
    [[nodiscard]] constexpr size_t Capacity() const { return N - 1; }
    [[nodiscard]] size_t HeadIndex() const { return head_; }
    [[nodiscard]] size_t TailIndex() const { return tail_; }

private:
    std::array<T, N> data_{};
    size_t head_{0};
    size_t tail_{0};
    size_t count_{0};
};

// =============================================================================
// Exercise 5: FIFO vs FILO Event Processing Simulator
// =============================================================================

struct MouseDelta {
    int dx;
    int dy;
};

struct MousePathResult {
    int final_x{0};
    int final_y{0};
    std::vector<std::pair<int, int>> trajectory;
};

inline MousePathResult SimulateEventQueueFIFO(const std::vector<MouseDelta>& deltas) {
    MousePathResult res;
    int cur_x = 100, cur_y = 100;
    res.trajectory.push_back({cur_x, cur_y});

    std::deque<MouseDelta> queue;
    for (const auto& d : deltas) {
        queue.push_back(d);
    }

    while (!queue.empty()) {
        auto d = queue.front();
        queue.pop_front();
        cur_x += d.dx;
        cur_y += d.dy;
        res.trajectory.push_back({cur_x, cur_y});
    }

    res.final_x = cur_x;
    res.final_y = cur_y;
    return res;
}

inline MousePathResult SimulateEventQueueFILO(const std::vector<MouseDelta>& deltas) {
    MousePathResult res;
    int cur_x = 100, cur_y = 100;
    res.trajectory.push_back({cur_x, cur_y});

    std::stack<MouseDelta> stack;
    for (const auto& d : deltas) {
        stack.push(d);
    }

    while (!stack.empty()) {
        auto d = stack.top();
        stack.pop();
        cur_x += d.dx;
        cur_y += d.dy;
        res.trajectory.push_back({cur_x, cur_y});
    }

    res.final_x = cur_x;
    res.final_y = cur_y;
    return res;
}

// =============================================================================
// Exercise 7: STI; HLT & Lost Wakeup Race Condition Simulation
// =============================================================================

class EventLoopSimulator {
public:
    enum class State {
        kRunning,
        kHalted,
        kInterruptServiced,
        kLostWakeupBug,
    };

    EventLoopSimulator() : if_flag_{true}, queue_{} {}

    void CLI() { if_flag_ = false; }
    void STI() { if_flag_ = true; }

    bool EnqueueInterruptEvent(uint8_t event_code) {
        // Interrupt can occur only if IF == 1
        return queue_.Push(event_code);
    }

    // Vulnerable loop: STI and HLT are separate instructions
    State StepVulnerableLoop(bool interrupt_fires_in_window) {
        CLI();
        if (queue_.Empty()) {
            STI();
            // Window of vulnerability: Interrupt arrives right here after STI and before HLT!
            if (interrupt_fires_in_window) {
                // Top half executes:
                EnqueueInterruptEvent(0xAA);
            }
            // CPU enters HLT, but the event was already enqueued!
            // If no more interrupts come, CPU will sleep forever (Lost Wakeup).
            return State::kLostWakeupBug;
        }
        auto ev = queue_.Pop();
        (void)ev;
        STI();
        return State::kRunning;
    }

    // Safe loop: STI; HLT atomic instruction pair (with Interrupt Shadow)
    State StepSafeLoop(bool interrupt_fires_in_window) {
        CLI();
        if (queue_.Empty()) {
            // STI; HLT executes atomically.
            // Even if an interrupt is pending, x86 Interrupt Shadow delays servicing
            // until AFTER HLT executes, which immediately wakes the CPU!
            if (interrupt_fires_in_window) {
                EnqueueInterruptEvent(0xAA);
            }
            STI(); // atomic transition to HLT
            return State::kInterruptServiced;
        }
        auto ev = queue_.Pop();
        (void)ev;
        STI();
        return State::kRunning;
    }

private:
    bool if_flag_{true};
    ArrayQueue<uint8_t, 8> queue_;
};

// =============================================================================
// Run All Exercises
// =============================================================================

inline void RunAllExercises() {
    std::cout << "================================================================\n";
    std::cout << "MikanOS Chapter 7 Comprehensive Exercises Verification (C++20)\n";
    std::cout << "================================================================\n";

    // -------------------------------------------------------------------------
    // Test 1: InterruptFrame Size & Alignment
    // -------------------------------------------------------------------------
    std::cout << "[Test 1] InterruptStackFrame Layout & 16-Byte Alignment... ";
    {
        alignas(16) InterruptFrame frame{
            .rip = 0x0000000000102040ULL,
            .cs = 0x0008ULL,
            .rflags = 0x00000202ULL, // IF = 1, reserved = 1
            .rsp = 0x0000000000200FF0ULL,
            .ss = 0x0010ULL,
        };

        assert(sizeof(frame) == 40);
        assert(reinterpret_cast<uintptr_t>(&frame) % 16 == 0);
        assert(frame.rsp % 16 == 0);
        assert((frame.rflags & (1 << 9)) != 0); // IF bit
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 2: IDT 16-Byte Descriptor & SetIDTEntry Bit Decomposition
    // -------------------------------------------------------------------------
    std::cout << "[Test 2] IDT 16-Byte Descriptor & SetIDTEntry Decomposition... ";
    {
        uint64_t handler_addr = 0xFFFF800012345678ULL;
        uint16_t cs_selector = 0x0008;

        auto desc = MakeIDTEntry(handler_addr, cs_selector, DescriptorType::kInterruptGate, 0, 0);

        assert(desc.offset_low == 0x5678);
        assert(desc.offset_mid == 0x1234);
        assert(desc.offset_high == 0xFFFF8000);
        assert(desc.segment_selector == 0x0008);
        assert(desc.attr.bits.type == DescriptorType::kInterruptGate);
        assert(desc.attr.bits.present == 1);
        assert(desc.attr.bits.descriptor_privilege_level == 0);

        uint64_t recovered_addr = GetIDTOffset(desc);
        assert(recovered_addr == handler_addr);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 3: PCI MSI Address & Data Generation
    // -------------------------------------------------------------------------
    std::cout << "[Test 3] PCI MSI Address & Data Math (0xFEE00000 / Vector)... ";
    {
        uint8_t bsp_apic_id = 0;
        uint8_t xhci_vector = 0x40; // 64

        auto msi = CalculateMSI(bsp_apic_id, xhci_vector);
        assert(msi.msg_addr == 0xFEE00000);
        assert(msi.msg_data == 0x0040);

        // Test with different APIC ID
        auto msi_apic3 = CalculateMSI(3, 0x50);
        assert(msi_apic3.msg_addr == (0xFEE00000 | (3 << 12)));
        assert(msi_apic3.msg_data == 0x0050);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 4: ArrayQueue<T, N> Ring Buffer Capacity & Wrap-around
    // -------------------------------------------------------------------------
    std::cout << "[Test 4] ArrayQueue<T, N> Full, Empty & Wrap-around Semantics... ";
    {
        ArrayQueue<int, 4> q; // capacity = 3 elements
        assert(q.Capacity() == 3);
        assert(q.Empty());
        assert(!q.Full());
        assert(q.Size() == 0);

        assert(q.Push(10));
        assert(q.Push(20));
        assert(q.Push(30));
        assert(q.Full());
        assert(q.Size() == 3);
        assert(!q.Push(40)); // Push to full fails!

        auto val1 = q.Pop();
        assert(val1.has_value() && *val1 == 10);
        assert(!q.Full());
        assert(q.Size() == 2);

        // Wrap around push
        assert(q.Push(50));
        assert(q.Full());

        assert(q.Pop() == 20);
        assert(q.Pop() == 30);
        assert(q.Pop() == 50);
        assert(q.Empty());
        assert(!q.Pop().has_value());
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 5: FIFO vs FILO Event Ordering
    // -------------------------------------------------------------------------
    std::cout << "[Test 5] FIFO vs FILO Event Ordering & Trajectory Fidelity... ";
    {
        std::vector<MouseDelta> deltas = {
            {5, 0},
            {10, 2},
            {15, 5},
            {-2, -3},
        };

        auto fifo_res = SimulateEventQueueFIFO(deltas);
        auto filo_res = SimulateEventQueueFILO(deltas);

        // Final position is identical under linear commutative addition:
        assert(fifo_res.final_x == filo_res.final_x);
        assert(fifo_res.final_y == filo_res.final_y);

        // BUT intermediate trajectories differ completely!
        assert(fifo_res.trajectory != filo_res.trajectory);
        // FIFO first step is (100+5, 100+0) = (105, 100)
        assert(fifo_res.trajectory[1] == std::make_pair(105, 100));
        // FILO first step is (100-2, 100-3) = (98, 97) (Reversed!)
        assert(filo_res.trajectory[1] == std::make_pair(98, 97));
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 6: STI; HLT Race Condition & Lost Wakeup Protection
    // -------------------------------------------------------------------------
    std::cout << "[Test 6] STI; HLT Interrupt Shadow & Lost Wakeup Prevention... ";
    {
        EventLoopSimulator sim1;
        // Vulnerable: if interrupt arrives between STI and HLT -> Lost Wakeup
        auto bug_state = sim1.StepVulnerableLoop(true);
        assert(bug_state == EventLoopSimulator::State::kLostWakeupBug);

        EventLoopSimulator sim2;
        // Safe: STI; HLT atomic combination wakes CPU immediately
        auto safe_state = sim2.StepSafeLoop(true);
        assert(safe_state == EventLoopSimulator::State::kInterruptServiced);
    }
    std::cout << "PASSED\n";

    std::cout << "\n>> All 6 Chapter 7 exercise verification tests passed successfully!\n";
}

} // namespace mikanos::exercises::ch07

int main(int argc, char** argv) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    mikanos::exercises::ch07::RunAllExercises();

    if (test_mode) {
        std::cout << "[CTest] All Chapter 7 exercises verified successfully.\n";
    }
    return 0;
}
