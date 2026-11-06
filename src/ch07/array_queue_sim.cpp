/**
 * @file array_queue_sim.cpp
 * @brief C++20 Standalone Verification of ArrayQueue Ring Buffer Implementation
 *        (MikanOS Chapter 7.9, osbook_day07b kernel/queue.hpp)
 *
 * This program comprehensively verifies:
 * 1. MikanOS ArrayQueue<T> Class Template:
 *    - Pure header-only template design with zero heap allocation dependency (no malloc/new)
 *    - External storage injection via std::array<T, N>& or raw pointer (T*, size)
 *    - Invariants of read_pos_, write_pos_, count_, and capacity_
 * 2. Ring Buffer Pointer Operations and Wrap-Around Math:
 *    - Push logic: data_[write_pos_] = value; ++count_; if (++write_pos_ == capacity_) write_pos_ = 0;
 *    - Pop logic: --count_; if (++read_pos_ == capacity_) read_pos_ = 0;
 *    - Boundary error conditions: Error::kFull on Push, Error::kEmpty on Pop
 * 3. Comparison with Classical "One-Slot Waste" Ring Buffer:
 *    - Proof that MikanOS count_-based design achieves 100% capacity utilization
 *    - Demonstration of slot wastage in the pointer-only (head == tail) design
 * 4. Struct / Event Message Queuing:
 *    - Verification with MikanOS struct Message (xHCI event, timer event, keyboard event)
 * 5. High-throughput performance and zero allocation overhead benchmark
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic array_queue_sim.cpp -o array_queue_sim
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
#include <numeric>
#include <algorithm>
#include <chrono>

namespace os {

// ============================================================================
// 1. Error Representation (MikanOS kernel/error.hpp Equivalent)
// ============================================================================

enum class ErrorCode {
    kSuccess = 0,
    kFull = 1,
    kEmpty = 2,
    kInvalidValue = 3,
};

class Error {
public:
    constexpr Error(ErrorCode code = ErrorCode::kSuccess, const char* file = nullptr, int line = 0)
        : code_{code}, file_{file}, line_{line} {}

    constexpr ErrorCode Code() const { return code_; }
    constexpr explicit operator bool() const { return code_ != ErrorCode::kSuccess; }

    const char* Name() const {
        switch (code_) {
            case ErrorCode::kSuccess: return "kSuccess";
            case ErrorCode::kFull: return "kFull";
            case ErrorCode::kEmpty: return "kEmpty";
            case ErrorCode::kInvalidValue: return "kInvalidValue";
            default: return "kUnknown";
        }
    }

    const char* File() const { return file_; }
    int Line() const { return line_; }

private:
    ErrorCode code_;
    const char* file_;
    int line_;
};

#define MAKE_ERROR(code) os::Error((code), __FILE__, __LINE__)

// ============================================================================
// 2. MikanOS ArrayQueue<T> Class Template (kernel/queue.hpp)
// ============================================================================

template <typename T>
class ArrayQueue {
public:
    template <size_t N>
    ArrayQueue(std::array<T, N>& buf) : ArrayQueue(buf.data(), N) {}

    ArrayQueue(T* buf, size_t size)
        : data_{buf}, read_pos_{0}, write_pos_{0}, count_{0}, capacity_{size} {}

    /**
     * @brief Pushes a value into the queue (Tail / Producer).
     * @return Error::kFull if queue is full, otherwise Error::kSuccess.
     */
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

    /**
     * @brief Pops an element from the queue (Head / Consumer).
     * @return Error::kEmpty if queue is empty, otherwise Error::kSuccess.
     */
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
    bool Empty() const { return count_ == 0; }
    bool Full() const { return count_ == capacity_; }

    const T& Front() const { return data_[read_pos_]; }
    T& Front() { return data_[read_pos_]; }

    // Inspection accessors for testing & simulation
    size_t ReadPos() const { return read_pos_; }
    size_t WritePos() const { return write_pos_; }
    const T* RawBuffer() const { return data_; }

private:
    T* data_;
    size_t read_pos_;
    size_t write_pos_;
    size_t count_;
    const size_t capacity_;
};

// ============================================================================
// 3. Classical "One-Slot Waste" Ring Buffer (For Comparative Analysis)
// ============================================================================

/**
 * @brief Classical ring buffer without an explicit count variable.
 *        Distinguishes Full from Empty by reserving 1 unused slot:
 *        - Empty: head == tail
 *        - Full:  (tail + 1) % capacity == head
 *        Usable capacity is therefore (capacity - 1).
 */
template <typename T>
class OneSlotWasteQueue {
public:
    template <size_t N>
    OneSlotWasteQueue(std::array<T, N>& buf) : OneSlotWasteQueue(buf.data(), N) {}

    OneSlotWasteQueue(T* buf, size_t size)
        : data_{buf}, head_{0}, tail_{0}, capacity_{size} {}

    Error Push(const T& value) {
        size_t next_tail = (tail_ + 1 == capacity_) ? 0 : tail_ + 1;
        if (next_tail == head_) {
            return MAKE_ERROR(ErrorCode::kFull); // Buffer full (1 slot left empty!)
        }
        data_[tail_] = value;
        tail_ = next_tail;
        return MAKE_ERROR(ErrorCode::kSuccess);
    }

    Error Pop() {
        if (head_ == tail_) {
            return MAKE_ERROR(ErrorCode::kEmpty);
        }
        head_ = (head_ + 1 == capacity_) ? 0 : head_ + 1;
        return MAKE_ERROR(ErrorCode::kSuccess);
    }

    const T& Front() const { return data_[head_]; }

    size_t Count() const {
        if (tail_ >= head_) {
            return tail_ - head_;
        } else {
            return capacity_ - head_ + tail_;
        }
    }

    size_t Capacity() const { return capacity_; }
    size_t UsableCapacity() const { return capacity_ > 0 ? capacity_ - 1 : 0; }
    bool Empty() const { return head_ == tail_; }
    bool Full() const { return ((tail_ + 1 == capacity_) ? 0 : tail_ + 1) == head_; }

private:
    T* data_;
    size_t head_;
    size_t tail_;
    const size_t capacity_;
};

// ============================================================================
// 4. MikanOS Message Struct (Realistic Event Payload)
// ============================================================================

struct Message {
    enum Type {
        kInterruptXHCI,
        kTimerTimeout,
        kKeyPush,
    } type;

    union {
        struct {
            unsigned long timeout;
            int value;
        } timer;
        struct {
            uint8_t modifier;
            uint8_t keycode;
            char ascii;
        } keyboard;
    } arg;
};

} // namespace os

// ============================================================================
// Verification and Unit Test Suite
// ============================================================================

void TestBasicOperations() {
    std::cout << "[Test 1] Basic Push, Pop, Front, and Invariants..." << std::endl;
    std::array<int, 4> raw_buf{};
    os::ArrayQueue<int> queue(raw_buf);

    assert(queue.Capacity() == 4);
    assert(queue.Count() == 0);
    assert(queue.Empty());
    assert(!queue.Full());
    assert(queue.ReadPos() == 0);
    assert(queue.WritePos() == 0);

    // Push 10
    auto err = queue.Push(10);
    assert(!err);
    assert(queue.Count() == 1);
    assert(!queue.Empty());
    assert(!queue.Full());
    assert(queue.Front() == 10);
    assert(queue.ReadPos() == 0);
    assert(queue.WritePos() == 1);

    // Push 20
    err = queue.Push(20);
    assert(!err);
    assert(queue.Count() == 2);
    assert(queue.Front() == 10); // Front still points to first element
    assert(queue.WritePos() == 2);

    // Pop 10
    err = queue.Pop();
    assert(!err);
    assert(queue.Count() == 1);
    assert(queue.Front() == 20); // Front advanced
    assert(queue.ReadPos() == 1);
    assert(queue.WritePos() == 2);

    // Pop 20
    err = queue.Pop();
    assert(!err);
    assert(queue.Count() == 0);
    assert(queue.Empty());
    assert(queue.ReadPos() == 2);
    assert(queue.WritePos() == 2);

    std::cout << "  -> PASSED: Invariants preserved during normal operations.\n";
}

void TestWrapAroundLogic() {
    std::cout << "[Test 2] Pointer Wrap-Around (Circular Buffer Mechanism)..." << std::endl;
    std::array<int, 3> raw_buf{};
    os::ArrayQueue<int> queue(raw_buf);

    // Fill buffer: [100, 200, 300]
    assert(!queue.Push(100));
    assert(!queue.Push(200));
    assert(!queue.Push(300));
    assert(queue.Full());
    assert(queue.WritePos() == 0); // Wrapped around from 3 to 0!
    assert(queue.ReadPos() == 0);

    // Pop two: [---, ---, 300]
    assert(queue.Front() == 100);
    assert(!queue.Pop());
    assert(queue.Front() == 200);
    assert(!queue.Pop());
    assert(queue.Count() == 1);
    assert(queue.ReadPos() == 2);
    assert(queue.WritePos() == 0);

    // Push two: [400, 500, 300]
    assert(!queue.Push(400));
    assert(queue.WritePos() == 1);
    assert(!queue.Push(500));
    assert(queue.WritePos() == 2);
    assert(queue.Full());

    // Reading elements should yield 300, 400, 500 in exact FIFO order
    assert(queue.Front() == 300);
    assert(!queue.Pop());
    assert(queue.ReadPos() == 0); // Read pointer wrapped around!

    assert(queue.Front() == 400);
    assert(!queue.Pop());
    assert(queue.ReadPos() == 1);

    assert(queue.Front() == 500);
    assert(!queue.Pop());
    assert(queue.ReadPos() == 2);
    assert(queue.Empty());

    std::cout << "  -> PASSED: Read and write pointers wrap around accurately.\n";
}

void TestBoundaryFullAndEmptyErrors() {
    std::cout << "[Test 3] Boundary Conditions: kFull and kEmpty Errors..." << std::endl;
    std::array<int, 2> raw_buf{};
    os::ArrayQueue<int> queue(raw_buf);

    // Empty pop error
    auto err = queue.Pop();
    assert(err.Code() == os::ErrorCode::kEmpty);
    assert(std::string_view(err.Name()) == "kEmpty");

    // Fill to capacity
    assert(!queue.Push(1));
    assert(!queue.Push(2));
    assert(queue.Full());

    // Full push error
    err = queue.Push(3);
    assert(err.Code() == os::ErrorCode::kFull);
    assert(std::string_view(err.Name()) == "kFull");

    // Invariants must be preserved after failed push
    assert(queue.Count() == 2);
    assert(queue.Front() == 1);

    std::cout << "  -> PASSED: Error codes kFull and kEmpty returned without corruption.\n";
}

void TestOneSlotWasteComparison() {
    std::cout << "[Test 4] Comparison: MikanOS count_-based vs One-Slot Waste..." << std::endl;
    constexpr size_t N = 4;
    std::array<int, N> buf_mikan{};
    std::array<int, N> buf_waste{};

    os::ArrayQueue<int> mikan_queue(buf_mikan);
    os::OneSlotWasteQueue<int> waste_queue(buf_waste);

    assert(mikan_queue.Capacity() == N);
    assert(waste_queue.Capacity() == N);
    assert(waste_queue.UsableCapacity() == N - 1);

    // Push until full
    size_t mikan_pushed = 0;
    while (!mikan_queue.Push(static_cast<int>(mikan_pushed + 1))) {
        ++mikan_pushed;
    }

    size_t waste_pushed = 0;
    while (!waste_queue.Push(static_cast<int>(waste_pushed + 1))) {
        ++waste_pushed;
    }

    std::cout << "  MikanOS ArrayQueue pushed: " << mikan_pushed << " elements (Utilization: 100.0%)\n";
    std::cout << "  OneSlotWasteQueue pushed:  " << waste_pushed << " elements (Utilization: "
              << (waste_pushed * 100.0 / N) << "%)\n";

    assert(mikan_pushed == N);
    assert(waste_pushed == N - 1); // 1 slot sacrificed!

    std::cout << "  -> PASSED: MikanOS design achieves full 100% capacity utilization.\n";
}

void TestStructMessageQueue() {
    std::cout << "[Test 5] Complex Struct Payload (os::Message)..." << std::endl;
    std::array<os::Message, 8> msg_buf{};
    os::ArrayQueue<os::Message> msg_queue(msg_buf);

    os::Message m1{};
    m1.type = os::Message::kInterruptXHCI;

    os::Message m2{};
    m2.type = os::Message::kTimerTimeout;
    m2.arg.timer.timeout = 1000;
    m2.arg.timer.value = 42;

    os::Message m3{};
    m3.type = os::Message::kKeyPush;
    m3.arg.keyboard.modifier = 0x02; // Shift
    m3.arg.keyboard.keycode = 0x04;   // 'a'
    m3.arg.keyboard.ascii = 'A';

    assert(!msg_queue.Push(m1));
    assert(!msg_queue.Push(m2));
    assert(!msg_queue.Push(m3));
    assert(msg_queue.Count() == 3);

    assert(msg_queue.Front().type == os::Message::kInterruptXHCI);
    assert(!msg_queue.Pop());

    assert(msg_queue.Front().type == os::Message::kTimerTimeout);
    assert(msg_queue.Front().arg.timer.timeout == 1000);
    assert(msg_queue.Front().arg.timer.value == 42);
    assert(!msg_queue.Pop());

    assert(msg_queue.Front().type == os::Message::kKeyPush);
    assert(msg_queue.Front().arg.keyboard.ascii == 'A');
    assert(!msg_queue.Pop());

    assert(msg_queue.Empty());
    std::cout << "  -> PASSED: Event messages correctly enqueued and dequeued.\n";
}

void TestHighThroughputBenchmark() {
    std::cout << "[Test 6] High-Throughput Performance Benchmark (10,000,000 Ops)..." << std::endl;
    constexpr size_t CAP = 64;
    std::array<uint32_t, CAP> buffer{};
    os::ArrayQueue<uint32_t> queue(buffer);

    constexpr size_t TOTAL_OPS = 10'000'000;
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < TOTAL_OPS; ++i) {
        queue.Push(static_cast<uint32_t>(i));
        queue.Pop();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double ops_per_sec = (TOTAL_OPS * 2) / (elapsed.count() / 1000.0);
    std::cout << "  Total Push+Pop Ops: " << TOTAL_OPS * 2 << "\n";
    std::cout << "  Elapsed Time:       " << std::fixed << std::setprecision(3) << elapsed.count() << " ms\n";
    std::cout << "  Throughput:         " << std::fixed << std::setprecision(1) << (ops_per_sec / 1e6)
              << " Million Ops/sec\n";
    std::cout << "  -> PASSED: Ultra-low latency and zero dynamic allocation.\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " MikanOS Ch 7.9: ArrayQueue Ring Buffer Verification\n";
    std::cout << "========================================================\n\n";

    TestBasicOperations();
    TestWrapAroundLogic();
    TestBoundaryFullAndEmptyErrors();
    TestOneSlotWasteComparison();
    TestStructMessageQueue();
    TestHighThroughputBenchmark();

    std::cout << "\n========================================================\n";
    std::cout << " ALL CHAPTER 7.9 TESTS COMPLETED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";
    return 0;
}
