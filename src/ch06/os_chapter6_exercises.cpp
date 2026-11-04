/**
 * @file os_chapter6_exercises.cpp
 * @brief Chapter 6 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 6:
 * 1. Vector2D Math & Mouse Cursor Sprite Rendering / Erasing Algorithm (Section 6.1)
 * 2. USB 3.0 / xHCI Architecture, TRB Format & Cycle Bit Toggle Mechanism (Section 6.2)
 * 3. PCI Configuration Space Address Derivation (MakeAddress) & Class Code Filter (Section 6.3)
 * 4. PCI BAR0 32-bit/64-bit Address Decoding & Sign Extension Safety (Section 6.4 / Column 6.2)
 * 5. Kernel Logger Log-Level Filtering Mechanism (Section 6.4 / Column 6.1)
 * 6. xHCI Polling Event Loop & Observer Callback Pipeline (Section 6.4)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch06 {

// =============================================================================
// Exercise 1: Vector2D Math & Mouse Cursor Sprite Rendering / Erasing
// =============================================================================

template <typename T>
struct Vector2D {
    T x{0};
    T y{0};

    constexpr Vector2D() = default;
    constexpr Vector2D(T x_, T y_) : x{x_}, y{y_} {}

    constexpr Vector2D operator+(const Vector2D& rhs) const {
        return Vector2D{x + rhs.x, y + rhs.y};
    }

    constexpr Vector2D operator-(const Vector2D& rhs) const {
        return Vector2D{x - rhs.x, y - rhs.y};
    }

    constexpr Vector2D& operator+=(const Vector2D& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    constexpr bool operator==(const Vector2D& rhs) const = default;
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    constexpr bool operator==(const PixelColor& rhs) const = default;
};

inline constexpr PixelColor kDesktopBgColor{45, 118, 180};
inline constexpr PixelColor kCursorBlack{0, 0, 0};
inline constexpr PixelColor kCursorWhite{255, 255, 255};

inline constexpr int kMouseCursorWidth = 15;
inline constexpr int kMouseCursorHeight = 24;

inline constexpr const char* const kMouseCursorShape[kMouseCursorHeight] = {
    "@              ",
    "@@             ",
    "@.@            ",
    "@..@           ",
    "@...@          ",
    "@....@         ",
    "@.....@        ",
    "@......@       ",
    "@.......@      ",
    "@........@     ",
    "@.........@    ",
    "@..........@   ",
    "@...........@  ",
    "@............@ ",
    "@......@@@@@@@@",
    "@......@       ",
    "@....@..@      ",
    "@...@ @..@     ",
    "@..@   @..@    ",
    "@.@     @..@   ",
    "@@       @..@  ",
    "@         @@   ",
    "               ",
    "               "
};

class MockFrameBuffer {
public:
    MockFrameBuffer(int width, int height, PixelColor fill_color = kDesktopBgColor)
        : width_(width), height_(height), buffer_(width * height, fill_color) {}

    void WritePixel(int x, int y, PixelColor color) {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            buffer_[y * width_ + x] = color;
        }
    }

    PixelColor GetPixel(int x, int y) const {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            return buffer_[y * width_ + x];
        }
        return PixelColor{0, 0, 0};
    }

    int Width() const { return width_; }
    int Height() const { return height_; }

private:
    int width_;
    int height_;
    std::vector<PixelColor> buffer_;
};

class SimulatedMouseCursor {
public:
    SimulatedMouseCursor(MockFrameBuffer& fb, Vector2D<int> initial_pos, PixelColor erase_color = kDesktopBgColor)
        : fb_(fb), position_(initial_pos), erase_color_(erase_color) {
        DrawCursor(position_);
    }

    void MoveRelative(Vector2D<int> disp) {
        EraseCursor(position_);
        position_ += disp;
        // Clamp to screen boundaries
        position_.x = std::clamp(position_.x, 0, fb_.Width() - 1);
        position_.y = std::clamp(position_.y, 0, fb_.Height() - 1);
        DrawCursor(position_);
    }

    Vector2D<int> GetPosition() const { return position_; }

private:
    void EraseCursor(Vector2D<int> pos) {
        for (int dy = 0; dy < kMouseCursorHeight; ++dy) {
            for (int dx = 0; dx < kMouseCursorWidth; ++dx) {
                if (kMouseCursorShape[dy][dx] != ' ') {
                    fb_.WritePixel(pos.x + dx, pos.y + dy, erase_color_);
                }
            }
        }
    }

    void DrawCursor(Vector2D<int> pos) {
        for (int dy = 0; dy < kMouseCursorHeight; ++dy) {
            for (int dx = 0; dx < kMouseCursorWidth; ++dx) {
                char pixel = kMouseCursorShape[dy][dx];
                if (pixel == '@') {
                    fb_.WritePixel(pos.x + dx, pos.y + dy, kCursorBlack);
                } else if (pixel == '.') {
                    fb_.WritePixel(pos.x + dx, pos.y + dy, kCursorWhite);
                }
            }
        }
    }

    MockFrameBuffer& fb_;
    Vector2D<int> position_;
    PixelColor erase_color_;
};

// =============================================================================
// Exercise 2: USB 3.0 / xHCI Architecture, TRB & Cycle Bit Logic
// =============================================================================

struct alignas(16) SimulatedTRB {
    uint64_t parameter{0};
    uint32_t status{0};
    uint32_t control{0};

    // TRB Type extraction (bits 15:10 of control field)
    uint32_t Type() const {
        return (control >> 10) & 0x3f;
    }

    // Cycle bit extraction (bit 0 of control field)
    bool CycleBit() const {
        return (control & 1u) != 0;
    }

    // Set Cycle Bit
    void SetCycleBit(bool c) {
        if (c) {
            control |= 1u;
        } else {
            control &= ~1u;
        }
    }
};
static_assert(sizeof(SimulatedTRB) == 16, "TRB must be exactly 16 bytes");

class SimulatedRingBuffer {
public:
    explicit SimulatedRingBuffer(size_t capacity)
        : ring_(capacity), cycle_bit_(true), producer_idx_(0), consumer_idx_(0) {}

    // Producer enqueue (OS software producing TRB)
    bool Enqueue(uint64_t param, uint32_t status, uint32_t control_without_cycle) {
        size_t next_idx = (producer_idx_ + 1) % ring_.size();
        if (next_idx == consumer_idx_) {
            return false; // Ring full
        }

        SimulatedTRB& trb = ring_[producer_idx_];
        trb.parameter = param;
        trb.status = status;
        trb.control = control_without_cycle;
        trb.SetCycleBit(cycle_bit_);

        producer_idx_ = next_idx;
        if (producer_idx_ == 0) {
            // Producer wrapped around: invert cycle bit
            cycle_bit_ = !cycle_bit_;
        }
        return true;
    }

    // Consumer dequeue (xHCI host controller consuming TRB)
    std::optional<SimulatedTRB> Dequeue(bool current_consumer_cycle) {
        if (consumer_idx_ == producer_idx_) {
            return std::nullopt; // Empty
        }

        const SimulatedTRB& trb = ring_[consumer_idx_];
        // xHCI hardware checks if TRB Cycle bit matches Consumer Cycle State (CCS)
        if (trb.CycleBit() != current_consumer_cycle) {
            return std::nullopt; // Not yet ready
        }

        SimulatedTRB result = trb;
        consumer_idx_ = (consumer_idx_ + 1) % ring_.size();
        return result;
    }

    bool ProducerCycleBit() const { return cycle_bit_; }
    size_t ProducerIndex() const { return producer_idx_; }
    size_t ConsumerIndex() const { return consumer_idx_; }

private:
    std::vector<SimulatedTRB> ring_;
    bool cycle_bit_;
    size_t producer_idx_;
    size_t consumer_idx_;
};

// =============================================================================
// Exercise 3: PCI Configuration Space (MakeAddress) & Class Code Filter
// =============================================================================

inline constexpr uint16_t kConfigAddress = 0x0cf8;
inline constexpr uint16_t kConfigData    = 0x0cfc;

constexpr uint32_t MakeAddress(uint8_t bus, uint8_t device, uint8_t function, uint8_t reg_addr) {
    auto shl = [](uint32_t x, unsigned int bits) {
        return x << bits;
    };

    return shl(1, 31) // Enable bit (Bit 31 = 1)
        | shl(bus, 16)
        | shl(device, 11)
        | shl(function, 8)
        | (reg_addr & 0xfc); // Register offset must be 4-byte aligned (bits 1:0 = 0)
}

struct ClassCode {
    uint8_t base;
    uint8_t sub;
    uint8_t interface;

    constexpr bool Match(uint8_t b) const { return base == b; }
    constexpr bool Match(uint8_t b, uint8_t s) const { return Match(b) && sub == s; }
    constexpr bool Match(uint8_t b, uint8_t s, uint8_t i) const { return Match(b, s) && interface == i; }
};

// USB 3.0 xHCI controller class code: Class=0x0C, SubClass=0x03, Interface=0x30
inline constexpr ClassCode kClassCodeXHCI{0x0C, 0x03, 0x30};

// =============================================================================
// Exercise 4: PCI BAR0 32-bit/64-bit Address Decoding & Sign Extension Safety
// =============================================================================

struct BAR0Decoder {
    static constexpr uint64_t DecodeAddress(uint32_t bar_lower, uint32_t bar_upper = 0) {
        // Bit 0: 0 = Memory Space, 1 = I/O Space
        bool is_io = (bar_lower & 1u) != 0;
        if (is_io) {
            return bar_lower & ~0x3u;
        }

        // Bits 2:1: Memory Type (00b = 32-bit, 10b = 64-bit)
        uint8_t mem_type = (bar_lower >> 1) & 0x3u;
        if (mem_type == 0x02) { // 64-bit address space
            uint64_t addr = (static_cast<uint64_t>(bar_upper) << 32) | bar_lower;
            // Prevent sign extension pitfall: ~static_cast<uint64_t>(0xf)
            return addr & ~static_cast<uint64_t>(0xf);
        } else { // 32-bit address space
            return bar_lower & ~0xfu;
        }
    }

    static constexpr bool Is64Bit(uint32_t bar_lower) {
        return ((bar_lower >> 1) & 0x3u) == 0x02;
    }
};

// =============================================================================
// Exercise 5: Kernel Logger Log-Level Filtering Mechanism
// =============================================================================

enum class LogLevel {
    kError = 3,
    kWarn = 4,
    kInfo = 5,
    kDebug = 6,
};

class SimulatedLogger {
public:
    explicit SimulatedLogger(LogLevel threshold = LogLevel::kWarn)
        : threshold_(threshold) {}

    void SetLogLevel(LogLevel level) {
        threshold_ = level;
    }

    LogLevel GetLogLevel() const {
        return threshold_;
    }

    bool ShouldLog(LogLevel level) const {
        return static_cast<int>(level) <= static_cast<int>(threshold_);
    }

    bool Log(LogLevel level, std::string_view msg) {
        if (ShouldLog(level)) {
            logs_.push_back(std::string(msg));
            return true;
        }
        return false;
    }

    const std::vector<std::string>& Logs() const { return logs_; }
    void Clear() { logs_.clear(); }

private:
    LogLevel threshold_;
    std::vector<std::string> logs_;
};

// =============================================================================
// Exercise 6: xHCI Polling Event Loop & Observer Callback Pipeline
// =============================================================================

struct MousePacket {
    int8_t disp_x{0};
    int8_t disp_y{0};
    uint8_t buttons{0};
};

using MouseObserverFn = void(*)(int8_t dx, int8_t dy);

class SimulatedXHCIPollingController {
public:
    void RegisterMouseObserver(MouseObserverFn observer) {
        observer_ = observer;
    }

    void PushPacket(MousePacket packet) {
        packet_queue_.push_back(packet);
    }

    // ProcessEvent simulation: polls event ring and delivers to observer
    bool ProcessEvent() {
        if (packet_queue_.empty()) {
            return false;
        }
        MousePacket pkt = packet_queue_.front();
        packet_queue_.erase(packet_queue_.begin());

        if (observer_) {
            observer_(pkt.disp_x, pkt.disp_y);
        }
        return true;
    }

    size_t PendingEvents() const { return packet_queue_.size(); }

private:
    MouseObserverFn observer_{nullptr};
    std::vector<MousePacket> packet_queue_;
};

// Global accumulator for observer callback test
inline int g_accumulated_dx = 0;
inline int g_accumulated_dy = 0;
inline void TestObserverCallback(int8_t dx, int8_t dy) {
    g_accumulated_dx += dx;
    g_accumulated_dy += dy;
}

// =============================================================================
// Verification Suite Execution
// =============================================================================

void RunAllExercises() {
    std::cout << "=========================================================\n";
    std::cout << " Chapter 6 Exercises Verification Suite (C++20 Standalone)\n";
    std::cout << "=========================================================\n\n";

    // -------------------------------------------------------------------------
    // Test 1: Vector2D & Mouse Cursor Erasing / Redrawing
    // -------------------------------------------------------------------------
    std::cout << "[Test 1] Vector2D & Mouse Cursor Rendering / Erasing... ";
    {
        constexpr Vector2D<int> v1{100, 200};
        constexpr Vector2D<int> v2{15, -25};
        constexpr Vector2D<int> v3 = v1 + v2;
        static_assert(v3.x == 115 && v3.y == 175, "Vector2D addition failed");

        MockFrameBuffer fb(800, 600, kDesktopBgColor);
        SimulatedMouseCursor cursor(fb, Vector2D<int>{200, 200});

        // Verify initial cursor drawn at (200, 200)
        // Top-left pixel of shape is '@' (Black)
        assert(fb.GetPixel(200, 200) == kCursorBlack);
        // Pixel at (200 + 1, 200 + 2) is '.' (White)
        assert(fb.GetPixel(201, 202) == kCursorWhite);

        // Move cursor by (+50, +30)
        cursor.MoveRelative(Vector2D<int>{50, 30});
        assert(cursor.GetPosition() == (Vector2D<int>{250, 230}));

        // Old position must be erased back to desktop background color!
        assert(fb.GetPixel(200, 200) == kDesktopBgColor);
        assert(fb.GetPixel(201, 202) == kDesktopBgColor);

        // New position must contain cursor pixels
        assert(fb.GetPixel(250, 230) == kCursorBlack);
        assert(fb.GetPixel(251, 232) == kCursorWhite);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 2: USB 3.0 / xHCI TRB & Ring Buffer Cycle Bit Toggle
    // -------------------------------------------------------------------------
    std::cout << "[Test 2] xHCI TRB & Cycle Bit Toggle Mechanism... ";
    {
        SimulatedTRB trb;
        trb.control = (32u << 10) | 1u; // Type 32 (TransferEvent), Cycle bit = 1
        assert(trb.Type() == 32);
        assert(trb.CycleBit() == true);
        trb.SetCycleBit(false);
        assert(trb.CycleBit() == false);

        // Ring buffer with capacity 4 (effective usable slots = 3)
        SimulatedRingBuffer ring(4);
        assert(ring.ProducerCycleBit() == true);

        // Enqueue 3 TRBs without wrapping
        assert(ring.Enqueue(0x1000, 0, (1u << 10)));
        assert(ring.Enqueue(0x2000, 0, (1u << 10)));
        assert(ring.Enqueue(0x3000, 0, (1u << 10)));
        // Ring full check
        assert(!ring.Enqueue(0x4000, 0, (1u << 10)));

        // Dequeue first TRB
        auto opt_trb1 = ring.Dequeue(true);
        assert(opt_trb1.has_value());
        assert(opt_trb1->parameter == 0x1000);
        assert(opt_trb1->CycleBit() == true);

        // Now producer can wrap around to index 0
        assert(ring.Enqueue(0x4000, 0, (1u << 10)));
        // Since it wrapped, ProducerCycleBit should have toggled to false!
        assert(ring.ProducerCycleBit() == false);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 3: PCI Configuration Space MakeAddress & Class Code Filter
    // -------------------------------------------------------------------------
    std::cout << "[Test 3] PCI MakeAddress & xHCI Class Code Verification... ";
    {
        // Test MakeAddress for Bus 0, Device 0x14, Function 0, Reg 0 (Typical xHCI on Intel/QEMU)
        uint32_t addr = MakeAddress(0, 0x14, 0, 0);
        // Bit 31: Enable (1) -> 0x80000000
        // Bus 0: bits 23:16 = 0
        // Dev 0x14 (20): bits 15:11 -> (20 << 11) = 0xA000
        // Func 0: bits 10:8 = 0
        // Reg 0: bits 7:0 = 0
        // Total expected = 0x8000A000
        assert(addr == 0x8000A000u);

        // Test alignment mask of reg_addr: 0x05 -> masked to 0x04
        uint32_t addr_masked = MakeAddress(1, 2, 3, 0x07);
        assert((addr_masked & 0xff) == 0x04);

        // Test ClassCode matching
        ClassCode xhci{0x0C, 0x03, 0x30};
        assert(xhci.Match(0x0C));
        assert(xhci.Match(0x0C, 0x03));
        assert(xhci.Match(0x0C, 0x03, 0x30));
        assert(!xhci.Match(0x01, 0x06, 0x01)); // SATA controller mismatch
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 4: PCI BAR0 32/64-bit Decoding & Sign Extension Safety
    // -------------------------------------------------------------------------
    std::cout << "[Test 4] PCI BAR0 Decoding & Sign Extension Prevention... ";
    {
        // 32-bit BAR: MMIO address 0xFE000000 (bits 2:1 = 00b, bit 0 = 0)
        uint32_t bar32 = 0xFE000000u;
        assert(!BAR0Decoder::Is64Bit(bar32));
        assert(BAR0Decoder::DecodeAddress(bar32) == 0xFE000000ull);

        // 64-bit BAR: bits 2:1 = 10b (0x4), lower = 0xFE000004, upper = 0x00000001
        uint32_t bar64_lower = 0xFE000004u;
        uint32_t bar64_upper = 0x00000001u;
        assert(BAR0Decoder::Is64Bit(bar64_lower));
        uint64_t decoded_64 = BAR0Decoder::DecodeAddress(bar64_lower, bar64_upper);
        assert(decoded_64 == 0x00000001FE000000ull);

        // Sign extension safety check:
        // In 32-bit signed int, ~0xf is 0xfffffff0.
        // If improperly cast to 64-bit without unsigned qualification,
        // it sign-extends to 0xfffffffffffffff0 (corrupting high bits).
        // With static_cast<uint64_t>(0xf), ~0xf is 0xfffffffffffffff0, but explicit uint64_t prevents compiler warnings.
        uint64_t mask = ~static_cast<uint64_t>(0xf);
        assert((0x123456789abcdefull & mask) == 0x123456789abcde0ull);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 5: Kernel Logger Log-Level Filtering Mechanism
    // -------------------------------------------------------------------------
    std::cout << "[Test 5] Logger Severity Threshold Filtering... ";
    {
        SimulatedLogger logger(LogLevel::kWarn); // Threshold = kWarn (4)

        // Error (3) <= Warn (4) -> Logged
        assert(logger.Log(LogLevel::kError, "Critical hardware error"));
        // Warn (4) <= Warn (4) -> Logged
        assert(logger.Log(LogLevel::kWarn, "Device warning"));
        // Info (5) > Warn (4) -> Filtered out (not logged)
        assert(!logger.Log(LogLevel::kInfo, "Informational message"));
        // Debug (6) > Warn (4) -> Filtered out (not logged)
        assert(!logger.Log(LogLevel::kDebug, "Verbose debug trace"));

        assert(logger.Logs().size() == 2);

        // Change log level to kDebug -> All messages pass
        logger.SetLogLevel(LogLevel::kDebug);
        assert(logger.Log(LogLevel::kDebug, "Now debug passes"));
        assert(logger.Logs().size() == 3);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 6: xHCI Polling & Observer Pipeline
    // -------------------------------------------------------------------------
    std::cout << "[Test 6] xHCI Polling Event Loop & Observer Callback... ";
    {
        SimulatedXHCIPollingController controller;
        g_accumulated_dx = 0;
        g_accumulated_dy = 0;

        controller.RegisterMouseObserver(TestObserverCallback);

        // Push 2 mouse movement packets
        controller.PushPacket(MousePacket{10, -5, 0});
        controller.PushPacket(MousePacket{-3, 8, 0});
        assert(controller.PendingEvents() == 2);

        // Poll first event
        assert(controller.ProcessEvent());
        assert(g_accumulated_dx == 10);
        assert(g_accumulated_dy == -5);
        assert(controller.PendingEvents() == 1);

        // Poll second event
        assert(controller.ProcessEvent());
        assert(g_accumulated_dx == 7);
        assert(g_accumulated_dy == 3);
        assert(controller.PendingEvents() == 0);

        // Poll when empty -> returns false
        assert(!controller.ProcessEvent());
    }
    std::cout << "PASSED\n";

    std::cout << "\n>> All 6 Chapter 6 exercise verification tests passed successfully!\n";
}

} // namespace mikanos::exercises::ch06

int main(int argc, char** argv) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    mikanos::exercises::ch06::RunAllExercises();

    if (test_mode) {
        std::cout << "[CTest] All Chapter 6 exercises verified successfully.\n";
    }
    return 0;
}
