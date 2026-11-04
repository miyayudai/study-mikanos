/**
 * @file mouse_polling_sim.cpp
 * @brief C++20 Standalone Simulation of Mouse Input via xHCI Polling (MikanOS Chapter 6.4 / osbook_day06c)
 *
 * This file implements and verifies:
 * 1. PCI Base Address Register (BAR0) math: 32-bit vs 64-bit address decoding and MMIO base extraction
 * 2. Column 6.2 verification: The mystery of `static_cast<uint64_t>(0xf)` and sign extension safety
 * 3. Kernel Logger (Column 6.1): Log levels (kError, kWarn, kInfo, kDebug) and threshold filtering
 * 4. MouseCursor class: Background color erasing (EraseMouseCursor) and sprite redraw (MoveRelative)
 * 5. xHCI event loop polling emulation: Event ring dequeuing, observer pattern, and mouse movement
 *
 * Can be compiled standalone:
 *   g++ -std=c++20 -O2 -Wall -Wextra mouse_polling_sim.cpp -o mouse_polling_sim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cstdarg>
#include <map>
#include <tuple>
#include <optional>

// ============================================================================
// 1. Error Handling & WithError (Matching MikanOS kernel/error.hpp)
// ============================================================================

class Error {
public:
    enum Code {
        kSuccess,
        kFull,
        kEmpty,
        kNoEnoughMemory,
        kIndexOutOfRange,
        kHostControllerNotHalted,
        kInvalidSlotID,
        kPortNotConnected,
        kInvalidEndpointNumber,
        kTransferRingNotSet,
        kAlreadyAllocated,
        kNotImplemented,
        kInvalidDescriptor,
        kBufferTooSmall,
        kUnknownDevice,
        kNoCorrespondingSetupStage,
        kTransferFailed,
        kInvalidPhase,
        kUnknownXHCISpeedID,
        kNoWaiter,
        kLastOfCode,
    };

    constexpr Error(Code code, const char* file = "", int line = 0)
        : code_{code}, line_{line}, file_{file} {}

    constexpr operator bool() const {
        return this->code_ != kSuccess;
    }

    constexpr Code Cause() const {
        return this->code_;
    }

    const char* Name() const {
        static constexpr std::array code_names_{
            "kSuccess",
            "kFull",
            "kEmpty",
            "kNoEnoughMemory",
            "kIndexOutOfRange",
            "kHostControllerNotHalted",
            "kInvalidSlotID",
            "kPortNotConnected",
            "kInvalidEndpointNumber",
            "kTransferRingNotSet",
            "kAlreadyAllocated",
            "kNotImplemented",
            "kInvalidDescriptor",
            "kBufferTooSmall",
            "kUnknownDevice",
            "kNoCorrespondingSetupStage",
            "kTransferFailed",
            "kInvalidPhase",
            "kUnknownXHCISpeedID",
            "kNoWaiter",
        };
        static_assert(kLastOfCode == code_names_.size());
        return code_names_[static_cast<int>(this->code_)];
    }

    const char* File() const { return file_; }
    int Line() const { return line_; }

private:
    Code code_;
    int line_;
    const char* file_;
};

#define MAKE_ERROR(code) Error((code), __FILE__, __LINE__)

template <class T>
struct WithError {
    T value;
    Error error;
};

// ============================================================================
// 2. Vector2D, PixelColor, and MockPixelWriter (graphics.hpp, mouse.hpp)
// ============================================================================

template <typename T>
struct Vector2D {
    T x, y;

    constexpr Vector2D& operator+=(const Vector2D& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    constexpr bool operator==(const Vector2D& rhs) const noexcept {
        return x == rhs.x && y == rhs.y;
    }
};

struct PixelColor {
    uint8_t r, g, b;

    constexpr bool operator==(const PixelColor& rhs) const noexcept {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }
};

class MockPixelWriter {
public:
    static constexpr int kWidth = 800;
    static constexpr int kHeight = 600;

    MockPixelWriter(PixelColor fill_color = {45, 118, 237}) {
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                buffer_[y][x] = fill_color;
            }
        }
    }

    void Write(int x, int y, const PixelColor& c) {
        if (x >= 0 && x < kWidth && y >= 0 && y < kHeight) {
            buffer_[y][x] = c;
            ++write_count_;
        }
    }

    PixelColor At(int x, int y) const {
        if (x >= 0 && x < kWidth && y >= 0 && y < kHeight) {
            return buffer_[y][x];
        }
        return {0, 0, 0};
    }

    size_t WriteCount() const { return write_count_; }
    void ResetWriteCount() { write_count_ = 0; }

private:
    PixelColor buffer_[kHeight][kWidth];
    size_t write_count_{0};
};

// ============================================================================
// 3. MouseCursor Implementation (MikanOS kernel/mouse.cpp)
// ============================================================================

namespace {
    const int kMouseCursorWidth = 15;
    const int kMouseCursorHeight = 24;
    const char mouse_cursor_shape[kMouseCursorHeight][kMouseCursorWidth + 1] = {
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
        "@....@@.@      ",
        "@...@ @.@      ",
        "@..@   @.@     ",
        "@.@    @.@     ",
        "@@      @.@    ",
        "@       @.@    ",
        "         @.@   ",
        "         @@@   ",
    };

    void DrawMouseCursor(MockPixelWriter* pixel_writer, Vector2D<int> position) {
        for (int dy = 0; dy < kMouseCursorHeight; ++dy) {
            for (int dx = 0; dx < kMouseCursorWidth; ++dx) {
                if (mouse_cursor_shape[dy][dx] == '@') {
                    pixel_writer->Write(position.x + dx, position.y + dy, {0, 0, 0});
                } else if (mouse_cursor_shape[dy][dx] == '.') {
                    pixel_writer->Write(position.x + dx, position.y + dy, {255, 255, 255});
                }
            }
        }
    }

    void EraseMouseCursor(MockPixelWriter* pixel_writer, Vector2D<int> position,
                          PixelColor erase_color) {
        for (int dy = 0; dy < kMouseCursorHeight; ++dy) {
            for (int dx = 0; dx < kMouseCursorWidth; ++dx) {
                if (mouse_cursor_shape[dy][dx] != ' ') {
                    pixel_writer->Write(position.x + dx, position.y + dy, erase_color);
                }
            }
        }
    }
}

class MouseCursor {
public:
    MouseCursor(MockPixelWriter* writer, PixelColor erase_color,
                Vector2D<int> initial_position)
        : pixel_writer_{writer},
          erase_color_{erase_color},
          position_{initial_position} {
        DrawMouseCursor(pixel_writer_, position_);
    }

    void MoveRelative(Vector2D<int> displacement) {
        EraseMouseCursor(pixel_writer_, position_, erase_color_);
        position_ += displacement;
        DrawMouseCursor(pixel_writer_, position_);
    }

    Vector2D<int> Position() const { return position_; }

private:
    MockPixelWriter* pixel_writer_{nullptr};
    PixelColor erase_color_;
    Vector2D<int> position_;
};

// ============================================================================
// 4. Kernel Logger Simulation (MikanOS kernel/logger.cpp, Column 6.1)
// ============================================================================

enum LogLevel {
    kError = 3,
    kWarn  = 4,
    kInfo  = 6,
    kDebug = 7,
};

namespace {
    LogLevel current_log_level = kWarn;
    std::vector<std::pair<LogLevel, std::string>> log_history;
}

void SetLogLevel(LogLevel level) {
    current_log_level = level;
}

int Log(LogLevel level, const char* format, ...) {
    if (level > current_log_level) {
        return 0;
    }

    va_list ap;
    char s[1024];

    va_start(ap, format);
    int result = vsnprintf(s, sizeof(s), format, ap);
    va_end(ap);

    log_history.emplace_back(level, std::string(s));
    return result;
}

// ============================================================================
// 5. PCI BAR Math & Column 6.2 (kernel/pci.cpp)
// ============================================================================

constexpr uint8_t CalcBarAddress(unsigned int bar_index) {
    return 0x10 + 4 * bar_index;
}

struct MockPCIDevice {
    uint8_t bus;
    uint8_t device;
    uint8_t function;
    std::array<uint32_t, 6> bar_regs;
};

WithError<uint64_t> ReadBar(const MockPCIDevice& dev, unsigned int bar_index) {
    if (bar_index >= 6) {
        return {0, MAKE_ERROR(Error::kIndexOutOfRange)};
    }

    const uint32_t bar = dev.bar_regs[bar_index];

    // Check Bit 2: 0 = 32-bit address, 1 = 64-bit address (Bit 2:1 == 10b -> 4u)
    if ((bar & 4u) == 0) {
        return {bar, MAKE_ERROR(Error::kSuccess)};
    }

    // 64-bit address requires 2 consecutive registers
    if (bar_index >= 5) {
        return {0, MAKE_ERROR(Error::kIndexOutOfRange)};
    }

    const uint32_t bar_upper = dev.bar_regs[bar_index + 1];
    const uint64_t combined = static_cast<uint64_t>(bar) | (static_cast<uint64_t>(bar_upper) << 32);
    return {combined, MAKE_ERROR(Error::kSuccess)};
}

// ============================================================================
// 6. xHCI Polling & Observer Emulation (kernel/main.cpp)
// ============================================================================

using MouseObserverFunc = void (*)(int8_t displacement_x, int8_t displacement_y);

class MockXHCIEngine {
public:
    void SetConnected(bool connected) { connected_ = connected; }
    bool IsConnected() const { return connected_; }

    void PushMouseEvent(int8_t dx, int8_t dy) {
        event_queue_.emplace_back(dx, dy);
    }

    Error ProcessEvent(MouseObserverFunc observer) {
        if (event_queue_.empty()) {
            return MAKE_ERROR(Error::kSuccess);
        }
        auto [dx, dy] = event_queue_.front();
        event_queue_.erase(event_queue_.begin());
        if (observer) {
            observer(dx, dy);
        }
        return MAKE_ERROR(Error::kSuccess);
    }

private:
    bool connected_{true};
    std::vector<std::pair<int8_t, int8_t>> event_queue_;
};

// Global simulated cursor & observer
namespace {
    MouseCursor* g_mouse_cursor = nullptr;

    void SimulatedMouseObserver(int8_t displacement_x, int8_t displacement_y) {
        if (g_mouse_cursor) {
            g_mouse_cursor->MoveRelative({displacement_x, displacement_y});
        }
    }
}

// ============================================================================
// 7. Unit Tests
// ============================================================================

void TestBarAddressCalculation() {
    for (unsigned int i = 0; i < 6; ++i) {
        assert(CalcBarAddress(i) == (0x10 + 4 * i));
    }

    // Test 32-bit BAR decoding
    MockPCIDevice dev32{0, 4, 0, {0xFEA00000, 0, 0, 0, 0, 0}};
    auto bar32 = ReadBar(dev32, 0);
    assert(!bar32.error);
    assert(bar32.value == 0xFEA00000);

    // Test 64-bit BAR decoding (Bit 2 == 1 -> (bar & 4u) != 0)
    // Lower: 0xFEA00004 (Base: 0xFEA00000, Type: 64-bit), Upper: 0x00000002
    MockPCIDevice dev64{0, 4, 0, {0xFEA00004, 0x00000002, 0, 0, 0, 0}};
    auto bar64 = ReadBar(dev64, 0);
    assert(!bar64.error);
    assert(bar64.value == 0x00000002FEA00004ULL);

    // Column 6.2 math: clearing lower 4 flag bits
    uint64_t mmio_base = bar64.value & ~static_cast<uint64_t>(0xf);
    assert(mmio_base == 0x00000002FEA00000ULL);

    std::cout << "[PASS] TestBarAddressCalculation: 32-bit & 64-bit BAR math verified." << std::endl;
}

void TestStaticCastColumnMystery() {
    // Demonstration of Column 6.2:
    // literal 0xf is signed 32-bit int
    int lit_32 = 0xf;
    int tilde_32 = ~lit_32; // 0xFFFFFFF0 (-16)
    (void)tilde_32;

    // With explicit static_cast<uint64_t>(0xf):
    uint64_t cast_64 = static_cast<uint64_t>(0xf);
    uint64_t mask_64 = ~cast_64;
    assert(mask_64 == 0xFFFFFFFFFFFFFFF0ULL);

    uint64_t high_addr = 0x12345678FEA00004ULL;
    uint64_t cleared = high_addr & mask_64;
    assert(cleared == 0x12345678FEA00000ULL);

    std::cout << "[PASS] TestStaticCastColumnMystery: static_cast<uint64_t>(0xf) verified." << std::endl;
}

void TestKernelLoggerFiltering() {
    log_history.clear();
    SetLogLevel(kWarn); // Threshold = 4

    Log(kError, "Critical failure: code %d", 500); // Level 3 <= 4 -> Logged
    Log(kWarn, "Warning: port reset timeout");      // Level 4 <= 4 -> Logged
    Log(kInfo, "Found USB device at port 1");      // Level 6 > 4 -> Ignored
    Log(kDebug, "TRB Cycle bit toggled");          // Level 7 > 4 -> Ignored

    assert(log_history.size() == 2);
    assert(log_history[0].first == kError);
    assert(log_history[1].first == kWarn);

    // Change log level to kDebug
    SetLogLevel(kDebug);
    Log(kDebug, "Debug msg after level change");
    assert(log_history.size() == 3);

    std::cout << "[PASS] TestKernelLoggerFiltering: Logger thresholds verified." << std::endl;
}

void TestMouseCursorMoveRelativeAndErase() {
    const PixelColor bg_color{45, 118, 237};
    MockPixelWriter writer(bg_color);

    Vector2D<int> init_pos{300, 200};
    MouseCursor cursor(&writer, bg_color, init_pos);

    // Initial position should have mouse cursor drawn (top-left is '@' = black)
    assert(writer.At(300, 200) == PixelColor({0, 0, 0}));
    assert(cursor.Position() == init_pos);

    // Move relative: dx = +10, dy = -5
    cursor.MoveRelative({10, -5});
    assert(cursor.Position().x == 310);
    assert(cursor.Position().y == 195);

    // Old position (300, 200) must have been erased to background color
    assert(writer.At(300, 200) == bg_color);

    // New position (310, 195) must now contain the cursor
    assert(writer.At(310, 195) == PixelColor({0, 0, 0}));

    std::cout << "[PASS] TestMouseCursorMoveRelativeAndErase: Movement and background restore verified." << std::endl;
}

void TestXHCIPollingEventLoop() {
    const PixelColor bg_color{45, 118, 237};
    MockPixelWriter writer(bg_color);
    MouseCursor cursor(&writer, bg_color, {300, 200});
    g_mouse_cursor = &cursor;

    MockXHCIEngine xhc;
    xhc.PushMouseEvent(5, 5);
    xhc.PushMouseEvent(-2, 3);

    // Simulate polling loop: 1st event
    Error err1 = xhc.ProcessEvent(SimulatedMouseObserver);
    assert(!err1);
    assert(cursor.Position() == (Vector2D<int>{305, 205}));

    // 2nd event
    Error err2 = xhc.ProcessEvent(SimulatedMouseObserver);
    assert(!err2);
    assert(cursor.Position() == (Vector2D<int>{303, 208}));

    // Queue is empty now: should return kSuccess immediately
    Error err3 = xhc.ProcessEvent(SimulatedMouseObserver);
    assert(!err3);
    assert(cursor.Position() == (Vector2D<int>{303, 208}));

    std::cout << "[PASS] TestXHCIPollingEventLoop: Polling event loop and cursor observer verified." << std::endl;
}

void RunAllTests() {
    std::cout << "Running Chapter 6.4 Mouse Input via xHCI Polling Standalone Tests..." << std::endl;
    TestBarAddressCalculation();
    TestStaticCastColumnMystery();
    TestKernelLoggerFiltering();
    TestMouseCursorMoveRelativeAndErase();
    TestXHCIPollingEventLoop();
    std::cout << "\nAll Chapter 6.4 mouse polling simulation tests PASSED successfully!" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        RunAllTests();
        return 0;
    }

    std::cout << "=== MikanOS Chapter 6.4 Mouse Input via xHCI Polling Demo ===" << std::endl;
    RunAllTests();
    return 0;
}
