/**
 * @file os_chapter3_exercises.cpp
 * @brief Chapter 3 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 3:
 * 1. x86-64 Register Hierarchy & 32-bit Zero-Extension Behavior
 * 2. System V AMD64 ABI Red Zone (128-byte) Stack Protection vs. Interrupt Corruption
 * 3. UEFI GOP Framebuffer Stride (PixelsPerScanLine) & Pixel Formats (RGB vs BGR)
 * 4. System V AMD64 ABI Calling Convention (RDI, RSI, RDX, RCX, R8, R9) & Kernel Entry Handoff
 * 5. UEFI Status Error Detection & Type-safe reinterpret_cast
 * 6. x86-64 SIB (Scale-Index-Base) Effective Address Computation
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch03 {

// =============================================================================
// Exercise 1: x86-64 Register File & 32-bit Zero-Extension Simulation
// =============================================================================

class MockRegister64 {
public:
    uint64_t full{0};

    void write64(uint64_t val) noexcept {
        full = val;
    }

    uint64_t read64() const noexcept {
        return full;
    }

    // Writing to 32-bit sub-register ALWAYS zero-extends to upper 32 bits
    void write32(uint32_t val) noexcept {
        full = static_cast<uint64_t>(val);
    }

    uint32_t read32() const noexcept {
        return static_cast<uint32_t>(full & 0xFFFFFFFFULL);
    }

    // Writing to 16-bit preserves upper 48 bits
    void write16(uint16_t val) noexcept {
        full = (full & ~0xFFFFULL) | static_cast<uint64_t>(val);
    }

    uint16_t read16() const noexcept {
        return static_cast<uint16_t>(full & 0xFFFFULL);
    }

    // Writing to low 8-bit preserves upper 56 bits
    void write8_low(uint8_t val) noexcept {
        full = (full & ~0xFFULL) | static_cast<uint64_t>(val);
    }

    uint8_t read8_low() const noexcept {
        return static_cast<uint8_t>(full & 0xFFULL);
    }
};

// =============================================================================
// Exercise 2: Red Zone (128-byte) Trap & Interrupt Simulation
// =============================================================================

class StackSimulator {
public:
    static constexpr size_t kStackSize = 4096;
    static constexpr size_t kRedZoneBytes = 128;

    std::vector<uint8_t> memory;
    uintptr_t base_addr;
    uintptr_t rsp;

    StackSimulator() : memory(kStackSize, 0xCC), base_addr(0x7FFF0000), rsp(base_addr + kStackSize) {}

    // Function using Red Zone writes into [RSP - 128, RSP) WITHOUT moving RSP
    void write_red_zone(size_t offset_below_rsp, uint8_t value) {
        assert(offset_below_rsp <= kRedZoneBytes && offset_below_rsp > 0);
        uintptr_t target = rsp - offset_below_rsp;
        size_t idx = target - base_addr;
        memory[idx] = value;
    }

    uint8_t read_red_zone(size_t offset_below_rsp) const {
        assert(offset_below_rsp <= kRedZoneBytes && offset_below_rsp > 0);
        uintptr_t target = rsp - offset_below_rsp;
        size_t idx = target - base_addr;
        return memory[idx];
    }

    // Asynchronous interrupt occurs: CPU automatically pushes RIP, CS, RFLAGS, RSP, SS onto stack
    void trigger_hardware_interrupt() {
        // Hardware pushes interrupt frame (5 * 8 = 40 bytes) below current RSP!
        for (size_t i = 1; i <= 40; ++i) {
            uintptr_t target = rsp - i;
            size_t idx = target - base_addr;
            memory[idx] = 0xAA; // Overwritten by CPU interrupt frame!
        }
        rsp -= 40;
    }

    // Safe kernel function with -mno-red-zone explicitly decrements RSP before writing
    void allocate_frame_safe(size_t bytes) {
        rsp -= bytes;
    }
};

// =============================================================================
// Exercise 3: GOP Framebuffer & Stride Calculation
// =============================================================================

enum class PixelFormat {
    kRgb,
    kBgr,
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};
};

class FrameBufferEngine {
public:
    uint32_t width;
    uint32_t height;
    uint32_t ppsl; // PixelsPerScanLine (stride)
    PixelFormat format;
    std::vector<uint8_t> buffer;

    FrameBufferEngine(uint32_t w, uint32_t h, uint32_t stride, PixelFormat fmt)
        : width(w), height(h), ppsl(stride), format(fmt), buffer(stride * h * 4, 0) {}

    uint32_t compute_byte_offset(uint32_t x, uint32_t y) const noexcept {
        return 4 * (y * ppsl + x);
    }

    void draw_pixel(uint32_t x, uint32_t y, PixelColor color) {
        if (x >= width || y >= height) return;
        uint32_t offset = compute_byte_offset(x, y);
        if (format == PixelFormat::kRgb) {
            buffer[offset + 0] = color.r;
            buffer[offset + 1] = color.g;
            buffer[offset + 2] = color.b;
            buffer[offset + 3] = 0; // Reserved
        } else {
            buffer[offset + 0] = color.b;
            buffer[offset + 1] = color.g;
            buffer[offset + 2] = color.r;
            buffer[offset + 3] = 0; // Reserved
        }
    }

    PixelColor get_pixel(uint32_t x, uint32_t y) const {
        if (x >= width || y >= height) return {};
        uint32_t offset = compute_byte_offset(x, y);
        if (format == PixelFormat::kRgb) {
            return {buffer[offset + 0], buffer[offset + 1], buffer[offset + 2]};
        } else {
            return {buffer[offset + 2], buffer[offset + 1], buffer[offset + 0]};
        }
    }

    void fill_white() {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                draw_pixel(x, y, {255, 255, 255});
            }
        }
    }
};

// =============================================================================
// Exercise 4: System V AMD64 ABI Argument Passing Simulation
// =============================================================================

struct CpuAbiRegisters {
    uint64_t rdi{0};
    uint64_t rsi{0};
    uint64_t rdx{0};
    uint64_t rcx{0};
    uint64_t r8{0};
    uint64_t r9{0};
    uint64_t rax{0};
};

// Simulation of kernel entry handoff: KernelMain(uint64_t fb_base, uint64_t fb_size)
inline void MockKernelEntry(const CpuAbiRegisters& regs, uint64_t& out_fb_base, uint64_t& out_fb_size) {
    // Under System V AMD64 ABI, 1st arg is in RDI, 2nd arg is in RSI
    out_fb_base = regs.rdi;
    out_fb_size = regs.rsi;
}

// =============================================================================
// Exercise 5: UEFI Status Code & Type-Safe reinterpret_cast
// =============================================================================

using EfiStatus = uint64_t;
constexpr EfiStatus kEfiSuccess = 0;
constexpr EfiStatus kEfiHighBit = 0x8000000000000000ULL;
constexpr EfiStatus kEfiBufferTooSmall = kEfiHighBit | 5;
constexpr EfiStatus kEfiNotFound = kEfiHighBit | 14;

constexpr bool EfiError(EfiStatus status) noexcept {
    return (status & kEfiHighBit) != 0;
}

// =============================================================================
// Exercise 6: SIB (Scale-Index-Base) Address Computation
// =============================================================================

inline uintptr_t compute_sib_address(uintptr_t base, uint64_t index, uint8_t scale, int32_t disp) noexcept {
    return base + (index * scale) + disp;
}

} // namespace mikanos::exercises::ch03

// =============================================================================
// Self-Testing & Verification Runner
// =============================================================================

int main(int argc, char* argv[]) {
    using namespace mikanos::exercises::ch03;
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }
    if (test_mode) {
        std::cout << "[Info] Running in test mode.\n";
    }

    std::cout << "========================================================\n";
    std::cout << " MikanOS Chapter 3 Comprehensive Exercises (C++20)\n";
    std::cout << "========================================================\n\n";

    // 1. Verify 32-bit zero-extension rule
    {
        std::cout << "[1] Verifying x86-64 Register Zero-Extension Rule...\n";
        MockRegister64 reg;
        reg.write64(0xFFFFFFFFFFFFFFFFULL);
        assert(reg.read64() == 0xFFFFFFFFFFFFFFFFULL);

        reg.write32(0x12345678U);
        // High 32 bits MUST be cleared to 0!
        assert(reg.read64() == 0x0000000012345678ULL);
        assert(reg.read32() == 0x12345678U);

        // 16-bit write preserves upper bits
        reg.write16(0xABCDU);
        assert(reg.read64() == 0x000000001234ABCDULL);

        // 8-bit low write preserves upper bits
        reg.write8_low(0x55U);
        assert(reg.read64() == 0x000000001234AB55ULL);
        std::cout << "    -> Passed: 32-bit write zero-extension and sub-register preservation verified.\n\n";
    }

    // 2. Verify Red Zone trap & interrupt stack corruption
    {
        std::cout << "[2] Verifying System V AMD64 ABI Red Zone Behavior...\n";
        StackSimulator stack;
        uintptr_t initial_rsp = stack.rsp;

        // Leaf function writes 0x42 into Red Zone at RSP - 16
        stack.write_red_zone(16, 0x42);
        assert(stack.read_red_zone(16) == 0x42);
        assert(stack.rsp == initial_rsp); // RSP was NOT moved!

        // Hardware interrupt strikes!
        stack.trigger_hardware_interrupt();
        // The interrupt handler pushed hardware frame, corrupting Red Zone at RSP - 16!
        uint8_t corrupted_val = stack.memory[initial_rsp - 16 - stack.base_addr];
        assert(corrupted_val == 0xAA); // Overwritten!
        std::cout << "    -> Passed: Red Zone corruption by asynchronous interrupt demonstrated.\n";

        // Now demonstrate safe compilation with -mno-red-zone
        StackSimulator safe_stack;
        safe_stack.allocate_frame_safe(64); // RSP explicitly decremented
        safe_stack.memory[safe_stack.rsp - safe_stack.base_addr + 16] = 0x99;
        safe_stack.trigger_hardware_interrupt();
        // The data is above new RSP, so it remains intact!
        assert(safe_stack.memory[safe_stack.rsp + 40 - safe_stack.base_addr + 16] == 0x99);
        std::cout << "    -> Passed: -mno-red-zone stack frame safety verified.\n\n";
    }

    // 3. Verify GOP Framebuffer Stride & Pixel Formats
    {
        std::cout << "[3] Verifying GOP Framebuffer Stride & Formats (RGB vs BGR)...\n";
        const uint32_t w = 800;
        const uint32_t h = 600;
        const uint32_t ppsl = 832; // Stride includes 32-pixel hardware padding!

        FrameBufferEngine fb_rgb(w, h, ppsl, PixelFormat::kRgb);
        FrameBufferEngine fb_bgr(w, h, ppsl, PixelFormat::kBgr);

        assert(fb_rgb.compute_byte_offset(100, 50) == 4 * (50 * 832 + 100));
        assert(fb_rgb.buffer.size() == 832 * 600 * 4);

        // Draw pixel at (10, 20) with pure Red (255, 0, 0)
        fb_rgb.draw_pixel(10, 20, {255, 0, 0});
        fb_bgr.draw_pixel(10, 20, {255, 0, 0});

        uint32_t off = fb_rgb.compute_byte_offset(10, 20);
        // In RGB: byte 0 is R (255), byte 2 is B (0)
        assert(fb_rgb.buffer[off + 0] == 255 && fb_rgb.buffer[off + 2] == 0);
        // In BGR: byte 0 is B (0), byte 2 is R (255)
        assert(fb_bgr.buffer[off + 0] == 0 && fb_bgr.buffer[off + 2] == 255);

        // Both engines correctly return logical PixelColor
        assert(fb_rgb.get_pixel(10, 20).r == 255);
        assert(fb_bgr.get_pixel(10, 20).r == 255);

        fb_rgb.fill_white();
        assert(fb_rgb.get_pixel(799, 599).r == 255);
        assert(fb_rgb.get_pixel(799, 599).g == 255);
        assert(fb_rgb.get_pixel(799, 599).b == 255);
        std::cout << "    -> Passed: Framebuffer stride math & RGB/BGR memory layout verified.\n\n";
    }

    // 4. Verify System V AMD64 ABI Argument Passing
    {
        std::cout << "[4] Verifying System V AMD64 ABI Calling Convention...\n";
        CpuAbiRegisters regs;
        regs.rdi = 0xE0000000ULL; // 1st arg: FrameBufferBase
        regs.rsi = 0x00800000ULL; // 2nd arg: FrameBufferSize (8 MiB)

        uint64_t fb_base = 0;
        uint64_t fb_size = 0;
        MockKernelEntry(regs, fb_base, fb_size);

        assert(fb_base == 0xE0000000ULL);
        assert(fb_size == 0x00800000ULL);
        std::cout << "    -> Passed: KernelMain(fb_base in RDI, fb_size in RSI) handoff verified.\n\n";
    }

    // 5. Verify UEFI Status Code & Type-Safe reinterpret_cast
    {
        std::cout << "[5] Verifying UEFI Status Code & Casts...\n";
        assert(!EfiError(kEfiSuccess));
        assert(EfiError(kEfiBufferTooSmall));
        assert(EfiError(kEfiNotFound));

        uint64_t raw_addr = 0x100000ULL;
        auto* ptr = reinterpret_cast<uint8_t*>(raw_addr);
        uintptr_t roundtrip_addr = reinterpret_cast<uintptr_t>(ptr);
        assert(roundtrip_addr == raw_addr);
        std::cout << "    -> Passed: EFI_STATUS high-bit error macro & reinterpret_cast verified.\n\n";
    }

    // 6. Verify SIB Address Computation
    {
        std::cout << "[6] Verifying SIB (Scale-Index-Base) Effective Address Computation...\n";
        uintptr_t base = 0x10000000ULL;
        uint64_t index = 42;
        uint8_t scale = 4; // 4 bytes per pixel
        int32_t disp = 16;

        uintptr_t ea = compute_sib_address(base, index, scale, disp);
        assert(ea == (0x10000000ULL + 42 * 4 + 16));
        std::cout << "    -> Passed: SIB address computation [base + index*4 + 16] verified.\n\n";
    }

    std::cout << "========================================================\n";
    std::cout << " All Chapter 3 Exercises C++20 Verifications PASSED!\n";
    std::cout << "========================================================\n";

    return 0;
}
