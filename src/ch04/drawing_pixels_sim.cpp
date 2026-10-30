/**
 * @file drawing_pixels_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 4.2 (Drawing Pixels Freely & ABI)
 *
 * Covers:
 * 1. FrameBufferConfig struct memory layout, member offsets, and 64-bit alignment (sizeof == 24)
 * 2. WritePixel mathematics: 2D coordinate (x, y) mapping with scanline stride
 * 3. Color channel encoding: kPixelRGBResv8BitPerColor vs kPixelBGRResv8BitPerColor
 * 4. Boundary checking and rectangular drawing algorithms matching osbook_day04b
 * 5. Application Binary Interface (ABI) calling convention comparison:
 *    - System V AMD64 ABI (Linux/Kernel) vs Microsoft x64 ABI (Windows/UEFI)
 *    - Register assignment order (RDI vs RCX)
 *    - Callee-saved vs Caller-saved registers (RDI/RSI role inversion)
 *    - Shadow Space (32B) and Red Zone (128B)
 *    - The UEFI-to-Kernel bridge: omitting EFIAPI in Main.c
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::ch04_pixels {

// ============================================================================
// 1. MikanOS FrameBufferConfig and Pixel Color Types (Matching day04b)
// ============================================================================

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    enum PixelFormat pixel_format;
};

// Compile-time verification of struct memory layout and alignment
static_assert(sizeof(FrameBufferConfig) == 24, "FrameBufferConfig size must be exactly 24 bytes in 64-bit ABI");
static_assert(alignof(FrameBufferConfig) == 8, "FrameBufferConfig alignment must be 8 bytes");
static_assert(offsetof(FrameBufferConfig, frame_buffer) == 0, "frame_buffer offset must be 0");
static_assert(offsetof(FrameBufferConfig, pixels_per_scan_line) == 8, "pixels_per_scan_line offset must be 8");
static_assert(offsetof(FrameBufferConfig, horizontal_resolution) == 12, "horizontal_resolution offset must be 12");
static_assert(offsetof(FrameBufferConfig, vertical_resolution) == 16, "vertical_resolution offset must be 16");
static_assert(offsetof(FrameBufferConfig, pixel_format) == 20, "pixel_format offset must be 20");

struct PixelColor {
    uint8_t r, g, b;
};

/**
 * @brief Write single pixel to frame buffer matching MikanOS WritePixel.
 *
 * @param config FrameBufferConfig containing VRAM base, stride, resolution, format.
 * @param x Horizontal pixel position (0 <= x < horizontal_resolution).
 * @param y Vertical pixel position (0 <= y < vertical_resolution).
 * @param c PixelColor (r, g, b).
 * @return 0 on success, -1 on out-of-bounds or invalid format.
 */
int WritePixel(const FrameBufferConfig& config, int x, int y, const PixelColor& c) {
    if (x < 0 || static_cast<uint32_t>(x) >= config.horizontal_resolution ||
        y < 0 || static_cast<uint32_t>(y) >= config.vertical_resolution) {
        return -1;
    }

    const int pixel_position = config.pixels_per_scan_line * y + x;
    uint8_t* p = &config.frame_buffer[4 * pixel_position];

    if (config.pixel_format == kPixelRGBResv8BitPerColor) {
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0;  // Reserved byte
    } else if (config.pixel_format == kPixelBGRResv8BitPerColor) {
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0;  // Reserved byte
    } else {
        return -1;
    }
    return 0;
}

/**
 * @brief Fill a rectangle with specified color.
 *
 * @return Number of successfully written pixels.
 */
int FillRectangle(const FrameBufferConfig& config, int x, int y, int width, int height, const PixelColor& c) {
    int count = 0;
    for (int dy = 0; dy < height; ++dy) {
        for (int dx = 0; dx < width; ++dx) {
            if (WritePixel(config, x + dx, y + dy, c) == 0) {
                ++count;
            }
        }
    }
    return count;
}

// ============================================================================
// 2. Application Binary Interface (ABI) Specification & Analysis
// ============================================================================

enum class AbiKind {
    SystemV_AMD64,
    Microsoft_x64,
};

struct AbiCallingConvention {
    std::string name;
    AbiKind kind;
    std::vector<std::string> integer_arg_registers;
    std::vector<std::string> callee_saved_registers;
    std::vector<std::string> caller_saved_registers;
    size_t shadow_space_bytes;
    size_t red_zone_bytes;
    size_t stack_alignment_bytes;
};

const AbiCallingConvention kSystemV_AMD64{
    .name = "System V AMD64 ABI (Linux, BSD, MikanOS Kernel)",
    .kind = AbiKind::SystemV_AMD64,
    .integer_arg_registers = {"RDI", "RSI", "RDX", "RCX", "R8", "R9"},
    .callee_saved_registers = {"RBX", "RSP", "RBP", "R12", "R13", "R14", "R15"},
    .caller_saved_registers = {"RAX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "R10", "R11"},
    .shadow_space_bytes = 0,
    .red_zone_bytes = 128,
    .stack_alignment_bytes = 16,
};

const AbiCallingConvention kMicrosoft_x64{
    .name = "Microsoft x64 ABI (Windows, UEFI Firmware)",
    .kind = AbiKind::Microsoft_x64,
    .integer_arg_registers = {"RCX", "RDX", "R8", "R9"},
    .callee_saved_registers = {"RBX", "RBP", "RDI", "RSI", "RSP", "R12", "R13", "R14", "R15"},
    .caller_saved_registers = {"RAX", "RCX", "RDX", "R8", "R9", "R10", "R11"},
    .shadow_space_bytes = 32,
    .red_zone_bytes = 0,
    .stack_alignment_bytes = 16,
};

struct UefiKernelHandoffResult {
    bool success;
    std::string caller_abi;
    std::string passed_register;
    std::string callee_expected_register;
    std::string diagnostic;
};

/**
 * @brief Simulates the UEFI bootloader call to KernelMain.
 *
 * @param use_efiapi If true, caller function pointer uses EFIAPI (__attribute__((ms_abi))).
 *                   If false, caller function pointer uses default System V AMD64 ABI.
 */
UefiKernelHandoffResult SimulateUefiKernelHandoff(bool use_efiapi) {
    if (!use_efiapi) {
        return {
            .success = true,
            .caller_abi = "System V AMD64 ABI (EFIAPI omitted in typedef)",
            .passed_register = "RDI",
            .callee_expected_register = "RDI",
            .diagnostic = "Matched! Caller places FrameBufferConfig pointer in RDI; KernelMain reads RDI.",
        };
    } else {
        return {
            .success = false,
            .caller_abi = "Microsoft x64 ABI (EFIAPI specified in typedef)",
            .passed_register = "RCX",
            .callee_expected_register = "RDI",
            .diagnostic = "Mismatch! Caller placed pointer in RCX, but KernelMain read uninitialized RDI (#PF).",
        };
    }
}

// ============================================================================
// 3. Test Runner
// ============================================================================

void RunTests() {
    std::cout << "[Test 1] FrameBufferConfig struct memory layout & static_assert verification..." << std::endl;
    {
        assert(sizeof(FrameBufferConfig) == 24);
        assert(alignof(FrameBufferConfig) == 8);
        assert(offsetof(FrameBufferConfig, frame_buffer) == 0);
        assert(offsetof(FrameBufferConfig, pixels_per_scan_line) == 8);
        assert(offsetof(FrameBufferConfig, horizontal_resolution) == 12);
        assert(offsetof(FrameBufferConfig, vertical_resolution) == 16);
        assert(offsetof(FrameBufferConfig, pixel_format) == 20);
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 2] WritePixel with RGB format..." << std::endl;
    {
        constexpr uint32_t kWidth = 4;
        constexpr uint32_t kHeight = 4;
        constexpr uint32_t kStride = 4;
        std::vector<uint8_t> vram(kStride * kHeight * 4, 0);

        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = kStride,
            .horizontal_resolution = kWidth,
            .vertical_resolution = kHeight,
            .pixel_format = kPixelRGBResv8BitPerColor,
        };

        // Write pixel at (1, 2) with Red=250, Green=120, Blue=50
        PixelColor color{250, 120, 50};
        int ret = WritePixel(config, 1, 2, color);
        assert(ret == 0);

        // Byte offset = 4 * (2 * 4 + 1) = 4 * 9 = 36
        size_t offset = 36;
        assert(vram[offset + 0] == 250); // Red
        assert(vram[offset + 1] == 120); // Green
        assert(vram[offset + 2] == 50);  // Blue
        assert(vram[offset + 3] == 0);   // Reserved
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 3] WritePixel with BGR format..." << std::endl;
    {
        constexpr uint32_t kWidth = 4;
        constexpr uint32_t kHeight = 4;
        constexpr uint32_t kStride = 4;
        std::vector<uint8_t> vram(kStride * kHeight * 4, 0);

        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = kStride,
            .horizontal_resolution = kWidth,
            .vertical_resolution = kHeight,
            .pixel_format = kPixelBGRResv8BitPerColor,
        };

        // Write pixel at (2, 1) with Red=10, Green=200, Blue=90
        PixelColor color{10, 200, 90};
        int ret = WritePixel(config, 2, 1, color);
        assert(ret == 0);

        // Byte offset = 4 * (1 * 4 + 2) = 4 * 6 = 24
        size_t offset = 24;
        assert(vram[offset + 0] == 90);  // Blue (first in BGR)
        assert(vram[offset + 1] == 200); // Green
        assert(vram[offset + 2] == 10);  // Red (third in BGR)
        assert(vram[offset + 3] == 0);   // Reserved
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 4] Boundary checks (Out-of-bounds rejection)..." << std::endl;
    {
        constexpr uint32_t kWidth = 800;
        constexpr uint32_t kHeight = 600;
        std::vector<uint8_t> vram(kWidth * kHeight * 4, 0);

        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = kWidth,
            .horizontal_resolution = kWidth,
            .vertical_resolution = kHeight,
            .pixel_format = kPixelRGBResv8BitPerColor,
        };

        PixelColor c{255, 255, 255};
        assert(WritePixel(config, -1, 100, c) == -1);
        assert(WritePixel(config, 100, -1, c) == -1);
        assert(WritePixel(config, 800, 100, c) == -1);  // x == width (out of bounds)
        assert(WritePixel(config, 100, 600, c) == -1);  // y == height (out of bounds)
        assert(WritePixel(config, 799, 599, c) == 0);   // valid bottom-right
        assert(WritePixel(config, 0, 0, c) == 0);       // valid top-left
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 5] Stride padding handling (pixels_per_scan_line > horizontal_resolution)..." << std::endl;
    {
        constexpr uint32_t kWidth = 800;
        constexpr uint32_t kHeight = 600;
        constexpr uint32_t kStride = 1024; // 224 pixels padding per line
        std::vector<uint8_t> vram(kStride * kHeight * 4, 0);

        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = kStride,
            .horizontal_resolution = kWidth,
            .vertical_resolution = kHeight,
            .pixel_format = kPixelRGBResv8BitPerColor,
        };

        // Write pixel at (0, 1) - start of 2nd line
        PixelColor c{12, 34, 56};
        WritePixel(config, 0, 1, c);

        // Byte offset must be 4 * (1 * 1024 + 0) = 4096
        assert(vram[4096 + 0] == 12);
        assert(vram[4096 + 1] == 34);
        assert(vram[4096 + 2] == 56);

        // Padding pixels must remain untouched (0)
        for (size_t p = 800; p < 1024; ++p) {
            size_t pad_offset = 4 * p;
            assert(vram[pad_offset + 0] == 0);
            assert(vram[pad_offset + 1] == 0);
            assert(vram[pad_offset + 2] == 0);
        }
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 6] Full screen fill & rectangle drawing (MikanOS day04b simulation)..." << std::endl;
    {
        constexpr uint32_t kWidth = 400;
        constexpr uint32_t kHeight = 300;
        std::vector<uint8_t> vram(kWidth * kHeight * 4, 0);

        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = kWidth,
            .horizontal_resolution = kWidth,
            .vertical_resolution = kHeight,
            .pixel_format = kPixelRGBResv8BitPerColor,
        };

        // 1. Fill screen white
        int bg_count = FillRectangle(config, 0, 0, kWidth, kHeight, {255, 255, 255});
        assert(bg_count == kWidth * kHeight);

        // Verify background is white
        auto get_pixel = [&](int x, int y) -> PixelColor {
            size_t off = 4 * (y * kWidth + x);
            return {vram[off], vram[off + 1], vram[off + 2]};
        };
        PixelColor bg = get_pixel(50, 50);
        assert(bg.r == 255 && bg.g == 255 && bg.b == 255);

        // 2. Draw green rectangle at (100, 100) size 200x100
        int rect_count = FillRectangle(config, 100, 100, 200, 100, {0, 255, 0});
        assert(rect_count == 200 * 100);

        // Inside green rectangle
        PixelColor inside = get_pixel(150, 150);
        assert(inside.r == 0 && inside.g == 255 && inside.b == 0);

        // Outside rectangle remains white
        PixelColor outside = get_pixel(50, 50);
        assert(outside.r == 255 && outside.g == 255 && outside.b == 255);
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 7] ABI calling convention specifications..." << std::endl;
    {
        // System V AMD64 ABI: 6 argument registers starting with RDI
        assert(kSystemV_AMD64.integer_arg_registers.size() == 6);
        assert(kSystemV_AMD64.integer_arg_registers[0] == "RDI");
        assert(kSystemV_AMD64.integer_arg_registers[1] == "RSI");
        assert(kSystemV_AMD64.shadow_space_bytes == 0);
        assert(kSystemV_AMD64.red_zone_bytes == 128);

        // Microsoft x64 ABI: 4 argument registers starting with RCX
        assert(kMicrosoft_x64.integer_arg_registers.size() == 4);
        assert(kMicrosoft_x64.integer_arg_registers[0] == "RCX");
        assert(kMicrosoft_x64.integer_arg_registers[1] == "RDX");
        assert(kMicrosoft_x64.shadow_space_bytes == 32);
        assert(kMicrosoft_x64.red_zone_bytes == 0);

        // RDI/RSI role inversion:
        // In MS ABI, RDI and RSI are callee-saved (Preserved)
        auto ms_has_rdi = std::find(kMicrosoft_x64.callee_saved_registers.begin(),
                                    kMicrosoft_x64.callee_saved_registers.end(), "RDI") != kMicrosoft_x64.callee_saved_registers.end();
        assert(ms_has_rdi);

        // In System V ABI, RDI and RSI are caller-saved (Volatile arguments)
        auto sysv_has_rdi = std::find(kSystemV_AMD64.caller_saved_registers.begin(),
                                      kSystemV_AMD64.caller_saved_registers.end(), "RDI") != kSystemV_AMD64.caller_saved_registers.end();
        assert(sysv_has_rdi);
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 8] UEFI to Kernel ABI bridge & EFIAPI analysis..." << std::endl;
    {
        auto res_without_efiapi = SimulateUefiKernelHandoff(false);
        assert(res_without_efiapi.success);
        assert(res_without_efiapi.passed_register == "RDI");
        assert(res_without_efiapi.callee_expected_register == "RDI");

        auto res_with_efiapi = SimulateUefiKernelHandoff(true);
        assert(!res_with_efiapi.success);
        assert(res_with_efiapi.passed_register == "RCX");
        assert(res_with_efiapi.callee_expected_register == "RDI");
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "\nAll Chapter 4.2 C++20 tests passed successfully!" << std::endl;
}

} // namespace mikan::ch04_pixels

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "=== MikanOS Chapter 4.2: Drawing Pixels Freely & ABI Simulator (C++20) ===" << std::endl;
    try {
        mikan::ch04_pixels::RunTests();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
