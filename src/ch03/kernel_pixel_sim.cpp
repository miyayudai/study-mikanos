/**
 * @file kernel_pixel_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 3.5 (Drawing Pixels from Kernel)
 *
 * Covers:
 * 1. Bootloader to Kernel transition and argument passing (frame_buffer_base, frame_buffer_size)
 * 2. ABI bridge mechanics: Microsoft x64 ABI (RCX, RDX) vs System V AMD64 ABI (RDI, RSI)
 * 3. The EFIAPI modifier omission that enables seamless C function pointer invocation
 * 4. Kernel entry point (KernelMain) with reinterpret_cast<uint8_t*> and frame_buffer[i] = i % 256
 * 5. Periodicity analysis: 256-byte / 64-pixel cycle, vertical stripes vs slanted moire patterns
 * 6. CPU state transition and the role of the HLT privileged instruction
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::kernel_pixel_sim {

// ============================================================================
// 1. Calling Convention and ABI Model
// ============================================================================

enum class CallingConvention {
    kMicrosoftX64,  // UEFI default: RCX, RDX, R8, R9
    kSystemVAmd64,  // Linux/ELF default: RDI, RSI, RDX, RCX, R8, R9
};

struct RegisterState {
    uint64_t rax{0};
    uint64_t rbx{0};
    uint64_t rcx{0};
    uint64_t rdx{0};
    uint64_t rsi{0};
    uint64_t rdi{0};
    uint64_t r8{0};
    uint64_t r9{0};
    uint64_t rsp{0};
    uint64_t rbp{0};
    uint64_t rip{0};
};

/**
 * @brief Simulates compiler argument preparation for a 2-argument function call.
 */
RegisterState PrepareArguments(CallingConvention abi, uint64_t arg1, uint64_t arg2) {
    RegisterState regs{};
    if (abi == CallingConvention::kMicrosoftX64) {
        // Microsoft x64 ABI: 1st in RCX, 2nd in RDX
        regs.rcx = arg1;
        regs.rdx = arg2;
    } else {
        // System V AMD64 ABI: 1st in RDI, 2nd in RSI
        regs.rdi = arg1;
        regs.rsi = arg2;
    }
    return regs;
}

// ============================================================================
// 2. Simulated Kernel Implementation (osbook_day03c kernel/main.cpp)
// ============================================================================

/**
 * @brief Simulated KernelMain matching day03c implementation:
 *
 * extern "C" void KernelMain(uint64_t frame_buffer_base,
 *                            uint64_t frame_buffer_size) {
 *   uint8_t* frame_buffer = reinterpret_cast<uint8_t*>(frame_buffer_base);
 *   for (uint64_t i = 0; i < frame_buffer_size; ++i) {
 *     frame_buffer[i] = i % 256;
 *   }
 *   while (1) __asm__("hlt");
 * }
 */
void SimulatedKernelMain(uint64_t frame_buffer_base, uint64_t frame_buffer_size) {
    auto* frame_buffer = reinterpret_cast<uint8_t*>(frame_buffer_base);
    for (uint64_t i = 0; i < frame_buffer_size; ++i) {
        frame_buffer[i] = static_cast<uint8_t>(i % 256);
    }
}

// ============================================================================
// 3. Stride and Color Periodicity Analysis
// ============================================================================

struct PixelColorRGB {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};
};

enum class PixelFormat {
    kRgbReserved8Bit,
    kBgrReserved8Bit,
};

struct StrideAnalysisResult {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t cycle_pixels;
    uint32_t stride_pixel_mod;
    uint32_t stride_byte_mod;
    bool is_pure_vertical;
};

StrideAnalysisResult AnalyzeStrideGeometry(uint32_t width, uint32_t height, uint32_t stride) {
    constexpr uint32_t kCycleBytes = 256;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kCyclePixels = kCycleBytes / kBytesPerPixel; // 64 pixels

    uint32_t stride_pixel_mod = stride % kCyclePixels;
    uint32_t stride_byte_mod = (stride * kBytesPerPixel) % kCycleBytes;
    bool is_pure_vertical = (stride_byte_mod == 0);

    return StrideAnalysisResult{
        .width = width,
        .height = height,
        .stride = stride,
        .cycle_pixels = kCyclePixels,
        .stride_pixel_mod = stride_pixel_mod,
        .stride_byte_mod = stride_byte_mod,
        .is_pure_vertical = is_pure_vertical,
    };
}

PixelColorRGB GetPixelColorAt(
    const uint8_t* buffer,
    uint32_t x,
    uint32_t y,
    uint32_t stride,
    PixelFormat format
) {
    size_t offset = (static_cast<size_t>(y) * stride + x) * 4;
    uint8_t b0 = buffer[offset];
    uint8_t b1 = buffer[offset + 1];
    uint8_t b2 = buffer[offset + 2];

    if (format == PixelFormat::kBgrReserved8Bit) {
        // b0=B, b1=G, b2=R
        return PixelColorRGB{.r = b2, .g = b1, .b = b0};
    } else {
        // b0=R, b1=G, b2=B
        return PixelColorRGB{.r = b0, .g = b1, .b = b2};
    }
}

// ============================================================================
// 4. Verification Tests
// ============================================================================

void TestAbiCallingConventionBridge() {
    constexpr uint64_t kDummyBase = 0x80000000;
    constexpr uint64_t kDummySize = 1024 * 768 * 4;

    // 1. With EFIAPI: compiler emits Microsoft x64 code (RCX, RDX)
    RegisterState ms_regs = PrepareArguments(CallingConvention::kMicrosoftX64, kDummyBase, kDummySize);
    assert(ms_regs.rcx == kDummyBase);
    assert(ms_regs.rdx == kDummySize);
    assert(ms_regs.rdi == 0);
    assert(ms_regs.rsi == 0);

    // 2. WITHOUT EFIAPI in Linux GCC/Clang: compiler emits System V AMD64 code (RDI, RSI)
    RegisterState sysv_regs = PrepareArguments(CallingConvention::kSystemVAmd64, kDummyBase, kDummySize);
    assert(sysv_regs.rdi == kDummyBase);
    assert(sysv_regs.rsi == kDummySize);
    assert(sysv_regs.rcx == 0);
    assert(sysv_regs.rdx == 0);

    // 3. Verify that KernelMain expects RDI and RSI:
    // KernelMain(uint64_t frame_buffer_base, uint64_t frame_buffer_size)
    // In System V ABI, arg1 is in RDI, arg2 is in RSI.
    assert(sysv_regs.rdi == kDummyBase);
    assert(sysv_regs.rsi == kDummySize);

    std::cout << "[PASS] ABI Calling Convention Bridge Test" << std::endl;
}

void TestSimulatedKernelMainExecution() {
    [[maybe_unused]] constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 480;
    constexpr uint32_t kStride = 640;
    constexpr size_t kBufferSize = static_cast<size_t>(kStride) * kHeight * 4;

    std::vector<uint8_t> memory_pool(kBufferSize, 0);
    uint64_t base_addr = reinterpret_cast<uint64_t>(memory_pool.data());

    // Execute kernel main
    SimulatedKernelMain(base_addr, kBufferSize);

    // Verify pattern: frame_buffer[i] == i % 256
    for (size_t i = 0; i < 256; ++i) {
        assert(memory_pool[i] == static_cast<uint8_t>(i));
    }
    for (size_t i = 256; i < 512; ++i) {
        assert(memory_pool[i] == static_cast<uint8_t>(i % 256));
    }
    assert(memory_pool[kBufferSize - 1] == static_cast<uint8_t>((kBufferSize - 1) % 256));

    // Verify pixel 0 (bytes 0, 1, 2, 3) in BGR format:
    // byte0: B=0, byte1: G=1, byte2: R=2, byte3: Resv=3
    PixelColorRGB p0 = GetPixelColorAt(memory_pool.data(), 0, 0, kStride, PixelFormat::kBgrReserved8Bit);
    assert(p0.b == 0);
    assert(p0.g == 1);
    assert(p0.r == 2);

    // Verify pixel 1 (bytes 4, 5, 6, 7):
    PixelColorRGB p1 = GetPixelColorAt(memory_pool.data(), 1, 0, kStride, PixelFormat::kBgrReserved8Bit);
    assert(p1.b == 4);
    assert(p1.g == 5);
    assert(p1.r == 6);

    // Verify pixel 63 (bytes 252, 253, 254, 255):
    PixelColorRGB p63 = GetPixelColorAt(memory_pool.data(), 63, 0, kStride, PixelFormat::kBgrReserved8Bit);
    assert(p63.b == 252);
    assert(p63.g == 253);
    assert(p63.r == 254);

    // Verify pixel 64 (bytes 256, 257, 258, 259) wraps around to 0, 1, 2:
    PixelColorRGB p64 = GetPixelColorAt(memory_pool.data(), 64, 0, kStride, PixelFormat::kBgrReserved8Bit);
    assert(p64.b == 0);
    assert(p64.g == 1);
    assert(p64.r == 2);

    std::cout << "[PASS] Simulated KernelMain Execution Test" << std::endl;
}

void TestStrideGeometryAndMoireAnalysis() {
    // Case 1: Stride is a multiple of 64 (e.g. 1024)
    auto res_1024 = AnalyzeStrideGeometry(1024, 768, 1024);
    assert(res_1024.cycle_pixels == 64);
    assert(res_1024.stride_pixel_mod == 0);
    assert(res_1024.stride_byte_mod == 0);
    assert(res_1024.is_pure_vertical == true);

    // Case 2: Stride is a multiple of 64 (e.g. 640) -> 640 % 64 == 0!
    auto res_640 = AnalyzeStrideGeometry(640, 480, 640);
    assert(res_640.stride_pixel_mod == 0);
    assert(res_640.is_pure_vertical == true);

    // Case 3: Stride is NOT a multiple of 64 (e.g. 800) -> 800 % 64 == 32
    auto res_800 = AnalyzeStrideGeometry(800, 600, 800);
    assert(res_800.stride_pixel_mod == 32);
    assert(res_800.stride_byte_mod == 128); // 800 * 4 = 3200 = 256 * 12 + 128
    assert(res_800.is_pure_vertical == false);

    // In res_800, row 0 and row 1 have a 32-pixel (128-byte) horizontal phase offset:
    std::vector<uint8_t> buffer_800(800 * 600 * 4);
    SimulatedKernelMain(reinterpret_cast<uint64_t>(buffer_800.data()), buffer_800.size());

    PixelColorRGB r0_p0 = GetPixelColorAt(buffer_800.data(), 0, 0, 800, PixelFormat::kBgrReserved8Bit);
    PixelColorRGB r1_p0 = GetPixelColorAt(buffer_800.data(), 0, 1, 800, PixelFormat::kBgrReserved8Bit);

    // Row 0, Pixel 0 is byte 0 (B=0, G=1, R=2)
    assert(r0_p0.b == 0);
    // Row 1, Pixel 0 offset is 800 * 4 = 3200. 3200 % 256 = 128.
    // bytes 128, 129, 130: B=128, G=129, R=130
    assert(r1_p0.b == 128);
    assert(r1_p0.g == 129);
    assert(r1_p0.r == 130);

    // Thus Row 1, Pixel 32 offset is 3200 + 32 * 4 = 3200 + 128 = 3328. 3328 % 256 = 0!
    PixelColorRGB r1_p32 = GetPixelColorAt(buffer_800.data(), 32, 1, 800, PixelFormat::kBgrReserved8Bit);
    assert(r1_p32.b == 0);
    assert(r1_p32.g == 1);
    assert(r1_p32.r == 2);

    std::cout << "[PASS] Stride Geometry and Moire Analysis Test" << std::endl;
}

} // namespace mikan::kernel_pixel_sim

// ============================================================================
// Main Execution
// ============================================================================

int main() {
    std::cout << "========================================================\n"
              << " MikanOS Chapter 3.5: Kernel Pixel Simulation (day03c)  \n"
              << "========================================================\n\n";

    mikan::kernel_pixel_sim::TestAbiCallingConventionBridge();
    mikan::kernel_pixel_sim::TestSimulatedKernelMainExecution();
    mikan::kernel_pixel_sim::TestStrideGeometryAndMoireAnalysis();

    std::cout << "\n--------------------------------------------------------\n"
              << " ABI Handoff & Register Verification Summary:          \n"
              << "--------------------------------------------------------\n"
              << " Bootloader (EDK II): Caller omitted EFIAPI modifier   \n"
              << " Code generated     : mov rdi, base; mov rsi, size     \n"
              << " Kernel (ELF / C++) : KernelMain(rdi, rsi)             \n"
              << " Status             : Seamless Register Compatibility! \n"
              << " Pattern Cycle      : 256 Bytes / 64 Pixels per Period \n"
              << " Stride Geometry    : 1024 -> Pure Vertical Stripes    \n"
              << "                      800  -> Slanted Moire (32px shift)\n"
              << "--------------------------------------------------------\n"
              << "[SUCCESS] All Chapter 3.5 C++20 verifications passed!\n";
    return 0;
}
