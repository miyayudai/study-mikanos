/**
 * @file gop_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 3.4 (UEFI GOP & Framebuffer)
 *
 * Covers:
 * 1. UEFI Graphics Output Protocol (GOP) data structures and GUID resolution
 * 2. EFI_GRAPHICS_OUTPUT_MODE_INFORMATION and Mode pointer hierarchy
 * 3. PixelFormat byte order (PixelRedGreenBlueReserved8BitPerColor vs PixelBlueGreenRedReserved8BitPerColor)
 * 4. HorizontalResolution vs PixelsPerScanLine (scanline stride & padding) offset math
 * 5. Direct VRAM MMIO pointer write and whole-screen white fill (frame_buffer[i] = 255)
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

namespace mikan::gop_sim {

// ============================================================================
// 1. UEFI GOP Specification Data Structures (UEFI Spec 2.9+)
// ============================================================================

enum class PixelFormat : uint32_t {
    kPixelRedGreenBlueReserved8BitPerColor = 0,
    kPixelBlueGreenRedReserved8BitPerColor = 1,
    kPixelBitMask = 2,
    kPixelBltOnly = 3,
    kPixelFormatMax = 4
};

struct PixelBitmask {
    uint32_t red_mask{0x000000FF};
    uint32_t green_mask{0x0000FF00};
    uint32_t blue_mask{0x00FF0000};
    uint32_t reserved_mask{0xFF000000};
};

struct FrameBufferInfo {
    uint32_t version{0};
    uint32_t horizontal_resolution{800};
    uint32_t vertical_resolution{600};
    PixelFormat pixel_format{PixelFormat::kPixelRedGreenBlueReserved8BitPerColor};
    PixelBitmask pixel_information{};
    uint32_t pixels_per_scan_line{800};
};

struct FrameBufferMode {
    uint32_t max_mode{1};
    uint32_t mode{0};
    FrameBufferInfo* info{nullptr};
    uint64_t size_of_info{sizeof(FrameBufferInfo)};
    uint64_t frame_buffer_base{0x80000000}; // Simulated physical MMIO address
    uint64_t frame_buffer_size{0};
};

// Simulated function signature for EFI_GRAPHICS_OUTPUT_PROTOCOL
struct GraphicsOutputProtocol {
    void* query_mode{nullptr};
    void* set_mode{nullptr};
    void* blt{nullptr};
    FrameBufferMode* mode{nullptr};
};

// ============================================================================
// 2. PixelFormat to String (matching MikanOS Main.c GetPixelFormatUnicode)
// ============================================================================

std::string_view GetPixelFormatString(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::kPixelRedGreenBlueReserved8BitPerColor:
            return "PixelRedGreenBlueReserved8BitPerColor";
        case PixelFormat::kPixelBlueGreenRedReserved8BitPerColor:
            return "PixelBlueGreenRedReserved8BitPerColor";
        case PixelFormat::kPixelBitMask:
            return "PixelBitMask";
        case PixelFormat::kPixelBltOnly:
            return "PixelBltOnly";
        case PixelFormat::kPixelFormatMax:
            return "PixelFormatMax";
        default:
            return "InvalidPixelFormat";
    }
}

// ============================================================================
// 3. GOP & VRAM Framebuffer Simulator
// ============================================================================

class GopSimulator {
public:
    GopSimulator(uint32_t h_res, uint32_t v_res, uint32_t stride, PixelFormat fmt, uint64_t base_addr = 0x80000000)
        : info_{0, h_res, v_res, fmt, {}, stride},
          mode_{1, 0, &info_, sizeof(FrameBufferInfo), base_addr, static_cast<uint64_t>(stride) * v_res * 4},
          gop_{nullptr, nullptr, nullptr, &mode_},
          vram_(static_cast<size_t>(mode_.frame_buffer_size), 0)
    {
        assert(stride >= h_res && "PixelsPerScanLine cannot be less than HorizontalResolution");
    }

    [[nodiscard]] const GraphicsOutputProtocol& protocol() const noexcept { return gop_; }
    [[nodiscard]] const FrameBufferMode& mode() const noexcept { return mode_; }
    [[nodiscard]] const FrameBufferInfo& info() const noexcept { return info_; }
    [[nodiscard]] const std::vector<uint8_t>& vram() const noexcept { return vram_; }
    [[nodiscard]] std::vector<uint8_t>& vram() noexcept { return vram_; }

    /**
     * @brief Calculate physical byte offset into framebuffer using scanline stride.
     * Formula: (y * pixels_per_scan_line + x) * 4
     */
    [[nodiscard]] size_t CalculateByteOffset(uint32_t x, uint32_t y) const {
        assert(x < info_.pixels_per_scan_line && "X coordinate exceeds scanline stride");
        assert(y < info_.vertical_resolution && "Y coordinate exceeds vertical resolution");
        return (static_cast<size_t>(y) * info_.pixels_per_scan_line + x) * 4;
    }

    /**
     * @brief Fallacious offset calculation using horizontal_resolution instead of stride.
     * Demonstrates the skew/slant bug when stride > width.
     */
    [[nodiscard]] size_t CalculateBuggyByteOffset(uint32_t x, uint32_t y) const {
        return (static_cast<size_t>(y) * info_.horizontal_resolution + x) * 4;
    }

    /**
     * @brief Direct pointer write filling whole framebuffer with 255 (0xFF), exactly matching MikanOS day03b.
     */
    void FillWhite() {
        uint8_t* frame_buffer = vram_.data();
        for (size_t i = 0; i < mode_.frame_buffer_size; ++i) {
            frame_buffer[i] = 255;
        }
    }

    /**
     * @brief Write a single pixel with format-specific color packing.
     */
    void WritePixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
        if (x >= info_.horizontal_resolution || y >= info_.vertical_resolution) {
            return;
        }
        size_t offset = CalculateByteOffset(x, y);
        if (info_.pixel_format == PixelFormat::kPixelRedGreenBlueReserved8BitPerColor) {
            vram_[offset + 0] = r;
            vram_[offset + 1] = g;
            vram_[offset + 2] = b;
            vram_[offset + 3] = 0;
        } else if (info_.pixel_format == PixelFormat::kPixelBlueGreenRedReserved8BitPerColor) {
            vram_[offset + 0] = b;
            vram_[offset + 1] = g;
            vram_[offset + 2] = r;
            vram_[offset + 3] = 0;
        }
    }

    /**
     * @brief Read back color at (x, y) as (R, G, B) tuple.
     */
    [[nodiscard]] std::tuple<uint8_t, uint8_t, uint8_t> GetPixelRGB(uint32_t x, uint32_t y) const {
        size_t offset = CalculateByteOffset(x, y);
        if (info_.pixel_format == PixelFormat::kPixelRedGreenBlueReserved8BitPerColor) {
            return {vram_[offset + 0], vram_[offset + 1], vram_[offset + 2]};
        } else if (info_.pixel_format == PixelFormat::kPixelBlueGreenRedReserved8BitPerColor) {
            return {vram_[offset + 2], vram_[offset + 1], vram_[offset + 0]};
        }
        return {vram_[offset], vram_[offset], vram_[offset]};
    }

    /**
     * @brief Draw a colored solid rectangle.
     */
    void FillRectangle(uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint8_t r, uint8_t g, uint8_t b) {
        for (uint32_t dy = 0; dy < h; ++dy) {
            for (uint32_t dx = 0; dx < w; ++dx) {
                WritePixel(x + dx, y + dy, r, g, b);
            }
        }
    }

private:
    FrameBufferInfo info_;
    FrameBufferMode mode_;
    GraphicsOutputProtocol gop_;
    std::vector<uint8_t> vram_;
};

} // namespace mikan::gop_sim

// ============================================================================
// 4. Test Suite and Standalone Verification
// ============================================================================

int run_tests() {
    using namespace mikan::gop_sim;

    std::cout << "\n>>> STARTING CHAPTER 3.4 (UEFI GOP & FRAMEBUFFER) TESTS <<<\n" << std::endl;

    std::cout << "[TEST] 1. GOP Specification Structs & String Conversion..." << std::endl;
    {
        assert(GetPixelFormatString(PixelFormat::kPixelRedGreenBlueReserved8BitPerColor) ==
               "PixelRedGreenBlueReserved8BitPerColor");
        assert(GetPixelFormatString(PixelFormat::kPixelBlueGreenRedReserved8BitPerColor) ==
               "PixelBlueGreenRedReserved8BitPerColor");
        assert(GetPixelFormatString(PixelFormat::kPixelBitMask) == "PixelBitMask");
        assert(GetPixelFormatString(PixelFormat::kPixelBltOnly) == "PixelBltOnly");
    }
    std::cout << "  -> PASSED: GOP format strings verified." << std::endl;

    std::cout << "[TEST] 2. Scanline Stride Offset Math & Memory Calculations..." << std::endl;
    {
        // Standard resolution: 800x600, with hardware stride padding: 832 (common GPU alignment)
        const uint32_t width = 800;
        const uint32_t height = 600;
        const uint32_t stride = 832;
        GopSimulator sim(width, height, stride, PixelFormat::kPixelRedGreenBlueReserved8BitPerColor);

        // FrameBufferSize calculation
        assert(sim.mode().frame_buffer_size == static_cast<uint64_t>(stride) * height * 4);
        assert(sim.mode().frame_buffer_size == 832 * 600 * 4); // 1,996,800 bytes

        // Line 0 offset
        assert(sim.CalculateByteOffset(0, 0) == 0);
        assert(sim.CalculateByteOffset(10, 0) == 40);

        // Line 1 offset: MUST step by stride (832 * 4), NOT width (800 * 4)
        assert(sim.CalculateByteOffset(0, 1) == 832 * 4); // 3328
        assert(sim.CalculateByteOffset(0, 2) == 832 * 4 * 2);

        // Verify the bug: If using width instead of stride, line 1 is displaced by (832-800)*4 = 128 bytes
        size_t correct_offset = sim.CalculateByteOffset(0, 1);
        size_t buggy_offset = sim.CalculateBuggyByteOffset(0, 1);
        assert(correct_offset - buggy_offset == (stride - width) * 4); // 128 byte skew per line
    }
    std::cout << "  -> PASSED: Scanline stride calculations and skew trap verified." << std::endl;

    std::cout << "[TEST] 3. Direct VRAM Write and MikanOS Day03b White Fill..." << std::endl;
    {
        GopSimulator sim(800, 600, 800, PixelFormat::kPixelRedGreenBlueReserved8BitPerColor);

        // Initially zero
        for (uint8_t byte : sim.vram()) {
            assert(byte == 0);
        }

        // Fill all with 255 (MikanOS: frame_buffer[i] = 255)
        sim.FillWhite();

        // Every single byte must be 255
        for (uint8_t byte : sim.vram()) {
            assert(byte == 255);
        }

        // Every pixel must read as (255, 255, 255)
        auto [r0, g0, b0] = sim.GetPixelRGB(0, 0);
        assert(r0 == 255 && g0 == 255 && b0 == 255);

        auto [r_last, g_last, b_last] = sim.GetPixelRGB(799, 599);
        assert(r_last == 255 && g_last == 255 && b_last == 255);
    }
    std::cout << "  -> PASSED: Day03b whole-screen white fill verified." << std::endl;

    std::cout << "[TEST] 4. Pixel Format Byte Packing (RGBR vs BGRR)..." << std::endl;
    {
        // Test RGB format
        GopSimulator sim_rgb(10, 10, 10, PixelFormat::kPixelRedGreenBlueReserved8BitPerColor);
        sim_rgb.WritePixel(2, 3, 0x11, 0x22, 0x33);
        size_t off_rgb = sim_rgb.CalculateByteOffset(2, 3);
        assert(sim_rgb.vram()[off_rgb + 0] == 0x11); // R
        assert(sim_rgb.vram()[off_rgb + 1] == 0x22); // G
        assert(sim_rgb.vram()[off_rgb + 2] == 0x33); // B
        assert(sim_rgb.vram()[off_rgb + 3] == 0x00); // Reserved

        // Test BGR format
        GopSimulator sim_bgr(10, 10, 10, PixelFormat::kPixelBlueGreenRedReserved8BitPerColor);
        sim_bgr.WritePixel(2, 3, 0x11, 0x22, 0x33);
        size_t off_bgr = sim_bgr.CalculateByteOffset(2, 3);
        assert(sim_bgr.vram()[off_bgr + 0] == 0x33); // B
        assert(sim_bgr.vram()[off_bgr + 1] == 0x22); // G
        assert(sim_bgr.vram()[off_bgr + 2] == 0x11); // R
        assert(sim_bgr.vram()[off_bgr + 3] == 0x00); // Reserved
    }
    std::cout << "  -> PASSED: RGBR vs BGRR byte ordering verified." << std::endl;

    std::cout << "\n>>> ALL CHAPTER 3.4 UEFI GOP SIMULATION TESTS PASSED! <<<\n" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string_view(argv[1]) == "--test") {
        return run_tests();
    }

    std::cout << "=== MikanOS Chapter 3.4 UEFI GOP Simulator ===" << std::endl;
    std::cout << "Run with --test to execute verification suite." << std::endl;
    return 0;
}
