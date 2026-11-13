/**
 * @file os_chapter10_high_speed_counter.cpp
 * @brief Chapter 10.3: High-Speed Counter & Screen Flicker / Tearing Analysis (C++20)
 *
 * Demonstrates:
 * 1. High-Speed Counter Window Dimensions (160x52, reduced from 160x68 in osbook_day10b)
 * 2. Busy-Loop Counter Loop (`sprintf`, `FillRectangle`, `WriteString`, `LayerManager::Draw`)
 * 3. Counter Text Erasing & Overwriting Math (24, 28, size 80x16, "%010u")
 * 4. Mathematical & Bandwidth Analysis of Screen Flickering and Tearing:
 *    - Full Redraw (3.14 MB) vs. Diff Update (5.12 KB): ~614.4x Bandwidth Waste
 *    - Lack of V-Sync / Raster Scan Collision without Double Buffering
 * 5. C++20 Standalone Executable Verification & Pixel Assertions
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
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::study {

// =============================================================================
// 1. Geometric Data Types and Colors
// =============================================================================

template <typename T>
struct Vector2D {
    T x, y;

    constexpr bool operator==(const Vector2D& rhs) const noexcept {
        return x == rhs.x && y == rhs.y;
    }
};

template <typename T>
struct Rectangle {
    Vector2D<T> pos;
    Vector2D<T> size;
};

struct PixelColor {
    uint8_t r, g, b;

    constexpr bool operator==(const PixelColor& rhs) const noexcept {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }
};

constexpr PixelColor ToColor(uint32_t c) noexcept {
    return {
        static_cast<uint8_t>((c >> 16) & 0xff),
        static_cast<uint8_t>((c >> 8) & 0xff),
        static_cast<uint8_t>(c & 0xff)
    };
}

constexpr PixelColor kDesktopBGColor{0x00, 0x80, 0x80};   // Dark Cyan/Green
constexpr PixelColor kWindowBGColor{0xc6, 0xc6, 0xc6};    // Light Gray
constexpr PixelColor kHighlightColor{0xff, 0xff, 0xff};   // Direct White
constexpr PixelColor kShadowColor{0x84, 0x84, 0x84};      // Dark Gray Shadow
constexpr PixelColor kDarkShadowColor{0x00, 0x00, 0x00};  // Black Shadow
constexpr PixelColor kTitleBarColor{0x00, 0x00, 0x84};    // Classic Navy Blue

// =============================================================================
// 2. PixelWriter Interface and Window Buffer
// =============================================================================

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(Vector2D<int> pos, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

void FillRectangle(PixelWriter& writer, Vector2D<int> pos, Vector2D<int> size, const PixelColor& c) {
    for (int dy = 0; dy < size.y; ++dy) {
        for (int dx = 0; dx < size.x; ++dx) {
            writer.Write({pos.x + dx, pos.y + dy}, c);
        }
    }
}

// Minimal 8x16 Hankaku font subset for digits '0'-'9', space, and window title
// Digits '0'-'9' standard bitmap representation
const uint8_t kDigitFont[10][16] = {
    {0x00, 0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00}, // 0
    {0x00, 0x10, 0x30, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x38, 0x00, 0x00, 0x00, 0x00}, // 1
    {0x00, 0x38, 0x44, 0x04, 0x04, 0x08, 0x10, 0x20, 0x40, 0x40, 0x44, 0x7c, 0x00, 0x00, 0x00, 0x00}, // 2
    {0x00, 0x38, 0x44, 0x04, 0x04, 0x18, 0x04, 0x04, 0x04, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00}, // 3
    {0x00, 0x08, 0x18, 0x28, 0x48, 0x48, 0x48, 0x7c, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00, 0x00}, // 4
    {0x00, 0x7c, 0x40, 0x40, 0x78, 0x44, 0x04, 0x04, 0x04, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00}, // 5
    {0x00, 0x18, 0x20, 0x40, 0x40, 0x78, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00}, // 6
    {0x00, 0x7c, 0x44, 0x04, 0x08, 0x08, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00}, // 7
    {0x00, 0x38, 0x44, 0x44, 0x44, 0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38, 0x00, 0x00, 0x00, 0x00}, // 8
    {0x00, 0x38, 0x44, 0x44, 0x44, 0x44, 0x3c, 0x04, 0x04, 0x04, 0x08, 0x30, 0x00, 0x00, 0x00, 0x00}, // 9
};

void WriteDigit(PixelWriter& writer, Vector2D<int> pos, char c, const PixelColor& color) {
    if (c >= '0' && c <= '9') {
        int idx = c - '0';
        for (int dy = 0; dy < 16; ++dy) {
            uint8_t row = kDigitFont[idx][dy];
            for (int dx = 0; dx < 8; ++dx) {
                if ((row << dx) & 0x80) {
                    writer.Write({pos.x + dx, pos.y + dy}, color);
                }
            }
        }
    }
}

void WriteString(PixelWriter& writer, Vector2D<int> pos, std::string_view s, const PixelColor& color) {
    for (size_t i = 0; i < s.size(); ++i) {
        WriteDigit(writer, {pos.x + static_cast<int>(i) * 8, pos.y}, s[i], color);
    }
}

// =============================================================================
// 3. Window & Window Decoration (osbook_day10b -> osbook_day10c)
// =============================================================================

class Window {
public:
    class WindowWriter : public PixelWriter {
    public:
        WindowWriter(Window& window) : window_{window} {}
        void Write(Vector2D<int> pos, const PixelColor& c) override {
            window_.Write(pos, c);
        }
        int Width() const override { return window_.Width(); }
        int Height() const override { return window_.Height(); }
    private:
        Window& window_;
    };

    Window(int width, int height)
        : width_{width}, height_{height}, data_(width * height, kWindowBGColor), writer_{*this} {}

    void Write(Vector2D<int> pos, const PixelColor& c) {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            data_[pos.y * width_ + pos.x] = c;
        }
    }

    PixelColor At(Vector2D<int> pos) const {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            return data_[pos.y * width_ + pos.x];
        }
        return {0, 0, 0};
    }

    int Width() const noexcept { return width_; }
    int Height() const noexcept { return height_; }
    PixelWriter* Writer() noexcept { return &writer_; }

private:
    int width_, height_;
    std::vector<PixelColor> data_;
    WindowWriter writer_;
};

void DrawWindow(PixelWriter& writer, [[maybe_unused]] const char* title) {
    auto fill_rect = [&writer](Vector2D<int> pos, Vector2D<int> size, uint32_t c) {
        FillRectangle(writer, pos, size, ToColor(c));
    };

    const int win_w = writer.Width();
    const int win_h = writer.Height();

    // 3D Bevel Window Borders
    fill_rect({0, 0},         {win_w, 1},             0xc6c6c6);
    fill_rect({1, 1},         {win_w - 2, 1},         0xffffff);
    fill_rect({0, 0},         {1, win_h},             0xc6c6c6);
    fill_rect({1, 1},         {1, win_h - 2},         0xffffff);
    fill_rect({win_w - 2, 1}, {1, win_h - 2},         0x848484);
    fill_rect({win_w - 1, 0}, {1, win_h},             0x000000);
    fill_rect({2, 2},         {win_w - 4, win_h - 4}, 0xc6c6c6);
    fill_rect({3, 3},         {win_w - 6, 18},        0x000084);
    fill_rect({1, win_h - 2}, {win_w - 2, 1},         0x848484);
    fill_rect({0, win_h - 1}, {win_w, 1},             0x000000);
}

// =============================================================================
// 4. Bandwidth and Screen Flicker Math Analysis
// =============================================================================

struct BandwidthMetrics {
    size_t screen_w{1024};
    size_t screen_h{768};
    size_t bpp{4}; // 32 bits = 4 bytes
    size_t counter_w{80};
    size_t counter_h{16};

    size_t FullRedrawBytes() const noexcept {
        return screen_w * screen_h * bpp;
    }

    size_t CounterDiffBytes() const noexcept {
        return counter_w * counter_h * bpp;
    }

    double WasteFactor() const noexcept {
        return static_cast<double>(FullRedrawBytes()) / CounterDiffBytes();
    }

    double BandwidthMBs(double update_rate_hz) const noexcept {
        return (FullRedrawBytes() * update_rate_hz) / (1024.0 * 1024.0);
    }
};

} // namespace mikanos::study

// =============================================================================
// Main Verification Runner
// =============================================================================

int main() {
    using namespace mikanos::study;
    std::cout << "================================================================\n";
    std::cout << "MikanOS Chapter 10.3: High-Speed Counter & Screen Flicker Math\n";
    std::cout << "================================================================\n";

    // 1. Verify Window Size Dimensions (160x52 in day10c, reduced from 160x68 in day10b)
    const int kWinWidth = 160;
    const int kWinHeight = 52;
    Window window(kWinWidth, kWinHeight);
    DrawWindow(*window.Writer(), "Hello Window");

    assert(window.Width() == 160);
    assert(window.Height() == 52);
    std::cout << "[PASS] Window dimensions verified: " << window.Width() << "x" << window.Height() << " (160x52)\n";

    // Verify 3D Bevel Outer & Inner Borders
    assert(window.At({0, 0}) == ToColor(0xc6c6c6));
    assert(window.At({1, 1}) == ToColor(0xffffff));
    assert(window.At({kWinWidth - 1, kWinHeight - 1}) == ToColor(0x000000));
    assert(window.At({kWinWidth - 2, kWinHeight - 2}) == ToColor(0x848484));
    assert(window.At({3, 3}) == ToColor(0x000084)); // Title bar corner
    std::cout << "[PASS] 3D Bevel border colors verified\n";

    // 2. High-Speed Counter Simulation: Erase & Write Cycle
    char str[128];
    unsigned int count = 0;

    // Simulate 10 iterations of the counter loop (osbook_day10c)
    for (int step = 1; step <= 10; ++step) {
        ++count;
        std::snprintf(str, sizeof(str), "%010u", count);

        // FillRectangle(*main_window->Writer(), {24, 28}, {8 * 10, 16}, {0xc6, 0xc6, 0xc6});
        FillRectangle(*window.Writer(), {24, 28}, {8 * 10, 16}, {0xc6, 0xc6, 0xc6});

        // WriteString(*main_window->Writer(), {24, 28}, str, {0, 0, 0});
        WriteString(*window.Writer(), {24, 28}, str, {0, 0, 0});
    }

    assert(std::string(str) == "0000000010");
    std::cout << "[PASS] Counter formatted string verified: \"" << str << "\"\n";

    // Verify digit '0' at first char position {24, 28}: top border of '0' has pixels
    // kDigitFont[0][1] = 0x38 -> bits 2, 3, 4 are set -> x = 24 + 2, 3, 4
    assert(window.At({24 + 2, 28 + 1}) == ToColor(0x000000));
    assert(window.At({24 + 3, 28 + 1}) == ToColor(0x000000));
    assert(window.At({24 + 4, 28 + 1}) == ToColor(0x000000));
    assert(window.At({24 + 0, 28 + 0}) == ToColor(0xc6c6c6)); // Background around glyph
    std::cout << "[PASS] Counter glyph pixel rendering and background erase verified\n";

    // 3. Quantitative Bandwidth Waste Analysis
    BandwidthMetrics bm;
    std::cout << "\n--- Bandwidth & Redraw Waste Analysis ---\n";
    std::cout << "Screen resolution: " << bm.screen_w << "x" << bm.screen_h << " (32bpp)\n";
    std::cout << "Full screen redraw size: " << bm.FullRedrawBytes() << " bytes ("
              << std::fixed << std::setprecision(2) << bm.FullRedrawBytes() / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "Counter diff update size: " << bm.CounterDiffBytes() << " bytes ("
              << bm.CounterDiffBytes() / 1024.0 << " KB)\n";

    assert(bm.FullRedrawBytes() == 1024 * 768 * 4);
    assert(bm.CounterDiffBytes() == 80 * 16 * 4);
    assert(std::abs(bm.WasteFactor() - 614.4) < 0.1);

    std::cout << "Waste factor (Full / Diff): " << std::setprecision(1) << bm.WasteFactor() << "x\n";
    std::cout << "[PASS] Bandwidth waste factor strictly matches 614.4x\n";

    // 4. Busy-Loop Redraw Throughput Analysis
    std::cout << "\n--- Busy-Loop Memory Bus Load ---\n";
    for (double rate : {60.0, 500.0, 1000.0, 2000.0}) {
        std::cout << "  At " << std::setw(6) << static_cast<int>(rate) << " Hz: "
                  << std::setw(8) << std::setprecision(2) << bm.BandwidthMBs(rate) << " MB/s ("
                  << std::setprecision(2) << bm.BandwidthMBs(rate) / 1024.0 << " GB/s)\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "All C++20 Chapter 10.3 Assertions PASSED Successfully!\n";
    std::cout << "================================================================\n";
    return 0;
}
