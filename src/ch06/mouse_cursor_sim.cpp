/**
 * @file mouse_cursor_sim.cpp
 * @brief Chapter 6.1: Mouse Cursor & Desktop GUI Simulation (MikanOS day06a)
 *
 * Implements Vector2D<T>, DrawRectangle, FillRectangle, mouse_cursor_shape sprite,
 * and complete day06a desktop rendering with standalone C++20 verification.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <string>
#include <memory>

// =============================================================================
// 1. Vector2D Template & Pixel Color Data Structures
// =============================================================================

template <typename T>
struct Vector2D {
    T x, y;

    constexpr Vector2D() : x{0}, y{0} {}
    constexpr Vector2D(T x_, T y_) : x{x_}, y{y_} {}

    constexpr Vector2D operator+(const Vector2D& rhs) const {
        return Vector2D{x + rhs.x, y + rhs.y};
    }

    constexpr Vector2D operator-(const Vector2D& rhs) const {
        return Vector2D{x - rhs.x, y - rhs.y};
    }

    constexpr bool operator==(const Vector2D& rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    constexpr bool operator!=(const Vector2D& rhs) const {
        return !(*this == rhs);
    }
};

struct PixelColor {
    uint8_t r, g, b;

    constexpr bool operator==(const PixelColor& rhs) const {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }

    constexpr bool operator!=(const PixelColor& rhs) const {
        return !(*this == rhs);
    }
};

// =============================================================================
// 2. FrameBufferConfig and PixelWriter Classes
// =============================================================================

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    PixelFormat pixel_format;
};

class PixelWriter {
public:
    PixelWriter(const FrameBufferConfig& config) : config_{config} {}
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;

protected:
    uint8_t* PixelAt(int x, int y) {
        return config_.frame_buffer + 4 * (config_.pixels_per_scan_line * y + x);
    }

private:
    const FrameBufferConfig& config_;
};

class RGBResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;
    virtual void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
    }
};

class BGRResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;
    virtual void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
    }
};

// =============================================================================
// 3. Rectangle Drawing Algorithms
// =============================================================================

void DrawRectangle(PixelWriter& writer, const Vector2D<int>& pos,
                   const Vector2D<int>& size, const PixelColor& c) {
    for (int dx = 0; dx < size.x; ++dx) {
        writer.Write(pos.x + dx, pos.y, c);
        writer.Write(pos.x + dx, pos.y + size.y - 1, c);
    }
    for (int dy = 1; dy < size.y - 1; ++dy) {
        writer.Write(pos.x, pos.y + dy, c);
        writer.Write(pos.x + size.x - 1, pos.y + dy, c);
    }
}

void FillRectangle(PixelWriter& writer, const Vector2D<int>& pos,
                   const Vector2D<int>& size, const PixelColor& c) {
    for (int dy = 0; dy < size.y; ++dy) {
        for (int dx = 0; dx < size.x; ++dx) {
            writer.Write(pos.x + dx, pos.y + dy, c);
        }
    }
}

// =============================================================================
// 4. Mouse Cursor Sprite (15x24) and Desktop UI Colors
// =============================================================================

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

const PixelColor kDesktopBGColor{45, 118, 237};
const PixelColor kDesktopFGColor{255, 255, 255};
const PixelColor kTaskbarColor{1, 8, 17};
const PixelColor kStartButtonColor{80, 80, 80};
const PixelColor kStartIconColor{160, 160, 160};

struct MouseCursorStats {
    int black_pixels = 0;
    int white_pixels = 0;
    int transparent_pixels = 0;
};

MouseCursorStats DrawMouseCursor(PixelWriter& writer, const Vector2D<int>& pos) {
    MouseCursorStats stats;
    for (int dy = 0; dy < kMouseCursorHeight; ++dy) {
        for (int dx = 0; dx < kMouseCursorWidth; ++dx) {
            char ch = mouse_cursor_shape[dy][dx];
            if (ch == '@') {
                writer.Write(pos.x + dx, pos.y + dy, {0, 0, 0});
                stats.black_pixels++;
            } else if (ch == '.') {
                writer.Write(pos.x + dx, pos.y + dy, {255, 255, 255});
                stats.white_pixels++;
            } else {
                stats.transparent_pixels++;
            }
        }
    }
    return stats;
}

// =============================================================================
// 5. Verification & Standalone Test Runner
// =============================================================================

void RunTests() {
    std::cout << "[TEST] 1. Vector2D<T> Memory Layout & Operations..." << std::endl;
    static_assert(sizeof(Vector2D<int>) == 8, "Vector2D<int> must be exactly 8 bytes (no padding or vptr)");
    static_assert(sizeof(Vector2D<uint32_t>) == 8, "Vector2D<uint32_t> must be 8 bytes");
    static_assert(sizeof(Vector2D<double>) == 16, "Vector2D<double> must be 16 bytes");

    Vector2D<int> v1{10, 20};
    Vector2D<int> v2{5, 15};
    Vector2D<int> v_sum = v1 + v2;
    Vector2D<int> v_diff = v1 - v2;

    assert(v_sum.x == 15 && v_sum.y == 35);
    assert(v_diff.x == 5 && v_diff.y == 5);
    assert(v1 == Vector2D<int>(10, 20));
    assert(v1 != v2);

    std::cout << "[TEST] 2. FrameBuffer and PixelWriter Allocation..." << std::endl;
    const int kWidth = 800;
    const int kHeight = 600;
    const int kStride = 800;
    std::vector<uint8_t> fb_memory(kStride * kHeight * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = fb_memory.data(),
        .pixels_per_scan_line = static_cast<uint32_t>(kStride),
        .horizontal_resolution = static_cast<uint32_t>(kWidth),
        .vertical_resolution = static_cast<uint32_t>(kHeight),
        .pixel_format = kPixelRGBResv8BitPerColor
    };

    RGBResv8BitPerColorPixelWriter writer{config};

    auto GetPixel = [&](int x, int y) -> PixelColor {
        size_t offset = 4 * (kStride * y + x);
        return PixelColor{fb_memory[offset], fb_memory[offset + 1], fb_memory[offset + 2]};
    };

    std::cout << "[TEST] 3. FillRectangle & DrawRectangle Verification..." << std::endl;
    // Fill desktop background
    FillRectangle(writer, {0, 0}, {kWidth, kHeight - 50}, kDesktopBGColor);
    assert(GetPixel(0, 0) == kDesktopBGColor);
    assert(GetPixel(400, 250) == kDesktopBGColor);
    assert(GetPixel(kWidth - 1, kHeight - 51) == kDesktopBGColor);

    // Verify taskbar
    FillRectangle(writer, {0, kHeight - 50}, {kWidth, 50}, kTaskbarColor);
    assert(GetPixel(0, kHeight - 50) == kTaskbarColor);
    assert(GetPixel(kWidth - 1, kHeight - 1) == kTaskbarColor);

    // Verify start button
    FillRectangle(writer, {0, kHeight - 50}, {kWidth / 5, 50}, kStartButtonColor);
    assert(GetPixel(0, kHeight - 50) == kStartButtonColor);
    assert(GetPixel(kWidth / 5 - 1, kHeight - 1) == kStartButtonColor);
    assert(GetPixel(kWidth / 5, kHeight - 50) == kTaskbarColor); // Boundary check

    // Verify start icon (outline only)
    Vector2D<int> icon_pos{10, kHeight - 40};
    Vector2D<int> icon_size{30, 30};
    DrawRectangle(writer, icon_pos, icon_size, kStartIconColor);

    // Outline should be start icon color
    assert(GetPixel(icon_pos.x, icon_pos.y) == kStartIconColor);
    assert(GetPixel(icon_pos.x + icon_size.x - 1, icon_pos.y) == kStartIconColor);
    assert(GetPixel(icon_pos.x, icon_pos.y + icon_size.y - 1) == kStartIconColor);
    assert(GetPixel(icon_pos.x + icon_size.x - 1, icon_pos.y + icon_size.y - 1) == kStartIconColor);
    // Interior must remain start button color (not overwritten!)
    assert(GetPixel(icon_pos.x + 5, icon_pos.y + 5) == kStartButtonColor);

    std::cout << "[TEST] 4. Mouse Cursor Sprite & Color Keying Verification..." << std::endl;
    Vector2D<int> cursor_pos{200, 100};
    MouseCursorStats stats = DrawMouseCursor(writer, cursor_pos);

    assert(stats.black_pixels > 0);
    assert(stats.white_pixels > 0);
    assert(stats.transparent_pixels > 0);
    assert(stats.black_pixels + stats.white_pixels + stats.transparent_pixels == kMouseCursorWidth * kMouseCursorHeight);

    // Tip of arrow at (200, 100) must be '@' (black)
    assert(GetPixel(200, 100) == PixelColor({0, 0, 0}));
    // Inside arrow (e.g. at (202, 103)) must be '.' (white)
    assert(GetPixel(202, 103) == PixelColor({255, 255, 255}));
    // Transparent pixel (e.g. at (200 + 14, 100)) must remain desktop background!
    assert(GetPixel(200 + 14, 100) == kDesktopBGColor);

    std::cout << "[TEST] 5. BGR PixelWriter Output Check..." << std::endl;
    config.pixel_format = kPixelBGRResv8BitPerColor;
    BGRResv8BitPerColorPixelWriter bgr_writer{config};
    bgr_writer.Write(0, 0, PixelColor{0x12, 0x34, 0x56});
    // In memory: p[0]=B, p[1]=G, p[2]=R
    assert(fb_memory[0] == 0x56);
    assert(fb_memory[1] == 0x34);
    assert(fb_memory[2] == 0x12);

    std::cout << "[SUCCESS] All Chapter 6.1 mouse cursor & desktop simulation tests PASSED!" << std::endl;
}

int main(int argc, char* argv[]) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    if (test_mode) {
        RunTests();
        return 0;
    }

    std::cout << "=== MikanOS Chapter 6.1: Mouse Cursor & Desktop GUI Simulator ===" << std::endl;
    std::cout << "Vector2D<int> size: " << sizeof(Vector2D<int>) << " bytes (Zero overhead)" << std::endl;
    std::cout << "Mouse cursor sprite dimensions: " << kMouseCursorWidth << "x" << kMouseCursorHeight << " pixels" << std::endl;
    std::cout << "Desktop background: RGB(45, 118, 237)" << std::endl;
    std::cout << "Taskbar: (0, 550), 800x50, RGB(1, 8, 17)" << std::endl;
    std::cout << "Start button: (0, 550), 160x50, RGB(80, 80, 80)" << std::endl;
    std::cout << "Start icon border: (10, 560), 30x30, RGB(160, 160, 160)" << std::endl;
    std::cout << "Mouse cursor position: (200, 100)" << std::endl;
    std::cout << "Pass --test to execute verification suite." << std::endl;
    return 0;
}
