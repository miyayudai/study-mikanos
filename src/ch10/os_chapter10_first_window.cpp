/**
 * @file os_chapter10_first_window.cpp
 * @brief Chapter 10.2: First GUI Window Decoration & 3D Bevel Math (C++20)
 *
 * Demonstrates:
 * 1. 3D Bevel Window Border Drawing (osbook_day10b: DrawWindow)
 *    - Geometric optics of top-left light source model L = (-1, -1, 1)
 *    - 1-pixel highlight/shadow double border composition
 *    - Color mappings:
 *        Outer top/left: 0xc6c6c6 (Indirect light highlight)
 *        Inner top/left: 0xffffff (Direct light highlight)
 *        Inner bottom/right: 0x848484 (Subtle drop shadow)
 *        Outer bottom/right: 0x000000 (Deep drop shadow)
 * 2. Classic Navy Blue Title Bar (0x000084) and Title String Alignment
 * 3. Close Button (16x14) 4-color Palette Bitmap Decoding & Placement
 * 4. Window Creation, Client Area Separation, and Layer Compositing
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

namespace mikanos::study {

// =============================================================================
// 1. Geometric Data Types and Colors
// =============================================================================

template <typename T>
struct Vector2D {
    T x{0};
    T y{0};

    constexpr Vector2D() = default;
    constexpr Vector2D(T x, T y) : x{x}, y{y} {}

    constexpr Vector2D operator+(const Vector2D& rhs) const { return {x + rhs.x, y + rhs.y}; }
    constexpr Vector2D operator-(const Vector2D& rhs) const { return {x - rhs.x, y - rhs.y}; }
    Vector2D& operator+=(const Vector2D& rhs) { x += rhs.x; y += rhs.y; return *this; }
    Vector2D& operator-=(const Vector2D& rhs) { x -= rhs.x; y -= rhs.y; return *this; }
    constexpr bool operator==(const Vector2D& rhs) const = default;
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    constexpr bool operator==(const PixelColor& rhs) const = default;
    constexpr bool operator!=(const PixelColor& rhs) const = default;
};

constexpr PixelColor ToColor(uint32_t c) {
    return {
        static_cast<uint8_t>((c >> 16) & 0xff),
        static_cast<uint8_t>((c >> 8) & 0xff),
        static_cast<uint8_t>(c & 0xff)
    };
}

constexpr PixelColor kColorWhite     = ToColor(0xffffff);
constexpr PixelColor kColorLightGray = ToColor(0xc6c6c6);
constexpr PixelColor kColorDarkGray  = ToColor(0x848484);
constexpr PixelColor kColorBlack     = ToColor(0x000000);
constexpr PixelColor kColorTitleBar  = ToColor(0x000084);

// =============================================================================
// 2. PixelWriter Interface and Memory Implementation
// =============================================================================

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(Vector2D<int> pos, const PixelColor& c) = 0;
    virtual PixelColor Read(Vector2D<int> pos) const = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

class MemoryPixelWriter : public PixelWriter {
public:
    MemoryPixelWriter(int width, int height)
        : width_{width}, height_{height}, buffer_(width * height, kColorLightGray) {}

    void Write(Vector2D<int> pos, const PixelColor& c) override {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            buffer_[pos.y * width_ + pos.x] = c;
        }
    }

    PixelColor Read(Vector2D<int> pos) const override {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            return buffer_[pos.y * width_ + pos.x];
        }
        return {0, 0, 0};
    }

    int Width() const override { return width_; }
    int Height() const override { return height_; }

    const std::vector<PixelColor>& Buffer() const { return buffer_; }

private:
    int width_{0};
    int height_{0};
    std::vector<PixelColor> buffer_;
};

void FillRectangle(PixelWriter& writer, Vector2D<int> pos, Vector2D<int> size, const PixelColor& c) {
    for (int dy = 0; dy < size.y; ++dy) {
        for (int dx = 0; dx < size.x; ++dx) {
            writer.Write(pos + Vector2D<int>{dx, dy}, c);
        }
    }
}

// Minimal ASCII 8x16 font rendering stub for C++ standalone testing
void WriteAsciiChar(PixelWriter& writer, Vector2D<int> pos, char c, const PixelColor& color) {
    // 8x16 simple bounding box / glyph approximation
    // Cross / dot indicator for non-space characters
    if (c != ' ') {
        for (int dy = 2; dy < 14; ++dy) {
            writer.Write(pos + Vector2D<int>{1, dy}, color);
        }
        for (int dx = 1; dx < 7; ++dx) {
            writer.Write(pos + Vector2D<int>{dx, 2}, color);
            writer.Write(pos + Vector2D<int>{dx, 8}, color);
        }
    }
}

void WriteString(PixelWriter& writer, Vector2D<int> pos, std::string_view s, const PixelColor& color) {
    for (size_t i = 0; i < s.size(); ++i) {
        WriteAsciiChar(writer, pos + Vector2D<int>{static_cast<int>(i * 8), 0}, s[i], color);
    }
}

// =============================================================================
// 3. MikanOS Close Button Bitmap (16x14)
// =============================================================================

constexpr int kCloseButtonWidth = 16;
constexpr int kCloseButtonHeight = 14;

constexpr const char kCloseButtonPattern[kCloseButtonHeight][kCloseButtonWidth + 1] = {
    "...............@",
    ".:::::::::::::$@",
    ".:::::::::::::$@",
    ".:::@@::::@@::$@",
    ".::::@@::@@:::$@",
    ".:::::@@@@::::$@",
    ".::::::@@:::::$@",
    ".:::::@@@@::::$@",
    ".::::@@::@@:::$@",
    ".:::@@::::@@::$@",
    ".:::::::::::::$@",
    ".:::::::::::::$@",
    ".$$$$$$$$$$$$$$@",
    "@@@@@@@@@@@@@@@@",
};

// =============================================================================
// 4. DrawWindow Implementation (osbook_day10b)
// =============================================================================

void DrawWindow(PixelWriter& writer, const char* title) {
    auto fill_rect = [&writer](Vector2D<int> pos, Vector2D<int> size, uint32_t c) {
        FillRectangle(writer, pos, size, ToColor(c));
    };
    const auto win_w = writer.Width();
    const auto win_h = writer.Height();

    // 1. 3D Bevel Outer & Inner Borders
    fill_rect({0, 0},         {win_w, 1},             0xc6c6c6); // Outer top
    fill_rect({1, 1},         {win_w - 2, 1},         0xffffff); // Inner top
    fill_rect({0, 0},         {1, win_h},             0xc6c6c6); // Outer left
    fill_rect({1, 1},         {1, win_h - 2},         0xffffff); // Inner left
    fill_rect({win_w - 2, 1}, {1, win_h - 2},         0x848484); // Inner right
    fill_rect({win_w - 1, 0}, {1, win_h},             0x000000); // Outer right
    fill_rect({2, 2},         {win_w - 4, win_h - 4}, 0xc6c6c6); // Body background
    fill_rect({3, 3},         {win_w - 6, 18},        0x000084); // Title bar
    fill_rect({1, win_h - 2}, {win_w - 2, 1},         0x848484); // Inner bottom
    fill_rect({0, win_h - 1}, {win_w, 1},             0x000000); // Outer bottom

    // 2. Title String (at x=24, y=4)
    WriteString(writer, {24, 4}, title, ToColor(0xffffff));

    // 3. Close Button (at top-right: (win_w - 5 - kCloseButtonWidth, 5))
    for (int y = 0; y < kCloseButtonHeight; ++y) {
        for (int x = 0; x < kCloseButtonWidth; ++x) {
            PixelColor c = ToColor(0xffffff);
            char sym = kCloseButtonPattern[y][x];
            if (sym == '@') {
                c = ToColor(0x000000);
            } else if (sym == '$') {
                c = ToColor(0x848484);
            } else if (sym == ':') {
                c = ToColor(0xc6c6c6);
            }
            writer.Write({win_w - 5 - kCloseButtonWidth + x, 5 + y}, c);
        }
    }
}

// =============================================================================
// 5. Window Class Representation
// =============================================================================

class Window {
public:
    Window(int width, int height)
        : width_{width}, height_{height}, writer_{width, height} {}

    PixelWriter& Writer() { return writer_; }
    const PixelWriter& Writer() const { return writer_; }

    int Width() const { return width_; }
    int Height() const { return height_; }

    void SetTransparentColor(std::optional<PixelColor> tc) {
        transparent_color_ = tc;
    }

    std::optional<PixelColor> TransparentColor() const {
        return transparent_color_;
    }

    PixelColor At(Vector2D<int> pos) const {
        return writer_.Read(pos);
    }

    // DrawTo with 2D geometric clipping (from Section 10.1)
    void DrawTo(PixelWriter& dst, Vector2D<int> position) const {
        if (!transparent_color_) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    dst.Write(position + Vector2D<int>{x, y}, At({x, y}));
                }
            }
            return;
        }

        const auto tc = *transparent_color_;
        int y_start = std::max(0, -position.y);
        int y_end   = std::min(height_, dst.Height() - position.y);
        int x_start = std::max(0, -position.x);
        int x_end   = std::min(width_, dst.Width() - position.x);

        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                const auto c = At({x, y});
                if (c != tc) {
                    dst.Write(position + Vector2D<int>{x, y}, c);
                }
            }
        }
    }

private:
    int width_{0};
    int height_{0};
    MemoryPixelWriter writer_;
    std::optional<PixelColor> transparent_color_{std::nullopt};
};

} // namespace mikanos::study

// =============================================================================
// 6. Test Driver & Verification Suite
// =============================================================================

int main() {
    using namespace mikanos::study;

    std::cout << "=================================================================\n";
    std::cout << "[Chapter 10.2] First GUI Window Decoration & 3D Bevel Math (C++20)\n";
    std::cout << "=================================================================\n\n";

    constexpr int kWinW = 160;
    constexpr int kWinH = 68;

    auto window = std::make_shared<Window>(kWinW, kWinH);
    DrawWindow(window->Writer(), "Hello Window");
    WriteString(window->Writer(), {24, 28}, "Welcome to", {0, 0, 0});
    WriteString(window->Writer(), {24, 44}, " MikanOS world!", {0, 0, 0});

    // Test 1: Border pixels verification (Highlight & Shadow edges)
    std::cout << "1. Validating 3D Bevel double border pixels...\n";
    assert(window->At({0, 0}) == kColorLightGray);              // Outer top-left corner
    assert(window->At({kWinW - 1, 0}) == kColorBlack);          // Outer top-right
    assert(window->At({0, kWinH - 1}) == kColorBlack);          // Outer bottom-left
    assert(window->At({kWinW - 1, kWinH - 1}) == kColorBlack);  // Outer bottom-right corner

    assert(window->At({1, 1}) == kColorWhite);                  // Inner top-left highlight
    assert(window->At({kWinW - 2, 1}) == kColorDarkGray);       // Inner top-right shadow
    assert(window->At({1, kWinH - 2}) == kColorDarkGray);       // Inner bottom-left shadow
    assert(window->At({kWinW - 2, kWinH - 2}) == kColorDarkGray); // Inner bottom-right shadow
    assert(window->At({2, 2}) == kColorLightGray);              // Window body background
    std::cout << "   [PASS] 3D Bevel highlight and shadow pixels match specifications.\n";

    // Test 2: Title bar verification (3, 3) 154x18
    std::cout << "2. Validating classic navy blue title bar (0x000084)...\n";
    assert(window->At({3, 3}) == kColorTitleBar);
    assert(window->At({10, 10}) == kColorTitleBar);
    assert(window->At({kWinW - 22, 15}) == kColorTitleBar);
    std::cout << "   [PASS] Title bar region correctly filled with 0x000084.\n";

    // Test 3: Close button verification (at x = kWinW - 5 - 16 = 139, y = 5)
    std::cout << "3. Validating 16x14 Close Button bitmap rendering...\n";
    int btn_x = kWinW - 5 - kCloseButtonWidth;
    int btn_y = 5;

    // Top-left pixel of close button is '.' -> 0xffffff
    assert(window->At({btn_x, btn_y}) == kColorWhite);
    // Top-right pixel of close button is '@' -> 0x000000
    assert(window->At({btn_x + kCloseButtonWidth - 1, btn_y}) == kColorBlack);
    // Bottom-left pixel of close button is '@' -> 0x000000
    assert(window->At({btn_x, btn_y + kCloseButtonHeight - 1}) == kColorBlack);
    // Cross center pixels (e.g., row 6, col 7 & 8 -> '@' -> 0x000000)
    assert(window->At({btn_x + 7, btn_y + 6}) == kColorBlack);
    assert(window->At({btn_x + 8, btn_y + 6}) == kColorBlack);
    // Inner shadow '$' -> 0x848484 at row 12, col 1
    assert(window->At({btn_x + 1, btn_y + 12}) == kColorDarkGray);
    std::cout << "   [PASS] Close button 4-color palette and glyph matched.\n";

    // Test 4: Window compositing onto desktop screen
    std::cout << "4. Validating window composition onto desktop screen...\n";
    MemoryPixelWriter desktop(640, 400);
    FillRectangle(desktop, {0, 0}, {640, 400}, ToColor(0x2d7689)); // Desktop teal

    Vector2D<int> win_pos{300, 100};
    window->DrawTo(desktop, win_pos);

    // Verify desktop received the window pixels
    assert(desktop.Read(win_pos + Vector2D<int>{1, 1}) == kColorWhite);
    assert(desktop.Read(win_pos + Vector2D<int>{10, 10}) == kColorTitleBar);
    assert(desktop.Read({50, 50}) == ToColor(0x2d7689)); // Background untouched
    std::cout << "   [PASS] Window composited onto desktop at (300, 100) successfully.\n";

    std::cout << "\n=================================================================\n";
    std::cout << "All Section 10.2 C++20 Verification Tests Passed Successfully!\n";
    std::cout << "=================================================================\n";
    return 0;
}
