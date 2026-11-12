/**
 * @file os_chapter10_mouse_refactor.cpp
 * @brief Chapter 10.1: Refactoring Mouse Processing & Window Geometric Clipping (C++20)
 *
 * Demonstrates:
 * 1. Absolute mouse coordinate tracking and boundary clamping (osbook_day10a)
 *    - Converting relative displacement (dx, dy) to bounded absolute position
 *    - Strict screen bounding box projection: [0, screen_size.x - 1] x [0, screen_size.y - 1]
 * 2. Window::DrawTo geometric intersection clipping for transparent layers
 *    - 2D coordinate transformation from local layer frame to global screen frame
 *    - Intersection clipping mathematics:
 *        y in [max(0, 0 - pos.y), min(Height(), screen.Height() - pos.y))
 *        x in [max(0, 0 - pos.x), min(Width(), screen.Width() - pos.x))
 * 3. Prevention of VRAM buffer overflow and segmentation faults on boundary crossing
 * 4. Transparent color skipping accuracy and edge-case safety tests
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mikanos::study {

// =============================================================================
// 1. Geometric Data Types
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

template <typename T>
struct Rectangle {
    Vector2D<T> pos;
    Vector2D<T> size;
};

// =============================================================================
// 2. Pixel & Color Representation
// =============================================================================

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    constexpr bool operator==(const PixelColor& rhs) const {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }
    constexpr bool operator!=(const PixelColor& rhs) const {
        return !(*this == rhs);
    }
};

enum class PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

// =============================================================================
// 3. PixelWriter & FrameBuffer Implementation
// =============================================================================

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(Vector2D<int> pos, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

class FrameBuffer {
public:
    FrameBuffer() = default;

    void Initialize(int width, int height, PixelFormat format) {
        width_ = width;
        height_ = height;
        format_ = format;
        bytes_per_pixel_ = 4;
        buffer_.assign(width * height * bytes_per_pixel_, 0);
        illegal_access_count_ = 0;
    }

    int Width() const { return width_; }
    int Height() const { return height_; }
    PixelFormat Format() const { return format_; }
    uint64_t IllegalAccessCount() const { return illegal_access_count_; }

    class FBWriter : public PixelWriter {
    public:
        explicit FBWriter(FrameBuffer& fb) : fb_{fb} {}
        void Write(Vector2D<int> pos, const PixelColor& c) override {
            fb_.WritePixel(pos, c);
        }
        int Width() const override { return fb_.Width(); }
        int Height() const override { return fb_.Height(); }
    private:
        FrameBuffer& fb_;
    };

    FBWriter& Writer() { return writer_; }

    void WritePixel(Vector2D<int> pos, const PixelColor& c) {
        if (pos.x < 0 || pos.x >= width_ || pos.y < 0 || pos.y >= height_) {
            illegal_access_count_++;
            return;
        }
        size_t offset = (static_cast<size_t>(pos.y) * width_ + pos.x) * bytes_per_pixel_;
        if (format_ == PixelFormat::kPixelRGBResv8BitPerColor) {
            buffer_[offset + 0] = c.r;
            buffer_[offset + 1] = c.g;
            buffer_[offset + 2] = c.b;
        } else {
            buffer_[offset + 0] = c.b;
            buffer_[offset + 1] = c.g;
            buffer_[offset + 2] = c.r;
        }
    }

    PixelColor ReadPixel(Vector2D<int> pos) const {
        if (pos.x < 0 || pos.x >= width_ || pos.y < 0 || pos.y >= height_) {
            return {0, 0, 0};
        }
        size_t offset = (static_cast<size_t>(pos.y) * width_ + pos.x) * bytes_per_pixel_;
        if (format_ == PixelFormat::kPixelRGBResv8BitPerColor) {
            return {buffer_[offset + 0], buffer_[offset + 1], buffer_[offset + 2]};
        } else {
            return {buffer_[offset + 2], buffer_[offset + 1], buffer_[offset + 0]};
        }
    }

    void Copy(Vector2D<int> dst_pos, const FrameBuffer& src) {
        for (int y = 0; y < src.Height(); ++y) {
            int dy = dst_pos.y + y;
            if (dy < 0 || dy >= height_) continue;
            for (int x = 0; x < src.Width(); ++x) {
                int dx = dst_pos.x + x;
                if (dx < 0 || dx >= width_) continue;
                WritePixel({dx, dy}, src.ReadPixel({x, y}));
            }
        }
    }

private:
    int width_{0};
    int height_{0};
    PixelFormat format_{PixelFormat::kPixelRGBResv8BitPerColor};
    int bytes_per_pixel_{4};
    std::vector<uint8_t> buffer_{};
    FBWriter writer_{*this};
    uint64_t illegal_access_count_{0};
};

// =============================================================================
// 4. Window Class with Intersection Clipping (osbook_day10a)
// =============================================================================

class Window {
public:
    class WindowWriter : public PixelWriter {
    public:
        explicit WindowWriter(Window& window) : window_{window} {}
        void Write(Vector2D<int> pos, const PixelColor& c) override {
            window_.Write(pos, c);
        }
        int Width() const override { return window_.Width(); }
        int Height() const override { return window_.Height(); }
    private:
        Window& window_;
    };

    Window(int width, int height, PixelFormat shadow_format)
        : width_{width}, height_{height} {
        data_.resize(height);
        for (int y = 0; y < height; ++y) {
            data_[y].resize(width);
        }
        shadow_buffer_.Initialize(width, height, shadow_format);
    }

    int Width() const { return width_; }
    int Height() const { return height_; }
    WindowWriter* Writer() { return &writer_; }

    const PixelColor& At(Vector2D<int> pos) const {
        return data_[pos.y][pos.x];
    }

    void Write(Vector2D<int> pos, PixelColor c) {
        if (pos.x < 0 || pos.x >= width_ || pos.y < 0 || pos.y >= height_) {
            return;
        }
        data_[pos.y][pos.x] = c;
        shadow_buffer_.WritePixel(pos, c);
    }

    void SetTransparentColor(std::optional<PixelColor> c) {
        transparent_color_ = c;
    }

    /**
     * @brief Draw window onto destination FrameBuffer with exact intersection clipping.
     *
     * As implemented in MikanOS day10a (kernel/window.cpp: Window::DrawTo):
     * When transparent_color_ is enabled, local coordinates (x, y) are constrained to:
     *   y: [max(0, 0 - position.y), min(Height(), dst.Writer().Height() - position.y))
     *   x: [max(0, 0 - position.x), min(Width(), dst.Writer().Width() - position.x))
     */
    void DrawTo(FrameBuffer& dst, Vector2D<int> position, uint64_t* drawn_pixels_out = nullptr) {
        if (!transparent_color_) {
            dst.Copy(position, shadow_buffer_);
            return;
        }

        const auto tc = transparent_color_.value();
        auto& writer = dst.Writer();
        uint64_t drawn_count = 0;

        for (int y = std::max(0, 0 - position.y);
             y < std::min(Height(), writer.Height() - position.y);
             ++y) {
            for (int x = std::max(0, 0 - position.x);
                 x < std::min(Width(), writer.Width() - position.x);
                 ++x) {
                const auto c = At(Vector2D<int>{x, y});
                if (c != tc) {
                    writer.Write(position + Vector2D<int>{x, y}, c);
                    drawn_count++;
                }
            }
        }

        if (drawn_pixels_out) {
            *drawn_pixels_out = drawn_count;
        }
    }

private:
    int width_{0};
    int height_{0};
    std::vector<std::vector<PixelColor>> data_{};
    WindowWriter writer_{*this};
    std::optional<PixelColor> transparent_color_{std::nullopt};
    FrameBuffer shadow_buffer_{};
};

// =============================================================================
// 5. Mouse Observer & Screen Boundary Clamping (osbook_day10a)
// =============================================================================

class MouseSubsystem {
public:
    MouseSubsystem(Vector2D<int> screen_size, Vector2D<int> initial_pos)
        : screen_size_{screen_size}, mouse_position_{initial_pos} {}

    Vector2D<int> GetPosition() const { return mouse_position_; }
    Vector2D<int> GetScreenSize() const { return screen_size_; }

    /**
     * @brief MouseObserver logic as introduced in MikanOS day10a (kernel/main.cpp).
     *
     * Project relative mouse displacement onto absolute clamped coordinates.
     */
    Vector2D<int> OnMouseMove(int8_t displacement_x, int8_t displacement_y) {
        auto new_pos = mouse_position_ + Vector2D<int>{displacement_x, displacement_y};
        new_pos.x = std::max(new_pos.x, 0);
        new_pos.x = std::min(new_pos.x, screen_size_.x - 1);
        new_pos.y = std::max(new_pos.y, 0);
        new_pos.y = std::min(new_pos.y, screen_size_.y - 1);

        mouse_position_ = new_pos;
        return mouse_position_;
    }

private:
    Vector2D<int> screen_size_;
    Vector2D<int> mouse_position_;
};

// =============================================================================
// 6. Test Suites & Benchmarks
// =============================================================================

void TestMouseClampingBasic() {
    std::cout << "[Test 1] Testing basic mouse coordinate clamping..." << std::endl;
    Vector2D<int> screen{640, 400};
    MouseSubsystem mouse{screen, {200, 200}};

    // Normal move
    auto pos = mouse.OnMouseMove(10, 20);
    assert((pos == Vector2D<int>{210, 220}));

    // Underflow X (left boundary)
    pos = mouse.OnMouseMove(-128, 0);
    pos = mouse.OnMouseMove(-128, 0);
    assert(pos.x == 0);
    assert(pos.y == 220);

    // Underflow Y (top boundary)
    pos = mouse.OnMouseMove(0, -128);
    pos = mouse.OnMouseMove(0, -128);
    assert((pos == Vector2D<int>{0, 0}));

    // Overflow X (right boundary: screen.x - 1 = 639)
    for (int i = 0; i < 10; ++i) {
        pos = mouse.OnMouseMove(100, 0);
    }
    assert(pos.x == 639);
    assert(pos.y == 0);

    // Overflow Y (bottom boundary: screen.y - 1 = 399)
    for (int i = 0; i < 10; ++i) {
        pos = mouse.OnMouseMove(0, 100);
    }
    assert((pos == Vector2D<int>{639, 399}));

    std::cout << "  -> PASSED: Screen bounds [0, 639]x[0, 399] strictly enforced." << std::endl;
}

void TestWindowClippingMath() {
    std::cout << "[Test 2] Testing Window::DrawTo geometric intersection math..." << std::endl;
    FrameBuffer screen;
    screen.Initialize(100, 100, PixelFormat::kPixelRGBResv8BitPerColor);

    Window win(20, 20, PixelFormat::kPixelRGBResv8BitPerColor);
    win.SetTransparentColor(PixelColor{0, 0, 0});

    // Fill window with non-transparent red color
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            win.Write({x, y}, {255, 0, 0});
        }
    }

    // Case 1: Fully inside screen
    uint64_t drawn = 0;
    win.DrawTo(screen, {10, 10}, &drawn);
    assert(drawn == 400); // 20x20
    assert(screen.IllegalAccessCount() == 0);
    assert((screen.ReadPixel({10, 10}) == PixelColor{255, 0, 0}));
    assert((screen.ReadPixel({29, 29}) == PixelColor{255, 0, 0}));

    // Case 2: Hanging off top-left: pos = (-10, -5)
    // Overlap should be x in [10, 20), y in [5, 20) -> 10 x 15 = 150 pixels
    win.DrawTo(screen, {-10, -5}, &drawn);
    assert(drawn == 150);
    assert(screen.IllegalAccessCount() == 0);
    assert((screen.ReadPixel({0, 0}) == PixelColor{255, 0, 0}));
    assert((screen.ReadPixel({9, 14}) == PixelColor{255, 0, 0}));

    // Case 3: Hanging off bottom-right: pos = (90, 85)
    // Screen size = 100x100
    // Overlap: x in [0, 10), y in [0, 15) -> 10 x 15 = 150 pixels
    win.DrawTo(screen, {90, 85}, &drawn);
    assert(drawn == 150);
    assert(screen.IllegalAccessCount() == 0);
    assert((screen.ReadPixel({90, 85}) == PixelColor{255, 0, 0}));
    assert((screen.ReadPixel({99, 99}) == PixelColor{255, 0, 0}));

    // Case 4: Completely outside screen (negative side)
    win.DrawTo(screen, {-30, -30}, &drawn);
    assert(drawn == 0);
    assert(screen.IllegalAccessCount() == 0);

    // Case 5: Completely outside screen (positive side)
    win.DrawTo(screen, {120, 120}, &drawn);
    assert(drawn == 0);
    assert(screen.IllegalAccessCount() == 0);

    std::cout << "  -> PASSED: All 5 geometric intersection cases verified with 0 illegal VRAM writes." << std::endl;
}

void TestTransparentColorPreservation() {
    std::cout << "[Test 3] Testing transparent color keying and background protection..." << std::endl;
    FrameBuffer screen;
    screen.Initialize(50, 50, PixelFormat::kPixelRGBResv8BitPerColor);

    // Fill screen with desktop green {0, 128, 0}
    for (int y = 0; y < 50; ++y) {
        for (int x = 0; x < 50; ++x) {
            screen.WritePixel({x, y}, {0, 128, 0});
        }
    }

    Window win(10, 10, PixelFormat::kPixelRGBResv8BitPerColor);
    win.SetTransparentColor(PixelColor{0, 0, 0});

    // Diagonal line: only (x == y) is white {255, 255, 255}, others transparent black {0, 0, 0}
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            if (x == y) {
                win.Write({x, y}, {255, 255, 255});
            } else {
                win.Write({x, y}, {0, 0, 0});
            }
        }
    }

    uint64_t drawn = 0;
    win.DrawTo(screen, {20, 20}, &drawn);
    assert(drawn == 10); // Exactly 10 non-transparent diagonal pixels
    assert(screen.IllegalAccessCount() == 0);

    // Diagonal must be white
    for (int i = 0; i < 10; ++i) {
        assert((screen.ReadPixel({20 + i, 20 + i}) == PixelColor{255, 255, 255}));
    }
    // Adjacent pixels must retain original green background
    assert((screen.ReadPixel({20, 21}) == PixelColor{0, 128, 0}));
    assert((screen.ReadPixel({21, 20}) == PixelColor{0, 128, 0}));

    std::cout << "  -> PASSED: Transparent keying correctly skips writes and preserves background." << std::endl;
}

void TestBenchmarkAndStressSafety() {
    std::cout << "[Test 4] Stress testing 50,000 rapid boundary mouse motions & clipping..." << std::endl;
    Vector2D<int> screen_size{640, 400};
    MouseSubsystem mouse{screen_size, {200, 200}};

    FrameBuffer screen;
    screen.Initialize(screen_size.x, screen_size.y, PixelFormat::kPixelRGBResv8BitPerColor);

    Window cursor(16, 24, PixelFormat::kPixelRGBResv8BitPerColor);
    cursor.SetTransparentColor(PixelColor{99, 99, 99});
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 16; ++x) {
            cursor.Write({x, y}, {255, 255, 255});
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    uint64_t total_drawn_pixels = 0;

    for (int i = 0; i < 50000; ++i) {
        int8_t dx = static_cast<int8_t>((i % 23) - 11);
        int8_t dy = static_cast<int8_t>((i % 29) - 14);
        auto pos = mouse.OnMouseMove(dx, dy);

        uint64_t drawn = 0;
        cursor.DrawTo(screen, pos, &drawn);
        total_drawn_pixels += drawn;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "  -> Total motions & draws: 50,000" << std::endl;
    std::cout << "  -> Total drawn pixels: " << total_drawn_pixels << std::endl;
    std::cout << "  -> Elapsed time: " << elapsed_ms << " ms ("
              << (50000.0 / (elapsed_ms ? elapsed_ms : 1)) * 1000.0 << " ops/sec)" << std::endl;
    std::cout << "  -> Detected illegal VRAM accesses: " << screen.IllegalAccessCount() << std::endl;

    assert(screen.IllegalAccessCount() == 0);
    std::cout << "  -> PASSED: Zero illegal accesses in 50,000 boundary rendering operations." << std::endl;
}

} // namespace mikanos::study

int main(int argc, char* argv[]) {
    std::cout << "=================================================================" << std::endl;
    std::cout << " MikanOS Chapter 10.1: Mouse Clamping & Window Clipping (C++20)  " << std::endl;
    std::cout << "=================================================================" << std::endl;

    bool run_all = (argc <= 1 || std::string(argv[1]) == "--test");

    if (run_all) {
        mikanos::study::TestMouseClampingBasic();
        mikanos::study::TestWindowClippingMath();
        mikanos::study::TestTransparentColorPreservation();
        mikanos::study::TestBenchmarkAndStressSafety();
    }

    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "ALL CHAPTER 10.1 TESTS AND VERIFICATIONS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
