/**
 * @file os_chapter9_optimizing_layer_redraw.cpp
 * @brief Chapter 9.5: Optimizing Layer Compositing with FrameBuffer & Line-by-Line memcpy (C++20)
 *
 * Demonstrates:
 * 1. FrameBuffer class abstraction unifying hardware VRAM and off-screen memory buffers
 * 2. Window class with shadow buffer (FrameBuffer shadow_buffer_)
 * 3. Fast-path compositing: replacing O(W * H) individual PixelWriter::Write virtual calls
 *    with line-by-line memcpy (FrameBuffer::Copy) for opaque layers (e.g. desktop background)
 * 4. 2D geometric clipping mathematics (handling negative offsets, screen boundaries, and partial overlap)
 * 5. High-precision performance benchmark measuring speedup of memcpy over pixel-by-pixel compositing
 * 6. Full layer compositing integration test verifying painter's algorithm correctness
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mikanos::study {

// =============================================================================
// 1. Core Geometric and Color Data Types
// =============================================================================

template <typename T>
struct Vector2D {
    T x, y;

    constexpr Vector2D() : x{0}, y{0} {}
    constexpr Vector2D(T x, T y) : x{x}, y{y} {}

    constexpr Vector2D operator+(const Vector2D& rhs) const {
        return Vector2D{x + rhs.x, y + rhs.y};
    }
    constexpr Vector2D operator-(const Vector2D& rhs) const {
        return Vector2D{x - rhs.x, y - rhs.y};
    }
    Vector2D& operator+=(const Vector2D& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
    bool operator==(const Vector2D& rhs) const {
        return x == rhs.x && y == rhs.y;
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

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

inline int BitsPerPixel(PixelFormat format) {
    switch (format) {
        case kPixelRGBResv8BitPerColor: return 32;
        case kPixelBGRResv8BitPerColor: return 32;
    }
    return -1;
}

struct FrameBufferConfig {
    uint8_t* frame_buffer{nullptr};
    uint32_t pixels_per_scan_line{0};
    uint32_t horizontal_resolution{0};
    uint32_t vertical_resolution{0};
    PixelFormat pixel_format{kPixelRGBResv8BitPerColor};
};

// =============================================================================
// 2. PixelWriter & Concrete Implementations
// =============================================================================

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(Vector2D<int> pos, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

class FrameBufferWriter : public PixelWriter {
public:
    explicit FrameBufferWriter(const FrameBufferConfig& config)
        : config_{config} {}
    virtual ~FrameBufferWriter() = default;

    int Width() const override { return config_.horizontal_resolution; }
    int Height() const override { return config_.vertical_resolution; }

protected:
    uint8_t* PixelAt(Vector2D<int> pos) {
        return config_.frame_buffer + 4 * (config_.pixels_per_scan_line * pos.y + pos.x);
    }
    const FrameBufferConfig& config_;
};

class RGBResv8BitPerColorPixelWriter : public FrameBufferWriter {
public:
    using FrameBufferWriter::FrameBufferWriter;

    void Write(Vector2D<int> pos, const PixelColor& c) override {
        if (pos.x < 0 || pos.x >= static_cast<int>(config_.horizontal_resolution) ||
            pos.y < 0 || pos.y >= static_cast<int>(config_.vertical_resolution)) {
            return;
        }
        auto p = PixelAt(pos);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0;
    }
};

class BGRResv8BitPerColorPixelWriter : public FrameBufferWriter {
public:
    using FrameBufferWriter::FrameBufferWriter;

    void Write(Vector2D<int> pos, const PixelColor& c) override {
        if (pos.x < 0 || pos.x >= static_cast<int>(config_.horizontal_resolution) ||
            pos.y < 0 || pos.y >= static_cast<int>(config_.vertical_resolution)) {
            return;
        }
        auto p = PixelAt(pos);
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0;
    }
};

// =============================================================================
// 3. FrameBuffer Class (Matching MikanOS kernel/frame_buffer.cpp)
// =============================================================================

struct CopyStats {
    int lines_copied{0};
    size_t bytes_copied{0};
    bool clipped{false};
};

class FrameBuffer {
public:
    FrameBuffer() = default;
    ~FrameBuffer() = default;
    FrameBuffer(const FrameBuffer&) = delete;
    FrameBuffer& operator=(const FrameBuffer&) = delete;

    bool Initialize(const FrameBufferConfig& config) {
        config_ = config;
        const auto bpp = BitsPerPixel(config_.pixel_format);
        if (bpp <= 0) {
            return false;
        }

        if (config_.frame_buffer) {
            buffer_.resize(0);
        } else {
            size_t size = static_cast<size_t>((bpp + 7) / 8) *
                          config_.horizontal_resolution * config_.vertical_resolution;
            buffer_.resize(size, 0);
            config_.frame_buffer = buffer_.data();
            config_.pixels_per_scan_line = config_.horizontal_resolution;
        }

        switch (config_.pixel_format) {
            case kPixelRGBResv8BitPerColor:
                writer_ = std::make_unique<RGBResv8BitPerColorPixelWriter>(config_);
                break;
            case kPixelBGRResv8BitPerColor:
                writer_ = std::make_unique<BGRResv8BitPerColorPixelWriter>(config_);
                break;
            default:
                return false;
        }
        return true;
    }

    CopyStats Copy(Vector2D<int> pos, const FrameBuffer& src) {
        CopyStats stats{};
        if (config_.pixel_format != src.config_.pixel_format) {
            return stats;
        }

        const auto bpp = BitsPerPixel(config_.pixel_format);
        if (bpp <= 0) {
            return stats;
        }

        const int dst_width = static_cast<int>(config_.horizontal_resolution);
        const int dst_height = static_cast<int>(config_.vertical_resolution);
        const int src_width = static_cast<int>(src.config_.horizontal_resolution);
        const int src_height = static_cast<int>(src.config_.vertical_resolution);

        const int copy_start_dst_x = std::max(pos.x, 0);
        const int copy_start_dst_y = std::max(pos.y, 0);
        const int copy_end_dst_x = std::min(pos.x + src_width, dst_width);
        const int copy_end_dst_y = std::min(pos.y + src_height, dst_height);

        if (copy_start_dst_x >= copy_end_dst_x || copy_start_dst_y >= copy_end_dst_y) {
            stats.clipped = true;
            return stats;
        }

        const auto bytes_per_pixel = static_cast<size_t>((bpp + 7) / 8);
        const auto bytes_per_copy_line = bytes_per_pixel * (copy_end_dst_x - copy_start_dst_x);

        stats.clipped = (copy_start_dst_x != pos.x || copy_start_dst_y != pos_y(pos) ||
                         copy_end_dst_x != pos.x + src_width || copy_end_dst_y != pos.y + src_height);

        const int src_start_x = copy_start_dst_x - pos.x;
        const int src_start_y = copy_start_dst_y - pos.y;

        uint8_t* dst_buf = config_.frame_buffer + bytes_per_pixel *
            (config_.pixels_per_scan_line * copy_start_dst_y + copy_start_dst_x);
        const uint8_t* src_buf = src.config_.frame_buffer + bytes_per_pixel *
            (src.config_.pixels_per_scan_line * src_start_y + src_start_x);

        for (int dy = 0; dy < copy_end_dst_y - copy_start_dst_y; ++dy) {
            std::memcpy(dst_buf, src_buf, bytes_per_copy_line);
            dst_buf += bytes_per_pixel * config_.pixels_per_scan_line;
            src_buf += bytes_per_pixel * src.config_.pixels_per_scan_line;
            stats.lines_copied++;
            stats.bytes_copied += bytes_per_copy_line;
        }

        return stats;
    }

    PixelColor ReadPixel(Vector2D<int> pos) const {
        if (pos.x < 0 || pos.x >= static_cast<int>(config_.horizontal_resolution) ||
            pos.y < 0 || pos.y >= static_cast<int>(config_.vertical_resolution)) {
            return PixelColor{0, 0, 0};
        }
        const uint8_t* p = config_.frame_buffer + 4 * (config_.pixels_per_scan_line * pos.y + pos.x);
        if (config_.pixel_format == kPixelRGBResv8BitPerColor) {
            return PixelColor{p[0], p[1], p[2]};
        } else {
            return PixelColor{p[2], p[1], p[0]};
        }
    }

    PixelWriter& Writer() { return *writer_; }
    [[nodiscard]] const FrameBufferConfig& Config() const { return config_; }

private:
    static int pos_y(Vector2D<int> pos) { return pos.y; }

    FrameBufferConfig config_{};
    std::vector<uint8_t> buffer_{};
    std::unique_ptr<FrameBufferWriter> writer_{};
};

// =============================================================================
// 4. Window Class with Shadow Buffer (Matching MikanOS kernel/window.cpp)
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
            data_[y].resize(width, PixelColor{0, 0, 0});
        }

        FrameBufferConfig config{};
        config.frame_buffer = nullptr;
        config.horizontal_resolution = width;
        config.vertical_resolution = height;
        config.pixel_format = shadow_format;

        shadow_buffer_.Initialize(config);
    }

    void DrawTo(FrameBuffer& dst, Vector2D<int> position) {
        if (!transparent_color_) {
            // Fast Path: line-by-line memcpy
            dst.Copy(position, shadow_buffer_);
            last_draw_mode_ = "memcpy";
            return;
        }

        // Slow Path: pixel-by-pixel with transparency check
        last_draw_mode_ = "pixel_by_pixel";
        const auto tc = transparent_color_.value();
        auto& writer = dst.Writer();
        for (int y = 0; y < Height(); ++y) {
            for (int x = 0; x < Width(); ++x) {
                const auto c = At(Vector2D<int>{x, y});
                if (c != tc) {
                    writer.Write(position + Vector2D<int>{x, y}, c);
                }
            }
        }
    }

    void SetTransparentColor(std::optional<PixelColor> c) {
        transparent_color_ = c;
    }

    WindowWriter* Writer() { return &writer_; }

    [[nodiscard]] const PixelColor& At(Vector2D<int> pos) const {
        return data_[pos.y][pos.x];
    }

    void Write(Vector2D<int> pos, PixelColor c) {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            data_[pos.y][pos.x] = c;
            shadow_buffer_.Writer().Write(pos, c);
        }
    }

    [[nodiscard]] int Width() const { return width_; }
    [[nodiscard]] int Height() const { return height_; }
    [[nodiscard]] const std::string& LastDrawMode() const { return last_draw_mode_; }
    [[nodiscard]] const FrameBuffer& ShadowBuffer() const { return shadow_buffer_; }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> data_{};
    WindowWriter writer_{*this};
    std::optional<PixelColor> transparent_color_{std::nullopt};
    FrameBuffer shadow_buffer_{};
    std::string last_draw_mode_{"none"};
};

// =============================================================================
// 5. Layer & LayerManager (Matching MikanOS kernel/layer.cpp)
// =============================================================================

class Layer {
public:
    explicit Layer(unsigned int id) : id_{id} {}

    [[nodiscard]] unsigned int ID() const { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }

    [[nodiscard]] std::shared_ptr<Window> GetWindow() const { return window_; }

    Layer& Move(Vector2D<int> pos) {
        pos_ = pos;
        return *this;
    }

    Layer& MoveRelative(Vector2D<int> pos_diff) {
        pos_ += pos_diff;
        return *this;
    }

    [[nodiscard]] Vector2D<int> GetPosition() const { return pos_; }

    void DrawTo(FrameBuffer& screen) const {
        if (window_) {
            window_->DrawTo(screen, pos_);
        }
    }

private:
    unsigned int id_;
    Vector2D<int> pos_{0, 0};
    std::shared_ptr<Window> window_{};
};

class LayerManager {
public:
    void SetWriter(FrameBuffer* screen) {
        screen_ = screen;
    }

    Layer& NewLayer() {
        ++latest_id_;
        return *layers_.emplace_back(std::make_unique<Layer>(latest_id_));
    }

    void Draw() const {
        for (auto layer : layer_stack_) {
            layer->DrawTo(*screen_);
        }
    }

    void Move(unsigned int id, Vector2D<int> new_position) {
        FindLayer(id)->Move(new_position);
    }

    void MoveRelative(unsigned int id, Vector2D<int> pos_diff) {
        FindLayer(id)->MoveRelative(pos_diff);
    }

    void UpDown(unsigned int id, int new_height) {
        if (new_height < 0) {
            Hide(id);
            return;
        }
        if (new_height > static_cast<int>(layer_stack_.size())) {
            new_height = static_cast<int>(layer_stack_.size());
        }

        auto layer = FindLayer(id);
        auto old_pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        auto new_pos = layer_stack_.begin() + new_height;

        if (old_pos == layer_stack_.end()) {
            layer_stack_.insert(new_pos, layer);
            return;
        }

        if (new_pos == layer_stack_.end()) {
            --new_pos;
        }
        layer_stack_.erase(old_pos);
        layer_stack_.insert(new_pos, layer);
    }

    void Hide(unsigned int id) {
        auto layer = FindLayer(id);
        auto pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (pos != layer_stack_.end()) {
            layer_stack_.erase(pos);
        }
    }

    [[nodiscard]] size_t StackSize() const { return layer_stack_.size(); }

private:
    Layer* FindLayer(unsigned int id) {
        auto pred = [id](const std::unique_ptr<Layer>& elem) {
            return elem->ID() == id;
        };
        auto it = std::find_if(layers_.begin(), layers_.end(), pred);
        if (it == layers_.end()) {
            return nullptr;
        }
        return it->get();
    }

    FrameBuffer* screen_{nullptr};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    unsigned int latest_id_{0};
};

// =============================================================================
// 6. Test Suites and Validation Routines
// =============================================================================

void TestFrameBufferInitialization() {
    std::cout << "[Test 1] FrameBuffer Initialization (VRAM vs Off-screen)... ";

    // 1. Off-screen FrameBuffer (allocates dynamic buffer)
    FrameBuffer offscreen;
    FrameBufferConfig cfg1{};
    cfg1.horizontal_resolution = 640;
    cfg1.vertical_resolution = 480;
    cfg1.pixel_format = kPixelRGBResv8BitPerColor;
    assert(offscreen.Initialize(cfg1));
    assert(offscreen.Config().pixels_per_scan_line == 640);
    assert(offscreen.Config().frame_buffer != nullptr);

    // 2. Hardware VRAM Screen (uses existing buffer pointer)
    std::vector<uint8_t> vram_mem(1024 * 768 * 4, 0);
    FrameBuffer screen;
    FrameBufferConfig cfg2{};
    cfg2.frame_buffer = vram_mem.data();
    cfg2.pixels_per_scan_line = 1024;
    cfg2.horizontal_resolution = 1024;
    cfg2.vertical_resolution = 768;
    cfg2.pixel_format = kPixelRGBResv8BitPerColor;
    assert(screen.Initialize(cfg2));
    assert(screen.Config().frame_buffer == vram_mem.data());

    std::cout << "PASSED" << std::endl;
}

void TestFrameBufferCopyAndClipping() {
    std::cout << "[Test 2] FrameBuffer::Copy Line-by-Line memcpy & Clipping Math... ";

    FrameBuffer dst;
    FrameBufferConfig dst_cfg{};
    dst_cfg.horizontal_resolution = 100;
    dst_cfg.vertical_resolution = 100;
    dst_cfg.pixel_format = kPixelRGBResv8BitPerColor;
    assert(dst.Initialize(dst_cfg));

    FrameBuffer src;
    FrameBufferConfig src_cfg{};
    src_cfg.horizontal_resolution = 40;
    src_cfg.vertical_resolution = 30;
    src_cfg.pixel_format = kPixelRGBResv8BitPerColor;
    assert(src.Initialize(src_cfg));

    // Fill source with red
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) {
            src.Writer().Write({x, y}, PixelColor{255, 0, 0});
        }
    }

    // Case A: Full interior copy (at pos {10, 20})
    auto stats_a = dst.Copy({10, 20}, src);
    assert(stats_a.lines_copied == 30);
    assert(stats_a.bytes_copied == 30 * 40 * 4);
    assert(!stats_a.clipped);
    assert(dst.ReadPixel({10, 20}) == (PixelColor{255, 0, 0}));
    assert(dst.ReadPixel({49, 49}) == (PixelColor{255, 0, 0}));
    assert(dst.ReadPixel({9, 19}) == (PixelColor{0, 0, 0}));

    // Case B: Partial clipping (at pos {80, 80} -> extends beyond 100x100)
    // Clipped to: x in [80, 100) -> 20 px wide, y in [80, 100) -> 20 px high
    auto stats_b = dst.Copy({80, 80}, src);
    assert(stats_b.lines_copied == 20);
    assert(stats_b.bytes_copied == 20 * 20 * 4);
    assert(stats_b.clipped);
    assert(dst.ReadPixel({80, 80}) == (PixelColor{255, 0, 0}));
    assert(dst.ReadPixel({99, 99}) == (PixelColor{255, 0, 0}));

    // Case C: Negative coordinate clipping (at pos {-10, -5})
    // Clipped to: x in [0, 30) -> 30 px wide, y in [0, 25) -> 25 px high
    auto stats_c = dst.Copy({-10, -5}, src);
    assert(stats_c.lines_copied == 25);
    assert(stats_c.bytes_copied == 25 * 30 * 4);
    assert(stats_c.clipped);
    assert(dst.ReadPixel({0, 0}) == (PixelColor{255, 0, 0}));
    assert(dst.ReadPixel({29, 24}) == (PixelColor{255, 0, 0}));

    // Case D: Completely outside (at pos {150, 150})
    auto stats_d = dst.Copy({150, 150}, src);
    assert(stats_d.lines_copied == 0);
    assert(stats_d.bytes_copied == 0);
    assert(stats_d.clipped);

    std::cout << "PASSED" << std::endl;
}

void TestWindowFastPathVsSlowPath() {
    std::cout << "[Test 3] Window Fast-Path (memcpy) vs Slow-Path (Transparency)... ";

    FrameBuffer dst;
    FrameBufferConfig dst_cfg{};
    dst_cfg.horizontal_resolution = 200;
    dst_cfg.vertical_resolution = 200;
    dst_cfg.pixel_format = kPixelRGBResv8BitPerColor;
    assert(dst.Initialize(dst_cfg));

    // 1. Opaque window (Desktop background): should use memcpy
    Window bg_win(200, 200, kPixelRGBResv8BitPerColor);
    bg_win.Write({50, 50}, PixelColor{10, 20, 30});
    bg_win.DrawTo(dst, {0, 0});
    assert(bg_win.LastDrawMode() == "memcpy");
    assert(dst.ReadPixel({50, 50}) == (PixelColor{10, 20, 30}));

    // 2. Transparent window (Mouse cursor): should use pixel_by_pixel
    Window mouse_win(16, 24, kPixelRGBResv8BitPerColor);
    mouse_win.SetTransparentColor(PixelColor{99, 99, 99});
    mouse_win.Write({0, 0}, PixelColor{99, 99, 99}); // transparent pixel
    mouse_win.Write({1, 1}, PixelColor{255, 255, 255}); // visible pixel
    mouse_win.DrawTo(dst, {10, 10});
    assert(mouse_win.LastDrawMode() == "pixel_by_pixel");
    assert(dst.ReadPixel({10, 10}) == (PixelColor{0, 0, 0})); // not overwritten by {99,99,99}
    assert(dst.ReadPixel({11, 11}) == (PixelColor{255, 255, 255}));

    std::cout << "PASSED" << std::endl;
}

void TestCompositingPerformanceBenchmark() {
    std::cout << "[Test 4] Performance Benchmark: Pixel-by-Pixel vs Line-by-Line memcpy..." << std::endl;

    constexpr int kWidth = 1024;
    constexpr int kHeight = 768;
    constexpr int kIterations = 100;

    FrameBuffer screen;
    FrameBufferConfig screen_cfg{};
    screen_cfg.horizontal_resolution = kWidth;
    screen_cfg.vertical_resolution = kHeight;
    screen_cfg.pixel_format = kPixelRGBResv8BitPerColor;
    assert(screen.Initialize(screen_cfg));

    Window win(kWidth, kHeight, kPixelRGBResv8BitPerColor);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            win.Write({x, y}, PixelColor{static_cast<uint8_t>(x & 0xFF), static_cast<uint8_t>(y & 0xFF), 128});
        }
    }

    // Benchmark A: Fast-path (memcpy)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        win.DrawTo(screen, {0, 0});
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double memcpy_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIterations;

    // Benchmark B: Simulated slow-path (Pixel-by-pixel with virtual dispatch)
    auto& writer = screen.Writer();
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) { // fewer iterations because it's slow
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                writer.Write({x, y}, win.At({x, y}));
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double slow_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / 5;

    double speedup = slow_ms / std::max(memcpy_ms, 0.0001);
    double throughput_gb_s = (kWidth * kHeight * 4.0) / (memcpy_ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "  - Individual PixelWriter: " << std::fixed << std::setprecision(2) << slow_ms << " ms / frame" << std::endl;
    std::cout << "  - Line-by-Line memcpy:    " << std::fixed << std::setprecision(3) << memcpy_ms << " ms / frame" << std::endl;
    std::cout << "  - Speedup Ratio:          " << std::fixed << std::setprecision(1) << speedup << "x faster!" << std::endl;
    std::cout << "  - memcpy Throughput:      " << std::fixed << std::setprecision(2) << throughput_gb_s << " GB/s" << std::endl;

    assert(speedup > 2.0); // memcpy must be significantly faster
    std::cout << "  -> Benchmark PASSED" << std::endl;
}

void TestLayerManagerCompositingPipeline() {
    std::cout << "[Test 5] Full LayerManager Compositing Integration... ";

    FrameBuffer screen;
    FrameBufferConfig cfg{};
    cfg.horizontal_resolution = 320;
    cfg.vertical_resolution = 240;
    cfg.pixel_format = kPixelRGBResv8BitPerColor;
    assert(screen.Initialize(cfg));

    LayerManager lm;
    lm.SetWriter(&screen);

    // Layer 1: Desktop Background (320x240, blue)
    auto bg_win = std::make_shared<Window>(320, 240, kPixelRGBResv8BitPerColor);
    for (int y = 0; y < 240; ++y) {
        for (int x = 0; x < 320; ++x) {
            bg_win->Write({x, y}, PixelColor{0, 100, 200});
        }
    }
    auto& bg_layer = lm.NewLayer().SetWindow(bg_win).Move({0, 0});
    lm.UpDown(bg_layer.ID(), 0);

    // Layer 2: Transparent Mouse Cursor (10x10, yellow with transparent border)
    auto mouse_win = std::make_shared<Window>(10, 10, kPixelRGBResv8BitPerColor);
    mouse_win->SetTransparentColor(PixelColor{0, 0, 0});
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            if (x >= 2 && x <= 7 && y >= 2 && y <= 7) {
                mouse_win->Write({x, y}, PixelColor{255, 255, 0});
            } else {
                mouse_win->Write({x, y}, PixelColor{0, 0, 0}); // transparent
            }
        }
    }
    auto& mouse_layer = lm.NewLayer().SetWindow(mouse_win).Move({50, 50});
    lm.UpDown(mouse_layer.ID(), 1);

    // Render composited frame
    lm.Draw();

    // Verify background intact outside mouse
    assert(screen.ReadPixel({10, 10}) == (PixelColor{0, 100, 200}));

    // Verify mouse transparent edge preserved background
    assert(screen.ReadPixel({50, 50}) == (PixelColor{0, 100, 200}));

    // Verify mouse center drawn yellow
    assert(screen.ReadPixel({54, 54}) == (PixelColor{255, 255, 0}));

    // Move mouse and redraw
    lm.Move(mouse_layer.ID(), {100, 100});
    lm.Draw();

    // Previous mouse position now shows desktop background again
    assert(screen.ReadPixel({54, 54}) == (PixelColor{0, 100, 200}));
    // New mouse position shows yellow
    assert(screen.ReadPixel({104, 104}) == (PixelColor{255, 255, 0}));

    std::cout << "PASSED" << std::endl;
}

} // namespace mikanos::study

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "MikanOS Chapter 9.5: Layer Compositing Optimization Verification" << std::endl;
    std::cout << "=================================================================" << std::endl;

    mikanos::study::TestFrameBufferInitialization();
    mikanos::study::TestFrameBufferCopyAndClipping();
    mikanos::study::TestWindowFastPathVsSlowPath();
    mikanos::study::TestCompositingPerformanceBenchmark();
    mikanos::study::TestLayerManagerCompositingPipeline();

    std::cout << "=================================================================" << std::endl;
    std::cout << "All C++20 Section 9.5 validation checks successfully passed!" << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
