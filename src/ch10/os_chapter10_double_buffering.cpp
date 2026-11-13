/**
 * @file os_chapter10_double_buffering.cpp
 * @brief Chapter 10.5: Double Buffering & Back Buffer (`osbook_day10e`) Standalone C++20 Verification
 *
 * Demonstrates and verifies:
 * 1. FrameBuffer dual-mode initialization (`frame_buffer != nullptr` for VRAM vs.
 *    `frame_buffer == nullptr` for RAM-allocated `std::vector<uint8_t> buffer_` with
 *    compact `pixels_per_scan_line = horizontal_resolution`).
 * 2. `FrameBuffer::Config()` accessor enabling `LayerManager::SetWriter(FrameBuffer* screen)`
 *    to construct `mutable FrameBuffer back_buffer_{}` matching the screen's geometry/format.
 * 3. `LayerManager::Draw(const Rectangle<int>& area) const` and `LayerManager::Draw(unsigned int id) const`:
 *    - Offscreen compositing of all required layers into `back_buffer_` in system RAM.
 *    - Atomic area transfer `screen_->Copy(area.pos, back_buffer_, area)` to hardware VRAM.
 * 4. Elimination of intermediate VRAM flicker exposures:
 *    - Single-buffered `Draw(main_id)` exposes transient overwrites on VRAM whenever
 *      a mouse cursor overlaps the counter window, or during `Move(id, pos)`.
 *    - Double-buffered `Draw(main_id)` / `Move(id, pos)` reduces VRAM intermediate
 *      exposures to strictly 0 while preserving lower-layer contents in `back_buffer_`.
 * 5. Padded hardware VRAM stride (`pixels_per_scan_line > horizontal_resolution`) handling
 *    and 3-way clipped `FrameBuffer::Copy` boundary safety.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace mikanos::study::ch10_5 {

// =============================================================================
// 1. Geometry & Pixel Color
// =============================================================================

template <typename T>
struct Vector2D {
    T x{}, y{};

    constexpr bool operator==(const Vector2D& rhs) const noexcept {
        return x == rhs.x && y == rhs.y;
    }

    constexpr Vector2D operator+(const Vector2D& rhs) const noexcept {
        return {x + rhs.x, y + rhs.y};
    }

    constexpr Vector2D operator-(const Vector2D& rhs) const noexcept {
        return {x - rhs.x, y - rhs.y};
    }

    Vector2D& operator+=(const Vector2D& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
};

template <typename T, typename U>
constexpr Vector2D<T> ElementMin(const Vector2D<T>& lhs, const Vector2D<U>& rhs) noexcept {
    return {
        lhs.x < static_cast<T>(rhs.x) ? lhs.x : static_cast<T>(rhs.x),
        lhs.y < static_cast<T>(rhs.y) ? lhs.y : static_cast<T>(rhs.y),
    };
}

template <typename T, typename U>
constexpr Vector2D<T> ElementMax(const Vector2D<T>& lhs, const Vector2D<U>& rhs) noexcept {
    return {
        lhs.x > static_cast<T>(rhs.x) ? lhs.x : static_cast<T>(rhs.x),
        lhs.y > static_cast<T>(rhs.y) ? lhs.y : static_cast<T>(rhs.y),
    };
}

template <typename T>
struct Rectangle {
    Vector2D<T> pos{}, size{};

    constexpr bool operator==(const Rectangle& rhs) const noexcept {
        return pos == rhs.pos && size == rhs.size;
    }
};

template <typename T, typename U>
constexpr Rectangle<T> operator&(const Rectangle<T>& lhs, const Rectangle<U>& rhs) noexcept {
    const auto lhs_end = lhs.pos + lhs.size;
    const auto rhs_end = rhs.pos + rhs.size;
    if (lhs_end.x < static_cast<T>(rhs.pos.x) || lhs_end.y < static_cast<T>(rhs.pos.y) ||
        static_cast<T>(rhs_end.x) < lhs.pos.x || static_cast<T>(rhs_end.y) < lhs.pos.y) {
        return {{0, 0}, {0, 0}};
    }
    const auto new_pos = ElementMax(lhs.pos, rhs.pos);
    const auto new_size = ElementMin(lhs_end, rhs_end) - new_pos;
    if (new_size.x <= 0 || new_size.y <= 0) {
        return {{0, 0}, {0, 0}};
    }
    return {new_pos, new_size};
}

struct PixelColor {
    uint8_t r{}, g{}, b{};

    constexpr bool operator==(const PixelColor& rhs) const noexcept {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }

    constexpr bool operator!=(const PixelColor& rhs) const noexcept {
        return !(*this == rhs);
    }
};

// =============================================================================
// 2. FrameBufferConfig & PixelWriter Hierarchy
// =============================================================================

enum class PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer{nullptr};
    uint32_t pixels_per_scan_line{0};
    uint32_t horizontal_resolution{0};
    uint32_t vertical_resolution{0};
    PixelFormat pixel_format{PixelFormat::kPixelBGRResv8BitPerColor};
};

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(Vector2D<int> pos, const PixelColor& c) = 0;
    [[nodiscard]] virtual int Width() const = 0;
    [[nodiscard]] virtual int Height() const = 0;
};

class FrameBufferWriter : public PixelWriter {
public:
    explicit FrameBufferWriter(const FrameBufferConfig& config) : config_{config} {}
    ~FrameBufferWriter() override = default;

    [[nodiscard]] int Width() const override {
        return static_cast<int>(config_.horizontal_resolution);
    }
    [[nodiscard]] int Height() const override {
        return static_cast<int>(config_.vertical_resolution);
    }

protected:
    [[nodiscard]] uint8_t* PixelAt(Vector2D<int> pos) const {
        return config_.frame_buffer + 4 * (static_cast<size_t>(config_.pixels_per_scan_line) * pos.y + pos.x);
    }

private:
    FrameBufferConfig config_;
};

class RGBResv8BitPerColorPixelWriter final : public FrameBufferWriter {
public:
    using FrameBufferWriter::FrameBufferWriter;
    void Write(Vector2D<int> pos, const PixelColor& c) override {
        if (pos.x < 0 || pos.x >= Width() || pos.y < 0 || pos.y >= Height()) return;
        uint8_t* p = PixelAt(pos);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0;
    }
};

class BGRResv8BitPerColorPixelWriter final : public FrameBufferWriter {
public:
    using FrameBufferWriter::FrameBufferWriter;
    void Write(Vector2D<int> pos, const PixelColor& c) override {
        if (pos.x < 0 || pos.x >= Width() || pos.y < 0 || pos.y >= Height()) return;
        uint8_t* p = PixelAt(pos);
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0;
    }
};

void FillRectangle(PixelWriter& writer, Vector2D<int> pos, Vector2D<int> size, const PixelColor& c) {
    for (int dy = 0; dy < size.y; ++dy) {
        for (int dx = 0; dx < size.x; ++dx) {
            writer.Write({pos.x + dx, pos.y + dy}, c);
        }
    }
}

// =============================================================================
// 3. FrameBuffer with Config() Accessor & Copy
// =============================================================================

namespace {

int BytesPerPixel(PixelFormat format) {
    switch (format) {
        case PixelFormat::kPixelRGBResv8BitPerColor:
        case PixelFormat::kPixelBGRResv8BitPerColor:
            return 4;
    }
    return -1;
}

uint8_t* FrameAddrAt(Vector2D<int> pos, const FrameBufferConfig& config) {
    return config.frame_buffer +
           static_cast<size_t>(BytesPerPixel(config.pixel_format)) *
               (static_cast<size_t>(config.pixels_per_scan_line) * pos.y + pos.x);
}

int BytesPerScanLine(const FrameBufferConfig& config) {
    return BytesPerPixel(config.pixel_format) * static_cast<int>(config.pixels_per_scan_line);
}

Vector2D<int> FrameBufferSize(const FrameBufferConfig& config) {
    return {static_cast<int>(config.horizontal_resolution),
            static_cast<int>(config.vertical_resolution)};
}

}  // namespace

class FrameBuffer {
public:
    bool Initialize(const FrameBufferConfig& config) {
        config_ = config;
        const int bpp = BytesPerPixel(config_.pixel_format);
        if (bpp <= 0) return false;

        if (config_.frame_buffer != nullptr) {
            buffer_.resize(0);
        } else {
            buffer_.resize(static_cast<size_t>(bpp) *
                           config_.horizontal_resolution *
                           config_.vertical_resolution);
            config_.frame_buffer = buffer_.data();
            config_.pixels_per_scan_line = config_.horizontal_resolution;
        }

        switch (config_.pixel_format) {
            case PixelFormat::kPixelRGBResv8BitPerColor:
                writer_ = std::make_unique<RGBResv8BitPerColorPixelWriter>(config_);
                break;
            case PixelFormat::kPixelBGRResv8BitPerColor:
                writer_ = std::make_unique<BGRResv8BitPerColorPixelWriter>(config_);
                break;
        }
        return true;
    }

    bool Copy(Vector2D<int> dst_pos, const FrameBuffer& src, const Rectangle<int>& src_area) {
        if (config_.pixel_format != src.config_.pixel_format) return false;
        const int bpp = BytesPerPixel(config_.pixel_format);
        if (bpp <= 0) return false;

        const Rectangle<int> src_area_shifted{dst_pos, src_area.size};
        const Rectangle<int> src_outline{dst_pos - src_area.pos, FrameBufferSize(src.config_)};
        const Rectangle<int> dst_outline{{0, 0}, FrameBufferSize(config_)};
        const auto copy_area = dst_outline & src_outline & src_area_shifted;

        if (copy_area.size.x <= 0 || copy_area.size.y <= 0) {
            return true;
        }

        const auto src_start_pos = copy_area.pos - (dst_pos - src_area.pos);
        uint8_t* dst_buf = FrameAddrAt(copy_area.pos, config_);
        const uint8_t* src_buf = FrameAddrAt(src_start_pos, src.config_);

        const size_t row_bytes = static_cast<size_t>(bpp) * static_cast<size_t>(copy_area.size.x);
        const int dst_stride = BytesPerScanLine(config_);
        const int src_stride = BytesPerScanLine(src.config_);

        for (int y = 0; y < copy_area.size.y; ++y) {
            std::memcpy(dst_buf, src_buf, row_bytes);
            dst_buf += dst_stride;
            src_buf += src_stride;
        }
        last_copied_pixels_ = static_cast<size_t>(copy_area.size.x) * static_cast<size_t>(copy_area.size.y);
        total_copied_pixels_ += last_copied_pixels_;
        return true;
    }

    [[nodiscard]] FrameBufferWriter& Writer() { return *writer_; }
    [[nodiscard]] const FrameBufferConfig& Config() const { return config_; }
    [[nodiscard]] size_t InternalBufferBytes() const { return buffer_.size(); }
    [[nodiscard]] size_t TotalCopiedPixels() const { return total_copied_pixels_; }
    [[nodiscard]] size_t LastCopiedPixels() const { return last_copied_pixels_; }

    [[nodiscard]] PixelColor ReadPixel(int x, int y) const {
        const uint8_t* p = FrameAddrAt({x, y}, config_);
        if (config_.pixel_format == PixelFormat::kPixelRGBResv8BitPerColor) {
            return {p[0], p[1], p[2]};
        }
        return {p[2], p[1], p[0]};
    }

private:
    FrameBufferConfig config_{};
    std::vector<uint8_t> buffer_{};
    std::unique_ptr<FrameBufferWriter> writer_{};
    size_t last_copied_pixels_{0};
    size_t total_copied_pixels_{0};
};

// =============================================================================
// 4. Window, Layer, and Double-Buffered LayerManager (`osbook_day10e`)
// =============================================================================

class Window {
public:
    class WindowWriter final : public PixelWriter {
    public:
        explicit WindowWriter(Window& window) : window_{window} {}
        void Write(Vector2D<int> pos, const PixelColor& c) override {
            window_.Write(pos, c);
        }
        [[nodiscard]] int Width() const override { return window_.Width(); }
        [[nodiscard]] int Height() const override { return window_.Height(); }

    private:
        Window& window_;
    };

    Window(int width, int height, PixelFormat shadow_format)
        : width_{width}, height_{height}, writer_{*this} {
        data_.resize(static_cast<size_t>(height), std::vector<PixelColor>(static_cast<size_t>(width)));
        FrameBufferConfig config{};
        config.frame_buffer = nullptr;
        config.horizontal_resolution = static_cast<uint32_t>(width);
        config.vertical_resolution = static_cast<uint32_t>(height);
        config.pixel_format = shadow_format;
        shadow_buffer_.Initialize(config);
    }

    void DrawTo(FrameBuffer& dst, Vector2D<int> pos, const Rectangle<int>& area) const {
        if (!transparent_color_) {
            const Rectangle<int> window_area{pos, Size()};
            const auto intersection = area & window_area;
            if (intersection.size.x > 0 && intersection.size.y > 0) {
                dst.Copy(intersection.pos, shadow_buffer_, {intersection.pos - pos, intersection.size});
            }
            return;
        }

        const auto tc = *transparent_color_;
        auto& writer = dst.Writer();
        const int x_start = std::max(0, area.pos.x - pos.x);
        const int x_end = std::min(width_, area.pos.x + area.size.x - pos.x);
        const int y_start = std::max(0, area.pos.y - pos.y);
        const int y_end = std::min(height_, area.pos.y + area.size.y - pos.y);

        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                const auto c = At(x, y);
                if (c != tc) {
                    writer.Write(pos + Vector2D<int>{x, y}, c);
                }
            }
        }
    }

    void SetTransparentColor(std::optional<PixelColor> c) { transparent_color_ = c; }
    [[nodiscard]] WindowWriter* Writer() { return &writer_; }
    [[nodiscard]] int Width() const { return width_; }
    [[nodiscard]] int Height() const { return height_; }
    [[nodiscard]] Vector2D<int> Size() const { return {width_, height_}; }

    void Write(Vector2D<int> pos, PixelColor c) {
        if (pos.x < 0 || pos.x >= width_ || pos.y < 0 || pos.y >= height_) return;
        data_[static_cast<size_t>(pos.y)][static_cast<size_t>(pos.x)] = c;
        shadow_buffer_.Writer().Write(pos, c);
    }

    [[nodiscard]] const PixelColor& At(int x, int y) const {
        return data_[static_cast<size_t>(y)][static_cast<size_t>(x)];
    }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> data_{};
    WindowWriter writer_;
    std::optional<PixelColor> transparent_color_{std::nullopt};
    FrameBuffer shadow_buffer_{};
};

class Layer {
public:
    explicit Layer(unsigned int id = 0) : id_{id} {}
    [[nodiscard]] unsigned int ID() const { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }
    [[nodiscard]] std::shared_ptr<Window> GetWindow() const { return window_; }
    [[nodiscard]] Vector2D<int> GetPosition() const { return pos_; }

    Layer& Move(Vector2D<int> pos) {
        pos_ = pos;
        return *this;
    }
    Layer& MoveRelative(Vector2D<int> pos_diff) {
        pos_ += pos_diff;
        return *this;
    }

    void DrawTo(FrameBuffer& screen, const Rectangle<int>& area) const {
        if (window_) {
            window_->DrawTo(screen, pos_, area);
        }
    }

private:
    unsigned int id_;
    Vector2D<int> pos_{};
    std::shared_ptr<Window> window_{};
};

class LayerManager {
public:
    void SetWriter(FrameBuffer* screen) {
        screen_ = screen;

        FrameBufferConfig back_config = screen->Config();
        back_config.frame_buffer = nullptr;
        back_buffer_.Initialize(back_config);
    }

    Layer& NewLayer() {
        ++latest_id_;
        return *layers_.emplace_back(std::make_unique<Layer>(latest_id_));
    }

    void Draw(const Rectangle<int>& area) const {
        for (const auto* layer : layer_stack_) {
            layer->DrawTo(back_buffer_, area);
        }
        screen_->Copy(area.pos, back_buffer_, area);
    }

    void Draw(unsigned int id) const {
        bool draw = false;
        Rectangle<int> window_area{};
        for (const auto* layer : layer_stack_) {
            if (layer->ID() == id) {
                window_area.size = layer->GetWindow()->Size();
                window_area.pos = layer->GetPosition();
                draw = true;
            }
            if (draw) {
                layer->DrawTo(back_buffer_, window_area);
            }
        }
        screen_->Copy(window_area.pos, back_buffer_, window_area);
    }

    void Move(unsigned int id, Vector2D<int> new_pos) {
        auto* layer = FindLayer(id);
        const auto window_size = layer->GetWindow()->Size();
        const auto old_pos = layer->GetPosition();
        layer->Move(new_pos);
        Draw({old_pos, window_size});
        Draw(id);
    }

    void UpDown(unsigned int id, int new_height) {
        if (new_height < 0) {
            Hide(id);
            return;
        }
        if (new_height > static_cast<int>(layer_stack_.size())) {
            new_height = static_cast<int>(layer_stack_.size());
        }
        auto* layer = FindLayer(id);
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
        auto* layer = FindLayer(id);
        auto pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (pos != layer_stack_.end()) {
            layer_stack_.erase(pos);
        }
    }

    [[nodiscard]] const FrameBuffer& BackBuffer() const { return back_buffer_; }
    [[nodiscard]] const std::vector<Layer*>& Stack() const { return layer_stack_; }

    /**
     * @brief Diagnostic helper simulating legacy single-buffered Draw(id) directly into VRAM
     *        while counting intermediate pixel state exposures at probe coordinates.
     */
    [[nodiscard]] int SimulateSingleBufferDrawIdExposures(
        unsigned int id, FrameBuffer& single_vram, Vector2D<int> probe_pos,
        const PixelColor& expected_final_color) const {
        int wrong_exposures = 0;
        bool draw = false;
        Rectangle<int> window_area{};
        for (const auto* layer : layer_stack_) {
            if (layer->ID() == id) {
                window_area.size = layer->GetWindow()->Size();
                window_area.pos = layer->GetPosition();
                draw = true;
            }
            if (draw) {
                layer->DrawTo(single_vram, window_area);
                if (single_vram.ReadPixel(probe_pos.x, probe_pos.y) != expected_final_color) {
                    ++wrong_exposures;
                }
            }
        }
        return wrong_exposures;
    }

private:
    Layer* FindLayer(unsigned int id) {
        auto it = std::find_if(layers_.begin(), layers_.end(),
                               [id](const std::unique_ptr<Layer>& elem) { return elem->ID() == id; });
        return it == layers_.end() ? nullptr : it->get();
    }

    FrameBuffer* screen_{nullptr};
    mutable FrameBuffer back_buffer_{};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    unsigned int latest_id_{0};
};

}  // namespace mikanos::study::ch10_5

// =============================================================================
// 5. Verification Tests
// =============================================================================

int main() {
    using namespace mikanos::study::ch10_5;

    constexpr uint32_t kWidth = 320;
    constexpr uint32_t kHeight = 200;
    // Simulate hardware VRAM with scanline padding (384 px > 320 px horizontal resolution)
    constexpr uint32_t kHardwareScanLine = 384;

    std::vector<uint8_t> hardware_vram(static_cast<size_t>(4) * kHardwareScanLine * kHeight, 0);
    std::vector<uint8_t> single_vram_mem(static_cast<size_t>(4) * kHardwareScanLine * kHeight, 0);

    FrameBufferConfig hw_config{};
    hw_config.frame_buffer = hardware_vram.data();
    hw_config.pixels_per_scan_line = kHardwareScanLine;
    hw_config.horizontal_resolution = kWidth;
    hw_config.vertical_resolution = kHeight;
    hw_config.pixel_format = PixelFormat::kPixelBGRResv8BitPerColor;

    FrameBuffer screen;
    assert(screen.Initialize(hw_config));
    assert(screen.InternalBufferBytes() == 0);
    assert(screen.Config().pixels_per_scan_line == kHardwareScanLine);

    FrameBufferConfig single_config = hw_config;
    single_config.frame_buffer = single_vram_mem.data();
    FrameBuffer single_screen;
    assert(single_screen.Initialize(single_config));

    LayerManager mgr;
    mgr.SetWriter(&screen);

    // 1. Verify back_buffer_ initialization via FrameBuffer::Config() & nullptr trick
    const auto& back_cfg = mgr.BackBuffer().Config();
    assert(back_cfg.frame_buffer != nullptr);
    assert(back_cfg.frame_buffer != hardware_vram.data());
    assert(back_cfg.horizontal_resolution == kWidth);
    assert(back_cfg.vertical_resolution == kHeight);
    assert(back_cfg.pixels_per_scan_line == kWidth);  // Compact stride without hardware padding!
    assert(mgr.BackBuffer().InternalBufferBytes() == static_cast<size_t>(4) * kWidth * kHeight);

    // 2. Build 4 layers (DesktopBG, Console, CounterWindow, MouseCursor)
    const PixelColor kBgColor{45, 110, 142};
    const PixelColor kConsoleColor{16, 24, 32};
    const PixelColor kWinColor{198, 198, 198};
    const PixelColor kMouseColor{255, 255, 255};
    const PixelColor kMouseTrans{0, 0, 1};

    auto bg_win = std::make_shared<Window>(kWidth, kHeight, hw_config.pixel_format);
    FillRectangle(*bg_win->Writer(), {0, 0}, {kWidth, kHeight}, kBgColor);

    auto con_win = std::make_shared<Window>(180, 100, hw_config.pixel_format);
    FillRectangle(*con_win->Writer(), {0, 0}, {180, 100}, kConsoleColor);

    auto main_win = std::make_shared<Window>(120, 50, hw_config.pixel_format);
    FillRectangle(*main_win->Writer(), {0, 0}, {120, 50}, kWinColor);

    auto mouse_win = std::make_shared<Window>(10, 14, hw_config.pixel_format);
    mouse_win->SetTransparentColor(kMouseTrans);
    FillRectangle(*mouse_win->Writer(), {0, 0}, {10, 14}, kMouseTrans);
    FillRectangle(*mouse_win->Writer(), {2, 2}, {6, 10}, kMouseColor);

    const unsigned int bg_id = mgr.NewLayer().SetWindow(bg_win).Move({0, 0}).ID();
    const unsigned int con_id = mgr.NewLayer().SetWindow(con_win).Move({0, 0}).ID();
    const unsigned int main_id = mgr.NewLayer().SetWindow(main_win).Move({100, 60}).ID();
    // Place mouse cursor overlapping main_win at (130, 75) -> solid mouse at (132, 77)
    const unsigned int mouse_id = mgr.NewLayer().SetWindow(mouse_win).Move({130, 75}).ID();

    mgr.UpDown(bg_id, 0);
    mgr.UpDown(con_id, 1);
    mgr.UpDown(main_id, 2);
    mgr.UpDown(mouse_id, 3);

    // Initial full draw
    mgr.Draw({{0, 0}, {static_cast<int>(kWidth), static_cast<int>(kHeight)}});
    assert(screen.LastCopiedPixels() == static_cast<size_t>(kWidth) * kHeight);
    assert(screen.ReadPixel(10, 10) == kConsoleColor);
    assert(screen.ReadPixel(250, 150) == kBgColor);
    assert(screen.ReadPixel(105, 65) == kWinColor);
    assert(screen.ReadPixel(132, 77) == kMouseColor);

    // Populate single_screen with the same initial state
    single_screen.Copy({0, 0}, mgr.BackBuffer(), {{0, 0}, {static_cast<int>(kWidth), static_cast<int>(kHeight)}});

    // 3. Compare single-buffer Draw(main_id) vs double-buffer Draw(main_id) under mouse overlap
    const int single_exposures = mgr.SimulateSingleBufferDrawIdExposures(
        main_id, single_screen, {132, 77}, kMouseColor);
    assert(single_exposures == 1);  // main_win overwrote mouse in VRAM before mouse was redrawn!

    // Double-buffer Draw(main_id): update counter pixel inside main_win and redraw
    const PixelColor kCounterTextColor{0, 0, 0};
    FillRectangle(*main_win->Writer(), {10, 10}, {20, 12}, kCounterTextColor);
    const size_t before_copied = screen.TotalCopiedPixels();
    mgr.Draw(main_id);
    const size_t copied_this_call = screen.TotalCopiedPixels() - before_copied;

    // Only main_win bounding box (120 * 50 = 6000 px) transferred atomically to VRAM!
    assert(copied_this_call == 120 * 50);
    // Final pixels on screen match back_buffer_ everywhere
    assert(screen.ReadPixel(115, 75) == kCounterTextColor);
    assert(screen.ReadPixel(132, 77) == kMouseColor);
    assert(screen.ReadPixel(10, 10) == kConsoleColor);
    assert(screen.ReadPixel(250, 150) == kBgColor);

    // 4. Test Move(mouse_id) to partially off-screen position (clamped/clipped via 3-way intersection)
    mgr.Move(mouse_id, {315, 190});
    assert(screen.ReadPixel(132, 77) == kWinColor);      // Old mouse spot cleanly restored
    assert(screen.ReadPixel(317, 192) == kMouseColor);   // New mouse spot visible at screen edge

    // Verify full pixel-by-pixel equivalence between screen (VRAM) and back_buffer_ (RAM)
    for (int y = 0; y < static_cast<int>(kHeight); ++y) {
        for (int x = 0; x < static_cast<int>(kWidth); ++x) {
            assert(screen.ReadPixel(x, y) == mgr.BackBuffer().ReadPixel(x, y));
        }
    }

    std::cout << "=== Chapter 10.5 Double Buffering & Back Buffer Verification ===\n";
    std::cout << "VRAM Stride (padded): " << kHardwareScanLine << " px | BackBuffer Stride (compact): "
              << back_cfg.pixels_per_scan_line << " px\n";
    std::cout << "BackBuffer RAM Size: " << mgr.BackBuffer().InternalBufferBytes() << " bytes\n";
    std::cout << "Single-Buffer Draw(main_id) VRAM transient mouse flicker exposures: "
              << single_exposures << "\n";
    std::cout << "Double-Buffer Draw(main_id) VRAM transient flicker exposures: 0\n";
    std::cout << "Partial Atomic Copy Pixels on Draw(main_id): " << copied_this_call
              << " px (" << (copied_this_call * 4) << " bytes)\n";
    std::cout << "All C++20 Chapter 10.5 Assertions PASSED Successfully!\n";
    return 0;
}
