/**
 * @file os_chapter9_layer_intro.cpp
 * @brief Chapter 9.1: Introduction to Layer Compositing Simulation (C++20)
 *
 * Demonstrates:
 * 1. Vector2D<T> 2D coordinate arithmetic and transformations
 * 2. PixelColor and transparent color keying (std::optional<PixelColor>)
 * 3. Window & WindowWriter 2D offscreen buffer with clipping
 * 4. Layer position management and DrawTo pipeline
 * 5. LayerManager z-order stack (UpDown / Hide) and Painter's Algorithm compositing
 * 6. Background destruction problem recreation vs Layer compositing resolution
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mikanos::study {

// 1. Vector2D template
template <typename T>
struct Vector2D {
    T x, y;

    template <typename U>
    Vector2D<T>& operator+=(const Vector2D<U>& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    template <typename U>
    Vector2D<T>& operator-=(const Vector2D<U>& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }
};

template <typename T, typename U>
auto operator+(const Vector2D<T>& lhs, const Vector2D<U>& rhs)
    -> Vector2D<decltype(lhs.x + rhs.x)> {
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}

template <typename T, typename U>
auto operator-(const Vector2D<T>& lhs, const Vector2D<U>& rhs)
    -> Vector2D<decltype(lhs.x - rhs.x)> {
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}

// 2. PixelColor definition
struct PixelColor {
    uint8_t r, g, b;

    constexpr bool operator==(const PixelColor& rhs) const noexcept {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }

    constexpr bool operator!=(const PixelColor& rhs) const noexcept {
        return !(*this == rhs);
    }
};

// 3. PixelWriter interface and MemoryPixelWriter
class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

class MemoryPixelWriter : public PixelWriter {
public:
    MemoryPixelWriter(int width, int height, PixelColor bg_color = {0, 0, 0})
        : width_{width}, height_{height}, buffer_(height, std::vector<PixelColor>(width, bg_color)) {}

    void Write(int x, int y, const PixelColor& c) override {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            buffer_[y][x] = c;
        }
    }

    int Width() const override { return width_; }
    int Height() const override { return height_; }

    const PixelColor& At(int x, int y) const {
        assert(x >= 0 && x < width_ && y >= 0 && y < height_);
        return buffer_[y][x];
    }

    PixelColor& At(int x, int y) {
        assert(x >= 0 && x < width_ && y >= 0 && y < height_);
        return buffer_[y][x];
    }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> buffer_;
};

// 4. Window and WindowWriter
class Window {
public:
    class WindowWriter : public PixelWriter {
    public:
        explicit WindowWriter(Window& window) : window_{window} {}
        void Write(int x, int y, const PixelColor& c) override {
            window_.At(x, y) = c;
        }
        int Width() const override { return window_.Width(); }
        int Height() const override { return window_.Height(); }

    private:
        Window& window_;
    };

    Window(int width, int height) : width_{width}, height_{height} {
        data_.resize(height, std::vector<PixelColor>(width, {0, 0, 0}));
    }

    ~Window() = default;
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    void DrawTo(PixelWriter& writer, Vector2D<int> position) {
        if (!transparent_color_) {
            for (int y = 0; y < Height(); ++y) {
                for (int x = 0; x < Width(); ++x) {
                    writer.Write(position.x + x, position.y + y, At(x, y));
                }
            }
            return;
        }

        const auto tc = transparent_color_.value();
        for (int y = 0; y < Height(); ++y) {
            for (int x = 0; x < Width(); ++x) {
                const auto c = At(x, y);
                if (c != tc) {
                    writer.Write(position.x + x, position.y + y, c);
                }
            }
        }
    }

    void SetTransparentColor(std::optional<PixelColor> c) {
        transparent_color_ = c;
    }

    std::optional<PixelColor> TransparentColor() const {
        return transparent_color_;
    }

    WindowWriter* Writer() { return &writer_; }

    PixelColor& At(int x, int y) {
        assert(x >= 0 && x < width_ && y >= 0 && y < height_);
        return data_[y][x];
    }

    const PixelColor& At(int x, int y) const {
        assert(x >= 0 && x < width_ && y >= 0 && y < height_);
        return data_[y][x];
    }

    int Width() const { return width_; }
    int Height() const { return height_; }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> data_{};
    WindowWriter writer_{*this};
    std::optional<PixelColor> transparent_color_{std::nullopt};
};

// 5. Layer
class Layer {
public:
    explicit Layer(unsigned int id = 0) : id_{id}, pos_{0, 0} {}
    unsigned int ID() const { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }

    std::shared_ptr<Window> GetWindow() const { return window_; }

    Vector2D<int> GetPosition() const { return pos_; }

    Layer& Move(Vector2D<int> pos) {
        pos_ = pos;
        return *this;
    }

    Layer& MoveRelative(Vector2D<int> pos_diff) {
        pos_ += pos_diff;
        return *this;
    }

    void DrawTo(PixelWriter& writer) const {
        if (window_) {
            window_->DrawTo(writer, pos_);
        }
    }

private:
    unsigned int id_;
    Vector2D<int> pos_;
    std::shared_ptr<Window> window_{nullptr};
};

// 6. LayerManager
class LayerManager {
public:
    void SetWriter(PixelWriter* writer) {
        writer_ = writer;
    }

    Layer& NewLayer() {
        ++latest_id_;
        return *layers_.emplace_back(std::make_unique<Layer>(latest_id_));
    }

    void Draw() const {
        assert(writer_ != nullptr);
        for (auto* layer : layer_stack_) {
            layer->DrawTo(*writer_);
        }
    }

    void Move(unsigned int id, Vector2D<int> new_position) {
        if (auto* layer = FindLayer(id)) {
            layer->Move(new_position);
        }
    }

    void MoveRelative(unsigned int id, Vector2D<int> pos_diff) {
        if (auto* layer = FindLayer(id)) {
            layer->MoveRelative(pos_diff);
        }
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
        if (!layer) return;

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
        if (!layer) return;
        auto pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (pos != layer_stack_.end()) {
            layer_stack_.erase(pos);
        }
    }

    Layer* FindLayer(unsigned int id) {
        auto it = std::find_if(layers_.begin(), layers_.end(),
            [id](const std::unique_ptr<Layer>& elem) { return elem->ID() == id; });
        return (it != layers_.end()) ? it->get() : nullptr;
    }

    size_t StackSize() const { return layer_stack_.size(); }
    const std::vector<Layer*>& Stack() const { return layer_stack_; }

private:
    PixelWriter* writer_{nullptr};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    unsigned int latest_id_{0};
};

} // namespace mikanos::study

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    using namespace mikanos::study;

    std::cout << "=========================================================\n";
    std::cout << " Chapter 9.1: Introduction to Layer Compositing Sim (C++20)\n";
    std::cout << "=========================================================\n";

    // 1. Vector2D Arithmetic
    Vector2D<int> p1{100, 200};
    Vector2D<int> diff{15, -20};
    auto p2 = p1 + diff;
    assert(p2.x == 115 && p2.y == 180);
    p1 += diff;
    assert(p1.x == 115 && p1.y == 180);
    p1 -= diff;
    assert(p1.x == 100 && p1.y == 200);
    std::cout << "[PASS] Vector2D<T> coordinate operations verified.\n";

    // 2. Window and WindowWriter Verification
    constexpr int kWinW = 20;
    constexpr int kWinH = 20;
    auto test_win = std::make_shared<Window>(kWinW, kWinH);
    auto* win_writer = test_win->Writer();
    assert(win_writer->Width() == kWinW);
    assert(win_writer->Height() == kWinH);

    // Fill test window with Red {255, 0, 0} and a Blue center dot {0, 0, 255}
    PixelColor c_red{255, 0, 0};
    PixelColor c_blue{0, 0, 255};
    PixelColor c_green{0, 255, 0};
    for (int y = 0; y < kWinH; ++y) {
        for (int x = 0; x < kWinW; ++x) {
            win_writer->Write(x, y, c_red);
        }
    }
    win_writer->Write(10, 10, c_blue);
    assert(test_win->At(0, 0) == c_red);
    assert(test_win->At(10, 10) == c_blue);
    std::cout << "[PASS] Window buffer & WindowWriter verified.\n";

    // 3. Screen Compositing & Transparent Color Keying
    constexpr int kScrW = 100;
    constexpr int kScrH = 100;
    PixelColor c_bg{45, 118, 237}; // Desktop blue
    MemoryPixelWriter screen(kScrW, kScrH, c_bg);

    // Set Red as transparent color
    test_win->SetTransparentColor(c_red);
    test_win->DrawTo(screen, {20, 30});

    // Verification:
    // (20, 30) was Red, so it should be skipped and remain c_bg
    assert(screen.At(20, 30) == c_bg);
    // (20+10, 30+10) = (30, 40) was Blue, so it should be written
    assert(screen.At(30, 40) == c_blue);
    std::cout << "[PASS] Window DrawTo with Transparent Color verified.\n";

    // 4. LayerManager and UpDown / Hide Verification
    LayerManager mgr;
    mgr.SetWriter(&screen);

    auto bg_win = std::make_shared<Window>(kScrW, kScrH);
    for (int y = 0; y < kScrH; ++y) {
        for (int x = 0; x < kScrW; ++x) {
            bg_win->At(x, y) = c_bg;
        }
    }

    auto fg_win = std::make_shared<Window>(10, 10);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            fg_win->At(x, y) = c_green;
        }
    }

    auto& bg_layer = mgr.NewLayer();
    bg_layer.SetWindow(bg_win).Move({0, 0});
    unsigned int bg_id = bg_layer.ID();

    auto& fg_layer = mgr.NewLayer();
    fg_layer.SetWindow(fg_win).Move({10, 10});
    unsigned int fg_id = fg_layer.ID();

    // Set UpDown order: bg_layer at 0, fg_layer at 1
    mgr.UpDown(bg_id, 0);
    mgr.UpDown(fg_id, 1);
    assert(mgr.StackSize() == 2);
    assert(mgr.Stack()[0]->ID() == bg_id);
    assert(mgr.Stack()[1]->ID() == fg_id);

    // Draw and verify green appears at (10, 10)
    mgr.Draw();
    assert(screen.At(10, 10) == c_green);
    assert(screen.At(0, 0) == c_bg);

    // Invert order: fg_layer at 0, bg_layer at 1 (bg covers fg)
    mgr.UpDown(bg_id, 1);
    mgr.UpDown(fg_id, 0);
    assert(mgr.Stack()[0]->ID() == fg_id);
    assert(mgr.Stack()[1]->ID() == bg_id);

    mgr.Draw();
    // Since bg_layer is on top, (10, 10) is now overwritten by c_bg!
    assert(screen.At(10, 10) == c_bg);
    std::cout << "[PASS] LayerManager UpDown z-order reordering verified.\n";

    // 5. Hide and MoveRelative
    mgr.Hide(bg_id);
    assert(mgr.StackSize() == 1);
    assert(mgr.Stack()[0]->ID() == fg_id);

    mgr.MoveRelative(fg_id, {5, 5});
    assert(mgr.FindLayer(fg_id)->GetPosition().x == 15);
    assert(mgr.FindLayer(fg_id)->GetPosition().y == 15);

    mgr.Draw();
    assert(screen.At(15, 15) == c_green);
    std::cout << "[PASS] LayerManager Hide and MoveRelative verified.\n";

    // 6. Background Destruction Problem vs Layer Compositing
    // In naive approach without layers, moving a cursor at (50, 50) over text
    // and erasing it with background color destroys text permanently.
    // In Layer approach:
    MemoryPixelWriter test_screen(100, 100, c_bg);
    LayerManager compositing_mgr;
    compositing_mgr.SetWriter(&test_screen);

    // Layer 0: Background with text "A" (drawn as white pixels)
    PixelColor c_white{255, 255, 255};
    auto text_layer_win = std::make_shared<Window>(100, 100);
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            text_layer_win->At(x, y) = c_bg;
        }
    }
    text_layer_win->At(50, 50) = c_white; // Text pixel

    // Layer 1: Cursor (10x10) with transparent border and black core
    PixelColor c_black{0, 0, 0};
    PixelColor c_cursor_trans{99, 99, 99};
    auto cursor_win = std::make_shared<Window>(10, 10);
    cursor_win->SetTransparentColor(c_cursor_trans);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            cursor_win->At(x, y) = c_cursor_trans;
        }
    }
    cursor_win->At(0, 0) = c_black; // Cursor pointer tip

    auto& l_text = compositing_mgr.NewLayer();
    l_text.SetWindow(text_layer_win).Move({0, 0});
    auto& l_cursor = compositing_mgr.NewLayer();
    l_cursor.SetWindow(cursor_win).Move({50, 50}); // Overlaps text!

    compositing_mgr.UpDown(l_text.ID(), 0);
    compositing_mgr.UpDown(l_cursor.ID(), 1);

    // Initial render: (50, 50) is cursor tip (Black)
    compositing_mgr.Draw();
    assert(test_screen.At(50, 50) == c_black);

    // Move cursor away to (70, 70) and redraw
    compositing_mgr.Move(l_cursor.ID(), {70, 70});
    compositing_mgr.Draw();

    // Crucial check: (50, 50) is RESTORED to text pixel (White)! Not destroyed!
    assert(test_screen.At(50, 50) == c_white);
    assert(test_screen.At(70, 70) == c_black);
    std::cout << "[PASS] Background destruction completely resolved via Layer Compositing!\n";

    std::cout << "✅ All Chapter 9.1 Layer Compositing C++20 tests passed successfully!\n";
    return 0;
}
