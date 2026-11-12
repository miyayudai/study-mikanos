/**
 * @file os_chapter9_window_layer.cpp
 * @brief Chapter 9.3: Window, Layer, and LayerManager Architecture with Smart Pointer Ownership (C++20)
 *
 * Demonstrates:
 * 1. Window class with inner WindowWriter implementing PixelWriter interface
 * 2. 2D offscreen pixel buffer (std::vector<std::vector<PixelColor>>) and transparent color keying
 * 3. Layer class wrapping coordinates and shared Window ownership (std::shared_ptr<Window>)
 * 4. LayerManager managing Layer pool with exclusive ownership (std::vector<std::unique_ptr<Layer>>)
 *    and z-order rendering stack with non-owning observers (std::vector<Layer*>)
 * 5. UpDown algorithm: insertion, reordering, clipping, and negative Hide()
 * 6. Column 9.1: C++ Smart Pointers (unique_ptr, shared_ptr, use_count, control block, and non-owning raw pointers)
 * 7. Comprehensive pixel compositing and boundary assertions
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

// =============================================================================
// 1. Basic Geometric and Color Types
// =============================================================================

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
bool operator==(const Vector2D<T>& lhs, const Vector2D<U>& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

struct PixelColor {
    uint8_t r, g, b;

    bool operator==(const PixelColor& rhs) const = default;
};

// Abstract PixelWriter interface (System V ABI polymorphic base)
class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

// Simulated Screen VRAM PixelWriter
class ScreenBufferWriter : public PixelWriter {
public:
    ScreenBufferWriter(int width, int height, PixelColor bg_color = {45, 118, 237})
        : width_{width}, height_{height}, buffer_(height, std::vector<PixelColor>(width, bg_color)) {}

    void Write(int x, int y, const PixelColor& c) override {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            buffer_[y][x] = c;
        }
    }

    int Width() const override { return width_; }
    int Height() const override { return height_; }

    const PixelColor& At(int x, int y) const {
        return buffer_[y][x];
    }

    void Clear(PixelColor color) {
        for (auto& row : buffer_) {
            std::fill(row.begin(), row.end(), color);
        }
    }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> buffer_;
};

// =============================================================================
// 2. Window and WindowWriter Classes (osbook_day09a: kernel/window.hpp)
// =============================================================================

class Window {
public:
    /**
     * @brief Inner class providing PixelWriter interface linked to this Window.
     */
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
        data_.resize(height);
        for (int y = 0; y < height; ++y) {
            data_[y].resize(width, PixelColor{0, 0, 0});
        }
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

    WindowWriter* Writer() {
        return &writer_;
    }

    PixelColor& At(int x, int y) {
        return data_[y][x];
    }

    const PixelColor& At(int x, int y) const {
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

// =============================================================================
// 3. Layer and LayerManager Classes (osbook_day09a: kernel/layer.hpp)
// =============================================================================

class Layer {
public:
    explicit Layer(unsigned int id = 0) : id_{id}, pos_{0, 0} {}
    unsigned int ID() const { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }

    std::shared_ptr<Window> GetWindow() const {
        return window_;
    }

    Vector2D<int> GetPosition() const {
        return pos_;
    }

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
        if (!writer_) return;
        for (auto layer : layer_stack_) {
            layer->DrawTo(*writer_);
        }
    }

    void Move(unsigned int id, Vector2D<int> new_position) {
        if (auto layer = FindLayer(id)) {
            layer->Move(new_position);
        }
    }

    void MoveRelative(unsigned int id, Vector2D<int> pos_diff) {
        if (auto layer = FindLayer(id)) {
            layer->MoveRelative(pos_diff);
        }
    }

    void UpDown(unsigned int id, int new_height) {
        if (new_height < 0) {
            Hide(id);
            return;
        }
        if (static_cast<size_t>(new_height) > layer_stack_.size()) {
            new_height = static_cast<int>(layer_stack_.size());
        }

        auto layer = FindLayer(id);
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
        auto layer = FindLayer(id);
        if (!layer) return;
        auto pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (pos != layer_stack_.end()) {
            layer_stack_.erase(pos);
        }
    }

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

    const std::vector<Layer*>& LayerStack() const {
        return layer_stack_;
    }

    size_t LayerCount() const {
        return layers_.size();
    }

private:
    PixelWriter* writer_{nullptr};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    unsigned int latest_id_{0};
};

} // namespace mikanos::study

// =============================================================================
// 4. Test Suite and Verification
// =============================================================================

int main() {
    using namespace mikanos::study;

    std::cout << "====================================================================\n";
    std::cout << " MikanOS Chapter 9.3: Window, Layer & LayerManager Architecture (C++20)\n";
    std::cout << "====================================================================\n\n";

    // -------------------------------------------------------------------------
    // Test 1: Window and WindowWriter Pixel Manipulation with Color Keying
    // -------------------------------------------------------------------------
    std::cout << "[Test 1] Window and WindowWriter Offscreen Buffer & Transparency...\n";
    {
        Window win(10, 8);
        assert(win.Width() == 10);
        assert(win.Height() == 8);

        auto writer = win.Writer();
        assert(writer != nullptr);
        assert(writer->Width() == 10);
        assert(writer->Height() == 8);

        // Fill background with key color
        const PixelColor transparent_key{99, 99, 99};
        win.SetTransparentColor(transparent_key);
        assert(win.TransparentColor().has_value());
        assert(win.TransparentColor().value() == transparent_key);

        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 10; ++x) {
                writer->Write(x, y, transparent_key);
            }
        }

        // Draw a red diagonal line
        const PixelColor red{255, 0, 0};
        for (int i = 0; i < 8; ++i) {
            writer->Write(i, i, red);
        }

        // Render onto ScreenBufferWriter
        ScreenBufferWriter screen(20, 20, PixelColor{45, 118, 237});
        win.DrawTo(screen, {5, 5});

        // The diagonal line must be red
        for (int i = 0; i < 8; ++i) {
            assert(screen.At(5 + i, 5 + i) == red);
        }
        // Non-line pixel in window must be transparent (screen original color preserved)
        assert(screen.At(5, 6) == (PixelColor{45, 118, 237}));
        assert(screen.At(6, 5) == (PixelColor{45, 118, 237}));

        std::cout << "  -> PASSED: WindowWriter and transparent color skipping verified.\n";
    }

    // -------------------------------------------------------------------------
    // Test 2: Layer Coordinates and Delegation
    // -------------------------------------------------------------------------
    std::cout << "[Test 2] Layer Coordinate Transformation and Method Chaining...\n";
    {
        Layer layer(42);
        assert(layer.ID() == 42);
        assert(layer.GetPosition() == (Vector2D<int>{0, 0}));

        layer.Move({100, 50}).MoveRelative({10, -20});
        assert(layer.GetPosition() == (Vector2D<int>{110, 30}));

        auto win = std::make_shared<Window>(16, 16);
        layer.SetWindow(win);
        assert(layer.GetWindow() == win);

        std::cout << "  -> PASSED: Layer position and window assignment verified.\n";
    }

    // -------------------------------------------------------------------------
    // Test 3: LayerManager UpDown Stack Manipulation & Boundary Conditions
    // -------------------------------------------------------------------------
    std::cout << "[Test 3] LayerManager UpDown Stack Reordering & Clipping...\n";
    {
        LayerManager mgr;
        ScreenBufferWriter screen(100, 100);
        mgr.SetWriter(&screen);

        auto& l1 = mgr.NewLayer(); // id = 1
        auto& l2 = mgr.NewLayer(); // id = 2
        auto& l3 = mgr.NewLayer(); // id = 3
        assert(mgr.LayerCount() == 3);
        assert(l1.ID() == 1);
        assert(l2.ID() == 2);
        assert(l3.ID() == 3);

        // Initially layer_stack_ is empty
        assert(mgr.LayerStack().empty());

        // Insert l1 at 0 -> [1]
        mgr.UpDown(1, 0);
        assert(mgr.LayerStack().size() == 1);
        assert(mgr.LayerStack()[0]->ID() == 1);

        // Insert l2 at 1 (top) -> [1, 2]
        mgr.UpDown(2, 1);
        assert(mgr.LayerStack().size() == 2);
        assert(mgr.LayerStack()[0]->ID() == 1);
        assert(mgr.LayerStack()[1]->ID() == 2);

        // Insert l3 between l1 and l2 at height 1 -> [1, 3, 2]
        mgr.UpDown(3, 1);
        assert(mgr.LayerStack().size() == 3);
        assert(mgr.LayerStack()[0]->ID() == 1);
        assert(mgr.LayerStack()[1]->ID() == 3);
        assert(mgr.LayerStack()[2]->ID() == 2);

        // Reorder l1 to top (height 100 exceeds size 3 -> clipped to 3) -> [3, 2, 1]
        mgr.UpDown(1, 100);
        assert(mgr.LayerStack().size() == 3);
        assert(mgr.LayerStack()[0]->ID() == 3);
        assert(mgr.LayerStack()[1]->ID() == 2);
        assert(mgr.LayerStack()[2]->ID() == 1);

        // Hide l2 by negative height (-1) -> [3, 1]
        mgr.UpDown(2, -1);
        assert(mgr.LayerStack().size() == 2);
        assert(mgr.LayerStack()[0]->ID() == 3);
        assert(mgr.LayerStack()[1]->ID() == 1);

        // Explicit Hide on l3 -> [1]
        mgr.Hide(3);
        assert(mgr.LayerStack().size() == 1);
        assert(mgr.LayerStack()[0]->ID() == 1);

        std::cout << "  -> PASSED: UpDown insertion, reordering, clipping and Hide() verified.\n";
    }

    // -------------------------------------------------------------------------
    // Test 4: Column 9.1 Smart Pointer Ownership Semantics (unique_ptr vs shared_ptr)
    // -------------------------------------------------------------------------
    std::cout << "[Test 4] Column 9.1: std::unique_ptr vs std::shared_ptr Ownership...\n";
    {
        // 1. unique_ptr exclusive ownership test
        auto u1 = std::make_unique<int>(100);
        assert(u1 != nullptr);
        assert(*u1 == 100);

        // Transfer ownership via std::move
        auto u2 = std::move(u1);
        assert(u1 == nullptr); // u1 is now empty
        assert(u2 != nullptr);
        assert(*u2 == 100);

        // 2. shared_ptr reference count and control block test
        auto s1 = std::make_shared<Window>(20, 20);
        assert(s1.use_count() == 1);

        {
            auto s2 = s1; // copy shared_ptr
            assert(s1.use_count() == 2);
            assert(s2.use_count() == 2);
            assert(s1.get() == s2.get());

            Layer temp_layer(99);
            temp_layer.SetWindow(s2);
            assert(s1.use_count() == 3);
        } // s2 and temp_layer go out of scope

        assert(s1.use_count() == 1);

        std::cout << "  -> PASSED: Smart pointer ownership and reference count lifecycle verified.\n";
    }

    // -------------------------------------------------------------------------
    // Test 5: Full Compositing with Painter's Algorithm and Mouse Movement
    // -------------------------------------------------------------------------
    std::cout << "[Test 5] End-to-End Layer Compositing & Non-destructive Movement...\n";
    {
        ScreenBufferWriter vram(50, 40, PixelColor{0, 0, 0});
        LayerManager mgr;
        mgr.SetWriter(&vram);

        // Layer 1: Background (50x40, Blue, transparent=nullopt)
        auto bg_win = std::make_shared<Window>(50, 40);
        const PixelColor blue{0, 0, 255};
        for (int y = 0; y < 40; ++y) {
            for (int x = 0; x < 50; ++x) {
                bg_win->Writer()->Write(x, y, blue);
            }
        }
        // Write white text pixel at (20, 20)
        const PixelColor white{255, 255, 255};
        bg_win->Writer()->Write(20, 20, white);

        auto& bg_layer = mgr.NewLayer();
        bg_layer.SetWindow(bg_win).Move({0, 0});
        mgr.UpDown(bg_layer.ID(), 0);

        // Layer 2: Mouse Cursor (4x4, Green, transparent=Magenta)
        auto cur_win = std::make_shared<Window>(4, 4);
        const PixelColor magenta{255, 0, 255};
        const PixelColor green{0, 255, 0};
        cur_win->SetTransparentColor(magenta);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                cur_win->Writer()->Write(x, y, magenta);
            }
        }
        // Top-left pixel of cursor is Green
        cur_win->Writer()->Write(0, 0, green);

        auto& cur_layer = mgr.NewLayer();
        cur_layer.SetWindow(cur_win).Move({20, 20});
        mgr.UpDown(cur_layer.ID(), 1);

        // Frame 1: Cursor is directly on top of white text pixel at (20, 20)
        mgr.Draw();
        assert(vram.At(20, 20) == green); // Cursor pixel covers text
        assert(vram.At(21, 20) == blue);  // Magenta key is transparent, reveals background

        // Move cursor away to (30, 30)
        mgr.Move(cur_layer.ID(), {30, 30});
        mgr.Draw();

        // White text pixel at (20, 20) is preserved without background destruction!
        assert(vram.At(20, 20) == white);
        assert(vram.At(30, 30) == green);

        std::cout << "  -> PASSED: Non-destructive layer compositing verified.\n";
    }

    std::cout << "\n✅ ALL C++20 UNIT TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
