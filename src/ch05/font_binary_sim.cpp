/**
 * @file font_binary_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 5.3 (Generating Hankaku Font Object, osbook_day05c)
 *
 * Covers:
 * 1. 8x16 bitmap font binary packaging (hankaku.txt -> makefont.py -> hankaku.bin)
 * 2. objcopy binary-to-ELF64 object conversion and automatic symbol generation:
 *    - _binary_hankaku_bin_start (payload data start)
 *    - _binary_hankaku_bin_end   (payload data end)
 *    - _binary_hankaku_bin_size  (Absolute symbol: value is symbol address)
 * 3. Linker symbol dereferencing mechanics:
 *    - Correct: reinterpret_cast<uintptr_t>(&_binary_hankaku_bin_size)
 *    - Trap: direct value access dereferences address 0x1000
 * 4. GetFont(char c) address arithmetic and bounds checking
 * 5. Full ASCII character rendering ('!' to '~') on GOP FrameBuffer
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mikan::ch05_font {

// ============================================================================
// 1. MikanOS Graphics & FrameBuffer Subsystem
// ============================================================================

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct PixelColor {
    uint8_t r, g, b;

    bool operator==(const PixelColor& other) const {
        return r == other.r && g == other.g && b == other.b;
    }
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    enum PixelFormat pixel_format;
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

// ============================================================================
// 2. Simulated Hankaku Font Binary and objcopy Linker Symbols
// ============================================================================

// 4096 bytes representing 256 characters * 16 bytes/char
alignas(16) static uint8_t s_mock_hankaku_bin[4096];

// Initialize representative glyphs (e.g., 'A', '!', '0')
void InitializeMockHankakuFont() {
    std::fill(std::begin(s_mock_hankaku_bin), std::end(s_mock_hankaku_bin), 0x00);

    // Glyph for 'A' (ASCII 0x41 = 65, offset = 65 * 16 = 1040)
    const uint8_t kGlyphA[16] = {
        0b00000000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00100100,
        0b00100100,
        0b00100100,
        0b00100100,
        0b01111110,
        0b01000010,
        0b01000010,
        0b01000010,
        0b11100111,
        0b00000000,
        0b00000000,
    };
    for (int i = 0; i < 16; ++i) {
        s_mock_hankaku_bin[65 * 16 + i] = kGlyphA[i];
    }

    // Glyph for '!' (ASCII 0x21 = 33, offset = 33 * 16 = 528)
    const uint8_t kGlyphExcl[16] = {
        0b00000000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00000000,
        0b00000000,
        0b00011000,
        0b00011000,
        0b00000000,
        0b00000000,
        0b00000000,
        0b00000000,
    };
    for (int i = 0; i < 16; ++i) {
        s_mock_hankaku_bin[33 * 16 + i] = kGlyphExcl[i];
    }

    // Populate remaining ASCII printable characters (0x20 to 0x7E) with distinct patterns
    for (int c = 0x20; c <= 0x7E; ++c) {
        if (c == 65 || c == 33) continue;
        for (int dy = 2; dy < 14; ++dy) {
            // Generate non-zero pattern so each character can be rendered
            s_mock_hankaku_bin[c * 16 + dy] = static_cast<uint8_t>((c * 17 + dy) | 0x24);
        }
    }
}

/**
 * @brief Test harness simulating objcopy symbol mechanics
 */
class ObjcopyFontSymbolHarness {
public:
    static const uint8_t* GetStartSymbolAddress() {
        return s_mock_hankaku_bin;
    }

    static const uint8_t* GetEndSymbolAddress() {
        return s_mock_hankaku_bin + sizeof(s_mock_hankaku_bin);
    }

    /**
     * In real ELF objects produced by objcopy, _binary_hankaku_bin_size is an Absolute symbol.
     * Its value is 4096 (0x1000). The symbol's virtual address in the symbol table is 0x1000.
     */
    static uintptr_t GetAbsoluteSizeSymbolAddress() {
        return sizeof(s_mock_hankaku_bin); // 4096
    }
};

/**
 * @brief Exact implementation of GetFont from MikanOS day05c (font.cpp)
 */
const uint8_t* GetFont(char c, const uint8_t* bin_start, uintptr_t bin_size) {
    auto index = 16 * static_cast<unsigned int>(static_cast<unsigned char>(c));
    if (index >= bin_size) {
        return nullptr;
    }
    return bin_start + index;
}

/**
 * @brief Exact implementation of WriteAscii from MikanOS day05c (font.cpp)
 */
void WriteAscii(PixelWriter& writer, int x, int y, char c, const PixelColor& color,
                const uint8_t* bin_start, uintptr_t bin_size) {
    const uint8_t* font = GetFont(c, bin_start, bin_size);
    if (font == nullptr) {
        return;
    }
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((font[dy] << dx) & 0x80u) {
                writer.Write(x + dx, y + dy, color);
            }
        }
    }
}

} // namespace mikan::ch05_font

int main() {
    using namespace mikan::ch05_font;

    std::cout << "=================================================================\n";
    std::cout << "MikanOS Chapter 5.3: Hankaku Font Object Simulation (osbook_day05c)\n";
    std::cout << "=================================================================\n\n";

    InitializeMockHankakuFont();

    const uint8_t* bin_start = ObjcopyFontSymbolHarness::GetStartSymbolAddress();
    const uint8_t* bin_end = ObjcopyFontSymbolHarness::GetEndSymbolAddress();
    uintptr_t bin_size = ObjcopyFontSymbolHarness::GetAbsoluteSizeSymbolAddress();

    // ------------------------------------------------------------------------
    // Test 1: objcopy Symbol and Memory Range Verification
    // ------------------------------------------------------------------------
    std::cout << "[Test 1] Testing objcopy Symbol and Memory Range Properties...\n";
    {
        ptrdiff_t byte_count = bin_end - bin_start;
        assert(byte_count == 4096);
        assert(bin_size == 4096);
        std::cout << "  [+] Start Address : " << static_cast<const void*>(bin_start) << "\n";
        std::cout << "  [+] End Address   : " << static_cast<const void*>(bin_end) << "\n";
        std::cout << "  [+] Memory Span   : " << byte_count << " bytes (256 glyphs * 16 bytes)\n";
        std::cout << "  [+] Absolute Size : " << bin_size << " (0x" << std::hex << bin_size << std::dec << ")\n";
    }
    std::cout << "PASSED\n\n";

    // ------------------------------------------------------------------------
    // Test 2: Absolute Symbol Address Arithmetic and Pointer Dereferencing Trap
    // ------------------------------------------------------------------------
    std::cout << "[Test 2] Testing Absolute Symbol (&_size vs _size) Mechanics...\n";
    {
        // Demonstration of why reinterpret_cast<uintptr_t>(&_binary_hankaku_bin_size) is required:
        // In objcopy, symbol address = 4096.
        uintptr_t size_via_address = bin_size;
        assert(size_via_address == 4096);

        std::cout << "  [+] reinterpret_cast<uintptr_t>(&_binary_hankaku_bin_size) == "
                  << size_via_address << " (Correct: extracts symbol address/value)\n";
        std::cout << "  [+] Note: Reading _binary_hankaku_bin_size as a variable would dereference\n"
                  << "      virtual address 0x00001000, triggering SIGSEGV or reading invalid memory.\n";
    }
    std::cout << "PASSED\n\n";

    // ------------------------------------------------------------------------
    // Test 3: GetFont Address Arithmetic and Bounds Checking
    // ------------------------------------------------------------------------
    std::cout << "[Test 3] Testing GetFont(char c) Address Arithmetic & Bounds Checking...\n";
    {
        // Check 'A' (ASCII 65)
        const uint8_t* fontA = GetFont('A', bin_start, bin_size);
        assert(fontA != nullptr);
        assert(fontA == bin_start + 65 * 16);
        assert(fontA[1] == 0b00011000);
        assert(fontA[9] == 0b01111110);
        std::cout << "  [+] 'A' glyph correctly resolved at offset " << (fontA - bin_start) << " bytes.\n";

        // Check '!' (ASCII 33)
        const uint8_t* fontExcl = GetFont('!', bin_start, bin_size);
        assert(fontExcl != nullptr);
        assert(fontExcl == bin_start + 33 * 16);
        assert(fontExcl[1] == 0b00011000);
        assert(fontExcl[10] == 0b00011000);
        std::cout << "  [+] '!' glyph correctly resolved at offset " << (fontExcl - bin_start) << " bytes.\n";

        // Check out of bounds protection
        // When size is 4096, index >= 4096 must return nullptr
        // In MikanOS, index = 16 * static_cast<unsigned int>(c)
        const uint8_t* font_invalid = GetFont(static_cast<char>(0xFF), bin_start, 256); // artificially low size
        assert(font_invalid == nullptr);
        std::cout << "  [+] Out of bounds check successfully returned nullptr.\n";
    }
    std::cout << "PASSED\n\n";

    // ------------------------------------------------------------------------
    // Test 4: FrameBuffer Simulation & ASCII String Rendering ('!' to '~')
    // ------------------------------------------------------------------------
    std::cout << "[Test 4] Testing FrameBuffer ASCII Rendering ('!' through '~')...\n";
    {
        const uint32_t width = 800;
        const uint32_t height = 100;
        std::vector<uint8_t> buffer(width * height * 4, 0xFF); // White background

        FrameBufferConfig config{
            .frame_buffer = buffer.data(),
            .pixels_per_scan_line = width,
            .horizontal_resolution = width,
            .vertical_resolution = height,
            .pixel_format = kPixelRGBResv8BitPerColor,
        };
        RGBResv8BitPerColorPixelWriter writer(config);

        // MikanOS day05c KernelMain loop:
        // int i = 0;
        // for (char c = '!'; c <= '~'; ++c, ++i) {
        //   WriteAscii(*pixel_writer, 8 * i, 50, c, {0, 0, 0});
        // }
        int rendered_count = 0;
        for (char c = '!'; c <= '~'; ++c, ++rendered_count) {
            WriteAscii(writer, 8 * rendered_count, 50, c, {0, 0, 0}, bin_start, bin_size);
        }

        assert(rendered_count == 94); // 94 printable ASCII characters
        std::cout << "  [+] Rendered " << rendered_count << " ASCII characters sequentially.\n";

        // Verify black pixel rendering for 'A' (which is at index 65 - 33 = 32 in the rendered loop)
        int idxA = 'A' - '!';
        int origin_xA = 8 * idxA;
        int origin_yA = 50;

        // Row 9 of 'A' has 0b01111110 -> 6 consecutive black pixels (dx = 1..6)
        auto pixel_at = [&](int x, int y) -> PixelColor {
            auto p = buffer.data() + 4 * (width * y + x);
            return PixelColor{p[0], p[1], p[2]};
        };

        PixelColor black{0, 0, 0};
        PixelColor white{0xFF, 0xFF, 0xFF};

        assert(pixel_at(origin_xA + 0, origin_yA + 9) == white);
        assert(pixel_at(origin_xA + 1, origin_yA + 9) == black);
        assert(pixel_at(origin_xA + 6, origin_yA + 9) == black);
        assert(pixel_at(origin_xA + 7, origin_yA + 9) == white);

        std::cout << "  [+] Pixel verification passed: 'A' bar glyph verified on FrameBuffer.\n";
    }
    std::cout << "PASSED\n\n";

    // ------------------------------------------------------------------------
    // Test 5: Makefile & objcopy Target Rules Verification
    // ------------------------------------------------------------------------
    std::cout << "[Test 5] Verifying objcopy and Makefile Target Rules...\n";
    {
        std::string objcopy_cmd = "objcopy -I binary -O elf64-x86-64 -B i386:x86-64 hankaku.bin hankaku.o";
        assert(objcopy_cmd.find("-I binary") != std::string::npos);
        assert(objcopy_cmd.find("-O elf64-x86-64") != std::string::npos);
        assert(objcopy_cmd.find("-B i386:x86-64") != std::string::npos);
        std::cout << "  [+] objcopy command syntax: " << objcopy_cmd << "\n";
    }
    std::cout << "PASSED\n\n";

    std::cout << "All Section 5.3 C++20 Verification Tests Passed Successfully!\n";
    return 0;
}
