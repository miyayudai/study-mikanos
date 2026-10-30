/**
 * @file separate_compilation_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 5.2 (Separate Compilation & Linking, osbook_day05b)
 *
 * Covers:
 * 1. Modular decomposition: separation into graphics.hpp/cpp, font.hpp/cpp, and main.cpp
 * 2. Declaration vs Definition: function prototypes, vtable generation, and One Definition Rule (ODR)
 * 3. Include guard mechanisms: #pragma once behavior and translation unit scoping
 * 4. Symbol table and linker resolution: defined (T/R/D), local (t/r), and undefined (U) symbol resolution
 * 5. Makefile dependency generation: clang++ -MM dependency DAG and incremental rebuild detection
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace mikan::ch05_sep {

// ============================================================================
// 1. MikanOS Modular Subsystems (Day05b Architecture)
// ============================================================================

// --- frame_buffer_config.hpp ---
enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    enum PixelFormat pixel_format;
};

// --- graphics.hpp / graphics.cpp ---
struct PixelColor {
    uint8_t r, g, b;
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

// --- font.hpp / font.cpp ---
// File-scope constant in font.cpp (internal linkage, not exported to global symbol table)
namespace {
    const uint8_t kFontA[16] = {
        0b00000000, //
        0b00011000, //    **
        0b00011000, //    **
        0b00011000, //    **
        0b00011000, //    **
        0b00100100, //   *  *
        0b00100100, //   *  *
        0b00100100, //   *  *
        0b00100100, //   *  *
        0b01111110, //  ******
        0b01000010, //  *    *
        0b01000010, //  *    *
        0b01000010, //  *    *
        0b11100111, // ***  ***
        0b00000000, //
        0b00000000, //
    };
}

void WriteAscii(PixelWriter& writer, int x, int y, char c, const PixelColor& color) {
    if (c != 'A') {
        return;
    }
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((kFontA[dy] << dx) & 0x80u) {
                writer.Write(x + dx, y + dy, color);
            }
        }
    }
}

// ============================================================================
// 2. Linker & Symbol Resolution Simulation in C++20
// ============================================================================

enum class SymbolKind {
    DefinedText,      // 'T': Function in code section
    DefinedData,      // 'D': Initialized data
    DefinedRoData,    // 'R': Read-only constant / vtable
    LocalSymbol,      // 't' / 'r': Static or anonymous namespace symbol
    Undefined,        // 'U': External unresolved reference
};

struct SymbolEntry {
    std::string name;
    std::string mangled;
    SymbolKind kind;
    uint32_t size;
    uint64_t resolved_address{0};
};

class TranslationUnitSimulator {
public:
    explicit TranslationUnitSimulator(std::string name, std::string source_file)
        : name_{std::move(name)}, source_file_{std::move(source_file)} {}

    void AddDefinedSymbol(std::string mangled, std::string demangled, SymbolKind kind, uint32_t size) {
        symbols_.push_back({std::move(demangled), std::move(mangled), kind, size, 0});
    }

    void AddUndefinedSymbol(std::string mangled, std::string demangled) {
        symbols_.push_back({std::move(demangled), std::move(mangled), SymbolKind::Undefined, 0, 0});
    }

    const std::string& Name() const { return name_; }
    const std::string& SourceFile() const { return source_file_; }
    const std::vector<SymbolEntry>& Symbols() const { return symbols_; }

private:
    std::string name_;
    std::string source_file_;
    std::vector<SymbolEntry> symbols_;
};

class LinkerEngineSimulator {
public:
    explicit LinkerEngineSimulator(uint64_t base_address = 0x00100000)
        : base_address_{base_address} {}

    void AddObjectFile(const TranslationUnitSimulator& obj) {
        objects_.push_back(obj);
    }

    struct LinkResult {
        bool success{false};
        std::map<std::string, uint64_t> symbol_addresses;
        std::vector<std::string> duplicate_errors;
        std::vector<std::string> undefined_errors;
    };

    LinkResult Link() {
        LinkResult res;
        res.success = true;

        uint64_t current_addr = base_address_;
        std::map<std::string, std::string> global_definitions; // mangled -> obj_name

        // Pass 1: Collect definitions and check for duplicate symbols (ODR violation)
        for (const auto& obj : objects_) {
            for (const auto& sym : obj.Symbols()) {
                if (sym.kind != SymbolKind::Undefined && sym.kind != SymbolKind::LocalSymbol) {
                    if (global_definitions.count(sym.mangled)) {
                        res.success = false;
                        res.duplicate_errors.push_back(
                            "Duplicate symbol '" + sym.name + "' defined in " +
                            global_definitions[sym.mangled] + " and " + obj.Name());
                    } else {
                        global_definitions[sym.mangled] = obj.Name();
                        res.symbol_addresses[sym.mangled] = current_addr;
                        current_addr += std::max(sym.size, 16u);
                    }
                }
            }
        }

        // Pass 2: Resolve undefined symbols
        for (const auto& obj : objects_) {
            for (const auto& sym : obj.Symbols()) {
                if (sym.kind == SymbolKind::Undefined) {
                    if (!res.symbol_addresses.count(sym.mangled)) {
                        res.success = false;
                        res.undefined_errors.push_back(
                            "Undefined reference to '" + sym.name + "' from " + obj.Name());
                    }
                }
            }
        }

        return res;
    }

private:
    uint64_t base_address_;
    std::vector<TranslationUnitSimulator> objects_;
};

// ============================================================================
// 3. Makefile Dependency Graph Simulation
// ============================================================================

class MakeDependencyGraphSimulator {
public:
    void AddRule(std::string target_obj, std::string source_cpp, std::vector<std::string> headers) {
        rules_[target_obj] = {std::move(source_cpp), std::move(headers)};
    }

    std::string GenerateDotDFile(const std::string& target_obj) const {
        auto it = rules_.find(target_obj);
        if (it == rules_.end()) return "";

        std::ostringstream oss;
        oss << target_obj << ": " << it->second.source_cpp;
        for (const auto& h : it->second.headers) {
            oss << " \\\n  " << h;
        }
        oss << "\n";
        return oss.str();
    }

    std::vector<std::string> EvaluateRebuildTargets(const std::string& modified_file) const {
        std::vector<std::string> targets;
        for (const auto& [target, rule] : rules_) {
            if (rule.source_cpp == modified_file) {
                targets.push_back(target);
            } else {
                for (const auto& h : rule.headers) {
                    if (h == modified_file) {
                        targets.push_back(target);
                        break;
                    }
                }
            }
        }
        std::sort(targets.begin(), targets.end());
        return targets;
    }

private:
    struct Rule {
        std::string source_cpp;
        std::vector<std::string> headers;
    };
    std::map<std::string, Rule> rules_;
};

} // namespace mikan::ch05_sep

// ============================================================================
// 4. Test Runner & Verification Suite
// ============================================================================

int main() {
    using namespace mikan::ch05_sep;
    std::cout << "================================================================\n";
    std::cout << "MikanOS Chapter 5.2 Separate Compilation & Linking Verification\n";
    std::cout << "================================================================\n\n";

    // Test 1: Day05b Modular Subsystems Execution
    std::cout << "[Test 1] Testing Day05b Modular Subsystems Execution...\n";
    {
        constexpr uint32_t width = 80;
        constexpr uint32_t height = 80;
        std::vector<uint8_t> vram(width * height * 4, 255); // White background

        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = width,
            .horizontal_resolution = width,
            .vertical_resolution = height,
            .pixel_format = kPixelRGBResv8BitPerColor,
        };

        alignas(RGBResv8BitPerColorPixelWriter) char writer_buf[sizeof(RGBResv8BitPerColorPixelWriter)];
        auto* writer = new (writer_buf) RGBResv8BitPerColorPixelWriter(config);

        // Draw "AA" as in Day05b
        WriteAscii(*writer, 10, 10, 'A', {0, 0, 0});
        WriteAscii(*writer, 18, 10, 'A', {0, 0, 0});

        // Verify lit pixel at crossbar of first 'A': (10 + 3, 10 + 9) = (13, 19)
        uint32_t offset1 = 4 * (19 * width + 13);
        assert(vram[offset1 + 0] == 0 && vram[offset1 + 1] == 0 && vram[offset1 + 2] == 0);

        // Verify lit pixel at crossbar of second 'A': (18 + 3, 10 + 9) = (21, 19)
        uint32_t offset2 = 4 * (19 * width + 21);
        assert(vram[offset2 + 0] == 0 && vram[offset2 + 1] == 0 && vram[offset2 + 2] == 0);

        std::cout << "  [+] FrameBuffer rendering with modular PixelWriter & WriteAscii succeeded.\n";
    }
    std::cout << "PASSED\n\n";

    // Test 2: Symbol Resolution in Day05b Linker Simulation
    std::cout << "[Test 2] Testing Day05b Linker Symbol Resolution (graphics.o + font.o + main.o)...\n";
    {
        TranslationUnitSimulator graphics_o("graphics.o", "graphics.cpp");
        graphics_o.AddDefinedSymbol("_ZN30RGBResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
                                    "RGBResv8BitPerColorPixelWriter::Write()", SymbolKind::DefinedText, 48);
        graphics_o.AddDefinedSymbol("_ZN30BGRResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
                                    "BGRResv8BitPerColorPixelWriter::Write()", SymbolKind::DefinedText, 48);
        graphics_o.AddDefinedSymbol("_ZTV30RGBResv8BitPerColorPixelWriter",
                                    "vtable for RGBResv8BitPerColorPixelWriter", SymbolKind::DefinedRoData, 32);
        graphics_o.AddDefinedSymbol("_ZTV30BGRResv8BitPerColorPixelWriter",
                                    "vtable for BGRResv8BitPerColorPixelWriter", SymbolKind::DefinedRoData, 32);

        TranslationUnitSimulator font_o("font.o", "font.cpp");
        font_o.AddDefinedSymbol("_Z10WriteAsciiR11PixelWriteriicRK10PixelColor",
                                "WriteAscii()", SymbolKind::DefinedText, 96);
        font_o.AddDefinedSymbol("_ZL6kFontA", "kFontA (static)", SymbolKind::LocalSymbol, 16);

        TranslationUnitSimulator main_o("main.o", "main.cpp");
        main_o.AddDefinedSymbol("KernelMain", "KernelMain()", SymbolKind::DefinedText, 256);
        main_o.AddUndefinedSymbol("_ZN30RGBResv8BitPerColorPixelWriter5WriteEiiRK10PixelColor",
                                  "RGBResv8BitPerColorPixelWriter::Write()");
        main_o.AddUndefinedSymbol("_ZTV30RGBResv8BitPerColorPixelWriter",
                                  "vtable for RGBResv8BitPerColorPixelWriter");
        main_o.AddUndefinedSymbol("_Z10WriteAsciiR11PixelWriteriicRK10PixelColor",
                                  "WriteAscii()");

        LinkerEngineSimulator linker(0x00100000);
        linker.AddObjectFile(graphics_o);
        linker.AddObjectFile(font_o);
        linker.AddObjectFile(main_o);

        auto result = linker.Link();
        assert(result.success);
        assert(result.duplicate_errors.empty());
        assert(result.undefined_errors.empty());
        assert(result.symbol_addresses.count("KernelMain") == 1);
        assert(result.symbol_addresses.count("_Z10WriteAsciiR11PixelWriteriicRK10PixelColor") == 1);

        std::cout << "  [+] All 3 object files resolved successfully into kernel.elf without errors.\n";
        for (const auto& [mangled, addr] : result.symbol_addresses) {
            std::cout << "      0x" << std::hex << std::uppercase << addr << " : " << mangled << "\n";
        }
        std::cout << std::dec;
    }
    std::cout << "PASSED\n\n";

    // Test 3: Detecting ODR Violations (Multiple Definitions Error)
    std::cout << "[Test 3] Testing ODR Violation Detection (Header with implementation included in multiple .cpp)...\n";
    {
        TranslationUnitSimulator font_o("font.o", "font.cpp");
        font_o.AddDefinedSymbol("WriteAscii", "WriteAscii()", SymbolKind::DefinedText, 96);

        TranslationUnitSimulator bad_main_o("main.o", "main.cpp");
        bad_main_o.AddDefinedSymbol("WriteAscii", "WriteAscii()", SymbolKind::DefinedText, 96); // duplicate!

        LinkerEngineSimulator linker(0x00100000);
        linker.AddObjectFile(font_o);
        linker.AddObjectFile(bad_main_o);

        auto result = linker.Link();
        assert(!result.success);
        assert(result.duplicate_errors.size() == 1);
        std::cout << "  [+] Linker correctly caught ODR violation: " << result.duplicate_errors[0] << "\n";
    }
    std::cout << "PASSED\n\n";

    // Test 4: Makefile Dependency DAG and Incremental Rebuild Tracking
    std::cout << "[Test 4] Testing Makefile Dependency DAG and Incremental Rebuild Tracking...\n";
    {
        MakeDependencyGraphSimulator dag;
        dag.AddRule("graphics.o", "graphics.cpp", {"graphics.hpp", "frame_buffer_config.hpp"});
        dag.AddRule("font.o", "font.cpp", {"font.hpp", "graphics.hpp"});
        dag.AddRule("main.o", "main.cpp", {"frame_buffer_config.hpp", "graphics.hpp", "font.hpp"});

        // Verify .main.d generation
        std::string main_d = dag.GenerateDotDFile("main.o");
        assert(main_d.find("main.o: main.cpp") != std::string::npos);
        assert(main_d.find("graphics.hpp") != std::string::npos);
        assert(main_d.find("font.hpp") != std::string::npos);

        // Case A: Modify graphics.hpp -> Rebuild graphics.o, font.o, main.o
        auto targets1 = dag.EvaluateRebuildTargets("graphics.hpp");
        std::vector<std::string> expected1 = {"font.o", "graphics.o", "main.o"};
        assert(targets1 == expected1);

        // Case B: Modify font.cpp -> Rebuild only font.o
        auto targets2 = dag.EvaluateRebuildTargets("font.cpp");
        std::vector<std::string> expected2 = {"font.o"};
        assert(targets2 == expected2);

        // Case C: Modify font.hpp -> Rebuild font.o and main.o (not graphics.o)
        auto targets3 = dag.EvaluateRebuildTargets("font.hpp");
        std::vector<std::string> expected3 = {"font.o", "main.o"};
        assert(targets3 == expected3);

        std::cout << "  [+] Makefile dependency evaluation matches clang++ -MM behavior exactly.\n";
    }
    std::cout << "PASSED\n\n";

    std::cout << "All Section 5.2 C++20 Verification Tests Passed Successfully!\n";
    return 0;
}
