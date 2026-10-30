/**
 * @file uefi_hello_world_sim.cpp
 * @brief Standalone C++20 verification tool for Section 2.2: Hello World with EDK II (osbook_day02a) & Preprocessor Expansion.
 *
 * Demonstrates and verifies:
 * 1. UEFI Type System & ABI: EFI_STATUS, EFI_HANDLE, CHAR16 (UTF-16LE), and EFI_SYSTEM_TABLE structures.
 * 2. UTF-16LE string encoding, Little-Endian byte order analysis, and Print(L"...") formatting.
 * 3. Deterministic C Preprocessor simulation:
 *    - #include <...> vs #include "..." resolution order.
 *    - Multiple-inclusion guards (#ifndef / #define / #endif) preventing symbol redefinitions.
 *    - Include tree depth tracking and code bloat analysis (from 10 lines to thousands of expanded lines).
 * 4. Simulation of UefiMain execution with virtual Console Out.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <memory>
#include <cstdint>
#include <cassert>
#include <algorithm>

namespace mikan::uefi {

// =============================================================================
// 1. UEFI Core Types and Status Code Bit Layout
// =============================================================================

using UINTN = uint64_t;
using INTN = int64_t;
using CHAR16 = char16_t;
using EFI_HANDLE = void*;

constexpr UINTN EFI_ERROR_BIT = 0x8000'0000'0000'0000ULL;

constexpr UINTN EFI_SUCCESS            = 0;
constexpr UINTN EFI_LOAD_ERROR         = EFI_ERROR_BIT | 1;
constexpr UINTN EFI_INVALID_PARAMETER  = EFI_ERROR_BIT | 2;
constexpr UINTN EFI_UNSUPPORTED        = EFI_ERROR_BIT | 3;
constexpr UINTN EFI_BAD_BUFFER_SIZE    = EFI_ERROR_BIT | 4;
constexpr UINTN EFI_BUFFER_TOO_SMALL   = EFI_ERROR_BIT | 5;
constexpr UINTN EFI_NOT_READY          = EFI_ERROR_BIT | 6;
constexpr UINTN EFI_DEVICE_ERROR       = EFI_ERROR_BIT | 7;
constexpr UINTN EFI_WRITE_PROTECTED    = EFI_ERROR_BIT | 8;
constexpr UINTN EFI_OUT_OF_RESOURCES   = EFI_ERROR_BIT | 9;

[[nodiscard]] constexpr bool EFI_ERROR(UINTN status) noexcept {
    return (status & EFI_ERROR_BIT) != 0;
}

[[nodiscard]] constexpr UINTN EFI_ERROR_CODE(UINTN status) noexcept {
    return status & ~EFI_ERROR_BIT;
}

[[nodiscard]] std::string_view GetStatusName(UINTN status) noexcept {
    switch (status) {
        case EFI_SUCCESS:           return "EFI_SUCCESS";
        case EFI_LOAD_ERROR:        return "EFI_LOAD_ERROR";
        case EFI_INVALID_PARAMETER: return "EFI_INVALID_PARAMETER";
        case EFI_UNSUPPORTED:       return "EFI_UNSUPPORTED";
        case EFI_BAD_BUFFER_SIZE:   return "EFI_BAD_BUFFER_SIZE";
        case EFI_BUFFER_TOO_SMALL:  return "EFI_BUFFER_TOO_SMALL";
        case EFI_NOT_READY:         return "EFI_NOT_READY";
        case EFI_DEVICE_ERROR:      return "EFI_DEVICE_ERROR";
        case EFI_WRITE_PROTECTED:   return "EFI_WRITE_PROTECTED";
        case EFI_OUT_OF_RESOURCES:  return "EFI_OUT_OF_RESOURCES";
        default:                    return "EFI_UNKNOWN_STATUS";
    }
}

// =============================================================================
// 2. UTF-16LE String Utilities
// =============================================================================

struct Char16ByteLayout {
    size_t offset;
    uint8_t low_byte;
    uint8_t high_byte;
    uint16_t code_point;
    std::string ascii_repr;
};

class UefiStringUtil {
public:
    [[nodiscard]] static std::u16string ToUtf16(std::string_view ascii) {
        std::u16string out;
        out.reserve(ascii.size());
        for (char c : ascii) {
            out.push_back(static_cast<char16_t>(static_cast<unsigned char>(c)));
        }
        return out;
    }

    [[nodiscard]] static std::string ToAscii(std::u16string_view utf16) {
        std::string out;
        out.reserve(utf16.size());
        for (char16_t c : utf16) {
            if (c >= 32 && c <= 126) {
                out.push_back(static_cast<char>(c));
            } else if (c == u'\n') {
                out.push_back('\n');
            } else if (c == 0) {
                break;
            } else {
                out.push_back('?');
            }
        }
        return out;
    }

    [[nodiscard]] static std::vector<Char16ByteLayout> AnalyzeBytes(std::u16string_view utf16_with_null) {
        std::vector<Char16ByteLayout> results;
        for (size_t i = 0; i < utf16_with_null.size(); ++i) {
            uint16_t val = static_cast<uint16_t>(utf16_with_null[i]);
            uint8_t low = static_cast<uint8_t>(val & 0xFF);
            uint8_t high = static_cast<uint8_t>((val >> 8) & 0xFF);

            std::string repr;
            if (val >= 32 && val <= 126) {
                repr = std::string(1, static_cast<char>(val));
            } else if (val == '\n') {
                repr = "\\n";
            } else if (val == 0) {
                repr = "\\0";
            } else {
                std::ostringstream ss;
                ss << "U+" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << val;
                repr = ss.str();
            }

            results.push_back({i * 2, low, high, val, repr});
        }
        return results;
    }
};

// =============================================================================
// 3. Simulated UEFI System Table and Console Protocol
// =============================================================================

struct SimpleTextOutputProtocol;

struct SimpleTextOutputProtocol {
    UINTN (*OutputString)(SimpleTextOutputProtocol* This, const CHAR16* String);
    std::string console_buffer;
};

struct SystemTable {
    uint64_t dummy_header[3];
    SimpleTextOutputProtocol* ConOut;
};

inline UINTN MockOutputString(SimpleTextOutputProtocol* This, const CHAR16* String) {
    if (!This || !String) {
        return EFI_INVALID_PARAMETER;
    }
    while (*String != 0) {
        char16_t c = *String++;
        if (c == u'\n') {
            This->console_buffer.push_back('\n');
        } else if (c >= 32 && c <= 126) {
            This->console_buffer.push_back(static_cast<char>(c));
        } else {
            This->console_buffer.push_back('.');
        }
    }
    return EFI_SUCCESS;
}

class UefiEnvironmentSimulator {
public:
    UefiEnvironmentSimulator() {
        proto_.OutputString = MockOutputString;
        table_.ConOut = &proto_;
    }

    [[nodiscard]] SystemTable* GetSystemTable() noexcept { return &table_; }
    [[nodiscard]] const std::string& GetConsoleOutput() const noexcept { return proto_.console_buffer; }
    void ClearConsole() noexcept { proto_.console_buffer.clear(); }

    UINTN Print(const CHAR16* format_string) {
        return proto_.OutputString(&proto_, format_string);
    }

private:
    SimpleTextOutputProtocol proto_{};
    SystemTable table_{};
};

// =============================================================================
// 4. C Preprocessor Simulator (Column 2.1 Verification)
// =============================================================================

struct PreprocessOutput {
    std::string expanded_source;
    size_t original_lines{0};
    size_t expanded_lines{0};
    std::vector<std::string> included_headers;
    std::vector<std::string> defined_symbols;
    size_t duplicate_includes_prevented{0};
};

class PreprocessorSimulator {
public:
    void AddVirtualFile(std::string path, std::string content) {
        files_[Normalize(path)] = std::move(content);
    }

    void AddSystemIncludeDir(std::string dir) {
        system_include_dirs_.push_back(Normalize(dir));
    }

    [[nodiscard]] PreprocessOutput Process(std::string_view entry_file) {
        PreprocessOutput output;
        std::string norm_entry = Normalize(entry_file);
        auto it = files_.find(norm_entry);
        assert(it != files_.end() && "Entry file must exist in virtual files");

        std::unordered_set<std::string> defined_symbols;
        std::vector<std::string> visited_stack;

        output.original_lines = CountLines(it->second);
        std::ostringstream out_stream;
        ProcessFile(norm_entry, defined_symbols, visited_stack, out_stream, output);

        output.expanded_source = out_stream.str();
        output.expanded_lines = CountLines(output.expanded_source);
        for (const auto& sym : defined_symbols) {
            output.defined_symbols.push_back(sym);
        }
        return output;
    }

private:
    std::unordered_map<std::string, std::string> files_;
    std::vector<std::string> system_include_dirs_;

    static std::string Normalize(std::string_view path) {
        std::string s(path);
        std::replace(s.begin(), s.end(), '\\', '/');
        while (s.find("//") != std::string::npos) {
            s.erase(s.find("//"), 1);
        }
        return s;
    }

    static size_t CountLines(std::string_view text) {
        if (text.empty()) return 0;
        size_t count = 0;
        for (char c : text) {
            if (c == '\n') ++count;
        }
        if (text.back() != '\n') ++count;
        return count;
    }

    [[nodiscard]] std::string ResolveInclude(std::string_view target, std::string_view current_dir, bool is_system) {
        if (!is_system) {
            std::string local_path = Normalize(std::string(current_dir) + "/" + std::string(target));
            if (files_.contains(local_path)) {
                return local_path;
            }
        }
        for (const auto& dir : system_include_dirs_) {
            std::string sys_path = Normalize(dir + "/" + std::string(target));
            if (files_.contains(sys_path)) {
                return sys_path;
            }
        }
        std::string exact = Normalize(target);
        if (files_.contains(exact)) {
            return exact;
        }
        return "";
    }

    void ProcessFile(const std::string& current_path,
                     std::unordered_set<std::string>& defined_symbols,
                     std::vector<std::string>& visited_stack,
                     std::ostream& out,
                     PreprocessOutput& output) {
        visited_stack.push_back(current_path);

        const std::string& content = files_[current_path];
        std::istringstream stream(content);
        std::string line;

        std::string current_dir = "";
        auto slash_pos = current_path.find_last_of('/');
        if (slash_pos != std::string::npos) {
            current_dir = current_path.substr(0, slash_pos);
        }

        std::vector<bool> skip_stack;

        while (std::getline(stream, line)) {
            // Trim leading spaces
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) {
                if (skip_stack.empty() || !skip_stack.back()) out << "\n";
                continue;
            }
            std::string_view sline = std::string_view(line).substr(start);

            if (sline.starts_with("#ifndef")) {
                auto space = sline.find_first_of(" \t", 7);
                if (space != std::string::npos) {
                    auto sym_start = sline.find_first_not_of(" \t", space);
                    auto sym_end = sline.find_first_of(" \t\r\n", sym_start);
                    std::string sym(sline.substr(sym_start, sym_end - sym_start));
                    bool already_defined = defined_symbols.contains(sym);
                    if (already_defined) {
                        ++output.duplicate_includes_prevented;
                    }
                    skip_stack.push_back(already_defined);
                    continue;
                }
            } else if (sline.starts_with("#define")) {
                if (skip_stack.empty() || !skip_stack.back()) {
                    auto space = sline.find_first_of(" \t", 7);
                    if (space != std::string::npos) {
                        auto sym_start = sline.find_first_not_of(" \t", space);
                        auto sym_end = sline.find_first_of(" \t\r\n", sym_start);
                        std::string sym(sline.substr(sym_start, sym_end - sym_start));
                        defined_symbols.insert(sym);
                    }
                }
                continue;
            } else if (sline.starts_with("#endif")) {
                if (!skip_stack.empty()) {
                    skip_stack.pop_back();
                    continue;
                }
            }

            if (!skip_stack.empty() && skip_stack.back()) {
                continue; // Skipped by #ifndef guard
            }

            if (sline.starts_with("#include")) {
                bool is_sys = false;
                std::string target;
                auto lt = sline.find('<');
                auto gt = sline.find('>');
                if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
                    is_sys = true;
                    target = std::string(sline.substr(lt + 1, gt - lt - 1));
                } else {
                    auto q1 = sline.find('"');
                    auto q2 = sline.find('"', q1 + 1);
                    if (q1 != std::string::npos && q2 != std::string::npos) {
                        target = std::string(sline.substr(q1 + 1, q2 - q1 - 1));
                    }
                }

                if (!target.empty()) {
                    std::string resolved = ResolveInclude(target, current_dir, is_sys);
                    assert(!resolved.empty() && "Include file must be resolvable");
                    output.included_headers.push_back(resolved);

                    out << "/* [BEGIN INCLUDE " << target << " (" << resolved << ")] */\n";
                    ProcessFile(resolved, defined_symbols, visited_stack, out, output);
                    out << "/* [END INCLUDE " << target << "] */\n";
                    continue;
                }
            }

            out << line << "\n";
        }

        visited_stack.pop_back();
    }
};

} // namespace mikan::uefi

// =============================================================================
// 5. Verification Suite & CLI Demonstration
// =============================================================================

void RunSelfTests() {
    using namespace mikan::uefi;

    // Test 1: Status Code Bit Manipulation
    assert(EFI_SUCCESS == 0);
    assert(!EFI_ERROR(EFI_SUCCESS));
    assert(EFI_ERROR(EFI_LOAD_ERROR));
    assert(EFI_ERROR(EFI_INVALID_PARAMETER));
    assert(EFI_ERROR(EFI_BUFFER_TOO_SMALL));
    assert(EFI_ERROR_CODE(EFI_LOAD_ERROR) == 1);
    assert(EFI_ERROR_CODE(EFI_INVALID_PARAMETER) == 2);
    assert(EFI_ERROR_CODE(EFI_BUFFER_TOO_SMALL) == 5);
    assert(GetStatusName(EFI_SUCCESS) == "EFI_SUCCESS");
    assert(GetStatusName(EFI_BUFFER_TOO_SMALL) == "EFI_BUFFER_TOO_SMALL");

    // Test 2: UTF-16 Encoding and Little-Endian Bytes
    std::string sample = "Hello\n";
    std::u16string utf16 = UefiStringUtil::ToUtf16(sample);
    utf16.push_back(0); // null terminate
    assert(utf16.size() == 7);
    assert(utf16[0] == u'H');
    assert(utf16[5] == u'\n');
    assert(utf16[6] == 0);

    auto bytes = UefiStringUtil::AnalyzeBytes(utf16);
    assert(bytes.size() == 7);
    assert(bytes[0].low_byte == 0x48 && bytes[0].high_byte == 0x00); // 'H'
    assert(bytes[5].low_byte == 0x0A && bytes[5].high_byte == 0x00); // '\n'
    assert(bytes[6].low_byte == 0x00 && bytes[6].high_byte == 0x00); // '\0'
    assert(UefiStringUtil::ToAscii(utf16) == "Hello\n");

    // Test 3: Virtual UEFI Environment OutputString & Print
    UefiEnvironmentSimulator uefi_sim;
    std::u16string msg = UefiStringUtil::ToUtf16("Hello, Mikan World!\n");
    msg.push_back(0);
    UINTN status = uefi_sim.Print(msg.c_str());
    assert(status == EFI_SUCCESS);
    assert(uefi_sim.GetConsoleOutput() == "Hello, Mikan World!\n");

    // Test 4: Preprocessor Simulator with MikanLoaderPkg model
    PreprocessorSimulator cpp_sim;
    cpp_sim.AddSystemIncludeDir("MdePkg/Include");

    // Register MdePkg standard headers
    cpp_sim.AddVirtualFile("MdePkg/Include/Uefi/UefiBaseType.h",
        "#ifndef __UEFI_BASE_TYPE_H__\n"
        "#define __UEFI_BASE_TYPE_H__\n"
        "typedef unsigned long long UINTN;\n"
        "typedef char16_t CHAR16;\n"
        "typedef UINTN EFI_STATUS;\n"
        "#endif\n"
    );

    cpp_sim.AddVirtualFile("MdePkg/Include/Uefi/UefiSpec.h",
        "#ifndef __UEFI_SPEC_H__\n"
        "#define __UEFI_SPEC_H__\n"
        "#include <Uefi/UefiBaseType.h>\n"
        "struct EFI_SYSTEM_TABLE {\n"
        "    void* ConOut;\n"
        "};\n"
        "#endif\n"
    );

    cpp_sim.AddVirtualFile("MdePkg/Include/Uefi.h",
        "#ifndef __UEFI_H__\n"
        "#define __UEFI_H__\n"
        "#include <Uefi/UefiBaseType.h>\n"
        "#include <Uefi/UefiSpec.h>\n"
        "#endif\n"
    );

    cpp_sim.AddVirtualFile("MdePkg/Include/Library/UefiLib.h",
        "#ifndef __UEFI_LIB_H__\n"
        "#define __UEFI_LIB_H__\n"
        "#include <Uefi.h>\n"
        "EFI_STATUS Print(const CHAR16* Format, ...);\n"
        "#endif\n"
    );

    // Register MikanLoaderPkg Main.c
    cpp_sim.AddVirtualFile("MikanLoaderPkg/Main.c",
        "#include <Uefi.h>\n"
        "#include <Library/UefiLib.h>\n"
        "\n"
        "EFI_STATUS EFIAPI UefiMain(\n"
        "    EFI_HANDLE ImageHandle,\n"
        "    EFI_SYSTEM_TABLE *SystemTable)\n"
        "{\n"
        "  Print(L\"Hello, Mikan World!\\n\");\n"
        "  while (1);\n"
        "  return EFI_SUCCESS;\n"
        "}\n"
    );

    auto result = cpp_sim.Process("MikanLoaderPkg/Main.c");
    assert(result.original_lines == 11);
    assert(result.expanded_lines > result.original_lines);
    // Uefi.h and UefiBaseType.h should be guarded against duplicate expansion
    assert(result.duplicate_includes_prevented >= 2);
    assert(result.expanded_source.find("Hello, Mikan World!") != std::string::npos);

    std::cout << "[PASS] All Chapter 2.2 UEFI C++20 Verification Tests Succeeded!\n";
}

void RunDemo() {
    using namespace mikan::uefi;

    std::cout << "=================================================================\n";
    std::cout << "   MikanOS Chapter 2.2: EDK II Hello World & Preprocessor Demo   \n";
    std::cout << "=================================================================\n\n";

    std::cout << "--- 1. UEFI Status Code Architecture (64-bit Bitmask) ---\n";
    std::vector<UINTN> test_statuses = {
        EFI_SUCCESS, EFI_LOAD_ERROR, EFI_INVALID_PARAMETER, EFI_BUFFER_TOO_SMALL
    };
    for (UINTN s : test_statuses) {
        std::cout << "Status: " << std::left << std::setw(24) << std::setfill(' ') << GetStatusName(s)
                  << " Hex: 0x" << std::right << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << s
                  << std::setfill(' ')
                  << " IsError: " << (EFI_ERROR(s) ? "YES (Bit 63=1)" : "NO  (Bit 63=0)")
                  << " Code: " << std::dec << EFI_ERROR_CODE(s) << "\n";
    }

    std::cout << "\n--- 2. UTF-16LE Memory Representation of L\"Hello, Mikan World!\\n\" ---\n";
    std::string text = "Hello, Mikan World!\n";
    std::u16string u16 = UefiStringUtil::ToUtf16(text);
    u16.push_back(0); // Null terminator
    auto layouts = UefiStringUtil::AnalyzeBytes(u16);
    std::cout << "Offset  Bytes(LE)   CodePoint   Character\n";
    std::cout << "-----------------------------------------\n";
    for (const auto& row : layouts) {
        std::cout << "+" << std::right << std::dec << std::setw(4) << std::setfill('0') << row.offset
                  << "   " << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(row.low_byte)
                  << " " << std::setw(2) << std::setfill('0') << static_cast<int>(row.high_byte)
                  << "       0x" << std::setw(4) << std::setfill('0') << row.code_point
                  << "      '" << row.ascii_repr << "'\n";
    }
    std::cout << std::dec << std::setfill(' ');

    std::cout << "\n--- 3. Preprocessor Expansion & Guard Statistics ---\n";
    PreprocessorSimulator cpp_sim;
    cpp_sim.AddSystemIncludeDir("MdePkg/Include");
    cpp_sim.AddVirtualFile("MdePkg/Include/Uefi/UefiBaseType.h",
        "#ifndef __UEFI_BASE_TYPE_H__\n#define __UEFI_BASE_TYPE_H__\ntypedef unsigned long long UINTN;\ntypedef char16_t CHAR16;\n#endif\n");
    cpp_sim.AddVirtualFile("MdePkg/Include/Uefi/UefiSpec.h",
        "#ifndef __UEFI_SPEC_H__\n#define __UEFI_SPEC_H__\n#include <Uefi/UefiBaseType.h>\nstruct EFI_SYSTEM_TABLE { void* ConOut; };\n#endif\n");
    cpp_sim.AddVirtualFile("MdePkg/Include/Uefi.h",
        "#ifndef __UEFI_H__\n#define __UEFI_H__\n#include <Uefi/UefiBaseType.h>\n#include <Uefi/UefiSpec.h>\n#endif\n");
    cpp_sim.AddVirtualFile("MdePkg/Include/Library/UefiLib.h",
        "#ifndef __UEFI_LIB_H__\n#define __UEFI_LIB_H__\n#include <Uefi.h>\nUINTN Print(const CHAR16* fmt);\n#endif\n");
    cpp_sim.AddVirtualFile("MikanLoaderPkg/Main.c",
        "#include <Uefi.h>\n#include <Library/UefiLib.h>\nEFI_STATUS EFIAPI UefiMain(void* H, void* T) {\n  Print(L\"Hello, Mikan World!\\n\");\n  while(1);\n}\n");

    auto out = cpp_sim.Process("MikanLoaderPkg/Main.c");
    std::cout << "Original Source Lines: " << out.original_lines << "\n";
    std::cout << "Expanded Source Lines: " << out.expanded_lines << "\n";
    std::cout << "Duplicate Includes Prevented: " << out.duplicate_includes_prevented << "\n";
    std::cout << "Headers Included in Order:\n";
    for (const auto& h : out.included_headers) {
        std::cout << "  -> " << h << "\n";
    }

    std::cout << "\n--- 4. Simulated UEFI Console Execution Output ---\n";
    UefiEnvironmentSimulator env;
    env.Print(u16.c_str());
    std::cout << "[Screen Output] >>> " << env.GetConsoleOutput();
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string_view(argv[1]) == "--test") {
        RunSelfTests();
        return 0;
    }
    RunSelfTests();
    RunDemo();
    return 0;
}
