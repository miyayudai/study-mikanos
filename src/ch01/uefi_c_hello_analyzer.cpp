/**
 * @file uefi_c_hello_analyzer.cpp
 * @brief Standalone C++20 verification tool for Section 1.9: Hello World in C & Column 1.1: PE, COFF, ELF.
 *
 * Demonstrates and verifies:
 * 1. UEFI minimal structure definitions and offset math (dummy[52], ConsoleOutHandle, ConOut).
 * 2. C arrow operator (->) dereference chains and object-oriented 'This' pointer pattern.
 * 3. UTF-16LE wide string encoding and decoding.
 * 4. Binary format comparison: PE32+ (PE/COFF 64-bit) vs ELF64 headers and structures.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <sstream>

namespace mikan {

// ============================================================================
// 1. Minimal UEFI Types & Structures (as in hello.c)
// ============================================================================

using CHAR16 = uint16_t;
using EFI_STATUS = uint64_t;
using EFI_HANDLE = void*;

constexpr EFI_STATUS EFI_SUCCESS = 0;
constexpr EFI_STATUS EFI_INVALID_PARAMETER = 0x8000000000000002ULL;

struct EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL;

using EFI_TEXT_STRING = EFI_STATUS (*)(
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL* This,
    CHAR16* String
);

#pragma pack(push, 1)
struct EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL {
    void* dummy;                // 8 bytes: Reset() function pointer
    EFI_TEXT_STRING OutputString; // 8 bytes: OutputString() function pointer
};

struct EFI_SYSTEM_TABLE {
    char dummy[52];                                  // 52 bytes: Headers, Vendor, Revision, ConIn
    EFI_HANDLE ConsoleOutHandle;                     // 8 bytes: ConsoleOut handle
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL* ConOut;         // 8 bytes: Console Out protocol pointer
};
#pragma pack(pop)

// ============================================================================
// 2. Binary Format Headers (PE32+ vs ELF64)
// ============================================================================

#pragma pack(push, 1)
// PE32+ DOS Header (64 bytes)
struct ImageDosHeader {
    uint16_t e_magic;    // Magic number: "MZ" (0x5A4D)
    uint16_t e_cblp;
    uint16_t e_cp;
    uint16_t e_crlc;
    uint16_t e_cparhdr;
    uint16_t e_minalloc;
    uint16_t e_maxalloc;
    uint16_t e_ss;
    uint16_t e_sp;
    uint16_t e_csum;
    uint16_t e_ip;
    uint16_t e_cs;
    uint16_t e_lfarlc;
    uint16_t e_ovno;
    uint16_t e_res[4];
    uint16_t e_oemid;
    uint16_t e_oeminfo;
    uint16_t e_res2[10];
    int32_t  e_lfanew;   // File address of new exe header (PE Header offset)
};

// COFF File Header (20 bytes)
struct ImageFileHeader {
    uint16_t Machine;              // 0x8664 = x86-64
    uint16_t NumberOfSections;
    uint32_t TimeDateStamp;
    uint32_t PointerToSymbolTable;
    uint32_t NumberOfSymbols;
    uint16_t SizeOfOptionalHeader;
    uint16_t Characteristics;
};

// PE32+ Optional Header Standard Fields (112 bytes base)
struct ImageOptionalHeader64 {
    uint16_t Magic;                // 0x020B for PE32+ (64-bit)
    uint8_t  MajorLinkerVersion;
    uint8_t  MinorLinkerVersion;
    uint32_t SizeOfCode;
    uint32_t SizeOfInitializedData;
    uint32_t SizeOfUninitializedData;
    uint32_t AddressOfEntryPoint;  // RVA of entry point (e.g. EfiMain)
    uint32_t BaseOfCode;
    uint64_t ImageBase;            // Preferred load address
    uint32_t SectionAlignment;
    uint32_t FileAlignment;
    uint16_t MajorOperatingSystemVersion;
    uint16_t MinorOperatingSystemVersion;
    uint16_t MajorImageVersion;
    uint16_t MinorImageVersion;
    uint16_t MajorSubsystemVersion;
    uint16_t MinorSubsystemVersion;
    uint32_t Win32VersionValue;
    uint32_t SizeOfImage;
    uint32_t SizeOfHeaders;
    uint32_t CheckSum;
    uint16_t Subsystem;            // 10 = IMAGE_SUBSYSTEM_EFI_APPLICATION
    uint16_t DllCharacteristics;
    uint64_t SizeOfStackReserve;
    uint64_t SizeOfStackCommit;
    uint64_t SizeOfHeapReserve;
    uint64_t SizeOfHeapCommit;
    uint32_t LoaderFlags;
    uint32_t NumberOfRvaAndSizes;
};

// ELF64 Header (64 bytes)
struct Elf64_Ehdr {
    unsigned char e_ident[16];     // Magic: 0x7F, 'E', 'L', 'F', ...
    uint16_t      e_type;          // 2 = ET_EXEC, 3 = ET_DYN (PIE)
    uint16_t      e_machine;       // 0x3E = x86-64
    uint32_t      e_version;
    uint64_t      e_entry;         // Virtual address of entry point
    uint64_t      e_phoff;         // Program header table file offset
    uint64_t      e_shoff;         // Section header table file offset
    uint32_t      e_flags;
    uint16_t      e_ehsize;        // Header size (64 bytes)
    uint16_t      e_phentsize;     // Program header entry size (56 bytes)
    uint16_t      e_phnum;
    uint16_t      e_shentsize;     // Section header entry size (64 bytes)
    uint16_t      e_shnum;
    uint16_t      e_shstrndx;
};

// ELF64 Program Header (56 bytes)
struct Elf64_Phdr {
    uint32_t p_type;               // 1 = PT_LOAD
    uint32_t p_flags;              // 1=X, 2=W, 4=R
    uint64_t p_offset;             // Segment file offset
    uint64_t p_vaddr;              // Segment virtual address
    uint64_t p_paddr;              // Segment physical address
    uint64_t p_filesz;             // Segment size in file
    uint64_t p_memsz;              // Segment size in memory
    uint64_t p_align;              // Alignment (e.g. 4096 or 2097152)
};
#pragma pack(pop)

// ============================================================================
// 3. Simulated Firmware & Execution Tracker
// ============================================================================

static std::vector<std::string> g_firmware_console_log;

EFI_STATUS SimulatedOutputString(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL* This, CHAR16* String) {
    if (!This || !String) {
        return EFI_INVALID_PARAMETER;
    }
    // Convert UTF-16LE characters to UTF-8 for console output
    std::string text;
    for (size_t i = 0; String[i] != 0; ++i) {
        char c = static_cast<char>(String[i] & 0x7F);
        text += c;
    }
    g_firmware_console_log.push_back(text);
    return EFI_SUCCESS;
}

} // namespace mikan

int main() {
    using namespace mikan;

    std::cout << "================================================================\n";
    std::cout << " MikanOS Ch1.9: UEFI C Data Structures & Binary Format Inspector\n";
    std::cout << "================================================================\n\n";

    // 1. Inspect UEFI System Table Layout
    std::cout << "[1] UEFI Structure Layout & Offset Verification\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "sizeof(EFI_SYSTEM_TABLE):               " << sizeof(EFI_SYSTEM_TABLE) << " bytes\n";
    std::cout << "offsetof(EFI_SYSTEM_TABLE, dummy):      " << offsetof(EFI_SYSTEM_TABLE, dummy) << " (0x00)\n";
    std::cout << "offsetof(EFI_SYSTEM_TABLE, ConsoleOutHandle): " << offsetof(EFI_SYSTEM_TABLE, ConsoleOutHandle) << " (0x34 = 52)\n";
    std::cout << "offsetof(EFI_SYSTEM_TABLE, ConOut):     " << offsetof(EFI_SYSTEM_TABLE, ConOut) << " (0x3C = 60)\n";
    std::cout << "sizeof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL): " << sizeof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL) << " bytes\n";
    std::cout << "offsetof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL, dummy): " << offsetof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL, dummy) << " (0x00)\n";
    std::cout << "offsetof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL, OutputString): " << offsetof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL, OutputString) << " (0x08)\n";

    assert(sizeof(EFI_SYSTEM_TABLE) == 68);
    assert(offsetof(EFI_SYSTEM_TABLE, ConsoleOutHandle) == 52);
    assert(offsetof(EFI_SYSTEM_TABLE, ConOut) == 60);
    assert(sizeof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL) == 16);
    assert(offsetof(EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL, OutputString) == 8);
    std::cout << ">> Offset and Size Assertions PASSED!\n\n";

    // 2. Simulated Execution of hello.c C-code
    std::cout << "[2] Simulating SystemTable->ConOut->OutputString(SystemTable->ConOut, L\"...\")\n";
    std::cout << "----------------------------------------------------------------\n";

    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL text_out{};
    text_out.dummy = nullptr;
    text_out.OutputString = SimulatedOutputString;

    EFI_SYSTEM_TABLE system_table{};
    system_table.ConsoleOutHandle = reinterpret_cast<EFI_HANDLE>(0x12345678ULL);
    system_table.ConOut = &text_out;

    // UTF-16LE string: L"Hello, world!\n"
    CHAR16 hello_str[] = {
        'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n', 0
    };

    // Call through arrow operator chain
    EFI_STATUS status = system_table.ConOut->OutputString(system_table.ConOut, hello_str);
    std::cout << "Call return status: " << status << " (EFI_SUCCESS)\n";
    std::cout << "Firmware console output: \"" << g_firmware_console_log.back() << "\"\n";

    assert(status == EFI_SUCCESS);
    assert(!g_firmware_console_log.empty());
    assert(g_firmware_console_log.back() == "Hello, world!\n");
    std::cout << ">> UEFI API Simulation PASSED!\n\n";

    // 3. Binary Format Structure Comparison (PE32+ vs ELF64)
    std::cout << "[3] Binary Format Comparison: PE32+ (PE/COFF) vs ELF64\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "PE32+ DOS Header Size:        " << sizeof(ImageDosHeader) << " bytes (Magic: 0x5A4D 'MZ')\n";
    std::cout << "PE32+ COFF File Header Size:  " << sizeof(ImageFileHeader) << " bytes (Machine: 0x8664 x86-64)\n";
    std::cout << "PE32+ Optional Header Size:   " << sizeof(ImageOptionalHeader64) << " bytes (Magic: 0x020B PE32+)\n";
    std::cout << "ELF64 Header Size:            " << sizeof(Elf64_Ehdr) << " bytes (Magic: 0x7F 'ELF')\n";
    std::cout << "ELF64 Program Header Size:    " << sizeof(Elf64_Phdr) << " bytes (PT_LOAD segments)\n";

    assert(sizeof(ImageDosHeader) == 64);
    assert(sizeof(ImageFileHeader) == 20);
    assert(sizeof(ImageOptionalHeader64) == 112);
    assert(sizeof(Elf64_Ehdr) == 64);
    assert(sizeof(Elf64_Phdr) == 56);
    std::cout << ">> Binary Header Structures Assertions PASSED!\n\n";

    std::cout << "================================================================\n";
    std::cout << " All Section 1.9 C++20 checks successfully verified!\n";
    std::cout << "================================================================\n";

    return 0;
}
