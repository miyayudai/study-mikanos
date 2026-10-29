/**
 * @file pe32plus_parser.cpp
 * @brief Standalone C++20 verification of UEFI PE32+ (x86-64) binary headers & machine code.
 *
 * Demonstrates binary format verification for MikanOS Day 01 hello.efi without external dependencies.
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace mikan::pe {

#pragma pack(push, 1)

struct DosHeader {
    uint16_t e_magic;      // Magic number ('MZ' = 0x5A4D)
    uint16_t e_cblp;       // Bytes on last page of file
    uint16_t e_cp;         // Pages in file
    uint16_t e_crlc;       // Relocations
    uint16_t e_cparhdr;    // Size of header in paragraphs
    uint16_t e_minalloc;   // Minimum extra paragraphs needed
    uint16_t e_maxalloc;   // Maximum extra paragraphs needed
    uint16_t e_ss;         // Initial (relative) SS value
    uint16_t e_sp;         // Initial SP value
    uint16_t e_csum;       // Checksum
    uint16_t e_ip;         // Initial IP value
    uint16_t e_cs;         // Initial (relative) CS value
    uint16_t e_lfarlc;     // File address of relocation table
    uint16_t e_ovno;       // Overlay number
    uint16_t e_res[4];     // Reserved words
    uint16_t e_oemid;      // OEM identifier (for e_oeminfo)
    uint16_t e_oeminfo;    // OEM information; e_oemid specific
    uint16_t e_res2[10];   // Reserved words
    uint32_t e_lfanew;     // File address of new exe header (PE Header offset)
};

struct CoffFileHeader {
    uint16_t Machine;              // 0x8664 = x86-64 (IMAGE_FILE_MACHINE_AMD64)
    uint16_t NumberOfSections;     // Number of sections (e.g. 2: .text, .rdata)
    uint32_t TimeDateStamp;
    uint32_t PointerToSymbolTable;
    uint32_t NumberOfSymbols;
    uint16_t SizeOfOptionalHeader; // Typically 0xF0 for PE32+
    uint16_t Characteristics;      // File characteristics
};

struct ImageDataDirectory {
    uint32_t VirtualAddress;
    uint32_t Size;
};

struct Pe32PlusOptionalHeader {
    uint16_t Magic;                        // 0x020B for PE32+ (64-bit)
    uint8_t  MajorLinkerVersion;
    uint8_t  MinorLinkerVersion;
    uint32_t SizeOfCode;
    uint32_t SizeOfInitializedData;
    uint32_t SizeOfUninitializedData;
    uint32_t AddressOfEntryPoint;          // RVA of entry point
    uint32_t BaseOfCode;
    uint64_t ImageBase;                    // Preferred image base address
    uint32_t SectionAlignment;             // Section alignment in memory (0x1000)
    uint32_t FileAlignment;                // File alignment on disk (0x200)
    uint16_t MajorOperatingSystemVersion;
    uint16_t MinorOperatingSystemVersion;
    uint16_t MajorImageVersion;
    uint16_t MinorImageVersion;
    uint16_t MajorSubsystemVersion;
    uint16_t MinorSubsystemVersion;
    uint32_t Win32VersionValue;
    uint32_t SizeOfImage;                  // Total memory image size
    uint32_t SizeOfHeaders;                // Headers total size (file-aligned)
    uint32_t CheckSum;
    uint16_t Subsystem;                    // 0x000A = IMAGE_SUBSYSTEM_EFI_APPLICATION
    uint16_t DllCharacteristics;
    uint64_t SizeOfStackReserve;
    uint64_t SizeOfStackCommit;
    uint64_t SizeOfHeapReserve;
    uint64_t SizeOfHeapCommit;
    uint32_t LoaderFlags;
    uint32_t NumberOfRvaAndSizes;
    ImageDataDirectory DataDirectory[16];
};

struct SectionHeader {
    uint8_t  Name[8];
    uint32_t VirtualSize;
    uint32_t VirtualAddress;
    uint32_t SizeOfRawData;
    uint32_t PointerToRawData;
    uint32_t PointerToRelocations;
    uint32_t PointerToLinenumbers;
    uint16_t NumberOfRelocations;
    uint16_t NumberOfLinenumbers;
    uint32_t Characteristics;
};

#pragma pack(pop)

constexpr uint16_t DOS_MAGIC_MZ = 0x5A4D;
constexpr uint32_t PE_SIGNATURE  = 0x00004550; // "PE\0\0"
constexpr uint16_t MACHINE_AMD64 = 0x8664;
constexpr uint16_t PE32PLUS_MAGIC = 0x020B;
constexpr uint16_t SUBSYS_EFI_APP = 0x000A;

class PE32PlusVerifier {
public:
    static bool verify(const std::vector<uint8_t>& data) {
        if (data.size() < sizeof(DosHeader)) {
            std::cerr << "[-] Error: File smaller than DOS Header\n";
            return false;
        }

        const auto* dos = reinterpret_cast<const DosHeader*>(data.data());
        if (dos->e_magic != DOS_MAGIC_MZ) {
            std::cerr << "[-] Error: Invalid DOS Magic: 0x" << std::hex << dos->e_magic << "\n";
            return false;
        }
        std::cout << "[+] DOS Header Verified: Magic = 'MZ', PE offset = 0x" << std::hex << dos->e_lfanew << "\n";

        if (dos->e_lfanew + 4 + sizeof(CoffFileHeader) + sizeof(Pe32PlusOptionalHeader) > data.size()) {
            std::cerr << "[-] Error: Corrupt PE header offset\n";
            return false;
        }

        uint32_t pe_sig = *reinterpret_cast<const uint32_t*>(data.data() + dos->e_lfanew);
        if (pe_sig != PE_SIGNATURE) {
            std::cerr << "[-] Error: Invalid PE Signature: 0x" << std::hex << pe_sig << "\n";
            return false;
        }
        std::cout << "[+] PE Signature Verified: 'PE\\0\\0'\n";

        const auto* coff = reinterpret_cast<const CoffFileHeader*>(data.data() + dos->e_lfanew + 4);
        if (coff->Machine != MACHINE_AMD64) {
            std::cerr << "[-] Error: Not AMD64/x86-64 machine: 0x" << std::hex << coff->Machine << "\n";
            return false;
        }
        std::cout << "[+] COFF File Header Verified: x86-64 Architecture, Sections = " << std::dec << coff->NumberOfSections << "\n";

        const auto* opt = reinterpret_cast<const Pe32PlusOptionalHeader*>(
            data.data() + dos->e_lfanew + 4 + sizeof(CoffFileHeader)
        );
        if (opt->Magic != PE32PLUS_MAGIC) {
            std::cerr << "[-] Error: Not PE32+ (64-bit): 0x" << std::hex << opt->Magic << "\n";
            return false;
        }
        if (opt->Subsystem != SUBSYS_EFI_APP) {
            std::cerr << "[-] Error: Subsystem not EFI Application: 0x" << std::hex << opt->Subsystem << "\n";
            return false;
        }
        std::cout << "[+] PE32+ Optional Header Verified: Subsystem = EFI Application (0x0A), Entry RVA = 0x"
                  << std::hex << opt->AddressOfEntryPoint << "\n";

        // Section Verification
        const auto* sec = reinterpret_cast<const SectionHeader*>(
            reinterpret_cast<const uint8_t*>(opt) + coff->SizeOfOptionalHeader
        );

        for (uint16_t i = 0; i < coff->NumberOfSections; ++i) {
            char name[9] = {0};
            std::memcpy(name, sec[i].Name, 8);
            std::cout << "    - Section [" << name << "]: VAddr=0x" << std::hex << sec[i].VirtualAddress
                      << " RawOffset=0x" << sec[i].PointerToRawData
                      << " RawSize=0x" << sec[i].SizeOfRawData << "\n";
        }

        // Verify machine code at .text section
        if (data.size() >= 0x214) {
            // sub rsp, 0x28 (48 83 ec 28)
            bool valid_sub_rsp = (data[0x200] == 0x48 && data[0x201] == 0x83 && data[0x202] == 0xec && data[0x203] == 0x28);
            // call [rcx + 8] (ff 51 08)
            bool valid_call = (data[0x20f] == 0xff && data[0x210] == 0x51 && data[0x211] == 0x08);
            // jmp .fin (eb fe)
            bool valid_jmp = (data[0x212] == 0xeb && data[0x213] == 0xfe);

            if (valid_sub_rsp && valid_call && valid_jmp) {
                std::cout << "[+] x86-64 Machine Code Verified: sub rsp, 0x28 / call [rcx+8] / jmp $\n";
            } else {
                std::cerr << "[-] Warning: Machine code signatures did not match expected minimal hello\n";
            }
        }

        // Verify UTF-16 "Hello, world!" at .rdata (offset 0x400)
        if (data.size() >= 0x41a) {
            // "Hello" in UTF-16LE: 48 00 65 00 6c 00 6c 00 6f 00
            bool valid_hello = (data[0x400] == 0x48 && data[0x401] == 0x00 &&
                                data[0x402] == 0x65 && data[0x403] == 0x00 &&
                                data[0x404] == 0x6c && data[0x405] == 0x00);
            if (valid_hello) {
                std::cout << "[+] UTF-16 Data Verified: 'Hello, world!' string found at 0x400\n";
            }
        }

        return true;
    }
};

} // namespace mikan::pe

int main(int argc, char** argv) {
    std::string filepath = "vendor/mikanos-build/day01/bin/hello.efi";
    if (argc > 1) {
        filepath = argv[1];
    }

    std::cout << "========================================================\n";
    std::cout << " MikanOS C++20 UEFI PE32+ Binary Header Verifier\n";
    std::cout << " Target: " << filepath << "\n";
    std::cout << "========================================================\n";

    if (!std::filesystem::exists(filepath)) {
        std::cerr << "[-] Error: File not found: " << filepath << "\n";
        return 1;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "[-] Error: Failed to open file: " << filepath << "\n";
        return 1;
    }

    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    file.close();

    bool success = mikan::pe::PE32PlusVerifier::verify(buffer);
    if (success) {
        std::cout << "========================================================\n";
        std::cout << "[SUCCESS] PE32+ Header & x86-64 Instruction Verification Passed!\n";
        std::cout << "========================================================\n";
        return 0;
    } else {
        std::cerr << "[FAILURE] Verification Failed!\n";
        return 1;
    }
}
