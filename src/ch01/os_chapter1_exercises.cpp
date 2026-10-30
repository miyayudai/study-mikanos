/**
 * @file os_chapter1_exercises.cpp
 * @brief Chapter 1 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 1:
 * 1. PE32+ (64-bit UEFI) Header Parsing, Magic Verification & Section Alignment
 * 2. Microsoft x64 Calling Convention Stack Frame Simulation (Shadow Space & 16-byte Alignment)
 * 3. 64-bit UEFI EFI_SYSTEM_TABLE Offset Layout & ConOut->OutputString Simulation
 * 4. MBR 32-bit vs GPT 64-bit LBA Addressing & Linux dev_t Bit Mathematics
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch01 {

// =============================================================================
// Exercise 1: PE32+ (64-bit UEFI) Header Parser & Magic Validator
// =============================================================================

#pragma pack(push, 1)
struct DosHeader {
    uint16_t e_magic;     // Magic number: 0x5A4D ('MZ')
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
    uint32_t e_lfanew;    // File address of PE header (typically 0x80 or 0x40)
};

struct CoffFileHeader {
    uint16_t machine;               // 0x8664 = AMD64 / x86-64
    uint16_t number_of_sections;
    uint32_t time_date_stamp;
    uint32_t pointer_to_symbol_table;
    uint32_t number_of_symbols;
    uint16_t size_of_optional_header;
    uint16_t characteristics;
};

struct Pe32PlusOptionalHeader {
    uint16_t magic;                 // 0x020B for PE32+ (64-bit)
    uint8_t  major_linker_version;
    uint8_t  minor_linker_version;
    uint32_t size_of_code;
    uint32_t size_of_initialized_data;
    uint32_t size_of_uninitialized_data;
    uint32_t address_of_entry_point;
    uint32_t base_of_code;
    uint64_t image_base;
    uint32_t section_alignment;
    uint32_t file_alignment;
    uint16_t major_operating_system_version;
    uint16_t minor_operating_system_version;
    uint16_t major_image_version;
    uint16_t minor_image_version;
    uint16_t major_subsystem_version;
    uint16_t minor_subsystem_version;
    uint32_t win32_version_value;
    uint32_t size_of_image;
    uint32_t size_of_headers;
    uint32_t check_sum;
    uint16_t subsystem;             // 0x000A = IMAGE_SUBSYSTEM_EFI_APPLICATION
    uint16_t dll_characteristics;
    uint64_t size_of_stack_reserve;
    uint64_t size_of_stack_commit;
    uint64_t size_of_heap_reserve;
    uint64_t size_of_heap_commit;
    uint32_t loader_flags;
    uint32_t number_of_rva_and_sizes;
};
#pragma pack(pop)

struct PeValidationResult {
    bool is_valid = false;
    bool is_pe32_plus = false;
    bool is_x86_64 = false;
    bool is_efi_app = false;
    uint32_t entry_point = 0;
    uint64_t image_base = 0;
    std::string error_message;
};

class Pe32PlusValidator {
public:
    static PeValidationResult Validate(const std::vector<uint8_t>& binary) {
        PeValidationResult res;
        if (binary.size() < sizeof(DosHeader)) {
            res.error_message = "Binary smaller than DOS Header";
            return res;
        }

        const auto* dos = reinterpret_cast<const DosHeader*>(binary.data());
        if (dos->e_magic != 0x5A4D) { // 'MZ'
            res.error_message = "Invalid DOS magic: expected 0x5A4D ('MZ')";
            return res;
        }

        if (dos->e_lfanew + 4 + sizeof(CoffFileHeader) + sizeof(Pe32PlusOptionalHeader) > binary.size()) {
            res.error_message = "PE header offset extends beyond binary bounds";
            return res;
        }

        const uint8_t* pe_ptr = binary.data() + dos->e_lfanew;
        uint32_t pe_sig = *reinterpret_cast<const uint32_t*>(pe_ptr);
        if (pe_sig != 0x00004550) { // 'PE\0\0'
            res.error_message = "Invalid PE signature: expected 0x00004550 ('PE\\0\\0')";
            return res;
        }

        const auto* coff = reinterpret_cast<const CoffFileHeader*>(pe_ptr + 4);
        res.is_x86_64 = (coff->machine == 0x8664);

        const auto* opt = reinterpret_cast<const Pe32PlusOptionalHeader*>(pe_ptr + 4 + sizeof(CoffFileHeader));
        res.is_pe32_plus = (opt->magic == 0x020B);
        res.is_efi_app = (opt->subsystem == 0x000A);
        res.entry_point = opt->address_of_entry_point;
        res.image_base = opt->image_base;

        res.is_valid = res.is_pe32_plus && res.is_x86_64 && res.is_efi_app;
        return res;
    }

    static std::vector<uint8_t> CreateMinimalSyntheticUefiBinary() {
        std::vector<uint8_t> bin(512, 0); // Single 512B sector

        auto* dos = reinterpret_cast<DosHeader*>(bin.data());
        dos->e_magic = 0x5A4D;
        dos->e_lfanew = 0x40; // PE starts at 0x40

        uint8_t* pe_ptr = bin.data() + 0x40;
        *reinterpret_cast<uint32_t*>(pe_ptr) = 0x00004550; // 'PE\0\0'

        auto* coff = reinterpret_cast<CoffFileHeader*>(pe_ptr + 4);
        coff->machine = 0x8664; // AMD64
        coff->number_of_sections = 1;
        coff->size_of_optional_header = sizeof(Pe32PlusOptionalHeader);
        coff->characteristics = 0x0022; // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE

        auto* opt = reinterpret_cast<Pe32PlusOptionalHeader*>(pe_ptr + 4 + sizeof(CoffFileHeader));
        opt->magic = 0x020B; // PE32+
        opt->address_of_entry_point = 0x1000;
        opt->image_base = 0x0000000140000000ULL;
        opt->section_alignment = 0x1000;
        opt->file_alignment = 0x0200;
        opt->subsystem = 0x000A; // EFI_APPLICATION

        return bin;
    }
};

// =============================================================================
// Exercise 2: Microsoft x64 Calling Convention Frame Simulator
// =============================================================================

struct MicrosoftX64Registers {
    uint64_t rcx = 0;  // 1st argument (or 'this' pointer in member function)
    uint64_t rdx = 0;  // 2nd argument
    uint64_t r8  = 0;  // 3rd argument
    uint64_t r9  = 0;  // 4th argument
    uint64_t rax = 0;  // Return value
    uint64_t rsp = 0;  // Stack pointer
};

class MicrosoftX64FrameSimulator {
public:
    static constexpr size_t kShadowSpaceBytes = 32;      // 4 qwords reserved for RCX, RDX, R8, R9
    static constexpr size_t kRequiredAlignment = 16;     // RSP % 16 == 0 before CALL

    struct CallContext {
        MicrosoftX64Registers regs;
        std::vector<uint64_t> stack_memory;
        bool is_aligned_at_call = false;
        bool has_shadow_space = false;
    };

    static CallContext PrepareCall(uint64_t initial_rsp,
                                   uint64_t arg1, uint64_t arg2,
                                   uint64_t arg3 = 0, uint64_t arg4 = 0) {
        CallContext ctx;
        ctx.regs.rcx = arg1;
        ctx.regs.rdx = arg2;
        ctx.regs.r8  = arg3;
        ctx.regs.r9  = arg4;

        // Allocate 32-byte shadow space on stack
        uint64_t call_rsp = initial_rsp - kShadowSpaceBytes;

        // Check 16-byte alignment before CALL
        ctx.is_aligned_at_call = (call_rsp % kRequiredAlignment == 0);
        ctx.has_shadow_space = true;
        ctx.regs.rsp = call_rsp;

        // Simulate stack space (4 slots for caller save)
        ctx.stack_memory.resize(4, 0);
        return ctx;
    }
};

// =============================================================================
// Exercise 3: 64-bit UEFI EFI_SYSTEM_TABLE Layout & ConOut Simulation
// =============================================================================

struct MockEfiTableHeader {
    uint64_t signature;
    uint32_t revision;
    uint32_t header_size;
    uint32_t crc32;
    uint32_t reserved;
};

class MockSimpleTextOutput;

struct MockEfiSystemTable {
    MockEfiTableHeader hdr;           // Offset 0 (24 bytes)
    const wchar_t* firmware_vendor;   // Offset 24 (8 bytes)
    uint32_t firmware_revision;       // Offset 32 (4 bytes)
    uint32_t padding;                 // Offset 36 (4 bytes alignment)
    void* console_in_handle;          // Offset 40 (8 bytes)
    void* con_in;                     // Offset 48 (8 bytes)
    void* console_out_handle;         // Offset 56 (8 bytes)
    MockSimpleTextOutput* con_out;    // Offset 64 (0x40, 8 bytes)
    // Followed by StdErr, RuntimeServices, BootServices...
};

class MockSimpleTextOutput {
public:
    uint64_t (*reset)(MockSimpleTextOutput* This, bool ExtendedVerification) = nullptr;
    uint64_t (*output_string)(MockSimpleTextOutput* This, const char16_t* String) = nullptr;

    std::u16string printed_log;

    MockSimpleTextOutput() {
        reset = [](MockSimpleTextOutput*, bool) -> uint64_t { return 0; };
        output_string = [](MockSimpleTextOutput* This, const char16_t* String) -> uint64_t {
            if (!This || !String) return 0x8000000000000002ULL; // EFI_INVALID_PARAMETER
            while (*String) {
                This->printed_log.push_back(*String);
                ++String;
            }
            return 0; // EFI_SUCCESS
        };
    }
};

// =============================================================================
// Exercise 4: MBR vs GPT Capacity & Linux dev_t Mathematics
// =============================================================================

class StorageMathCalculator {
public:
    static constexpr uint64_t kSectorSizeBytes = 512;

    static uint64_t CalculateMbrMaxCapacityBytes() {
        // 32-bit LBA addressing
        constexpr uint64_t max_sectors = 1ULL << 32;
        return max_sectors * kSectorSizeBytes; // 2 TiB = 2,199,023,255,552 bytes
    }

    static double CalculateMbrMaxCapacityTiB() {
        return static_cast<double>(CalculateMbrMaxCapacityBytes()) / (1ULL << 40);
    }

    static double CalculateGptMaxCapacityZiB() {
        // 64-bit LBA addressing: 2^64 * 512 bytes = 2^73 bytes = 8 ZiB (zebibytes)
        // 1 ZiB = 2^70 bytes
        return static_cast<double>(1ULL << (73 - 70)); // 2^3 = 8.0 ZiB
    }

    static uint64_t PackLinuxDevT(uint32_t major, uint32_t minor) {
        // Glibc standard dev_t bit packing:
        // major is split: bits [31:20] and [19:8]
        // In modern Linux kernel (dev_t = 32-bit):
        // major = (dev >> 20) & 0xfff, minor = dev & 0xfffff (12-bit major, 20-bit minor)
        return (static_cast<uint64_t>(major & 0x0FFF) << 20) | (minor & 0x000FFFFF);
    }

    static std::pair<uint32_t, uint32_t> UnpackLinuxDevT(uint64_t dev) {
        uint32_t major = static_cast<uint32_t>((dev >> 20) & 0x0FFF);
        uint32_t minor = static_cast<uint32_t>(dev & 0x000FFFFF);
        return {major, minor};
    }
};

} // namespace mikanos::exercises::ch01

// =============================================================================
// Main Verification Runner
// =============================================================================

int main() {
    using namespace mikanos::exercises::ch01;

    std::cout << "=========================================================\n";
    std::cout << " MikanOS Chapter 1 Exercises Verification (C++20)\n";
    std::cout << "=========================================================\n\n";

    // 1. Verify PE32+ Binary Header Parsing
    std::cout << "[Step 1] Verifying PE32+ UEFI Binary Parser...\n";
    auto synthetic_bin = Pe32PlusValidator::CreateMinimalSyntheticUefiBinary();
    auto pe_res = Pe32PlusValidator::Validate(synthetic_bin);
    assert(pe_res.is_valid);
    assert(pe_res.is_pe32_plus);
    assert(pe_res.is_x86_64);
    assert(pe_res.is_efi_app);
    assert(pe_res.entry_point == 0x1000);
    assert(pe_res.image_base == 0x140000000ULL);
    std::cout << "  ✓ Synthetic PE32+ binary validated (Entry: 0x"
              << std::hex << pe_res.entry_point << ", ImageBase: 0x" << pe_res.image_base << std::dec << ")\n";

    // 2. Verify Microsoft x64 Calling Convention Simulator
    std::cout << "[Step 2] Verifying Microsoft x64 Calling Convention & Stack Alignment...\n";
    uint64_t initial_stack = 0x7FFF0000; // 16-byte aligned address
    assert(initial_stack % 16 == 0);
    auto ctx = MicrosoftX64FrameSimulator::PrepareCall(initial_stack, 0x1111, 0x2222, 0x3333, 0x4444);
    assert(ctx.is_aligned_at_call);
    assert(ctx.has_shadow_space);
    assert(ctx.regs.rcx == 0x1111);
    assert(ctx.regs.rdx == 0x2222);
    assert(ctx.regs.r8  == 0x3333);
    assert(ctx.regs.r9  == 0x4444);
    assert(ctx.regs.rsp == initial_stack - 32);
    assert(ctx.regs.rsp % 16 == 0);
    std::cout << "  ✓ Microsoft x64 Calling Convention validated (RCX, RDX, R8, R9, Shadow=32B, RSP%16=0)\n";

    // 3. Verify 64-bit EFI_SYSTEM_TABLE Structure & ConOut Offset
    std::cout << "[Step 3] Verifying 64-bit EFI_SYSTEM_TABLE Memory Offsets...\n";
    MockSimpleTextOutput mock_conout;
    MockEfiSystemTable sys_table{};
    sys_table.con_out = &mock_conout;

    size_t con_out_offset = offsetof(MockEfiSystemTable, con_out);
    assert(con_out_offset == 64); // Exactly 64 bytes (0x40)
    std::cout << "  ✓ EFI_SYSTEM_TABLE->ConOut offset confirmed at: " << con_out_offset << " bytes (0x40)\n";

    // Simulate OutputString call via function pointer
    const char16_t msg[] = u"Hello, MikanOS UEFI!\r\n";
    uint64_t uefi_status = sys_table.con_out->output_string(sys_table.con_out, msg);
    assert(uefi_status == 0);
    assert(mock_conout.printed_log == msg);
    std::cout << "  ✓ Simulated ConOut->OutputString invocation successful (Printed 22 UTF-16LE code units)\n";

    // 4. Verify Storage Math (MBR 2TiB, GPT 8ZiB, Linux dev_t)
    std::cout << "[Step 4] Verifying MBR vs GPT & Linux dev_t Mathematics...\n";
    uint64_t mbr_bytes = StorageMathCalculator::CalculateMbrMaxCapacityBytes();
    double mbr_tib = StorageMathCalculator::CalculateMbrMaxCapacityTiB();
    double gpt_zib = StorageMathCalculator::CalculateGptMaxCapacityZiB();

    assert(mbr_bytes == 2199023255552ULL);
    assert(mbr_tib == 2.0);
    assert(gpt_zib == 8.0);
    std::cout << "  ✓ MBR Max Capacity: " << mbr_tib << " TiB (" << mbr_bytes << " bytes)\n";
    std::cout << "  ✓ GPT Max Capacity: " << gpt_zib << " ZiB (2^73 bytes)\n";

    uint32_t orig_maj = 8;  // SCSI / SATA disk
    uint32_t orig_min = 17; // sdb1
    uint64_t packed_dev = StorageMathCalculator::PackLinuxDevT(orig_maj, orig_min);
    auto [unpacked_maj, unpacked_min] = StorageMathCalculator::UnpackLinuxDevT(packed_dev);
    assert(unpacked_maj == orig_maj);
    assert(unpacked_min == orig_min);
    std::cout << "  ✓ Linux dev_t packing verified: (" << orig_maj << ", " << orig_min
              << ") -> 0x" << std::hex << packed_dev << std::dec << " -> (" << unpacked_maj << ", " << unpacked_min << ")\n";

    std::cout << "\n=========================================================\n";
    std::cout << " All Chapter 1 C++20 Exercises Passed Successfully!\n";
    std::cout << "=========================================================\n";
    return 0;
}
