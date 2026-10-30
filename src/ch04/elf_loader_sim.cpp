/**
 * @file elf_loader_sim.cpp
 * @brief C++20 Simulation and Low-Level Verification of MikanOS Chapter 4.5 (ELF Loader & .bss Zero-Clear)
 *
 * Covers:
 * 1. ELF-64 specifications: Elf64_Ehdr (64B) and Elf64_Phdr (56B) binary layouts and static assertions
 * 2. Address range calculation: CalcLoadAddressRange() for PT_LOAD segments and 4KiB page count math
 * 3. Segment deployment & .bss clearing: CopyLoadSegments() with filesz copy and memsz - filesz zero-fill
 * 4. Two-stage bootloader architecture: AllocatePool -> CalcRange -> AllocatePages -> Copy -> FreePool
 * 5. Direct entry point extraction: reading e_entry at offset 24 from kernel base address
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace mikan::ch04_elf {

// ============================================================================
// 1. ELF-64 Type Definitions and Header Structures (kernel/elf.hpp compatible)
// ============================================================================

using Elf64_Addr   = uintptr_t;
using Elf64_Off    = uint64_t;
using Elf64_Half   = uint16_t;
using Elf64_Word   = uint32_t;
using Elf64_Sword  = int32_t;
using Elf64_Xword  = uint64_t;
using Elf64_Sxword = int64_t;

constexpr size_t EI_NIDENT = 16;
constexpr uint8_t ELF_MAG0 = 0x7f;
constexpr uint8_t ELF_MAG1 = 'E';
constexpr uint8_t ELF_MAG2 = 'L';
constexpr uint8_t ELF_MAG3 = 'F';

constexpr Elf64_Half ET_EXEC = 2;
constexpr Elf64_Half ET_DYN  = 3;
constexpr Elf64_Half EM_X86_64 = 62;

constexpr Elf64_Word PT_NULL    = 0;
constexpr Elf64_Word PT_LOAD    = 1;
constexpr Elf64_Word PT_DYNAMIC = 2;
constexpr Elf64_Word PT_INTERP  = 3;
constexpr Elf64_Word PT_NOTE    = 4;
constexpr Elf64_Word PT_SHLIB   = 5;
constexpr Elf64_Word PT_PHDR    = 6;
constexpr Elf64_Word PT_TLS     = 7;

constexpr Elf64_Word PF_X = 1;  // Execute
constexpr Elf64_Word PF_W = 2;  // Write
constexpr Elf64_Word PF_R = 4;  // Read

#pragma pack(push, 1)
struct Elf64_Ehdr {
    unsigned char e_ident[EI_NIDENT];
    Elf64_Half    e_type;
    Elf64_Half    e_machine;
    Elf64_Word    e_version;
    Elf64_Addr    e_entry;
    Elf64_Off     e_phoff;
    Elf64_Off     e_shoff;
    Elf64_Word    e_flags;
    Elf64_Half    e_ehsize;
    Elf64_Half    e_phentsize;
    Elf64_Half    e_phnum;
    Elf64_Half    e_shentsize;
    Elf64_Half    e_shnum;
    Elf64_Half    e_shstrndx;
};

struct Elf64_Phdr {
    Elf64_Word  p_type;
    Elf64_Word  p_flags;
    Elf64_Off   p_offset;
    Elf64_Addr  p_vaddr;
    Elf64_Addr  p_paddr;
    Elf64_Xword p_filesz;
    Elf64_Xword p_memsz;
    Elf64_Xword p_align;
};
#pragma pack(pop)

static_assert(sizeof(Elf64_Ehdr) == 64, "Elf64_Ehdr must be exactly 64 bytes");
static_assert(sizeof(Elf64_Phdr) == 56, "Elf64_Phdr must be exactly 56 bytes");
static_assert(offsetof(Elf64_Ehdr, e_entry) == 24, "e_entry must be at byte offset 24 in Elf64_Ehdr");
static_assert(offsetof(Elf64_Ehdr, e_phoff) == 32, "e_phoff must be at byte offset 32 in Elf64_Ehdr");
static_assert(offsetof(Elf64_Phdr, p_vaddr) == 16, "p_vaddr must be at byte offset 16 in Elf64_Phdr");
static_assert(offsetof(Elf64_Phdr, p_filesz) == 32, "p_filesz must be at byte offset 32 in Elf64_Phdr");
static_assert(offsetof(Elf64_Phdr, p_memsz) == 40, "p_memsz must be at byte offset 40 in Elf64_Phdr");

// ============================================================================
// 2. MikanLoaderPkg Core Loader Logic (Main.c exact implementation)
// ============================================================================

/**
 * @brief Calculate the minimum and maximum virtual address spanning all PT_LOAD segments.
 *
 * Direct reproduction of CalcLoadAddressRange() in MikanLoaderPkg/Main.c:
 *   Elf64_Phdr* phdr = (Elf64_Phdr*)((UINT64)ehdr + ehdr->e_phoff);
 *   *first = MAX_UINT64;
 *   *last = 0;
 *   for (Elf64_Half i = 0; i < ehdr->e_phnum; ++i) {
 *     if (phdr[i].p_type != PT_LOAD) continue;
 *     *first = MIN(*first, phdr[i].p_vaddr);
 *     *last = MAX(*last, phdr[i].p_vaddr + phdr[i].p_memsz);
 *   }
 */
inline void CalcLoadAddressRange(const Elf64_Ehdr* ehdr, uint64_t* first, uint64_t* last) {
    auto phdr = reinterpret_cast<const Elf64_Phdr*>(
        reinterpret_cast<uintptr_t>(ehdr) + ehdr->e_phoff);
    *first = UINT64_MAX;
    *last = 0;
    for (Elf64_Half i = 0; i < ehdr->e_phnum; ++i) {
        if (phdr[i].p_type != PT_LOAD) continue;
        *first = std::min(*first, static_cast<uint64_t>(phdr[i].p_vaddr));
        *last = std::max(*last, static_cast<uint64_t>(phdr[i].p_vaddr + phdr[i].p_memsz));
    }
}

/**
 * @brief Calculate the number of 4KiB pages required to encompass the address range.
 *
 * Formula: (kernel_last_addr - kernel_first_addr + 0xfff) / 0x1000
 */
inline uint64_t CalculateRequiredPages(uint64_t first, uint64_t last, uint64_t page_size = 0x1000) {
    if (last <= first) return 0;
    return (last - first + page_size - 1) / page_size;
}

/**
 * @brief Copy PT_LOAD segments to destination memory and zero-fill remaining .bss.
 *
 * Direct reproduction of CopyLoadSegments() in MikanLoaderPkg/Main.c:
 *   Elf64_Phdr* phdr = (Elf64_Phdr*)((UINT64)ehdr + ehdr->e_phoff);
 *   for (Elf64_Half i = 0; i < ehdr->e_phnum; ++i) {
 *     if (phdr[i].p_type != PT_LOAD) continue;
 *     UINT64 segm_in_file = (UINT64)ehdr + phdr[i].p_offset;
 *     CopyMem((VOID*)phdr[i].p_vaddr, (VOID*)segm_in_file, phdr[i].p_filesz);
 *     UINTN remain_bytes = phdr[i].p_memsz - phdr[i].p_filesz;
 *     SetMem((VOID*)(phdr[i].p_vaddr + phdr[i].p_filesz), remain_bytes, 0);
 *   }
 */
inline void CopyLoadSegments(
    const Elf64_Ehdr* ehdr,
    uint8_t* memory_base,
    uint64_t memory_base_vaddr)
{
    auto phdr = reinterpret_cast<const Elf64_Phdr*>(
        reinterpret_cast<uintptr_t>(ehdr) + ehdr->e_phoff);
    for (Elf64_Half i = 0; i < ehdr->e_phnum; ++i) {
        if (phdr[i].p_type != PT_LOAD) continue;

        auto segm_in_file = reinterpret_cast<const uint8_t*>(
            reinterpret_cast<uintptr_t>(ehdr) + phdr[i].p_offset);
        
        uint8_t* dest = memory_base + (phdr[i].p_vaddr - memory_base_vaddr);

        // Copy file content: CopyMem((VOID*)phdr[i].p_vaddr, (VOID*)segm_in_file, phdr[i].p_filesz)
        if (phdr[i].p_filesz > 0) {
            std::memcpy(dest, segm_in_file, phdr[i].p_filesz);
        }

        // Zero-fill .bss: SetMem((VOID*)(phdr[i].p_vaddr + phdr[i].p_filesz), remain_bytes, 0)
        uint64_t remain_bytes = phdr[i].p_memsz - phdr[i].p_filesz;
        if (remain_bytes > 0) {
            std::memset(dest + phdr[i].p_filesz, 0, remain_bytes);
        }
    }
}

// ============================================================================
// 3. Synthetic ELF Binary Construction Helper
// ============================================================================

struct SegmentSpec {
    Elf64_Word type;
    Elf64_Word flags;
    Elf64_Addr vaddr;
    std::vector<uint8_t> data;
    Elf64_Xword memsz;
    Elf64_Xword align;
};

inline std::vector<uint8_t> BuildSyntheticElf(
    Elf64_Addr entry_point,
    const std::vector<SegmentSpec>& segments,
    Elf64_Half machine = EM_X86_64,
    bool include_header_in_seg0 = false)
{
    Elf64_Ehdr ehdr{};
    ehdr.e_ident[0] = ELF_MAG0;
    ehdr.e_ident[1] = ELF_MAG1;
    ehdr.e_ident[2] = ELF_MAG2;
    ehdr.e_ident[3] = ELF_MAG3;
    ehdr.e_ident[4] = 2; // 64-bit
    ehdr.e_ident[5] = 1; // Little endian
    ehdr.e_ident[6] = 1; // Version 1
    ehdr.e_ident[7] = 0; // System V ABI
    ehdr.e_type = ET_EXEC;
    ehdr.e_machine = machine;
    ehdr.e_version = 1;
    ehdr.e_entry = entry_point;
    ehdr.e_phoff = sizeof(Elf64_Ehdr);
    ehdr.e_ehsize = sizeof(Elf64_Ehdr);
    ehdr.e_phentsize = sizeof(Elf64_Phdr);
    ehdr.e_phnum = static_cast<Elf64_Half>(segments.size());

    // Calculate header space
    const size_t phdrs_size = segments.size() * sizeof(Elf64_Phdr);
    const size_t headers_size = sizeof(Elf64_Ehdr) + phdrs_size;
    size_t current_file_offset = headers_size;
    if (current_file_offset % 16 != 0) {
        current_file_offset += (16 - (current_file_offset % 16));
    }

    std::vector<Elf64_Phdr> phdrs;
    phdrs.reserve(segments.size());
    std::vector<std::vector<uint8_t>> payloads;
    payloads.reserve(segments.size());

    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& seg = segments[i];
        Elf64_Phdr ph{};
        ph.p_type = seg.type;
        ph.p_flags = seg.flags;
        ph.p_vaddr = seg.vaddr;
        ph.p_paddr = seg.vaddr;
        ph.p_align = seg.align;

        if (i == 0 && include_header_in_seg0 && seg.type == PT_LOAD) {
            // LLD style: Segment 0 starts at file offset 0, covering Ehdr and Phdrs
            ph.p_offset = 0;
            ph.p_filesz = current_file_offset + seg.data.size();
            ph.p_memsz = std::max<Elf64_Xword>(seg.memsz, ph.p_filesz);
            current_file_offset += seg.data.size();
        } else {
            ph.p_offset = current_file_offset;
            ph.p_filesz = seg.data.size();
            ph.p_memsz = std::max<Elf64_Xword>(seg.memsz, seg.data.size());
            current_file_offset += seg.data.size();
        }

        phdrs.push_back(ph);
        payloads.push_back(seg.data);
    }

    // Assemble full ELF file binary
    std::vector<uint8_t> file_bytes(current_file_offset, 0);
    std::memcpy(file_bytes.data(), &ehdr, sizeof(Elf64_Ehdr));
    std::memcpy(file_bytes.data() + ehdr.e_phoff, phdrs.data(), phdrs_size);

    for (size_t i = 0; i < segments.size(); ++i) {
        if (!payloads[i].empty()) {
            if (i == 0 && include_header_in_seg0 && phdrs[i].p_type == PT_LOAD) {
                // Payload is placed after headers in file
                const size_t seg0_payload_offset = headers_size + (headers_size % 16 != 0 ? 16 - (headers_size % 16) : 0);
                std::memcpy(file_bytes.data() + seg0_payload_offset, payloads[i].data(), payloads[i].size());
            } else {
                std::memcpy(file_bytes.data() + phdrs[i].p_offset, payloads[i].data(), payloads[i].size());
            }
        }
    }

    return file_bytes;
}

// ============================================================================
// 4. Verification Test Cases
// ============================================================================

/**
 * @brief Test 1: Verify Elf64_Ehdr & Elf64_Phdr structure layouts and offsets.
 */
void TestElfHeaderLayoutAndOffsets() {
    std::cout << "[Test 1] Verifying ELF-64 header structures and offsets... ";

    assert(sizeof(Elf64_Ehdr) == 64);
    assert(sizeof(Elf64_Phdr) == 56);

    Elf64_Ehdr ehdr{};
    ehdr.e_entry = 0x100000;

    // Verify entry address access via byte offset 24
    uint64_t read_entry = *reinterpret_cast<const uint64_t*>(
        reinterpret_cast<const uint8_t*>(&ehdr) + 24);
    assert(read_entry == 0x100000);

    std::cout << "PASSED (Ehdr: 64B, Phdr: 56B, EntryOffset: 24)" << std::endl;
}

/**
 * @brief Test 2: Verify CalcLoadAddressRange and required page count calculation.
 */
void TestCalcLoadAddressRangeAndPages() {
    std::cout << "[Test 2] Verifying CalcLoadAddressRange & page calculation... ";

    // Construct mock segments:
    // Seg 0 (Code): vaddr 0x100000, filesz 0x800, memsz 0x800 (PT_LOAD)
    // Seg 1 (Dynamic): vaddr 0x0, filesz 0x100, memsz 0x100 (PT_DYNAMIC: should be ignored!)
    // Seg 2 (Data+BSS): vaddr 0x101000, filesz 0x400, memsz 0x1200 (PT_LOAD) -> span up to 0x102200
    std::vector<SegmentSpec> segs = {
        {PT_LOAD, PF_R | PF_X, 0x100000, std::vector<uint8_t>(0x800, 0x90), 0x800, 0x1000},
        {PT_DYNAMIC, PF_R | PF_W, 0x500000, std::vector<uint8_t>(0x100, 0x00), 0x100, 0x8},
        {PT_LOAD, PF_R | PF_W, 0x101000, std::vector<uint8_t>(0x400, 0xAA), 0x1200, 0x1000},
    };

    auto elf_data = BuildSyntheticElf(0x100050, segs);
    auto ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf_data.data());

    uint64_t first = 0, last = 0;
    CalcLoadAddressRange(ehdr, &first, &last);

    assert(first == 0x100000);
    assert(last == (0x101000 + 0x1200));  // 0x102200

    // Range length = 0x2200 (8704 bytes)
    // Required 4KiB pages: (0x2200 + 0xfff) / 0x1000 = (0x31ff) / 0x1000 = 3 pages (0x3000 = 12288 bytes)
    uint64_t pages = CalculateRequiredPages(first, last, 0x1000);
    assert(pages == 3);

    std::cout << "PASSED (first: 0x" << std::hex << first << ", last: 0x" << last
              << ", pages: " << std::dec << pages << ")" << std::endl;
}

/**
 * @brief Test 3: Verify segment data copy and strict .bss zero-fill.
 */
void TestSegmentCopyAndBssZeroClear() {
    std::cout << "[Test 3] Verifying CopyLoadSegments and .bss zero-clearing... ";

    // Seg 0: Code (vaddr 0x100000, filesz 0x200, memsz 0x200)
    // Seg 1: Data + BSS (vaddr 0x101000, filesz 0x100 [0xBB], memsz 0x300 [0x200 BSS])
    std::vector<uint8_t> code_payload(0x200, 0xCC); // int 3 instructions
    std::vector<uint8_t> data_payload(0x100, 0x55); // initialized data

    std::vector<SegmentSpec> segs = {
        {PT_LOAD, PF_R | PF_X, 0x100000, code_payload, 0x200, 0x1000},
        {PT_LOAD, PF_R | PF_W, 0x101000, data_payload, 0x300, 0x1000},
    };

    auto elf_data = BuildSyntheticElf(0x100000, segs);
    auto ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf_data.data());

    uint64_t first = 0, last = 0;
    CalcLoadAddressRange(ehdr, &first, &last);
    assert(first == 0x100000);
    assert(last == 0x101300);

    // Prepare dirty allocated memory (simulating dirty physical RAM filled with 0xFF)
    const size_t total_span = last - first;
    std::vector<uint8_t> memory_pool(total_span, 0xFF);

    CopyLoadSegments(ehdr, memory_pool.data(), first);

    // Verify Seg 0 (code at offset 0x000 - 0x200)
    for (size_t i = 0; i < 0x200; ++i) {
        assert(memory_pool[i] == 0xCC);
    }

    // Verify gap between Seg 0 and Seg 1 (0x200 to 0x1000) remains untouched (or dirty)
    // In actual OS, pages are allocated, but segments are copied to their respective vaddr
    assert(memory_pool[0x500] == 0xFF);

    // Verify Seg 1 Data (0x101000 - 0x100000 = offset 0x1000, size 0x100)
    const size_t data_offset = 0x1000;
    for (size_t i = 0; i < 0x100; ++i) {
        assert(memory_pool[data_offset + i] == 0x55);
    }

    // Verify Seg 1 BSS (offset 0x1100, size 0x200): MUST BE 0x00!
    const size_t bss_offset = 0x1100;
    for (size_t i = 0; i < 0x200; ++i) {
        assert(memory_pool[bss_offset + i] == 0x00);
    }

    std::cout << "PASSED (Code: CC, Data: 55, BSS: strictly zeroed)" << std::endl;
}

/**
 * @brief Test 4: Verify full two-stage loader flow and entry point resolution.
 */
void TestTwoStageLoaderLifecycle() {
    std::cout << "[Test 4] Verifying two-stage loading flow & entry point resolution... ";

    constexpr Elf64_Addr expected_entry = 0x100080;
    std::vector<SegmentSpec> segs = {
        {PT_LOAD, PF_R | PF_X, 0x100000, std::vector<uint8_t>(0x100, 0x90), 0x100, 0x1000},
    };

    auto elf_data = BuildSyntheticElf(expected_entry, segs, EM_X86_64, true);

    // Stage 1: Load entire file into temporary buffer (AllocatePool simulation)
    auto temp_pool = std::make_unique<uint8_t[]>(elf_data.size());
    std::memcpy(temp_pool.get(), elf_data.data(), elf_data.size());

    // Stage 2: Calculate address range from temporary buffer
    auto kernel_ehdr = reinterpret_cast<const Elf64_Ehdr*>(temp_pool.get());
    uint64_t k_first = 0, k_last = 0;
    CalcLoadAddressRange(kernel_ehdr, &k_first, &k_last);

    uint64_t num_pages = CalculateRequiredPages(k_first, k_last, 0x1000);
    assert(num_pages == 1);

    // Stage 3: Allocate dedicated physical pages at k_first (AllocatePages simulation)
    auto kernel_dest = std::make_unique<uint8_t[]>(num_pages * 0x1000);
    std::memset(kernel_dest.get(), 0xAA, num_pages * 0x1000);

    // Stage 4: Copy segments and zero-fill BSS
    CopyLoadSegments(kernel_ehdr, kernel_dest.get(), k_first);

    // Stage 5: Free temporary buffer (FreePool simulation)
    temp_pool.reset();

    // Stage 6: Retrieve entry point from kernel_first_addr + 24
    // (Because Seg 0 contains the Ehdr header at offset 0 in MikanOS kernel binary)
    // Or directly from kernel_ehdr->e_entry
    uint64_t entry_addr = *reinterpret_cast<const uint64_t*>(kernel_dest.get() + 24);
    assert(entry_addr == expected_entry);

    std::cout << "PASSED (Entry point: 0x" << std::hex << entry_addr << " verified)" << std::dec << std::endl;
}

/**
 * @brief Test 5: Verify robustness against non-LOAD segments and empty files.
 */
void TestLoaderEdgeCases() {
    std::cout << "[Test 5] Verifying loader edge cases (no PT_LOAD, empty)... ";

    // Empty segments list
    std::vector<SegmentSpec> segs_empty{};
    auto elf_empty = BuildSyntheticElf(0x100000, segs_empty);
    auto ehdr_empty = reinterpret_cast<const Elf64_Ehdr*>(elf_empty.data());

    uint64_t first = 0, last = 0;
    CalcLoadAddressRange(ehdr_empty, &first, &last);
    assert(first == UINT64_MAX && last == 0);
    assert(CalculateRequiredPages(first, last) == 0);

    // Only non-LOAD segments
    std::vector<SegmentSpec> segs_note = {
        {PT_NOTE, PF_R, 0x1000, std::vector<uint8_t>(64, 0), 64, 4},
    };
    auto elf_note = BuildSyntheticElf(0x100000, segs_note);
    auto ehdr_note = reinterpret_cast<const Elf64_Ehdr*>(elf_note.data());

    CalcLoadAddressRange(ehdr_note, &first, &last);
    assert(first == UINT64_MAX && last == 0);

    std::cout << "PASSED (Non-LOAD handled safely)" << std::endl;
}

}  // namespace mikan::ch04_elf

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " MikanOS Chapter 4.5: ELF Loader & .bss Zero-Clear Verification  " << std::endl;
    std::cout << "=================================================================" << std::endl;

    mikan::ch04_elf::TestElfHeaderLayoutAndOffsets();
    mikan::ch04_elf::TestCalcLoadAddressRangeAndPages();
    mikan::ch04_elf::TestSegmentCopyAndBssZeroClear();
    mikan::ch04_elf::TestTwoStageLoaderLifecycle();
    mikan::ch04_elf::TestLoaderEdgeCases();

    std::cout << "=================================================================" << std::endl;
    std::cout << " All Section 4.5 C++20 Verification Tests Passed Successfully!   " << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
