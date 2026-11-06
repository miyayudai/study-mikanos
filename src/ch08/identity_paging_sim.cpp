/**
 * @file identity_paging_sim.cpp
 * @brief Chapter 8.6 Identity Paging & 4-Level Paging Architecture Simulator
 *
 * MikanOS Chapter 8.6: Setting Up Identity Paging (osbook_day08b/c)
 * Demonstrates:
 *  1. Need for OS-managed page tables in x86-64 Long Mode (CR0.PG = 1 invariant).
 *  2. 48-bit Canonical Virtual Address decomposition (Sign extension, PML4, PDP, PD, PT, Offset).
 *  3. 2MiB Huge Page mechanism (PS-bit 7 in PDE) eliminating 32,768 Level-1 Page Tables.
 *  4. 64GiB Identity Mapping setup (PML4[0] -> 64 PDP entries -> 64 PDs x 512 x 2MiB).
 *  5. Flag bitmask derivations (0x003 for directory pointers, 0x083 for 2MiB huge pages).
 *  6. 4KiB alignment constraints (alignas(4096)) and low 12-bit address usage.
 *  7. CR3 register loading (SetCR3 / MOV CR3, RDI) and automatic TLB flush semantics.
 *  8. Hardware MMU address translation simulation verifying Virtual == Physical across 64GiB.
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <cstring>
#include <sstream>

namespace mikanos {

// Page sizes and capacity constants
constexpr uint64_t kPageSize4K = 4096;
constexpr uint64_t kPageSize2M = 512 * kPageSize4K;         // 2,097,152 (2MiB)
constexpr uint64_t kPageSize1G = 512 * kPageSize2M;         // 1,073,741,824 (1GiB)
constexpr size_t kPageDirectoryCount = 64;                  // Covers 64 GiB
constexpr uint64_t kTotalMappedBytes = kPageDirectoryCount * kPageSize1G; // 64GiB

/**
 * @brief 48-bit Canonical Linear Address representation matching MikanOS LinearAddress4Level
 */
union LinearAddress4Level {
    uint64_t value;

    struct {
        uint64_t offset : 12; // [11:0]  4KiB in-page offset
        uint64_t page   : 9;  // [20:12] Level 1 Page Table index
        uint64_t dir    : 9;  // [29:21] Level 2 Page Directory index
        uint64_t pdp    : 9;  // [38:30] Level 3 Page Directory Pointer index
        uint64_t pml4   : 9;  // [47:39] Level 4 PML4 index
        uint64_t sign_ext : 16; // [63:48] Sign extension
    } __attribute__((packed)) parts;

    int Part(int page_map_level) const {
        switch (page_map_level) {
        case 0: return parts.offset;
        case 1: return parts.page;
        case 2: return parts.dir;
        case 3: return parts.pdp;
        case 4: return parts.pml4;
        default: return 0;
        }
    }

    uint64_t Offset2M() const {
        return value & 0x1FFFFFu; // 21 bits [20:0]
    }

    bool IsCanonical() const {
        uint64_t bit47 = (value >> 47) & 1;
        uint64_t expected_sign = bit47 ? 0xFFFFu : 0x0000u;
        return parts.sign_ext == expected_sign;
    }
};

/**
 * @brief 64-bit Page Table Entry matching x86-64 hardware and MikanOS PageMapEntry
 */
union PageMapEntry {
    uint64_t data;

    struct {
        uint64_t present        : 1;  // Bit 0: 1 = present in memory
        uint64_t writable       : 1;  // Bit 1: 1 = read/write, 0 = read-only
        uint64_t user           : 1;  // Bit 2: 1 = User (Ring 3), 0 = Supervisor (Ring 0)
        uint64_t write_through  : 1;  // Bit 3: 1 = write-through caching
        uint64_t cache_disable  : 1;  // Bit 4: 1 = disable cache
        uint64_t accessed       : 1;  // Bit 5: 1 = accessed by CPU
        uint64_t dirty          : 1;  // Bit 6: 1 = written to (PTE / 2MB PDE)
        uint64_t huge_page      : 1;  // Bit 7: 1 = 2MiB Huge Page (in PDE) or 1GiB (in PDPTE)
        uint64_t global_page    : 1;  // Bit 8: 1 = global (not flushed on CR3 reload)
        uint64_t available1     : 3;  // Bits 11:9: Available for OS use
        uint64_t addr           : 40; // Bits 51:12: Physical address [51:12]
        uint64_t available2     : 11; // Bits 62:52: Available for OS use
        uint64_t execute_disable: 1;  // Bit 63: 1 = instruction execution disabled (NX)
    } __attribute__((packed)) bits;

    uint64_t PhysicalAddress() const {
        return bits.addr << 12;
    }

    uint64_t HugePageBaseAddress() const {
        return data & 0x000FFFFFFFFE0000ULL;
    }
};

/**
 * @brief Simulator replicating MikanOS Identity Paging setup
 */
class IdentityPagingSimulator {
public:
    // Page tables must be 4KiB-aligned (alignas(4096))
    alignas(kPageSize4K) std::array<uint64_t, 512> pml4_table{};
    alignas(kPageSize4K) std::array<uint64_t, 512> pdp_table{};
    alignas(kPageSize4K) std::array<std::array<uint64_t, 512>, kPageDirectoryCount> page_directory{};

    uint64_t cr3{0};

    IdentityPagingSimulator() {
        SetupIdentityPageTable();
    }

    /**
     * @brief Setup 64GiB Identity Page Table (Direct reproduction of MikanOS kernel/paging.cpp)
     */
    void SetupIdentityPageTable() {
        // PML4[0] points to PDP table: 0x003 = Present (bit 0) | Writable (bit 1)
        pml4_table[0] = reinterpret_cast<uint64_t>(&pdp_table[0]) | 0x003ULL;

        for (size_t i_pdpt = 0; i_pdpt < page_directory.size(); ++i_pdpt) {
            // Each PDP entry points to the corresponding Page Directory: 0x003
            pdp_table[i_pdpt] = reinterpret_cast<uint64_t>(&page_directory[i_pdpt]) | 0x003ULL;

            for (size_t i_pd = 0; i_pd < 512; ++i_pd) {
                // Calculate physical base address for each 2MiB huge page frame
                uint64_t phys_base = i_pdpt * kPageSize1G + i_pd * kPageSize2M;
                // 0x083 = Present (0x1) | Writable (0x2) | Page Size / Huge Page (0x80)
                page_directory[i_pdpt][i_pd] = phys_base | 0x083ULL;
            }
        }

        // Set CR3 register pointing to PML4 table
        SetCR3(reinterpret_cast<uint64_t>(&pml4_table[0]));
    }

    void SetCR3(uint64_t value) {
        cr3 = value;
    }

    /**
     * @brief Hardware MMU Address Translation Simulation
     */
    struct TranslationResult {
        bool success{false};
        uint64_t linear_address{0};
        uint64_t physical_address{0};
        bool is_identity{false};
        std::string error_message{};
        int pml4_idx{0};
        int pdp_idx{0};
        int pd_idx{0};
        uint64_t offset_2mb{0};
    };

    TranslationResult Translate(uint64_t linear_addr) const {
        TranslationResult res;
        res.linear_address = linear_addr;

        LinearAddress4Level vaddr{.value = linear_addr};
        if (!vaddr.IsCanonical()) {
            res.success = false;
            res.error_message = "Non-canonical virtual address (bits 63:48 do not match bit 47)";
            return res;
        }

        res.pml4_idx = vaddr.parts.pml4;
        res.pdp_idx = vaddr.parts.pdp;
        res.pd_idx = vaddr.parts.dir;
        res.offset_2mb = vaddr.Offset2M();

        // 1. Check PML4
        PageMapEntry pml4e{.data = pml4_table[vaddr.parts.pml4]};
        if (!pml4e.bits.present) {
            res.success = false;
            res.error_message = "PML4 entry is not present (Page Fault #PF)";
            return res;
        }

        // 2. Check PDP
        if (vaddr.parts.pdp >= kPageDirectoryCount) {
            res.success = false;
            res.error_message = "PDP index exceeds mapped 64GiB limit";
            return res;
        }

        PageMapEntry pdpe{.data = pdp_table[vaddr.parts.pdp]};
        if (!pdpe.bits.present) {
            res.success = false;
            res.error_message = "PDP entry is not present (Page Fault #PF)";
            return res;
        }

        // 3. Check Page Directory
        PageMapEntry pde{.data = page_directory[vaddr.parts.pdp][vaddr.parts.dir]};
        if (!pde.bits.present) {
            res.success = false;
            res.error_message = "PD entry is not present (Page Fault #PF)";
            return res;
        }

        if (!pde.bits.huge_page) {
            res.success = false;
            res.error_message = "PD entry is not configured as 2MiB Huge Page (PS bit is 0)";
            return res;
        }

        // 4. Resolve 2MiB physical address
        uint64_t phys_base = pde.HugePageBaseAddress();
        res.physical_address = phys_base + res.offset_2mb;
        res.is_identity = (res.physical_address == res.linear_address);
        res.success = true;
        return res;
    }

    /**
     * @brief Verify 4KiB alignment of table base addresses
     */
    bool VerifyAlignments() const {
        if (reinterpret_cast<uintptr_t>(&pml4_table[0]) % kPageSize4K != 0) return false;
        if (reinterpret_cast<uintptr_t>(&pdp_table[0]) % kPageSize4K != 0) return false;
        for (size_t i = 0; i < kPageDirectoryCount; ++i) {
            if (reinterpret_cast<uintptr_t>(&page_directory[i][0]) % kPageSize4K != 0) return false;
        }
        return true;
    }
};

} // namespace mikanos

void PrintHeader(const std::string& title) {
    std::cout << "\n======================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "======================================================================\n";
}

int main(int argc, char** argv) {
    bool run_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            run_tests = true;
        }
    }

    mikanos::IdentityPagingSimulator sim;

    if (!run_tests) {
        PrintHeader("MikanOS Chapter 8.6: Identity Paging & 4-Level Paging Architecture");
        std::cout << "Configuration Summary:\n";
        std::cout << "  PML4 Table Address: 0x" << std::hex << reinterpret_cast<uintptr_t>(&sim.pml4_table[0]) << "\n";
        std::cout << "  PDP Table Address : 0x" << reinterpret_cast<uintptr_t>(&sim.pdp_table[0]) << "\n";
        std::cout << "  PD Count          : " << std::dec << mikanos::kPageDirectoryCount << " tables\n";
        std::cout << "  Total Mapped Space: " << (mikanos::kTotalMappedBytes / (1024ULL * 1024 * 1024)) << " GiB\n";
        std::cout << "  CR3 Register      : 0x" << std::hex << sim.cr3 << std::dec << "\n\n";

        std::cout << "Sample Translations across 64GiB Address Space:\n";
        std::vector<uint64_t> test_addrs = {
            0x00000000ULL,               // 0 (Beginning of RAM)
            0x00100000ULL,               // 1MiB (Kernel load address)
            0x00200000ULL,               // 2MiB boundary (PD[0][1])
            0x40000000ULL,               // 1GiB boundary (PDP[1], PD[1][0])
            0x80000000ULL,               // 2GiB boundary (PDP[2])
            0xFFFFFFFFULL,               // 4GiB - 1 (Top of 32-bit space)
            0x100000000ULL,              // 4GiB (PDP[4])
            0xF80000000ULL,              // 62GiB (PDP[62])
            0xFFFE00000ULL,              // 64GiB - 2MiB (PDP[63], PD[63][511])
        };

        for (uint64_t addr : test_addrs) {
            auto res = sim.Translate(addr);
            std::cout << "  Virtual: 0x" << std::hex << std::setw(10) << std::setfill('0') << addr
                      << " -> Physical: 0x" << std::setw(10) << std::setfill('0') << res.physical_address
                      << " [PML4:" << std::dec << res.pml4_idx
                      << ", PDP:" << res.pdp_idx
                      << ", PD:" << res.pd_idx
                      << ", Off2M: 0x" << std::hex << res.offset_2mb << "]"
                      << " (Identity: " << (res.is_identity ? "MATCH" : "FAIL") << ")\n";
        }
    } else {
        // Run comprehensive unit tests
        // Test 1: Alignment verification (4KiB = 0x1000 boundary)
        assert(sim.VerifyAlignments());
        assert(sim.cr3 % mikanos::kPageSize4K == 0);

        // Test 2: PML4[0] points to PDP with 0x003 flags
        mikanos::PageMapEntry pml4_0{.data = sim.pml4_table[0]};
        assert(pml4_0.bits.present == 1);
        assert(pml4_0.bits.writable == 1);
        assert(pml4_0.bits.huge_page == 0);
        assert(pml4_0.PhysicalAddress() == reinterpret_cast<uintptr_t>(&sim.pdp_table[0]));

        // Test 3: Unmapped PML4 entries [1..511] must be 0
        for (size_t i = 1; i < 512; ++i) {
            assert(sim.pml4_table[i] == 0);
        }

        // Test 4: PDP entries [0..63] point to page_directory[i] with 0x003 flags
        for (size_t i = 0; i < mikanos::kPageDirectoryCount; ++i) {
            mikanos::PageMapEntry pdpe{.data = sim.pdp_table[i]};
            assert(pdpe.bits.present == 1);
            assert(pdpe.bits.writable == 1);
            assert(pdpe.bits.huge_page == 0);
            assert(pdpe.PhysicalAddress() == reinterpret_cast<uintptr_t>(&sim.page_directory[i][0]));
        }

        // Test 5: PDP unmapped entries [64..511] must be 0
        for (size_t i = mikanos::kPageDirectoryCount; i < 512; ++i) {
            assert(sim.pdp_table[i] == 0);
        }

        // Test 6: Page Directory entries must have 0x083 flags and correct 2MiB physical base addresses
        for (size_t i_pdpt = 0; i_pdpt < mikanos::kPageDirectoryCount; ++i_pdpt) {
            for (size_t i_pd = 0; i_pd < 512; ++i_pd) {
                uint64_t expected_base = i_pdpt * mikanos::kPageSize1G + i_pd * mikanos::kPageSize2M;
                uint64_t raw_entry = sim.page_directory[i_pdpt][i_pd];
                assert((raw_entry & 0xFFF) == 0x083); // Flags: P=1, RW=1, PS=1

                mikanos::PageMapEntry pde{.data = raw_entry};
                assert(pde.bits.present == 1);
                assert(pde.bits.writable == 1);
                assert(pde.bits.huge_page == 1);
                assert(pde.HugePageBaseAddress() == expected_base);
            }
        }

        // Test 7: Virtual == Physical Identity Mapping verification across multiple key addresses
        std::vector<uint64_t> sample_addresses = {
            0x00000000ULL,
            0x00001000ULL,
            0x00100000ULL,
            0x001FFFFFULL,
            0x00200000ULL,
            0x3FE00000ULL,
            0x40000000ULL,
            0x80000000ULL,
            0x100000000ULL,
            0x200000000ULL,
            0xF00000000ULL,
            0xFFFE00000ULL,
            0xFFFFFFFFFULL, // 64GiB - 1
        };

        for (uint64_t addr : sample_addresses) {
            auto res = sim.Translate(addr);
            assert(res.success);
            assert(res.is_identity);
            assert(res.physical_address == addr);
        }

        // Test 8: Out-of-bounds address rejection (>= 64GiB)
        uint64_t beyond_64g = 0x1000000000ULL; // 64GiB
        auto res_beyond = sim.Translate(beyond_64g);
        assert(!res_beyond.success);
        assert(res_beyond.error_message.find("PDP index exceeds") != std::string::npos);

        // Test 9: Non-canonical address rejection
        uint64_t non_canonical = 0x0001000000000000ULL; // bit 48 is 1 without sign extension
        auto res_non_canon = sim.Translate(non_canonical);
        assert(!res_non_canon.success);
        assert(res_non_canon.error_message.find("Non-canonical") != std::string::npos);

        std::cout << "All C++20 Verification Unit Tests Passed Successfully!\n";
    }

    return 0;
}
