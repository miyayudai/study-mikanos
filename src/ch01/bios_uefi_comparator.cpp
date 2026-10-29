/**
 * @file bios_uefi_comparator.cpp
 * @brief Standalone C++20 verification tool for Section 1.7: Booting with UEFI BIOS.
 *
 * Verifies low-level memory layouts of MBR (512 bytes) and GPT (GUID Partition Table)
 * structures, computes 16-bit real-mode linear addresses with A20 wraparound,
 * and contrasts Legacy BIOS multi-stage mode ascension vs UEFI 64-bit Long Mode boot.
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <array>
#include <string>
#include <vector>

#pragma pack(push, 1)

/**
 * @brief 16-byte MBR partition table entry.
 */
struct MbrPartitionEntry {
    uint8_t  boot_indicator;   // 0x80: bootable/active, 0x00: inactive
    uint8_t  starting_chs[3];  // Cylinder-Head-Sector start
    uint8_t  partition_type;   // 0x07: NTFS/exFAT, 0x0B/0x0C: FAT32, 0xEF: ESP
    uint8_t  ending_chs[3];    // CHS end
    uint32_t starting_lba;     // 32-bit starting LBA
    uint32_t sector_count;     // 32-bit sector count
};
static_assert(sizeof(MbrPartitionEntry) == 16, "MbrPartitionEntry must be exactly 16 bytes");

/**
 * @brief 512-byte Master Boot Record (MBR) sector (LBA 0).
 */
struct MbrSector {
    uint8_t           bootstrap_code[446];  // 446 bytes real-mode machine code
    MbrPartitionEntry partition_table[4];   // 4 * 16 = 64 bytes partition table
    uint16_t          boot_signature;       // 0xAA55 (little-endian: 0x55, 0xAA)
};
static_assert(sizeof(MbrSector) == 512, "MbrSector must be exactly 512 bytes");

/**
 * @brief 92-byte EFI Partition Table Header (LBA 1).
 */
struct GptHeader {
    uint8_t  signature[8];              // "EFI PART"
    uint32_t revision;                  // 0x00010000 (v1.0)
    uint32_t header_size;               // 92 bytes
    uint32_t header_crc32;              // CRC32 of header
    uint32_t reserved;                  // Must be zero
    uint64_t current_lba;               // LBA 1
    uint64_t backup_lba;                // Last LBA of disk
    uint64_t first_usable_lba;          // LBA 34
    uint64_t last_usable_lba;           // backup_lba - 33
    uint8_t  disk_guid[16];             // Unique disk GUID
    uint64_t partition_entry_lba;       // LBA 2
    uint32_t num_partition_entries;     // 128
    uint32_t size_of_partition_entry;   // 128 bytes
    uint32_t partition_array_crc32;     // CRC32 of partition entries
};
static_assert(sizeof(GptHeader) == 92, "GptHeader must be exactly 92 bytes");

/**
 * @brief 128-byte GPT Partition Entry.
 */
struct GptEntry {
    uint8_t  partition_type_guid[16];   // e.g. ESP: C12A7328-F81F-11D2-BA4B-00A0C93EC93B
    uint8_t  unique_partition_guid[16]; // Unique partition GUID
    uint64_t starting_lba;              // Starting LBA
    uint64_t ending_lba;                // Ending LBA
    uint64_t attributes;                // Bit flags
    uint16_t partition_name[36];        // 36 UTF-16LE characters (72 bytes)
};
static_assert(sizeof(GptEntry) == 128, "GptEntry must be exactly 128 bytes");

#pragma pack(pop)

/**
 * @brief Computes 16-bit real-mode linear address with optional A20 gate simulation.
 *
 * Formula: LinearAddress = (Segment << 4) + Offset.
 * When A20 is disabled, bit 20 (0x100000) is masked off (wrapped around to 0x000000).
 */
uint32_t calculate_real_mode_linear(uint16_t segment, uint16_t offset, bool a20_enabled) {
    uint32_t raw_linear = (static_cast<uint32_t>(segment) << 4) + offset;
    if (a20_enabled) {
        return raw_linear;
    }
    return raw_linear & ~(1U << 20);
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  Chapter 1.7: Booting with UEFI BIOS Architectural Comparator   \n";
    std::cout << "=================================================================\n\n";

    // 1. Structure Size & Layout Verification
    std::cout << "[1] Low-Level Structure Size & Alignment Verification:\n";
    std::cout << "  - sizeof(MbrPartitionEntry) : " << sizeof(MbrPartitionEntry) << " bytes (Expected: 16)\n";
    std::cout << "  - sizeof(MbrSector)         : " << sizeof(MbrSector) << " bytes (Expected: 512)\n";
    std::cout << "  - sizeof(GptHeader)         : " << sizeof(GptHeader) << " bytes (Expected: 92)\n";
    std::cout << "  - sizeof(GptEntry)          : " << sizeof(GptEntry) << " bytes (Expected: 128)\n";
    std::cout << "  -> All static assertions passed successfully!\n\n";

    // 2. Real Mode Segment:Offset & A20 Gate Wraparound Calculation
    std::cout << "[2] Real Mode Segment:Offset & A20 Gate Wraparound Simulation:\n";
    struct AddressTestCase {
        uint16_t seg;
        uint16_t off;
        const char* desc;
    };

    AddressTestCase cases[] = {
        {0x0000, 0x7C00, "BIOS MBR Load Target (0x07C00)"},
        {0xFFFF, 0x0000, "Reset Vector Base (0xFFFF0)"},
        {0xFFFF, 0x0010, "1MB Boundary Exact Wrap Test (0x100000)"},
        {0xFFFF, 0x0020, "Above 1MB Boundary Test (0x100010)"},
    };

    std::cout << "  " << std::left << std::setw(20) << "Segment:Offset"
              << std::setw(15) << "A20 Enabled"
              << std::setw(15) << "A20 Disabled"
              << "Description\n";
    std::cout << "  ---------------------------------------------------------------------------\n";

    for (const auto& tc : cases) {
        uint32_t addr_a20_on = calculate_real_mode_linear(tc.seg, tc.off, true);
        uint32_t addr_a20_off = calculate_real_mode_linear(tc.seg, tc.off, false);

        std::cout << "  0x" << std::right << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << tc.seg
                  << ":0x" << std::right << std::setw(4) << std::setfill('0') << tc.off << "  "
                  << "0x" << std::right << std::setw(6) << std::setfill('0') << addr_a20_on << "        "
                  << "0x" << std::right << std::setw(6) << std::setfill('0') << addr_a20_off << "        "
                  << std::dec << std::setfill(' ')
                  << tc.desc
                  << (addr_a20_on != addr_a20_off ? " [WRAPPED!]" : "")
                  << "\n";
    }
    std::cout << "\n";

    // 3. Storage Capacity Comparison (MBR 32-bit vs GPT 64-bit LBA)
    std::cout << "[3] Storage Addressing Limits (Sector Size = 512 bytes):\n";
    double mbr_max_tib = (static_cast<double>(1ULL << 32) * 512.0) / (1024.0 * 1024.0 * 1024.0 * 1024.0);
    double gpt_max_zb  = (static_cast<double>(~0ULL) * 512.0) / 1e21; // Decimal ZB
    std::cout << "  - MBR (32-bit LBA) Max Capacity: " << std::fixed << std::setprecision(2)
              << mbr_max_tib << " TiB (2.199 TB) -> Cannot address >2TB modern drives!\n";
    std::cout << "  - GPT (64-bit LBA) Max Capacity: " << std::fixed << std::setprecision(2)
              << gpt_max_zb << " Zettabytes (9.44 * 10^21 bytes) -> Virtually limitless.\n\n";

    // 4. Synthesize Mock MBR & Protective MBR
    std::cout << "[4] Mock MBR & Protective MBR Partition Setup:\n";
    MbrSector mbr{};
    mbr.boot_signature = 0xAA55;
    mbr.partition_table[0].boot_indicator = 0x80; // Bootable
    mbr.partition_table[0].partition_type = 0xEF; // EFI System Partition
    mbr.partition_table[0].starting_lba   = 2048;
    mbr.partition_table[0].sector_count   = 204800; // 100 MiB

    std::cout << "  MBR Partition #1: Bootable=" << (mbr.partition_table[0].boot_indicator == 0x80 ? "YES" : "NO")
              << ", Type=0x" << std::hex << std::uppercase << static_cast<int>(mbr.partition_table[0].partition_type)
              << std::dec << ", StartLBA=" << mbr.partition_table[0].starting_lba
              << ", Sectors=" << mbr.partition_table[0].sector_count
              << " (" << (mbr.partition_table[0].sector_count * 512 / (1024 * 1024)) << " MiB)\n";
    std::cout << "  Boot Signature: 0x" << std::hex << std::uppercase << mbr.boot_signature
              << " (" << (mbr.boot_signature == 0xAA55 ? "VALID" : "INVALID") << ")\n\n";

    std::cout << "[5] Architectural Conclusion:\n";
    std::cout << "  - Legacy BIOS requires complex real->protected->long mode transitions in assembly.\n";
    std::cout << "  - UEFI firmware completes all CPU mode transitions before invoking EfiMain().\n";
    std::cout << "  - MikanOS boots directly in 64-bit Long Mode with full C++ support from day 1.\n";
    std::cout << "=================================================================\n";
    return 0;
}
