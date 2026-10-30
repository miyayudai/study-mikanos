/**
 * @file first_kernel_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 3.3 (First Kernel & Red Zone)
 *
 * Covers:
 * 1. ELF64 Header structure and entry point resolution at offset 24 (0x18)
 * 2. UEFI Bootloader to Kernel transition lifecycle (Open, GetInfo, AllocatePages at 1MiB, ExitBootServices)
 * 3. MapKey consistency and ExitBootServices retry mechanics
 * 4. System V AMD64 ABI Red Zone (128 bytes) mechanics and interrupt stack corruption trap
 * 5. Kernel compilation flags (-ffreestanding, -mno-red-zone, -fno-exceptions, -fno-rtti)
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mikan::kernel_sim {

// ============================================================================
// 1. Standard 64-bit ELF Header Structure (Elf64_Ehdr)
// ============================================================================

#pragma pack(push, 1)
struct Elf64_Ehdr {
    unsigned char e_ident[16]; // ELF Identification (Magic, Class, Data, Version, OS-ABI)
    uint16_t      e_type;      // Object file type (2 = ET_EXEC)
    uint16_t      e_machine;   // Architecture (0x3E = 62 = EM_X86_64)
    uint32_t      e_version;   // Object file version (1 = EV_CURRENT)
    uint64_t      e_entry;     // Entry point virtual address (OFFSET 24 / 0x18)
    uint64_t      e_phoff;     // Program header table file offset
    uint64_t      e_shoff;     // Section header table file offset
    uint32_t      e_flags;     // Processor-specific flags
    uint16_t      e_ehsize;    // ELF header size in bytes (64 bytes)
    uint16_t      e_phentsize; // Program header table entry size
    uint16_t      e_phnum;     // Program header table entry count
    uint16_t      e_shentsize; // Section header table entry size
    uint16_t      e_shnum;     // Section header table entry count
    uint16_t      e_shstrndx;  // Section name string table index
};
#pragma pack(pop)

static_assert(sizeof(Elf64_Ehdr) == 64, "Elf64_Ehdr must be exactly 64 bytes");
static_assert(offsetof(Elf64_Ehdr, e_entry) == 24, "e_entry must reside exactly at byte offset 24 (0x18)");

// ============================================================================
// 2. Bootloader Transition Simulation
// ============================================================================

enum class EFI_STATUS : uint64_t {
    SUCCESS = 0,
    BUFFER_TOO_SMALL = 5,
    INVALID_PARAMETER = 2,
};

struct MemoryMapState {
    uint64_t map_key{100};
    bool modified{false};
};

class BootloaderSimulator {
public:
    static constexpr uint64_t KERNEL_LOAD_ADDR = 0x100000; // 1MiB physical boundary
    uint64_t kernel_base_addr{0};
    uint64_t kernel_file_size{0};
    std::vector<uint8_t> memory_space;
    MemoryMapState memmap;
    bool boot_services_active{true};
    bool kernel_called{false};
    uint64_t executed_entry_point{0};

    BootloaderSimulator() : memory_space(2 * 1024 * 1024, 0) {}

    // Simulates opening "\\kernel.elf" and obtaining size
    void open_and_stat_kernel(uint64_t file_size) {
        kernel_file_size = file_size;
    }

    // Simulates gBS->AllocatePages(AllocateAddress, EfiLoaderData, pages, &kernel_base_addr)
    void allocate_pages_at_1mb() {
        kernel_base_addr = KERNEL_LOAD_ADDR;
        // Memory allocation dynamically changes the UEFI memory map!
        memmap.map_key += 15;
        memmap.modified = true;
    }

    // Simulates writing ELF binary into allocated memory
    void load_kernel_binary(const Elf64_Ehdr& header) {
        assert(kernel_base_addr >= KERNEL_LOAD_ADDR);
        std::memcpy(&memory_space[kernel_base_addr], &header, sizeof(Elf64_Ehdr));
    }

    // Simulates gBS->ExitBootServices(image_handle, key)
    bool exit_boot_services(uint64_t passed_key, bool allow_retry = true) {
        if (passed_key != memmap.map_key) {
            if (!allow_retry) {
                return false;
            }
            // MikanOS retry logic: GetMemoryMap() renews the key
            passed_key = memmap.map_key;
        }

        boot_services_active = false;
        return true;
    }

    // Simulates the exact MikanOS kernel jump code:
    // UINT64 entry_addr = *(UINT64*)(kernel_base_addr + 24);
    // typedef void EntryPointType(void);
    // EntryPointType* entry_point = (EntryPointType*)entry_addr;
    // entry_point();
    uint64_t jump_to_kernel() {
        assert(!boot_services_active && "Cannot jump to kernel before ExitBootServices!");

        const uint8_t* base_ptr = &memory_space[kernel_base_addr];
        uint64_t entry_addr = *reinterpret_cast<const uint64_t*>(base_ptr + 24);

        executed_entry_point = entry_addr;
        kernel_called = true;
        return entry_addr;
    }
};

// ============================================================================
// 3. System V AMD64 ABI Red Zone Simulator
// ============================================================================

struct StackVar {
    std::string name;
    uint64_t address;
    int64_t offset_from_rsp;
    uint64_t value;
    bool destroyed{false};
};

class RedZoneSimulator {
public:
    static constexpr uint64_t INITIAL_RSP = 0x00007FFFFFFFE000ULL;
    uint64_t rsp{INITIAL_RSP};
    std::unordered_map<uint64_t, uint64_t> stack_memory;
    std::vector<StackVar> tracked_vars;
    bool no_red_zone_mode{false};

    void set_no_red_zone(bool enable) {
        no_red_zone_mode = enable;
    }

    void allocate_local_variables(const std::vector<std::pair<std::string, uint64_t>>& vars) {
        if (!no_red_zone_mode) {
            // RED ZONE ENABLED (Standard User ABI):
            // Compiler DOES NOT decrement RSP! It uses [RSP - 8], [RSP - 16] up to [RSP - 128].
            int64_t offset = -8;
            for (const auto& [name, val] : vars) {
                uint64_t addr = rsp + offset;
                stack_memory[addr] = val;
                tracked_vars.push_back({name, addr, offset, val, false});
                offset -= 8;
            }
        } else {
            // -mno-red-zone ENABLED (Kernel ABI):
            // Compiler explicitly executes `sub rsp, N` BEFORE writing local variables.
            uint64_t frame_size = ((vars.size() * 8 + 15) / 16) * 16;
            rsp -= frame_size; // Safely shift RSP down!

            int64_t offset = 0;
            for (const auto& [name, val] : vars) {
                uint64_t addr = rsp + offset;
                stack_memory[addr] = val;
                tracked_vars.push_back({name, addr, offset, val, false});
                offset += 8;
            }
        }
    }

    // Hardware pushes: SS, RSP, RFLAGS, CS, RIP (5 words = 40 bytes)
    void trigger_cpu_interrupt() {
        uint64_t push_ptr = rsp;
        uint64_t interrupt_frame[5] = {
            0x10,                // Saved SS
            rsp,                 // Saved RSP
            0x0000000000000202,  // Saved RFLAGS (IF=1)
            0x08,                // Saved CS
            0x0000000000100080ULL// Saved RIP
        };

        for (int i = 0; i < 5; ++i) {
            push_ptr -= 8;
            for (auto& var : tracked_vars) {
                if (var.address == push_ptr) {
                    var.destroyed = true; // Overwritten by CPU interrupt hardware!
                }
            }
            stack_memory[push_ptr] = interrupt_frame[i];
        }
        rsp = push_ptr;
    }

    size_t count_corrupted_variables() const {
        size_t count = 0;
        for (const auto& v : tracked_vars) {
            if (v.destroyed) ++count;
        }
        return count;
    }
};

} // namespace mikan::kernel_sim

// ============================================================================
// Self-Testing Suite for CMake & pytest
// ============================================================================

int run_tests() {
    using namespace mikan::kernel_sim;

    std::cout << "[TEST] 1. ELF64 Header Structure and Offset 24..." << std::endl;
    {
        Elf64_Ehdr ehdr;
        std::memset(&ehdr, 0, sizeof(ehdr));
        ehdr.e_ident[0] = 0x7F;
        ehdr.e_ident[1] = 'E';
        ehdr.e_ident[2] = 'L';
        ehdr.e_ident[3] = 'F';
        ehdr.e_ident[4] = 2; // 64-bit
        ehdr.e_ident[5] = 1; // Little endian
        ehdr.e_ident[6] = 1; // Version
        ehdr.e_type = 2;     // ET_EXEC
        ehdr.e_machine = 0x3E; // EM_X86_64
        ehdr.e_version = 1;
        ehdr.e_entry = 0x00100000ULL; // 1MiB entry point

        assert(sizeof(ehdr) == 64);
        assert(offsetof(Elf64_Ehdr, e_entry) == 24);

        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&ehdr);
        uint64_t dereferenced = *reinterpret_cast<const uint64_t*>(raw + 24);
        assert(dereferenced == 0x00100000ULL);
    }
    std::cout << "  -> PASSED: ELF64 offset 24 dereference verified." << std::endl;

    std::cout << "[TEST] 2. Bootloader Transition & ExitBootServices Retry..." << std::endl;
    {
        BootloaderSimulator loader;
        loader.open_and_stat_kernel(4096);
        loader.allocate_pages_at_1mb();

        Elf64_Ehdr ehdr;
        std::memset(&ehdr, 0, sizeof(ehdr));
        ehdr.e_entry = 0x00100050ULL; // Sample entry point
        loader.load_kernel_binary(ehdr);

        // Stale map key fails without retry
        bool fail_without_retry = loader.exit_boot_services(100, false);
        assert(!fail_without_retry);

        // With retry (MikanOS behavior), succeeds!
        bool success_with_retry = loader.exit_boot_services(100, true);
        assert(success_with_retry);
        assert(!loader.boot_services_active);

        // Jump to kernel
        uint64_t entry = loader.jump_to_kernel();
        assert(entry == 0x00100050ULL);
        assert(loader.kernel_called);
    }
    std::cout << "  -> PASSED: Bootloader transition and ExitBootServices verified." << std::endl;

    std::cout << "[TEST] 3. Red Zone Stack Corruption Trap in Kernel Mode..." << std::endl;
    {
        RedZoneSimulator sim;
        sim.set_no_red_zone(false); // Standard ABI (uses Red Zone without moving RSP)

        // Leaf function places 3 local variables in Red Zone [RSP-8, RSP-16, RSP-24]
        sim.allocate_local_variables({
            {"var_a", 0x11111111ULL},
            {"var_b", 0x22222222ULL},
            {"var_c", 0x33333333ULL}
        });

        // Interrupt arrives unexpectedly!
        sim.trigger_cpu_interrupt();

        // Variables in Red Zone are clobbered by CPU hardware push
        assert(sim.count_corrupted_variables() > 0);
    }
    std::cout << "  -> PASSED: Red Zone stack destruction reproduced." << std::endl;

    std::cout << "[TEST] 4. Safe Stack with -mno-red-zone in Kernel Mode..." << std::endl;
    {
        RedZoneSimulator sim;
        sim.set_no_red_zone(true); // Kernel ABI (-mno-red-zone: sub rsp, N first)

        sim.allocate_local_variables({
            {"var_a", 0x11111111ULL},
            {"var_b", 0x22222222ULL},
            {"var_c", 0x33333333ULL}
        });

        // Interrupt arrives!
        sim.trigger_cpu_interrupt();

        // ZERO variables corrupted because RSP was already below them!
        assert(sim.count_corrupted_variables() == 0);
    }
    std::cout << "  -> PASSED: -mno-red-zone immunity to interrupt corruption verified." << std::endl;

    std::cout << "\n>>> ALL FIRST KERNEL & RED ZONE SIMULATION TESTS PASSED! <<<\n" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string_view(argv[1]) == "--test") {
        return run_tests();
    }

    std::cout << "=== MikanOS Chapter 3.3 First Kernel Simulator ===" << std::endl;
    std::cout << "Run with --test to execute verification suite." << std::endl;
    return 0;
}
