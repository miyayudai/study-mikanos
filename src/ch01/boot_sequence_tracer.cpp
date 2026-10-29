/**
 * @file boot_sequence_tracer.cpp
 * @brief C++20 Host-side CPU Reset Vector & UEFI Boot Sequence Tracer
 * 
 * Part of study-mikanos: Chapter 1.5 "What Actually Happened Behind the Scenes"
 * Demonstrates:
 *   1. Hardware reset vector calculation (0xFFFFFFF0 / 4GiB - 16B) and CPU state
 *   2. UEFI Platform Initialization (PI) 5-phase execution flow (SEC, PEI, DXE, BDS, TSL)
 *   3. Architecture-specific UEFI fallback boot path resolution (\\EFI\\BOOT\\BOOTX64.EFI)
 *   4. UEFI Boot Manager decision tree (NVRAM BootOrder vs Removable Media Fallback)
 *   5. PE32+ loader relocation and Microsoft x64 calling convention register state
 *   6. EFI_SYSTEM_TABLE ConOut->OutputString simulation with UTF-16 decoding
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <memory>
#include <map>

namespace mikanos {

// -----------------------------------------------------------------------------
// 1. CPU Modes and Hardware Reset State
// -----------------------------------------------------------------------------

enum class CpuMode {
    RealMode16,
    ProtectedMode32,
    CompatibilityMode64,
    LongMode64
};

inline const char* cpu_mode_to_string(CpuMode mode) {
    switch (mode) {
        case CpuMode::RealMode16: return "16-bit Real Mode (with 4GiB CS base)";
        case CpuMode::ProtectedMode32: return "32-bit Protected Mode (Flat)";
        case CpuMode::CompatibilityMode64: return "IA-32e Compatibility Mode";
        case CpuMode::LongMode64: return "64-bit Long Mode (IA-32e 4-Level Paging)";
    }
    return "Unknown";
}

enum class Architecture {
    X86_64,
    AArch64,
    IA32,
    ARM32,
    RiscV64
};

inline const char* arch_to_string(Architecture arch) {
    switch (arch) {
        case Architecture::X86_64: return "x86-64 (AMD64 / Intel 64)";
        case Architecture::AArch64: return "AArch64 (ARM 64-bit)";
        case Architecture::IA32: return "IA-32 (x86 32-bit)";
        case Architecture::ARM32: return "ARM 32-bit (Thumb-2)";
        case Architecture::RiscV64: return "RISC-V 64-bit";
    }
    return "Unknown";
}

/**
 * @brief Returns canonical UEFI fallback filename per UEFI Spec Section 3.5.1
 */
inline std::string get_uefi_fallback_filename(Architecture arch) {
    switch (arch) {
        case Architecture::X86_64: return "BOOTX64.EFI";
        case Architecture::AArch64: return "BOOTAA64.EFI";
        case Architecture::IA32: return "BOOTIA32.EFI";
        case Architecture::ARM32: return "BOOTARM.EFI";
        case Architecture::RiscV64: return "BOOTRISCV64.EFI";
    }
    return "BOOT.EFI";
}

inline std::string get_uefi_fallback_path(Architecture arch) {
    return "\\EFI\\BOOT\\" + get_uefi_fallback_filename(arch);
}

/**
 * @brief CPU State immediately after Power-On Reset (Intel SDM Vol 3A, Chap 9.1)
 */
struct CpuResetState {
    uint32_t rip;          // 0x0000FFF0 (16-bit IP)
    uint16_t cs_selector;  // 0xF000
    uint64_t cs_base;      // 0xFFFF0000 (Hidden base in descriptor cache)
    uint32_t cr0;          // 0x60000010 (CD=1, NW=1, PE=0: Real Mode)
    uint32_t cr4;          // 0x00000000
    uint32_t eflags;       // 0x00000002 (Reserved bit 1 set)
    uint32_t edx_bist;     // Built-In Self Test result & CPU model

    [[nodiscard]] uint64_t get_physical_reset_vector() const noexcept {
        return cs_base + rip; // 0xFFFF0000 + 0xFFF0 = 0xFFFFFFF0 (4GiB - 16 bytes)
    }

    [[nodiscard]] std::string format_details() const {
        std::ostringstream oss;
        oss << "=== x86-64 CPU Power-On Reset State (Intel SDM Vol 3A 9.1) ===\n";
        oss << "  Initial Physical Address (Reset Vector): 0x" 
            << std::hex << std::uppercase << get_physical_reset_vector() << " (4GiB - 16 bytes)\n";
        oss << "  CS:IP Selector / Offset: 0x" << std::hex << cs_selector << ":0x" << rip << "\n";
        oss << "  CS Hidden Base Address:  0x" << std::hex << cs_base << "\n";
        oss << "  CR0 Register:            0x" << std::hex << cr0 
            << " (PE=" << (cr0 & 1) << " RealMode, CD=" << ((cr0 >> 30) & 1) << " CacheDisable)\n";
        oss << "  EFLAGS Register:         0x" << std::hex << eflags << "\n";
        oss << "  EDX BIST Result:         0x" << std::hex << edx_bist 
            << ((edx_bist == 0) ? " (BIST PASSED)" : " (BIST Signature)") << "\n";
        return oss.str();
    }
};

// -----------------------------------------------------------------------------
// 2. UEFI Platform Initialization (PI) 5 Phases
// -----------------------------------------------------------------------------

enum class BootPhase {
    SEC,  // Security Phase
    PEI,  // Pre-EFI Initialization
    DXE,  // Driver Execution Environment
    BDS,  // Boot Device Selection
    TSL,  // Transient System Load (UEFI App / OS Loader)
    RT,   // Run Time
    AL    // After Life
};

inline const char* boot_phase_to_string(BootPhase phase) {
    switch (phase) {
        case BootPhase::SEC: return "SEC (Security Phase)";
        case BootPhase::PEI: return "PEI (Pre-EFI Initialization Phase)";
        case BootPhase::DXE: return "DXE (Driver Execution Environment Phase)";
        case BootPhase::BDS: return "BDS (Boot Device Selection Phase)";
        case BootPhase::TSL: return "TSL (Transient System Load Phase)";
        case BootPhase::RT:  return "RT (Run Time Phase)";
        case BootPhase::AL:  return "AL (After Life Phase)";
    }
    return "Unknown";
}

struct PhaseInfo {
    BootPhase phase;
    CpuMode cpu_mode;
    std::string memory_backing;
    std::string primary_role;
    std::vector<std::string> major_events;
};

inline std::vector<PhaseInfo> get_uefi_phases_info() {
    return {
        {
            BootPhase::SEC,
            CpuMode::ProtectedMode32,
            "Cache-as-RAM (CAR / Temporary RAM in L1/L2 Cache)",
            "Hardware root of trust, CAR init, transition out of 16-bit Reset Vector",
            {
                "CPU executes at Reset Vector 0xFFFFFFF0",
                "Switches to 32-bit Protected Mode",
                "Configures MSRs to enable Cache-as-RAM (CAR)",
                "Hands off control and PPI table pointer to PEI Foundation"
            }
        },
        {
            BootPhase::PEI,
            CpuMode::ProtectedMode32,
            "CAR -> Main DRAM (after Memory Controller Init)",
            "Initialize CPU, Chipset, Memory Controller, Discover DRAM size",
            {
                "Execute PEIMs (PEI Modules)",
                "Initialize Memory Controller (DRAM Timing, Calibration)",
                "Migrate temporary stack/heap from CAR to physical DRAM",
                "Build HOB (Hand-Off Block) list describing system memory and hardware state",
                "Pass HOB list to DXE Foundation"
            }
        },
        {
            BootPhase::DXE,
            CpuMode::LongMode64,
            "Physical DRAM (4-Level Identity Paging enabled, CR3 loaded)",
            "Switch to 64-bit Long Mode, load device drivers, produce UEFI Protocols",
            {
                "Build initial 4-level page tables (PML4 -> PDP -> PD -> PT)",
                "Enable IA32_EFER.LME, CR4.PAE, and CR0.PG to enter 64-bit Long Mode",
                "Initialize UEFI Core Services (BootServices, RuntimeServices)",
                "Load architectural protocols: GOP (Graphics), PCI Root, xHCI/USB, FAT/Disk",
                "Construct Protocol Database and Device Path tree"
            }
        },
        {
            BootPhase::BDS,
            CpuMode::LongMode64,
            "Physical DRAM (UEFI Boot Services available)",
            "Execute Boot Policy, parse NVRAM BootOrder, resolve Boot Device",
            {
                "Connect drivers to console devices (ConIn, ConOut, ConErr)",
                "Read NVRAM variables (BootOrder, Boot0000, Boot0001, etc.)",
                "Probe Removable Storage (USB, Virtual VFAT) if NVRAM target fails",
                "Locate \\EFI\\BOOT\\BOOTX64.EFI on FAT volume (Fallback boot path)",
                "Load image into memory via gBS->LoadImage()",
                "Transfer control via gBS->StartImage()"
            }
        },
        {
            BootPhase::TSL,
            CpuMode::LongMode64,
            "Physical DRAM (EFI System Table & BootServices active)",
            "Execute OS Loader / UEFI Application (MikanOS BOOTX64.EFI)",
            {
                "Invoke EfiMain(ImageHandle, SystemTable)",
                "Pass RCX=ImageHandle, RDX=SystemTable (Microsoft x64 ABI)",
                "Application draws to screen via ConOut->OutputString() or GOP",
                "Optional: Call ExitBootServices() to terminate UEFI and take full control"
            }
        }
    };
}

// -----------------------------------------------------------------------------
// 3. UEFI Boot Manager & Device Resolution Simulation
// -----------------------------------------------------------------------------

struct FileNode {
    std::string path; // Normalized path e.g. "EFI/BOOT/BOOTX64.EFI"
    std::vector<uint8_t> data;
    size_t size;
};

struct StoragePartition {
    std::string label;
    std::string partition_type; // "GPT ESP", "MBR FAT", "Raw"
    std::string fs_type;        // "FAT32", "FAT16", "FAT12", "NTFS", "EXT4"
    bool is_removable;          // true for USB stick / QEMU virtual FAT
    std::map<std::string, FileNode> files;

    [[nodiscard]] bool has_file(const std::string& normalized_path) const {
        return files.find(normalized_path) != files.end();
    }
};

struct BootOrderEntry {
    uint16_t boot_index; // e.g. 0x0001 for "Boot0001"
    std::string description;
    std::string target_device_label;
    std::string target_file_path; // e.g. "\\EFI\\fedora\\grubx64.efi"
    bool is_enabled;
};

struct BootResolutionResult {
    bool success;
    std::string selected_device_label;
    std::string resolved_file_path;
    std::string resolution_method; // "NVRAM BootOrder" or "Removable Fallback"
    std::vector<std::string> log_trace;
};

class UefiBootManagerSimulator {
public:
    explicit UefiBootManagerSimulator(Architecture arch = Architecture::X86_64)
        : target_arch_(arch) {}

    void add_partition(const StoragePartition& part) {
        partitions_.push_back(part);
    }

    void add_boot_order_entry(const BootOrderEntry& entry) {
        boot_order_.push_back(entry);
    }

    [[nodiscard]] BootResolutionResult resolve_boot_target() const {
        BootResolutionResult res;
        res.success = false;
        res.log_trace.push_back("BDS Phase: Starting UEFI Boot Target Resolution");

        // Step 1: Scan NVRAM BootOrder entries
        res.log_trace.push_back("Step 1: Inspecting NVRAM BootOrder entries (" + 
                                std::to_string(boot_order_.size()) + " configured)");
        for (const auto& entry : boot_order_) {
            if (!entry.is_enabled) {
                res.log_trace.push_back("  Skipping disabled entry: Boot" + 
                                        format_hex4(entry.boot_index) + " (" + entry.description + ")");
                continue;
            }

            res.log_trace.push_back("  Checking Boot" + format_hex4(entry.boot_index) + 
                                    ": " + entry.description + " -> " + entry.target_file_path);

            // Find matching partition
            for (const auto& part : partitions_) {
                if (part.label == entry.target_device_label) {
                    std::string norm = normalize_path(entry.target_file_path);
                    if (is_uefi_compliant_fs(part.fs_type) && part.has_file(norm)) {
                        res.success = true;
                        res.selected_device_label = part.label;
                        res.resolved_file_path = entry.target_file_path;
                        res.resolution_method = "NVRAM BootOrder (Boot" + format_hex4(entry.boot_index) + ")";
                        res.log_trace.push_back("  --> MATCH FOUND on partition [" + part.label + "]");
                        return res;
                    } else {
                        res.log_trace.push_back("  Target file not found on partition [" + part.label + "]");
                    }
                }
            }
        }

        // Step 2: Removable Media Fallback (UEFI Spec 3.5.1.1)
        res.log_trace.push_back("Step 2: No valid NVRAM entry booted. Initiating Removable Media Fallback");
        std::string expected_fallback = get_uefi_fallback_path(target_arch_);
        std::string expected_norm = normalize_path(expected_fallback);

        res.log_trace.push_back("  Target Architecture: " + std::string(arch_to_string(target_arch_)));
        res.log_trace.push_back("  Standard Fallback Path: " + expected_fallback);

        for (const auto& part : partitions_) {
            res.log_trace.push_back("  Probing partition [" + part.label + "] (Type: " + 
                                    part.partition_type + ", FS: " + part.fs_type + 
                                    ", Removable: " + (part.is_removable ? "Yes" : "No") + ")");

            if (!is_uefi_compliant_fs(part.fs_type)) {
                res.log_trace.push_back("    Ignored: File system is not UEFI compatible (FAT12/16/32 required)");
                continue;
            }

            if (part.has_file(expected_norm)) {
                res.success = true;
                res.selected_device_label = part.label;
                res.resolved_file_path = expected_fallback;
                res.resolution_method = "Removable Media Fallback (" + expected_fallback + ")";
                res.log_trace.push_back("    --> SUCCESS: Located fallback binary " + expected_fallback);
                return res;
            } else {
                res.log_trace.push_back("    Fallback file " + expected_fallback + " not present");
            }
        }

        res.log_trace.push_back("BDS Phase Error: No bootable device or fallback binary located!");
        return res;
    }

private:
    Architecture target_arch_;
    std::vector<StoragePartition> partitions_;
    std::vector<BootOrderEntry> boot_order_;

    static bool is_uefi_compliant_fs(const std::string& fs) {
        return fs == "FAT32" || fs == "FAT16" || fs == "FAT12";
    }

    static std::string normalize_path(std::string p) {
        std::replace(p.begin(), p.end(), '\\', '/');
        while (!p.empty() && p.front() == '/') {
            p.erase(p.begin());
        }
        std::transform(p.begin(), p.end(), p.begin(), ::toupper);
        return p;
    }

    static std::string format_hex4(uint16_t v) {
        std::ostringstream oss;
        oss << std::hex << std::uppercase << std::setfill('0') << std::setw(4) << v;
        return oss.str();
    }
};

// -----------------------------------------------------------------------------
// 4. Microsoft x64 Calling Convention & UEFI Entry Verification
// -----------------------------------------------------------------------------

struct UefiCallingConvention {
    uint64_t rcx_image_handle;   // Arg 1: EFI_HANDLE ImageHandle
    uint64_t rdx_system_table;   // Arg 2: EFI_SYSTEM_TABLE* SystemTable
    uint64_t r8;                 // Arg 3: (Unused in EfiMain)
    uint64_t r9;                 // Arg 4: (Unused in EfiMain)
    uint64_t rsp;                // Stack Pointer
    uint32_t shadow_space_bytes; // 32 bytes (4 * 8-byte slots)

    [[nodiscard]] bool is_stack_aligned_before_call() const noexcept {
        // According to Microsoft x64 ABI / System V:
        // Before CALL instruction, RSP % 16 must equal 0.
        // Immediately after CALL pushes 8-byte return address, RSP % 16 == 8.
        return (rsp % 16) == 0;
    }

    [[nodiscard]] bool is_shadow_space_valid() const noexcept {
        return shadow_space_bytes >= 32;
    }

    [[nodiscard]] std::string format_calling_state() const {
        std::ostringstream oss;
        oss << "=== UEFI EfiMain Calling Convention (Microsoft x64 ABI) ===\n";
        oss << "  RCX (Arg 1: ImageHandle):  0x" << std::hex << rcx_image_handle << "\n";
        oss << "  RDX (Arg 2: SystemTable):  0x" << std::hex << rdx_system_table << "\n";
        oss << "  RSP (Stack Pointer):       0x" << std::hex << rsp 
            << " (16-byte Aligned: " << (is_stack_aligned_before_call() ? "YES" : "NO") << ")\n";
        oss << "  Shadow Space Allocated:    " << std::dec << shadow_space_bytes << " bytes "
            << (is_shadow_space_valid() ? "[VALID]" : "[INVALID]") << "\n";
        return oss.str();
    }
};

// -----------------------------------------------------------------------------
// 5. UTF-16 OutputString Simulation
// -----------------------------------------------------------------------------

struct MockConOut {
    std::vector<std::string> output_lines;

    void output_string_utf16(const uint16_t* utf16_str) {
        if (!utf16_str) return;
        std::string utf8;
        for (size_t i = 0; utf16_str[i] != 0; ++i) {
            uint16_t code = utf16_str[i];
            if (code < 0x80) {
                utf8.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
                utf8.push_back(static_cast<char>(0xC0 | (code >> 6)));
                utf8.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
                utf8.push_back(static_cast<char>(0xE0 | (code >> 12)));
                utf8.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                utf8.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
        }
        output_lines.push_back(utf8);
    }
};

// -----------------------------------------------------------------------------
// 6. Comprehensive Self-Test
// -----------------------------------------------------------------------------

inline void run_self_tests() {
    // Test 1: Reset Vector math
    CpuResetState reset_state{
        .rip = 0xFFF0,
        .cs_selector = 0xF000,
        .cs_base = 0xFFFF0000,
        .cr0 = 0x60000010,
        .cr4 = 0,
        .eflags = 0x00000002,
        .edx_bist = 0
    };
    assert(reset_state.get_physical_reset_vector() == 0xFFFFFFF0ULL);
    assert((0x100000000ULL - reset_state.get_physical_reset_vector()) == 16);

    // Test 2: Fallback filenames
    assert(get_uefi_fallback_filename(Architecture::X86_64) == "BOOTX64.EFI");
    assert(get_uefi_fallback_filename(Architecture::AArch64) == "BOOTAA64.EFI");
    assert(get_uefi_fallback_filename(Architecture::IA32) == "BOOTIA32.EFI");
    assert(get_uefi_fallback_path(Architecture::X86_64) == "\\EFI\\BOOT\\BOOTX64.EFI");

    // Test 3: Boot Manager resolution with removable fallback
    UefiBootManagerSimulator bm(Architecture::X86_64);

    StoragePartition usb_part{
        .label = "USB_MIKANOS",
        .partition_type = "MBR FAT",
        .fs_type = "FAT32",
        .is_removable = true,
        .files = {
            {"EFI/BOOT/BOOTX64.EFI", FileNode{.path = "EFI/BOOT/BOOTX64.EFI", .data = {}, .size = 1024}}
        }
    };
    bm.add_partition(usb_part);

    auto result = bm.resolve_boot_target();
    assert(result.success);
    assert(result.selected_device_label == "USB_MIKANOS");
    assert(result.resolved_file_path == "\\EFI\\BOOT\\BOOTX64.EFI");

    // Test 4: Microsoft x64 ABI validation
    UefiCallingConvention abi{
        .rcx_image_handle = 0x10000000,
        .rdx_system_table = 0x20000000,
        .r8 = 0,
        .r9 = 0,
        .rsp = 0x7FFFFFFF0000, // Aligned to 16
        .shadow_space_bytes = 32
    };
    assert(abi.is_stack_aligned_before_call());
    assert(abi.is_shadow_space_valid());

    // Test 5: Mock ConOut UTF-16 decoding
    MockConOut conout;
    uint16_t sample_utf16[] = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\r', '\n', 0};
    conout.output_string_utf16(sample_utf16);
    assert(conout.output_lines.size() == 1);
    assert(conout.output_lines[0] == "Hello, world!\r\n");

    std::cout << "All C++20 Boot Sequence Tracer Self-Tests Passed Successfully!" << std::endl;
}

} // namespace mikanos

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    bool run_tests = false;
    bool show_trace = false;
    bool show_phases = false;

    if (argc == 1) {
        // Default: display full trace & phases
        show_trace = true;
        show_phases = true;
    } else {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--test" || arg == "-t") run_tests = true;
            if (arg == "--trace") show_trace = true;
            if (arg == "--phases") show_phases = true;
        }
    }

    if (run_tests) {
        mikanos::run_self_tests();
        return 0;
    }

    std::cout << "=================================================================\n";
    std::cout << "🍊 MikanOS Chapter 1.5: Behind the Scenes Boot Sequence Tracer\n";
    std::cout << "=================================================================\n\n";

    if (show_phases) {
        std::cout << ">>> 1. Hardware Reset State <<<\n";
        mikanos::CpuResetState reset_state{
            .rip = 0xFFF0,
            .cs_selector = 0xF000,
            .cs_base = 0xFFFF0000,
            .cr0 = 0x60000010,
            .cr4 = 0,
            .eflags = 0x00000002,
            .edx_bist = 0
        };
        std::cout << reset_state.format_details() << "\n";

        std::cout << ">>> 2. UEFI Platform Initialization (PI) 5-Phase Architecture <<<\n";
        auto phases = mikanos::get_uefi_phases_info();
        for (const auto& p : phases) {
            std::cout << "-----------------------------------------------------------------\n";
            std::cout << "Phase: " << mikanos::boot_phase_to_string(p.phase) << "\n";
            std::cout << "  CPU Mode: " << mikanos::cpu_mode_to_string(p.cpu_mode) << "\n";
            std::cout << "  Memory:   " << p.memory_backing << "\n";
            std::cout << "  Role:     " << p.primary_role << "\n";
            std::cout << "  Key Events:\n";
            for (const auto& ev : p.major_events) {
                std::cout << "    * " << ev << "\n";
            }
        }
        std::cout << "\n";
    }

    if (show_trace) {
        std::cout << ">>> 3. UEFI Boot Manager Fallback Resolution Simulation <<<\n";
        mikanos::UefiBootManagerSimulator bm(mikanos::Architecture::X86_64);

        // Scenario: Internal NVMe drive with missing OS, and USB Stick with MikanOS
        mikanos::StoragePartition internal_nvme{
            .label = "NVMe_ESP",
            .partition_type = "GPT ESP",
            .fs_type = "FAT32",
            .is_removable = false,
            .files = {} // No bootloader
        };
        mikanos::BootOrderEntry nvram_entry{
            .boot_index = 0x0001,
            .description = "Ubuntu GRUB",
            .target_device_label = "NVMe_ESP",
            .target_file_path = "\\EFI\\ubuntu\\shimx64.efi",
            .is_enabled = true
        };
        bm.add_partition(internal_nvme);
        bm.add_boot_order_entry(nvram_entry);

        mikanos::StoragePartition usb_stick{
            .label = "MIKANOS_USB",
            .partition_type = "MBR FAT",
            .fs_type = "FAT32",
            .is_removable = true,
            .files = {
                {"EFI/BOOT/BOOTX64.EFI", mikanos::FileNode{.path = "EFI/BOOT/BOOTX64.EFI", .data = {}, .size = 1440}}
            }
        };
        bm.add_partition(usb_stick);

        auto result = bm.resolve_boot_target();
        for (const auto& line : result.log_trace) {
            std::cout << line << "\n";
        }
        std::cout << "\nResult Summary:\n";
        std::cout << "  Resolution Success: " << (result.success ? "YES" : "NO") << "\n";
        std::cout << "  Method:             " << result.resolution_method << "\n";
        std::cout << "  Boot Device:        [" << result.selected_device_label << "]\n";
        std::cout << "  Loaded File:        " << result.resolved_file_path << "\n\n";

        std::cout << ">>> 4. EfiMain Calling Convention (Microsoft x64 ABI) <<<\n";
        mikanos::UefiCallingConvention abi{
            .rcx_image_handle = 0x7E120000,
            .rdx_system_table = 0x7E200000,
            .r8 = 0,
            .r9 = 0,
            .rsp = 0x7FFF0000,
            .shadow_space_bytes = 32
        };
        std::cout << abi.format_calling_state() << "\n";

        std::cout << ">>> 5. Mock UEFI ConOut Output <<<\n";
        mikanos::MockConOut conout;
        uint16_t msg[] = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\r', '\n', 0};
        conout.output_string_utf16(msg);
        for (const auto& line : conout.output_lines) {
            std::cout << "  [ConOut Display]: " << line;
        }
    }

    return 0;
}
