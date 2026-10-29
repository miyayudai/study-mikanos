/**
 * @file qemu_launch_builder.cpp
 * @brief C++20 Host-side QEMU Launch Configuration Builder & Virtual Hardware Validator
 * 
 * Part of study-mikanos: Chapter 1.4 "Running on Emulator (QEMU & OVMF)"
 * Demonstrates:
 *   1. Dynamic construction and validation of qemu-system-x86_64 arguments
 *   2. OVMF UEFI firmware mapping modes (PFLASH vs BIOS direct)
 *   3. Virtual FAT32 BPB and Directory Entry packed binary structures
 *   4. Command-line generation and options inspection
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

namespace mikanos {

// -----------------------------------------------------------------------------
// Packed Binary Structures for FAT32 / UEFI ESP
// -----------------------------------------------------------------------------
#pragma pack(push, 1)

/**
 * @brief BIOS Parameter Block (BPB) and FAT32 Extended Boot Record
 */
struct Fat32Bpb {
    uint8_t  jmp_boot[3];        // 0xEB, 0x58, 0x90
    char     oem_name[8];        // "MSDOS5.0"
    uint16_t bytes_per_sector;   // 512
    uint8_t  sectors_per_cluster;// 8 (4096 bytes / cluster)
    uint16_t reserved_sectors;   // Typically 32
    uint8_t  num_fats;           // 2
    uint16_t root_entry_count;   // 0 for FAT32
    uint16_t total_sectors_16;   // 0 for FAT32
    uint8_t  media_type;         // 0xF8 (Fixed disk)
    uint16_t fat_size_16;        // 0 for FAT32
    uint16_t sectors_per_track;  // e.g. 63
    uint16_t num_heads;          // e.g. 255
    uint32_t hidden_sectors;     // e.g. 0
    uint32_t total_sectors_32;   // Total sectors in volume

    // FAT32 Extended Fields
    uint32_t sectors_per_fat32;  // Sectors occupied by one FAT
    uint16_t ext_flags;          // Active FAT mirroring
    uint16_t fs_version;         // 0x0000
    uint32_t root_cluster;       // Typically 2
    uint16_t fs_info_sector;     // 1
    uint16_t backup_boot_sector; // 6
    uint8_t  reserved[12];
    uint8_t  drive_number;       // 0x80
    uint8_t  reserved1;
    uint8_t  boot_signature;     // 0x29
    uint32_t volume_id;
    char     volume_label[11];   // "NO NAME    "
    char     fs_type[8];         // "FAT32   "
};
static_assert(sizeof(Fat32Bpb) == 90, "Fat32Bpb must be exactly 90 bytes");

/**
 * @brief Standard 32-byte FAT Directory Entry (Short Directory Entry)
 */
struct FatDirectoryEntry {
    char     name[11];           // 8.3 filename ("BOOTX64 EFI")
    uint8_t  attributes;         // 0x20: Archive, 0x10: Subdirectory, etc.
    uint8_t  nt_reserved;
    uint8_t  creation_time_tenth;
    uint16_t creation_time;
    uint16_t creation_date;
    uint16_t last_access_date;
    uint16_t first_cluster_high; // High 16 bits of start cluster
    uint16_t write_time;
    uint16_t write_date;
    uint16_t first_cluster_low;  // Low 16 bits of start cluster
    uint32_t file_size;          // File size in bytes
};
static_assert(sizeof(FatDirectoryEntry) == 32, "FatDirectoryEntry must be exactly 32 bytes");

#pragma pack(pop)

// -----------------------------------------------------------------------------
// QEMU Configuration Enums & Builder Classes
// -----------------------------------------------------------------------------

enum class FirmwareMode {
    BiosDirect,     // -bios OVMF.fd
    PflashSplit,    // -drive if=pflash (CODE read-only + VARS rw)
    PflashUnified,  // -drive if=pflash (single file)
};

enum class DriveFormat {
    VfatDirectory,  // -drive format=raw,file=fat:rw:<dir>
    RawImage,       // -drive format=raw,file=<path>
    Qcow2Image,     // -drive format=qcow2,file=<path>
};

struct QemuLaunchOptions {
    int memory_mb = 1024;
    int smp_cores = 1;
    std::string cpu_model = "qemu64";
    bool enable_kvm = false;
    bool enable_xhci = true;
    bool enable_usb_mouse = false;
    std::string monitor = "stdio";
    std::string display = "default";
    std::string net = "none";
    bool gdb_debug = false;
    int gdb_port = 1234;

    FirmwareMode firmware_mode = FirmwareMode::PflashSplit;
    std::string ovmf_code_path = "/usr/share/OVMF/OVMF_CODE.fd";
    std::string ovmf_vars_path = "/usr/share/OVMF/OVMF_VARS.fd";

    DriveFormat drive_format = DriveFormat::VfatDirectory;
    std::string disk_target = "disk_img";

    std::vector<std::string> extra_args;
};

class QemuLaunchBuilder {
public:
    explicit QemuLaunchBuilder(QemuLaunchOptions opts = {})
        : opts_(std::move(opts)) {}

    [[nodiscard]] std::vector<std::string> BuildArgs() const {
        std::vector<std::string> args;
        args.emplace_back("qemu-system-x86_64");

        // 1. RAM & CPU
        args.emplace_back("-m");
        args.emplace_back(std::to_string(opts_.memory_mb) + "M");

        if (opts_.smp_cores > 1) {
            args.emplace_back("-smp");
            args.emplace_back(std::to_string(opts_.smp_cores));
        }

        args.emplace_back("-cpu");
        args.emplace_back(opts_.cpu_model);

        if (opts_.enable_kvm) {
            args.emplace_back("-enable-kvm");
        }

        // 2. Firmware (OVMF)
        if (opts_.firmware_mode == FirmwareMode::BiosDirect) {
            args.emplace_back("-bios");
            args.emplace_back(opts_.ovmf_code_path);
        } else if (opts_.firmware_mode == FirmwareMode::PflashSplit) {
            args.emplace_back("-drive");
            args.emplace_back("if=pflash,format=raw,readonly=on,file=" + opts_.ovmf_code_path);
            if (!opts_.ovmf_vars_path.empty()) {
                args.emplace_back("-drive");
                args.emplace_back("if=pflash,format=raw,file=" + opts_.ovmf_vars_path);
            }
        } else if (opts_.firmware_mode == FirmwareMode::PflashUnified) {
            args.emplace_back("-drive");
            args.emplace_back("if=pflash,format=raw,file=" + opts_.ovmf_code_path);
        }

        // 3. Storage Drive
        if (opts_.drive_format == DriveFormat::VfatDirectory) {
            args.emplace_back("-drive");
            args.emplace_back("format=raw,file=fat:rw:" + opts_.disk_target);
        } else if (opts_.drive_format == DriveFormat::RawImage) {
            args.emplace_back("-drive");
            args.emplace_back("format=raw,file=" + opts_.disk_target);
        } else if (opts_.drive_format == DriveFormat::Qcow2Image) {
            args.emplace_back("-drive");
            args.emplace_back("format=qcow2,file=" + opts_.disk_target);
        }

        // 4. USB 3.0 (xHCI)
        if (opts_.enable_xhci) {
            args.emplace_back("-device");
            args.emplace_back("nec-usb-xhci,id=xhci");
            if (opts_.enable_usb_mouse) {
                args.emplace_back("-device");
                args.emplace_back("usb-mouse,bus=xhci.0");
            }
        }

        // 5. Monitor & Display
        if (!opts_.monitor.empty()) {
            args.emplace_back("-monitor");
            args.emplace_back(opts_.monitor);
        }
        if (opts_.display != "default") {
            args.emplace_back("-display");
            args.emplace_back(opts_.display);
        }

        // 6. Network
        if (opts_.net == "none") {
            args.emplace_back("-net");
            args.emplace_back("none");
        }

        // 7. GDB Debug
        if (opts_.gdb_debug) {
            args.emplace_back("-gdb");
            args.emplace_back("tcp::" + std::to_string(opts_.gdb_port));
            args.emplace_back("-S");
        }

        // 8. Extra user args
        for (const auto& arg : opts_.extra_args) {
            args.push_back(arg);
        }

        return args;
    }

    [[nodiscard]] std::string BuildCommandString() const {
        auto args = BuildArgs();
        std::ostringstream oss;
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) oss << " ";
            oss << args[i];
        }
        return oss.str();
    }

    [[nodiscard]] std::vector<std::string> Validate() const {
        std::vector<std::string> errors;
        if (opts_.memory_mb < 128) {
            errors.push_back("Memory must be at least 128MB for UEFI OVMF");
        }
        if (opts_.ovmf_code_path.empty()) {
            errors.push_back("OVMF code path cannot be empty");
        }
        if (opts_.firmware_mode == FirmwareMode::PflashSplit && opts_.ovmf_vars_path.empty()) {
            errors.push_back("PflashSplit mode requires non-empty OVMF vars path");
        }
        if (opts_.disk_target.empty()) {
            errors.push_back("Disk target cannot be empty");
        }
        return errors;
    }

    const QemuLaunchOptions& Options() const { return opts_; }
    QemuLaunchOptions& MutableOptions() { return opts_; }

private:
    QemuLaunchOptions opts_;
};

} // namespace mikanos

// -----------------------------------------------------------------------------
// Standalone Verification & Demonstration
// -----------------------------------------------------------------------------
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "  MikanOS C++20 QEMU Launch Builder & FAT32 Inspector   \n";
    std::cout << "========================================================\n\n";

    // 1. Verify Packed FAT32 Binary Structure Layouts
    std::cout << "[Step 1] Validating FAT32 BPB & Directory Entry Structures...\n";
    mikanos::Fat32Bpb bpb{};
    std::memset(&bpb, 0, sizeof(bpb));
    bpb.bytes_per_sector = 512;
    bpb.sectors_per_cluster = 8;
    bpb.reserved_sectors = 32;
    bpb.num_fats = 2;
    bpb.root_cluster = 2;
    std::memcpy(bpb.fs_type, "FAT32   ", 8);

    assert(sizeof(mikanos::Fat32Bpb) == 90);
    assert(bpb.bytes_per_sector * bpb.sectors_per_cluster == 4096);
    std::cout << "  - Fat32Bpb size: " << sizeof(mikanos::Fat32Bpb) << " bytes (OK)\n";
    std::cout << "  - Cluster size : " << (bpb.bytes_per_sector * bpb.sectors_per_cluster) << " bytes (OK)\n";

    mikanos::FatDirectoryEntry entry{};
    std::memset(&entry, 0, sizeof(entry));
    std::memcpy(entry.name, "BOOTX64 EFI", 11);
    entry.attributes = 0x20; // Archive file
    entry.first_cluster_high = 0x0000;
    entry.first_cluster_low = 0x0004;
    entry.file_size = 1488; // e.g. hello.efi size

    uint32_t start_cluster = (static_cast<uint32_t>(entry.first_cluster_high) << 16) | entry.first_cluster_low;
    assert(sizeof(mikanos::FatDirectoryEntry) == 32);
    assert(start_cluster == 4);
    std::cout << "  - FatDirectoryEntry size: " << sizeof(mikanos::FatDirectoryEntry) << " bytes (OK)\n";
    std::cout << "  - Entry name   : " << std::string(entry.name, 11) << "\n";
    std::cout << "  - Start cluster: " << start_cluster << " (OK)\n\n";

    // 2. Test QEMU Command Construction with Default MikanOS Settings
    std::cout << "[Step 2] Testing QEMU Command Construction (MikanOS Standard)...\n";
    mikanos::QemuLaunchOptions default_opts;
    default_opts.memory_mb = 1024;
    default_opts.firmware_mode = mikanos::FirmwareMode::PflashSplit;
    default_opts.ovmf_code_path = "/usr/share/OVMF/OVMF_CODE.fd";
    default_opts.ovmf_vars_path = "/usr/share/OVMF/OVMF_VARS.fd";
    default_opts.drive_format = mikanos::DriveFormat::VfatDirectory;
    default_opts.disk_target = "disk_img";
    default_opts.enable_xhci = true;
    default_opts.monitor = "stdio";

    mikanos::QemuLaunchBuilder builder(default_opts);
    auto errors = builder.Validate();
    assert(errors.empty());

    auto args = builder.BuildArgs();
    std::cout << "  - Total arguments: " << args.size() << "\n";
    std::cout << "  - Command line:\n    " << builder.BuildCommandString() << "\n\n";

    // Verify critical arguments are present
    auto has_arg = [&](const std::string& key) {
        return std::find(args.begin(), args.end(), key) != args.end();
    };
    assert(has_arg("-m"));
    assert(has_arg("1024M"));
    assert(has_arg("-monitor"));
    assert(has_arg("stdio"));
    assert(has_arg("-device"));
    assert(has_arg("nec-usb-xhci,id=xhci"));
    assert(has_arg("-net"));
    assert(has_arg("none"));

    // 3. Test BIOS Direct Mode (-bios)
    std::cout << "[Step 3] Testing -bios direct loading mode...\n";
    mikanos::QemuLaunchOptions bios_opts = default_opts;
    bios_opts.firmware_mode = mikanos::FirmwareMode::BiosDirect;
    bios_opts.ovmf_code_path = "/path/to/OVMF.fd";

    mikanos::QemuLaunchBuilder bios_builder(bios_opts);
    auto bios_args = bios_builder.BuildArgs();
    auto it_bios = std::find(bios_args.begin(), bios_args.end(), "-bios");
    assert(it_bios != bios_args.end());
    assert(*(it_bios + 1) == "/path/to/OVMF.fd");
    std::cout << "  - Command line:\n    " << bios_builder.BuildCommandString() << "\n\n";

    // 4. Test GDB Debug Mode
    std::cout << "[Step 4] Testing GDB remote debugging mode (-gdb tcp::1234 -S)...\n";
    mikanos::QemuLaunchOptions gdb_opts = default_opts;
    gdb_opts.gdb_debug = true;
    gdb_opts.gdb_port = 1234;

    mikanos::QemuLaunchBuilder gdb_builder(gdb_opts);
    auto gdb_args = gdb_builder.BuildArgs();
    auto it_gdb = std::find(gdb_args.begin(), gdb_args.end(), "-gdb");
    assert(it_gdb != gdb_args.end());
    assert(*(it_gdb + 1) == "tcp::1234");
    assert(std::find(gdb_args.begin(), gdb_args.end(), "-S") != gdb_args.end());
    std::cout << "  - Command line:\n    " << gdb_builder.BuildCommandString() << "\n\n";

    // 5. Test Parameter Validation
    std::cout << "[Step 5] Testing configuration validation rules...\n";
    mikanos::QemuLaunchOptions invalid_opts;
    invalid_opts.memory_mb = 64; // Too small
    invalid_opts.ovmf_code_path = ""; // Empty
    invalid_opts.firmware_mode = mikanos::FirmwareMode::PflashSplit;
    invalid_opts.ovmf_vars_path = ""; // Missing vars in split mode
    invalid_opts.disk_target = "";

    mikanos::QemuLaunchBuilder invalid_builder(invalid_opts);
    auto val_errors = invalid_builder.Validate();
    std::cout << "  - Expected validation errors caught: " << val_errors.size() << "\n";
    for (const auto& err : val_errors) {
        std::cout << "    * " << err << "\n";
    }
    assert(val_errors.size() >= 3);

    std::cout << "\n✅ All C++20 QEMU Launch Builder & FAT32 tests passed successfully!\n";
    return 0;
}
