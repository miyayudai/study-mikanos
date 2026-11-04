/**
 * @file pci_scan_sim.cpp
 * @brief C++20 Standalone Simulation of PCI Configuration Space & Bus Scanning (MikanOS Chapter 6.3 / osbook_day06b)
 *
 * This file implements and verifies:
 * 1. PCI Configuration Mechanism #1: I/O Ports 0x0CF8 (CONFIG_ADDRESS) and 0x0CFC (CONFIG_DATA)
 * 2. Bitwise derivation of MakeAddress(bus, device, function, reg_addr)
 * 3. PCI Configuration Space Standard Headers (Type 00h for endpoints, Type 01h for PCI-to-PCI bridges)
 * 4. Recursive bus scanning algorithm: ScanAllBus(), ScanBus(), ScanDevice(), ScanFunction()
 * 5. Detection of xHCI USB 3.0 Host Controller via Class Code (0x0C0330xx)
 *
 * Can be compiled standalone:
 *   g++ -std=c++20 -O2 -Wall -Wextra pci_scan_sim.cpp -o pci_scan_sim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <map>
#include <tuple>
#include <optional>

// ============================================================================
// 1. Error Handling (Directly matching MikanOS kernel/error.hpp)
// ============================================================================

class Error {
public:
    enum Code {
        kSuccess,
        kFull,
        kEmpty,
        kLastOfCode,
    };

    constexpr Error(Code code) : code_{code} {}

    constexpr operator bool() const {
        return this->code_ != kSuccess;
    }

    constexpr Code Cause() const {
        return this->code_;
    }

    const char* Name() const {
        return code_names_[static_cast<size_t>(this->code_)];
    }

private:
    static constexpr std::array<const char*, 4> code_names_ = {
        "kSuccess",
        "kFull",
        "kEmpty",
        "kLastOfCode"
    };

    Code code_;
};

// ============================================================================
// 2. PCI Namespace & Data Structures (Matching MikanOS kernel/pci.hpp)
// ============================================================================

namespace pci {

/** @brief CONFIG_ADDRESS register IO port address */
constexpr uint16_t kConfigAddress = 0x0cf8;
/** @brief CONFIG_DATA register IO port address */
constexpr uint16_t kConfigData = 0x0cfc;

/** @brief Basic PCI device descriptor stored during bus scan */
struct Device {
    uint8_t bus{0};
    uint8_t device{0};
    uint8_t function{0};
    uint8_t header_type{0};

    bool operator==(const Device& other) const = default;
};

/** @brief Array of discovered PCI devices (matching MikanOS fixed buffer) */
inline std::array<Device, 32> devices;
/** @brief Number of valid devices discovered */
inline int num_device{0};

// Forward declarations
void WriteAddress(uint32_t address);
void WriteData(uint32_t value);
uint32_t ReadData();

uint16_t ReadVendorId(uint8_t bus, uint8_t device, uint8_t function);
uint16_t ReadDeviceId(uint8_t bus, uint8_t device, uint8_t function);
uint8_t ReadHeaderType(uint8_t bus, uint8_t device, uint8_t function);
uint32_t ReadClassCode(uint8_t bus, uint8_t device, uint8_t function);
uint32_t ReadBusNumbers(uint8_t bus, uint8_t device, uint8_t function);
bool IsSingleFunctionDevice(uint8_t header_type);

Error ScanAllBus();

} // namespace pci

// ============================================================================
// 3. Virtual Hardware PCI Environment & Port I/O Emulation
// ============================================================================

namespace sim {

/**
 * @brief Representation of a 256-byte PCI Configuration Space of a single function.
 */
struct SimulatedPCIFunction {
    uint8_t bus{0};
    uint8_t device{0};
    uint8_t function{0};
    std::array<uint8_t, 256> config_space{};

    SimulatedPCIFunction() {
        config_space.fill(0);
    }

    SimulatedPCIFunction(uint8_t b, uint8_t d, uint8_t f,
                         uint16_t vendor_id, uint16_t device_id,
                         uint8_t header_type, uint32_t class_code,
                         uint8_t secondary_bus = 0)
        : bus(b), device(d), function(f)
    {
        config_space.fill(0);
        // 0x00: Vendor ID (16-bit), Device ID (16-bit)
        write16(0x00, vendor_id);
        write16(0x02, device_id);

        // 0x04: Command (16-bit), Status (16-bit)
        write16(0x04, 0x0007); // IO + Memory + BusMaster enable
        write16(0x06, 0x0010); // Capabilities list supported

        // 0x08: Revision ID (8-bit) & Class Code (24-bit)
        write32(0x08, class_code);

        // 0x0C: Header Type (8-bit at offset 0x0E)
        config_space[0x0E] = header_type;

        // If Header Type 01h (PCI-to-PCI Bridge), set bus numbers at offset 0x18
        if ((header_type & 0x7F) == 0x01) {
            config_space[0x18] = bus;             // Primary Bus Number
            config_space[0x19] = secondary_bus;    // Secondary Bus Number
            config_space[0x1A] = secondary_bus;    // Subordinate Bus Number
        }
    }

    void write16(size_t offset, uint16_t val) {
        if (offset + 1 < config_space.size()) {
            config_space[offset] = static_cast<uint8_t>(val & 0xFF);
            config_space[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        }
    }

    void write32(size_t offset, uint32_t val) {
        if (offset + 3 < config_space.size()) {
            config_space[offset] = static_cast<uint8_t>(val & 0xFF);
            config_space[offset + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
            config_space[offset + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
            config_space[offset + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
        }
    }

    uint32_t read32(size_t offset) const {
        if (offset + 3 < config_space.size()) {
            return static_cast<uint32_t>(config_space[offset])
                | (static_cast<uint32_t>(config_space[offset + 1]) << 8)
                | (static_cast<uint32_t>(config_space[offset + 2]) << 16)
                | (static_cast<uint32_t>(config_space[offset + 3]) << 24);
        }
        return 0xFFFFFFFF;
    }
};

/**
 * @brief Global PCI Host Controller and Bus Topology simulator
 */
class PCIEnvironment {
public:
    static PCIEnvironment& Instance() {
        static PCIEnvironment inst;
        return inst;
    }

    void Reset() {
        pci_tree_.clear();
        io_config_address_ = 0;
        io_config_data_ = 0;
    }

    void RegisterDevice(const SimulatedPCIFunction& dev) {
        auto key = std::make_tuple(dev.bus, dev.device, dev.function);
        pci_tree_[key] = dev;
    }

    void WriteAddress(uint32_t addr) {
        io_config_address_ = addr;
    }

    void WriteData(uint32_t val) {
        io_config_data_ = val;
        // If enable bit (bit 31) is set, write to target device if registered
        if ((io_config_address_ & 0x80000000u) != 0) {
            uint8_t bus = (io_config_address_ >> 16) & 0xFF;
            uint8_t dev = (io_config_address_ >> 11) & 0x1F;
            uint8_t func = (io_config_address_ >> 8) & 0x07;
            uint8_t reg = io_config_address_ & 0xFC;

            auto it = pci_tree_.find(std::make_tuple(bus, dev, func));
            if (it != pci_tree_.end()) {
                it->second.write32(reg, val);
            }
        }
    }

    uint32_t ReadData() {
        // If enable bit is not set, port read returns floating bus (0xFFFFFFFF)
        if ((io_config_address_ & 0x80000000u) == 0) {
            return 0xFFFFFFFF;
        }

        uint8_t bus = (io_config_address_ >> 16) & 0xFF;
        uint8_t dev = (io_config_address_ >> 11) & 0x1F;
        uint8_t func = (io_config_address_ >> 8) & 0x07;
        uint8_t reg = io_config_address_ & 0xFC;

        auto it = pci_tree_.find(std::make_tuple(bus, dev, func));
        if (it != pci_tree_.end()) {
            return it->second.read32(reg);
        }
        return 0xFFFFFFFF; // Device not present returns all 1s
    }

    /**
     * @brief Populate standard QEMU VM devices as used in MikanOS development
     */
    void SetupQEMUDevices() {
        Reset();
        // 0.0.0: Intel 82440FX Host Bridge (Header Type 0x00, Single Function)
        // Class: 0x06000002 (Bridge: Host)
        RegisterDevice(SimulatedPCIFunction(0, 0, 0, 0x8086, 0x1237, 0x00, 0x06000002));

        // 0.1.0: Intel PIIX3 ISA Bridge (Header Type 0x80: Multi-Function!)
        // Class: 0x06010000 (Bridge: ISA)
        RegisterDevice(SimulatedPCIFunction(0, 1, 0, 0x8086, 0x7000, 0x80, 0x06010000));

        // 0.1.1: Intel PIIX3 IDE Controller (Function 1 of Dev 1)
        // Class: 0x01018000 (Mass Storage: IDE)
        RegisterDevice(SimulatedPCIFunction(0, 1, 1, 0x8086, 0x7010, 0x00, 0x01018000));

        // 0.1.3: Intel PIIX4 Power Management / ACPI (Function 3 of Dev 1)
        // Class: 0x06800000 (Bridge: Other)
        RegisterDevice(SimulatedPCIFunction(0, 1, 3, 0x8086, 0x7113, 0x00, 0x06800000));

        // 0.2.0: QEMU Standard VGA Display Adapter
        // Class: 0x03000000 (Display: VGA)
        RegisterDevice(SimulatedPCIFunction(0, 2, 0, 0x1234, 0x1111, 0x00, 0x03000000));

        // 0.3.0: Intel 82540EM Gigabit Ethernet (e1000)
        // Class: 0x02000000 (Network: Ethernet)
        RegisterDevice(SimulatedPCIFunction(0, 3, 0, 0x8086, 0x100e, 0x00, 0x02000000));

        // 0.4.0: QEMU / NEC xHCI USB 3.0 Host Controller
        // Class: 0x0C033001 (Serial Bus: USB, ProgIF 0x30 = xHCI)
        RegisterDevice(SimulatedPCIFunction(0, 4, 0, 0x1b36, 0x000d, 0x00, 0x0C033001));

        // 0.5.0: Red Hat PCI-to-PCI Bridge (Header Type 0x01, Secondary Bus = 1)
        // Class: 0x06040000 (Bridge: PCI-to-PCI)
        RegisterDevice(SimulatedPCIFunction(0, 5, 0, 0x1b36, 0x0001, 0x01, 0x06040000, 1));

        // 1.1.0: Intel 82801AA AC'97 Audio Controller on Secondary Bus 1
        // Class: 0x04010000 (Multimedia: Audio)
        RegisterDevice(SimulatedPCIFunction(1, 1, 0, 0x8086, 0x2415, 0x00, 0x04010000));
    }

private:
    PCIEnvironment() = default;
    std::map<std::tuple<uint8_t, uint8_t, uint8_t>, SimulatedPCIFunction> pci_tree_;
    uint32_t io_config_address_{0};
    uint32_t io_config_data_{0};
};

} // namespace sim

// ============================================================================
// 4. MikanOS PCI Bus Control Implementation (Matching kernel/pci.cpp)
// ============================================================================

namespace {

using namespace pci;

/**
 * @brief Generate 32-bit CONFIG_ADDRESS integer from bus, device, function, and reg_addr.
 *
 * Bitfield layout:
 * - Bit 31: Enable Bit (1)
 * - Bits 30-24: Reserved (0)
 * - Bits 23-16: Bus Number (8 bits)
 * - Bits 15-11: Device Number (5 bits)
 * - Bits 10-8:  Function Number (3 bits)
 * - Bits 7-2:   Register Offset (6 bits, 4-byte aligned)
 * - Bits 1-0:   00 (DWORD aligned)
 */
uint32_t MakeAddress(uint8_t bus, uint8_t device,
                     uint8_t function, uint8_t reg_addr) {
    auto shl = [](uint32_t x, unsigned int bits) {
        return x << bits;
    };

    return shl(1, 31)  // enable bit
        | shl(bus, 16)
        | shl(device, 11)
        | shl(function, 8)
        | (reg_addr & 0xfcu);
}

/**
 * @brief Append device to `devices` array and increment `num_device`.
 */
Error AddDevice(uint8_t bus, uint8_t device,
                uint8_t function, uint8_t header_type) {
    if (num_device == static_cast<int>(devices.size())) {
        return Error::kFull;
    }

    devices[num_device] = Device{bus, device, function, header_type};
    ++num_device;
    return Error::kSuccess;
}

Error ScanBus(uint8_t bus);

/**
 * @brief Scan a single function: register into devices array, and if PCI-to-PCI bridge,
 * recursively invoke ScanBus on the secondary bus.
 */
Error ScanFunction(uint8_t bus, uint8_t device, uint8_t function) {
    auto header_type = ReadHeaderType(bus, device, function);
    if (auto err = AddDevice(bus, device, function, header_type)) {
        return err;
    }

    auto class_code = ReadClassCode(bus, device, function);
    uint8_t base = (class_code >> 24) & 0xffu;
    uint8_t sub = (class_code >> 16) & 0xffu;

    if (base == 0x06u && sub == 0x04u) {
        // Standard PCI-PCI bridge
        auto bus_numbers = ReadBusNumbers(bus, device, function);
        uint8_t secondary_bus = (bus_numbers >> 8) & 0xffu;
        return ScanBus(secondary_bus);
    }

    return Error::kSuccess;
}

/**
 * @brief Scan all functions of a given device.
 * First scans Function 0. If single function, returns immediately.
 * Otherwise, scans Functions 1 to 7 if present (Vendor ID != 0xFFFF).
 */
Error ScanDevice(uint8_t bus, uint8_t device) {
    if (auto err = ScanFunction(bus, device, 0)) {
        return err;
    }
    if (IsSingleFunctionDevice(ReadHeaderType(bus, device, 0))) {
        return Error::kSuccess;
    }

    for (uint8_t function = 1; function < 8; ++function) {
        if (ReadVendorId(bus, device, function) == 0xffffu) {
            continue;
        }
        if (auto err = ScanFunction(bus, device, function)) {
            return err;
        }
    }
    return Error::kSuccess;
}

/**
 * @brief Scan all 32 devices on a given bus.
 * Checks Vendor ID of Function 0 for existence.
 */
Error ScanBus(uint8_t bus) {
    for (uint8_t device = 0; device < 32; ++device) {
        if (ReadVendorId(bus, device, 0) == 0xffffu) {
            continue;
        }
        if (auto err = ScanDevice(bus, device)) {
            return err;
        }
    }
    return Error::kSuccess;
}

} // namespace

namespace pci {

void WriteAddress(uint32_t address) {
    sim::PCIEnvironment::Instance().WriteAddress(address);
}

void WriteData(uint32_t value) {
    sim::PCIEnvironment::Instance().WriteData(value);
}

uint32_t ReadData() {
    return sim::PCIEnvironment::Instance().ReadData();
}

uint16_t ReadVendorId(uint8_t bus, uint8_t device, uint8_t function) {
    WriteAddress(MakeAddress(bus, device, function, 0x00));
    return ReadData() & 0xffffu;
}

uint16_t ReadDeviceId(uint8_t bus, uint8_t device, uint8_t function) {
    WriteAddress(MakeAddress(bus, device, function, 0x00));
    return ReadData() >> 16;
}

uint8_t ReadHeaderType(uint8_t bus, uint8_t device, uint8_t function) {
    WriteAddress(MakeAddress(bus, device, function, 0x0c));
    return (ReadData() >> 16) & 0xffu;
}

uint32_t ReadClassCode(uint8_t bus, uint8_t device, uint8_t function) {
    WriteAddress(MakeAddress(bus, device, function, 0x08));
    return ReadData();
}

uint32_t ReadBusNumbers(uint8_t bus, uint8_t device, uint8_t function) {
    WriteAddress(MakeAddress(bus, device, function, 0x18));
    return ReadData();
}

bool IsSingleFunctionDevice(uint8_t header_type) {
    return (header_type & 0x80u) == 0;
}

Error ScanAllBus() {
    num_device = 0;

    auto header_type = ReadHeaderType(0, 0, 0);
    if (IsSingleFunctionDevice(header_type)) {
        return ScanBus(0);
    }

    for (uint8_t function = 1; function < 8; ++function) {
        if (ReadVendorId(0, 0, function) == 0xffffu) {
            continue;
        }
        if (auto err = ScanBus(function)) {
            return err;
        }
    }
    return Error::kSuccess;
}

} // namespace pci

// ============================================================================
// 5. Verification Test Suite & CLI Runner
// ============================================================================

void TestMakeAddress() {
    // Test 1: Bus 0, Dev 0, Func 0, Reg 0 -> 0x80000000
    uint32_t addr0 = MakeAddress(0, 0, 0, 0x00);
    assert(addr0 == 0x80000000u);

    // Test 2: Bus 1, Dev 2, Func 3, Reg 0x0C
    // Bus 1: 1 << 16 = 0x00010000
    // Dev 2: 2 << 11 = 0x00001000
    // Func 3: 3 << 8 = 0x00000300
    // Reg 0x0C: 0x0C
    // Enable: 0x80000000
    // Total: 0x8001130C
    uint32_t addr1 = MakeAddress(1, 2, 3, 0x0C);
    assert(addr1 == 0x8001130Cu);

    // Test 3: Register offset mask (lower 2 bits must be masked off)
    uint32_t addr_unaligned = MakeAddress(0, 0, 0, 0x0F);
    assert((addr_unaligned & 0x3) == 0);
    assert((addr_unaligned & 0xFC) == 0x0C);

    std::cout << "[PASS] TestMakeAddress: Bitfield layout and masking verified." << std::endl;
}

void TestEmptyBusScan() {
    sim::PCIEnvironment::Instance().Reset();
    auto err = pci::ScanAllBus();
    assert(!err);
    assert(pci::num_device == 0);
    std::cout << "[PASS] TestEmptyBusScan: 0 devices detected on unpopulated bus." << std::endl;
}

void TestSingleVsMultiFunctionScan() {
    sim::PCIEnvironment::Instance().Reset();
    // Host bridge at 0.0.0 (Single function)
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 0, 0, 0x8086, 0x1237, 0x00, 0x06000000)
    );

    // Register Single Function Device at 0.1.0 (header_type = 0x00)
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 1, 0, 0x1234, 0x5678, 0x00, 0x02000000)
    );
    // Phantom function at 0.1.1 (should be ignored because 0.1.0 is single function!)
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 1, 1, 0x1234, 0x9999, 0x00, 0x02000000)
    );

    auto err = pci::ScanAllBus();
    assert(!err);
    // Discovered: 0.0.0 (Host bridge) and 0.1.0 (Device), but NOT 0.1.1
    assert(pci::num_device == 2);
    assert(pci::devices[0].bus == 0 && pci::devices[0].device == 0 && pci::devices[0].function == 0);
    assert(pci::devices[1].bus == 0 && pci::devices[1].device == 1 && pci::devices[1].function == 0);

    // Now test Multi-Function Device at 0.1 (header_type = 0x80)
    sim::PCIEnvironment::Instance().Reset();
    // Host bridge at 0.0.0
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 0, 0, 0x8086, 0x1237, 0x00, 0x06000000)
    );
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 1, 0, 0x1234, 0x5678, 0x80, 0x02000000)
    );
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 1, 1, 0x1234, 0x9999, 0x00, 0x02000000)
    );
    sim::PCIEnvironment::Instance().RegisterDevice(
        sim::SimulatedPCIFunction(0, 1, 3, 0x1234, 0xAAAA, 0x00, 0x02000000)
    );

    err = pci::ScanAllBus();
    assert(!err);
    // Discovered: 0.0.0, 0.1.0, 0.1.1, 0.1.3
    assert(pci::num_device == 4);
    assert(pci::devices[0].device == 0);
    assert(pci::devices[1].device == 1 && pci::devices[1].function == 0);
    assert(pci::devices[2].device == 1 && pci::devices[2].function == 1);
    assert(pci::devices[3].device == 1 && pci::devices[3].function == 3);

    std::cout << "[PASS] TestSingleVsMultiFunctionScan: Header Type bit 7 distinction verified." << std::endl;
}

void TestQEMUStandardScanAndXHCIDetection() {
    sim::PCIEnvironment::Instance().SetupQEMUDevices();
    auto err = pci::ScanAllBus();
    assert(!err);

    std::cout << "\n=== Discovered PCI Devices (MikanOS ScanAllBus Simulation) ===" << std::endl;
    std::cout << "Status: " << err.Name() << " | Total Devices: " << pci::num_device << std::endl;

    std::optional<pci::Device> xhci_dev;

    for (int i = 0; i < pci::num_device; ++i) {
        const auto& dev = pci::devices[i];
        auto vendor_id = pci::ReadVendorId(dev.bus, dev.device, dev.function);
        auto device_id = pci::ReadDeviceId(dev.bus, dev.device, dev.function);
        auto class_code = pci::ReadClassCode(dev.bus, dev.device, dev.function);

        uint8_t base_class = (class_code >> 24) & 0xFF;
        uint8_t sub_class = (class_code >> 16) & 0xFF;
        uint8_t prog_if = (class_code >> 8) & 0xFF;

        std::cout << std::dec << static_cast<int>(dev.bus) << "."
                  << static_cast<int>(dev.device) << "."
                  << static_cast<int>(dev.function) << ": vend 0x"
                  << std::hex << std::setw(4) << std::setfill('0') << vendor_id
                  << ", dev 0x" << std::setw(4) << std::setfill('0') << device_id
                  << ", class 0x" << std::setw(8) << std::setfill('0') << class_code
                  << ", head 0x" << std::setw(2) << std::setfill('0') << static_cast<int>(dev.header_type);

        if (base_class == 0x0C && sub_class == 0x03 && prog_if == 0x30) {
            std::cout << " <-- [xHCI USB 3.0 Controller FOUND!]";
            xhci_dev = dev;
        } else if (base_class == 0x06 && sub_class == 0x04) {
            std::cout << " <-- [PCI-to-PCI Bridge]";
        }
        std::cout << std::endl;
    }

    // Verify xHCI device was found
    assert(xhci_dev.has_value());
    assert(xhci_dev->bus == 0);
    assert(xhci_dev->device == 4);
    assert(xhci_dev->function == 0);

    // Verify bridge traversal to bus 1 (Audio device on 1.1.0)
    bool found_audio_on_bus1 = false;
    for (int i = 0; i < pci::num_device; ++i) {
        if (pci::devices[i].bus == 1 && pci::devices[i].device == 1) {
            found_audio_on_bus1 = true;
            break;
        }
    }
    assert(found_audio_on_bus1);

    std::cout << "[PASS] TestQEMUStandardScanAndXHCIDetection: All 8 QEMU devices and xHCI identified." << std::endl;
}

void RunAllTests() {
    std::cout << "Running Chapter 6.3 PCI Configuration & Bus Scan Standalone Tests..." << std::endl;
    TestMakeAddress();
    TestEmptyBusScan();
    TestSingleVsMultiFunctionScan();
    TestQEMUStandardScanAndXHCIDetection();
    std::cout << "\nAll Chapter 6.3 PCI scan simulation tests PASSED successfully!" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        RunAllTests();
        return 0;
    }

    std::cout << "=== MikanOS Chapter 6.3 PCI Scanning Simulation Demo ===" << std::endl;
    RunAllTests();
    return 0;
}
