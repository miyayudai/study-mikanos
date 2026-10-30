/**
 * @file save_memory_map_sim.cpp
 * @brief C++20 Simulation and Verification of UEFI File System Protocols and Memory Map CSV Export
 *
 * Simulates MikanOS Chapter 2.6 (osbook_day02b):
 * EFI_SIMPLE_FILE_SYSTEM_PROTOCOL, EFI_FILE_PROTOCOL, OpenRootDir,
 * and SaveMemoryMap CSV serialization with attribute bitmasking.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::uefi {

// ============================================================================
// UEFI Basic Types and Status Codes (UEFI Specification 2.8+)
// ============================================================================

using EFI_STATUS = uint64_t;
using EFI_PHYSICAL_ADDRESS = uint64_t;
using EFI_VIRTUAL_ADDRESS = uint64_t;
using EFI_HANDLE = void*;
using UINTN = size_t;
using UINT32 = uint32_t;
using UINT64 = uint64_t;
using CHAR8 = char;
using CHAR16 = char16_t;

constexpr EFI_STATUS EFI_SUCCESS             = 0ULL;
constexpr EFI_STATUS EFI_ERROR_MASK          = 0x8000'0000'0000'0000ULL;
constexpr EFI_STATUS EFI_INVALID_PARAMETER   = EFI_ERROR_MASK | 2ULL;
constexpr EFI_STATUS EFI_UNSUPPORTED         = EFI_ERROR_MASK | 3ULL;
constexpr EFI_STATUS EFI_BUFFER_TOO_SMALL    = EFI_ERROR_MASK | 5ULL;
constexpr EFI_STATUS EFI_DEVICE_ERROR        = EFI_ERROR_MASK | 7ULL;
constexpr EFI_STATUS EFI_ACCESS_DENIED       = EFI_ERROR_MASK | 15ULL;

constexpr uint64_t EFI_FILE_MODE_READ   = 0x0000'0000'0000'0001ULL;
constexpr uint64_t EFI_FILE_MODE_WRITE  = 0x0000'0000'0000'0002ULL;
constexpr uint64_t EFI_FILE_MODE_CREATE = 0x8000'0000'0000'0000ULL;

constexpr uint64_t EFI_MEMORY_RUNTIME = 0x8000'0000'0000'0000ULL;

// ============================================================================
// UEFI Memory Types and Descriptor Layout
// ============================================================================

enum class MemoryType : uint32_t {
    EfiReservedMemoryType      = 0,
    EfiLoaderCode              = 1,
    EfiLoaderData              = 2,
    EfiBootServicesCode        = 3,
    EfiBootServicesData        = 4,
    EfiRuntimeServicesCode     = 5,
    EfiRuntimeServicesData     = 6,
    EfiConventionalMemory      = 7,
    EfiUnusableMemory          = 8,
    EfiACPIReclaimMemory       = 9,
    EfiACPIMemoryNVS           = 10,
    EfiMemoryMappedIO          = 11,
    EfiMemoryMappedIOPortSpace = 12,
    EfiPalCode                 = 13,
    EfiPersistentMemory        = 14,
    EfiMaxMemoryType           = 15
};

const char* GetMemoryTypeName(MemoryType type) noexcept {
    switch (type) {
        case MemoryType::EfiReservedMemoryType:      return "EfiReservedMemoryType";
        case MemoryType::EfiLoaderCode:              return "EfiLoaderCode";
        case MemoryType::EfiLoaderData:              return "EfiLoaderData";
        case MemoryType::EfiBootServicesCode:        return "EfiBootServicesCode";
        case MemoryType::EfiBootServicesData:        return "EfiBootServicesData";
        case MemoryType::EfiRuntimeServicesCode:     return "EfiRuntimeServicesCode";
        case MemoryType::EfiRuntimeServicesData:     return "EfiRuntimeServicesData";
        case MemoryType::EfiConventionalMemory:      return "EfiConventionalMemory";
        case MemoryType::EfiUnusableMemory:          return "EfiUnusableMemory";
        case MemoryType::EfiACPIReclaimMemory:       return "EfiACPIReclaimMemory";
        case MemoryType::EfiACPIMemoryNVS:           return "EfiACPIMemoryNVS";
        case MemoryType::EfiMemoryMappedIO:          return "EfiMemoryMappedIO";
        case MemoryType::EfiMemoryMappedIOPortSpace: return "EfiMemoryMappedIOPortSpace";
        case MemoryType::EfiPalCode:                 return "EfiPalCode";
        case MemoryType::EfiPersistentMemory:        return "EfiPersistentMemory";
        case MemoryType::EfiMaxMemoryType:           return "EfiMaxMemoryType";
        default:                                     return "InvalidMemoryType";
    }
}

#pragma pack(push, 1)
struct EFI_MEMORY_DESCRIPTOR {
    uint32_t Type;
    uint32_t Pad;
    EFI_PHYSICAL_ADDRESS PhysicalStart;
    EFI_VIRTUAL_ADDRESS VirtualStart;
    uint64_t NumberOfPages;
    uint64_t Attribute;
};
#pragma pack(pop)

static_assert(sizeof(EFI_MEMORY_DESCRIPTOR) == 40, "EFI_MEMORY_DESCRIPTOR must be 40 bytes");

struct MemoryMap {
    UINTN buffer_size;
    void* buffer;
    UINTN map_size;
    UINTN map_key;
    UINTN descriptor_size;
    UINT32 descriptor_version;
};

// ============================================================================
// EFI_FILE_PROTOCOL & EFI_SIMPLE_FILE_SYSTEM_PROTOCOL Mock
// ============================================================================

struct EFI_FILE_PROTOCOL;

using EFI_FILE_OPEN = EFI_STATUS (*)(
    EFI_FILE_PROTOCOL* This,
    EFI_FILE_PROTOCOL** NewHandle,
    const char* FileName,
    uint64_t OpenMode,
    uint64_t Attributes
);

using EFI_FILE_CLOSE = EFI_STATUS (*)(
    EFI_FILE_PROTOCOL* This
);

using EFI_FILE_WRITE = EFI_STATUS (*)(
    EFI_FILE_PROTOCOL* This,
    UINTN* BufferSize,
    const void* Buffer
);

using EFI_FILE_READ = EFI_STATUS (*)(
    EFI_FILE_PROTOCOL* This,
    UINTN* BufferSize,
    void* Buffer
);

struct EFI_FILE_PROTOCOL {
    EFI_FILE_OPEN Open;
    EFI_FILE_CLOSE Close;
    EFI_FILE_WRITE Write;
    EFI_FILE_READ Read;

    // Internal simulated backend fields
    std::string name;
    uint64_t mode{0};
    uint64_t attributes{0};
    bool is_directory{false};
    bool is_closed{false};
    std::string content_buffer;
    std::vector<std::unique_ptr<EFI_FILE_PROTOCOL>> children;
};

// Forward declaration of static dispatchers
static EFI_STATUS MockFileOpen(EFI_FILE_PROTOCOL* This, EFI_FILE_PROTOCOL** NewHandle, const char* FileName, uint64_t OpenMode, uint64_t Attributes);
static EFI_STATUS MockFileClose(EFI_FILE_PROTOCOL* This);
static EFI_STATUS MockFileWrite(EFI_FILE_PROTOCOL* This, UINTN* BufferSize, const void* Buffer);
static EFI_STATUS MockFileRead(EFI_FILE_PROTOCOL* This, UINTN* BufferSize, void* Buffer);

std::unique_ptr<EFI_FILE_PROTOCOL> CreateMockFile(
    std::string name,
    uint64_t mode = EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE,
    uint64_t attributes = 0,
    bool is_directory = false
) {
    auto file = std::make_unique<EFI_FILE_PROTOCOL>();
    file->Open = MockFileOpen;
    file->Close = MockFileClose;
    file->Write = MockFileWrite;
    file->Read = MockFileRead;
    file->name = std::move(name);
    file->mode = mode;
    file->attributes = attributes;
    file->is_directory = is_directory;
    file->is_closed = false;
    return file;
}

static EFI_STATUS MockFileOpen(
    EFI_FILE_PROTOCOL* This,
    EFI_FILE_PROTOCOL** NewHandle,
    const char* FileName,
    uint64_t OpenMode,
    uint64_t Attributes
) {
    if (!This || !NewHandle || !FileName) {
        return EFI_INVALID_PARAMETER;
    }
    if (This->is_closed) {
        return EFI_DEVICE_ERROR;
    }
    if (!This->is_directory) {
        return EFI_UNSUPPORTED;
    }

    std::string path(FileName);
    while (!path.empty() && (path.front() == '\\' || path.front() == '/')) {
        path.erase(path.begin());
    }

    for (const auto& child : This->children) {
        if (child->name == path) {
            child->mode = OpenMode;
            child->is_closed = false;
            *NewHandle = child.get();
            return EFI_SUCCESS;
        }
    }

    if (OpenMode & EFI_FILE_MODE_CREATE) {
        auto new_file = CreateMockFile(path, OpenMode, Attributes, false);
        *NewHandle = new_file.get();
        This->children.push_back(std::move(new_file));
        return EFI_SUCCESS;
    }

    return EFI_DEVICE_ERROR;
}

static EFI_STATUS MockFileClose(EFI_FILE_PROTOCOL* This) {
    if (!This) {
        return EFI_INVALID_PARAMETER;
    }
    This->is_closed = true;
    return EFI_SUCCESS;
}

static EFI_STATUS MockFileWrite(
    EFI_FILE_PROTOCOL* This,
    UINTN* BufferSize,
    const void* Buffer
) {
    if (!This || !BufferSize || !Buffer) {
        return EFI_INVALID_PARAMETER;
    }
    if (This->is_closed) {
        return EFI_DEVICE_ERROR;
    }
    if (!(This->mode & EFI_FILE_MODE_WRITE)) {
        return EFI_ACCESS_DENIED;
    }
    if (This->is_directory) {
        return EFI_UNSUPPORTED;
    }

    const char* data = static_cast<const char*>(Buffer);
    This->content_buffer.append(data, *BufferSize);
    return EFI_SUCCESS;
}

static EFI_STATUS MockFileRead(
    EFI_FILE_PROTOCOL* This,
    UINTN* BufferSize,
    void* Buffer
) {
    if (!This || !BufferSize || !Buffer) {
        return EFI_INVALID_PARAMETER;
    }
    if (This->is_closed) {
        return EFI_DEVICE_ERROR;
    }
    return EFI_SUCCESS;
}

// ============================================================================
// Simple File System Protocol & Loaded Image Protocol Mocks
// ============================================================================

struct EFI_SIMPLE_FILE_SYSTEM_PROTOCOL;

using EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_OPEN_VOLUME = EFI_STATUS (*)(
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL* This,
    EFI_FILE_PROTOCOL** Root
);

struct EFI_SIMPLE_FILE_SYSTEM_PROTOCOL {
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_OPEN_VOLUME OpenVolume;
    std::unique_ptr<EFI_FILE_PROTOCOL> root_dir;
};

static EFI_STATUS MockOpenVolume(
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL* This,
    EFI_FILE_PROTOCOL** Root
) {
    if (!This || !Root) {
        return EFI_INVALID_PARAMETER;
    }
    *Root = This->root_dir.get();
    return EFI_SUCCESS;
}

std::unique_ptr<EFI_SIMPLE_FILE_SYSTEM_PROTOCOL> CreateMockSimpleFileSystem() {
    auto fs = std::make_unique<EFI_SIMPLE_FILE_SYSTEM_PROTOCOL>();
    fs->OpenVolume = MockOpenVolume;
    fs->root_dir = CreateMockFile("\\", EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE, 0, true);
    return fs;
}

// ============================================================================
// MikanOS Day02b SaveMemoryMap Implementation Reproduction
// ============================================================================

EFI_STATUS SaveMemoryMap(struct MemoryMap* map, EFI_FILE_PROTOCOL* file) {
    char buf[256];
    UINTN len;

    const char* header = "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n";
    len = std::strlen(header);
    EFI_STATUS status = file->Write(file, &len, header);
    if (status != EFI_SUCCESS) {
        return status;
    }

    EFI_PHYSICAL_ADDRESS iter;
    int i;
    for (iter = reinterpret_cast<EFI_PHYSICAL_ADDRESS>(map->buffer), i = 0;
         iter < reinterpret_cast<EFI_PHYSICAL_ADDRESS>(map->buffer) + map->map_size;
         iter += map->descriptor_size, i++) {
        auto* desc = reinterpret_cast<EFI_MEMORY_DESCRIPTOR*>(iter);
        int formatted_len = std::snprintf(
            buf, sizeof(buf),
            "%u, %x, %-s, %08lx, %lx, %lx\n",
            i,
            desc->Type,
            GetMemoryTypeName(static_cast<MemoryType>(desc->Type)),
            static_cast<unsigned long>(desc->PhysicalStart),
            static_cast<unsigned long>(desc->NumberOfPages),
            static_cast<unsigned long>(desc->Attribute & 0xffffflu)
        );

        if (formatted_len > 0) {
            len = static_cast<UINTN>(formatted_len);
            status = file->Write(file, &len, buf);
            if (status != EFI_SUCCESS) {
                return status;
            }
        }
    }

    return EFI_SUCCESS;
}

// ============================================================================
// Unit Verifications and Assertions
// ============================================================================

void TestFileProtocolOpenAndModes() {
    auto fs = CreateMockSimpleFileSystem();
    EFI_FILE_PROTOCOL* root = nullptr;
    EFI_STATUS st = fs->OpenVolume(fs.get(), &root);
    assert(st == EFI_SUCCESS);
    assert(root != nullptr);
    assert(root->is_directory);

    // Open file without CREATE flag (should fail for nonexistent file)
    EFI_FILE_PROTOCOL* file1 = nullptr;
    st = root->Open(root, &file1, "\\nonexistent", EFI_FILE_MODE_READ, 0);
    assert(st != EFI_SUCCESS);

    // Open file with CREATE flag
    st = root->Open(root, &file1, "\\memmap", EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE, 0);
    assert(st == EFI_SUCCESS);
    assert(file1 != nullptr);
    assert(!file1->is_closed);

    // Write to file
    const char* msg = "Hello UEFI File System!\n";
    UINTN len = std::strlen(msg);
    st = file1->Write(file1, &len, msg);
    assert(st == EFI_SUCCESS);
    assert(file1->content_buffer == msg);

    // Close file
    st = file1->Close(file1);
    assert(st == EFI_SUCCESS);
    assert(file1->is_closed);

    // Write after close should fail
    st = file1->Write(file1, &len, msg);
    assert(st == EFI_DEVICE_ERROR);

    std::cout << "  [PASS] TestFileProtocolOpenAndModes\n";
}

void TestSaveMemoryMapCsvFormatting() {
    // Construct simulated memory map buffer with 3 descriptors
    std::vector<EFI_MEMORY_DESCRIPTOR> descs = {
        {
            static_cast<uint32_t>(MemoryType::EfiBootServicesCode),
            0,
            0x0000'0000'0010'0000ULL,
            0x0000'0000'0010'0000ULL,
            0x20ULL, // 32 pages = 128 KiB
            0x0000'0000'0000'000fULL | EFI_MEMORY_RUNTIME, // Attributes with upper bit
        },
        {
            static_cast<uint32_t>(MemoryType::EfiConventionalMemory),
            0,
            0x0000'0000'0020'0000ULL,
            0x0000'0000'0020'0000ULL,
            0x1000ULL, // 4096 pages = 16 MiB
            0x0000'0000'0000'000fULL, // Attributes without runtime bit
        },
        {
            static_cast<uint32_t>(MemoryType::EfiMemoryMappedIO),
            0,
            0x0000'0000'FED0'0000ULL,
            0x0000'0000'FED0'0000ULL,
            0x1ULL, // 1 page = 4 KiB
            0x0000'0000'0000'0001ULL | EFI_MEMORY_RUNTIME,
        },
    };

    MemoryMap map{};
    map.buffer = descs.data();
    map.buffer_size = descs.size() * sizeof(EFI_MEMORY_DESCRIPTOR);
    map.map_size = map.buffer_size;
    map.descriptor_size = sizeof(EFI_MEMORY_DESCRIPTOR);
    map.descriptor_version = 1;

    auto file = CreateMockFile("memmap", EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE);
    EFI_STATUS st = SaveMemoryMap(&map, file.get());
    assert(st == EFI_SUCCESS);

    std::string csv = file->content_buffer;
    assert(csv.find("Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n") == 0);
    assert(csv.find("0, 3, EfiBootServicesCode, 00100000, 20, f\n") != std::string::npos);
    assert(csv.find("1, 7, EfiConventionalMemory, 00200000, 1000, f\n") != std::string::npos);
    assert(csv.find("2, b, EfiMemoryMappedIO, fed00000, 1, 1\n") != std::string::npos);

    // Verify attribute bitmask 0xffffflu stripped EFI_MEMORY_RUNTIME (Bit 63)
    assert(csv.find("8000") == std::string::npos);

    std::cout << "  [PASS] TestSaveMemoryMapCsvFormatting\n";
}

void TestVariableDescriptorSizeStepping() {
    // Simulate firmware with 48-byte descriptors (40 bytes data + 8 bytes vendor padding)
    constexpr size_t DESC_SIZE = 48;
    constexpr size_t NUM_DESCS = 3;
    std::vector<uint8_t> buffer(DESC_SIZE * NUM_DESCS, 0xCC);

    for (size_t i = 0; i < NUM_DESCS; ++i) {
        auto* desc = reinterpret_cast<EFI_MEMORY_DESCRIPTOR*>(buffer.data() + i * DESC_SIZE);
        desc->Type = static_cast<uint32_t>(MemoryType::EfiConventionalMemory);
        desc->Pad = 0;
        desc->PhysicalStart = 0x1000'0000ULL + i * 0x0100'0000ULL;
        desc->VirtualStart = desc->PhysicalStart;
        desc->NumberOfPages = 0x400ULL; // 1024 pages = 4 MiB
        desc->Attribute = 0x0fULL;
    }

    MemoryMap map{};
    map.buffer = buffer.data();
    map.buffer_size = buffer.size();
    map.map_size = buffer.size();
    map.descriptor_size = DESC_SIZE; // 48 bytes!
    map.descriptor_version = 1;

    auto file = CreateMockFile("memmap", EFI_FILE_MODE_READ | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_CREATE);
    EFI_STATUS st = SaveMemoryMap(&map, file.get());
    assert(st == EFI_SUCCESS);

    std::string csv = file->content_buffer;
    assert(csv.find("0, 7, EfiConventionalMemory, 10000000, 400, f\n") != std::string::npos);
    assert(csv.find("1, 7, EfiConventionalMemory, 11000000, 400, f\n") != std::string::npos);
    assert(csv.find("2, 7, EfiConventionalMemory, 12000000, 400, f\n") != std::string::npos);

    std::cout << "  [PASS] TestVariableDescriptorSizeStepping (48B padding handled correctly)\n";
}

} // namespace mikan::uefi

int main() {
    std::cout << "======================================================================\n";
    std::cout << "MikanOS Chapter 2.6: UEFI File System & Memory Map CSV C++20 Verification\n";
    std::cout << "======================================================================\n";

    mikan::uefi::TestFileProtocolOpenAndModes();
    mikan::uefi::TestSaveMemoryMapCsvFormatting();
    mikan::uefi::TestVariableDescriptorSizeStepping();

    std::cout << "\nAll Section 2.6 C++20 verifications passed successfully!\n";
    return 0;
}
