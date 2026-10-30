/**
 * @file pointer_assembly_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 3.7 (Pointers & Assembly Level)
 *
 * Covers:
 * 1. Data transfer instructions: MOV [addr], reg and MOV reg, [addr]
 * 2. Memory access size specifiers: BYTE PTR, WORD PTR, DWORD PTR, QWORD PTR
 * 3. Scale-Index-Base (SIB) hardware addressing: [Base + Index * Scale + Disp]
 * 4. LEA (Load Effective Address) vs MOV (Memory Dereference)
 * 5. Local variables, stack frame ([rbp - offset]), and address-of (&x)
 * 6. Pixel rendering loop disassembly: -O0 vs -O2 vs volatile semantics
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
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace mikan::pointer_assembly_sim {

// ============================================================================
// 1. Data Size Specifiers & Type Mapping
// ============================================================================

enum class DataSize : uint8_t {
    Byte  = 1,  // 8-bit  (uint8_t,  char)     -> BYTE PTR
    Word  = 2,  // 16-bit (uint16_t, short)    -> WORD PTR
    Dword = 4,  // 32-bit (uint32_t, int)      -> DWORD PTR
    Qword = 8   // 64-bit (uint64_t, pointers) -> QWORD PTR
};

[[nodiscard]] constexpr std::string_view DataSizeToSpecifier(DataSize size) noexcept {
    switch (size) {
        case DataSize::Byte:  return "BYTE PTR";
        case DataSize::Word:  return "WORD PTR";
        case DataSize::Dword: return "DWORD PTR";
        case DataSize::Qword: return "QWORD PTR";
    }
    return "UNKNOWN PTR";
}

template <typename T>
[[nodiscard]] constexpr DataSize TypeToDataSize() noexcept {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "Unsupported data type size for x86-64 memory operand");
    if constexpr (sizeof(T) == 1) return DataSize::Byte;
    if constexpr (sizeof(T) == 2) return DataSize::Word;
    if constexpr (sizeof(T) == 4) return DataSize::Dword;
    if constexpr (sizeof(T) == 8) return DataSize::Qword;
}

// Compile-time checks
static_assert(TypeToDataSize<uint8_t>() == DataSize::Byte);
static_assert(TypeToDataSize<uint16_t>() == DataSize::Word);
static_assert(TypeToDataSize<uint32_t>() == DataSize::Dword);
static_assert(TypeToDataSize<uint64_t>() == DataSize::Qword);
static_assert(TypeToDataSize<void*>() == DataSize::Qword);

// ============================================================================
// 2. SIB (Scale-Index-Base) Hardware Addressing Mode
// ============================================================================

/**
 * @brief x86-64 SIB (Scale-Index-Base) addressing calculation
 * 
 * Hardware encoding in ModR/M & SIB byte:
 * Scale bits (2 bits):
 *   00b = 1 (Scale factor 1)
 *   01b = 2 (Scale factor 2)
 *   10b = 4 (Scale factor 4)
 *   11b = 8 (Scale factor 8)
 * 
 * Formula:
 *   Effective Address = Base + (Index * Scale) + Displacement
 */
struct SIBAddress {
    uint64_t base{0};
    uint64_t index{0};
    uint8_t  scale{1};       // Must be 1, 2, 4, or 8
    int64_t  displacement{0};

    [[nodiscard]] constexpr bool IsValidScale() const noexcept {
        return scale == 1 || scale == 2 || scale == 4 || scale == 8;
    }

    [[nodiscard]] constexpr uint8_t ScaleToEncoding() const {
        switch (scale) {
            case 1: return 0b00;
            case 2: return 0b01;
            case 4: return 0b10;
            case 8: return 0b11;
            default: throw std::invalid_argument("Invalid SIB scale factor");
        }
    }

    [[nodiscard]] constexpr uint64_t CalculateEffectiveAddress() const noexcept {
        return base + (index * scale) + static_cast<uint64_t>(displacement);
    }

    [[nodiscard]] std::string ToAssemblyString(std::string_view base_reg = "rax",
                                              std::string_view index_reg = "rcx") const {
        std::ostringstream oss;
        oss << "[" << base_reg;
        if (index_reg != "") {
            oss << " + " << index_reg;
            if (scale > 1) {
                oss << "*" << static_cast<int>(scale);
            }
        }
        if (displacement > 0) {
            oss << " + " << displacement;
        } else if (displacement < 0) {
            oss << " - " << (-displacement);
        }
        oss << "]";
        return oss.str();
    }
};

// ============================================================================
// 3. Virtual x86-64 Machine & Memory Model (MOV vs LEA)
// ============================================================================

class VirtualCpuSimulator {
public:
    explicit VirtualCpuSimulator(size_t memory_size = 1024 * 1024)
        : memory_(memory_size, 0) {}

    // Register Bank
    enum class Reg : uint8_t {
        RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP,
        R8,  R9,  R10, R11, R12, R13, R14, R15,
        Count
    };

    static std::string_view RegName(Reg r, DataSize size = DataSize::Qword) noexcept {
        switch (r) {
            case Reg::RAX:
                if (size == DataSize::Byte) return "al";
                if (size == DataSize::Word) return "ax";
                if (size == DataSize::Dword) return "eax";
                return "rax";
            case Reg::RBX:
                if (size == DataSize::Byte) return "bl";
                if (size == DataSize::Word) return "bx";
                if (size == DataSize::Dword) return "ebx";
                return "rbx";
            case Reg::RCX:
                if (size == DataSize::Byte) return "cl";
                if (size == DataSize::Word) return "cx";
                if (size == DataSize::Dword) return "ecx";
                return "rcx";
            case Reg::RDX:
                if (size == DataSize::Byte) return "dl";
                if (size == DataSize::Word) return "dx";
                if (size == DataSize::Dword) return "edx";
                return "rdx";
            case Reg::RSI:
                if (size == DataSize::Byte) return "sil";
                if (size == DataSize::Word) return "si";
                if (size == DataSize::Dword) return "esi";
                return "rsi";
            case Reg::RDI:
                if (size == DataSize::Byte) return "dil";
                if (size == DataSize::Word) return "di";
                if (size == DataSize::Dword) return "edi";
                return "rdi";
            case Reg::RBP:
                if (size == DataSize::Byte) return "bpl";
                if (size == DataSize::Word) return "bp";
                if (size == DataSize::Dword) return "ebp";
                return "rbp";
            case Reg::RSP:
                if (size == DataSize::Byte) return "spl";
                if (size == DataSize::Word) return "sp";
                if (size == DataSize::Dword) return "esp";
                return "rsp";
            case Reg::R8:
                if (size == DataSize::Byte) return "r8b";
                if (size == DataSize::Word) return "r8w";
                if (size == DataSize::Dword) return "r8d";
                return "r8";
            case Reg::R9:
                if (size == DataSize::Byte) return "r9b";
                if (size == DataSize::Word) return "r9w";
                if (size == DataSize::Dword) return "r9d";
                return "r9";
            default:
                return "reg";
        }
    }

    void SetRegister(Reg r, uint64_t val) noexcept {
        registers_[static_cast<size_t>(r)] = val;
    }

    [[nodiscard]] uint64_t GetRegister(Reg r) const noexcept {
        return registers_[static_cast<size_t>(r)];
    }

    // Memory Bus Access
    void WriteMemory(uint64_t addr, uint64_t value, DataSize size) {
        size_t bytes = static_cast<size_t>(size);
        if (addr + bytes > memory_.size()) {
            throw std::out_of_range("Memory access violation: address out of bounds");
        }
        std::memcpy(&memory_[addr], &value, bytes);
        memory_write_count_++;
        bytes_written_ += bytes;
    }

    [[nodiscard]] uint64_t ReadMemory(uint64_t addr, DataSize size) {
        size_t bytes = static_cast<size_t>(size);
        if (addr + bytes > memory_.size()) {
            throw std::out_of_range("Memory access violation: address out of bounds");
        }
        uint64_t value = 0;
        std::memcpy(&value, &memory_[addr], bytes);
        memory_read_count_++;
        bytes_read_ += bytes;
        return value;
    }

    /**
     * @brief Execute MOV [mem], reg (Store to memory via pointer dereference)
     */
    void ExecuteMovStore(const SIBAddress& mem_addr, Reg src_reg, DataSize size) {
        uint64_t effective_addr = mem_addr.CalculateEffectiveAddress();
        uint64_t reg_val = GetRegister(src_reg);
        WriteMemory(effective_addr, reg_val, size);
        executed_instructions_.push_back(
            "mov " + std::string(DataSizeToSpecifier(size)) + " " +
            mem_addr.ToAssemblyString(RegName(Reg::RAX), RegName(Reg::RCX)) + ", " +
            std::string(RegName(src_reg, size)));
    }

    /**
     * @brief Execute MOV reg, [mem] (Load from memory via pointer dereference)
     */
    void ExecuteMovLoad(Reg dest_reg, const SIBAddress& mem_addr, DataSize size) {
        uint64_t effective_addr = mem_addr.CalculateEffectiveAddress();
        uint64_t mem_val = ReadMemory(effective_addr, size);
        SetRegister(dest_reg, mem_val);
        executed_instructions_.push_back(
            "mov " + std::string(RegName(dest_reg, size)) + ", " +
            std::string(DataSizeToSpecifier(size)) + " " +
            mem_addr.ToAssemblyString(RegName(Reg::RAX), RegName(Reg::RCX)));
    }

    /**
     * @brief Execute LEA reg, [mem] (Load Effective Address)
     * 
     * Key principle: LEA NEVER accesses memory bus! It purely performs arithmetic.
     */
    void ExecuteLea(Reg dest_reg, const SIBAddress& mem_addr) noexcept {
        uint64_t effective_addr = mem_addr.CalculateEffectiveAddress();
        SetRegister(dest_reg, effective_addr);
        // Note: No memory read/write counts are incremented!
        executed_instructions_.push_back(
            "lea " + std::string(RegName(dest_reg, DataSize::Qword)) + ", " +
            mem_addr.ToAssemblyString(RegName(Reg::RAX), RegName(Reg::RCX)));
    }

    [[nodiscard]] size_t GetMemoryWriteCount() const noexcept { return memory_write_count_; }
    [[nodiscard]] size_t GetMemoryReadCount() const noexcept { return memory_read_count_; }
    [[nodiscard]] size_t GetBytesWritten() const noexcept { return bytes_written_; }
    [[nodiscard]] const std::vector<std::string>& GetExecutedInstructions() const noexcept {
        return executed_instructions_;
    }

    void ClearInstructionLog() noexcept {
        executed_instructions_.clear();
        memory_write_count_ = 0;
        memory_read_count_ = 0;
        bytes_written_ = 0;
        bytes_read_ = 0;
    }

private:
    std::array<uint64_t, static_cast<size_t>(Reg::Count)> registers_{0};
    std::vector<uint8_t> memory_;
    size_t memory_write_count_{0};
    size_t memory_read_count_{0};
    size_t bytes_written_{0};
    size_t bytes_read_{0};
    std::vector<std::string> executed_instructions_;
};

// ============================================================================
// 4. Pixel Buffer Rendering Loop Sim (O0 vs O2 vs Volatile)
// ============================================================================

struct FrameBufferProfile {
    std::string optimization_level;
    bool is_volatile;
    size_t instructions_executed;
    size_t memory_writes;
    size_t total_bytes_written;
    std::vector<std::string> asm_snippet;
};

class PixelLoopDisassemblySimulator {
public:
    static FrameBufferProfile SimulateUnoptimizedO0(size_t pixel_count) {
        // -O0: Every iteration loads 'i' from stack, computes pointer, stores value, increments 'i'
        // Pattern:
        //   mov DWORD PTR [rbp-4], 0
        // .L2:
        //   mov eax, DWORD PTR [rbp-4]
        //   cdqe
        //   mov rdx, QWORD PTR [rbp-16]
        //   add rax, rdx
        //   mov BYTE PTR [rax], 255
        //   add DWORD PTR [rbp-4], 1
        //   cmp DWORD PTR [rbp-4], N
        //   jl .L2
        size_t instrs_per_iter = 8;
        size_t loop_overhead = 2; // initial setup and exit
        return {
            .optimization_level = "-O0 (No Optimization)",
            .is_volatile = false,
            .instructions_executed = loop_overhead + pixel_count * instrs_per_iter,
            .memory_writes = pixel_count, // each pixel written once
            .total_bytes_written = pixel_count * 1,
            .asm_snippet = {
                "mov    DWORD PTR [rbp-4], 0",
                ".L2:",
                "mov    eax, DWORD PTR [rbp-4]",
                "cdqe",
                "mov    rdx, QWORD PTR [rbp-16]",
                "add    rax, rdx",
                "mov    BYTE PTR [rax], 255",
                "add    DWORD PTR [rbp-4], 1",
                "cmp    DWORD PTR [rbp-4], 100",
                "jl     .L2"
            }
        };
    }

    static FrameBufferProfile SimulateOptimizedO2(size_t pixel_count) {
        // -O2: Vectorized with SSE/AVX (16 bytes per store with movups/movaps)
        // plus remainder scalar loop
        size_t vector_size = 16;
        size_t vector_iters = pixel_count / vector_size;
        size_t remainder = pixel_count % vector_size;

        // Setup: movdqa / movaps with 0xFF broadcast
        size_t setup_instrs = 4;
        size_t vector_instrs = vector_iters * 3; // movups, add, cmp/jne
        size_t remainder_instrs = remainder * 3;

        return {
            .optimization_level = "-O2 (SIMD Vectorized & Unrolled)",
            .is_volatile = false,
            .instructions_executed = setup_instrs + vector_instrs + remainder_instrs,
            .memory_writes = vector_iters + remainder,
            .total_bytes_written = pixel_count,
            .asm_snippet = {
                "movdqa xmm0, XMMWORD PTR .LC0[rip]  ; 16 bytes of 0xFF broadcast",
                "xor    eax, eax",
                ".L3:",
                "movups XMMWORD PTR [rdi+rax], xmm0  ; 16 pixels per instruction!",
                "add    rax, 16",
                "cmp    rax, 96",
                "jne    .L3",
                "; Remainder scalar stores for remaining 4 bytes",
                "mov    DWORD PTR [rdi+96], -1"
            }
        };
    }

    static FrameBufferProfile SimulateVolatileO2(size_t pixel_count) {
        // -O2 with volatile: Must emit EXACT individual writes, cannot vectorize across volatile boundary
        // Pattern:
        //   xor    eax, eax
        // .L4:
        //   mov    BYTE PTR [rdi+rax], 255
        //   add    rax, 1
        //   cmp    rax, 100
        //   jne    .L4
        size_t instrs_per_iter = 3;
        size_t setup = 2;
        return {
            .optimization_level = "-O2 with volatile uint8_t*",
            .is_volatile = true,
            .instructions_executed = setup + pixel_count * instrs_per_iter,
            .memory_writes = pixel_count,
            .total_bytes_written = pixel_count * 1,
            .asm_snippet = {
                "xor    eax, eax",
                ".L4:",
                "mov    BYTE PTR [rdi+rax], 255      ; Guaranteed memory write!",
                "add    rax, 1",
                "cmp    rax, 100",
                "jne    .L4"
            }
        };
    }
};

} // namespace mikan::pointer_assembly_sim

// ============================================================================
// Main & Self-Verification Suite
// ============================================================================

int main() {
    using namespace mikan::pointer_assembly_sim;
    std::cout << "============================================================\n";
    std::cout << "[MikanOS Chapter 3.7] Pointer & Assembly Level Simulation\n";
    std::cout << "============================================================\n\n";

    // 1. Data size specifiers
    std::cout << "[1] Verifying Data Size Specifiers:\n";
    assert(DataSizeToSpecifier(DataSize::Byte) == "BYTE PTR");
    assert(DataSizeToSpecifier(DataSize::Word) == "WORD PTR");
    assert(DataSizeToSpecifier(DataSize::Dword) == "DWORD PTR");
    assert(DataSizeToSpecifier(DataSize::Qword) == "QWORD PTR");
    std::cout << "  ✓ All 4 size specifiers verified successfully.\n\n";

    // 2. SIB Addressing calculations
    std::cout << "[2] Verifying SIB (Scale-Index-Base) Calculations:\n";
    SIBAddress sib1{ .base = 0x1000, .index = 10, .scale = 4, .displacement = 16 };
    assert(sib1.IsValidScale());
    assert(sib1.ScaleToEncoding() == 0b10);
    // 0x1000 + (10 * 4) + 16 = 4096 + 40 + 16 = 4152 = 0x1038
    assert(sib1.CalculateEffectiveAddress() == 0x1038);
    std::cout << "  ✓ SIB [0x1000 + 10*4 + 16] = 0x" << std::hex << sib1.CalculateEffectiveAddress() << std::dec << "\n";
    std::cout << "  ✓ Asm format: " << sib1.ToAssemblyString("rdi", "rsi") << "\n\n";

    // 3. Virtual CPU: MOV vs LEA
    std::cout << "[3] Verifying Virtual CPU: MOV (Dereference) vs LEA (Calculation):\n";
    VirtualCpuSimulator cpu(65536);
    cpu.SetRegister(VirtualCpuSimulator::Reg::RAX, 0x2000); // Base pointer
    cpu.SetRegister(VirtualCpuSimulator::Reg::RCX, 5);      // Index i = 5
    cpu.SetRegister(VirtualCpuSimulator::Reg::RDX, 0xCAFEBABE); // Value to store

    SIBAddress mem_op{ .base = 0x2000, .index = 5, .scale = 4, .displacement = 0 };
    // Effective address: 0x2000 + 20 = 0x2014

    // Test LEA: lea rdi, [rax + rcx*4]
    cpu.ExecuteLea(VirtualCpuSimulator::Reg::RDI, mem_op);
    assert(cpu.GetRegister(VirtualCpuSimulator::Reg::RDI) == 0x2014);
    assert(cpu.GetMemoryReadCount() == 0);
    assert(cpu.GetMemoryWriteCount() == 0);
    std::cout << "  ✓ LEA computed effective address: 0x" << std::hex
              << cpu.GetRegister(VirtualCpuSimulator::Reg::RDI) << std::dec << " (NO memory access)\n";

    // Test MOV Store: mov DWORD PTR [rax + rcx*4], edx
    cpu.ExecuteMovStore(mem_op, VirtualCpuSimulator::Reg::RDX, DataSize::Dword);
    assert(cpu.GetMemoryWriteCount() == 1);
    assert(cpu.GetBytesWritten() == 4);

    // Test MOV Load: mov ebx, DWORD PTR [rax + rcx*4]
    cpu.ExecuteMovLoad(VirtualCpuSimulator::Reg::RBX, mem_op, DataSize::Dword);
    assert(cpu.GetRegister(VirtualCpuSimulator::Reg::RBX) == 0xCAFEBABE);
    assert(cpu.GetMemoryReadCount() == 1);
    std::cout << "  ✓ MOV stored and loaded back value: 0x" << std::hex
              << cpu.GetRegister(VirtualCpuSimulator::Reg::RBX) << std::dec << " (2 memory accesses)\n";

    std::cout << "  Executed instructions:\n";
    for (const auto& inst : cpu.GetExecutedInstructions()) {
        std::cout << "    - " << inst << "\n";
    }
    std::cout << "\n";

    // 4. Pixel loop disassembly simulation
    std::cout << "[4] Verifying Pixel Loop Disassembly & Optimization Profiles:\n";
    auto o0 = PixelLoopDisassemblySimulator::SimulateUnoptimizedO0(100);
    auto o2 = PixelLoopDisassemblySimulator::SimulateOptimizedO2(100);
    auto vol = PixelLoopDisassemblySimulator::SimulateVolatileO2(100);

    std::cout << "  Profile 1: " << o0.optimization_level << " -> " << o0.instructions_executed << " instructions\n";
    std::cout << "  Profile 2: " << o2.optimization_level << " -> " << o2.instructions_executed << " instructions (Speedup: "
              << static_cast<double>(o0.instructions_executed) / o2.instructions_executed << "x)\n";
    std::cout << "  Profile 3: " << vol.optimization_level << " -> " << vol.instructions_executed << " instructions (Guaranteed "
              << vol.memory_writes << " writes)\n";

    assert(o2.instructions_executed < o0.instructions_executed);
    assert(vol.memory_writes == 100);

    std::cout << "\n============================================================\n";
    std::cout << "✓ ALL TESTS PASSED SUCCESSFULLY! (Chapter 3.7 Verification OK)\n";
    std::cout << "============================================================\n";
    return 0;
}
