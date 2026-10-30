/**
 * @file registers_sim.cpp
 * @brief C++20 Simulation and Verification of x86-64 Registers Architecture
 *
 * Simulates and verifies MikanOS Chapter 3.2:
 * - 64-bit General-Purpose Registers (RAX..R15)
 * - Sub-register access hierarchies (64/32/16/8-bit)
 * - Crucial 32-bit Zero-Extension Rule (writing 32-bit registers clears upper 32 bits)
 * - Partial preservation when writing 16-bit and 8-bit registers (AH/AL, etc.)
 * - Instruction Pointer (RIP) & Stack Pointer (RSP) LIFO stack operations
 * - RFLAGS register layout and mathematical derivation of status flags (CF, ZF, SF, OF, PF)
 * - Control Registers (CR0, CR2, CR3, CR4, CR8) and MSR (IA32_EFER)
 * - System V AMD64 ABI Calling Convention (argument registers, caller-saved vs callee-saved)
 */

#include <cassert>
#include <cctype>
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

namespace mikan::registers {

// ============================================================================
// 1. x86-64 Register File and Sub-register Aliasing Model
// ============================================================================

enum class RegId {
    RAX, RBX, RCX, RDX,
    RSI, RDI, RBP, RSP,
    R8,  R9,  R10, R11,
    R12, R13, R14, R15,
    RIP,
    COUNT
};

class X86_64RegisterFile {
public:
    uint64_t gpr[16]{0};
    uint64_t rip{0x00100000ULL};

    // Virtual memory buffer for stack operations
    std::unordered_map<uint64_t, uint64_t> memory_words;

    X86_64RegisterFile() {
        // Standard initial stack pointer
        gpr[static_cast<size_t>(RegId::RSP)] = 0x0000000007FE3DC0ULL;
    }

    uint64_t get_rip() const { return rip; }
    void set_rip(uint64_t addr) { rip = addr; }

    uint64_t get_rsp() const { return gpr[static_cast<size_t>(RegId::RSP)]; }
    void set_rsp(uint64_t addr) { gpr[static_cast<size_t>(RegId::RSP)] = addr; }

    // Read full 64-bit register
    uint64_t read_64(RegId reg) const {
        return gpr[static_cast<size_t>(reg)];
    }

    // Write full 64-bit register
    void write_64(RegId reg, uint64_t value) {
        gpr[static_cast<size_t>(reg)] = value;
    }

    // Read 32-bit sub-register (EAX..R15D)
    uint32_t read_32(RegId reg) const {
        return static_cast<uint32_t>(gpr[static_cast<size_t>(reg)] & 0xFFFFFFFFULL);
    }

    /**
     * @brief Write 32-bit sub-register with ZERO-EXTENSION.
     * In x86-64, writes to 32-bit registers (e.g. MOV EAX, 0x12345678)
     * automatically zero-extend to the full 64-bit register, zeroing the upper 32 bits!
     */
    void write_32(RegId reg, uint32_t value) {
        gpr[static_cast<size_t>(reg)] = static_cast<uint64_t>(value);
    }

    // Read 16-bit sub-register (AX..R15W)
    uint16_t read_16(RegId reg) const {
        return static_cast<uint16_t>(gpr[static_cast<size_t>(reg)] & 0xFFFFULL);
    }

    /**
     * @brief Write 16-bit sub-register PRESERVING upper 48 bits.
     * Unlike 32-bit writes, 16-bit writes do NOT zero-extend; the upper 48 bits remain unchanged.
     */
    void write_16(RegId reg, uint16_t value) {
        uint64_t current = gpr[static_cast<size_t>(reg)];
        gpr[static_cast<size_t>(reg)] = (current & ~0xFFFFULL) | static_cast<uint64_t>(value);
    }

    // Read 8-bit low sub-register (AL, BL, CL, DL, SIL, DIL, BPL, SPL, R8B..R15B)
    uint8_t read_8l(RegId reg) const {
        return static_cast<uint8_t>(gpr[static_cast<size_t>(reg)] & 0xFFULL);
    }

    /**
     * @brief Write 8-bit low sub-register PRESERVING upper 56 bits.
     */
    void write_8l(RegId reg, uint8_t value) {
        uint64_t current = gpr[static_cast<size_t>(reg)];
        gpr[static_cast<size_t>(reg)] = (current & ~0xFFULL) | static_cast<uint64_t>(value);
    }

    // Read 8-bit high sub-register (AH, BH, CH, DH - only valid for RAX..RDX)
    uint8_t read_8h(RegId reg) const {
        assert(reg == RegId::RAX || reg == RegId::RBX || reg == RegId::RCX || reg == RegId::RDX);
        return static_cast<uint8_t>((gpr[static_cast<size_t>(reg)] >> 8) & 0xFFULL);
    }

    /**
     * @brief Write 8-bit high sub-register PRESERVING upper 48 bits and lowest 8 bits.
     */
    void write_8h(RegId reg, uint8_t value) {
        assert(reg == RegId::RAX || reg == RegId::RBX || reg == RegId::RCX || reg == RegId::RDX);
        uint64_t current = gpr[static_cast<size_t>(reg)];
        gpr[static_cast<size_t>(reg)] = (current & ~0xFF00ULL) | (static_cast<uint64_t>(value) << 8);
    }

    // Stack operations: push / pop
    void push(uint64_t val) {
        uint64_t& rsp = gpr[static_cast<size_t>(RegId::RSP)];
        rsp -= 8;
        memory_words[rsp] = val;
    }

    uint64_t pop() {
        uint64_t& rsp = gpr[static_cast<size_t>(RegId::RSP)];
        uint64_t val = memory_words[rsp];
        rsp += 8;
        return val;
    }
};

// ============================================================================
// 2. RFLAGS Register Model and Flag Arithmetic Derivation
// ============================================================================

struct RFlags {
    uint64_t raw{0x0000000000000202ULL}; // Bit 1 is reserved and always 1, IF=1

    static constexpr uint64_t CF_MASK = 1ULL << 0;  // Carry Flag
    static constexpr uint64_t RES1_MASK = 1ULL << 1; // Reserved bit 1 (always 1)
    static constexpr uint64_t PF_MASK = 1ULL << 2;  // Parity Flag
    static constexpr uint64_t AF_MASK = 1ULL << 4;  // Auxiliary Carry Flag
    static constexpr uint64_t ZF_MASK = 1ULL << 6;  // Zero Flag
    static constexpr uint64_t SF_MASK = 1ULL << 7;  // Sign Flag
    static constexpr uint64_t TF_MASK = 1ULL << 8;  // Trap Flag
    static constexpr uint64_t IF_MASK = 1ULL << 9;  // Interrupt Enable Flag
    static constexpr uint64_t DF_MASK = 1ULL << 10; // Direction Flag
    static constexpr uint64_t OF_MASK = 1ULL << 11; // Overflow Flag
    static constexpr uint64_t IOPL_MASK = 3ULL << 12; // I/O Privilege Level (bits 12..13)
    static constexpr uint64_t NT_MASK = 1ULL << 14; // Nested Task
    static constexpr uint64_t RF_MASK = 1ULL << 16; // Resume Flag
    static constexpr uint64_t VM_MASK = 1ULL << 17; // Virtual-8086 Mode
    static constexpr uint64_t AC_MASK = 1ULL << 18; // Alignment Check
    static constexpr uint64_t VIF_MASK = 1ULL << 19; // Virtual Interrupt Flag
    static constexpr uint64_t VIP_MASK = 1ULL << 20; // Virtual Interrupt Pending
    static constexpr uint64_t ID_MASK = 1ULL << 21; // Identification Flag

    bool cf() const { return (raw & CF_MASK) != 0; }
    bool pf() const { return (raw & PF_MASK) != 0; }
    bool af() const { return (raw & AF_MASK) != 0; }
    bool zf() const { return (raw & ZF_MASK) != 0; }
    bool sf() const { return (raw & SF_MASK) != 0; }
    bool tf() const { return (raw & TF_MASK) != 0; }
    bool if_flag() const { return (raw & IF_MASK) != 0; }
    bool df() const { return (raw & DF_MASK) != 0; }
    bool of() const { return (raw & OF_MASK) != 0; }
    uint8_t iopl() const { return static_cast<uint8_t>((raw >> 12) & 0x3); }

    void set_cf(bool b) { raw = b ? (raw | CF_MASK) : (raw & ~CF_MASK); }
    void set_zf(bool b) { raw = b ? (raw | ZF_MASK) : (raw & ~ZF_MASK); }
    void set_sf(bool b) { raw = b ? (raw | SF_MASK) : (raw & ~SF_MASK); }
    void set_of(bool b) { raw = b ? (raw | OF_MASK) : (raw & ~OF_MASK); }
    void set_pf(bool b) { raw = b ? (raw | PF_MASK) : (raw & ~PF_MASK); }
    void set_if(bool b) { raw = b ? (raw | IF_MASK) : (raw & ~IF_MASK); }

    // Parity calculation for lowest 8 bits (1 if even number of 1-bits, 0 if odd)
    static bool calculate_parity(uint8_t byte) {
        int ones = 0;
        for (int i = 0; i < 8; ++i) {
            if ((byte >> i) & 1) ++ones;
        }
        return (ones % 2) == 0;
    }

    /**
     * @brief Updates arithmetic status flags for addition (res = a + b).
     * @tparam T uint64_t, uint32_t, uint16_t, or uint8_t
     */
    template <typename T>
    void update_add(T a, T b) {
        using UnsignedT = std::make_unsigned_t<T>;
        using SignedT = std::make_signed_t<T>;

        UnsignedT res = static_cast<UnsignedT>(a + b);
        size_t bit_width = sizeof(T) * 8;
        UnsignedT sign_mask = static_cast<UnsignedT>(1ULL << (bit_width - 1));

        // Carry Flag: unsigned overflow
        set_cf(res < a);

        // Zero Flag: result is 0
        set_zf(res == 0);

        // Sign Flag: sign bit of result is 1
        set_sf((res & sign_mask) != 0);

        // Overflow Flag: signed overflow (positive + positive = negative OR negative + negative = positive)
        SignedT sa = static_cast<SignedT>(a);
        SignedT sb = static_cast<SignedT>(b);
        SignedT sres = static_cast<SignedT>(res);
        set_of(((sa > 0 && sb > 0 && sres < 0) || (sa < 0 && sb < 0 && sres > 0)));

        // Parity Flag: based on lowest byte
        set_pf(calculate_parity(static_cast<uint8_t>(res & 0xFF)));

        // Ensure reserved bit 1 is always set
        raw |= RES1_MASK;
    }

    /**
     * @brief Updates arithmetic status flags for subtraction (res = a - b).
     */
    template <typename T>
    void update_sub(T a, T b) {
        using UnsignedT = std::make_unsigned_t<T>;
        using SignedT = std::make_signed_t<T>;

        UnsignedT res = static_cast<UnsignedT>(a - b);
        size_t bit_width = sizeof(T) * 8;
        UnsignedT sign_mask = static_cast<UnsignedT>(1ULL << (bit_width - 1));

        // Carry Flag: unsigned borrow (a < b)
        set_cf(a < b);

        // Zero Flag: result is 0 (a == b)
        set_zf(res == 0);

        // Sign Flag: sign bit of result is 1
        set_sf((res & sign_mask) != 0);

        // Overflow Flag: signed overflow (pos - neg = neg OR neg - pos = pos)
        SignedT sa = static_cast<SignedT>(a);
        SignedT sb = static_cast<SignedT>(b);
        SignedT sres = static_cast<SignedT>(res);
        set_of(((sa >= 0 && sb < 0 && sres < 0) || (sa < 0 && sb >= 0 && sres >= 0)));

        // Parity Flag: lowest byte
        set_pf(calculate_parity(static_cast<uint8_t>(res & 0xFF)));

        raw |= RES1_MASK;
    }

    std::string format_flags() const {
        std::ostringstream oss;
        oss << "[";
        oss << (cf() ? "CF " : "");
        oss << (pf() ? "PF " : "");
        oss << (af() ? "AF " : "");
        oss << (zf() ? "ZF " : "");
        oss << (sf() ? "SF " : "");
        oss << (if_flag() ? "IF " : "");
        oss << (df() ? "DF " : "");
        oss << (of() ? "OF " : "");
        oss << "IOPL=" << static_cast<int>(iopl()) << "]";
        return oss.str();
    }
};

// ============================================================================
// 3. Control Registers (CR0..CR4, CR8) and MSR Model
// ============================================================================

struct ControlRegisters {
    uint64_t cr0{0x80010033ULL}; // PG=1, WP=1, ET=1, PE=1
    uint64_t cr2{0ULL};         // Page fault linear address
    uint64_t cr3{0x0000000007FA0000ULL}; // PML4 base physical address
    uint64_t cr4{0x00000668ULL}; // PAE=1, OSFXSR=1, OSXMMEXCPT=1
    uint64_t cr8{0ULL};         // Task Priority Register (TPR)
    uint64_t ia32_efer{0x00000D01ULL}; // LME=1, LMA=1, NXE=1, SCE=1

    // CR0 bit tests
    bool is_protected_mode() const { return (cr0 & (1ULL << 0)) != 0; } // PE
    bool is_write_protect() const { return (cr0 & (1ULL << 16)) != 0; } // WP
    bool is_paging_enabled() const { return (cr0 & (1ULL << 31)) != 0; } // PG

    // CR3 PML4 physical base address (bits 12..51)
    uint64_t get_pml4_base_address() const {
        return cr3 & 0x000FFFFFFFFFF000ULL;
    }

    // CR4 bit tests
    bool is_pae_enabled() const { return (cr4 & (1ULL << 5)) != 0; } // PAE
    bool is_osfxsr_enabled() const { return (cr4 & (1ULL << 9)) != 0; } // OSFXSR (SSE support)

    // IA32_EFER bit tests
    bool is_syscall_extension_enabled() const { return (ia32_efer & (1ULL << 0)) != 0; } // SCE
    bool is_long_mode_active() const { return (ia32_efer & (1ULL << 10)) != 0; }        // LMA
    bool is_no_execute_enabled() const { return (ia32_efer & (1ULL << 11)) != 0; }      // NXE
};

// ============================================================================
// 4. System V AMD64 ABI Calling Convention Simulator
// ============================================================================

class SystemVAbiSimulator {
public:
    // First 6 integer/pointer arguments passed in registers
    static constexpr RegId ARG_REGS[6] = {
        RegId::RDI, // arg 0
        RegId::RSI, // arg 1
        RegId::RDX, // arg 2
        RegId::RCX, // arg 3
        RegId::R8,  // arg 4
        RegId::R9   // arg 5
    };

    static bool is_callee_saved(RegId reg) {
        return reg == RegId::RBX || reg == RegId::RSP || reg == RegId::RBP ||
               reg == RegId::R12 || reg == RegId::R13 || reg == RegId::R14 || reg == RegId::R15;
    }

    static bool is_caller_saved(RegId reg) {
        return reg == RegId::RAX || reg == RegId::RCX || reg == RegId::RDX ||
               reg == RegId::RSI || reg == RegId::RDI || reg == RegId::R8 ||
               reg == RegId::R9  || reg == RegId::R10 || reg == RegId::R11;
    }

    /**
     * @brief Prepare register state for a function call with given arguments.
     * Arguments 0..5 placed in RDI..R9; arguments 6..N pushed on stack.
     */
    static void prepare_call(X86_64RegisterFile& rf, const std::vector<uint64_t>& args) {
        for (size_t i = 0; i < args.size(); ++i) {
            if (i < 6) {
                rf.write_64(ARG_REGS[i], args[i]);
            } else {
                // Stack arguments are pushed in reverse order (System V ABI)
                // For demonstration, push sequentially
                rf.push(args[i]);
            }
        }
    }
};

} // namespace mikan::registers

// ============================================================================
// Test Suite
// ============================================================================

int run_tests() {
    using namespace mikan::registers;

    std::cout << "[TEST] 1. x86-64 Sub-register Zero-Extension Rule..." << std::endl;
    {
        X86_64RegisterFile rf;
        rf.write_64(RegId::RAX, 0xFFFFFFFFFFFFFFFFULL);
        assert(rf.read_64(RegId::RAX) == 0xFFFFFFFFFFFFFFFFULL);

        // Write 32-bit: MUST ZERO-EXTEND to upper 32 bits!
        rf.write_32(RegId::RAX, 0x12345678U);
        assert(rf.read_32(RegId::RAX) == 0x12345678U);
        assert(rf.read_64(RegId::RAX) == 0x0000000012345678ULL); // Upper 32 bits MUST BE ZERO

        // Write 16-bit: MUST PRESERVE upper 48 bits!
        rf.write_64(RegId::RAX, 0x1122334455667788ULL);
        rf.write_16(RegId::RAX, 0xAA55U);
        assert(rf.read_64(RegId::RAX) == 0x112233445566AA55ULL);

        // Write 8-bit low (AL): MUST PRESERVE upper 56 bits!
        rf.write_8l(RegId::RAX, 0x99U);
        assert(rf.read_64(RegId::RAX) == 0x112233445566AA99ULL);

        // Write 8-bit high (AH): MUST PRESERVE upper 48 bits and lowest 8 bits!
        rf.write_8h(RegId::RAX, 0x42U);
        assert(rf.read_64(RegId::RAX) == 0x1122334455664299ULL);
    }
    std::cout << "  -> PASSED: 32-bit zero-extension and partial sub-register preservation verified." << std::endl;

    std::cout << "[TEST] 2. Stack LIFO and 8-byte alignment..." << std::endl;
    {
        X86_64RegisterFile rf;
        uint64_t initial_rsp = 0x07FE3DC0ULL;
        rf.set_rsp(initial_rsp);

        rf.push(0xDEADBEEFCAFEBABEULL);
        assert(rf.get_rsp() == initial_rsp - 8);

        rf.push(0x1122334455667788ULL);
        assert(rf.get_rsp() == initial_rsp - 16);

        uint64_t val1 = rf.pop();
        assert(val1 == 0x1122334455667788ULL);
        assert(rf.get_rsp() == initial_rsp - 8);

        uint64_t val2 = rf.pop();
        assert(val2 == 0xDEADBEEFCAFEBABEULL);
        assert(rf.get_rsp() == initial_rsp);
    }
    std::cout << "  -> PASSED: Stack push/pop LIFO verified." << std::endl;

    std::cout << "[TEST] 3. RFLAGS Status Flag Calculation..." << std::endl;
    {
        RFlags flags;

        // Zero result: 5 - 5 = 0
        flags.update_sub<uint64_t>(5, 5);
        assert(flags.zf() == true);
        assert(flags.cf() == false);
        assert(flags.sf() == false);
        assert(flags.of() == false);

        // Carry and sign: 3 - 5
        flags.update_sub<uint64_t>(3, 5);
        assert(flags.zf() == false);
        assert(flags.cf() == true);  // Unsigned borrow
        assert(flags.sf() == true);  // Result is 0xFFFFFFFFFFFFFFFE (MSB=1)
        assert(flags.of() == false); // Signed 3 - 5 = -2 (no signed overflow)

        // Signed overflow in 32-bit: 0x7FFFFFFF + 1
        flags.update_add<uint32_t>(0x7FFFFFFFU, 1U);
        assert(flags.of() == true);  // Pos + Pos = Neg
        assert(flags.sf() == true);  // MSB is 1
        assert(flags.cf() == false); // No unsigned carry out of 32-bit
        assert(flags.zf() == false);

        // Reserved bit 1
        assert((flags.raw & 0x2) != 0);
    }
    std::cout << "  -> PASSED: RFLAGS status flags arithmetic verified." << std::endl;

    std::cout << "[TEST] 4. Control Registers & MSR..." << std::endl;
    {
        ControlRegisters cr;
        assert(cr.is_paging_enabled() == true);
        assert(cr.is_protected_mode() == true);
        assert(cr.is_write_protect() == true);
        assert(cr.is_pae_enabled() == true);
        assert(cr.is_long_mode_active() == true);
        assert(cr.is_no_execute_enabled() == true);
        assert(cr.get_pml4_base_address() == 0x0000000007FA0000ULL);
    }
    std::cout << "  -> PASSED: Control registers and MSR flags verified." << std::endl;

    std::cout << "[TEST] 5. System V AMD64 ABI Calling Convention..." << std::endl;
    {
        X86_64RegisterFile rf;
        std::vector<uint64_t> args = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70};
        SystemVAbiSimulator::prepare_call(rf, args);

        assert(rf.read_64(RegId::RDI) == 0x10);
        assert(rf.read_64(RegId::RSI) == 0x20);
        assert(rf.read_64(RegId::RDX) == 0x30);
        assert(rf.read_64(RegId::RCX) == 0x40);
        assert(rf.read_64(RegId::R8)  == 0x50);
        assert(rf.read_64(RegId::R9)  == 0x60);

        // 7th arg was pushed on stack
        assert(rf.pop() == 0x70);

        // Caller/Callee saved checks
        assert(SystemVAbiSimulator::is_caller_saved(RegId::RAX) == true);
        assert(SystemVAbiSimulator::is_caller_saved(RegId::RDI) == true);
        assert(SystemVAbiSimulator::is_callee_saved(RegId::RBX) == true);
        assert(SystemVAbiSimulator::is_callee_saved(RegId::RBP) == true);
        assert(SystemVAbiSimulator::is_callee_saved(RegId::R12) == true);
    }
    std::cout << "  -> PASSED: System V ABI argument passing verified." << std::endl;

    std::cout << "\n>>> ALL REGISTER SIMULATION TESTS PASSED SUCCESSFULLY! <<<\n" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string_view(argv[1]) == "--test") {
        return run_tests();
    }

    std::cout << "=== x86-64 Architecture Register Simulator (MikanOS Chapter 3.2) ===" << std::endl;
    mikan::registers::X86_64RegisterFile rf;
    rf.write_64(mikan::registers::RegId::RAX, 0xFFFFFFFFFFFFFFFFULL);
    std::cout << "Initial RAX: 0x" << std::hex << rf.read_64(mikan::registers::RegId::RAX) << std::endl;

    rf.write_32(mikan::registers::RegId::RAX, 0x12345678U);
    std::cout << "After MOV EAX, 0x12345678 (Zero-Extension Rule):" << std::endl;
    std::cout << "  EAX: 0x" << std::hex << rf.read_32(mikan::registers::RegId::RAX) << std::endl;
    std::cout << "  RAX: 0x" << std::hex << std::setfill('0') << std::setw(16)
              << rf.read_64(mikan::registers::RegId::RAX) << " (Upper 32 bits cleared!)" << std::endl;

    mikan::registers::RFlags flags;
    flags.update_sub<uint64_t>(100, 100);
    std::cout << "\nFlags after (100 - 100): " << flags.format_flags() << std::endl;

    return 0;
}
