/**
 * @file interrupt_vector_sim.cpp
 * @brief C++20 Standalone Verification of x86-64 Interrupt Vectors and Exceptions
 *        (MikanOS Chapter 7.3 / osbook_day07a)
 *
 * This program verifies:
 * 1. x86-64 Interrupt Vector space partitioning: 256 vectors total (0x00 to 0xFF)
 *    - Vectors 0x00-0x1F (0-31): CPU Architecture-defined exceptions & interrupts
 *    - Vectors 0x20-0xFF (32-255): User-defined / peripheral hardware interrupts
 * 2. CPU Exception classification: Fault vs Trap vs Abort vs Interrupt semantics
 *    - Fault: Instruction restartable (saved RIP = faulting instruction address)
 *    - Trap: Instruction complete notification (saved RIP = next instruction address)
 *    - Abort: Unrecoverable hardware/system disaster (saved RIP unpredictable)
 * 3. Hardware ErrorCode stack semantics:
 *    - Vectors 8 (#DF), 10 (#TS), 11 (#NP), 12 (#SS), 13 (#GP), 14 (#PF), 17 (#AC), 21 (#CP)
 *    - Stack alignment with error code: 48 bytes (6 quadwords) vs 40 bytes (5 quadwords)
 * 4. MikanOS InterruptVector class design:
 *    - Type-safe namespacing: InterruptVector::kXHCI = 0x40 (64)
 *    - Selection rationale: Avoiding reserved exception conflicts and future system interrupts
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic interrupt_vector_sim.cpp -o interrupt_vector_sim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <optional>
#include <algorithm>

// ============================================================================
// 1. MikanOS Interrupt Vector Definition (kernel/interrupt.hpp)
// ============================================================================

/**
 * @brief Interrupt vector definitions in MikanOS kernel
 */
class InterruptVector {
 public:
  enum Number : uint8_t {
    kXHCI = 0x40,  // Vector 64: USB 3.0 xHCI Host Controller MSI Interrupt
  };
};

static_assert(InterruptVector::kXHCI == 0x40, "InterruptVector::kXHCI must be 0x40 (64)");

// ============================================================================
// 2. x86-64 Exception Classification and Specifications
// ============================================================================

enum class ExceptionType {
    Fault,      // Restartable; saved RIP points to faulting instruction (#PF, #GP, #DE)
    Trap,       // Notification; saved RIP points to next instruction (#BP, #OF)
    Abort,      // Non-restartable severe failure; RIP unpredictable (#DF, #MC)
    Interrupt,  // External asynchronous interrupt; saved RIP points to next instruction
};

constexpr std::string_view ExceptionTypeToString(ExceptionType type) noexcept {
    switch (type) {
        case ExceptionType::Fault: return "Fault (命令再実行型)";
        case ExceptionType::Trap: return "Trap (命令完了通知型)";
        case ExceptionType::Abort: return "Abort (回復不能障害型)";
        case ExceptionType::Interrupt: return "Interrupt (外部非同期割込)";
    }
    return "Unknown";
}

struct ExceptionSpec {
    uint8_t vector;
    std::string_view mnemonic;
    std::string_view name;
    ExceptionType type;
    bool has_error_code;
    std::string_view description;
};

// Complete table of Intel/AMD x86-64 CPU architecture exceptions
constexpr std::array<ExceptionSpec, 22> kCpuExceptions = {{
    { 0, "#DE", "Divide Error",               ExceptionType::Fault,     false, "DIV/IDIV ゼロ除算または商オーバーフロー" },
    { 1, "#DB", "Debug",                      ExceptionType::Fault,     false, "ハードウェアブレークポイント / TF=1" },
    { 2, "NMI", "Non-Maskable Interrupt",     ExceptionType::Interrupt, false, "マスク不可能外部割り込み (パリティ/ECCエラー)" },
    { 3, "#BP", "Breakpoint",                 ExceptionType::Trap,      false, "INT 3 (0xCC) ソフトウェアブレークポイント" },
    { 4, "#OF", "Overflow",                   ExceptionType::Trap,      false, "INTO 命令時の OF=1" },
    { 5, "#BR", "BOUND Range Exceeded",       ExceptionType::Fault,     false, "BOUND 命令配列境界超過 (32bitレガシー)" },
    { 6, "#UD", "Invalid Opcode",             ExceptionType::Fault,     false, "未定義機械語命令 / 無効プリフィックス" },
    { 7, "#NM", "Device Not Available",       ExceptionType::Fault,     false, "CR0.TS=1 時の FPU/SSE 命令実行" },
    { 8, "#DF", "Double Fault",               ExceptionType::Abort,     true,  "例外処理中の追加例外 (ErrorCode=0)" },
    { 9, "#MF", "Coprocessor Segment Overrun",ExceptionType::Abort,     false, "FPU セグメント超過 (レガシー)" },
    {10, "#TS", "Invalid TSS",                ExceptionType::Fault,     true,  "TSS セグメント記述子が無効" },
    {11, "#NP", "Segment Not Present",        ExceptionType::Fault,     true,  "セグメント記述子の Present ビットが 0" },
    {12, "#SS", "Stack-Segment Fault",        ExceptionType::Fault,     true,  "スタック制限超過または SS 不在" },
    {13, "#GP", "General Protection Fault",   ExceptionType::Fault,     true,  "特権違反、未整列メモリアクセス (SSE/AVX) 等" },
    {14, "#PF", "Page Fault",                 ExceptionType::Fault,     true,  "ページ不在または権限違反 (CR2=障害アドレス)" },
    {15, "#RSVD","Intel Reserved",            ExceptionType::Fault,     false, "予約領域" },
    {16, "#MF", "x87 FPU Floating-Point",     ExceptionType::Fault,     false, "x87 浮動小数点エラー" },
    {17, "#AC", "Alignment Check",            ExceptionType::Fault,     true,  "未整列メモリアクセス (CR0.AM=1 & EFLAGS.AC=1)" },
    {18, "#MC", "Machine Check",              ExceptionType::Abort,     false, "内部バスエラー / ハードウェア障害" },
    {19, "#XM", "SIMD Floating-Point",        ExceptionType::Fault,     false, "SSE/AVX 浮動小数点例外 (MXCSR)" },
    {20, "#VE", "Virtualization Exception",   ExceptionType::Fault,     false, "EPT 違反による仮想化例外" },
    {21, "#CP", "Control Protection",         ExceptionType::Fault,     true,  "CET シャドウスタック保護違反" },
}};

// ============================================================================
// 3. Hardware Stack Frame Layout Verification (with and without ErrorCode)
// ============================================================================

/**
 * @brief Standard 40-byte Interrupt Frame (pushed by CPU for hardware interrupts and exceptions without error code)
 */
struct StandardInterruptFrame {
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
};
static_assert(sizeof(StandardInterruptFrame) == 40, "StandardInterruptFrame size must be 40 bytes");
static_assert(offsetof(StandardInterruptFrame, rip) == 0);
static_assert(offsetof(StandardInterruptFrame, cs) == 8);
static_assert(offsetof(StandardInterruptFrame, rflags) == 16);
static_assert(offsetof(StandardInterruptFrame, rsp) == 24);
static_assert(offsetof(StandardInterruptFrame, ss) == 32);

/**
 * @brief 48-byte Interrupt Frame with ErrorCode (pushed by CPU for #GP, #PF, #DF, etc.)
 *
 * Notice that ErrorCode is pushed AFTER RIP, so it resides at [RSP + 0]!
 */
struct InterruptFrameWithErrorCode {
    uint64_t error_code;
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
};
static_assert(sizeof(InterruptFrameWithErrorCode) == 48, "InterruptFrameWithErrorCode size must be 48 bytes");
static_assert(offsetof(InterruptFrameWithErrorCode, error_code) == 0);
static_assert(offsetof(InterruptFrameWithErrorCode, rip) == 8);
static_assert(offsetof(InterruptFrameWithErrorCode, cs) == 16);
static_assert(offsetof(InterruptFrameWithErrorCode, rflags) == 24);
static_assert(offsetof(InterruptFrameWithErrorCode, rsp) == 32);
static_assert(offsetof(InterruptFrameWithErrorCode, ss) == 40);

// ============================================================================
// 4. Vector Dispatch Simulator
// ============================================================================

struct DispatchResult {
    uint8_t vector;
    std::string_view category;
    ExceptionType type;
    bool has_error_code;
    std::optional<uint64_t> error_code_val;
    uint64_t saved_rip;
    std::optional<uint64_t> return_address;
    size_t stack_bytes_pushed;
};

class VectorDispatcher {
 public:
    static bool IsCpuException(uint8_t vector) noexcept {
        return vector < 32;
    }

    static bool IsUserDefined(uint8_t vector) noexcept {
        return vector >= 32;
    }

    static std::optional<ExceptionSpec> FindExceptionSpec(uint8_t vector) noexcept {
        if (!IsCpuException(vector)) return std::nullopt;
        for (const auto& spec : kCpuExceptions) {
            if (spec.vector == vector) return spec;
        }
        return std::nullopt;
    }

    static DispatchResult Simulate(uint8_t vector, uint64_t current_rip, uint64_t next_rip, uint64_t error_code = 0) {
        if (IsCpuException(vector)) {
            auto spec_opt = FindExceptionSpec(vector);
            ExceptionType type = spec_opt ? spec_opt->type : ExceptionType::Fault;
            bool has_ec = spec_opt ? spec_opt->has_error_code : false;

            uint64_t saved_rip = (type == ExceptionType::Fault) ? current_rip : next_rip;
            std::optional<uint64_t> return_addr;
            if (type == ExceptionType::Fault) {
                return_addr = current_rip;  // Re-execute faulting instruction after fix
            } else if (type == ExceptionType::Trap) {
                return_addr = next_rip;     // Continue to next instruction
            } else { // Abort
                return_addr = std::nullopt; // System halt / Panic
            }

            return DispatchResult{
                .vector = vector,
                .category = "CPU Architecture Exception (0x00-0x1F)",
                .type = type,
                .has_error_code = has_ec,
                .error_code_val = has_ec ? std::make_optional(error_code) : std::nullopt,
                .saved_rip = saved_rip,
                .return_address = return_addr,
                .stack_bytes_pushed = has_ec ? sizeof(InterruptFrameWithErrorCode) : sizeof(StandardInterruptFrame),
            };
        } else {
            // User-defined / peripheral hardware interrupt
            return DispatchResult{
                .vector = vector,
                .category = (vector == InterruptVector::kXHCI) ? "MikanOS xHCI MSI Interrupt (0x40)"
                                                               : "User-Defined Interrupt (0x20-0xFF)",
                .type = ExceptionType::Interrupt,
                .has_error_code = false,
                .error_code_val = std::nullopt,
                .saved_rip = next_rip,
                .return_address = next_rip,
                .stack_bytes_pushed = sizeof(StandardInterruptFrame),
            };
        }
    }
};

// ============================================================================
// 5. Standalone Tests & Verification Runner
// ============================================================================

void RunVectorSpaceTests() {
    std::cout << "[Test 1] Verifying Vector Space Partitioning & Constants...\n";
    assert(VectorDispatcher::IsCpuException(0));
    assert(VectorDispatcher::IsCpuException(31));
    assert(!VectorDispatcher::IsCpuException(32));
    assert(!VectorDispatcher::IsCpuException(255));

    assert(VectorDispatcher::IsUserDefined(32));
    assert(VectorDispatcher::IsUserDefined(0x40));
    assert(VectorDispatcher::IsUserDefined(255));
    assert(!VectorDispatcher::IsUserDefined(31));

    assert(InterruptVector::kXHCI == 64);
    assert(InterruptVector::kXHCI == 0x40);
    std::cout << "  ✓ Vector 0-31 classified as CPU Exceptions.\n";
    std::cout << "  ✓ Vector 32-255 classified as User Defined.\n";
    std::cout << "  ✓ MikanOS InterruptVector::kXHCI == 0x40 (64) verified.\n\n";
}

void RunExceptionSpecsTests() {
    std::cout << "[Test 2] Verifying CPU Exception Specifications...\n";

    // #DE: Vector 0, Fault, No error code
    auto de = VectorDispatcher::FindExceptionSpec(0);
    assert(de.has_value());
    assert(de->mnemonic == "#DE");
    assert(de->type == ExceptionType::Fault);
    assert(!de->has_error_code);

    // #BP: Vector 3, Trap, No error code
    auto bp = VectorDispatcher::FindExceptionSpec(3);
    assert(bp.has_value());
    assert(bp->mnemonic == "#BP");
    assert(bp->type == ExceptionType::Trap);
    assert(!bp->has_error_code);

    // #DF: Vector 8, Abort, With error code
    auto df = VectorDispatcher::FindExceptionSpec(8);
    assert(df.has_value());
    assert(df->mnemonic == "#DF");
    assert(df->type == ExceptionType::Abort);
    assert(df->has_error_code);

    // #GP: Vector 13, Fault, With error code
    auto gp = VectorDispatcher::FindExceptionSpec(13);
    assert(gp.has_value());
    assert(gp->mnemonic == "#GP");
    assert(gp->type == ExceptionType::Fault);
    assert(gp->has_error_code);

    // #PF: Vector 14, Fault, With error code
    auto pf = VectorDispatcher::FindExceptionSpec(14);
    assert(pf.has_value());
    assert(pf->mnemonic == "#PF");
    assert(pf->type == ExceptionType::Fault);
    assert(pf->has_error_code);

    std::cout << "  ✓ #DE (0), #BP (3), #DF (8), #GP (13), #PF (14) specs match Intel SDM Vol.3A.\n\n";
}

void RunDispatchSimulationTests() {
    std::cout << "[Test 3] Verifying Dispatch Simulation for Fault / Trap / Abort / Hardware Interrupt...\n";

    const uint64_t current_rip = 0x100020;  // Address of instruction triggering interrupt
    const uint64_t next_rip    = 0x100025;  // Address of next sequential instruction

    // 1. Page Fault (#PF, vector 14): Fault with ErrorCode
    auto res_pf = VectorDispatcher::Simulate(14, current_rip, next_rip, 0x02); // Write fault
    assert(res_pf.type == ExceptionType::Fault);
    assert(res_pf.has_error_code);
    assert(res_pf.error_code_val == 0x02);
    assert(res_pf.saved_rip == current_rip); // Fault re-executes instruction
    assert(res_pf.return_address == current_rip);
    assert(res_pf.stack_bytes_pushed == 48);

    // 2. Breakpoint (#BP, vector 3): Trap without ErrorCode
    auto res_bp = VectorDispatcher::Simulate(3, current_rip, next_rip);
    assert(res_bp.type == ExceptionType::Trap);
    assert(!res_bp.has_error_code);
    assert(res_bp.saved_rip == next_rip); // Trap continues to next instruction
    assert(res_bp.return_address == next_rip);
    assert(res_bp.stack_bytes_pushed == 40);

    // 3. Double Fault (#DF, vector 8): Abort with ErrorCode
    auto res_df = VectorDispatcher::Simulate(8, current_rip, next_rip, 0x00);
    assert(res_df.type == ExceptionType::Abort);
    assert(res_df.has_error_code);
    assert(!res_df.return_address.has_value()); // Abort cannot recover
    assert(res_df.stack_bytes_pushed == 48);

    // 4. MikanOS xHCI MSI Interrupt (vector 0x40): External Interrupt without ErrorCode
    auto res_xhci = VectorDispatcher::Simulate(InterruptVector::kXHCI, current_rip, next_rip);
    assert(res_xhci.type == ExceptionType::Interrupt);
    assert(!res_xhci.has_error_code);
    assert(res_xhci.saved_rip == next_rip); // Resumes next instruction
    assert(res_xhci.return_address == next_rip);
    assert(res_xhci.stack_bytes_pushed == 40);

    std::cout << "  ✓ Fault preserves current RIP (0x" << std::hex << current_rip << ") for re-execution.\n";
    std::cout << "  ✓ Trap & Hardware Interrupt save next RIP (0x" << std::hex << next_rip << ") for normal progression.\n";
    std::cout << "  ✓ Abort safely identifies system unrecoverability.\n";
    std::cout << "  ✓ Stack size difference correctly calculated: 40 bytes (standard) vs 48 bytes (with ErrorCode).\n\n" << std::dec;
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  MikanOS Chapter 7.3: x86-64 Interrupt Vectors C++20 Verification\n";
    std::cout << "==================================================================\n\n";

    RunVectorSpaceTests();
    RunExceptionSpecsTests();
    RunDispatchSimulationTests();

    std::cout << "------------------------------------------------------------------\n";
    std::cout << "Summary Table of Key Vectors & MikanOS Interrupt Architecture:\n";
    std::cout << "------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(8) << "Vector"
              << std::setw(10) << "Symbol"
              << std::setw(26) << "Name"
              << std::setw(22) << "Type"
              << std::setw(12) << "ErrorCode"
              << "Role in MikanOS\n";
    std::cout << std::string(90, '-') << "\n";

    const std::array<uint8_t, 8> sample_vectors = {0, 3, 6, 8, 13, 14, 0x40, 0x20};
    for (uint8_t vec : sample_vectors) {
        std::cout << "0x" << std::hex << std::uppercase << std::setfill('0') << std::setw(2) << static_cast<int>(vec)
                  << std::dec << std::setfill(' ') << "   ";

        if (VectorDispatcher::IsCpuException(vec)) {
            auto spec = VectorDispatcher::FindExceptionSpec(vec);
            if (spec) {
                std::cout << std::left << std::setw(10) << spec->mnemonic
                          << std::setw(26) << spec->name
                          << std::setw(22) << ExceptionTypeToString(spec->type)
                          << std::setw(12) << (spec->has_error_code ? "あり" : "なし");
                if (vec == 13) std::cout << "後章でハンドラ登録 (#GP)\n";
                else if (vec == 14) std::cout << "第20章/27章でハンドラ登録 (#PF)\n";
                else std::cout << "ハードウェア例外\n";
            }
        } else if (vec == InterruptVector::kXHCI) {
            std::cout << std::left << std::setw(10) << "kXHCI"
                      << std::setw(26) << "xHCI USB Host Controller"
                      << std::setw(22) << ExceptionTypeToString(ExceptionType::Interrupt)
                      << std::setw(12) << "なし"
                      << "★ 第7章採用 (MSI割り込み)\n";
        } else {
            std::cout << std::left << std::setw(10) << "User"
                      << std::setw(26) << "User-Defined IRQ"
                      << std::setw(22) << ExceptionTypeToString(ExceptionType::Interrupt)
                      << std::setw(12) << "なし"
                      << "自由利用可能領域\n";
        }
    }

    std::cout << "==================================================================\n";
    std::cout << "All C++20 Interrupt Vector tests PASSED successfully!\n";
    std::cout << "==================================================================\n";

    return 0;
}
