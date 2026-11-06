/**
 * @file kernel_stack_sim.cpp
 * @brief Chapter 8.4 Kernel Stack Switching & System V AMD64 ABI Alignment Simulator
 *
 * MikanOS Chapter 8.4: Switching Kernel Stack (osbook_day08b)
 * Demonstrates:
 *  1. Danger of running kernel on UEFI-allocated stack after ExitBootServices.
 *  2. 1MiB dedicated kernel stack in .bss (alignas(16) uint8_t kernel_main_stack[1024 * 1024]).
 *  3. x86-64 Full Descending Stack arithmetics: RSP_init = stack_base + 1024 * 1024.
 *  4. System V AMD64 ABI 16-byte stack alignment invariants:
 *     - At `call` instruction: RSP % 16 == 0
 *     - Function entry (after 8-byte return address push): RSP % 16 == 8
 *     - After prologue `push rbp`: RSP % 16 == 0
 *  5. Transparent argument forwarding in RDI / RSI registers across stack switch.
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <cstring>

namespace mikanos {

constexpr size_t kKernelStackSize = 1024 * 1024; // 1 MiB
constexpr size_t kRequiredAlignment = 16;         // 16-byte alignment required by System V ABI

/**
 * @brief Simulated Kernel BSS Stack Buffer
 * Mirrors `alignas(16) uint8_t kernel_main_stack[1024 * 1024];` in MikanOS kernel/main.cpp
 */
alignas(16) uint8_t kernel_main_stack[kKernelStackSize];

/**
 * @brief Simulated UEFI Stack Buffer (for comparison and danger demonstration)
 */
alignas(16) uint8_t uefi_boot_stack[64 * 1024]; // 64 KiB typical UEFI firmware stack

/**
 * @brief Virtual CPU State simulating x86-64 registers during stack switch
 */
struct CpuState {
    uint64_t rip{0};
    uint64_t rsp{0};
    uint64_t rbp{0};
    uint64_t rdi{0}; // 1st argument (FrameBufferConfig pointer/ref)
    uint64_t rsi{0}; // 2nd argument (MemoryMap pointer/ref)
    uint64_t rax{0};
    bool halted{false};
};

/**
 * @brief Simulator for x86-64 Full Descending Stack and ABI Alignment Checks
 */
class KernelStackSimulator {
public:
    KernelStackSimulator() {
        std::memset(kernel_main_stack, 0xCC, kKernelStackSize); // Fill with 0xCC (INT 3 pattern)
        std::memset(uefi_boot_stack, 0xAA, sizeof(uefi_boot_stack));
    }

    static uintptr_t get_kernel_stack_base() {
        return reinterpret_cast<uintptr_t>(kernel_main_stack);
    }

    static uintptr_t get_kernel_stack_top() {
        return get_kernel_stack_base() + kKernelStackSize;
    }

    static uintptr_t get_uefi_stack_base() {
        return reinterpret_cast<uintptr_t>(uefi_boot_stack);
    }

    static uintptr_t get_uefi_stack_top() {
        return get_uefi_stack_base() + sizeof(uefi_boot_stack);
    }

    /**
     * @brief Verify that the allocated stack buffer satisfies the 16-byte alignment requirement.
     */
    static bool verify_stack_alignment() {
        uintptr_t base = get_kernel_stack_base();
        uintptr_t top = get_kernel_stack_top();
        return (base % kRequiredAlignment == 0) && (top % kRequiredAlignment == 0);
    }

    /**
     * @brief Check whether a given RSP satisfies the System V ABI alignment rule at `call`.
     */
    static bool is_valid_pre_call_rsp(uint64_t rsp) {
        return (rsp % kRequiredAlignment) == 0;
    }

    /**
     * @brief Check whether a given RSP satisfies the System V ABI alignment rule at function entry.
     */
    static bool is_valid_func_entry_rsp(uint64_t rsp) {
        return (rsp % kRequiredAlignment) == 8;
    }

    /**
     * @brief Simulates execution of KernelMain in asmfunc.asm:
     *
     *   KernelMain:
     *       mov rsp, kernel_main_stack + 1024 * 1024
     *       call KernelMainNewStack
     *   .fin:
     *       hlt
     *       jmp .fin
     */
    CpuState simulate_asm_stack_switch(uint64_t initial_rdi, uint64_t initial_rsi) {
        CpuState cpu;
        // Step 0: Initially running on UEFI firmware stack
        cpu.rsp = get_uefi_stack_top() - 8;
        cpu.rdi = initial_rdi;
        cpu.rsi = initial_rsi;
        cpu.rip = 0x100000; // Fake KernelMain entry address

        // Step 1: mov rsp, kernel_main_stack + 1024 * 1024
        uint64_t new_rsp = static_cast<uint64_t>(get_kernel_stack_top());
        cpu.rsp = new_rsp;

        // Invariant check: Pre-call RSP MUST be 16-byte aligned
        if (!is_valid_pre_call_rsp(cpu.rsp)) {
            throw std::runtime_error("System V ABI Violation: Pre-call RSP is not 16-byte aligned!");
        }

        // Step 2: call KernelMainNewStack
        // `call` pushes 8-byte return address (.fin address) onto stack
        uint64_t return_address = cpu.rip + 12; // address of .fin
        cpu.rsp -= 8;
        *reinterpret_cast<uint64_t*>(cpu.rsp) = return_address;

        // Invariant check: Function entry RSP MUST satisfy RSP % 16 == 8
        if (!is_valid_func_entry_rsp(cpu.rsp)) {
            throw std::runtime_error("System V ABI Violation: Function entry RSP is not 16n + 8 aligned!");
        }

        // Step 3: Verify arguments are intact (RDI and RSI preserved)
        if (cpu.rdi != initial_rdi || cpu.rsi != initial_rsi) {
            throw std::runtime_error("ABI Violation: Argument registers corrupted during stack switch!");
        }

        // Step 4: Simulate KernelMainNewStack execution and return
        // KernelMainNewStack function prologue:
        // push rbp
        cpu.rsp -= 8;
        *reinterpret_cast<uint64_t*>(cpu.rsp) = cpu.rbp;
        cpu.rbp = cpu.rsp;

        // After push rbp, RSP % 16 == 0 again (suitable for local variable alignment & SSE)
        if (!is_valid_pre_call_rsp(cpu.rsp)) {
            throw std::runtime_error("ABI Violation: Post-prologue RSP is not 16-byte aligned!");
        }

        // Function epilogue:
        // pop rbp
        cpu.rbp = *reinterpret_cast<uint64_t*>(cpu.rsp);
        cpu.rsp += 8;

        // ret: pop rip
        uint64_t popped_ret = *reinterpret_cast<uint64_t*>(cpu.rsp);
        cpu.rsp += 8;
        cpu.rip = popped_ret;

        // Step 5: CPU reaches .fin: hlt; jmp .fin
        if (cpu.rip == return_address) {
            cpu.halted = true;
        }

        return cpu;
    }

    /**
     * @brief Simulates stack overflow detection on a 1MiB stack.
     */
    bool test_stack_depth(size_t frames) {
        uint64_t cur_rsp = get_kernel_stack_top();
        uint64_t bottom = get_kernel_stack_base();

        for (size_t i = 0; i < frames; ++i) {
            if (cur_rsp < bottom + 64) {
                // Approaching stack bottom (stack overflow condition)
                return false;
            }
            cur_rsp -= 64; // e.g., 64 bytes per frame
        }
        return true;
    }
};

} // namespace mikanos

void run_tests() {
    using namespace mikanos;
    KernelStackSimulator sim;

    std::cout << "[INFO] Running Chapter 8.4 Kernel Stack Simulator Unit Tests...\n";

    // Test 1: Stack Buffer 16-byte Alignment
    assert(KernelStackSimulator::verify_stack_alignment());
    std::cout << "  ✓ Test 1: kernel_main_stack 16-byte alignment verified.\n";

    // Test 2: Stack size is exactly 1MiB
    assert(kKernelStackSize == 1024 * 1024);
    assert(KernelStackSimulator::get_kernel_stack_top() - KernelStackSimulator::get_kernel_stack_base() == kKernelStackSize);
    std::cout << "  ✓ Test 2: Kernel stack size (1,048,576 bytes = 1 MiB) verified.\n";

    // Test 3: Stack switch simulation & ABI Alignment Invariants
    uint64_t dummy_fb = 0xFFFF'8000'0000'1000ULL;
    uint64_t dummy_map = 0xFFFF'8000'0000'2000ULL;
    CpuState final_cpu = sim.simulate_asm_stack_switch(dummy_fb, dummy_map);

    assert(final_cpu.rdi == dummy_fb);
    assert(final_cpu.rsi == dummy_map);
    assert(final_cpu.halted == true);
    assert(final_cpu.rsp == KernelStackSimulator::get_kernel_stack_top());
    std::cout << "  ✓ Test 3: Stack switch execution, argument preservation, and .fin HLT verified.\n";

    // Test 4: Deep recursion / stack depth tolerance
    // 1MiB can safely hold 16,384 frames of 64 bytes each
    assert(sim.test_stack_depth(1000));
    assert(sim.test_stack_depth(10000));
    assert(!sim.test_stack_depth(20000)); // 20000 * 64 = 1.28 MiB > 1 MiB -> overflow caught
    std::cout << "  ✓ Test 4: Stack capacity and overflow boundary checking verified.\n";

    std::cout << "[PASS] All Chapter 8.4 kernel stack switching tests passed successfully!\n";
}

void print_report() {
    using namespace mikanos;
    std::cout << "====================================================================\n";
    std::cout << "  MikanOS Chapter 8.4: Kernel Stack Switching & System V ABI Report \n";
    std::cout << "====================================================================\n\n";

    std::cout << "1. Kernel Stack Memory Placement (.bss Section):\n";
    std::cout << "   - Base Address:    0x" << std::hex << KernelStackSimulator::get_kernel_stack_base() << std::dec << "\n";
    std::cout << "   - Top Address:     0x" << std::hex << KernelStackSimulator::get_kernel_stack_top() << std::dec << "\n";
    std::cout << "   - Size:            " << (kKernelStackSize / 1024) << " KiB (" << kKernelStackSize << " bytes)\n";
    std::cout << "   - Alignment:       " << (KernelStackSimulator::get_kernel_stack_base() % 16 == 0 ? "16-byte aligned (OK)" : "MISALIGNED!") << "\n\n";

    std::cout << "2. System V AMD64 ABI Stack Invariants:\n";
    std::cout << "   - Pre-call RSP:    Must satisfy RSP % 16 == 0\n";
    std::cout << "   - At Function Entry: (Pushed 8-byte return address) -> RSP % 16 == 8\n";
    std::cout << "   - After 'push rbp': (Pushed 8-byte saved RBP)       -> RSP % 16 == 0\n";
    std::cout << "   - SSE/AVX Safety:  Guarantees aligned movaps/movapd instructions never crash with #GP\n\n";

    std::cout << "3. Assembly Glue Logic (kernel/asmfunc.asm):\n";
    std::cout << "   KernelMain:\n";
    std::cout << "       mov rsp, kernel_main_stack + 1024 * 1024\n";
    std::cout << "       call KernelMainNewStack\n";
    std::cout << "   .fin:\n";
    std::cout << "       hlt\n";
    std::cout << "       jmp .fin\n\n";
    std::cout << "   Arguments in RDI (FrameBufferConfig) & RSI (MemoryMap) remain undisturbed.\n";
    std::cout << "====================================================================\n";
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        run_tests();
        return 0;
    }

    print_report();
    return 0;
}
