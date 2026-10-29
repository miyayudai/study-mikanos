/**
 * @file binary_mutation_tester.cpp
 * @brief C++20 Host-side Binary Mutation & Low-Level Empirical Experimentation Tool
 * 
 * Part of study-mikanos: Chapter 1.6 "Taking Action: Empirical Experimentation & Binary Mutation"
 * Demonstrates:
 *   1. Empirical OODA Loop in low-level systems programming (Observe -> Orient -> Decide -> Act)
 *   2. PE32+ byte-level binary patching & mutation (UTF-16LE strings and x86-64 machine instructions)
 *   3. 4 Key mutation experiments:
 *      - Case 1: Same-length UTF-16LE string replacement (clean display)
 *      - Case 2: Shortened string missing null terminator (memory garbage / trailing byte exposure)
 *      - Case 3: Oversized string exceeding raw section / file boundaries (overflow rejection)
 *      - Case 4: Machine code modification: breaking infinite loop 'jmp $' (stack imbalance / CPU fault)
 *   4. Byte-level Hexdiff generation and unified difference reporting
 *   5. Automated self-testing and CTest verification suite
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
#include <fstream>

namespace mikanos {

// -----------------------------------------------------------------------------
// 1. Mutation Outcome Enums & Structures
// -----------------------------------------------------------------------------

enum class MutationStatus {
    Success,
    TruncatedValid,
    NullTerminatorMissing,
    SectionOverflow,
    ExecutionCrash
};

inline const char* status_to_string(MutationStatus status) {
    switch (status) {
        case MutationStatus::Success: return "SUCCESS (Normal Display)";
        case MutationStatus::TruncatedValid: return "TRUNCATED_VALID (Shortened with Null Terminator)";
        case MutationStatus::NullTerminatorMissing: return "NULL_TERMINATOR_MISSING (Trailing Memory Exposed)";
        case MutationStatus::SectionOverflow: return "SECTION_OVERFLOW (Exceeded Raw Section Bounds)";
        case MutationStatus::ExecutionCrash: return "EXECUTION_CRASH (CPU Fault / Stack Imbalance)";
        default: return "UNKNOWN";
    }
}

struct BinaryPatch {
    size_t offset;
    std::vector<uint8_t> original_bytes;
    std::vector<uint8_t> patched_bytes;
    std::string description;
};

struct DiffEntry {
    size_t offset;
    uint8_t original_byte;
    uint8_t patched_byte;
};

struct ExecutionResult {
    MutationStatus status;
    std::string screen_output;
    bool halted;
    std::string cpu_state;
    std::string diagnostics;
};

// -----------------------------------------------------------------------------
// 2. Binary Mutator Engine
// -----------------------------------------------------------------------------

class BinaryMutator {
public:
    explicit BinaryMutator(std::vector<uint8_t> initial_data)
        : original_data_(initial_data), current_data_(std::move(initial_data)) {}

    [[nodiscard]] const std::vector<uint8_t>& data() const { return current_data_; }
    [[nodiscard]] size_t size() const { return current_data_.size(); }

    // Convert UTF-8 std::string to UTF-16LE bytes
    static std::vector<uint8_t> encode_utf16le(const std::string& text, bool null_terminate = true) {
        std::vector<uint8_t> result;
        result.reserve((text.size() + (null_terminate ? 1 : 0)) * 2);
        for (char c : text) {
            uint16_t code_point = static_cast<uint8_t>(c);
            result.push_back(static_cast<uint8_t>(code_point & 0xFF));
            result.push_back(static_cast<uint8_t>((code_point >> 8) & 0xFF));
        }
        if (null_terminate) {
            result.push_back(0x00);
            result.push_back(0x00);
        }
        return result;
    }

    // Decode UTF-16LE string from offset until 0x0000 or max_chars
    std::pair<std::string, bool> decode_utf16le(size_t offset, size_t max_chars = 128) const {
        std::string text;
        bool is_null_terminated = false;
        size_t limit = std::min(offset + max_chars * 2, current_data_.size());

        for (size_t ptr = offset; ptr + 1 < limit; ptr += 2) {
            uint16_t code = static_cast<uint16_t>(current_data_[ptr]) |
                            (static_cast<uint16_t>(current_data_[ptr + 1]) << 8);
            if (code == 0) {
                is_null_terminated = true;
                break;
            }
            if (code < 128) {
                text.push_back(static_cast<char>(code));
            } else {
                text.push_back('?');
            }
        }
        return {text, is_null_terminated};
    }

    // Search for byte pattern
    std::vector<size_t> find_pattern(const std::vector<uint8_t>& pattern) const {
        std::vector<size_t> matches;
        if (pattern.empty() || pattern.size() > current_data_.size()) return matches;

        for (size_t i = 0; i <= current_data_.size() - pattern.size(); ++i) {
            if (std::memcmp(&current_data_[i], pattern.data(), pattern.size()) == 0) {
                matches.push_back(i);
            }
        }
        return matches;
    }

    // Apply binary patch
    BinaryPatch apply_patch(size_t offset, const std::vector<uint8_t>& patch_bytes, const std::string& desc = "") {
        if (offset + patch_bytes.size() > current_data_.size()) {
            current_data_.resize(offset + patch_bytes.size(), 0x00);
        }

        std::vector<uint8_t> orig(current_data_.begin() + offset,
                                  current_data_.begin() + offset + patch_bytes.size());

        std::copy(patch_bytes.begin(), patch_bytes.end(), current_data_.begin() + offset);

        BinaryPatch patch{
            .offset = offset,
            .original_bytes = orig,
            .patched_bytes = patch_bytes,
            .description = desc
        };
        patches_.push_back(patch);
        return patch;
    }

    // Calculate diff against original
    [[nodiscard]] std::vector<DiffEntry> compute_diff() const {
        std::vector<DiffEntry> diffs;
        size_t max_size = std::max(original_data_.size(), current_data_.size());
        for (size_t i = 0; i < max_size; ++i) {
            uint8_t orig = (i < original_data_.size()) ? original_data_[i] : 0x00;
            uint8_t curr = (i < current_data_.size()) ? current_data_[i] : 0x00;
            if (orig != curr) {
                diffs.push_back(DiffEntry{.offset = i, .original_byte = orig, .patched_byte = curr});
            }
        }
        return diffs;
    }

    // Format diff table
    [[nodiscard]] std::string format_diff_table(size_t max_rows = 15) const {
        auto diffs = compute_diff();
        if (diffs.empty()) {
            return "No modifications detected (Data is bit-for-bit identical).";
        }
        std::ostringstream oss;
        oss << "Offset (Hex) | Original Byte | Patched Byte | Representation\n";
        oss << "-------------+---------------+--------------+----------------\n";
        for (size_t i = 0; i < std::min(diffs.size(), max_rows); ++i) {
            const auto& d = diffs[i];
            char orig_c = (d.original_byte >= 32 && d.original_byte <= 126) ? static_cast<char>(d.original_byte) : '.';
            char patch_c = (d.patched_byte >= 32 && d.patched_byte <= 126) ? static_cast<char>(d.patched_byte) : '.';
            oss << "0x" << std::hex << std::uppercase << std::setfill('0') << std::setw(6) << d.offset
                << "     | 0x" << std::setw(2) << static_cast<int>(d.original_byte) << " ('" << orig_c << "')   "
                << "| 0x" << std::setw(2) << static_cast<int>(d.patched_byte) << " ('" << patch_c << "')   "
                << "| Modified\n";
        }
        if (diffs.size() > max_rows) {
            oss << "... and " << std::dec << (diffs.size() - max_rows) << " more byte modifications.\n";
        }
        return oss.str();
    }

    // Simulate UEFI Execution Behavior
    [[nodiscard]] ExecutionResult simulate_execution() const {
        ExecutionResult res;
        res.status = MutationStatus::Success;
        res.halted = true;
        res.cpu_state = "HALTED (Looping at 0x212)";

        // 1. Check file size vs Section Capacity (original hello.efi is 1536 bytes / 0x600)
        constexpr size_t kMaxAllowedSize = 1536;
        if (current_data_.size() > kMaxAllowedSize) {
            res.status = MutationStatus::SectionOverflow;
            res.halted = false;
            res.cpu_state = "CRASH / BDS_IMAGE_REJECTED";
            res.diagnostics = "Binary size (" + std::to_string(current_data_.size()) + 
                              " bytes) exceeds raw section capacity (1536 bytes). UEFI BDS rejected load.";
            return res;
        }

        // 2. Check machine code at .text infinite loop (offset 0x212: EB FE = jmp $)
        constexpr size_t kLoopOffset = 0x212;
        if (current_data_.size() >= kLoopOffset + 2) {
            if (current_data_[kLoopOffset] != 0xEB || current_data_[kLoopOffset + 1] != 0xFE) {
                res.status = MutationStatus::ExecutionCrash;
                res.halted = false;
                res.cpu_state = "CPU_FAULT (General Protection / Stack Imbalance #GP)";
                res.diagnostics = "Infinite loop 'jmp $' at 0x212 modified. CPU fell through without stack restoration.";
                auto [str, _] = decode_utf16le(0x400, 32);
                res.screen_output = str;
                return res;
            }
        }

        // 3. Check string at .data / .rdata (offset 0x400)
        constexpr size_t kStrOffset = 0x400;
        auto [decoded, terminated] = decode_utf16le(kStrOffset, 64);
        res.screen_output = decoded;

        if (!terminated) {
            res.status = MutationStatus::NullTerminatorMissing;
            res.diagnostics = "UTF-16 string lacks 2-byte null terminator (0x0000). Trailing bytes exposed!";
        } else {
            auto [orig_str, _] = BinaryMutator(original_data_).decode_utf16le(kStrOffset);
            if (decoded == orig_str) {
                res.status = MutationStatus::Success;
                res.diagnostics = "Original string executed identically.";
            } else if (decoded.size() < orig_str.size()) {
                res.status = MutationStatus::TruncatedValid;
                res.diagnostics = "String shortened and safely terminated with 0x0000.";
            } else {
                res.status = MutationStatus::Success;
                res.diagnostics = "String mutated cleanly within section bounds.";
            }
        }

        return res;
    }

private:
    std::vector<uint8_t> original_data_;
    std::vector<uint8_t> current_data_;
    std::vector<BinaryPatch> patches_;
};

// -----------------------------------------------------------------------------
// 3. Synthetic UEFI Binary Generator for Testing
// -----------------------------------------------------------------------------

inline std::vector<uint8_t> create_synthetic_hello_efi() {
    std::vector<uint8_t> bin(1536, 0x00);
    // DOS header magic "MZ"
    bin[0] = 'M'; bin[1] = 'Z';
    // PE header offset at 0x3C -> 0x80
    bin[0x3C] = 0x80;
    // PE signature "PE\0\0"
    bin[0x80] = 'P'; bin[0x81] = 'E';

    // .text section code at 0x200
    // sub rsp, 0x28 (48 83 ec 28)
    bin[0x200] = 0x48; bin[0x201] = 0x83; bin[0x202] = 0xEC; bin[0x203] = 0x28;
    // infinite loop at 0x212: jmp $ (EB FE)
    bin[0x212] = 0xEB; bin[0x213] = 0xFE;

    // .data section UTF-16LE string at 0x400: "Hello, world!\0"
    auto str_bytes = BinaryMutator::encode_utf16le("Hello, world!", true);
    std::copy(str_bytes.begin(), str_bytes.end(), bin.begin() + 0x400);

    return bin;
}

} // namespace mikanos

// -----------------------------------------------------------------------------
// 4. Main Entrypoint & Self-Test Suite
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::cout << "=================================================================\n";
    std::cout << " 🍊 MikanOS Chapter 1.6: Binary Mutation & Empirical Tester\n";
    std::cout << "    (Low-Level Hands-On OODA Loop & Binary Patch Verification)\n";
    std::cout << "=================================================================\n\n";

    if (argc > 1 && std::string(argv[1]) == "--demo") {
        std::cout << "[Running in Interactive Demo Mode]\n\n";
    }

    // Step 1: Base Binary Setup
    auto base_bin = mikanos::create_synthetic_hello_efi();
    mikanos::BinaryMutator mutator(base_bin);

    std::cout << ">>> 1. Baseline Binary Verification <<<\n";
    std::cout << "  Size: " << mutator.size() << " bytes\n";
    auto initial_exec = mutator.simulate_execution();
    std::cout << "  Status: " << mikanos::status_to_string(initial_exec.status) << "\n";
    std::cout << "  Screen: \"" << initial_exec.screen_output << "\"\n";
    std::cout << "  CPU:    " << initial_exec.cpu_state << "\n\n";
    assert(initial_exec.status == mikanos::MutationStatus::Success);
    assert(initial_exec.screen_output == "Hello, world!");

    // Step 2: Case 1 - Same-Length UTF-16 Mutation
    std::cout << ">>> 2. Experiment Case 1: Same-Length Mutation (\"Hello, Mikan!\") <<<\n";
    mikanos::BinaryMutator case1_mutator(base_bin);
    auto patch1 = case1_mutator.apply_patch(
        0x400,
        mikanos::BinaryMutator::encode_utf16le("Hello, Mikan!", true),
        "Replace 'Hello, world!' with 'Hello, Mikan!'"
    );
    auto res1 = case1_mutator.simulate_execution();
    std::cout << "  Patched " << patch1.patched_bytes.size() << " bytes at offset 0x400\n";
    std::cout << "  Status: " << mikanos::status_to_string(res1.status) << "\n";
    std::cout << "  Screen: \"" << res1.screen_output << "\"\n";
    std::cout << case1_mutator.format_diff_table(6) << "\n";
    assert(res1.status == mikanos::MutationStatus::Success);
    assert(res1.screen_output == "Hello, Mikan!");

    // Step 3: Case 2 - Shortened String without Null Terminator
    std::cout << ">>> 3. Experiment Case 2: Missing Null Terminator (\"Hi!\") <<<\n";
    mikanos::BinaryMutator case2_mutator(base_bin);
    // Write only "Hi!" (6 bytes), leaving original trailing bytes "llo, world!\0"
    auto patch2 = case2_mutator.apply_patch(
        0x400,
        mikanos::BinaryMutator::encode_utf16le("Hi!", false),
        "Overwrite first 3 chars with 'Hi!' but no null terminator"
    );
    auto res2 = case2_mutator.simulate_execution();
    std::cout << "  Patched " << patch2.patched_bytes.size() << " bytes without null terminator\n";
    std::cout << "  Status: " << mikanos::status_to_string(res2.status) << "\n";
    std::cout << "  Screen: \"" << res2.screen_output << "\" (Memory remnant exposed!)\n\n";
    assert(res2.screen_output == "Hi!lo, world!");

    // Step 4: Case 3 - Section Overflow
    std::cout << ">>> 4. Experiment Case 3: Raw Section Overflow (>1536 bytes) <<<\n";
    mikanos::BinaryMutator case3_mutator(base_bin);
    std::vector<uint8_t> overflow_payload(200, 0x41);
    case3_mutator.apply_patch(1500, overflow_payload, "Inject payload crossing 1536-byte file bound");
    auto res3 = case3_mutator.simulate_execution();
    std::cout << "  New Binary Size: " << case3_mutator.size() << " bytes\n";
    std::cout << "  Status: " << mikanos::status_to_string(res3.status) << "\n";
    std::cout << "  CPU:    " << res3.cpu_state << "\n";
    std::cout << "  Diag:   " << res3.diagnostics << "\n\n";
    assert(res3.status == mikanos::MutationStatus::SectionOverflow);

    // Step 5: Case 4 - Breaking Infinite Loop (NOP substitution)
    std::cout << ">>> 5. Experiment Case 4: Break Infinite Loop (EB FE -> 90 90) <<<\n";
    mikanos::BinaryMutator case4_mutator(base_bin);
    std::vector<uint8_t> nop_bytes = {0x90, 0x90};
    case4_mutator.apply_patch(0x212, nop_bytes, "Replace 'jmp $' with 'nop nop'");
    auto res4 = case4_mutator.simulate_execution();
    std::cout << "  Patched opcode at 0x212 to 0x9090 (NOP NOP)\n";
    std::cout << "  Status: " << mikanos::status_to_string(res4.status) << "\n";
    std::cout << "  CPU:    " << res4.cpu_state << "\n";
    std::cout << "  Diag:   " << res4.diagnostics << "\n\n";
    assert(res4.status == mikanos::MutationStatus::ExecutionCrash);

    std::cout << "✅ All Chapter 1.6 C++20 Binary Mutation Verification Tests PASSED!\n";
    return 0;
}
