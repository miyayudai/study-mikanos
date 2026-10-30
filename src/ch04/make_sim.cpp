/**
 * @file make_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 4.1 (Introduction to Make & Bare-metal Build Automation)
 *
 * Covers:
 * 1. Makefile rule structure: Target, Prerequisites, Recipe
 * 2. Dependency Graph (DAG) construction and Topological Sort (DFS with cycle detection)
 * 3. Timestamp-based rebuild condition: mtime(target) < max(mtime(prerequisites))
 * 4. Automatic variables: $@ (Target), $< (First prerequisite), $^ (All prerequisites), $? (Newer prerequisites), $* (Stem)
 * 5. Phony targets (.PHONY: all clean) and pattern rules (%.o: %.cpp)
 * 6. MikanOS kernel compiler/linker flags verification (-ffreestanding, -mno-red-zone, --image-base 0x100000, etc.)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace mikan::make_sim {

// ============================================================================
// 1. Make Rule & Automatic Variables Expander
// ============================================================================

struct MakeRule {
    std::string target;
    std::vector<std::string> prerequisites;
    std::vector<std::string> recipes;
    bool is_pattern{false};

    [[nodiscard]] std::vector<std::string> ExpandRecipe(
        std::string_view actual_target,
        const std::vector<std::string>& actual_prereqs,
        const std::vector<std::string>& newer_prereqs = {},
        std::string_view stem = "") const {
        
        std::string first_prereq = actual_prereqs.empty() ? "" : actual_prereqs.front();
        
        // Build deduplicated all_prereqs string
        std::string all_prereqs;
        {
            std::set<std::string> seen;
            for (size_t i = 0; i < actual_prereqs.size(); ++i) {
                if (seen.insert(actual_prereqs[i]).second) {
                    if (!all_prereqs.empty()) all_prereqs += " ";
                    all_prereqs += actual_prereqs[i];
                }
            }
        }

        std::string newer_str;
        {
            const auto& src = newer_prereqs.empty() ? actual_prereqs : newer_prereqs;
            std::set<std::string> seen;
            for (size_t i = 0; i < src.size(); ++i) {
                if (seen.insert(src[i]).second) {
                    if (!newer_str.empty()) newer_str += " ";
                    newer_str += src[i];
                }
            }
        }

        std::vector<std::string> expanded;
        expanded.reserve(recipes.size());

        for (const auto& line : recipes) {
            std::string cmd = line;
            auto replace_all = [&](std::string_view token, std::string_view repl) {
                size_t pos = 0;
                while ((pos = cmd.find(token, pos)) != std::string::npos) {
                    cmd.replace(pos, token.length(), repl);
                    pos += repl.length();
                }
            };

            replace_all("$@", actual_target);
            replace_all("$<", first_prereq);
            replace_all("$^", all_prereqs);
            replace_all("$?", newer_str);
            replace_all("$*", stem);

            expanded.push_back(std::move(cmd));
        }

        return expanded;
    }
};

// ============================================================================
// 2. Make Dependency Graph & Build Engine
// ============================================================================

class MakeDependencyGraph {
public:
    void AddRule(std::string target, std::vector<std::string> prereqs, std::vector<std::string> recipes) {
        MakeRule rule{std::move(target), std::move(prereqs), std::move(recipes), false};
        rules_[rule.target] = std::move(rule);
    }

    void AddPatternRule(std::string target_pattern, std::string prereq_pattern, std::vector<std::string> recipes) {
        MakeRule rule{std::move(target_pattern), {std::move(prereq_pattern)}, std::move(recipes), true};
        pattern_rules_.push_back(std::move(rule));
    }

    void MarkPhony(std::string target) {
        phony_targets_.insert(std::move(target));
    }

    [[nodiscard]] bool IsPhony(const std::string& target) const {
        return phony_targets_.contains(target);
    }

    void SetFile(std::string name, double mtime) {
        timestamps_[std::move(name)] = mtime;
    }

    void RemoveFile(const std::string& name) {
        timestamps_.erase(name);
    }

    [[nodiscard]] bool FileExists(const std::string& name) const {
        return timestamps_.contains(name);
    }

    [[nodiscard]] double GetMtime(const std::string& name) const {
        auto it = timestamps_.find(name);
        return (it != timestamps_.end()) ? it->second : 0.0;
    }

    struct RuleResolution {
        const MakeRule* rule{nullptr};
        std::vector<std::string> prerequisites;
        std::string stem;
    };

    [[nodiscard]] RuleResolution ResolveRule(const std::string& target) const {
        auto it = rules_.find(target);
        if (it != rules_.end()) {
            return {&it->second, it->second.prerequisites, ""};
        }

        for (const auto& prule : pattern_rules_) {
            if (prule.target.starts_with('%')) {
                std::string suffix = prule.target.substr(1);
                if (target.ends_with(suffix)) {
                    std::string stem = target.substr(0, target.length() - suffix.length());
                    std::vector<std::string> prereqs;
                    for (const auto& p : prule.prerequisites) {
                        std::string resolved = p;
                        size_t pos = resolved.find('%');
                        if (pos != std::string::npos) {
                            resolved.replace(pos, 1, stem);
                        }
                        prereqs.push_back(std::move(resolved));
                    }
                    return {&prule, std::move(prereqs), std::move(stem)};
                }
            }
        }

        return {nullptr, {}, ""};
    }

    [[nodiscard]] std::vector<std::string> TopologicalSort(const std::string& root) const {
        std::map<std::string, int> state; // 0: unvisited, 1: visiting, 2: visited
        std::vector<std::string> order;

        auto dfs = [&](auto self, const std::string& node) -> void {
            int s = state[node];
            if (s == 1) {
                throw std::runtime_error("Cyclic dependency detected: " + node);
            }
            if (s == 2) return;

            state[node] = 1;
            auto res = ResolveRule(node);
            for (const auto& child : res.prerequisites) {
                self(self, child);
            }
            state[node] = 2;
            order.push_back(node);
        };

        dfs(dfs, root);
        return order;
    }

    struct RebuildCheckResult {
        bool needs_rebuild;
        std::string reason;
        std::vector<std::string> newer_prerequisites;
    };

    [[nodiscard]] RebuildCheckResult NeedsRebuild(const std::string& target) const {
        if (IsPhony(target)) {
            return {true, "Phony target always runs", {}};
        }

        if (!FileExists(target)) {
            return {true, "Target does not exist", {}};
        }

        double target_mtime = GetMtime(target);
        auto res = ResolveRule(target);

        std::vector<std::string> newer;
        for (const auto& p : res.prerequisites) {
            if (GetMtime(p) > target_mtime) {
                newer.push_back(p);
            }
        }

        if (!newer.empty()) {
            return {true, "Prerequisites newer than target", std::move(newer)};
        }

        return {false, "Target is up to date", {}};
    }

    std::vector<std::string> Build(const std::string& target, double current_time = 1000.0) {
        auto order = TopologicalSort(target);
        std::vector<std::string> executed_cmds;

        for (const auto& node : order) {
            auto res = ResolveRule(node);
            if (!res.rule) {
                if (!FileExists(node) && !IsPhony(node)) {
                    throw std::runtime_error("No rule to make target '" + node + "'");
                }
                continue;
            }

            auto check = NeedsRebuild(node);
            if (check.needs_rebuild) {
                current_time += 1.0;
                auto cmds = res.rule->ExpandRecipe(node, res.prerequisites, check.newer_prerequisites, res.stem);
                for (auto& cmd : cmds) {
                    executed_cmds.push_back(cmd);
                }
                if (!IsPhony(node)) {
                    SetFile(node, current_time);
                }
            }
        }

        return executed_cmds;
    }

private:
    std::map<std::string, MakeRule> rules_;
    std::vector<MakeRule> pattern_rules_;
    std::set<std::string> phony_targets_;
    std::map<std::string, double> timestamps_;
};

// ============================================================================
// 3. Compiler Flag Specification
// ============================================================================

struct CompilerFlagSpec {
    std::string_view flag;
    std::string_view category;
    std::string_view purpose;
    std::string_view baremetal_necessity;
    std::string_view omission_risk;
};

inline constexpr std::array<CompilerFlagSpec, 13> kMikanOSFlags = {{
    {"-O2", "Optimization", "最適化Lv2", "描画・転送性能向上と不要スタック退避削減", "描画遅延やバイナリ肥大化"},
    {"-Wall", "Diagnostics", "全主要警告有効化", "未初期化変数などのサイレントクラッシュ事前検知", "潜在バグによるOS異常停止"},
    {"-g", "Diagnostics", "DWARFデバッグ情報生成", "QEMU+GDBでのシンボル・行番号デバッグ", "逆アセンブル時に生アドレスのみで解析不能"},
    {"--target=x86_64-elf", "Target Arch", "ELF64ベアメタルクロス指定", "ホストOS非依存の完全静的ELF生成", "Linux用バイナリとなりブート不能"},
    {"-ffreestanding", "Runtime Env", "自立実行環境宣言", "libc暗黙依存・crt0スタートアップ排除", "存在しないlibcシンボル未解決エラー"},
    {"-mno-red-zone", "Target Arch", "128B Red Zone使用禁止", "非同期割り込み発生時のスタック破壊防止", "割り込みハンドラによるカーネルスタック破壊"},
    {"-fno-exceptions", "C++ Suppress", "C++例外処理無効化", "スタック巻き戻し・巨大ランタイム排除", "__gxx_personality_v0未解決リンクエラー"},
    {"-fno-rtti", "C++ Suppress", "ランタイム型情報無効化", "vtable内の不要メタデータ参照排除", "型記述メタデータ未解決リンクエラー"},
    {"-std=c++17", "Language", "C++17言語規格", "構造化束縛・constexpr if等モダン機能活用", "旧規格制約による型安全性低下"},
    {"--entry KernelMain", "Linker Spec", "ELFエントリポイント指定", "ブートローダからの直接ジャンプアドレス設定", "_start期待によるリンクエラーまたは暴走"},
    {"-z norelro", "Linker Spec", "RELROセグメント無効化", "ページング未設定環境でのセクション単純化", "ELFセグメントローダの複雑化・クラッシュ"},
    {"--image-base 0x100000", "Linker Spec", "配置基準物理アドレス1MiB固定", "BIOS/UEFI予約メモリ回避と空きRAM配置", "0x200000等の不一致配置による即死"},
    {"--static", "Linker Spec", "完全静的リンク", "動的共有ライブラリ参照の完全排除", "動的ローダ不在による起動失敗"}
}};

// ============================================================================
// 4. Verification Suite & Unit Tests
// ============================================================================

void RunTests() {
    std::cout << "[Test 1] Makefile rule and automatic variable expansion..." << std::endl;
    {
        MakeRule rule{
            "kernel.elf",
            {"main.o", "Makefile"},
            {"ld.lld $(LDFLAGS) -o $@ $^"},
            false
        };
        auto cmds = rule.ExpandRecipe("kernel.elf", {"main.o", "Makefile"}, {"main.o"}, "");
        assert(cmds.size() == 1);
        assert(cmds[0] == "ld.lld $(LDFLAGS) -o kernel.elf main.o Makefile");

        MakeRule pattern_rule{
            "%.o",
            {"%.cpp", "Makefile"},
            {"clang++ $(CPPFLAGS) $(CXXFLAGS) -c $<"},
            true
        };
        auto pcmds = pattern_rule.ExpandRecipe("main.o", {"main.cpp", "Makefile"}, {"main.cpp"}, "main");
        assert(pcmds.size() == 1);
        assert(pcmds[0] == "clang++ $(CPPFLAGS) $(CXXFLAGS) -c main.cpp");
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 2] Dependency Graph topological sort & cycle detection..." << std::endl;
    {
        MakeDependencyGraph graph;
        graph.AddRule("all", {"kernel.elf"}, {});
        graph.AddRule("kernel.elf", {"main.o", "Makefile"}, {"ld.lld -o $@ $^"});
        graph.AddPatternRule("%.o", "%.cpp", {"clang++ -c $<"});
        graph.MarkPhony("all");

        graph.SetFile("main.cpp", 100.0);
        graph.SetFile("Makefile", 100.0);

        auto order = graph.TopologicalSort("all");
        // Topological order must process prerequisites before targets:
        // main.cpp -> main.o -> Makefile -> kernel.elf -> all
        auto find_idx = [&](const std::string& name) {
            auto it = std::find(order.begin(), order.end(), name);
            assert(it != order.end());
            return std::distance(order.begin(), it);
        };
        assert(find_idx("main.cpp") < find_idx("main.o"));
        assert(find_idx("main.o") < find_idx("kernel.elf"));
        assert(find_idx("Makefile") < find_idx("kernel.elf"));
        assert(find_idx("kernel.elf") < find_idx("all"));

        // Test cyclic dependency detection
        MakeDependencyGraph cycle_graph;
        cycle_graph.AddRule("a", {"b"}, {});
        cycle_graph.AddRule("b", {"c"}, {});
        cycle_graph.AddRule("c", {"a"}, {});
        bool caught = false;
        try {
            std::ignore = cycle_graph.TopologicalSort("a");
        } catch (const std::runtime_error&) {
            caught = true;
        }
        assert(caught);
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 3] Timestamp comparison & incremental build..." << std::endl;
    {
        MakeDependencyGraph graph;
        graph.AddRule("all", {"kernel.elf"}, {});
        graph.AddRule("kernel.elf", {"main.o", "Makefile"}, {"ld.lld -o $@ $^"});
        graph.AddPatternRule("%.o", "%.cpp", {"clang++ -c $<"});
        graph.MarkPhony("all");

        graph.SetFile("main.cpp", 100.0);
        graph.SetFile("Makefile", 100.0);

        // Step 1: Initial build (no objects exist)
        auto cmds1 = graph.Build("all", 1000.0);
        assert(cmds1.size() == 2);
        assert(cmds1[0] == "clang++ -c main.cpp");
        assert(cmds1[1] == "ld.lld -o kernel.elf main.o Makefile");
        assert(graph.FileExists("main.o"));
        assert(graph.FileExists("kernel.elf"));

        // Step 2: Immediate rebuild with no source changes -> nothing to build
        auto cmds2 = graph.Build("all", 2000.0);
        assert(cmds2.empty());

        // Step 3: Modify main.cpp (newer timestamp)
        graph.SetFile("main.cpp", 2100.0);
        auto cmds3 = graph.Build("all", 3000.0);
        assert(cmds3.size() == 2);
        assert(cmds3[0] == "clang++ -c main.cpp");
        assert(cmds3[1] == "ld.lld -o kernel.elf main.o Makefile");

        // Step 4: Modify Makefile only -> kernel.elf is rebuilt
        graph.SetFile("Makefile", 4000.0);
        auto cmds4 = graph.Build("all", 5000.0);
        assert(!cmds4.empty());
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 4] Phony target protection..." << std::endl;
    {
        MakeDependencyGraph graph;
        graph.AddRule("clean", {}, {"rm -rf *.o"});
        graph.MarkPhony("clean");

        // Even if a file named 'clean' exists on disk, phony target must execute
        graph.SetFile("clean", 9999.0);
        auto cmds = graph.Build("clean", 10000.0);
        assert(cmds.size() == 1);
        assert(cmds[0] == "rm -rf *.o");
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "[Test 5] Compiler & Linker Flag specifications..." << std::endl;
    {
        assert(kMikanOSFlags.size() == 13);
        bool found_freestanding = false;
        bool found_red_zone = false;
        bool found_image_base = false;

        for (const auto& f : kMikanOSFlags) {
            if (f.flag == "-ffreestanding") found_freestanding = true;
            if (f.flag == "-mno-red-zone") found_red_zone = true;
            if (f.flag == "--image-base 0x100000") found_image_base = true;
        }

        assert(found_freestanding);
        assert(found_red_zone);
        assert(found_image_base);
    }
    std::cout << "  -> PASSED" << std::endl;

    std::cout << "All C++20 Make & Bare-metal Build tests passed successfully!" << std::endl;
}

} // namespace mikan::make_sim

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "=== MikanOS Chapter 4.1: Make & Bare-metal Build Simulator (C++20) ===" << std::endl;
    try {
        mikan::make_sim::RunTests();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
