// src/ch00/os_architecture_roadmap.cpp
// Section 0.5: Big Picture of MikanOS Architecture Roadmap and Tech Stack Selection Rationale
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::ch00 {

// 1. Architectural layers in MikanOS hierarchy
enum class ArchitectureLayer {
    kHAL,         // Hardware Abstraction Layer & Device Drivers (GOP, PCI, xHCI, APIC)
    kKernelCore,  // Core Kernel Subsystems (Memory, IDT, Paging, Scheduler, Syscall)
    kSubsystem,   // Window Manager, GUI Layers, FAT FileSystem, Terminal
    kUserland     // Ring 3 Applications, Standard Library (Newlib libc)
};

constexpr std::string_view LayerToString(ArchitectureLayer layer) {
    switch (layer) {
        case ArchitectureLayer::kHAL: return "Hardware / HAL";
        case ArchitectureLayer::kKernelCore: return "Kernel Core";
        case ArchitectureLayer::kSubsystem: return "OS Subsystem";
        case ArchitectureLayer::kUserland: return "Userland / Apps";
    }
    return "Unknown";
}

// 2. Definition of an OS Kernel Module and its dependency graph
struct KernelModule {
    std::string_view name;
    int chapter;
    ArchitectureLayer layer;
    std::vector<std::string_view> dependencies;
    std::string_view description;
};

// 3. Definition of an evolutionary phase in MikanOS
struct ArchitecturePhase {
    int phase_id;
    std::string_view name;
    std::string_view chapter_range;
    int start_chapter;
    int end_chapter;
    std::vector<std::string_view> key_milestones;
    std::vector<std::string_view> core_data_structures;
};

// 4. Multi-criteria architectural tech stack evaluation model
struct TechEvaluation {
    std::string_view technology;
    std::string_view category;
    int bit64_affinity;       // 1 - 5
    int graphics_support;     // 1 - 5
    int file_access;          // 1 - 5
    int simplicity;           // 1 - 5
    int type_safety_oop;      // 1 - 5

    [[nodiscard]] constexpr int TotalScore() const {
        return bit64_affinity + graphics_support + file_access + simplicity + type_safety_oop;
    }
};

class MikanOSRoadmapModel {
public:
    static const std::vector<ArchitecturePhase>& GetPhases() {
        static const std::vector<ArchitecturePhase> kPhases = {
            {
                1,
                "ブートとグラフィックス基盤",
                "第1章 〜 第5章",
                1,
                5,
                {
                    "UEFI EDK II ブートローダ (MikanLoaderPkg)",
                    "GOP (Graphics Output Protocol) フレームバッファ取得",
                    "PixelWriter による点・矩形描画ポリモーフィズム",
                    "Hankaku ビットマップフォント展開と Console クラス",
                    "Newlib 連携と printk 書式付き文字列出力",
                },
                {"PixelWriter", "PixelFormat", "Console", "FrameBufferConfig"},
            },
            {
                2,
                "入力デバイスと割り込み機構",
                "第6章 〜 第7章",
                6,
                7,
                {
                    "PCI コンフィギュレーション空間の体系的探索",
                    "xHCI (USB 3.0) ホストコントローラの発見と初期化",
                    "IDT (割り込み記述子表) 登録と割り込みハンドラ",
                    "MSI (Message Signaled Interrupts) によるハードウェア通知",
                    "ArrayQueue (リングバッファ/FIFO) によるイベント非同期配送",
                },
                {"IDTEntry", "InterruptDescriptorTable", "ArrayQueue", "PCIConfigAddress"},
            },
            {
                3,
                "メモリ管理とGUIレイヤ重ね合わせ",
                "第8章 〜 第10章",
                8,
                10,
                {
                    "GDT (Global Descriptor Table) セグメント初期化",
                    "BitmapMemoryManager による 4KiB 物理フレーム管理",
                    "new / delete 演算子オーバーロードとカーネルヒープ",
                    "Layer および LayerManager による前後関係（Z-order）管理",
                    "ShadowBuffer (ダブルバッファ) によるチラつき解消",
                },
                {"BitmapMemoryManager", "SegmentDescriptor", "Layer", "LayerManager", "Window"},
            },
            {
                4,
                "時間計測・キーボードとマルチタスク",
                "第11章 〜 第14章",
                11,
                14,
                {
                    "Local APIC タイマ割り込みと ACPI PM タイマ較正",
                    "TimerManager (優先度付きキュー) によるミリ秒タイマ",
                    "USB HID キーボードドライバとモディファイアキー解析",
                    "TaskContext と SwitchContext アセンブリによる協調的タスク切り替え",
                    "マルチレベル優先度付きランキューと HLT アイドルタスク",
                },
                {"TaskContext", "Task", "TaskManager", "Timer", "TimerManager"},
            },
            {
                5,
                "ターミナル・コマンドとファイルシステム",
                "第15章 〜 第18章",
                15,
                18,
                {
                    "Terminal タスクの独立スレッド化とアクティブ入力制御",
                    "キーイベントディスパッチと echo / clear / lspci コマンド",
                    "FAT32 BPB (BIOS Parameter Block) とディレクトリエントリ解析",
                    "クラスタチェーン追跡と FAT ファイル読み込み (ls コマンド)",
                    "ELF アプリケーションバイナリのロードと実行",
                },
                {"Terminal", "FAT32Header", "DirectoryEntry", "Elf64_Ehdr", "Elf64_Phdr"},
            },
            {
                6,
                "4階層ページングとシステムコール保護",
                "第19章 〜 第23章",
                19,
                23,
                {
                    "48ビット正規アドレスと 4階層ページテーブル (PML4/PDP/PD/PT)",
                    "カーネル空間とアプリケーション空間の仮想メモリ分離",
                    "Ring 0 (Kernel) と Ring 3 (User) の特権レベル分離",
                    "TSS (Task State Segment) と IST (Interrupt Stack Table)",
                    "SYSCALL / SYSRET 命令による低オーバーヘッドシステムコール",
                    "GUI アプリケーション API (ウィンドウ描画・イベント配送)",
                },
                {"PML4Table", "PDPTable", "PageDirectory", "TSS64", "SyscallArgs"},
            },
            {
                7,
                "実用OSへの洗練・IPCとアプリケーション",
                "第24章 〜 第30章",
                24,
                30,
                {
                    "複数ターミナルの同時起動と PML4 仮想空間の独立化",
                    "ファイル書き込み (FAT クラスタ割り当て) と標準入力抽象化",
                    "デマンドページング (Demand Paging) と Copy-on-Write (CoW)",
                    "UTF-8 デコードと FreeType による日本語フォント描画",
                    "パイプ (Pipe) と標準入出力リダイレクトによる IPC",
                    "実用アプリ群 (テキストビューア tview, 画像ビューア gview, blocks)",
                },
                {"FileDescriptor", "PipeDescriptor", "DemandPageManager", "FreeTypeFont"},
            },
        };
        return kPhases;
    }

    static const std::vector<KernelModule>& GetModules() {
        static const std::vector<KernelModule> kModules = {
            {"PixelWriter", 4, ArchitectureLayer::kHAL, {}, "GOPフレームバッファ描画抽象化"},
            {"Console", 5, ArchitectureLayer::kKernelCore, {"PixelWriter"}, "文字描画・スクロール・printk基盤"},
            {"PCIManager", 6, ArchitectureLayer::kHAL, {}, "PCIバス探索・デバイス特定"},
            {"xHCIDriver", 6, ArchitectureLayer::kHAL, {"PCIManager"}, "USB 3.0 コントローラ初期化"},
            {"IDTManager", 7, ArchitectureLayer::kKernelCore, {}, "割り込み記述子表登録・CPU例外ハンドラ"},
            {"ArrayQueue", 7, ArchitectureLayer::kKernelCore, {}, "リングバッファ型スレッドセーフ非同期キュー"},
            {"BitmapMemoryManager", 8, ArchitectureLayer::kKernelCore, {}, "4KiB物理ページフレーム割り当て"},
            {"LayerManager", 9, ArchitectureLayer::kSubsystem, {"PixelWriter"}, "複数ウィンドウZ-order重ね合わせ合成"},
            {"TimerManager", 11, ArchitectureLayer::kKernelCore, {"IDTManager"}, "Local APICタイマ割り込み管理"},
            {"TaskManager", 13, ArchitectureLayer::kKernelCore, {"TimerManager"}, "コンテキストスイッチと優先度スケジューラ"},
            {"Terminal", 15, ArchitectureLayer::kSubsystem, {"Console", "TaskManager", "LayerManager"}, "独立ターミナル・コマンド実行"},
            {"FATFileSystem", 17, ArchitectureLayer::kSubsystem, {}, "FAT32ファイル解析・読み書き"},
            {"ELFLoader", 18, ArchitectureLayer::kSubsystem, {"FATFileSystem"}, "ELF64バイナリ検証・セグメント配置"},
            {"PagingManager", 19, ArchitectureLayer::kKernelCore, {"BitmapMemoryManager"}, "4階層ページテーブル構成とメモリ保護"},
            {"SyscallHandler", 20, ArchitectureLayer::kKernelCore, {"TaskManager", "PagingManager"}, "SYSCALL/SYSRET特権境界ハンドリング"},
            {"AppWindowManager", 21, ArchitectureLayer::kSubsystem, {"LayerManager", "SyscallHandler"}, "ユーザーアプリ向けGUI API"},
            {"PipeManager", 29, ArchitectureLayer::kSubsystem, {"TaskManager"}, "プロセス間通信(IPC)用パイプバッファ"},
        };
        return kModules;
    }

    // Validates that all module dependencies reference earlier or contemporary chapters (no circular dependency)
    static bool ValidateDependencies() {
        const auto& modules = GetModules();
        std::map<std::string_view, int> module_chapters;
        for (const auto& m : modules) {
            module_chapters[m.name] = m.chapter;
        }

        for (const auto& m : modules) {
            for (const auto& dep : m.dependencies) {
                auto it = module_chapters.find(dep);
                if (it == module_chapters.end()) {
                    return false;  // Dependent module does not exist
                }
                if (it->second > m.chapter) {
                    return false;  // Backward time dependency!
                }
            }
        }
        return true;
    }

    // Calculates the longest dependency chain length (depth) for a given module
    static int CalculateDependencyDepth(std::string_view module_name) {
        const auto& modules = GetModules();
        const KernelModule* target = nullptr;
        for (const auto& m : modules) {
            if (m.name == module_name) {
                target = &m;
                break;
            }
        }
        if (!target || target->dependencies.empty()) {
            return 0;
        }

        int max_depth = 0;
        for (const auto& dep : target->dependencies) {
            max_depth = std::max(max_depth, 1 + CalculateDependencyDepth(dep));
        }
        return max_depth;
    }

    static std::map<ArchitectureLayer, size_t> GetLayerDistribution() {
        std::map<ArchitectureLayer, size_t> dist;
        for (const auto& m : GetModules()) {
            dist[m.layer]++;
        }
        return dist;
    }
};

class TechStackEvaluator {
public:
    static const std::vector<TechEvaluation>& GetFirmwareEvaluations() {
        static const std::vector<TechEvaluation> kEvals = {
            {"UEFI (EDK II)", "Firmware", 5, 5, 5, 2, 4},
            {"Legacy BIOS (MBR)", "Firmware", 1, 2, 2, 4, 2},
        };
        return kEvals;
    }

    static const std::vector<TechEvaluation>& GetLanguageEvaluations() {
        static const std::vector<TechEvaluation> kEvals = {
            {"C++ (C++17/20)", "Language", 5, 5, 5, 4, 5},
            {"Pure C (C99)", "Language", 5, 4, 4, 5, 2},
            {"Rust", "Language", 5, 4, 4, 2, 5},
        };
        return kEvals;
    }
};

}  // namespace mikanos::ch00

int main() {
    using namespace mikanos::ch00;

    std::cout << "=================================================================\n";
    std::cout << " MikanOS Architecture Roadmap & Tech Stack Verification (C++20)  \n";
    std::cout << "=================================================================\n\n";

    // 1. Verify 7 architectural phases
    const auto& phases = MikanOSRoadmapModel::GetPhases();
    assert(phases.size() == 7);
    assert(phases.front().start_chapter == 1);
    assert(phases.back().end_chapter == 30);

    std::cout << "--- 1. Seven Evolutionary Phases of MikanOS ---\n";
    for (const auto& phase : phases) {
        std::cout << "[Phase " << phase.phase_id << "] "
                  << std::setw(32) << std::left << phase.name
                  << " (" << phase.chapter_range << ") : "
                  << phase.key_milestones.size() << " milestones\n";
    }
    std::cout << "\n";

    // 2. Verify module dependency graph
    bool valid_deps = MikanOSRoadmapModel::ValidateDependencies();
    assert(valid_deps);
    std::cout << "--- 2. Module Dependency Graph Validation ---\n";
    std::cout << "Dependency graph DAG status: " << (valid_deps ? "VALID (No Cycles / Chronological)" : "INVALID") << "\n\n";

    const auto& modules = MikanOSRoadmapModel::GetModules();
    std::cout << std::setw(20) << std::left << "Module Name"
              << std::setw(10) << "Chapter"
              << std::setw(18) << "Layer"
              << std::setw(8)  << "Depth"
              << "Dependencies\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& m : modules) {
        int depth = MikanOSRoadmapModel::CalculateDependencyDepth(m.name);
        std::string deps_str;
        for (size_t i = 0; i < m.dependencies.size(); ++i) {
            if (i > 0) deps_str += ", ";
            deps_str += m.dependencies[i];
        }
        if (deps_str.empty()) deps_str = "(none)";

        std::cout << std::setw(20) << std::left << m.name
                  << "Ch." << std::setw(6) << m.chapter
                  << std::setw(18) << LayerToString(m.layer)
                  << std::setw(8) << depth
                  << deps_str << "\n";
    }
    std::cout << "\n";

    // 3. Verify Layer distribution
    auto layer_dist = MikanOSRoadmapModel::GetLayerDistribution();
    std::cout << "--- 3. Kernel Layer Distribution ---\n";
    for (const auto& [layer, count] : layer_dist) {
        std::cout << " - " << std::setw(18) << std::left << LayerToString(layer) << ": " << count << " modules\n";
    }
    std::cout << "\n";

    // 4. Verify Tech Stack evaluation scores
    std::cout << "--- 4. Technology Stack Selection Rationale ---\n";
    const auto& fw_evals = TechStackEvaluator::GetFirmwareEvaluations();
    for (const auto& eval : fw_evals) {
        std::cout << "[Firmware] " << std::setw(20) << std::left << eval.technology
                  << " Score: " << eval.TotalScore() << " / 25\n";
    }
    assert(fw_evals[0].TotalScore() > fw_evals[1].TotalScore()); // UEFI scores higher than BIOS

    const auto& lang_evals = TechStackEvaluator::GetLanguageEvaluations();
    for (const auto& eval : lang_evals) {
        std::cout << "[Language] " << std::setw(20) << std::left << eval.technology
                  << " Score: " << eval.TotalScore() << " / 25\n";
    }
    assert(lang_evals[0].TotalScore() >= lang_evals[1].TotalScore()); // C++ scores highest for educational balance

    std::cout << "\n[PASS] All 0.5 MikanOS Architecture Roadmap and Tech Stack assertions passed successfully!\n";
    return 0;
}
