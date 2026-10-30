# 🍊 MikanOS: ゼロからのOS自作入門 — 完全自己完結・理論解説 & C++/Python 検証リポジトリ (`study-mikanos`)

![Language](https://img.shields.io/badge/Language-C%2B%2B20%20%7C%20Python%203.10%2B-blue.svg)
![Architecture](https://img.shields.io/badge/Target-x86__64%20%7C%20UEFI-orange.svg)
![Tests](https://img.shields.io/badge/tests-287%20passed-brightgreen.svg)
![Progress](https://img.shields.io/badge/curriculum-16.2%25%20(37%2F228)-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📖 1. 書籍・教材概要

本リポジトリは、内田公太（uchan）氏の名著**『ゼロからのOS自作入門』（マイナビ出版, 2021年, 通称「みかん本」）**および公式リポジトリ（[`uchan-nos/mikanos`](https://github.com/uchan-nos/mikanos)）を題材とした、**完全自己完結型の現代OSアーキテクチャ・C++カーネル実装・対話型シミュレーション学習リポジトリ**です。

- **原著情報**: 内田公太 著『ゼロからのOS自作入門』（マイナビ出版, 2021年3月発売, 全31章＋付録）
- **完全自己完結保証**: 書籍や膨大なハードウェア仕様書（Intel SDM / UEFI Spec / xHCI Spec）を別途開かなくても、**各節の Jupyter Notebook を読むだけで、OSの動作原理・ビット構造・C++実装・アセンブリ連携・落とし穴のすべてが日本語で深く理解できる**ように設計されています。
- **全セル実行エラーゼロ保証**: 収録されているすべての Notebook は全セル正常実行済み（エラーゼロ）であり、対応する単体テスト（`pytest` および C++20 検証コード）が全件パスしています。

---

## 🌟 2. 主な特徴（Key Features）

1. **原著の全31章・全197節・全コラムを完全網羅する詳細 Notebook**
   - UEFI ブートローダ（EDK II）から、GOP ピクセル描画、PCI / USB 3.0 (xHCI) マウス・キーボードドライバ、IDT 割り込み、4階層ページング、マルチウィンドウ GUI、プリエンプティブ・マルチタスク、FAT ファイルシステム、ELF アプリ実行、システムコール、デマンドページング・Copy-on-Write、パイプ通信までを1節単位で徹底解剖します。
2. **Google Colab 完全対応（1クリック即時起動・自動環境セットアップ）**
   - すべてのノートブックの先頭セル（Cell 0）に **Google Colab 自動環境セットアップコード** が配備されています。バッジから開いた後、先頭セルを実行するだけで、リポジトリのクローン・依存ライブラリの導入・共通モジュールのインストール・作業ディレクトリの移動が全自動で完了し、追加のコマンド入力なしで直ちに学習を開始できます。
3. **ハードウェア構造図 & フレームバッファ出力の高解像度再現**
   - メモリマップ、ページテーブル、GDT/IDT、リングバッファ、および MikanOS の GOP フレームバッファ描画結果を `result/` ディレクトリに高解像度（DPI 300）で保存・可視化します。
4. **全章末の演習問題（Exercises）完備**
   - 各章末に理論・ビット計算・C++/Python 実装の穴埋め演習問題 Notebook（`{ch}/{ch}_Exercises.ipynb`）を配置し、`assert` による即時自己採点が可能です。
5. **共通 OS シミュレーション基盤パッケージ（`pip install -e .`）**
   - `common/` ディレクトリに GOP フレームバッファシミュレータ（`FrameBufferSimulator`）、4階層ページング計算機（`PageTableAddress`）、物理フレーム管理（`BitmapMemoryManager`）を提供しています。
6. **包括的単体テストスイート（`pytest` & Modern CMake C++20）**
   - OS コアデータ構造・アドレス計算・ビットフィールド操作の正確性を `tests/` で自動検証します。

---

## 📂 3. ディレクトリ構成

```text
study-mikanos/
├── README.md                  # 📖 公式ポータル・カリキュラム進捗・Colab直通リンク集
├── AGENT.md                   # 🤖 AIエージェント特化実行規範 (Agent.md シンボリックリンク完備)
├── TASK.md                    # 📋 全32章・228ユニット進捗台帳 & Active Lock 管理
├── CMakeLists.txt             # 🛠️ C++20 ホスト側OSコア構造検証ビルド定義
├── pyproject.toml             # 📦 Python 共通パッケージ定義 (pip install -e .)
├── requirements.txt           # 📦 Python 依存ライブラリ一覧
├── pytest.ini                 # 🧪 pytest 設定ファイル
├── common/                    # 🧩 共通 OS シミュレーション・可視化基盤パッケージ
│   ├── __init__.py
│   └── os_sim.py              #    (FrameBufferSimulator, PageTableAddress, BitmapMemoryManager)
├── src/                       # 💻 各章のスタンドアロン C++20 検証コード (src/ch00/, src/ch01/...)
├── tests/                     # ✅ 単体テストスイート (pytest & C++ 検証テスト)
│   └── test_smoke.py
└── 0/ 〜 31/                  # 📚 各章ディレクトリ (Notebook, Exercises, result/ 高解像度図版)
```

---

## 🚀 4. クイックスタート（2大導線）

### 導線 A: Google Colab によるブラウザ即時実行（環境構築ゼロ）
下の「**ノートブック一覧 & Google Colab リンク**」から読みたい節の **[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]** バッジをクリックし、先頭セル（Cell 0）を実行するだけで、ブラウザ上で C++20 コンパイル実行と OS シミュレーション・図版生成がすぐに行えます。

### 導線 B: ローカル環境でのセットアップ & テスト実行

```bash
# 1. リポジトリのクローン
git clone https://github.com/miyayudai/study-mikanos.git
cd study-mikanos

# 2. 依存パッケージと共通モジュールのインストール
pip install -r requirements.txt
pip install -e .

# 3. 単体テストスイートの実行
pytest tests/

# 4. C++20 スタンドアロン検証ターゲットのビルド (CMake)
cmake -B build -S .
cmake --build build
```

---

## 📊 5. 全章カリキュラムと進捗一覧

| Chapter | タイトル | 節数 | 対応タグ範囲 | 主なトピック | 演習問題 | 状態 |
|:---:|---|:---:|:---:|---|:---:|:---:|
| **第0章** | OSって個人で作れるの？ | 5 | - | OSの作り方、POSIX、ブート手順、MikanOS全体像 | [`0_Exercises.ipynb`](0/0_Exercises.ipynb) | `[x]` 完了 (6/6) |
| **第1章** | PCの仕組みとハローワールド | 9 | - | UEFIブート、PE32+/COFFとELF、最小UEFIアプリ | [`1_Exercises.ipynb`](1/1_Exercises.ipynb) | `[x]` 完了 (10/10) |
| **第2章** | EDK II入門とメモリマップ | 9 | `day02a`–`02b` | EDK II、`GetMemoryMap()`、ポインタとアロー演算子 | [`2_Exercises.ipynb`](2/2_Exercises.ipynb) | `[x]` 完了 (10/10) |
| **第3章** | 画面表示の練習とブートローダ | 7 | `day03a`–`03d` | QEMUモニタ、x86-64レジスタ、GOP描画、Red Zone | [`3_Exercises.ipynb`](3/3_Exercises.ipynb) | `[x]` 完了 (8/8) |
| **第4章** | ピクセル描画とmake入門 | 5 | `day04a`–`04d` | Makefile、System V ABI、`PixelWriter`、vtable、ELFローダ | `4_Exercises.ipynb` | `[-]` 進行中 (3/6) |
| **第5章** | 文字表示とコンソールクラス | 6 | `day05a`–`05f` | ビットマップフォント、分割コンパイル、`Console`、`printk` | `5_Exercises.ipynb` | `[ ]` 未着手 (0/7) |
| **第6章** | マウス入力とPCI | 4 | `day06a`–`06c` | マウスカーソル、PCI探索、xHCI初期化とポーリング | `6_Exercises.ipynb` | `[ ]` 未着手 (0/5) |
| **第7章** | 割り込みとFIFO | 10 | `day07a`–`07b` | IDT、MSI割り込み、リングバッファ `ArrayQueue` | `7_Exercises.ipynb` | `[ ]` 未着手 (0/11) |
| **第8章** | メモリ管理 | 7 | `day08a`–`08c` | カーネルスタック移動、GDT、初期ページング、`BitmapMemoryManager` | `8_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第9章** | 重ね合わせ処理 | 7 | `day09a`–`09e` | `sbrk`と`new`、`LayerManager`、差分再描画・スクロール高速化 | `9_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第10章** | ウィンドウ | 7 | `day10a`–`10g` | ウィンドウ描画、シャドウバッファによるチラつき解消、ドラッグ移動 | `10_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第11章** | タイマとACPI | 5 | `day11a`–`11e` | Local APICタイマ、`TimerManager`、RSDPとACPI PMタイマ | `11_Exercises.ipynb` | `[ ]` 未着手 (0/6) |
| **第12章** | キー入力 | 6 | `day12a`–`12f` | FADT、タイマ較正、USBキーボードHID、テキストボックス | `12_Exercises.ipynb` | `[ ]` 未着手 (0/7) |
| **第13章** | マルチタスク(1) | 5 | `day13a`–`13d` | `TaskContext`、`SwitchContext`アセンブリ、`TaskManager` | `13_Exercises.ipynb` | `[ ]` 未着手 (0/6) |
| **第14章** | マルチタスク(2) | 5 | `day14a`–`14d` | スリープ/起床、イベント待機、優先度スケジューラ、`TaskIdle` | `14_Exercises.ipynb` | `[ ]` 未着手 (0/6) |
| **第15章** | ターミナル | 4 | `day15a`–`15d` | メインタスク描画集約、アクティブウィンドウ、`Terminal` | `15_Exercises.ipynb` | `[ ]` 未着手 (0/5) |
| **第16章** | コマンド | 6 | `day16a`–`16f` | キー配送、`echo`/`clear`/`lspci`、コマンド履歴、省電力化 | `16_Exercises.ipynb` | `[ ]` 未着手 (0/7) |
| **第17章** | ファイルシステム | 5 | `day17a`–`17b` | FAT構造、BPB、ディレクトリエントリ、`ls`コマンド | `17_Exercises.ipynb` | `[ ]` 未着手 (0/6) |
| **第18章** | アプリケーション | 4 | `day18a`–`18d` | FATクラスタチェーン、ELFアプリ実行、逆ポーランド計算機 | `18_Exercises.ipynb` | `[ ]` 未着手 (0/5) |
| **第19章** | ページング | 8 | `day19a` | 48bit仮想アドレス、4階層ページング構築と解放 | `19_Exercises.ipynb` | `[ ]` 未着手 (0/9) |
| **第20章** | システムコール | 7 | `day20a`–`20e` | Ring 0/3分離、TSS、CPU例外ハンドラ、`SYSCALL`/`SYSRET` | `20_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第21章** | アプリからウィンドウ | 9 | `day21a`–`21f` | IST、`PutString`、`SyscallExit`、スタック復帰、ウィンドウ生成API | `21_Exercises.ipynb` | `[ ]` 未着手 (0/10) |
| **第22章** | グラフィックとイベント(1) | 7 | `day22a`–`22g` | `exit()`、矩形・ブレゼンハム線分描画、`ReadEvent`キー待機 | `22_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第23章** | グラフィックとイベント(2) | 5 | `day23a`–`23e` | マウスイベント、ペイントソフト、タイマ、ブロック崩しゲーム | `23_Exercises.ipynb` | `[ ]` 未着手 (0/6) |
| **第24章** | 複数のターミナル | 7 | `day24a`–`24g` | 複数ターミナル、タスク別PML4分離、悪意あるアプリからのOS保護 | `24_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第25章** | アプリでファイル読み込み | 3 | `day25a`–`25c` | サブディレクトリ探索、`open`/`read`、`grep`正規表現検索 | `25_Exercises.ipynb` | `[ ]` 未着手 (0/4) |
| **第26章** | アプリでファイル書き込み | 6 | `day26a`–`26d` | `FileDescriptor`抽象化、標準入力、EOF、FATファイル書き込み | `26_Exercises.ipynb` | `[ ]` 未着手 (0/7) |
| **第27章** | アプリのメモリ管理 | 4 | `day27a`–`27d` | デマンドページング、`mmap`、`memstat`、Copy-on-Write (CoW) | `27_Exercises.ipynb` | `[ ]` 未着手 (0/5) |
| **第28章** | 日本語表示とリダイレクト | 3 | `day28a`–`28c` | UTF-8デコーダ、FreeType日本語フォント描画、`>`リダイレクト | `28_Exercises.ipynb` | `[ ]` 未着手 (0/4) |
| **第29章** | アプリ間通信 | 9 | `day29a`–`29d` | 終了コード、パイプ（`PipeDescriptor`）、`sort`、共有メモリ | `29_Exercises.ipynb` | `[ ]` 未着手 (0/10) |
| **第30章** | おまけアプリ | 6 | `day30a`–`30f` | `PATH`解決、`more`/`cat`、閉じるボタン、`tview`、`gview` | `30_Exercises.ipynb` | `[ ]` 未着手 (0/7) |
| **第31章・付録** | これからの道 & 付録 A〜F | 7 | `v1.0` | 発展テーマ、QEMU/Docker環境、Git差分解析、EDK II、iPXE | - | `[ ]` 未着手 (0/7) |

---

## 📓 6. ノートブック一覧 & Google Colab リンク

### 第0章 OSって個人で作れるの？ (`0/`)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`0.1_How_to_Build_an_OS.ipynb`](0/0.1_How_to_Build_an_OS.ipynb) | 0.1 OSの作り方（コード規模の定量比較・既存改造 vs フルスクラッチ・段階的成長曲線） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/0/0.1_How_to_Build_an_OS.ipynb) |
| [`0.2_What_is_an_OS_and_POSIX.ipynb`](0/0.2_What_is_an_OS_and_POSIX.ipynb) | 0.2 そもそもOSって何？（ハードウェア抽象化・リソース管理・Ring 0/3 特権境界・POSIX規格とNewlib） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/0/0.2_What_is_an_OS_and_POSIX.ipynb) |
| [`0.3_Steps_to_Build_an_OS.ipynb`](0/0.3_Steps_to_Build_an_OS.ipynb) | 0.3 OS自作の手順（ビルドパイプライン・フリースタンディング環境 `-mno-red-zone`・UEFIブートメディアとQEMUモニタデバッグ） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/0/0.3_Steps_to_Build_an_OS.ipynb) |
| [`0.4_How_to_Enjoy_OS_Development.ipynb`](0/0.4_How_to_Enjoy_OS_Development.ipynb) | 0.4 OS自作の楽しみ方（スモールステップ開発・QEMU vs 実機ハイブリッド戦略・printfのない世界の5大デバッグ思考法） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/0/0.4_How_to_Enjoy_OS_Development.ipynb) |
| [`0.5_Big_Picture_of_MikanOS.ipynb`](0/0.5_Big_Picture_of_MikanOS.ipynb) | 0.5 OS自作の全体像（全30章・7大フェーズロードマップ・UEFI/x86-64/C++20技術的必然性と4層システムアーキテクチャ） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/0/0.5_Big_Picture_of_MikanOS.ipynb) |
| [`0_Exercises.ipynb`](0/0_Exercises.ipynb) | 第0章 演習問題（OSの役割・Ring 0/3特権分離・Freestanding検証・二分探索デバッガ・C++20ディスパッチャ・総合テスト10問） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/0/0_Exercises.ipynb) |

### 第1章 PCの仕組みとハローワールド (`1/`)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`1.1_Hello_World_Binary.ipynb`](1/1.1_Hello_World_Binary.ipynb) | 1.1 ハローワールド（PE32+ヘッダ構造・x86-64機械語解読・Microsoft x64呼び出し規約とスタック整列・バイナリ合成） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.1_Hello_World_Binary.ipynb) |
| [`1.2_Finding_USB_Device_Names.ipynb`](1/1.2_Finding_USB_Device_Names.ipynb) | 1.2 USBメモリのデバイス名の探し方（Linuxブロックデバイス層・dev_tビット数理・USBエニュメレーション・FAT32配置と誤破壊防止） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.2_Finding_USB_Device_Names.ipynb) |
| [`1.3_Working_with_WSL.ipynb`](1/1.3_Working_with_WSL.ipynb) | 1.3 WSLでのやり方（WSL1 vs WSL2 内部アーキテクチャ・DrvFs/9Pマウント数理・USB/IPパススルー・UEFIブート配置スクリプト） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.3_Working_with_WSL.ipynb) |
| [`1.4_Running_on_QEMU_OVMF.ipynb`](1/1.4_Running_on_QEMU_OVMF.ipynb) | 1.4 エミュレータでのやり方（QEMU仮想PCアーキテクチャ・OVMF Flash分離・仮想FAT32オンザフライ合成とUEFIブートフロー） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.4_Running_on_QEMU_OVMF.ipynb) |
| [`1.5_What_Actually_Happened.ipynb`](1/1.5_What_Actually_Happened.ipynb) | 1.5 結局、何をやったのか？（CPUリセットベクタ0xFFFFFFF0数理・UEFI PI 5大フェーズ・標準フォールバック探索決定木・PE32+再配置と画面描画トレース） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.5_What_Actually_Happened.ipynb) |
| [`1.6_Hands_On_Mindset.ipynb`](1/1.6_Hands_On_Mindset.ipynb) | 1.6 とにかく手を動かそう（実験・観察OODAループ・UTF-16LE文字列改変・ヌル終端欠落とゴミ露出・セクション超過・機械語無限ループ解除） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.6_Hands_On_Mindset.ipynb) |
| [`1.7_Booting_with_UEFI_BIOS.ipynb`](1/1.7_Booting_with_UEFI_BIOS.ipynb) | 1.7 UEFI BIOSによる起動（レガシーBIOS vs UEFI・16bitリアルモードとA20ゲート・MBR 2TiB限界 vs GPT 9.4ZB・Boot/Runtimeサービス分離） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.7_Booting_with_UEFI_BIOS.ipynb) |
| [`1.8_Tools_for_OS_Development.ipynb`](1/1.8_Tools_for_OS_Development.ipynb) | 1.8 OSを作る道具（フリースタンディング環境・クロスコンパイルとターゲットトリプル・LLVM/Clangモジュラー設計・4大必須フラグ） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.8_Tools_for_OS_Development.ipynb) |
| [`1.9_Hello_World_in_C_and_PE_COFF_ELF.ipynb`](1/1.9_Hello_World_in_C_and_PE_COFF_ELF.ipynb) | 1.9 C言語でハローワールド（最小Freestanding C実装・EFI_SYSTEM_TABLE 52Bオフセット数理・Thisポインタ・PE32+ vs ELF64徹底比較） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1.9_Hello_World_in_C_and_PE_COFF_ELF.ipynb) |
| [`1_Exercises.ipynb`](1/1_Exercises.ipynb) | 第1章 演習問題（PE32+/x64 ABI理論・dev_t/LBA数理・UTF-16LE・C++20検証・10問自己採点テスト・レーダーチャート） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/1/1_Exercises.ipynb) |

### 第2章 EDK II入門とメモリマップ (`2/`)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`2.1_Introduction_to_EDK2.ipynb`](2/2.1_Introduction_to_EDK2.ipynb) | 2.1 EDK II入門（EDK IIアーキテクチャ・MdePkgとMikanLoaderPkg・DEC/INF/DSCメタ仕様・依存性注入・6段階ビルドパイプライン） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.1_Introduction_to_EDK2.ipynb) |
| [`2.2_Hello_World_with_EDK2.ipynb`](2/2.2_Hello_World_with_EDK2.ipynb) | 2.2 EDK IIでハローワールド（`osbook_day02a`・`MikanLoaderPkg`構成・`UefiMain`とABI・UTF-16LE・コラム2.1 プリプロセッサ展開と探索パス・インクルードガード） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.2_Hello_World_with_EDK2.ipynb) |
| [`2.3_Main_Memory_Architecture.ipynb`](2/2.3_Main_Memory_Architecture.ipynb) | 2.3 メインメモリ（物理アドレス空間とバス幅数理・RAM vs MMIOのメカニズム・PCIホールとメモリスパース性・レガシー1MB PCマップ・C++20/Pythonシミュレーション） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.3_Main_Memory_Architecture.ipynb) |
| [`2.4_UEFI_Memory_Map_Concept.ipynb`](2/2.4_UEFI_Memory_Map_Concept.ipynb) | 2.4 メモリマップ（メモリマップの概念と役割・EFI_MEMORY_DESCRIPTOR 40B構造体数理・EFI_MEMORY_TYPE全16種分類・ExitBootServices前後のメモリライフサイクル・属性フラグ・可変長ディスクリプタとポインタ演算の罠） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.4_UEFI_Memory_Map_Concept.ipynb) |
| [`2.5_Getting_Memory_Map.ipynb`](2/2.5_Getting_Memory_Map.ipynb) | 2.5 メモリマップの取得（`osbook_day02b`・`gBS->GetMemoryMap()`引数仕様・`EFI_BUFFER_TOO_SMALL`・`struct MemoryMap`設計・16KiB固定バッファ vs 動的確保の動く標的問題・`MapKey`と`ExitBootServices()`整合性・可変長バイトポインタ走査） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.5_Getting_Memory_Map.ipynb) |
| [`2.6_Saving_Memory_Map_to_File.ipynb`](2/2.6_Saving_Memory_Map_to_File.ipynb) | 2.6 メモリマップのファイルへの保存（`osbook_day02b`・`EFI_SIMPLE_FILE_SYSTEM_PROTOCOL`・`EFI_FILE_PROTOCOL`・`OpenRootDir`プロトコル解決チェーン・UTF-16LEパスとOpenモード・`SaveMemoryMap`・可変長バイトポインタ走査・属性マスク `& 0xffffflu`・`Close`フラッシュ） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.6_Saving_Memory_Map_to_File.ipynb) |
| [`2.7_Analyzing_Memory_Map_Output.ipynb`](2/2.7_Analyzing_Memory_Map_Output.ipynb) | 2.7 メモリマップの確認（出力CSVのパース・EfiConventionalMemory空きRAM集計・ExitBootServices後のBootServicesCode/Data回収可能メモリ数理・物理アドレス断片化・MMIOホール・カーネル配置アドレス0x100000の必然性・C++20/Pythonシミュレーション） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.7_Analyzing_Memory_Map_Output.ipynb) |
| [`2.8_Pointers_Part1_Address_and_Pointer.ipynb`](2/2.8_Pointers_Part1_Address_and_Pointer.ipynb) | 2.8 ポインタ入門(1)：アドレスとポインタ（メモリバイト配列数理・64bitポインタサイズ不変性・アドレス演算子 &・リトルエンディアン・デリファレンス *・ポインタ型とアクセス幅・ポインタ算術スケーリング・メモリマップ走査キャスト・VRAM直接描画布石・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.8_Pointers_Part1_Address_and_Pointer.ipynb) |
| [`2.9_Pointers_and_Arrow_Operator.ipynb`](2/2.9_Pointers_and_Arrow_Operator.ipynb) | 2.9 ポインタとアロー演算子（構造体メモリレイアウト・自然アライメントとパディング数理・アロー演算子と(*p).member完全等価性・x86-64オフセット解決・コラム2.2 ポインタのポインタ(二重ポインタT**)・値渡しvsポインタ渡し・UEFI OpenRootDir出力引数パターン・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2.9_Pointers_and_Arrow_Operator.ipynb) |
| [`2_Exercises.ipynb`](2/2_Exercises.ipynb) | 第2章 演習問題（EDK IIビルド構成・メモリディスクリプタ・空きRAM集計・ポインタ算術・アライメント・10問自己採点テスト・レーダーチャート） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/2/2_Exercises.ipynb) |

### 第3章 画面表示の練習とブートローダ (`3/`)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`3.1_QEMU_Monitor_and_Debugging.ipynb`](3/3.1_QEMU_Monitor_and_Debugging.ipynb) | 3.1 QEMUモニタ（非侵襲ハードウェアインスペクタ・GUI/CLI操作モード・`info registers`レジスタ状態解読・`xp`物理メモリダンプとリトルエンディアン数理・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.1_QEMU_Monitor_and_Debugging.ipynb) |
| [`3.2_x86_64_Registers.ipynb`](3/3.2_x86_64_Registers.ipynb) | 3.2 レジスタ（x86-64汎用レジスタ体系・32ビットゼロ拡張の罠とアウトオブオーダー最適化・RIP/RSPスタック動作・RFLAGSフラグ演算数理・CR0/CR3/CR4・ABI呼び出し規約・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.2_x86_64_Registers.ipynb) |
| [`3.3_First_Kernel_and_Red_Zone.ipynb`](3/3.3_First_Kernel_and_Red_Zone.ipynb) | 3.3 初めてのカーネル（ブートローダからの脱皮・1MiB物理配置・ExitBootServicesの主権移譲とMapKeyリトライ・ELF64エントリポイント導出数理・コラム3.1レッドゾーンの罠・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.3_First_Kernel_and_Red_Zone.ipynb) |
| [`3.4_Drawing_Pixels_from_Bootloader.ipynb`](3/3.4_Drawing_Pixels_from_Bootloader.ipynb) | 3.4 ブートローダからピクセルを描く（`osbook_day03b`・UEFI GOPプロトコル探索とGUID解決・`LocateHandleBuffer`/`OpenProtocol`・階層モード構造体・走査線ストライド数理と斜め歪みバグ・VRAM直接ポインタ書き込みによる全画面白色描画・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.4_Drawing_Pixels_from_Bootloader.ipynb) |
| [`3.5_Drawing_Pixels_from_Kernel.ipynb`](3/3.5_Drawing_Pixels_from_Kernel.ipynb) | 3.5 カーネルからピクセルを描く（`osbook_day03c`・ブートローダ主導からカーネル自立への転換・ExitBootServices・引数引き渡し・Microsoft x64 vs System V AMD64 ABI境界問題・EFIAPI省略の奇跡・`reinterpret_cast`・`i % 256` カラーストライプとモアレ数理・HLT特権命令・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.5_Drawing_Pixels_from_Kernel.ipynb) |
| [`3.6_Error_Handling_and_Pointer_Casts.ipynb`](3/3.6_Error_Handling_and_Pointer_Casts.ipynb) | 3.6 エラー処理をしよう（`osbook_day03d`・サイレントクラッシュの撲滅・`EFI_STATUS` 64ビット構造・`EFI_ERROR` 2の補数数理・EDK II `%r` 書式指定・Graceful Degradation `\memmap` エラー無視・`HLT` 特権命令と CPU 省電力状態 `Halt()`・コラム3.2 ポインタのキャストと C++ 4大キャスト型安全性・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.6_Error_Handling_and_Pointer_Casts.ipynb) |
| [`3.7_Pointers_Part2_Assembly_Level.ipynb`](3/3.7_Pointers_Part2_Assembly_Level.ipynb) | 3.7 ポインタ入門(2)：ポインタとアセンブリ言語（CPU物理アドレッシング・MOV命令と間接参照[...]・メモリアクセスサイズ指定子・SIBハードウェア数理・LEA vs MOVの決定的な違い・スタック相対アドレッシングとダングリングポインタ・ピクセルループ逆アセンブル解析・SIMD自動ベクトル化・volatileの不可欠性・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3.7_Pointers_Part2_Assembly_Level.ipynb) |
| [`3_Exercises.ipynb`](3/3_Exercises.ipynb) | 第3章 演習問題（QEMUモニタ・x86-64レジスタ・Red Zone・UEFI GOPストライド数理・引数引き渡しABI・アセンブリSIB・C++20検証・10問自己採点テスト・レーダーチャート） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/3/3_Exercises.ipynb) |

### 第4章 ピクセル描画とmake入門 (`4/`)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`4.1_Introduction_to_Make.ipynb`](4/4.1_Introduction_to_Make.ipynb) | 4.1 make入門（`osbook_day04a`・kernel/Makefile・依存関係DAGとトポロジカルソート数理・タイムスタンプ比較則・自動変数・.PHONY保護・ベアメタルC++ビルドフラグ完全解読） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/4/4.1_Introduction_to_Make.ipynb) |
| [`4.2_Drawing_Pixels_Freely_and_ABI.ipynb`](4/4.2_Drawing_Pixels_Freely_and_ABI.ipynb) | 4.2 ピクセルを自在に描く（`osbook_day04b`・`FrameBufferConfig`構造体・VRAM2次元座標写像数理・スキャンラインストライド・RGB/BGR画素構成・白クリア＆矩形描画・コラム4.1 ABI徹底解剖・System V vs MS x64・EFIAPI省略の真相） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/4/4.2_Drawing_Pixels_Freely_and_ABI.ipynb) |
| [`4.3_Refactoring_with_Cpp_OOP.ipynb`](4/4.3_Refactoring_with_Cpp_OOP.ipynb) | 4.3 C++の機能を使って書き直す（`osbook_day04c`・`PixelWriter`抽象基底クラス・純粋仮想関数・継承コンストラクタ・配置new・フリースタンディング静的バッファ配置・コラム4.2 コンパイルエラーはお友達・vtable分岐予測・C++20/Python検証） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/study-mikanos/blob/main/4/4.3_Refactoring_with_Cpp_OOP.ipynb) |






---

## 💻 7. 共通ライブラリ利用例 (`common`)

`pip install -e .` により、MikanOS のハードウェア抽象化・メモリ管理・フレームバッファ描画を Python から直感的にシミュレートできます。

```python
from common import BitmapMemoryManager, FrameBufferSimulator, PageTableAddress

# 1. x86-64 4階層ページングの仮想アドレス分解 (例: アプリ配置空間 0xffff8000ffff0000)
addr_info = PageTableAddress.from_linear_address(0xFFFF_8000_FFFF_0000)
print(addr_info)
# -> PageTableAddress(pml4_index=256, pdp_index=3, pd_index=511, pt_index=496, offset=0)

# 2. ビットマップ物理メモリマネージャ (4KiB ページフレーム単位の First-Fit 割り当て)
mem_mgr = BitmapMemoryManager(num_frames=1024)
frame_id = mem_mgr.allocate(num_frames=4)
print(f"Allocated 4 frames starting at Frame #{frame_id} (Phys Addr: 0x{frame_id * 4096:x})")

# 3. UEFI GOP フレームバッファ描画シミュレーション
fb = FrameBufferSimulator(width=320, height=200)
fb.fill_rectangle(0, 0, 320, 172, (0, 78, 152))  # MikanOS デスクトップ背景色
fb.save_image("result/demo_desktop.png", title="MikanOS GOP FrameBuffer Simulation")
```

---

## 🛠️ 8. 技術スタック & 動作環境

- **推奨ホストOS**: Ubuntu 24.04 LTS / 22.04 LTS / 20.04 LTS (または Google Colab)
- **C++ コンパイラ**: GCC 13+ / Clang 14+ (`-std=c++20` / `-std=c++17`)
- **Python 環境**: Python 3.10+, NumPy, Matplotlib, Jupyter, `pytest`
- **ターゲットアーキテクチャ**: x86-64 (Intel 64), UEFI (EDK II), QEMU + OVMF

---

## 📜 9. ライセンス

本リポジトリの解説・シミュレーションコードは [MIT License](LICENSE) の下で公開されています。  
参照しているオリジナル MikanOS（[`uchan-nos/mikanos`](https://github.com/uchan-nos/mikanos)）は Apache License 2.0 の下で公開されています。
