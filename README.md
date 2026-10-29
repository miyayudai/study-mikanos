# 🍊 MikanOS: ゼロからのOS自作入門 — 完全自己完結・理論解説 & C++/Python 検証リポジトリ (`study-mikanos`)

![Language](https://img.shields.io/badge/Language-C%2B%2B20%20%7C%20Python%203.10%2B-blue.svg)
![Architecture](https://img.shields.io/badge/Target-x86__64%20%7C%20UEFI-orange.svg)
![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen.svg)
![Progress](https://img.shields.io/badge/curriculum-2.6%25%20(6%2F228)-yellow.svg)
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
| **第1章** | PCの仕組みとハローワールド | 9 | - | UEFIブート、PE32+/COFFとELF、最小UEFIアプリ | `1_Exercises.ipynb` | `[ ]` 未着手 (0/10) |
| **第2章** | EDK II入門とメモリマップ | 9 | `day02a`–`02b` | EDK II、`GetMemoryMap()`、ポインタとアロー演算子 | `2_Exercises.ipynb` | `[ ]` 未着手 (0/10) |
| **第3章** | 画面表示の練習とブートローダ | 7 | `day03a`–`03d` | QEMUモニタ、x86-64レジスタ、GOP描画、Red Zone | `3_Exercises.ipynb` | `[ ]` 未着手 (0/8) |
| **第4章** | ピクセル描画とmake入門 | 5 | `day04a`–`04d` | Makefile、System V ABI、`PixelWriter`、vtable、ELFローダ | `4_Exercises.ipynb` | `[ ]` 未着手 (0/6) |
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
