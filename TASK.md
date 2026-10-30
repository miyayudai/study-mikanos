# ゼロからのOS自作入門 (MikanOS) AI 実装・学習タスクリスト (`TASK.md`)

このドキュメントは、名著『**ゼロからのOS自作入門**』（内田公太 著、マイナビ出版 2021年）および公式リポジトリ（`uchan-nos/mikanos`）に基づき、64ビット自作OS「**MikanOS**」の理論解説、ハードウェア仕様（UEFI / x86-64 / PCI / xHCI / ページング / マルチタスク / FAT / システムコール）の導出、C++20 / Python ハイブリッド検証、図版・フレームバッファ再現、および全章末演習問題の作成・検証を管理するタスク台帳です。

> [!IMPORTANT]
> **学習者のための完全自己完結化方針**: 書籍や外部仕様書を開かなくても、**「ノートブックを読むだけで OS のハードウェア制御・メモリ構造・C++実装・落とし穴のすべてが日本語で深く理解できる」** 圧倒的な濃度の解説 Notebook を作成することを絶対要件とします。

---

## 🔒 現在の実行ステータス (Active Lock)
- **現在実行中のタスク**: なし (アイドル / Standby)
- **担当セクション**: なし
- **ステータス**: アイドル (Standby / ロック解除中)
- **直前完了タスク**: 第2章 2.4 メモリマップ (Issue #21)
※作業を開始する際は、まずこのセクションを更新し、対象タスクを `- [-]` に変更してから着手してください。

---

## 📊 全章構成・進捗概要 (Progress Summary)

| Chapter | タイトル | 節数 | 対応タグ範囲 | 主要トピック・実装機能 | 演習問題 | 状態 |
|:---:|---|:---:|:---:|---|:---:|:---:|
| **第0章** | OSって個人で作れるの？ | 5 | - | OSの役割、POSIX仕様、カーネルとドライバ、MikanOS全体アーキテクチャ | あり | 完了 (6/6) |
| **第1章** | PCの仕組みとハローワールド | 9 | - | UEFI BIOSブート、PE32+/COFFとELF、最小UEFIアプリのバイナリ構造とC実装 | あり | 完了 (10/10) |
| **第2章** | EDK II入門とメモリマップ | 9 | `day02a`–`02b` | EDK IIビルド機構、`GetMemoryMap()`、UEFIメモリディスクリプタとポインタ演算 | あり | 進行中 (4/10) |
| **第3章** | 画面表示の練習とブートローダ | 7 | `day03a`–`03d` | QEMUモニタ、x86-64レジスタ、GOPフレームバッファ、ELFカーネル呼び出し、Red Zone | あり | 未着手 (0/8) |
| **第4章** | ピクセル描画とmake入門 | 5 | `day04a`–`04d` | Makefile、System V AMD64 ABI、`PixelWriter`多態性とvtable、ELFセグメント配置 | あり | 未着手 (0/6) |
| **第5章** | 文字表示とコンソールクラス | 6 | `day05a`–`05f` | ビットマップフォント展開、分割コンパイル、Newlib `sprintf`、`Console`と`printk` | あり | 未着手 (0/7) |
| **第6章** | マウス入力とPCI | 4 | `day06a`–`06c` | マウスカーソル描画、PCIコンフィギュレーション空間探索、xHCI初期化とポーリング | あり | 未着手 (0/5) |
| **第7章** | 割り込みとFIFO | 10 | `day07a`–`07b` | IDT（割り込み記述子表）、MSI割り込み、リングバッファ（`ArrayQueue`）による高速化 | あり | 未着手 (0/11) |
| **第8章** | メモリ管理 | 7 | `day08a`–`08c` | GDT（セグメンテーション）、4階層ページテーブル初期設定、`BitmapMemoryManager` | あり | 未着手 (0/8) |
| **第9章** | 重ね合わせ処理 | 7 | `day09a`–`09e` | `new`演算子とヒープ、`Layer`と`LayerManager`、TSC時間計測と差分描画最適化 | あり | 未着手 (0/8) |
| **第10章** | ウィンドウ | 7 | `day10a`–`10g` | ウィンドウ描画、ダブルバッファリング（`ShadowBuffer`）によるチラつき解消、ドラッグ移動 | あり | 未着手 (0/8) |
| **第11章** | タイマとACPI | 5 | `day11a`–`11e` | Local APICタイマ割り込み、`TimerManager`（優先度付きキュー）、RSDPとACPI PMタイマ | あり | 未着手 (0/6) |
| **第12章** | キー入力 | 6 | `day12a`–`12f` | FADT解析、周波数較正、USBキーボードHIDドライバ、モディファイアキー、テキストボックス | あり | 未着手 (0/7) |
| **第13章** | マルチタスク(1) | 5 | `day13a`–`13d` | コンテキスト構造体（`TaskContext`）、`SwitchContext`アセンブリ、16バイトアライメント | あり | 未着手 (0/6) |
| **第14章** | マルチタスク(2) | 5 | `day14a`–`14d` | タスクのスリープとウェイクアップ、ランキュー、マルチレベル優先度スケジューラ、`HLT`アイドル | あり | 未着手 (0/6) |
| **第15章** | ターミナル | 4 | `day15a`–`15d` | 描画スレッド分離、アクティブウィンドウ制御、`Terminal`タスク、部分再描画高速化 | あり | 未着手 (0/5) |
| **第16章** | コマンド | 6 | `day16a`–`16f` | キーイベント配送、`echo` / `clear` / `lspci` コマンド実装、コマンド履歴リングバッファ | あり | 未着手 (0/7) |
| **第17章** | ファイルシステム | 5 | `day17a`–`17b` | FAT16/32構造、BPB（BIOS Parameter Block）、ディレクトリエントリ解析、`ls`コマンド | あり | 未着手 (0/6) |
| **第18章** | アプリケーション | 4 | `day18a`–`18d` | FATクラスタチェーン追跡、ELFアプリロードと実行、C++計算機アプリ（逆ポーランド記法） | あり | 未着手 (0/5) |
| **第19章** | ページング | 8 | `day19a` | 48ビット仮想アドレスと4階層ページング（PML4/PDP/PD/PT）、アプリ空間分離とクリーンアップ | あり | 未着手 (0/9) |
| **第20章** | システムコール | 7 | `day20a`–`20e` | 特権レベル（Ring 0 / Ring 3）、TSS、例外ハンドラ、`SYSCALL`/`SYSRET`命令とMSR設定 | あり | 未着手 (0/8) |
| **第21章** | アプリからウィンドウ | 9 | `day21a`–`21f` | IST（割り込みスタック表）、`LogString`/`PutString`、アプリ終了（`SyscallExit`）、GUI API | あり | 未着手 (0/10) |
| **第22章** | グラフィックとイベント(1) | 7 | `day22a`–`22g` | `exit()`対応、点・線描画API（ブレゼンハムアルゴリズム）、`ReadEvent`によるキー入力待機 | あり | 未着手 (0/8) |
| **第23章** | グラフィックとイベント(2) | 5 | `day23a`–`23e` | マウスイベント通知、お絵描きソフト（`paint`）、アプリ用タイマ、ブロック崩し（`blocks`） | あり | 未着手 (0/6) |
| **第24章** | 複数のターミナル | 7 | `day24a`–`24g` | マルチターミナル起動、PML4のタスク独立化、アプリの非同期起動、強制終了によるOS保護 | あり | 未着手 (0/8) |
| **第25章** | アプリでファイル読み込み | 3 | `day25a`–`25c` | サブディレクトリ探索、`open`/`read`システムコール（Newlib連携）、`grep`正規表現検索 | あり | 未着手 (0/4) |
| **第26章** | アプリでファイル書き込み | 6 | `day26a`–`26d` | ファイルディスクリプタ抽象化、標準入力、EOF/EOT、FATクラスタ割り当てとファイル書き込み | あり | 未着手 (0/7) |
| **第27章** | アプリのメモリ管理 | 4 | `day27a`–`27d` | デマンドページング（`sbrk`とPage Fault）、`mmap`、メモリ使用量計測、Copy-on-Write (CoW) | あり | 未着手 (0/5) |
| **第28章** | 日本語表示とリダイレクト | 3 | `day28a`–`28c` | UTF-8デコード（バイト列解析）、FreeTypeによる日本語フォント描画、標準出力リダイレクト（`>`） | あり | 未着手 (0/4) |
| **第29章** | アプリ間通信 | 9 | `day29a`–`29d` | 終了コード、パイプ（`PipeDescriptor`と`|`）、`sort`コマンド、共有メモリ | あり | 未着手 (0/10) |
| **第30章** | おまけアプリ | 6 | `day30a`–`30f` | `PATH`解決、`more`/`cat`コマンド、閉じるボタン、テキストビューア（`tview`）、画像表示（`gview`） | あり | 未着手 (0/7) |
| **第31章・付録** | これからの道 & 付録 A〜F | 7 | `v1.0` | OS自作の発展テーマ、QEMU/WSL環境、Git差分調査、EDK II構成、C++テンプレート、iPXE起動 | なし | 未着手 (0/7) |
| **合計** | **全32章構成 (0〜31章+付録)** | **197節** | **`day02a`–`day30f`** | **64-bit UEFI GUI マルチタスクOS 完全網羅** | **全31章分** | **20 / 228 ユニット完了** |

---

## 📋 Task Queue (全章・全節タスクリスト)

### 第0章 OSって個人で作れるの？ (`0/`)
- [x] **0.1 OSの作り方** (`0/0.1_How_to_Build_an_OS.ipynb`)
  - 既存OSの改造 vs フルスクラッチ自作、Linux 0.01・はりぼてOS・MikanOSの規模比較と可視化
- [x] **0.2 そもそもOSって何？ (含 コラム0.1: OSの仕様とPOSIX)** (`0/0.2_What_is_an_OS_and_POSIX.ipynb`)
  - ハードウェア抽象化、リソース管理、カーネルとシェル・アプリの関係、POSIX規格の意義
- [x] **0.3 OS自作の手順** (`0/0.3_Steps_to_Build_an_OS.ipynb`)
  - ソースコードから機械語バイナリ（コンパイル・アセンブル・リンク）、ブートメディア作成と起動フロー
- [x] **0.4 OS自作の楽しみ方** (`0/0.4_How_to_Enjoy_OS_Development.ipynb`)
  - スモールステップ開発、エミュレータ（QEMU）と実機動作、デバッグの思考法
- [x] **0.5 OS自作の全体像** (`0/0.5_Big_Picture_of_MikanOS.ipynb`)
  - MikanOS全30章のアーキテクチャロードマップ、UEFI・x86-64・C++17採用の技術的理由
- [x] **第0章 演習問題** (`0/0_Exercises.ipynb`)

### 第1章 PCの仕組みとハローワールド (`1/`)
- [x] **1.1 ハローワールド** (`1/1.1_Hello_World_Binary.ipynb`)
  - バイナリエディタによる最小UEFIアプリケーション（`BOOTX64.EFI`）の16進数ダンプ解析と構造検証
- [x] **1.2 USBメモリのデバイス名の探し方** (`1/1.2_Finding_USB_Device_Names.ipynb`)
  - Linuxにおけるブロックデバイス管理（`/dev/sdX`, `lsblk`, `dmesg`）とFATファイルシステムへの配置
- [x] **1.3 WSLでのやり方** (`1/1.3_Working_with_WSL.ipynb`)
  - Windows Subsystem for Linux (WSL2) とホストファイルシステム・ドライブマウントの仕組み
- [x] **1.4 エミュレータでのやり方** (`1/1.4_Running_on_QEMU_OVMF.ipynb`)
  - QEMUとOVMF（Open Virtual Machine Firmware: UEFIファームウェア）による仮想PCブートの仕組み
- [x] **1.5 結局、何をやったのか？** (`1/1.5_What_Actually_Happened.ipynb`)
  - CPUリセットベクタからUEFIファームウェア初期化、`/EFI/BOOT/BOOTX64.EFI` 自動検出までのシーケンス
- [x] **1.6 とにかく手を動かそう** (`1/1.6_Hands_On_Mindset.ipynb`)
  - 低レイヤ学習における実験・観察サイクルとバイナリ検証手法
- [x] **1.7 UEFI BIOSによる起動** (`1/1.7_Booting_with_UEFI_BIOS.ipynb`)
  - レガシーBIOS（16ビットリアルモード・MBR 512バイト制限）とUEFI（64ビットLong Mode・GPT/FAT）の決定的な違い
- [x] **1.8 OSを作る道具** (`1/1.8_Tools_for_OS_Development.ipynb`)
  - クロスコンパイラ（Clang/LLVM）、リンカ（LLD）、ヘッダ・ライブラリ非依存（Freestanding環境）の仕組み
- [x] **1.9 C言語でハローワールド (含 コラム1.1: PEとCOFFとELF)** (`1/1.9_Hello_World_in_C_and_PE_COFF_ELF.ipynb`)
  - `EfiMain` と `EFI_SYSTEM_TABLE`、`ConOut->OutputString` の構造体ポインタ呼び出し、PE32+/COFFヘッダとELFのバイナリ構造比較
- [x] **第1章 演習問題** (`1/1_Exercises.ipynb`)

### 第2章 EDK II入門とメモリマップ (`2/`)
- [x] **2.1 EDK II入門** (`2/2.1_Introduction_to_EDK2.ipynb`)
  - UEFI公式開発キット EDK II のアーキテクチャ、パッケージ（`.dec`, `.dsc`, `.inf`）の役割
- [x] **2.2 EDK IIでハローワールド (`osbook_day02a`) (含 コラム2.1: インクルード)** (`2/2.2_Hello_World_with_EDK2.ipynb`)
  - `MikanLoaderPkg` の構成、`UefiMain` 実装とプリプロセッサ `#include` の展開動作
- [x] **2.3 メインメモリ** (`2/2.3_Main_Memory_Architecture.ipynb`)
  - 物理アドレス空間、RAMとMMIO（Memory-Mapped I/O）領域、ファームウェア予約領域の配置
- [x] **2.4 メモリマップ** (`2/2.4_UEFI_Memory_Map_Concept.ipynb`)
  - `EFI_MEMORY_DESCRIPTOR` 構造体のビットレイアウト、`EFI_MEMORY_TYPE` の分類とOSにとっての意味
- [ ] **2.5 メモリマップの取得 (`osbook_day02b`)** (`2/2.5_Getting_Memory_Map.ipynb`)
  - `gBS->GetMemoryMap()` の引数仕様、バッファサイズ不足エラー（`EFI_BUFFER_TOO_SMALL`）への対処とMapKey
- [ ] **2.6 メモリマップのファイルへの保存** (`2/2.6_Saving_Memory_Map_to_File.ipynb`)
  - `EFI_SIMPLE_FILE_SYSTEM_PROTOCOL` と `EFI_FILE_PROTOCOL` を用いたCSV出力処理
- [ ] **2.7 メモリマップの確認** (`2/2.7_Analyzing_Memory_Map_Output.ipynb`)
  - 出力されたメモリマップCSVのパース、空き領域（`EfiConventionalMemory`）の集計とメモリ配置マップ可視化
- [ ] **2.8 ポインタ入門(1)：アドレスとポインタ** (`2/2.8_Pointers_Part1_Address_and_Pointer.ipynb`)
  - 64ビットアドレス空間におけるポインタ変数、アドレス演算子 `&` と間接参照演算子 `*` のメモリモデル
- [ ] **2.9 ポインタとアロー演算子 (含 コラム2.2: ポインタのポインタ)** (`2/2.9_Pointers_and_Arrow_Operator.ipynb`)
  - 構造体ポインタ `->` のオフセット計算と、UEFI APIで頻出する二重ポインタ（`void**`）の完全図解
- [ ] **第2章 演習問題** (`2/2_Exercises.ipynb`)

### 第3章 画面表示の練習とブートローダ (`3/`)
- [ ] **3.1 QEMUモニタ** (`3/3.1_QEMU_Monitor_and_Debugging.ipynb`)
  - `info registers`, `xp` 等による物理メモリ・CPU状態の検査手法
- [ ] **3.2 レジスタ** (`3/3.2_x86_64_Registers.ipynb`)
  - 汎用レジスタ（`RAX`〜`R15`）、命令ポインタ（`RIP`）、スタックポインタ（`RSP`）、制御レジスタ（`CR0`〜`CR4`）
- [ ] **3.3 初めてのカーネル (`osbook_day03a`) (含 コラム3.1: レッドゾーン)** (`3/3.3_First_Kernel_and_Red_Zone.ipynb`)
  - ブートローダでのカーネル読み込み、`ExitBootServices`、関数ポインタによるジャンプ、System V ABI Red Zone（128バイト）の罠と `-mno-red-zone`
- [ ] **3.4 ブートローダからピクセルを描く (`osbook_day03b`)** (`3/3.4_Drawing_Pixels_from_Bootloader.ipynb`)
  - UEFI GOP (`EFI_GRAPHICS_OUTPUT_PROTOCOL`) によるフレームバッファベースアドレスと解像度の取得
- [ ] **3.5 カーネルからピクセルを描く (`osbook_day03c`)** (`3/3.5_Drawing_Pixels_from_Kernel.ipynb`)
  - ブートローダからカーネルへのフレームバッファ情報の引き渡しと、ピクセルフォーマット別描画
- [ ] **3.6 エラー処理をしよう (`osbook_day03d`) (含 コラム3.2: ポインタのキャスト)** (`3/3.6_Error_Handling_and_Pointer_Casts.ipynb`)
  - `EFI_STATUS` のエラーチェック機構、`reinterpret_cast` とCスタイルキャストの違い
- [ ] **3.7 ポインタ入門(2)：ポインタとアセンブリ言語** (`3/3.7_Pointers_Part2_Assembly_Level.ipynb`)
  - ポインタ操作が x86-64 アセンブリ（`MOV [RAX], EDX` 等）にどうコンパイルされるかの逆アセンブル検証
- [ ] **第3章 演習問題** (`3/3_Exercises.ipynb`)

### 第4章 ピクセル描画とmake入門 (`4/`)
- [ ] **4.1 make入門 (`osbook_day04a`)** (`4/4.1_Introduction_to_Make.ipynb`)
  - 依存関係グラフ、ターゲット・必須条件・レシピ、自動変数（`$@`, `$<`, `$^`）とパターンルール
- [ ] **4.2 ピクセルを自在に描く (`osbook_day04b`) (含 コラム4.1: ABI)** (`4/4.2_Drawing_Pixels_Freely_and_ABI.ipynb`)
  - `FrameBufferConfig` 構造体、System V AMD64 ABI と Microsoft x64 ABI のレジスタ渡し規則の違い
- [ ] **4.3 C++の機能を使って書き直す (`osbook_day04c`) (含 コラム4.2: コンパイルエラーはお友達)** (`4/4.3_Refactoring_with_Cpp_OOP.ipynb`)
  - `PixelWriter` 基底クラスと `RGBResv8BitPerColorPixelWriter` / `BGRResv8BitPerColorPixelWriter`、配置 `new`（Placement new）
- [ ] **4.4 vtable** (`4/4.4_Virtual_Method_Table_vtable.ipynb`)
  - 仮想関数テーブル（vtable）と vptr のメモリレイアウト、純粋仮想関数と `__cxa_pure_virtual`
- [ ] **4.5 ローダを改良する (`osbook_day04d`)** (`4/4.5_Loading_ELF_Segments_Properly.ipynb`)
  - ELF形式（`Elf64_Ehdr`, `Elf64_Phdr`）の解析、`PT_LOAD` セグメントの配置と `.bss` ゼロクリア処理
- [ ] **第4章 演習問題** (`4/4_Exercises.ipynb`)

### 第5章 文字表示とコンソールクラス (`5/`)
- [ ] **5.1 文字を書いてみる (`osbook_day05a`) (含 コラム5.1: 参照とポインタ)** (`5/5.1_Drawing_Characters_and_References.ipynb`)
  - 8x16 ビットマップフォントのビット展開アルゴリズム、C++参照（Reference）とポインタの比較
- [ ] **5.2 分割コンパイル (`osbook_day05b`)** (`5/5.2_Separate_Compilation_and_Linking.ipynb`)
  - ヘッダファイル（宣言）とソースファイル（定義）、インクルードガード（`#pragma once`）、シンボル解決
- [ ] **5.3 フォントを増やそう (`osbook_day05c`)** (`5/5.3_Generating_Hankaku_Font_Object.ipynb`)
  - `hankaku.txt` からバイナリオブジェクト（`objcopy`）への変換とリンカシンボル（`_binary_hankaku_bin_start`）参照
- [ ] **5.4 文字列描画とsprintf() (`osbook_day05d`)** (`5/5.4_String_Rendering_and_sprintf.ipynb`)
  - Newlib 標準Cライブラリのリンク、可変長引数と `sprintf` による書式整形
- [ ] **5.5 コンソールクラス (`osbook_day05e`)** (`5/5.5_Implementing_Console_Class.ipynb`)
  - 25行×80列文字バッファ、改行（`\n`）処理と1行スクロール描画ロジック
- [ ] **5.6 printk() (`osbook_day05f`)** (`5/5.6_Implementing_printk.ipynb`)
  - `va_list`, `va_start`, `vsnprintf` を用いたカーネル用書式付き出力関数 `printk` の実装
- [ ] **第5章 演習問題** (`5/5_Exercises.ipynb`)

### 第6章 マウス入力とPCI (`6/`)
- [ ] **6.1 マウスカーソル (`osbook_day06a`)** (`6/6.1_Rendering_Mouse_Cursor.ipynb`)
  - デスクトップ背景描画とマウスカーソルスプライトの描画
- [ ] **6.2 USBホストドライバ** (`6/6.2_USB_3_0_xHCI_Architecture.ipynb`)
  - USB 3.0 と xHCI（eXtensible Host Controller Interface）の階層構造、エンドポイントとリング構造
- [ ] **6.3 PCIデバイスの探索 (`osbook_day06b`)** (`6/6.3_Scanning_PCI_Configuration_Space.ipynb`)
  - I/Oポート（`0xCF8` `CONFIG_ADDRESS`, `0xCFC` `CONFIG_DATA`）、`in`/`out` 命令によるPCIバス全探索とxHCI特定
- [ ] **6.4 ポーリングでマウス入力 (`osbook_day06c`) (含 コラム6.1, 6.2)** (`6/6.4_Mouse_Input_via_xHCI_Polling.ipynb`)
  - MMIOベースアドレス（BAR0）取得、USB HID マウスドライバ登録、ポーリングによるカーソル移動とログレベル制御
- [ ] **第6章 演習問題** (`6/6_Exercises.ipynb`)

### 第7章 割り込みとFIFO (`7/`)
- [ ] **7.1 割り込み (`osbook_day07a`)** (`7/7.1_Concept_of_Interrupts.ipynb`)
  - ポーリング方式のCPU浪費問題と、ハードウェア割り込みによるイベント駆動制御
- [ ] **7.2 割り込みハンドラ** (`7/7.2_Interrupt_Handlers_and_Compiler_Attribute.ipynb`)
  - `__attribute__((interrupt))` と `InterruptFrame`、通常関数呼び出しと割り込みリターン（`IRETQ`）の違い
- [ ] **7.3 割り込みベクタ** (`7/7.3_Interrupt_Vectors_in_x86_64.ipynb`)
  - CPU例外（0〜31）とユーザー定義割り込みベクタ（32〜255）の割り当て設計
- [ ] **7.4 割り込み記述子の設定** (`7/7.4_Configuring_IDT_Descriptors.ipynb`)
  - 16バイト IDT エントリ（`InterruptDescriptor`）のビット構造と `LIDT` 命令による登録
- [ ] **7.5 MSI割り込み** (`7/7.5_Configuring_PCI_MSI_Interrupts.ipynb`)
  - PCIケーパビリティリスト走査、MSI（Message Signaled Interrupts）のアドレス・データレジスタ設定（`0xFEE00000`）
- [ ] **7.6 割り込みのまとめ** (`7/7.6_Interrupt_Flow_Summary.ipynb`)
  - マウス移動から xHCI → MSI → Local APIC → CPU → IDT → ハンドラ起動までの完全トレース
- [ ] **7.7 割り込みハンドラの高速化 (`osbook_day07b`)** (`7/7.7_Fast_Interrupt_Handling_Top_Bottom_Half.ipynb`)
  - 割り込みハンドラ内で重い処理（描画等）を行う危険性と、遅延処理への分離
- [ ] **7.8 FIFOとFILO** (`7/7.8_FIFO_vs_FILO_Data_Structures.ipynb`)
  - イベント処理順序を保証するキュー（FIFO）とスタック（FILO）の比較
- [ ] **7.9 キューの実装** (`7/9_Implementing_Ring_Buffer_ArrayQueue.ipynb`)
  - リングバッファ `ArrayQueue<T>` のポインタ・インデックス循環ロジックと満杯/空判定
- [ ] **7.10 キューを使った割り込み高速化** (`7/7.10_Event_Loop_with_Interrupt_Queue.ipynb`)
  - `CLI` / `STI` と `STI; HLT` による競合回避（Race Condition防止）とイベントループの完成
- [ ] **第7章 演習問題** (`7/7_Exercises.ipynb`)

### 第8章 メモリ管理 (`8/`)
- [ ] **8.1 メモリ管理** (`8/8.1_Role_of_OS_Memory_Management.ipynb`)
  - なぜUEFI任せのままではいけないのか、物理メモリ管理と仮想メモリ管理の全体像
- [ ] **8.2 UEFIメモリマップ (`osbook_day08a`)** (`8/8.2_Inspecting_UEFI_Memory_Map_in_Kernel.ipynb`)
  - カーネル内での `MemoryMap` 走査と利用可能物理メモリ領域の特定
- [ ] **8.3 データ構造の移動 (`osbook_day08b`)** (`8/8.3_Relocating_Kernel_Data_Structures.ipynb`)
  - UEFI領域にあるデータの危険性と、カーネル管理下のメモリ領域への退避
- [ ] **8.4 スタック領域の移動** (`8/8.4_Switching_Kernel_Stack.ipynb`)
  - `.bss` 領域への1MiB独自カーネルスタック確保とアセンブリによる `RSP` 切り替え
- [ ] **8.5 セグメンテーションの設定** (`8/8.5_Setting_Up_GDT_Segmentation.ipynb`)
  - x86-64 における GDT（Global Descriptor Table）、コードセグメント記述子（DPL, Lビット）と `LGDT`
- [ ] **8.6 ページングの設定** (`8/8.6_Setting_Up_Identity_Paging.ipynb`)
  - PML4 / PDP / PD による 2MiB ページ単位の 64GiB 恒等マッピング（Identity Mapping）と `CR3` 設定
- [ ] **8.7 メモリ管理に挑戦 (`osbook_day08c`)** (`8/8.7_Implementing_BitmapMemoryManager.ipynb`)
  - 4KiB ページフレーム単位のビットマップ管理 `BitmapMemoryManager`（First-Fit 探索・確保・解放）の実装と検証
- [ ] **第8章 演習問題** (`8/8_Exercises.ipynb`)

### 第9章 重ね合わせ処理 (`9/`)
- [ ] **9.1 重ね合わせ処理 (`osbook_day09a`)** (`9/9.1_Introduction_to_Layer_Compositing.ipynb`)
  - マウス移動時に背景が消える問題の解決と、レイヤ（層）モデルの設計
- [ ] **9.2 new演算子** (`9/9.2_Enabling_Dynamic_Allocation_sbrk.ipynb`)
  - `BitmapMemoryManager` によるヒープ領域確保と Newlib `sbrk()` 実装による `new` / `std::vector` の有効化
- [ ] **9.3 重ね合わせ処理の原理 (含 コラム9.1: スマートポインタ)** (`9/9.3_Window_Layer_and_LayerManager.ipynb`)
  - `Window`（透過色付きバッファ）、`Layer`、`LayerManager`、および `std::shared_ptr` / `std::unique_ptr` による所有権管理
- [ ] **9.4 重ね合わせ処理の時間計測 (`osbook_day09b`)** (`9/9.4_Profiling_Compositing_Performance.ipynb`)
  - 全画面再描画によるマウス遅延の観察とボトルネック分析
- [ ] **9.5 重ね合わせ処理の高速化 (`osbook_day09c`)** (`9/9.5_Optimizing_Layer_Redraw_Bounding_Box.ipynb`)
  - 移動前後の矩形領域（`Rectangle`）の交差判定（Intersection）による部分再描画最適化
- [ ] **9.6 スクロール処理の時間計測 (`osbook_day09d`)** (`9/9.6_Measuring_Console_Scroll_Time.ipynb`)
  - コンソール1行スクロール時の全文字再描画コストの定量計測
- [ ] **9.7 スクロール処理の高速化 (`osbook_day09e`)** (`9/9.7_Optimizing_Scroll_with_memcpy.ipynb`)
  - ピクセルバッファの `memcpy` / `memmove` 一括転送によるスクロール劇的高速化
- [ ] **第9章 演習問題** (`9/9_Exercises.ipynb`)

### 第10章 ウィンドウ (`10/`)
- [ ] **10.1 もっとマウス (`osbook_day10a`)** (`10/10.1_Refactoring_Mouse_Class.ipynb`)
  - `Mouse` クラスの分離とボタン状態・座標クリッピング処理
- [ ] **10.2 はじめてのウィンドウ (`osbook_day10b`)** (`10/10.2_Drawing_First_GUI_Window.ipynb`)
  - タイトルバー・閉じるボタン・枠線を持つ GUI ウィンドウの描画ロジック
- [ ] **10.3 高速カウンタ (`osbook_day10c`)** (`10/10.3_High_Speed_Counter_Flicker_Problem.ipynb`)
  - ループ内での連続再描画と、画面のチラつき（Flickering / Tearing）が発生するメカニズム
- [ ] **10.4 チラチラ解消 (`osbook_day10d`)** (`10/10.4_Eliminating_Flicker_with_Partial_Draw.ipynb`)
  - 指定レイヤより上のレイヤだけを描画する `Draw(id)` 改良と限界
- [ ] **10.5 バックバッファ (`osbook_day10e`)** (`10/10.5_Double_Buffering_and_Shadow_Buffer.ipynb`)
  - `FrameBuffer` クラスとシャドウバッファ（VRAM直接読み書き回避＋オフスクリーン合成）による完全なチラつき解消
- [ ] **10.6 ウィンドウのドラッグ移動 (`osbook_day10f`)** (`10/10.6_Dragging_Windows_with_Mouse.ipynb`)
  - マウス左ボタン押下判定、最前面を除くレイヤ探索（`FindLayerByPosition`）と追従移動
- [ ] **10.7 ウィンドウだけドラッグ移動 (`osbook_day10g`)** (`10/10.7_Draggable_Layer_Flags.ipynb`)
  - デスクトップ背景レイヤがドラッグされてしまうバグの修正（`draggable_` フラグ設計）
- [ ] **第10章 演習問題** (`10/10_Exercises.ipynb`)

### 第11章 タイマとACPI (`11/`)
- [ ] **11.1 ソースコード整理 (`osbook_day11a`)** (`11/11.1_Kernel_Modularization.ipynb`)
  - 肥大化した `main.cpp` から各サブシステム初期化処理の分離とヘッダ整理
- [ ] **11.2 タイマ割り込み (`osbook_day11b`)** (`11/11.2_Local_APIC_Timer_Interrupts.ipynb`)
  - Local APIC タイマのレジスタ構造（Divide Config, LVT Timer, Initial Count）と周期割り込み発生
- [ ] **11.3 細かく時間を計る (`osbook_day11c`) (含 コラム11.1: volatileの必要性)** (`11/11.3_Tick_Counter_and_Volatile_Keyword.ipynb`)
  - `TimerManager::Tick()` と、コンパイラ最適化による無限ループ化を防ぐ `volatile` の意味
- [ ] **11.4 複数のタイマとタイムアウト通知 (`osbook_day11d`)** (`11/11.4_Priority_Queue_TimerManager.ipynb`)
  - `std::priority_queue` を用いた複数論理タイマの効率的管理とメインキューへのイベント通知
- [ ] **11.5 ACPI PMタイマとRSDP (`osbook_day11e`)** (`11/11.5_ACPI_Tables_and_RSDP_Validation.ipynb`)
  - ACPI（Advanced Configuration and Power Interface）仕様、UEFI設定テーブルからの RSDP / XSDT 探索とチェックサム検証
- [ ] **第11章 演習問題** (`11/11_Exercises.ipynb`)

### 第12章 キー入力 (`12/`)
- [ ] **12.1 FADTを探す (`osbook_day12a`)** (`12/12.1_Locating_ACPI_FADT.ipynb`)
  - XSDT からのシグネチャ `"FACP"`（FADT）探索と PM タイマ I/O ポートアドレス取得
- [ ] **12.2 ACPI PMタイマを使う (`osbook_day12b`)** (`12/12.2_Calibrating_LAPIC_Timer_with_ACPI_PM.ipynb`)
  - 3.579545MHz 固定周波数の ACPI PM タイマ（24bit/32bit）を用いた Local APIC タイマ周波数の実測較正
- [ ] **12.3 USBキーボードドライバ (`osbook_day12c`)** (`12/12.3_USB_HID_Keyboard_Driver.ipynb`)
  - USB HID Boot Protocol キーボードの 8バイト入力レポート解析とキーコード→ASCII変換テーブル
- [ ] **12.4 モディファイアキー (`osbook_day12d`)** (`12/12.4_Handling_Modifier_Keys_Shift_Ctrl.ipynb`)
  - Shift / Ctrl / Alt モディファイアビット判定と大文字・記号入力への対応
- [ ] **12.5 テキストボックス (`osbook_day12e`)** (`12/12.5_Drawing_Interactive_Textbox.ipynb`)
  - 陰影付きテキストボックス描画とキーボード入力文字のリアルタイム追記
- [ ] **12.6 カーソル (`osbook_day12f`)** (`12/12.6_Blinking_Text_Cursor_with_Timer.ipynb`)
  - 0.5秒周期のタイマイベントを利用したテキストカーソル点滅制御
- [ ] **第12章 演習問題** (`12/12_Exercises.ipynb`)

### 第13章 マルチタスク(1) (`13/`)
- [ ] **13.1 マルチタスクとコンテキスト** (`13/13.1_Multitasking_and_CPU_Context.ipynb`)
  - 時分割多重（Time Sharing）の原理と、タスク切り替え時に保存すべきCPUコンテキストの定義
- [ ] **13.2 コンテキストの切り替えに挑戦 (`osbook_day13a`) (含 コラム13.1: スタックのアライメント制約)** (`13/13.2_Manual_Context_Switch_and_Stack_Alignment.ipynb`)
  - 64バイト整列 `TaskContext`（CR3, RIP, RFLAGS, 汎用レジスタ, FXSAVE SSE領域）、`SwitchContext` アセンブリ、x86-64 ABIの `RSP % 16 == 8` アライメント規約
- [ ] **13.3 コンテキストスイッチの自動化 (`osbook_day13b`)** (`13/13.3_Preemptive_Multitasking_with_Timer.ipynb`)
  - 20ミリ秒周期の専用タイマによるプリエンプティブ（強制的）コンテキストスイッチ
- [ ] **13.4 マルチタスクの検証 (`osbook_day13c`)** (`13/13.4_Verifying_Concurrent_Execution.ipynb`)
  - 並行動作時の共有リソース競合（`printk` / `CLI`・`STI`）とデッドロック・画面破壊の防止
- [ ] **13.5 タスクを増やす (`osbook_day13d`)** (`13/13.5_Implementing_Task_and_TaskManager.ipynb`)
  - 任意の数のタスクを生成・ラウンドロビン切り替えできる `Task` / `TaskManager` クラスの実装
- [ ] **第13章 演習問題** (`13/13_Exercises.ipynb`)

### 第14章 マルチタスク(2) (`14/`)
- [ ] **14.1 スリープしてみる (`osbook_day14a`)** (`14/14.1_Task_Sleep_and_Run_Queue.ipynb`)
  - 実行可能キュー（`running_`）からの除外によるタスクスリープとCPU負荷削減
- [ ] **14.2 イベントが来たら起床する (`osbook_day14b`)** (`14/14.2_Per_Task_Event_Queue_and_Wakeup.ipynb`)
  - タスク固有のイベントキュー（`msgs_`）と `SendMessage` による自動ウェイクアップ機構
- [ ] **14.3 性能測定** (`14/14.3_Benchmarking_Task_Sleep_Efficiency.ipynb`)
  - ビジーループ時とスリープ導入時のカウンタ性能比較検証
- [ ] **14.4 タスクに優先度を付ける (`osbook_day14c`)** (`14/14.4_Multi_Level_Priority_Scheduling.ipynb`)
  - 複数レベルのランキューを持つ優先度ベーススケジューラ（Multi-level Priority Queue）の設計と実装
- [ ] **14.5 アイドルタスク (`osbook_day14d`)** (`14/14.5_Idle_Task_with_HLT.ipynb`)
  - 全タスクがスリープした際の番兵となる最低優先度 `TaskIdle`（`hlt` ループ）の導入
- [ ] **第14章 演習問題** (`14/14_Exercises.ipynb`)

### 第15章 ターミナル (`15/`)
- [ ] **15.1 ウィンドウ描画はメインスレッドで (`osbook_day15a`)** (`15/15.1_Delegating_Window_Draw_to_Main_Task.ipynb`)
  - マルチタスク下での `LayerManager::Draw` 競合問題と、`kLayer` メッセージによるメインタスク集約
- [ ] **15.2 アクティブウィンドウ (`osbook_day15b`) (含 コラム15.1)** (`15/15.2_Active_Window_Focus_Management.ipynb`)
  - クリックされたウィンドウのフォーカス切り替えとタイトルバー色変更（`ToplevelWindow`）
- [ ] **15.3 ターミナルウィンドウ (`osbook_day15c`)** (`15/15.3_Creating_Terminal_Task_and_Window.ipynb`)
  - 独立タスクとして動く `Terminal` クラスとプロンプト `>`・点滅カーソルの実装
- [ ] **15.4 描画の高速化 (`osbook_day15d`)** (`15/15.4_Partial_Area_Redraw_in_LayerMessage.ipynb`)
  - ウィンドウ全体ではなく変更された文字矩形領域のみを再描画するメッセージ拡張
- [ ] **第15章 演習問題** (`15/15_Exercises.ipynb`)

### 第16章 コマンド (`16/`)
- [ ] **16.1 ターミナルでキー入力 (`osbook_day16a`)** (`16/16.1_Routing_Key_Events_to_Active_Window.ipynb`)
  - アクティブウィンドウに紐づくタスクへのキー入力イベントルーティングとバックスペース処理
- [ ] **16.2 echoコマンド (`osbook_day16b`)** (`16/16.2_Command_Line_Parsing_and_Echo.ipynb`)
  - エンターキー押下時のコマンドライン文字列分割と `echo` コマンドの実装
- [ ] **16.3 clearコマンド (`osbook_day16c`)** (`16/16.3_Implementing_Clear_Command.ipynb`)
  - ターミナル表示領域のクリアとカーソル座標リセット
- [ ] **16.4 lspciコマンド (`osbook_day16d`)** (`16/16.4_Implementing_lspci_Command.ipynb`)
  - 検出済みPCIデバイス一覧のフォーマット出力
- [ ] **16.5 コマンド履歴 (`osbook_day16e`)** (`16/16.5_Command_History_with_Up_Down_Keys.ipynb`)
  - `std::deque` を用いた上下矢印キーによるコマンドヒストリ呼び出し
- [ ] **16.6 省電力化 (`osbook_day16f`)** (`16/16.6_Power_Saving_when_Inactive.ipynb`)
  - 非アクティブウィンドウのカーソル点滅タイマ停止による省電力・省リソース化
- [ ] **第16章 演習問題** (`16/16_Exercises.ipynb`)

### 第17章 ファイルシステム (`17/`)
- [ ] **17.1 ファイルとファイルシステム** (`17/17.1_Introduction_to_File_Systems_and_FAT.ipynb`)
  - ブロックストレージ上のデータ管理抽象化と FAT（File Allocation Table）の歴史・特徴
- [ ] **17.2 BIOSパラメータブロック** (`17/17.2_FAT_BIOS_Parameter_Block_BPB.ipynb`)
  - ブートセクタ先頭 `BPB` 構造体のフィールド解析と予約領域・FAT領域・データ領域のオフセット計算式
- [ ] **17.3 ディレクトリエントリ** (`17/17.3_FAT_Directory_Entry_Structure.ipynb`)
  - 32バイト `DirectoryEntry`（8.3形式ファイル名、属性 `Attr`、開始クラスタ番号、ファイルサイズ）のビット構造
- [ ] **17.4 ボリュームを読み出す (`osbook_day17a`) (含 コラム17.1)** (`17/17.4_Loading_FAT_Volume_into_Memory.ipynb`)
  - ブートローダによるディスクイメージ読み込みとカーネルへの引き渡し
- [ ] **17.5 lsコマンド (`osbook_day17b`)** (`17/17.5_Implementing_ls_Command.ipynb`)
  - ルートディレクトリクラスタの走査とファイル名一覧表示
- [ ] **第17章 演習問題** (`17/17_Exercises.ipynb`)

### 第18章 アプリケーション (`18/`)
- [ ] **18.1 ファイルアロケーションテーブル (`osbook_day18a`)** (`18/18.1_Traversing_FAT_Cluster_Chains.ipynb`)
  - FAT32 テーブルのクラスタチェーン追跡（`NextCluster`）と終端判定（`>= 0x0FFFFFF8`）
- [ ] **18.2 初アプリ (`osbook_day18b`)** (`18/18.2_Loading_and_Executing_First_ELF_App.ipynb`)
  - ファイル名検索（`FindFile`）、ELFファイルのメモリ展開とエントリポイント呼び出し
- [ ] **18.3 C++で計算機 (`osbook_day18c`)** (`18/18.3_RPN_Calculator_App_in_Cpp.ipynb`)
  - 逆ポーランド記法（RPN）計算機アプリ `rpn` の作成とコマンドライン引数（`argc`, `argv`）のスタック構築
- [ ] **18.4 標準ライブラリ (`osbook_day18d`)** (`18/18.4_Linking_Standard_Library_to_Apps.ipynb`)
  - アプリケーションへの Newlib / libc++ リンクとシステムコールスタブの必要性
- [ ] **第18章 演習問題** (`18/18_Exercises.ipynb`)

### 第19章 ページング (`19/`)
- [ ] **19.1 実行ファイルとメモリアドレス** (`19/19.1_Address_Collision_Problem_without_Paging.ipynb`)
  - 物理アドレス直指定での複数アプリ実行や不正メモリアクセスの問題点
- [ ] **19.2 アドレス変換 (含 コラム19.1: 事実上のアドレス)** (`19/19.2_Virtual_to_Physical_Address_Translation.ipynb`)
  - MMU（Memory Management Unit）による仮想アドレスから物理アドレスへの変換原理と正規アドレス（Canonical Address）
- [ ] **19.3 アプリのロードと実行 (`osbook_day19a`)** (`19/19.3_Overview_of_App_Paging_Execution.ipynb`)
  - 仮想アドレス空間 `0xffff8000ffff0000` 以降へのアプリ配置の全体設計
- [ ] **19.4 仮想アドレスと4階層ページング** (`19/19.4_4_Level_Paging_Bit_Structure.ipynb`)
  - 48ビット仮想アドレスの9ビット×4階層（PML4, PDP, PD, PT）＋12ビットオフセット分解と `PageMapEntry` フラグ
- [ ] **19.5 アプリケーションを後半で動かす** (`19/19.5_Canonical_Higher_Half_Address_Space.ipynb`)
  - カーネル空間（前半 `0x0000...`）とアプリケーション空間（後半 `0xffff8000...`）の分離設計
- [ ] **19.6 アプリケーションのロード** (`19/19.6_Loading_ELF_with_Dynamic_Page_Allocation.ipynb`)
  - ELF `PT_LOAD` セグメント範囲の計算とオンデマンドなページテーブル・物理フレーム割り当て
- [ ] **19.7 階層ページング構造の設定** (`19/19.7_Implementing_SetupPageMaps.ipynb`)
  - 再帰的な4階層ページテーブル構築アルゴリズム（`SetupPageMaps`）のステップ導出と検証
- [ ] **19.8 階層ページング構造の片付け (含 コラム19.2: 構造化束縛)** (`19/19.8_Cleaning_Up_Page_Tables_and_Structured_Bindings.ipynb`)
  - アプリ終了時の物理フレーム・ページテーブル全解放（`CleanPageMaps`）とC++17構造化束縛
- [ ] **第19章 演習問題** (`19/19_Exercises.ipynb`)

### 第20章 システムコール (`20/`)
- [ ] **20.1 アプリがOSの機能を使う方法 (`osbook_day20a`)** (`20/20.1_How_Apps_Invoke_OS_Services.ipynb`)
  - カーネル内部関数の直接呼び出しが危険な理由と、システムコールインターフェースの役割
- [ ] **20.2 OSを守ろう(1) (`osbook_day20b`)** (`20/20.2_CPU_Privilege_Levels_Ring0_and_Ring3.ipynb`)
  - CPU特権レベル（Ring 0 カーネルモード vs Ring 3 ユーザーモード）、GDTユーザーセグメント、ページテーブルの `User` ビットと `RETFIQ` / `SYSRET`
- [ ] **20.3 TSSを設定しよう (`osbook_day20c`)** (`20/20.3_Task_State_Segment_TSS_and_Ring_Transition.ipynb`)
  - Ring 3 実行中の割り込み発生時に Ring 0 スタック（`RSP0`）へ切り替える TSS（Task State Segment）と `LTR` 命令
- [ ] **20.4 バグ発見を手伝おう (`osbook_day20d`)** (`20/20.4_CPU_Exception_Handlers_PageFault_GPF.ipynb`)
  - `#PF`（ページフォルト, 14番）や `#GP`（一般保護例外, 13番）の例外ハンドラ登録とエラーコード・`CR2` 解析
- [ ] **20.5 システムコール (`osbook_day20e`)** (`20/20.5_SYSCALL_and_SYSRET_Architecture.ipynb`)
  - 高速システムコール命令 `SYSCALL` / `SYSRET` の動作仕様とレジスタ変化（`RCX`, `R11`, `RAX`）
- [ ] **20.6 システムコールの登録処理** (`20/20.6_Configuring_MSRs_IA32_LSTAR_STAR_FMASK.ipynb`)
  - MSR（Model Specific Register: `IA32_EFER`, `IA32_LSTAR`, `IA32_STAR`, `IA32_FMASK`）への `WRMSR` 設定
- [ ] **20.7 システムコールの本体** (`20/20.7_SyscallEntry_Assembly_and_Table_Dispatch.ipynb`)
  - `SyscallEntry` アセンブリでのOSスタック切り替えと、`syscall_table` による C++ 関数ディスパッチ
- [ ] **第20章 演習問題** (`20/20_Exercises.ipynb`)

### 第21章 アプリからウィンドウ (`21/`)
- [ ] **21.1 ISTを設定しよう (`osbook_day21a`)** (`21/21.1_Interrupt_Stack_Table_IST.ipynb`)
  - システムコール処理中のタイマ割り込みでユーザースタックを破壊しないための IST（Interrupt Stack Table）設定
- [ ] **21.2 文字列表示システムコール (`osbook_day21b`)** (`21/21.2_Terminal_Print_Syscall_PutString.ipynb`)
  - アプリ標準出力を現在のターミナルウィンドウへ描画する `SyscallPutString`
- [ ] **21.3 システムコールの作成** (`21/21.3_Designing_Syscall_Arguments_and_Return.ipynb`)
  - システムコールABI（`RAX` 番号、`RDI`, `RSI`, `RDX`, `R10`, `R8`, `R9` 引数）と64ビット戻り値構造体 `SyscallResult`
- [ ] **21.4 write()の作成** (`21/21.4_Connecting_Newlib_write_to_Syscall.ipynb`)
  - Newlib システムコールスタブ `write()` から `SyscallPutString` の呼び出しと `printf` の動作完成
- [ ] **21.5 終了システムコール (`osbook_day21c`)** (`21/21.5_Implementing_SyscallExit.ipynb`)
  - アプリケーションから安全にカーネルへ制御を戻す `SyscallExit`
- [ ] **21.6 スタックポインタの復帰** (`21/21.6_Saving_and_Restoring_OS_Stack_Pointer.ipynb`)
  - `CallApp` 時のカーネルレジスタ・`os_stack_ptr` 保存と `ExitApp` による長距離ジャンプ復帰
- [ ] **21.7 コード整理 (`osbook_day21d`)** (`21/21.7_Shared_App_Support_Files.ipynb`)
  - 各アプリ共通の `syscall.asm` / `newlib_support.c` の共通ビルド化
- [ ] **21.8 ウィンドウを開く (`osbook_day21e`)** (`21/21.8_SyscallOpenWindow_for_GUI_Apps.ipynb`)
  - アプリから新規 GUI ウィンドウを生成してレイヤIDを返す `SyscallOpenWindow`
- [ ] **21.9 ウィンドウに文字を書く (`osbook_day21f`)** (`21/21.9_SyscallWinWriteString.ipynb`)
  - レイヤID指定によるアプリウィンドウへの文字列描画システムコール
- [ ] **第21章 演習問題** (`21/21_Exercises.ipynb`)

### 第22章 グラフィックとイベント(1) (`22/`)
- [ ] **22.1 exit()を使う (`osbook_day22a`)** (`22/22.1_Supporting_Standard_exit_Function.ipynb`)
  - C標準ライブラリ `exit()` と `_exit()` システムコールスタブの実装
- [ ] **22.2 点を描く (`osbook_day22b`)** (`22/22.2_SyscallWinFillRectangle_and_Pixel_Drawing.ipynb`)
  - `SyscallWinFillRectangle` とマンデルブロ集合・グラデーション描画アプリ
- [ ] **22.3 タイマ値の取得 (`osbook_day22c`)** (`22/22.3_SyscallGetCurrentTick.ipynb`)
  - OS起動からの経過ティック値を取得する `SyscallGetCurrentTick`
- [ ] **22.4 ウィンドウ描画の最適化 (`osbook_day22d`)** (`22/22.4_Deferred_Redraw_Flags_in_WinSyscalls.ipynb`)
  - ピクセル単位の再描画を防ぎ、バッチ描画後に一括 `SyscallWinRedraw` する描画最適化
- [ ] **22.5 線を引く (`osbook_day22e`)** (`22/22.5_Bresenham_Line_Drawing_Algorithm.ipynb`)
  - ブレゼンハムの線分描画アルゴリズムの数理導出と `SyscallWinDrawLine` 実装
- [ ] **22.6 ウィンドウのクローズ (`osbook_day22f`)** (`22/22.6_SyscallCloseWindow.ipynb`)
  - アプリ終了時のウィンドウレイヤ破棄処理
- [ ] **22.7 キー入力を待つ (`osbook_day22g`)** (`22/22.7_SyscallReadEvent_and_AppEvent.ipynb`)
  - アプリがキー入力イベントをブロック待機して受け取る `SyscallReadEvent` と `AppEvent` 共用体
- [ ] **第22章 演習問題** (`22/22_Exercises.ipynb`)

### 第23章 グラフィックとイベント(2) (`23/`)
- [ ] **23.1 マウス入力 (`osbook_day23a`)** (`23/23.1_Routing_Mouse_Events_to_App_Windows.ipynb`)
  - ウィンドウローカル座標への変換と、ドラッグ・ボタン押下イベントのアプリ通知
- [ ] **23.2 お絵描きソフト (`osbook_day23b`)** (`23/23.2_Building_Paint_App.ipynb`)
  - マウスイベントと線分描画を組み合わせたペイントアプリ `paint` の実装
- [ ] **23.3 タイマコマンド (`osbook_day23c`)** (`23/23.3_SyscallCreateTimer_Absolute_and_Relative.ipynb`)
  - アプリ向け相対/絶対モード対応 `SyscallCreateTimer` と `timer` コマンド
- [ ] **23.4 アニメーション (`osbook_day23d`)** (`23/23.4_Frame_Rate_Control_and_Eye_Animation.ipynb`)
  - 周期タイマイベントを用いた滑らかなアニメーション描画（`eye` アプリ）
- [ ] **23.5 ブロック崩しゲーム (`osbook_day23e`)** (`23/23.5_Building_Breakout_Game_blocks.ipynb`)
  - キー入力・タイマ・衝突判定・ビット演算ブロック管理を統合した GUI ゲーム `blocks` の実装とシミュレーション
- [ ] **第23章 演習問題** (`23/23_Exercises.ipynb`)

### 第24章 複数のターミナル (`24/`)
- [ ] **24.1 ターミナルを増やす (`osbook_day24a`)** (`24/24.1_Spawning_Multiple_Terminals_with_F2.ipynb`)
  - `F2` ショートカットキーによる新規ターミナルタスクの動的生成
- [ ] **24.2 カーソル点滅を自分で (`osbook_day24b`)** (`24/24.2_Per_Terminal_Cursor_Timer.ipynb`)
  - 各ターミナルタスクが自身のIDに紐づくタイマで独立してカーソル点滅を行う設計
- [ ] **24.3 複数アプリの同時起動 (`osbook_day24c`)** (`24/24.3_Per_Task_PML4_Page_Table_Isolation.ipynb`)
  - タスクごとに独立した PML4 テーブルを持たせることで、同一仮想アドレス空間での複数アプリ同時実行を実現
- [ ] **24.4 ウィンドウの重なりのバグ修正 (`osbook_day24d`)** (`24/24.4_Fixing_Window_Z_Order_on_Close.ipynb`)
  - ウィンドウクローズ時のアクティブウィンドウ状態・レイヤ前後関係の整合性維持
- [ ] **24.5 ターミナル無しのアプリ起動 (`osbook_day24e`)** (`24/24.5_Launching_Standalone_App_with_Shift_F2.ipynb`)
  - `Shift+F2` または `run` コマンドによるバックグラウンド/スタンドアロンアプリ起動
- [ ] **24.6 OSをフリーズさせるアプリ (`osbook_day24f`)** (`24/24.6_Denial_of_Service_by_Malicious_Apps.ipynb`)
  - `cli` 特権命令実行や無限ループ、不正ポインタ渡しによってOSがハング・クラッシュする脆弱性の検証
- [ ] **24.7 OSを守ろう(2) (`osbook_day24g`)** (`24/24.7_Protecting_Kernel_and_Force_Killing_Apps.ipynb`)
  - CPU例外発生時のアプリ強制終了、システムコール引数のポインタ境界チェック、`F4` キーによるフリーズアプリの強制キル
- [ ] **第24章 演習問題** (`24/24_Exercises.ipynb`)

### 第25章 アプリでファイル読み込み (`25/`)
- [ ] **25.1 ディレクトリ対応 (`osbook_day25a`)** (`25/25.1_Hierarchical_Directory_Path_Resolution.ipynb`)
  - スラッシュ区切りパス（`/dir/file`）の階層探索アルゴリズムとサブディレクトリ対応 `FindFile`
- [ ] **25.2 ファイル読み込み (`osbook_day25b`)** (`25/25.2_File_Descriptors_open_and_read_Syscalls.ipynb`)
  - タスクごとのファイルディスクリプタ表（`files_`）、`FileDescriptor`、`open`/`read` システムコールと `cat` アプリ
- [ ] **25.3 正規表現検索 (`osbook_day25c`)** (`25/25.3_Implementing_grep_with_std_regex.ipynb`)
  - `std::ifstream` と `std::regex` を用いたパターン検索アプリ `grep` と `posix_memalign` 対応
- [ ] **第25章 演習問題** (`25/25_Exercises.ipynb`)

### 第26章 アプリでファイル書き込み (`26/`)
- [ ] **26.1 標準入力 (`osbook_day26a`)** (`26/26.1_Standard_Input_and_Keyboard_Forwarding.ipynb`)
  - 標準入力（`fd = 0`）の概念とターミナルからの文字入力ストリーム
- [ ] **26.2 ファイルディスクリプタの抽象化** (`26/26.2_Polymorphic_FileDescriptor_Hierarchy.ipynb`)
  - `FileDescriptor` 抽象基底クラスからの `FatFileDescriptor` と `TerminalFileDescriptor` の派生設計
- [ ] **26.3 キーボード入力を受け取る** (`26/26.3_Implementing_TerminalFileDescriptor_Read.ipynb`)
  - アプリ実行中のキーイベントを `TerminalFileDescriptor::Read` で受け取る仕組み
- [ ] **26.4 EOFとEOT (`osbook_day26b`)** (`26/26.4_Handling_Ctrl_D_EOT_and_EOF.ipynb`)
  - `Ctrl+D`（End of Transmission: `0x04`）による標準入力の EOF 通知処理
- [ ] **26.5 ファイル書き込み(1) (`osbook_day26c`)** (`26/26.5_Creating_New_Files_and_FAT_Cluster_Allocation.ipynb`)
  - `O_CREAT` による新規ディレクトリエントリ作成と、空きFATクラスタ（`0x00000000`）の探索・チェーン構築
- [ ] **26.6 ファイル書き込み(2) (`osbook_day26d`)** (`26/26.6_Implementing_FatFileDescriptor_Write.ipynb`)
  - クラスタ境界をまたぐ `FatFileDescriptor::Write` の実装とファイルサイズ更新
- [ ] **第26章 演習問題** (`26/26_Exercises.ipynb`)

### 第27章 アプリのメモリ管理 (`27/`)
- [ ] **27.1 デマンドページング (`osbook_day27a`)** (`27/27.1_Demand_Paging_and_Page_Fault_Handler.ipynb`)
  - 仮想アドレス範囲のみ予約し、実際のメモリアクセスで `#PF`（Page Fault）が起きた瞬間に物理フレームを割り当てる遅延確保（Demand Paging）
- [ ] **27.2 メモリマップトファイル (`osbook_day27b`)** (`27/27.2_Memory_Mapped_Files_SyscallMapFile.ipynb`)
  - ファイルをページ単位で仮想アドレス空間に直接マッピングする `SyscallMapFile`（`mmap`）の実装
- [ ] **27.3 メモリ使用量を測ろう (`osbook_day27c`)** (`27/27.3_Measuring_Physical_Memory_Consumption_memstat.ipynb`)
  - `memstat` コマンドによる使用中フレーム数とデマンドページング節約効果の定量測定
- [ ] **27.4 コピーオンライト (`osbook_day27d`)** (`27/27.4_Copy_on_Write_CoW_Paging.ipynb`)
  - 読み取り専用共有マッピングと、書き込み時の Page Fault（エラーコード `Write` ビット）を契機とした物理ページ複製（Copy-on-Write）の完全実装
- [ ] **第27章 演習問題** (`27/27_Exercises.ipynb`)

### 第28章 日本語表示とリダイレクト (`28/`)
- [ ] **28.1 日本語と文字コード (`osbook_day28a`)** (`28/28.1_UTF8_Encoding_and_Codepoint_Decoding.ipynb`)
  - ASCII / Shift_JIS / Unicode (UTF-8) のビット構造と、1〜4バイト UTF-8 シーケンスから `char32_t` コードポイントへのデコーダ実装
- [ ] **28.2 日本語フォント (`osbook_day28b`)** (`28/28.2_Rendering_Japanese_Fonts_with_FreeType.ipynb`)
  - FreeType ライブラリと IPAexゴシック（TrueType/OpenTypeアウトラインフォント）による日本語グリフのラスタライズ描画
- [ ] **28.3 リダイレクト (`osbook_day28c`)** (`28/28.3_Standard_Output_Redirection.ipynb`)
  - シェルにおける `>` リダイレクト構文解析と、標準出力（`fd = 1`）のファイル差し替え
- [ ] **第28章 演習問題** (`28/28_Exercises.ipynb`)

### 第29章 アプリ間通信 (`29/`)
- [ ] **29.1 終了コード (`osbook_day29a`)** (`29/29.1_App_Exit_Code_Display.ipynb`)
  - アプリの終了ステータス（`exit(status)`）をターミナルで受け取り表示する仕組み
- [ ] **29.2 パイプ (`osbook_day29b`)** (`29/29.2_Concept_of_Unix_Pipes.ipynb`)
  - プロセス間通信（IPC）としてのパイプ（`|`）の動作モデルとデータストリーム接続
- [ ] **29.3 コマンドラインの解析とタスクの起動** (`29/29.3_Parsing_Pipe_Syntax_and_Subtask_Spawning.ipynb`)
  - `cmd1 | cmd2` の分割と、左右のコマンドを並行タスクとして同時起動する仕組み
- [ ] **29.4 パイプ処理の本体 PipeDescriptor** (`29/29.4_Implementing_PipeDescriptor.ipynb`)
  - メッセージキューを介してタスク間でバイト列を送受信する `PipeDescriptor`（`Read` / `Write` / `FinishWrite`）
- [ ] **29.5 ターミナルの起動と終了** (`29/29.5_Sub_Terminal_Lifecycle_for_Pipes.ipynb`)
  - 非表示サブターミナルの生成と、パイプライン完了までの同期制御
- [ ] **29.6 タスクの終了** (`29/29.6_Task_Termination_and_Wait_Synchronization.ipynb`)
  - タスクの完全消滅処理と親タスクの待機・起床通知（`finish_waiter_`）
- [ ] **29.7 sortコマンド (`osbook_day29c`)** (`29/29.7_Implementing_sort_Filter_App.ipynb`)
  - 標準入力を読み込みソートして標準出力へ流すフィルタアプリ `sort` と `ls | sort` の連携検証
- [ ] **29.8 ターミナルのバグ修正 (`osbook_day29d`)** (`29/29.8_Fixing_Redraw_after_Pipe_Completion.ipynb`)
  - パイプ実行後のウィンドウ再描画漏れバグの解析と修正
- [ ] **29.9 共有メモリ** (`29/29.9_Shared_Memory_IPC_Design.ipynb`)
  - 同一物理ページを複数タスクの仮想アドレスにマッピングする共有メモリ（Shared Memory）の設計と原理
- [ ] **第29章 演習問題** (`29/29_Exercises.ipynb`)

### 第30章 おまけアプリ (`30/`)
- [ ] **30.1 アプリにパスを通す (`osbook_day30a`)** (`30/30.1_Application_Search_Path_Apps_Dir.ipynb`)
  - ルートディレクトリだけでなく `/apps` ディレクトリからも実行ファイルを自動検索する仕組み
- [ ] **30.2 moreコマンド (`osbook_day30b`)** (`30/30.2_Implementing_more_Pager_App.ipynb`)
  - パイプ入力を受け取り、1画面ごとにキー入力を待つページャ `more` の実装
- [ ] **30.3 catを入力に対応させる (`osbook_day30c`)** (`30/30.3_Updating_cat_for_Standard_Input.ipynb`)
  - 引数省略時に標準入力から読み込む `cat` のPOSIX風改良
- [ ] **30.4 閉じるボタン (`osbook_day30d`)** (`30/30.4_Clickable_Window_Close_Button.ipynb`)
  - ウィンドウ右上の `×` ボタンのクリック領域判定とアプリ終了イベント通知
- [ ] **30.5 テキストビューア (`osbook_day30e`)** (`30/30.5_Building_GUI_Text_Viewer_tview.ipynb`)
  - `mmap` と UTF-8 デコード、矢印キーによるページスクロールを備えた GUI テキストビューア `tview`
- [ ] **30.6 画像ビューア (`osbook_day30f`)** (`30/30.6_Building_JPEG_PNG_Image_Viewer_gview.ipynb`)
  - `stb_image` を用いた JPEG/PNG デコードと GUI 画像ビューア `gview` の実装
- [ ] **第30章 演習問題** (`30/30_Exercises.ipynb`)

### 第31章・付録 これからの道 & 付録 A〜F (`31/`)
- [ ] **31.1 これからの道（OS自作の発展テーマと次なる挑戦）** (`31/31.1_Future_Directions_in_OS_Development.ipynb`)
  - マルチコア（SMP / AP起動）、ネットワークスタック（NICドライバ / TCP/IP）、永続ストレージ（NVMe / AHCI）、セキュリティへの展望
- [ ] **付録A 開発環境のインストール（QEMU / OVMF / Docker / Ubuntu 24.04対策）** (`31/A_Development_Environment_Setup.ipynb`)
- [ ] **付録B MikanOSの入手とGitバージョン間差分解析手法** (`31/B_Getting_MikanOS_and_Git_Diff_Analysis.ipynb`)
- [ ] **付録C EDK IIのファイル構成と設定ファイル詳解** (`31/C_EDK2_Configuration_Files_Reference.ipynb`)
- [ ] **付録D OS実装で活用するC++テンプレート・イディオム詳解** (`31/D_Cpp_Templates_for_OS_Kernel.ipynb`)
- [ ] **付録E iPXEによる実機ネットワークブート実践** (`31/E_Network_Booting_with_iPXE.ipynb`)
- [ ] **付録F ASCIIコード表と文字エンコーディング早見表** (`31/F_ASCII_Code_Table_and_Encoding_Reference.ipynb`)
