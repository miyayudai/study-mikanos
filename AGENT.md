# 🤖 MikanOS & OS Architecture Study Guidelines (`study-mikanos`)

本ドキュメントは、名著**『ゼロからのOS自作入門』（内田公太 著、マイナビ出版 2021年、通称「みかん本」）**で構築するオリジナル64ビットOS **「MikanOS」** の理論解説、ハードウェア仕様（x86-64 / UEFI / PCI / xHCI / APIC / ページング）の数理・ビット構造導出、C++17/20 コア実装、QEMU/ホストシミュレーションによる画面・挙動再現、および全章末の演習問題を扱う学習リポジトリ（`study-mikanos`）における**AIエージェント特化実行規範**です。

本リポジトリでのすべての作業は、ルートの [**Universal Guidelines (`../agent-docs/AGENT.md`)**](../agent-docs/AGENT.md) および [**Study Repository Guidelines (`../agent-docs/studies/AGENT.md`)**](../agent-docs/studies/AGENT.md) を継承した上で、本書の規定に厳格に従ってください。

---

## 1. 学習者のための完全自己完結化方針（最大原則）

> [!IMPORTANT]
> **「書籍や外部仕様書（Intel SDM / UEFI Spec / xHCI Spec）を開かなくても、ノートブックを読むだけで OS の動作原理・レジスタ/メモリ構造・C++実装・落とし穴のすべてが深く理解できる」** 圧倒的な濃度の解説 Notebook を作成することを絶対要件とします。

1. **完全自己完結（Self-contained）の徹底**:
   - 各節の導入では、**「なぜこの仕組みが OS に必要なのか（動機）」「前節（前タグ `osbook_dayXX`）までの実装が抱えていた課題・限界」** を必ず明記してください。
   - 公式リポジトリ（`uchan-nos/mikanos`）の対応タグ（`osbook_day02a` 〜 `osbook_day30f`）のコード構造・差分を正確に反映し、なぜその C++ / アセンブリ記述になるのかを1行単位で丁寧に解説してください。
2. **英語技術用語・ハードウェア略語の併記**:
   - 専門用語や略語は初出時および重要箇所で英語正式名称を括弧書きで必ず併記してください。
   - 例: UEFI (*Unified Extensible Firmware Interface*), GOP (*Graphics Output Protocol*), ELF (*Executable and Linkable Format*), GDT (*Global Descriptor Table*), IDT (*Interrupt Descriptor Table*), MSI (*Message Signaled Interrupts*), xHCI (*eXtensible Host Controller Interface*), PML4 (*Page Map Level 4*), TSS (*Task State Segment*), IST (*Interrupt Stack Table*), CoW (*Copy-on-Write*) 等。
3. **小節（Subsection）およびコラムの完全網羅**:
   - 対象節に含まれるすべてのトピックおよび付随コラム（Column）を省略せず、Jupyter Notebook 内の見出し（Markdownセル `###`）として独立して解説してください。

---

## 2. ハードウェア構造・メモリレイアウト・数理の記述基準

1. **ビットフィールド・アドレス計算の完全導出**:
   - x86-64 の 4階層ページング（48ビット正規アドレス：`9 + 9 + 9 + 9 + 12` ビット分割）、GDT/IDT セグメント記述子・割り込み記述子のビット配置、PCI `CONFIG_ADDRESS`（バス番号・デバイス番号・ファンクション番号・レジスタオフセット）のビットシフト演算、FAT16/32 のクラスタ番号→セクタ番号→バイトオフセット変換式などを、LaTeX 数式とビット図でステップ・バイ・ステップで導出してください。
2. **メモリ構造・制御フローの視覚化（必須DoD）**:
   - スタックフレーム、コンテキストスイッチ時のレジスタ退避（`TaskContext`）、リングバッファ（FIFO）、レイヤ重ね合わせ（`LayerManager`）、ページテーブル階層などを **Mermaid ダイアグラム**、**ASCII アート**、および **matplotlib による高解像度図版（`{ch}/result/fig{ch}_{sec}_{desc}.png`）** として必ず視覚化・保存してください。

---

## 3. 対話型 C++20 コンパイル実行 & Python フレームバッファ/OSシミュレーション

Google Colab や軽量ローカル環境でも UEFI 実機再起動なしで即座に対話的学習・実験ができるよう、各節の Notebook では以下の **ハイブリッド検証方式** を標準とします：

1. **C++ コアロジックの対話型コンパイル & 実行**:
   - MikanOS のカーネル実装（`PixelWriter`, `Console`, `ArrayQueue`, `BitmapMemoryManager`, `Layer`, `TimerManager`, `Task`, `FAT` 等）をホスト側 C++20（`g++ -std=c++20 -O2 -Wall -Wextra`）でもコンパイル・単体検証可能な形で `src/chXX/` に切り出し、Notebook 上からコンパイル・実行して挙動とメモリ状態を実証します。
2. **Python (`common/os_sim.py`) によるフレームバッファ・メモリ・可視化プロット**:
   - GOP フレームバッファ描画（マウスカーソル、ウィンドウ、テキストボックス、ブロック崩し等）やメモリマップ・ページング・タスクスケジューリングの時系列挙動を Python / matplotlib で忠実に再現し、`{ch}/result/` に高解像度 PNG（DPI 300）として保存＆Notebook 内にインライン表示します。

---

## 4. 適用モードとタスク進行ルール

- **Git運用モード**: **モード A（自律高速モード Direct-to-Main with DoD）**
  - 作業ブランチや PR の作成は不要。最新の `main` 上で直接作業します。
  - **1回の実行ステップは原則「1つの節（Section）」または「その章の演習問題（Exercises）」**とします。「1章まるごと」の一括処理は内容の疎化を招くため厳禁です。
- **Active Lock と GitHub Issue 運用**:
  - 作業着手時は必ず [TASK.md](TASK.md) の対象タスクを `- [-]` に更新し、`Active Lock` を取得してください。
  - 1節（または1演習タスク）ごとに GitHub Issue（`[Study] Chapter X.Y: {タイトル}`）を起票し、DoD（全セル実行エラーゼロ、`result/` 図版生成、`pytest` 全件パス、`README.md` 進捗・Colabリンク同期）を満たしたら進捗コメントを追記して `main` へ直接 `git push` し、Issue をクローズして次の節へ自律進行してください。
