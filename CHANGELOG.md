# Changelog

All notable changes to ALICE-Sync will be documented in this file.

## [Unreleased]

### Added
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査する (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` と test) 静的は target/ 以外の設定・script・Rust を comment を除いて読み、`target-cpu=native`、素の literal でない target-cpu / target-feature / llvm-args の値、command や file から取った RUSTFLAGS、fast-math 系の flag、sse / sse2 を外す target-feature を検出する (意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く) 動的は新しい target dir で `cargo check -v` を走らせ、cargo が rustc に実際に渡した flag を読む 静的は entropy-guard job、動的は release-profile job (ubuntu / macOS、default / all-features / no-default) と preflight で実行
- `tests/float_environment.rs` (ALICE-DB と同一): round-to-nearest-even と subnormal が flush されないことを実行時に確かめる test test job (ubuntu / macOS / Windows) と release profile で `scripts/run_tests.sh` を通して実行 `alice-law-conformance.toml` の G14 を pass にした
- CI (`clippy.toml`、G10): `std::time::Instant::elapsed` / `std::time::SystemTime::elapsed` を disallowed-methods に追加 どちらも `::now()` と同じく wall clock を読むが、どちらの層も検査していなかった (`.elapsed(` の字面検査は `elapsed` という名の値全般を拾い過ぎるため、型解決する clippy 側だけに持たせた) `duration_since` はすでに保持した値どうしの引き算で clock を読まないため対象外 非 test code の `.elapsed()` 呼び出しは 0 件で、挙動の変更なし canary: clippy.toml にこの 2 entry があることを確認する test
- `tests/float_edge_golden.rs` と release profile の CI (`release-profile.yml`、ubuntu / macOS) を追加 float の境界 (`Fixed::from_f32` / `to_f32` / `Vec3Fixed`、`BandwidthStats` と `loss_rate` の比、`average_rtt`、`SpatialRegion`) の符号付きゼロ・1 step の前後・大きな逆符号値の打ち消し・0 除算の既定値を `to_bits` / raw 値で固定し、`scripts/run_tests.sh` (ALICE-DB と byte 一致、指定した test file が 1 本も走らなければ失敗) 経由で `analytic_oracle` と合わせて `cargo test --release` で走らせる preflight の full にも同じ step を追加 挙動の変更なし
- `README_JP.md` を追加 (README.md と同じ節構成の日本語版、コード例は英語版と同一) README.md の冒頭に日本語版への link を追加
- README の benchmark 節の測定環境の表記を機種名から `arm64` に変更 (数値は変更なし)
- `clippy.toml` に ALICE-Law の G1 の禁止一覧 (f32 / f64 の超越関数 24 個と `sin_cos`、`powi`) と G10 の `Instant::now` / `SystemTime::now` / `RandomState::new` を `disallowed-methods` として追加 transport 層の単調時計の読み取りは `src/clock.rs` の 1 箇所に集め、その 1 箇所だけに理由つきの許可を置く `scripts/entropy_guard.py` (CI の `entropy-guard` job) が src/ の乱数・時計の使用を読み、`clock.rs` の 1 件以外を拒否する 挙動の変更なし
- CI: libm を直接依存に足した時に G1 の禁止 (clippy.toml の `libm::` path と `scripts/libm_ban_check.py`) が無ければ失敗する検査 `scripts/libm_dependency_guard.py` (試験 `scripts/test_libm_dependency_guard.py`) を entropy-guard job と preflight に追加 (ALICE-DetMath と byte 一致)

### Changed
- CI: 公開 API の互換性の検査を `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB / ALICE-Analytics / ALICE-Edge / ALICE-Zip と同一) に置き換えた crates.io の最新の alice-sync を baseline にし、既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせ、互換でない変更を失敗させる 版が公開済みの 0.6.1 と同じ間は次の最小の版 0.6.2 として検査する 旧 job (security-audit.yml) は結果を問わない informational で、`rust-toolchain.toml` の 1.92.0 が cargo-semver-checks 0.50 の要求 (rustc 1.93 以上) に届かず毎回何も比較していなかった
- `rust-toolchain.toml` の pin を 1.92.0 から 1.98.1 に上げた (他の ALICE-* と同じ) `rust-version` (MSRV 1.88) は変えない
- CI: 公開する文面の検査の pin を ALICE-Law 5de698e (checker sha256 4b3d9542…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- **`alice-physics` の要求を `2.0` から `3.0` に上げた (2026-10-15)** 公開済の 3.0.1 に追従するため caret は major を跨がず、`^2.0` のままでは 3.x に到達しない `physics_bridge` の変更は無し (`new` + `with_*` / `Default` 構築のため) 実測: `cargo test --features physics` が 161 + 4 + doc 2 pass (lock の解決は 3.0.1)、`cargo check --all-features --all-targets` が 0 error `alice-physics` 3.0.1 は `alice-det-math` を `0.5` で引くので、この crate の det-math 世代は 0.3 から 0.5 に動く
- CI: 上限の無かった 15 job に `timeout-minutes` を付けた (止まった step は共有 runner を GitHub の上限の 6 時間まで占有しうる) 値は直近の成功した run 10 本の所要時間の最大の 2 倍・中央値の 4 倍の大きい方を 5 分単位に切り上げ、cargo を使う job は最低 15 分 (cache が外れた build のため)、他は最低 10 分 fuzz は 30 分で、`workflow_dispatch` の `duration_seconds` は 1 から 1200 の整数 (先頭の 0 は不可、0 は libFuzzer で無制限になる) でなければ理由を出して止める (それより長い run は build と合わせて上限を超え途中で切れる) 入力は script に埋め込まず `env` で渡す `scripts/workflow_timeouts.py` (試験 `scripts/test_workflow_timeouts.py`、ALICE-LOL と byte 一致) が全 job の上限を検査し、job を 1 つも読めなければ失敗する CI (actionlint job) と preflight で走る
- CI: どの workflow も `ci/**` の branch への push でも走るようにした (`push.branches` に `ci/**` を足した) main に入れる前の commit をその branch で検証するため これまでは main への push と pull request でしか走らず、branch の commit には CI の結果が無かった どの workflow の concurrency も、main では commit ごとに別の group にして run を止めない形にした (これまでは main の run も次の commit で止まり、止まった run は検証されないまま残っていた) main 以外では従来どおり古い run を止める path の条件がある workflow (fuzz / security-audit) は従来どおり対象の path が変わった時だけ走る
- **`alice-physics` の要求を `1.4` から `2.0` に上げた (2026-10-09)** 2.0.0 は `#[non_exhaustive]` の一括付与と enum の variant 追加を含む major bump で、caret は major を跨がないため `^1.4` のままでは公開済の 2.0.0 に永久に到達しない `physics_bridge` は 2026-09-30 に `FrameInput` / `NetcodeConfig` の struct literal を `new` + `with_*` / `Default` へ変えてあるので、呼び出し側の変更は無し 実測: `cargo check --features physics --all-targets` が 0 error (lock の解決は 2.0.0)、`scripts/preflight.sh` (full) が exit 0 `alice-physics` は `alice-det-math` を `^0.3` で引くので、この crate の det-math 世代は 0.2 から 0.3 に動く (0.4 に揃うには alice-physics の 0.4 版の publish が必要)
- **`physics_bridge` の `FrameInput` / `NetcodeConfig` 構築を struct literal から `new` + `with_*` / `Default` に変更 (2026-09-30)** alice-physics 側でこの 2 型に `#[non_exhaustive]` が付いても壊れない形にしておく 挙動は変更なし (`sync_input_to_physics` は同じ 4 field を同じ値で埋める、test 側の `player_count: 2` / `max_snapshots: 10` は既定値と同値なので `assert_eq!` で固定)
- **License: `AGPL-3.0-or-later` → `AGPL-3.0-or-later OR LicenseRef-Commercial` (dual-licensed、2026-09-27)** AGPL 側の条件は変更なし (既存 AGPL 利用者への影響ゼロ)、商用という選択肢が追加されただけ SPDX が AGPL 単独だと cargo-deny / FOSSA / SBOM に「商用オプションなし」と見えるため宣言を dual に 変更点: SPDX / `LICENSE` → `LICENSE-AGPL` rename / `LICENSE-COMMERCIAL.md` (商用トリガー 6 条件 = クローズド製品・商用 SaaS・エッジ・ファームウェア配布・plugin 再配布・プラットフォーム NDA・保証、社内利用は AGPL 側で無償と明記) / README の選択肢表 商用窓口は法人 `contact@extoria.co.jp`
- CI: `scripts/semver_gate.py` が cargo-semver-checks の出力を読む parser であることを `CARGO_OUTPUT_PARSER = True` で宣言し、`scripts/test_semver_gate.py` は module を `import semver_gate` で読むようにした (色の付いた出力を試験する検査器が parser とその試験を見つけられる形 実行時は従来どおり `CARGO_TERM_COLOR=never` と escape の除去) 共有 file の中身を揃える変更で、検査の挙動は変わらない
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) を追加し、植える呼び出しの表 `scripts/clippy_canary_calls.tsv` を `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成して CI が再生成した結果と byte で比べる (この repo は 58 件) `scripts/clippy_canary.py` / `scripts/test_clippy_canary.py` / `scripts/gen_clippy_canary_calls.py` / `scripts/test_gen_clippy_canary_calls.py` は ALICE-DB と byte 単位で同一 clippy job と preflight に 4 step を足した
- CI: `scripts/run_tests.sh` が名前ごとの最低の件数 (`float_environment=2`) を受け付けるようにし (ALICE-DB と同じ中身)、G14 の `tests/float_environment.rs` を名指す全 step を `float_environment=2` にした 2 つの試験の片方を `#[ignore]` にする・cfg で外す・名前を変えると、その step が失敗する `scripts/test_run_tests.py` も足し CI と preflight で走らせる
- CI: 公開する文面の検査の pin を ALICE-Law 779fa44 (checker sha256 e439fdd7…) に進めた 新しい版で HEAD の全 file に指摘 0 件

## [0.6.1] - 2026-09-17

### Fixed
- `codec_bridge`: `decompress_event_batch` が量子化 step を 1 に固定していたため step > 1 の batch が step 分の 1 に縮んで復号されていた → step を header (4 B、histogram の前) に載せて復号側で使う (header 1028 → 1032 B、0.6.0 の `data` とは非互換) / `to_symbols` の `|q| > 127` error を捨てて symbol が silent wrap していた → step を `⌈max|c| / 127⌉` まで広げる (8-bit 全域 data で ≤ 4)、`Result` は `expect` で fail fast
- `codec_bridge` fuzz 初回 (2 件): 破損 header の step (u32 最大) で alice-codec `dequantize` が乗算 overflow → 復号側 step を `MAX_STEP = 2¹⁶` に clamp / header 内 length と `original_len` の不一致で 4 GiB 割当 → 不一致は復号せず返す
- `alice-codec` 要求を 0.1.3 に (rANS の偏った histogram round-trip / 量子化 symbol wrap の修正版)、`codec` feature の oracle の `#[ignore]` を解除

### Added
- CI を canonical 構成に (ALICE-SDF 同等): `ci.yml` = fmt / clippy `-D warnings` 3 variant / test 3 OS (default + 全 native feature) / analytic oracle (async + physics + codec) / msrv `rust-version = 1.88` 実 compile / doc `-D warnings` / feature-powerset (cargo-hack depth 2、python 除外)、`security-audit.yml` (audit / deny / unused-deps / stub-guard / coverage / semver-checks)、`fuzz.yml` + `fuzz/` 3 target (packet decode + 再組立 / codec_bridge / CRDT merge の可換・冪等)、`deny.toml` (AGPL bridge 例外) `ci-unified.yml` (workflow_dispatch 専用、stub 生成) は削除、未使用の `crossbeam-channel` dep を撤去 (cargo-machete)、`bytes` 1.12.1 (RUSTSEC-2026-0007)
- `tests/analytic_oracle.rs` — 閉形式 oracle 8 本 + ignore 1 (2026-09-17): Q16.16 の dyadic 有理数 exact / 飽和 / i16 Q8.8 往復、SIMD add・sub が scalar と bit 一致 + batch、RFC 6298 の SRTT / RTTVAR / RTO を整数演算で逐語再現 + 定常 / clamp、header 往復 + 断片数 ⌈len/(MTU−4)⌉ + 逆順再組立、sequence / 重複、CRDT 代数 (可換 / 冪等 / LWW)、FNV-1a test vector + 最初の divergence frame、`physics` feature: i16 ↔ Fix128 の Q8.8 契約 (両方向・飽和) `codec` feature の oracle は alice-codec 0.1.3 publish 待ちで `#[ignore]` CI に async + physics の oracle step
- **i16 wire field (`InputFrame::movement` / `aim`、`EventKind::Motion` delta) の scale が 3 経路で食い違っていた** (oracle 先行 red): doc は Q8.8、`Fixed::from_i16` は `<< 6` (Q6.10、256 ↦ 0.25)、`physics_bridge` は整数 (`from_int`、256 ↦ 256.0) — 同じ入力が経路毎に 1024 倍違う変位になっていた → Q8.8 (256 ↦ 1.0) を唯一の法則に: `Fixed::from_i16` = `<< 8` / `to_i16` = `>> 8` + 飽和、`Vec3Simd::from_i16_array` は `Fixed::from_i16` 経由、`physics_bridge` は `Fix128::from_int(n).shr_bits(8)` と `⌊v·256⌋` 飽和 既存 unit test の `<< 6` / `.hi == n` pin を法則値に更新

### Changed
- README / lib.rs の全称 claim を実態に限定し、各 claim 行に `<!-- claim-test: fn -->` で検証 test を紐付け (strict-eval 検査 1、2026-09-17)
- bridge 系 sibling 6 crate (`alice-physics` 1.4 / `alice-db` 0.2.0-beta.2 / `alice-cache` 0.2 / `alice-auth` 0.5.1 (0.5.0 は release build 不能) / `alice-codec` 0.1 / `alice-analytics` 0.1) の依存を path から crates.io version に変更、CI の manifest-only stub (version 0.1.0 固定で `^1` / `^0.5` を満たせず 2026-09-14 から red) を撤去し `cargo check --lib --all-features` を追加 (bridge feature が公開版 API で compile することを CI が初めて確認)

### Fixed
- **FFI 71 関数の panic 隔離** (`src/ffi.rs`): 全 `extern "C"` の本体を `ffi_guard(sentinel, || ..)` で包み、panic は host を落とさず sentinel (null / 0 / −1 / NaN / ()) + `as_sync_last_error()` (新規、`as_sync_clear_last_error` / `as_sync_free_error_string` も) で通知 `[profile.release] panic = "abort"` を撤去 (abort では `catch_unwind` が機能しない、cdylib size が数 % 増) release profile で guard test 通過
- `analytics_bridge`: `alice_analytics::prelude` は存在しない module だった (bridge 追加以来 stub で一度も compile されていなかった) → `alice_analytics::sketch::{CountMinSketch1024x5, DDSketch256, HyperLogLog12}`
- `codec_bridge` test: 常に真の `u8 <= 255` assert (clippy `absurd_extreme_comparisons`) を長さ検証に置換、誤差上限 oracle `roundtrip_error_is_bounded` を追加 — 実測 MAE 24.6 / max 101 (0..63 鋸波) で定数予測より悪いため `#[ignore]` (bridge の量子化は要修正、未解決)

## [0.6.0] - 2026-02-23

### Added

- Input synchronization: `LockstepSession`, `RollbackSession`, `InputFrame`, `InputBuffer`
- Rollback netcode with prediction, confirmation, and automatic re-simulation
- `PhysicsRollbackSession` (feature `physics`): combined rollback + deterministic physics
- `codec_bridge` (feature `codec`): wavelet + rANS compression for event streams
- `analytics_bridge` (feature `analytics`): DDSketch/HLL/CMS probabilistic telemetry
- `auth_bridge` (feature `auth`): Ed25519 ZKP peer authentication via ALICE-Auth
- `cache_bridge` (feature `cache`): Markov oracle entity prefetching via ALICE-Cache
- `cloud_bridge` (feature `cloud`): star-topology multi-device spatial sync hub
- Python bindings: `WorldSoA`, `InputFrame`, `LockstepSession`, `RollbackSession`
- C-ABI FFI bindings: 71 `extern "C"` functions (feature `ffi`)
- Unity C# bindings: 71 `[DllImport]` wrappers + RAII handles (`bindings/unity/AliceSync.cs`)
- UE5 C++ bindings: 71 `extern "C"` declarations + 9 RAII handles (`bindings/ue5/AliceSync.h`)
- "Demon Mode" batch optimizations: sort by entity_id, coalesce same-entity updates
- Auto-SIMD detection for contiguous 8-slot ranges
- Comprehensive module documentation in `lib.rs`
- `CHANGELOG.md` and `CONTRIBUTING.md`
- 88 tests across core + bridge + FFI modules

### Changed

- Event batching re-sorts by slot index after entity-id sort for true memory locality
- `apply_motions_sorted` now 1.4x faster via prefetcher-friendly access patterns

### Fixed

- `codec_bridge` test used 256-byte input below 1028-byte header threshold
- Unused import warnings in `cloud_bridge` and unused `Result` warnings in `codec_bridge`
- Doc warnings from `dx:2`/`dz:2` parsed as intra-doc links

## [0.5.0] - 2026-02-20

### Added

- `WorldSoA`: complete Structure of Arrays world storage
- Vertical SIMD: process 8 entities' x/y/z coordinates in one instruction
- 8-wide parallel hash computation
- `Slot` abstraction for SoA indexing
- Python batch API: `apply_motions()`, `positions()` with GIL release

### Changed

- Motion reads 12 bytes/entity instead of 84 bytes (7x memory reduction)
- 100% cache efficiency: only touch data needed for each operation

## [0.4.0] - 2026-02-18

### Added

- SoA (Structure of Arrays) event storage: `MotionData`, `SpawnData`, `DespawnData`
- `SoAStats` for batch processing statistics
- Branchless property hash with arithmetic masking

### Changed

- Raw integer mixing replaces `Hasher` trait overhead
- Event enum tag eliminated in hot loops via SoA decomposition

## [0.3.0] - 2026-02-16

### Added

- `WorldHash`: XOR rolling hash (Zobrist-style) for O(1) incremental verification
- Vec direct indexing for O(1) entity lookup (no HashMap)
- `Entity` as `Copy` type (no heap allocation)
- SIMD `Vec3Simd` operations via `wide` crate
- `Vec3Fixed`: Q16.16 fixed-point 3D vector

### Changed

- World hash comparison reduced from O(N) full recalc to O(1) incremental

## [0.2.0] - 2026-02-14

### Added

- `Fixed`: Q16.16 fixed-point integer type for platform-independent determinism
- `Arena`: generational arena allocator with O(1) insert/remove/lookup
- bitcode serialization (61% smaller than bincode)
- `Protocol`: Handshake / Sync / Ack / HashCheck wire messages
- `EventStream`: ordered event history with sequence numbers

## [0.1.0] - 2026-02-12

### Added

- Core event system: `Event`, `EventKind`, `EventId`, `SeqNum`
- `Node`: P2P node with causal event ordering and world state
- `World`: AoS entity world with spawn/motion/despawn/property events
- `Entity`, `EntityProps`: fixed-size entity with 8 property slots
- bincode serialization
- 5 event types: Spawn, Motion, Despawn, Property, Tick
