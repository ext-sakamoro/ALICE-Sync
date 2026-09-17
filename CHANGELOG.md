# Changelog

All notable changes to ALICE-Sync will be documented in this file.

## [Unreleased]

## [0.6.1] - 2026-09-17

### Fixed
- `codec_bridge`: `decompress_event_batch` が量子化 step を 1 に固定していたため step > 1 の batch が step 分の 1 に縮んで復号されていた → step を header (4 B、histogram の前) に載せて復号側で使う (header 1028 → 1032 B、0.6.0 の `data` とは非互換) / `to_symbols` の `|q| > 127` error を捨てて symbol が silent wrap していた → step を `⌈max|c| / 127⌉` まで広げる (8-bit 全域 data で ≤ 4)、`Result` は `expect` で fail fast
- `codec_bridge` fuzz 初回 (2 件): 破損 header の step (u32 最大) で alice-codec `dequantize` が乗算 overflow → 復号側 step を `MAX_STEP = 2¹⁶` に clamp / header 内 length と `original_len` の不一致で 4 GiB 割当 → 不一致は復号せず返す
- `alice-codec` 要求を 0.1.3 に (rANS の偏った histogram round-trip / 量子化 symbol wrap の修正版)、`codec` feature の oracle の `#[ignore]` を解除

### Added
- CI を canonical 構成に (ALICE-SDF 同等): `ci.yml` = fmt / clippy `-D warnings` 3 variant / test 3 OS (default + 全 native feature) / analytic oracle (async + physics + codec) / msrv `rust-version = 1.88` 実 compile / doc `-D warnings` / feature-powerset (cargo-hack depth 2、python 除外)、`security-audit.yml` (audit / deny / unused-deps / stub-guard / coverage / semver-checks)、`fuzz.yml` + `fuzz/` 3 target (packet decode + 再組立 / codec_bridge / CRDT merge の可換・冪等)、`deny.toml` (AGPL bridge 例外) `ci-unified.yml` (workflow_dispatch 専用、stub 生成) は削除、未使用の `crossbeam-channel` dep を撤去 (cargo-machete)、`bytes` 1.12.1 (RUSTSEC-2026-0007)
- `tests/analytic_oracle.rs` — 閉形式 oracle 8 本 + ignore 1 (CLAUDE.md § 解析解突合テスト規律、2026-09-17): Q16.16 の dyadic 有理数 exact / 飽和 / i16 Q8.8 往復、SIMD add・sub が scalar と bit 一致 + batch、RFC 6298 の SRTT / RTTVAR / RTO を整数演算で逐語再現 + 定常 / clamp、header 往復 + 断片数 ⌈len/(MTU−4)⌉ + 逆順再組立、sequence / 重複、CRDT 代数 (可換 / 冪等 / LWW)、FNV-1a test vector + 最初の divergence frame、`physics` feature: i16 ↔ Fix128 の Q8.8 契約 (両方向・飽和) `codec` feature の oracle は alice-codec 0.1.3 publish 待ちで `#[ignore]` CI に async + physics の oracle step
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
