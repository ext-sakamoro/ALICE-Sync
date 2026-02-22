# Changelog

All notable changes to ALICE-Sync will be documented in this file.

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
- "Demon Mode" batch optimizations: sort by entity_id, coalesce same-entity updates
- Auto-SIMD detection for contiguous 8-slot ranges
- Comprehensive module documentation in `lib.rs`
- `CHANGELOG.md` and `CONTRIBUTING.md`
- 78 tests across core + bridge modules

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
