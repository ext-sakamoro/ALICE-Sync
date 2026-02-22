# Contributing to ALICE-Sync

## Prerequisites

- Rust 1.70.0 or later
- `cargo fmt` and `cargo clippy` installed

## Development Workflow

```bash
# Build (default features)
cargo build

# Run core tests
cargo test --features std

# Run all tests (including bridges)
cargo test --features all-bridges,cloud

# Run with specific bridge
cargo test --features physics
cargo test --features codec
cargo test --features auth

# Lint
cargo clippy --features std -- -W clippy::all

# Format
cargo fmt

# Benchmarks
cargo bench

# Doc generation
cargo doc --features std --no-deps --open
```

## Code Style

- Run `cargo fmt` before committing
- All clippy warnings must be resolved (`-W clippy::all`)
- Maintain `no_std` compatibility for core modules (not gated behind `std`)

## Determinism

This engine guarantees bit-exact results across all platforms. When contributing:

- Never use `f32` or `f64` in synchronization paths — use `Fixed` (Q16.16)
- Use `Vec3Fixed` instead of floating-point vectors
- Ensure XOR hash updates are commutative and associative
- Event application order must produce identical world state on all nodes
- Use stable sort with explicit comparators in batch processing

## Testing

- Unit tests go in `#[cfg(test)] mod tests` within each source file
- Aim for at least one test per public function
- Bridge tests should verify roundtrip correctness
- Determinism tests should verify hash equality across independent nodes

## Architecture

```
Core Modules (always available):
  arena.rs         — Generational arena allocator
  event.rs         — Event types + SoA storage
  fixed_point.rs   — Q16.16 fixed-point + SIMD
  node.rs          — P2P node with causal ordering
  protocol.rs      — Wire protocol messages
  world.rs         — AoS entity world
  world_soa.rs     — SoA world (Demon Mode)
  input_sync.rs    — Lockstep + Rollback sessions

Bridge Modules (feature-gated):
  physics_bridge.rs    — ALICE-Physics integration
  telemetry.rs         — ALICE-DB time-series
  cache_bridge.rs      — ALICE-Cache prefetching
  auth_bridge.rs       — ALICE-Auth Ed25519 ZKP
  codec_bridge.rs      — ALICE-Codec wavelet compression
  analytics_bridge.rs  — ALICE-Analytics sketches
  cloud_bridge.rs      — Multi-device spatial sync
```

## License

By contributing, you agree that your contributions will be licensed under AGPL-3.0.
