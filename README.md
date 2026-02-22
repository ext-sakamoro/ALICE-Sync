# ALICE-Sync

**P2P Synchronization via Event Diffing**

> "Don't send data. Send the delta."

## What is ALICE-Sync?

ALICE-Sync is a high-performance P2P state synchronization engine for games and distributed simulations. Instead of sending entire world states, it synchronizes only the **events** (spawn, move, despawn) between nodes.

### Key Benefits

- **Minimal Bandwidth**: Send 18-byte events instead of megabytes of world data
- **Bit-Exact Determinism**: Fixed-point arithmetic ensures identical results across all platforms
- **O(1) Sync Verification**: Instant hash comparison even with 10,000+ entities
- **Zero Allocation**: Copy-based entities with arena storage

## Installation

```toml
[dependencies]
alice-sync = "0.6"
```

## Quick Start

```rust
use alice_sync::{Node, NodeId, Event, EventKind};

fn main() {
    // Create two nodes with the same seed
    let mut node_a = Node::new(NodeId(1));
    let mut node_b = Node::new(NodeId(2));

    // Node A: Spawn an entity
    let spawn = Event::new(EventKind::Spawn {
        entity: 1,
        kind: 0,
        pos: [0, 0, 0],  // i16 fixed-point coordinates
    });
    node_a.apply_event(&spawn).unwrap();

    // Node A: Move the entity
    let motion = Event::new(EventKind::Motion {
        entity: 1,
        delta: [100, 0, 0],  // Move right
    });
    node_a.apply_event(&motion).unwrap();

    // Sync: Send events to Node B (not world data!)
    node_b.apply_event(&spawn).unwrap();
    node_b.apply_event(&motion).unwrap();

    // Verify: Both nodes have identical world state
    assert_eq!(node_a.world_hash(), node_b.world_hash());
    println!("Sync verified! Hash: {:016x}", node_a.world_hash().0);
}
```

## Batch Processing (High-Performance)

For high-throughput scenarios, use the optimized batch API:

```rust
use alice_sync::{WorldSoA, Event, EventKind, MotionData};

let mut world = WorldSoA::new(42);

// Spawn entities
for i in 0..1000u32 {
    world.apply(&Event::new(EventKind::Spawn {
        entity: i,
        kind: 0,
        pos: [0, 0, 0],
    })).unwrap();
}

// Batch motion updates (optimized: sort + coalesce + SIMD)
let motions: Vec<MotionData> = (0..1000)
    .map(|i| MotionData {
        entity: i as u32,
        delta_x: 1,
        delta_y: 0,
        delta_z: 0,
    })
    .collect();

world.apply_motions_sorted(motions);  // 2-5x faster than naive apply
```

## Event Types

| Event | Description | Size (bitcode) |
|-------|-------------|----------------|
| Spawn | Create entity at position | ~20 bytes |
| Motion | Move entity by delta | ~18 bytes |
| Despawn | Remove entity | ~10 bytes |
| Property | Set entity property | ~18 bytes |
| Tick | Advance frame | ~14 bytes |

## Benchmark Results

All benchmarks measured on Apple M1, `cargo bench --release`:

```
World Hash (10000 entities):
  Incremental O(1):    543 ps   ← Constant time!
  Full Recalc O(N):    46.5 µs  ← Traditional approach
  Speedup: ~85,000x

Entity Lookup:
  Vec[id] indexing:    1.15 ns  ← O(1) regardless of count

Batch Processing (1000 motions, 100 entities):
  Sorted + Coalesce:   17.3 µs  ← Demon Mode
  No optimization:     22.3 µs  ← Baseline
  Speedup: 1.3x

Coalesce Effect (same entity x1000):
  With coalesce:       6.5 µs   ← Merges to 1 update
  Without coalesce:    35 µs    ← 1000 separate updates
  Speedup: 5.5x

Serialization:
  bincode:             46 bytes
  bitcode:             18 bytes (61% smaller)

Single Event Apply:
  Motion event:        30 ns
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Node A                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ World Engine │←─│ Event Stream │←─│   Protocol   │←─ Net │
│  │ (Fixed-Point)│  │  (History)   │  │  (bitcode)   │       │
│  │  O(1) Hash   │  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
         ↑                   ↑                   ↑
         │                   │                   │
    Same hash           Same events         Same protocol
    (XOR rolling)       (i16 quantized)     (bitcode)
         │                   │                   │
         ↓                   ↓                   ↓
┌─────────────────────────────────────────────────────────────┐
│                         Node B                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ World Engine │←─│ Event Stream │←─│   Protocol   │←─ Net │
│  │ (Fixed-Point)│  │  (History)   │  │  (bitcode)   │       │
│  │  O(1) Hash   │  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Optimization History

### v0.6 "Demon Mode" - Sort-Based Batching

- Sort events by entity ID for cache-friendly access
- Coalesce same-entity updates (5.5x faster for bursts)
- Re-sort by slot for true memory locality
- Auto-detect contiguous ranges for SIMD

### v0.5 "Zen Mode" - Complete SoA

- Entity data decomposed into component arrays
- Vertical SIMD: process 8 entities per instruction
- 12 bytes/entity for motion vs 84 bytes (7x reduction)

### v0.4 "Ultra Mode" - Branchless & Unsafe

- Raw integer mixing (no Hasher trait overhead)
- Branchless property hash with arithmetic masking
- SoA event storage for cache efficiency

### v0.3 "God Mode" - O(1) Everything

- XOR rolling hash (Zobrist-style)
- Vec direct indexing (no HashMap)
- Entity as Copy (no heap allocation)
- SIMD Vec3 operations

### v0.2 - Determinism

- Fixed-point arithmetic (no f32 drift)
- Arena allocation
- bitcode serialization

## Key Techniques

### O(1) Incremental Hashing

```rust
// Each event updates hash in O(1) using XOR
pub fn apply_motion(&mut self, entity_id: u32, delta: [i16; 3]) {
    let old_hash = self.compute_entity_hash(entity_id);
    self.current_hash ^= old_hash;  // Remove old state

    self.update_position(entity_id, delta);

    let new_hash = self.compute_entity_hash(entity_id);
    self.current_hash ^= new_hash;  // Add new state
}
```

### Fixed-Point Determinism

```rust
// No floating-point = no platform-dependent rounding
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Fixed(pub i32);  // Q16.16 format

// Guaranteed identical results on x86, ARM, WASM
let a = Fixed::from_f32(1.5);
let b = Fixed::from_f32(0.7);
let c = a + b;  // Always exactly the same bits
```

## Input Synchronization (v0.6)

Deterministic input sync for multiplayer games — **~20 bytes/player/frame**.

### Sync Modes

| Mode | Best For | Latency Model |
|------|----------|---------------|
| **Lockstep** | RTS, turn-based, < 4 players | Wait for all inputs |
| **Rollback** | Fighting, FPS, action games | Predict + confirm + rollback |

### Usage

```rust
use alice_sync::{InputFrame, LockstepSession, RollbackSession};

// --- Lockstep (RTS / turn-based) ---
let mut session = LockstepSession::new(2);  // 2 players

let input = InputFrame::new(0, 0)  // frame 0, player 0
    .with_movement(100, 0, 0)
    .with_actions(0x01);

session.add_local_input(input);
session.add_remote_input(remote_input);

if session.ready_to_advance() {
    let inputs = session.advance().unwrap();
    // Apply all inputs deterministically
}

// --- Rollback (fighting / FPS) ---
let mut rollback = RollbackSession::new(2, 0, 8);  // 2 players, local=0, max_rollback=8

let predicted_inputs = rollback.add_local_input(local_input);
// Apply predicted_inputs immediately (no waiting!)

match rollback.add_remote_input(confirmed_input) {
    RollbackAction::None => {},           // Prediction was correct
    RollbackAction::Rollback { to_frame } => {
        // Restore snapshot, re-simulate from to_frame
    },
    RollbackAction::Desync { frame } => {
        // Unrecoverable — reconnect
    },
}
```

### Bandwidth Comparison

| Approach | Data/Frame | 60 FPS, 4 Players |
|----------|-----------|-------------------|
| State sync (100 bodies) | ~16 KB | 3.84 MB/s |
| **Input sync (ALICE)** | ~80 B | **19.2 KB/s** |
| Savings | | **99.5%** |

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `async` | No | Tokio async runtime integration |
| `simd` | No | SIMD acceleration hints |
| `python` | No | Python bindings (PyO3 + NumPy zero-copy) |
| `physics` | No | ALICE-Physics bridge (InputFrame ↔ FrameInput, PhysicsRollbackSession) |
| `telemetry` | No | Sync telemetry recording via ALICE-DB (RTT, rollback, desync metrics) |
| `cache` | No | Markov oracle entity prefetching via ALICE-Cache |
| `auth` | No | Ed25519 ZKP peer authentication via ALICE-Auth |
| `codec` | No | Wavelet + rANS event stream compression via ALICE-Codec |
| `analytics` | No | DDSketch/HLL/CMS probabilistic telemetry via ALICE-Analytics |
| `cloud` | No | Cloud-side multi-device spatial synchronization hub |
| `all-bridges` | No | Enable all integration bridges (`physics` + `telemetry` + `cache` + `auth` + `codec` + `analytics`) |

## Python Bindings (PyO3 + NumPy Zero-Copy)

```toml
[features]
python = ["pyo3", "numpy"]
```

### Optimization Layers

| Layer | Technique | Effect |
|-------|-----------|--------|
| L1 | GIL Release (`py.allow_threads`) | Parallel batch processing |
| L2 | Zero-Copy NumPy (`into_pyarray`) | No memcpy for positions |
| L3 | Batch API (SoA world operations) | FFI amortization |
| L4 | Rust backend (8-wide SIMD, Demon Mode) | Hardware-speed apply |

### Python API

```python
import alice_sync

# --- WorldSoA (high-performance SoA world) ---
world = alice_sync.WorldSoA(seed=42)
world.spawn(entity_id=0, kind=0, x=0, y=0, z=0)
world.spawn(entity_id=1, kind=0, x=100, y=0, z=0)

# Batch motions (GIL released, Demon Mode SIMD)
motions = [(0, 10, 0, 0), (1, -5, 3, 0)]  # (entity_id, dx, dy, dz)
world.apply_motions(motions)

positions = world.positions()  # NumPy (N, 3) int32 — zero-copy
print(f"Hash: {world.hash():016x}")

# --- Input sync ---
frame = alice_sync.InputFrame(frame=0, player_id=0, move_x=100, move_y=0, move_z=0)
data = frame.to_bytes()   # bitcode serialized (~18 bytes)
decoded = alice_sync.InputFrame.from_bytes(data)

# Lockstep session
session = alice_sync.LockstepSession(player_count=2)
session.add_local_input(frame)

# Rollback session
rollback = alice_sync.RollbackSession(player_count=2, local_player=0, max_rollback=8)
predicted = rollback.add_local_input(frame)
action = rollback.add_remote_input(remote_frame)  # "none" / "rollback:N" / "desync:N"
```

## Game Engine Pipeline (ALICE-Physics Bridge)

Enable with `--features physics` to bridge ALICE-Sync input synchronization with [ALICE-Physics](../ALICE-Physics) deterministic simulation.

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["physics"] }
```

### Architecture

```
Network ──► InputFrame (i16, 24B) ──► FrameInput (Fix128) ──► PhysicsWorld
                 ALICE-Sync                ALICE-Physics

PhysicsWorld ──► SimulationChecksum ──► WorldHash ──► Desync Verification
                   ALICE-Physics           ALICE-Sync
```

### PhysicsRollbackSession

Combined rollback input sync + deterministic physics in one struct:

```rust
use alice_sync::physics_bridge::PhysicsRollbackSession;
use alice_sync::InputFrame;
use alice_physics::{NetcodeConfig, RigidBody, Vec3Fix, Fix128};

let mut session = PhysicsRollbackSession::new(2, 0, 8, NetcodeConfig::default());

// Add player bodies to physics world
let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
let idx = session.sim.add_body(body);
session.sim.assign_player_body(0, idx);

// Game loop: advance with local input
let input = InputFrame::new(1, 0).with_movement(1, 0, 0);
let checksum = session.advance_frame(input);

// When remote input arrives:
match session.add_remote_input(remote_input) {
    RollbackAction::None => {},
    RollbackAction::Rollback { to_frame } => {
        session.handle_rollback(to_frame);  // auto re-simulate
    },
    RollbackAction::Desync { .. } => { /* reconnect */ },
}
```

### Bridge Functions

| Function | Description |
|----------|-------------|
| `sync_input_to_physics()` | InputFrame (i16 Q8.8) → FrameInput (Fix128) |
| `physics_input_to_sync()` | FrameInput (Fix128) → InputFrame (i16, truncate) |
| `sync_inputs_to_physics()` | Batch conversion |
| `physics_checksum_to_world_hash()` | SimulationChecksum(u64) → WorldHash(u64) |
| `world_hash_to_physics_checksum()` | WorldHash(u64) → SimulationChecksum(u64) |

## Sync Telemetry (ALICE-DB Integration)

Record network synchronization metrics as time-series data in [ALICE-DB](../ALICE-DB). Enable with `--features telemetry`.

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["telemetry"] }
```

### Channels

| Channel | Metric | Range |
|---------|--------|-------|
| 0 | Rollback count | frames rolled back per event |
| 1 | Desync severity | 0.0 = none, 1.0 = fatal |
| 2 | Prediction accuracy | 0.0..1.0 |
| 3 | RTT (ms) | round-trip time |
| 4 | Input delay | delay in frames |

### Usage

```rust
use alice_sync::telemetry::SyncTelemetry;

let telemetry = SyncTelemetry::new("./telemetry_data")?;

// Record during gameplay
telemetry.record_rtt(frame, 15.5)?;
telemetry.record_prediction_accuracy(frame, 0.95)?;
telemetry.record_rollback(frame, 3)?;

telemetry.flush()?;

// Query for analysis
let rtt_data = telemetry.scan_rtt(0, 3600)?;
let avg_rtt = telemetry.average_rtt(0, 3600)?;
let max_rollback = telemetry.max_rollback(0, 3600)?;

telemetry.close()?;
```

ALICE-DB's model-based compression fits telemetry naturally:
- Stable RTT → constant model (1 coefficient)
- Gradually improving prediction → linear model
- Periodic jitter → Fourier model

## Cross-Crate Bridges

### Cache Bridge (feature: `cache`)

CRDT-based distributed cache invalidation with [ALICE-Cache](../ALICE-Cache). When sync events modify shared state, the cache bridge propagates invalidation messages to ensure distributed cache consistency across nodes.

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["cache"] }
```

### Codec Bridge (feature: `codec`)

Wavelet + rANS compression for event stream batches via [ALICE-Codec](../ALICE-Codec). Reduces event batch wire size by applying 1D wavelet transform and entropy coding to serialized event data.

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["codec"] }
```

```rust
use alice_sync::codec_bridge::{compress_event_batch, decompress_event_batch, estimate_ratio};

// Compress serialized event batch
let compressed = compress_event_batch(&event_bytes, quality)?;

// Estimate compression ratio
let ratio = estimate_ratio(&event_bytes);

// Decompress
let original = decompress_event_batch(&compressed)?;
```

### Analytics Bridge (feature: `analytics`)

Real-time sync telemetry via [ALICE-Analytics](../ALICE-Analytics). Feeds event throughput, round-trip latency, unique peer count, and hash divergences into probabilistic sketches (DDSketch, HyperLogLog, Count-Min Sketch).

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["analytics"] }
```

```rust
use alice_sync::analytics_bridge::SyncTelemetry;

let mut tel = SyncTelemetry::new();
tel.record_throughput(1500.0);
tel.record_latency(42.0);
tel.record_peer(b"peer-001");
println!("p99 latency: {:.1}us", tel.latency_p99());
println!("unique peers: {:.0}", tel.unique_peers());
```

### Incoming Bridge: ALICE-Streaming-Protocol

[ALICE-Streaming-Protocol](../ALICE-Streaming-Protocol) connects to ALICE-Sync via its `sync_bridge` module (feature `sync`), enabling synchronized media stream state across P2P nodes.

### Build Profile Changes

- `[profile.release]`: Added `strip = true` for smaller release binaries
- `[profile.bench]`: Standardized bench profile added

## Use Cases

- **Multiplayer Games**: RTS, fighting games, physics puzzles
- **Game Engine Pipeline**: ALICE-Physics + ALICE-Sync for bit-exact rollback netcode
- **Distributed Simulation**: Weather, traffic, physics
- **Collaborative Editing**: Real-time document/3D model sync
- **Rollback Netcode**: GGPO-style predict/confirm/rollback
- **Lockstep Netcode**: Deterministic wait-for-all synchronization

## Protocol

```
Handshake:
  A → B: Hello { node_id, seq, world_hash }
  B → A: Hello { node_id, seq, world_hash }

Sync:
  A → B: Events { [event1, event2, ...] }  // bitcode encoded
  B → A: Ack { seq }

Consistency Check:
  A → B: HashCheck { seq, world_hash }
  B → A: HashCheck { seq, world_hash }  // O(1) comparison
```

## Test Suite

78 tests across core modules and bridge integrations:

| Scope | Command | Tests |
|-------|---------|-------|
| Core | `cargo test --features std` | 52 |
| + Physics | `cargo test --features std,physics` | 58 |
| + Telemetry | `cargo test --features std,telemetry` | 55 |
| All bridges | `cargo test --features all-bridges,cloud` | 77 |

```bash
cargo test --features std                  # Core tests (52)
cargo test --features all-bridges,cloud    # All tests (77)
cargo bench                                # Benchmarks
```

### Codec Compression Note

The `codec` bridge uses a 1028-byte header (4B length + 256×4B histogram).
Event batches smaller than ~2 KB will not benefit from wavelet compression —
the header overhead exceeds savings. For small batches, bitcode serialization
alone (18 bytes/event) is already optimal.

## License

AGPL-3.0

Commercial licensing available for enterprise use.

## Author

Moroya Sakamoto

---

*"The best sync is one where data never travels."*
