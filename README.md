# ALICE-Sync

**General-Purpose P2P Synchronization Infrastructure**

> "Don't send data. Send the delta."

## What is ALICE-Sync?

ALICE-Sync is a high-performance P2P state synchronization engine. Instead of sending entire world states, it synchronizes only the **events** (spawn, move, despawn) between nodes. With the `async` feature, it becomes a complete, self-contained P2P infrastructure — transport, discovery, pub/sub, CRDTs, and session management included.

### Key Benefits

- **Minimal Bandwidth**: Send 18-byte events instead of megabytes of world data
- **Bit-Exact Determinism**: Fixed-point arithmetic ensures identical results across all platforms
- **O(1) Sync Verification**: Instant hash comparison even with 10,000+ entities
- **Zero Allocation**: Copy-based entities with arena storage
- **Self-Contained P2P**: UDP/TCP transport, mDNS discovery, topic pub/sub, CRDTs — no external infrastructure needed
- **Multi-Mode Sync**: Lockstep, Rollback, CRDT, Event Sourcing, Snapshot

## Installation

```toml
[dependencies]
alice-sync = "0.6"

# Full P2P infrastructure
alice-sync = { version = "0.6", features = ["async"] }
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

## Async P2P Infrastructure

Enable with `--features async` to get a complete, self-contained P2P stack.

### Modules

| Module | Description |
|--------|-------------|
| `reliability` | UDP reliability layer: ACK tracking, RTT estimation, retransmission, fragmentation/reassembly |
| `transport` | `SyncTransport` trait + UDP, TCP, in-process Channel implementations |
| `crdt` | Conflict-free replicated data types: `LwwRegister`, `GCounter`, `PnCounter`, `OrSet`, `LwwMap` |
| `channel` | Topic-based pub/sub with per-topic filtering |
| `discovery` | mDNS LAN auto-discovery + manual peer registration + GC |
| `session` | High-level session driver: builder pattern, tick loop, 5 sync modes |

### Sync Modes

| Mode | Best For | Characteristics |
|------|----------|-----------------|
| **Lockstep** | RTS, turn-based, < 4 players | Waits for all inputs before advancing |
| **Rollback** | Fighting games, FPS, action | Predicts + corrects on mismatch |
| **CRDT** | Collaboration, IoT, databases | Eventual consistency, no coordination |
| **EventSourcing** | Audit logs, financial, legal | Append-only event log + replay |
| **Snapshot** | Late-join, recovery | Full state transfer |

### Session Example

```rust
use alice_sync::{SessionBuilder, SyncMode, UdpTransport};

#[tokio::main]
async fn main() {
    let transport = UdpTransport::bind("0.0.0.0:9000").await.unwrap();

    let session = SessionBuilder::new()
        .mode(SyncMode::Crdt)
        .node_id(1)
        .tick_rate(60)
        .name("my-node")
        .build(transport);

    // Add peers manually or let mDNS discover them
    session.add_peer("192.168.1.100:9000".parse().unwrap()).await;

    // Subscribe to session events
    let mut events = session.subscribe_events();

    // Start the tick loop
    session.start().await;

    // Use CRDT for conflict-free state sync
    session.crdt_set("player/1/name".into(), b"Alice".to_vec()).await;

    // Use topic pub/sub for structured routing
    let mut rx = session.pubsub().subscribe("game/room42/state", addr).await;
    session.pubsub().publish("game/room42/state", payload, None).await;
}
```

### CRDT Example

```rust
use alice_sync::{LwwMap, OrSet, GCounter, CrdtMergeable};

// LWW-Map: distributed key-value store
let mut node_a = LwwMap::new(1); // replica ID 1
let mut node_b = LwwMap::new(2); // replica ID 2

node_a.insert("key".to_string(), "value_a".to_string());
node_b.insert("key".to_string(), "value_b".to_string());

// Merge — highest timestamp wins, no conflicts
node_a.merge(&node_b);

// OR-Set: add/remove with add-wins semantics
let mut set = OrSet::new();
set.add(1, "tag_a".to_string());
set.add(2, "tag_b".to_string());
assert!(set.contains(&"tag_a".to_string()));

// G-Counter: distributed monotonic counter
let mut counter = GCounter::new();
counter.increment(1); // replica 1 increments
counter.increment(2); // replica 2 increments
assert_eq!(counter.value(), 2);
```

### Transport Abstraction

```rust
use alice_sync::{SyncTransport, UdpTransport, TcpTransport, ChannelTransport};

// UDP: low-latency, with reliability layer (ACK, retransmit, fragmentation)
let udp = UdpTransport::bind("0.0.0.0:9000").await?;

// TCP: reliable ordered delivery, length-prefixed framing
let tcp = TcpTransport::bind("0.0.0.0:9001").await?;

// Channel: in-process, for testing
let (a, b) = ChannelTransport::pair();
```

### Topic Pub/Sub

```rust
use alice_sync::PubSub;

let pubsub = PubSub::new();

// Subscribe peers to topics
let mut rx = pubsub.subscribe("sensor/temp/room1", peer_addr).await;

// Publish with optional filtering
pubsub.publish("sensor/temp/room1", data, Some(origin)).await;

// Get all subscribers for transport-level routing
let peers = pubsub.subscribers("sensor/temp/room1").await;
```

### Peer Discovery

```rust
use alice_sync::Discovery;
use std::sync::Arc;

let discovery = Arc::new(Discovery::new(local_addr, "my-node"));

// Manual peer registration
discovery.register_peer(addr, "peer-1", metadata).await;

// Event callbacks
discovery.on_event(Arc::new(|event| {
    println!("Peer event: {:?}", event);
})).await;

// Start mDNS GC background task
discovery.start_mdns_gc();

// Get live peers
let peers = discovery.peer_addrs().await;
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
  Incremental O(1):    543 ps   <- Constant time!
  Full Recalc O(N):    46.5 us  <- Traditional approach
  Speedup: ~85,000x

Entity Lookup:
  Vec[id] indexing:    1.15 ns  <- O(1) regardless of count

Batch Processing (1000 motions, 100 entities):
  Sorted + Coalesce:   17.3 us  <- Demon Mode
  No optimization:     22.3 us  <- Baseline
  Speedup: 1.3x

Coalesce Effect (same entity x1000):
  With coalesce:       6.5 us   <- Merges to 1 update
  Without coalesce:    35 us    <- 1000 separate updates
  Speedup: 5.5x

Serialization:
  bincode:             46 bytes
  bitcode:             18 bytes (61% smaller)

Single Event Apply:
  Motion event:        30 ns
```

## Architecture

```
                         ALICE-Sync P2P Stack

  Application Layer
  +-------------------------------------------------------------+
  |  SessionBuilder -> SyncSession                               |
  |    - 5 sync modes (Lockstep/Rollback/CRDT/EventSrc/Snapshot)|
  |    - Tick loop, event dispatch, stats                        |
  +-------------------------------------------------------------+
                            |
  +----------------+  +----------+  +-------------+
  | Topic Pub/Sub  |  |  CRDTs   |  |  Discovery  |
  | (channel.rs)   |  | LWW-Map  |  | mDNS + GC   |
  | Filter, Route  |  | OR-Set   |  | Manual reg  |
  +----------------+  | G/PN-Cnt |  +-------------+
                       +----------+
                            |
  Transport Layer
  +-------------------------------------------------------------+
  |  SyncTransport trait                                         |
  |    +----------+  +----------+  +-----------+                 |
  |    |   UDP    |  |   TCP    |  |  Channel  |                 |
  |    | + Reliab |  | Length-  |  | In-process|                 |
  |    | + Frag   |  | prefixed |  | (testing) |                 |
  |    +----------+  +----------+  +-----------+                 |
  +-------------------------------------------------------------+
                            |
  Core Engine
  +-------------------------------------------------------------+
  |  Node (causal ordering) <- Event Stream <- Protocol (bitcode)|
  |  World (AoS, O(1) hash)    WorldSoA (8-wide SIMD)           |
  |  Fixed-point (Q16.16)      Input Sync (Lockstep/Rollback)   |
  +-------------------------------------------------------------+
```

## Optimization History

### v0.6 "Demon Mode" - Sort-Based Batching + P2P Infrastructure

- Sort events by entity ID for cache-friendly access
- Coalesce same-entity updates (5.5x faster for bursts)
- Re-sort by slot for true memory locality
- Auto-detect contiguous ranges for SIMD
- **NEW**: Complete async P2P stack (transport, discovery, pub/sub, CRDTs, session)

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

## Input Synchronization

Deterministic input sync for multiplayer games — **~20 bytes/player/frame**.

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
        // Unrecoverable - reconnect
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
| `async` | No | Full P2P infrastructure: transport, discovery, pub/sub, CRDTs, session driver (requires tokio) |
| `simd` | No | SIMD acceleration hints |
| `python` | No | Python bindings (PyO3 0.28 + NumPy zero-copy) |
| `physics` | No | ALICE-Physics bridge (InputFrame <-> FrameInput, PhysicsRollbackSession) |
| `telemetry` | No | Sync telemetry recording via ALICE-DB (RTT, rollback, desync metrics) |
| `cache` | No | Markov oracle entity prefetching via ALICE-Cache |
| `auth` | No | Ed25519 ZKP peer authentication via ALICE-Auth |
| `codec` | No | Wavelet + rANS event stream compression via ALICE-Codec |
| `analytics` | No | DDSketch/HLL/CMS probabilistic telemetry via ALICE-Analytics |
| `ffi` | No | C-ABI FFI for Unity/UE5 (71 functions) |
| `cloud` | No | Cloud-side multi-device spatial synchronization hub |
| `all-bridges` | No | Enable all integration bridges (`physics` + `telemetry` + `cache` + `auth` + `codec` + `analytics`) |

## C-ABI FFI (Unity / UE5)

71 `extern "C"` functions covering all core types. Enable with `--features ffi`.

### Coverage

| Module | Functions | Description |
|--------|-----------|-------------|
| World (AoS) | 8 | new, free, hash, entity_count, frame, apply_event, get_position, recalculate_hash |
| WorldSoA | 5 | new, free, hash, entity_count, apply_event |
| Node | 11 | new, with_seed, free, hash, state, emit, apply, add_peer, entity_count, events_count, events_bytes |
| Event | 8 | new_motion, new_spawn, new_despawn, new_property, new_input, new_tick, free, size_bytes |
| EventStream | 7 | new, free, push, len, is_empty, current_seq, total_bytes |
| InputFrame | 10 | new, free, set/get movement, set/get actions, set/get aim, get_frame, get_player_id |
| InputFrameArray | 3 | free, len, get |
| Lockstep | 5 | new, free, add_local/remote_input, ready, advance, confirmed_frame, record_checksum |
| Rollback | 7 | new, free, add_local/remote_input, confirmed/predicted_frame, frames_ahead |
| Utilities | 4 | fixed_from_f32, fixed_to_f32, vec3_hash, version |

### Unity C# (`bindings/unity/AliceSync.cs`)

71 `[DllImport]` declarations with RAII `IDisposable` handles.

### UE5 C++ (`bindings/ue5/AliceSync.h`)

71 `extern "C"` declarations with 9 RAII `std::unique_ptr` handles.

## Python Bindings (PyO3 + NumPy Zero-Copy)

```toml
[features]
python = ["pyo3", "numpy"]
```

### Optimization Layers

| Layer | Technique | Effect |
|-------|-----------|--------|
| L1 | Zero-Copy NumPy (`into_pyarray`) | No memcpy for positions |
| L2 | Batch API (SoA world operations) | FFI amortization |
| L3 | Rust backend (8-wide SIMD, Demon Mode) | Hardware-speed apply |

### Python API

```python
import alice_sync

# --- WorldSoA (high-performance SoA world) ---
world = alice_sync.WorldSoA(seed=42)
world.spawn(entity_id=0, kind=0, x=0, y=0, z=0)
world.spawn(entity_id=1, kind=0, x=100, y=0, z=0)

# Batch motions (Demon Mode SIMD)
motions = [(0, 10, 0, 0), (1, -5, 3, 0)]  # (entity_id, dx, dy, dz)
world.apply_motions(motions)

positions = world.positions()  # NumPy (N, 3) int32 - zero-copy
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
Network --> InputFrame (i16, 24B) --> FrameInput (Fix128) --> PhysicsWorld
                 ALICE-Sync                ALICE-Physics

PhysicsWorld --> SimulationChecksum --> WorldHash --> Desync Verification
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
| `sync_input_to_physics()` | InputFrame (i16 Q8.8) -> FrameInput (Fix128) |
| `physics_input_to_sync()` | FrameInput (Fix128) -> InputFrame (i16, truncate) |
| `sync_inputs_to_physics()` | Batch conversion |
| `physics_checksum_to_world_hash()` | SimulationChecksum(u64) -> WorldHash(u64) |
| `world_hash_to_physics_checksum()` | WorldHash(u64) -> SimulationChecksum(u64) |

## Sync Telemetry (ALICE-DB Integration)

Record network synchronization metrics as time-series data in [ALICE-DB](../ALICE-DB). Enable with `--features telemetry`.

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

## Cross-Crate Bridges

### Cache Bridge (feature: `cache`)

CRDT-based distributed cache invalidation with [ALICE-Cache](../ALICE-Cache). When sync events modify shared state, the cache bridge propagates invalidation messages to ensure distributed cache consistency across nodes.

### Codec Bridge (feature: `codec`)

Wavelet + rANS compression for event stream batches via [ALICE-Codec](../ALICE-Codec). Reduces event batch wire size by applying 1D wavelet transform and entropy coding to serialized event data.

```rust
use alice_sync::codec_bridge::{compress_event_batch, decompress_event_batch, estimate_ratio};

let compressed = compress_event_batch(&event_bytes, quality)?;
let ratio = estimate_ratio(&event_bytes);
let original = decompress_event_batch(&compressed)?;
```

### Analytics Bridge (feature: `analytics`)

Real-time sync telemetry via [ALICE-Analytics](../ALICE-Analytics). Feeds event throughput, round-trip latency, unique peer count, and hash divergences into probabilistic sketches (DDSketch, HyperLogLog, Count-Min Sketch).

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

## Use Cases

- **Multiplayer Games**: RTS, fighting games, FPS, physics puzzles (Lockstep / Rollback)
- **Real-Time Collaboration**: Document co-editing, 3D model sync, whiteboarding (CRDT mode)
- **IoT / Edge Computing**: Sensor mesh synchronization, device state convergence (CRDT + mDNS discovery)
- **Distributed Databases**: Change-data-capture replication, multi-master sync (Event Sourcing + CRDTs)
- **Messaging Infrastructure**: Topic-based pub/sub, group chat delivery, notification fanout
- **Game Engine Pipeline**: ALICE-Physics + ALICE-Sync for bit-exact rollback netcode
- **Distributed Simulation**: Weather, traffic, physics (deterministic event diffing)
- **Edge Caching**: Distributed cache coherence with eventual consistency

## Protocol

```
Handshake:
  A -> B: Hello { node_id, seq, world_hash }
  B -> A: Hello { node_id, seq, world_hash }

Sync:
  A -> B: Events { [event1, event2, ...] }  // bitcode encoded
  B -> A: Ack { seq }

Consistency Check:
  A -> B: HashCheck { seq, world_hash }
  B -> A: HashCheck { seq, world_hash }  // O(1) comparison
```

## Test Suite

195 tests across core modules, async P2P infrastructure, bridge integrations, and FFI:

| Scope | Command | Tests |
|-------|---------|-------|
| Core | `cargo test` | 139 |
| + Async P2P | `cargo test --features async` | 195 |
| + Physics | `cargo test --features physics` | +6 |
| + All bridges | `cargo test --features all-bridges,cloud,ffi` | all |

```bash
cargo test                          # Core tests (139)
cargo test --features async         # Core + P2P infrastructure (195)
cargo bench                         # Benchmarks
```

### Quality

| Metric | Value |
|--------|-------|
| clippy (pedantic+nursery) | 0 warnings |
| fmt | clean |
| Tests (default) | 139 |
| Tests (async) | 195 |

## License

AGPL-3.0

Commercial licensing available for enterprise use.

## Author

Moroya Sakamoto

---

*"The best sync is one where data never travels."*
