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

## Use Cases

- **Multiplayer Games**: RTS, fighting games, physics puzzles
- **Distributed Simulation**: Weather, traffic, physics
- **Collaborative Editing**: Real-time document/3D model sync
- **Rollback Netcode**: Foundation for fighting game networking

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

## Running Benchmarks

```bash
cargo bench
```

## License

AGPL-3.0

Commercial licensing available for enterprise use.

## Author

Moroya Sakamoto

---

*"The best sync is one where data never travels."*
