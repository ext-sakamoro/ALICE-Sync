//! Cross-platform determinism goldens of ALICE-Sync's Law-bearing state.
//!
//! Each golden is the SHA-256 of the bytes a fixed scenario produces: the
//! Q16.16 arithmetic of `Fixed` / `Vec3Fixed`, the world state and rolling
//! hash after a fixed event sequence, and the merged state of the CRDTs
//! (with `async`, the feature that builds `crdt`). The digests are recorded
//! on one platform; every CI leg (debug and release) must reproduce them bit
//! for bit. They are also the pins of `SYNC_SEMANTICS_ID`
//! (`src/semantics.rs`): re-recording a golden after an intended change
//! means updating the pin there and the identifier.

use alice_sync::fixed_point::{Fixed, Vec3Fixed};
use alice_sync::{Event, EventKind, Node, NodeId, World};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The SHA-256 of `bytes` when it is not `expected`
fn golden_mismatch(bytes: &[u8], expected: &str) -> Option<String> {
    let got = digest(bytes);
    (got != expected).then_some(got)
}

fn assert_golden(bytes: &[u8], expected: &str, what: &str) {
    if let Some(got) = golden_mismatch(bytes, expected) {
        panic!("{what}: {got}, recorded {expected}");
    }
}

#[test]
fn checker_rejects_a_wrong_digest() {
    let recorded = digest(b"recorded bits");
    assert_eq!(golden_mismatch(b"recorded bits", &recorded), None);
    assert!(
        golden_mismatch(b"moved bits", &recorded).is_some(),
        "a scenario whose bits moved must not match its recorded digest"
    );
}

/// Inputs spanning signs, the step, saturation edges and large magnitudes.
const RAW: [i32; 14] = [
    0,
    1,
    -1,
    2,
    -2,
    65_536,
    -65_536,
    98_304,
    -98_304,
    1 << 30,
    -(1 << 30),
    i32::MAX,
    i32::MIN,
    123_456_789,
];
const FLOATS: [f32; 10] = [
    0.0, -0.0, 1.0, -1.0, 0.5, -1.5, 1.0e-5, 3.25, -32_767.5, 32_767.999,
];

pub const GOLDEN_FIXED: &str = "dc65b542978e3eb841faee4a7202f7416cee5cd6ce4fdd64cebc18a97ba31f9d";

#[test]
fn golden_fixed() {
    let mut out = Vec::new();
    for &a in &RAW {
        let fa = Fixed::from_bits(a);
        for &b in &RAW {
            let fb = Fixed::from_bits(b);
            out.extend_from_slice(&fa.saturating_add(fb).to_bits().to_le_bytes());
            out.extend_from_slice(&fa.saturating_sub(fb).to_bits().to_le_bytes());
            out.extend_from_slice(&fa.saturating_mul(fb).to_bits().to_le_bytes());
            // wrapping operators are defined on in-range operands only
            if a.unsigned_abs() < (1 << 20) && b.unsigned_abs() < (1 << 20) {
                out.extend_from_slice(&(fa + fb).to_bits().to_le_bytes());
                out.extend_from_slice(&(fa - fb).to_bits().to_le_bytes());
                out.extend_from_slice(&(fa * fb).to_bits().to_le_bytes());
            }
        }
        out.extend_from_slice(&fa.to_f32().to_bits().to_le_bytes());
        out.extend_from_slice(&fa.to_i16().to_le_bytes());
    }
    for &f in &FLOATS {
        let x = Fixed::from_f32(f);
        out.extend_from_slice(&x.to_bits().to_le_bytes());
        out.extend_from_slice(&x.to_f32().to_bits().to_le_bytes());
    }
    for n in [-32_768_i16, -1, 0, 1, 32_767] {
        out.extend_from_slice(&Fixed::from_i16(n).to_bits().to_le_bytes());
    }
    let v = Vec3Fixed::from_f32(1.25, -2.5, 0.0) + Vec3Fixed::from_i16_array([3, -4, 5]);
    out.extend_from_slice(&v.hash_bits().to_le_bytes());
    for c in v.to_f32_array() {
        out.extend_from_slice(&c.to_bits().to_le_bytes());
    }
    assert_golden(&out, GOLDEN_FIXED, "Fixed arithmetic golden moved");
}

pub const GOLDEN_WORLD: &str = "875410bef5edafaaff33e1e74a2c879b987e394f784f5ef361b6d828d531f593";

fn scenario() -> Vec<EventKind> {
    let mut ev = Vec::new();
    for i in 0..8u32 {
        let k = i as i16;
        ev.push(EventKind::Spawn {
            entity: i,
            kind: (i % 3) as u16,
            pos: [k * 100, -k * 7, 32_000 - k],
        });
    }
    for step in 0..12i16 {
        for i in 0..8u32 {
            let k = i as i16;
            ev.push(EventKind::Motion {
                entity: i,
                delta: [step - k, k * 3 - step, (step * k) % 97],
            });
        }
        if step % 4 == 3 {
            ev.push(EventKind::Property {
                entity: (step as u32) % 8,
                prop: 2,
                value: i32::from(step) * -1_000,
            });
        }
    }
    ev.push(EventKind::Despawn { entity: 5 });
    ev.push(EventKind::Motion {
        entity: 5,
        delta: [1, 1, 1],
    }); // despawned: ignored
    ev.push(EventKind::Spawn {
        entity: 9,
        kind: 7,
        pos: [i16::MIN, i16::MAX, 0],
    });
    ev
}

#[test]
fn golden_world() {
    let mut world = World::new(42);
    let mut out = Vec::new();
    for kind in scenario() {
        world.apply(&Event::new(kind)).expect("apply is infallible");
        out.extend_from_slice(&world.hash().0.to_le_bytes());
    }
    out.extend_from_slice(&world.recalculate_hash().0.to_le_bytes());
    let mut entities: Vec<_> = world.iter_entities().copied().collect();
    entities.sort_by_key(|e| e.id);
    for e in entities {
        out.extend_from_slice(&e.id.to_le_bytes());
        out.extend_from_slice(&e.kind.to_le_bytes());
        out.extend_from_slice(&e.position.hash_bits().to_le_bytes());
        out.extend_from_slice(&e.compute_hash().to_le_bytes());
    }
    assert_golden(&out, GOLDEN_WORLD, "world state golden moved");
}

pub const GOLDEN_RECEIVE: &str = "b0f4dd9b848374e314f1d88ae220e926d8a3d05d49d08b3a77d8d9673bb2b242";

/// The world a node builds from events of three peers (ALICE-Law G13): the world scenario split
/// across three origins that see some of each other's events (so their Lamport times interleave),
/// with entity 2 re-spawned by another origin, received in reverse and with every third event
/// delivered twice
#[test]
fn golden_receive() {
    let mut peers = [
        Node::new(NodeId(1)),
        Node::new(NodeId(2)),
        Node::new(NodeId(3)),
    ];
    for (i, kind) in scenario().into_iter().enumerate() {
        peers[i % 3]
            .emit(Event::new(kind))
            .expect("emit is infallible");
        if i % 10 == 9 {
            let seen: Vec<Event> = peers[i % 3].events().all().to_vec();
            for e in &seen {
                peers[(i + 1) % 3]
                    .apply_event(e)
                    .expect("apply is infallible");
            }
        }
    }
    peers[1]
        .emit(Event::new(EventKind::Spawn {
            entity: 2,
            kind: 11,
            pos: [-5, 5, -5],
        }))
        .expect("emit is infallible");
    let mut all: Vec<Event> = Vec::new();
    for p in &peers {
        all.extend(p.events().all().iter().cloned());
    }
    let mut node = Node::new(NodeId(9));
    for (i, e) in all.iter().rev().enumerate() {
        node.apply_event(e).expect("apply is infallible");
        if i % 3 == 0 {
            node.apply_event(e).expect("apply is infallible");
        }
    }
    let world = node.world();
    let mut out = Vec::new();
    out.extend_from_slice(&world.hash().0.to_le_bytes());
    out.extend_from_slice(&world.frame().to_le_bytes());
    for e in world.iter_entities() {
        out.extend_from_slice(&e.id.to_le_bytes());
        out.extend_from_slice(&e.kind.to_le_bytes());
        out.extend_from_slice(&e.position.hash_bits().to_le_bytes());
        out.extend_from_slice(&e.compute_hash().to_le_bytes());
    }
    assert_golden(&out, GOLDEN_RECEIVE, "receive golden moved");
}

pub const GOLDEN_CRDT: &str = "54f05313b39a7b671f53bf5c87fb4e37cb1b64f40b402ef98517047c4a730bc8";

#[cfg(feature = "async")]
#[test]
fn golden_crdt() {
    use alice_sync::crdt::{CrdtMergeable, GCounter, LwwRegister, OrSet, PnCounter};
    let mut out = Vec::new();

    let (mut g1, mut g2) = (GCounter::new(), GCounter::new());
    g1.increment_by(1, 5);
    g2.increment_by(2, 7);
    g2.increment_by(1, 3);
    g1.merge(&g2);
    out.extend_from_slice(&g1.value().to_le_bytes());

    let (mut p1, mut p2) = (PnCounter::new(), PnCounter::new());
    p1.increment(1);
    p1.increment(1);
    p2.decrement(2);
    p2.decrement(2);
    p2.decrement(2);
    p1.merge(&p2);
    out.extend_from_slice(&p1.value().to_le_bytes());

    let mut r1 = LwwRegister::new(10_i32, 1);
    let mut r2 = LwwRegister::new(20_i32, 2);
    r1.set_at(11, 5);
    r2.set_at(21, 5); // tie on timestamp: higher replica id wins
    r1.merge(&r2);
    out.extend_from_slice(&r1.value.to_le_bytes());
    out.extend_from_slice(&r1.timestamp.to_le_bytes());
    out.extend_from_slice(&r1.replica_id.to_le_bytes());

    let (mut s1, mut s2) = (OrSet::new(), OrSet::new());
    for v in [3_u32, 1, 4, 1, 5] {
        s1.add(1, v);
    }
    for v in [9_u32, 2, 6] {
        s2.add(2, v);
    }
    s1.remove(&4);
    s2.merge(&s1);
    for v in s2.values() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    assert_golden(&out, GOLDEN_CRDT, "CRDT merge golden moved");
}
