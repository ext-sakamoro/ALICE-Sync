//! Each statement uses one form that `.github/clippy-lib/clippy.toml` forbids in
//! the library code; `scripts/clippy_lib_gate.py` runs clippy with that
//! configuration and requires a `disallowed_methods` diagnostic naming each one

use alice_sync::world::World;
use alice_sync::world_soa::WorldSoA;
use alice_sync::{Event, EventKind, EventStream, MotionData, SpawnData};
use wide::i32x8;

pub fn world(e: Event, m: Vec<MotionData>, s: Vec<SpawnData>, st: EventStream) {
    let mut w = World::new(0);
    let _ = w.apply(&e);
    let _ = w.get_entity_mut(1);
    w.apply_motions_batch(&m);
    w.apply_spawns_batch(&s);
    let _ = w.apply_stream_batched(&st);
}

pub fn world_soa(e: Event, m: Vec<MotionData>) {
    let mut soa = WorldSoA::new(0);
    let d = i32x8::splat(1);
    let _ = soa.apply(&e);
    let _ = soa.storage.spawn(1, 0, 0, 0, 0);
    let _ = soa.storage.despawn(0);
    soa.storage.set_position(0, 1, 2, 3);
    soa.storage.set_property(0, 1, 2);
    soa.apply_motions_vertical_8(0, d, d, d);
    soa.apply_motions_batch(&m);
    soa.apply_motions_sorted(m.clone());
    soa.apply_motions_presorted(&m);
    soa.apply_motions_demon(m.clone());
    soa.apply_motions_nosort(&m);
    soa.apply_uniform_motion_vertical(0, 8, 1, 1, 1);
    let _ = EventKind::Tick { frame: 0 };
}
