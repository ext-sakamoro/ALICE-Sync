//! Oracle for the keyed lookup table of the worlds (ALICE-Law G10 allowance): the per-table key
//! changes only where entries sit, never an output. Two worlds built from the same events, each with
//! its own key, give bit-identical stored bytes, world hashes and entity lists, while the order the
//! tables hold their entries in (visible in `Debug`) differs, so the two keys really differed.

use alice_sync::persist::encode_state;
use alice_sync::{Event, EventKind, World, WorldSoA};

fn spawn(entity: u32) -> Event {
    Event::new(EventKind::Spawn {
        entity,
        kind: u16::try_from(entity % 7).unwrap(),
        pos: [i16::try_from(entity % 100).unwrap(), 1, -2],
    })
}

/// 2000 ids spread over the whole range, some despawned again
fn ids() -> Vec<u32> {
    (0..2000u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect()
}

fn build() -> World {
    let mut w = World::new(3);
    for id in ids() {
        w.apply(&spawn(id)).unwrap();
    }
    for id in ids().into_iter().step_by(5) {
        w.apply(&Event::new(EventKind::Despawn { entity: id }))
            .unwrap();
    }
    w.apply(&Event::new(EventKind::Property {
        entity: ids()[1],
        prop: 3,
        value: -9,
    }))
    .unwrap();
    w
}

#[test]
fn worlds_with_different_keys_store_the_same_bytes_and_hash() {
    let (a, b) = (build(), build());
    // the tables were keyed differently: their raw entry order differs
    assert_ne!(
        format!("{:?}", a.state()),
        format!("{:?}", b.state()),
        "the two worlds' tables have the same layout: the keys did not differ, so this test \
         compares nothing about them"
    );
    assert_eq!(encode_state(a.state()), encode_state(b.state()));
    assert_eq!(a.hash(), b.hash());
    let (mut ca, mut cb) = (a.clone(), b.clone());
    assert_eq!(ca.recalculate_hash(), cb.recalculate_hash());
    let list = |w: &World| w.iter_entities().map(|e| e.id).collect::<Vec<_>>();
    assert_eq!(list(&a), list(&b));
    let sorted = {
        let mut v = list(&a);
        v.sort_unstable();
        v
    };
    assert_eq!(list(&a), sorted, "entities are listed in id order");
    assert_eq!(a.state().live_count(), 1600);
}

#[test]
fn column_worlds_with_different_keys_hash_the_same() {
    let build_soa = || {
        let mut w = WorldSoA::new(3);
        for id in ids() {
            w.apply(&spawn(id)).unwrap();
        }
        for id in ids().into_iter().step_by(5) {
            w.apply(&Event::new(EventKind::Despawn { entity: id }))
                .unwrap();
        }
        w
    };
    let (mut a, mut b) = (build_soa(), build_soa());
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.recalculate_hash(), b.recalculate_hash());
}
