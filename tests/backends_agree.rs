//! Oracle for the two world types: `World` (entities in an arena) and `WorldSoA` (columns) hold the
//! same values for the same events, at one scale (positions are `Fixed`, Q16.16; an event carries
//! Q8.8) and under one entity hash, and a Spawn of an id already live replaces that entity.
//!
//! oracle: `Fixed::from_i16` (Q8.8 into Q16.16, `n << 8`) for the expected raw positions, and
//! `Entity::compute_hash` of the expected entity for the hash.

use alice_sync::fixed_point::{Fixed, Vec3Fixed};
use alice_sync::persist::{decode_state, encode_state};
use alice_sync::world::EntityProps;
use alice_sync::{Entity, Event, EventKind, World, WorldSoA, WorldState};

fn spawn(entity: u32, pos: [i16; 3]) -> Event {
    Event::new(EventKind::Spawn {
        entity,
        kind: 2,
        pos,
    })
}

fn motion(entity: u32, delta: [i16; 3]) -> Event {
    Event::new(EventKind::Motion { entity, delta })
}

fn both(events: &[Event]) -> (World, WorldSoA) {
    let mut w = World::new(77);
    let mut q = WorldSoA::new(77);
    for e in events {
        w.apply(e).unwrap();
        q.apply(e).unwrap();
    }
    (w, q)
}

/// The one live slot of a column world: its position
fn soa_position(q: &WorldSoA) -> (Fixed, Fixed, Fixed) {
    let st = q.storage();
    let slot = (0..4096u32).find(|&s| st.is_alive(s)).expect("a live slot");
    st.get_position(slot)
}

#[test]
fn a_spawn_and_a_motion_give_the_same_raw_position_and_hash_in_both_types() {
    let (w, q) = both(&[spawn(1, [3, -2, 1]), motion(1, [1, 0, 0])]);
    let want = Vec3Fixed {
        x: Fixed::from_i16(4),
        y: Fixed::from_i16(-2),
        z: Fixed::from_i16(1),
    };
    assert_eq!(
        (want.x.0, want.y.0, want.z.0),
        (1024, -512, 256),
        "Q8.8 into Q16.16"
    );
    let e = w.iter_entities().next().unwrap();
    assert_eq!(
        (e.position.x.0, e.position.y.0, e.position.z.0),
        (1024, -512, 256)
    );
    let (x, y, z) = soa_position(&q);
    assert_eq!((x.0, y.0, z.0), (1024, -512, 256));
    let expected = Entity {
        id: 1,
        kind: 2,
        position: want,
        properties: EntityProps::EMPTY,
    };
    assert_eq!(w.hash().0, expected.compute_hash());
    assert_eq!(q.hash().0, expected.compute_hash());
}

#[test]
fn properties_hash_the_same_in_both_types() {
    let (w, q) = both(&[
        spawn(7, [0, 0, 0]),
        Event::new(EventKind::Property {
            entity: 7,
            prop: 3,
            value: 41,
        }),
        Event::new(EventKind::Property {
            entity: 7,
            prop: 9,
            value: -5,
        }),
    ]);
    let mut properties = EntityProps::EMPTY;
    properties.set(3, 41);
    properties.set(9, -5);
    let expected = Entity {
        id: 7,
        kind: 2,
        position: Vec3Fixed::from_i16_array([0, 0, 0]),
        properties,
    };
    assert_eq!(w.hash().0, expected.compute_hash());
    assert_eq!(q.hash().0, expected.compute_hash());
}

#[test]
fn a_spawn_of_a_live_id_replaces_it_in_both_types() {
    let (w, q) = both(&[
        spawn(5, [1, 1, 1]),
        spawn(5, [9, 9, 9]),
        spawn(5, [2, 3, 4]),
    ]);
    let (fresh_w, fresh_q) = both(&[spawn(5, [2, 3, 4])]);
    assert_eq!(w.entity_count(), 1, "the earlier incarnations are gone");
    assert_eq!(q.entity_count(), 1);
    assert_eq!(w.hash(), fresh_w.hash());
    assert_eq!(q.hash(), fresh_q.hash());
    let (mut cw, mut cq) = (w.clone(), q.clone());
    assert_eq!(cw.recalculate_hash(), w.hash());
    assert_eq!(cq.recalculate_hash(), q.hash());
}

/// A seeded stream of events of every kind over ids 0..60
fn stream(seed: u64, n: usize) -> Vec<Event> {
    let mut x = seed;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    (0..n)
        .map(|_| {
            let r = next();
            let entity = u32::try_from(r % 60).unwrap();
            let small =
                |v: u64, m: u64| i16::try_from(v % m).unwrap() - i16::try_from(m / 2).unwrap();
            Event::new(match (r >> 8) % 9 {
                0..=2 => EventKind::Spawn {
                    entity,
                    kind: u16::try_from((r >> 16) % 4).unwrap(),
                    pos: [
                        small(r >> 20, 200),
                        small(r >> 28, 200),
                        small(r >> 36, 200),
                    ],
                },
                3 => EventKind::Despawn { entity },
                4..=6 => EventKind::Motion {
                    entity,
                    delta: [small(r >> 20, 30), small(r >> 28, 30), small(r >> 36, 30)],
                },
                7 => EventKind::Property {
                    entity,
                    prop: u16::try_from((r >> 16) % 20).unwrap(),
                    value: i32::try_from((r >> 24) % 2001).unwrap() - 1000,
                },
                _ => EventKind::Tick { frame: r >> 40 },
            })
        })
        .collect()
}

#[test]
fn both_types_store_the_same_bytes_for_the_same_events() {
    let mut compared = 0;
    for seed in 1..=40u64 {
        let events = stream(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15), 400);
        let (w, q) = both(&events);
        let from_columns = q.to_state();
        assert_eq!(
            encode_state(&from_columns),
            encode_state(w.state()),
            "seed {seed}"
        );
        assert_eq!(q.hash(), w.hash(), "seed {seed}");
        compared += 1;
    }
    assert_eq!(compared, 40);
}

#[test]
fn a_column_world_round_trips_through_the_stored_form() {
    for seed in 1..=20u64 {
        let events = stream(seed.wrapping_mul(0xd6e8_feb8_6659_fd93), 300);
        let (_, q) = both(&events);
        let bytes = encode_state(&q.to_state());
        let back =
            WorldSoA::from_state(&decode_state::<WorldState>(&bytes).unwrap().value).unwrap();
        assert_eq!(encode_state(&back.to_state()), bytes, "seed {seed}");
        assert_eq!(back.hash(), q.hash(), "seed {seed}");
        let mut copy = back.clone();
        assert_eq!(copy.recalculate_hash(), back.hash());
        assert_eq!(back.entity_count(), q.to_state().live_count());
    }
}
