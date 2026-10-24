//! Oracle for the two world types: `World` (entities in an arena) and `WorldSoA` (columns) hold the
//! same values for the same events, at one scale (positions are `Fixed`, Q16.16; an event carries
//! Q8.8) and under one entity hash, and a Spawn of an id already live replaces that entity.
//!
//! oracle: `Fixed::from_i16` (Q8.8 into Q16.16, `n << 8`) for the expected raw positions, and
//! `Entity::compute_hash` of the expected entity for the hash.

use alice_sync::fixed_point::{Fixed, Vec3Fixed};
use alice_sync::world::EntityProps;
use alice_sync::{Entity, Event, EventKind, World, WorldSoA};

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
    let mut w = World::new(0);
    let mut q = WorldSoA::new(0);
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
