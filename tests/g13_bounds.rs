//! Oracle for the bounds of `converge` (ALICE-Law G13 with G16): which entities count is decided by
//! the set of events received, never by the order they arrive in.
//!
//! oracle (independent of the bounded code path): the records kept are those of the `max_records`
//! smallest ids any event names; the world is what an unbounded record set makes of the events of
//! those ids, cut to its `max_live` smallest live ids. Every arrival order of the same events, with
//! duplicates, must give that world, with an incremental hash equal to a full recomputation.

use alice_sync::converge::Converge;
use alice_sync::event::{EventId, SeqNum};
use alice_sync::{Event, EventKind, World};

type EntityRow = (u32, u16, [i32; 3], [i32; 16], u16);

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    entities: Vec<EntityRow>,
    frame: u64,
    hash: u64,
    recomputed: u64,
}

fn rows(world: &World, keep: impl Fn(u32) -> bool) -> Vec<EntityRow> {
    world
        .iter_entities()
        .filter(|e| keep(e.id))
        .map(|e| {
            (
                e.id,
                e.kind,
                [e.position.x.0, e.position.y.0, e.position.z.0],
                e.properties.values,
                e.properties.active,
            )
        })
        .collect()
}

fn snapshot(world: &World) -> Snapshot {
    let mut copy = world.clone();
    Snapshot {
        entities: rows(world, |_| true),
        frame: world.frame(),
        hash: world.hash().0,
        recomputed: copy.recalculate_hash().0,
    }
}

/// xorshift64*: a fixed, seeded stream for the generated cases
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn entity_of(e: &Event) -> Option<u32> {
    match e.kind {
        EventKind::Spawn { entity, .. }
        | EventKind::Despawn { entity }
        | EventKind::Motion { entity, .. }
        | EventKind::Property { entity, .. } => Some(entity),
        _ => None,
    }
}

/// `n` stamped events from three origins over ids `0..ids`, keys distinct
fn events(rng: &mut Rng, n: usize, ids: u32) -> Vec<Event> {
    (0..n)
        .map(|i| {
            let entity = u32::try_from(rng.below(u64::from(ids))).unwrap();
            let kind = match rng.below(10) {
                0..=3 => EventKind::Spawn {
                    entity,
                    kind: u16::try_from(rng.below(3)).unwrap(),
                    pos: [i16::try_from(rng.below(50)).unwrap(), 1, 2],
                },
                4 | 5 => EventKind::Despawn { entity },
                6 | 7 => EventKind::Motion {
                    entity,
                    delta: [1, i16::try_from(rng.below(5)).unwrap(), 0],
                },
                8 => EventKind::Property {
                    entity,
                    prop: u16::try_from(rng.below(4)).unwrap(),
                    value: i32::try_from(rng.below(100)).unwrap(),
                },
                _ => EventKind::Tick {
                    frame: rng.below(20),
                },
            };
            Event {
                id: EventId(i as u64 + 1),
                seq: SeqNum(i as u64 + 1),
                origin: 1 + rng.below(3),
                timestamp: rng.below(12),
                kind,
            }
        })
        .collect()
}

fn apply_all(events: &[Event], max_live: usize, max_records: usize) -> World {
    let mut c = Converge::with_limits(max_live, max_records);
    let mut w = World::new(0);
    for e in events {
        c.apply(&mut w, e).unwrap();
    }
    w
}

/// The model: unbounded records over the kept ids, cut to the smallest live ids
fn model(events: &[Event], max_live: usize, max_records: usize) -> Vec<EntityRow> {
    let mut named: Vec<u32> = events.iter().filter_map(entity_of).collect();
    named.sort_unstable();
    named.dedup();
    named.truncate(max_records);
    let kept: Vec<Event> = events
        .iter()
        .filter(|e| entity_of(e).is_none_or(|id| named.contains(&id)))
        .cloned()
        .collect();
    let unbounded = apply_all(&kept, usize::MAX, usize::MAX);
    let live: Vec<u32> = unbounded
        .iter_entities()
        .map(|e| e.id)
        .take(max_live)
        .collect();
    rows(&unbounded, |id| live.contains(&id))
}

fn shuffled(rng: &mut Rng, events: &[Event]) -> Vec<Event> {
    let mut v = events.to_vec();
    for i in (1..v.len()).rev() {
        let j = usize::try_from(rng.below(i as u64 + 1)).unwrap();
        v.swap(i, j);
    }
    // a third of them delivered twice
    let extra: Vec<Event> = v.iter().step_by(3).cloned().collect();
    v.extend(extra);
    v
}

#[test]
fn bounded_worlds_are_the_same_for_every_arrival_order_and_match_the_model() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (max_live, max_records) = (3, 6);
    let mut compared = 0;
    let mut bound_bit = 0; // cases where a bound actually removed something
    for _ in 0..400 {
        let ev = events(&mut rng, 30, 10);
        let want = model(&ev, max_live, max_records);
        let first = snapshot(&apply_all(&ev, max_live, max_records));
        assert_eq!(first.entities, want, "events {ev:?}");
        assert_eq!(first.hash, first.recomputed);
        let unbounded = apply_all(&ev, usize::MAX, usize::MAX);
        if rows(&unbounded, |_| true) != want {
            bound_bit += 1;
        }
        let mut reversed = ev.clone();
        reversed.reverse();
        for order in [reversed, shuffled(&mut rng, &ev), shuffled(&mut rng, &ev)] {
            assert_eq!(snapshot(&apply_all(&order, max_live, max_records)), first);
            compared += 1;
        }
    }
    assert_eq!(compared, 1200);
    // the bounds must bite in a good share of the cases, or this test checks nothing about them
    assert!(
        bound_bit >= 100,
        "the bounds changed only {bound_bit} of 400 worlds"
    );
}

#[test]
fn a_smaller_id_evicts_the_largest_and_a_despawn_readmits_the_next() {
    let ev = |seq: u64, kind| Event {
        id: EventId(seq),
        seq: SeqNum(seq),
        origin: 1,
        timestamp: seq,
        kind,
    };
    let spawn = |entity| EventKind::Spawn {
        entity,
        kind: 0,
        pos: [0, 0, 0],
    };
    let mut c = Converge::with_limits(2, 10);
    let mut w = World::new(0);
    for (i, id) in [5u32, 7, 9].into_iter().enumerate() {
        c.apply(&mut w, &ev(i as u64 + 1, spawn(id))).unwrap();
    }
    let ids = |w: &World| w.iter_entities().map(|e| e.id).collect::<Vec<_>>();
    assert_eq!(ids(&w), [5, 7]);
    c.apply(&mut w, &ev(4, spawn(1))).unwrap();
    assert_eq!(ids(&w), [1, 5]);
    c.apply(&mut w, &ev(5, EventKind::Despawn { entity: 1 }))
        .unwrap();
    assert_eq!(ids(&w), [5, 7]);
    c.apply(&mut w, &ev(6, EventKind::Despawn { entity: 5 }))
        .unwrap();
    assert_eq!(ids(&w), [7, 9]);
    let mut copy = w.clone();
    assert_eq!(w.hash(), copy.recalculate_hash());
}

/// The world itself holds no more than `MAX_LIVE_ENTITIES`: a Spawn of a new id past it is refused,
/// a Spawn of a live id is not, and a stored world never holds more than a decode accepts
#[test]
fn the_world_refuses_a_new_entity_past_the_limit() {
    use alice_sync::world::MAX_LIVE_ENTITIES;
    let spawn = |entity| {
        Event::new(EventKind::Spawn {
            entity,
            kind: 0,
            pos: [0, 0, 0],
        })
    };
    let mut w = World::new(0);
    let limit = u32::try_from(MAX_LIVE_ENTITIES).unwrap();
    for id in 0..limit {
        w.apply(&spawn(id)).unwrap();
    }
    assert!(matches!(
        w.apply(&spawn(limit)),
        Err(alice_sync::SyncError::TooManyEntities { limit: l }) if l == MAX_LIVE_ENTITIES
    ));
    assert_eq!(w.iter_entities().count(), MAX_LIVE_ENTITIES);
    // a live id may spawn again (a new incarnation replaces it after its Despawn)
    w.apply(&Event::new(EventKind::Despawn { entity: 0 }))
        .unwrap();
    w.apply(&spawn(0)).unwrap();
    w.apply(&Event::new(EventKind::Despawn { entity: 7 }))
        .unwrap();
    w.apply(&spawn(limit)).unwrap();
    assert_eq!(w.iter_entities().count(), MAX_LIVE_ENTITIES);
}

#[test]
fn the_column_world_refuses_a_new_entity_past_the_limit() {
    // the same bound as World's (the_world_refuses_a_new_entity_past_the_limit), for WorldSoA
    use alice_sync::world::MAX_LIVE_ENTITIES;
    use alice_sync::WorldSoA;
    let spawn = |entity| {
        Event::new(EventKind::Spawn {
            entity,
            kind: 0,
            pos: [0, 0, 0],
        })
    };
    let mut w = WorldSoA::new(0);
    let limit = u32::try_from(MAX_LIVE_ENTITIES).unwrap();
    for id in 0..limit {
        w.apply(&spawn(id)).unwrap();
    }
    let before = w.hash();
    assert!(matches!(
        w.apply(&spawn(limit)),
        Err(alice_sync::SyncError::TooManyEntities { limit: l }) if l == MAX_LIVE_ENTITIES
    ));
    assert_eq!(w.entity_count(), MAX_LIVE_ENTITIES);
    assert_eq!(w.hash(), before, "the refused spawn changed nothing");
    // a live id may spawn again, and a freed place takes a new id
    w.apply(&spawn(0)).unwrap();
    w.apply(&Event::new(EventKind::Despawn { entity: 7 }))
        .unwrap();
    w.apply(&spawn(limit)).unwrap();
    assert_eq!(w.entity_count(), MAX_LIVE_ENTITIES);
}

#[test]
fn a_state_with_more_live_entities_than_the_limit_is_refused_by_the_column_world() {
    // a WorldState built field by field (no World holds more than the limit): the limit itself converts,
    // one more is refused
    use alice_sync::world::MAX_LIVE_ENTITIES;
    use alice_sync::{Entity, EntityProps, Vec3Fixed, WorldSoA, WorldState};
    let state_of = |n: u32| {
        let mut state = WorldState::default();
        for id in 0..n {
            let h = state.entities.insert(Entity {
                id,
                kind: 0,
                position: Vec3Fixed::ZERO,
                properties: EntityProps::default(),
            });
            state.set_handle(id, h);
        }
        state
    };
    let limit = u32::try_from(MAX_LIVE_ENTITIES).unwrap();
    let at = WorldSoA::from_state(&state_of(limit)).unwrap();
    assert_eq!(at.entity_count(), MAX_LIVE_ENTITIES);
    assert!(matches!(
        WorldSoA::from_state(&state_of(limit + 1)),
        Err(alice_sync::SyncError::TooManyEntities { limit: l }) if l == MAX_LIVE_ENTITIES
    ));
}
