//! ALICE-Law G13: the world built from events received from several peers does not depend on the
//! order they arrive in (the async transport delivers them in an order the scheduler picks), nor on a
//! duplicate delivery. A node applies the events it accepts (signed by their authors,
//! `alice_sync::authorship`) through `alice_sync::converge::Converge`; the arrival orders below are
//! applied through it directly, and the node's own path is checked at the end (signed nodes).
//!
//! The oracle is independent of `alice_sync::converge`: the expected world is the one a single
//! writer gets by applying each distinct event once, in key order (timestamp, origin, sequence),
//! to a plain `World`, where a Spawn replaces any earlier incarnation of its entity. Every arrival
//! order below must reproduce it exactly: entities (id, kind, position, properties), frame and hash.
//! The model keeps every event and prunes nothing, so it is also the differential check of the
//! node's pruning (Motions keyed before a greater Spawn dropped). Two different events under one key
//! (two nodes with one id) are both in the set; the model applies them in content order.

use std::collections::BTreeMap;

use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use alice_sync::authorship::{Author, AuthorshipError, SignedEvents, TrustPolicy};
use alice_sync::converge::Converge;
use alice_sync::event::Origin;
use alice_sync::{Event, EventKind, Node, NodeId, World};

/// One entity: id, kind, position (raw Q16.16), property values, active-property mask
type EntityRow = (u32, u16, [i32; 3], [i32; 16], u16);

/// The world compared: every entity in id order, the frame, the incremental hash and the
/// recomputed one
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    entities: Vec<EntityRow>,
    frame: u64,
    hash: u64,
    recomputed: u64,
}

fn snapshot(world: &World) -> Snapshot {
    let entities = world
        .iter_entities()
        .map(|e| {
            (
                e.id,
                e.kind,
                [e.position.x.0, e.position.y.0, e.position.z.0],
                e.properties.values,
                e.properties.active,
            )
        })
        .collect();
    let mut copy = world.clone();
    Snapshot {
        entities,
        frame: world.frame(),
        hash: world.hash().0,
        recomputed: copy.recalculate_hash().0,
    }
}

fn key(e: &Event) -> (u64, Origin, u64) {
    (e.timestamp, e.origin, e.seq.0)
}

/// The order the model applies distinct events in: by key; under one key (a collision of two
/// nodes with the same id), a Spawn after the other kinds (an event keyed at its Spawn belongs to an
/// earlier incarnation, a Despawn keyed at it does not end it) and otherwise by content, ascending,
/// so the greater Spawn, Property value and Tick frame are the ones left standing
fn model_order(e: &Event) -> (u64, Origin, u64, u8, [i64; 4]) {
    let content = match e.kind {
        EventKind::Spawn { kind, pos, .. } => [
            i64::from(kind),
            i64::from(pos[0]),
            i64::from(pos[1]),
            i64::from(pos[2]),
        ],
        EventKind::Property { prop, value, .. } => [i64::from(prop), i64::from(value), 0, 0],
        EventKind::Tick { frame } => [frame as i64, 0, 0, 0],
        _ => [0; 4],
    };
    let spawn_last = u8::from(matches!(e.kind, EventKind::Spawn { .. }));
    (e.timestamp, e.origin, e.seq.0, spawn_last, content)
}

/// The expected world: each distinct event once, in key order, on a plain World
fn expected(events: &[Event]) -> Snapshot {
    let mut distinct: Vec<Event> = Vec::new();
    for e in events {
        if !distinct.contains(e) {
            distinct.push(e.clone());
        }
    }
    distinct.sort_by_key(model_order);
    let mut world = World::new(0);
    for e in &distinct {
        if let EventKind::Spawn { entity, .. } = e.kind {
            world
                .apply(&Event::new(EventKind::Despawn { entity }))
                .unwrap();
        }
        world.apply(e).unwrap();
    }
    snapshot(&world)
}

fn received(events: &[Event]) -> Snapshot {
    let mut world = World::new(0);
    let mut converge = Converge::new();
    for e in events {
        converge.apply(&mut world, e).unwrap();
    }
    snapshot(&world)
}

/// Three peers that partly see each other's events, so their Lamport times interleave, emitting
/// every case the order could decide: a Motion and a Property before (and from a peer that has not
/// seen) their Spawn, the same Motion from two origins at the same time and sequence, two writes of
/// one property at the same time from different origins, a Motion keyed before a re-spawn, a
/// Despawn and a later re-Spawn, a Despawn keyed before a Spawn it arrives after, Ticks from every
/// peer, and events that leave the world alone
///
/// Recorded as (kind, timestamp, origin, sequence number) from three nodes a (1), b (2) and c (3)
/// emitting while delivering to each other: a spawns 1 and 4 and delivers to b and c; b and c each emit
/// two equal Motions of 4 (same times, sequences and deltas: all four apply) and write property 3 of
/// 4 at the same time (origin 3 > 2 wins, not the larger value); b moves 1 and despawns 3 (keyed before
/// c's Spawn of 3); c spawns 5 and delivers to b, which despawns 5; c spawns 2; a moves 2 before seeing
/// its Spawn (not applied), sees c, moves 2 and writes its property; b despawns 1 and delivers to a,
/// which re-spawns 1 (a new incarnation: none of the motions above) and moves it; b delivers to c, which
/// spawns 3 after the Despawn it may arrive after; Ticks from each and an Input. A fourth node made with
/// a's id emits a Spawn of 6 stamped like a's first event: a collision both events survive.
fn scenario() -> Vec<Event> {
    vec![
        stamped(
            EventKind::Spawn {
                entity: 1,
                kind: 7,
                pos: [10, 0, 0],
            },
            1,
            1,
            1,
        ),
        stamped(
            EventKind::Spawn {
                entity: 4,
                kind: 1,
                pos: [0, 0, 0],
            },
            2,
            1,
            2,
        ),
        stamped(
            EventKind::Motion {
                entity: 2,
                delta: [1, 2, 3],
            },
            3,
            1,
            3,
        ),
        stamped(
            EventKind::Motion {
                entity: 2,
                delta: [0, 0, 9],
            },
            8,
            1,
            4,
        ),
        stamped(
            EventKind::Property {
                entity: 2,
                prop: 1,
                value: -7,
            },
            9,
            1,
            5,
        ),
        stamped(
            EventKind::Spawn {
                entity: 1,
                kind: 8,
                pos: [0, 0, 0],
            },
            10,
            1,
            6,
        ),
        stamped(
            EventKind::Motion {
                entity: 1,
                delta: [1, 1, 1],
            },
            11,
            1,
            7,
        ),
        stamped(EventKind::Tick { frame: 5 }, 12, 1, 8),
        stamped(
            EventKind::Motion {
                entity: 4,
                delta: [5, 0, 0],
            },
            3,
            2,
            1,
        ),
        stamped(
            EventKind::Motion {
                entity: 4,
                delta: [5, 0, 0],
            },
            4,
            2,
            2,
        ),
        stamped(
            EventKind::Property {
                entity: 4,
                prop: 3,
                value: 100,
            },
            5,
            2,
            3,
        ),
        stamped(
            EventKind::Motion {
                entity: 1,
                delta: [5, 0, 0],
            },
            6,
            2,
            4,
        ),
        stamped(EventKind::Despawn { entity: 3 }, 7, 2, 5),
        stamped(EventKind::Despawn { entity: 5 }, 8, 2, 6),
        stamped(EventKind::Despawn { entity: 1 }, 9, 2, 7),
        stamped(EventKind::Tick { frame: 4 }, 10, 2, 8),
        stamped(
            EventKind::Motion {
                entity: 4,
                delta: [5, 0, 0],
            },
            3,
            3,
            1,
        ),
        stamped(
            EventKind::Motion {
                entity: 4,
                delta: [5, 0, 0],
            },
            4,
            3,
            2,
        ),
        stamped(
            EventKind::Property {
                entity: 4,
                prop: 3,
                value: 50,
            },
            5,
            3,
            3,
        ),
        stamped(
            EventKind::Spawn {
                entity: 5,
                kind: 6,
                pos: [1, 1, 1],
            },
            6,
            3,
            4,
        ),
        stamped(
            EventKind::Spawn {
                entity: 2,
                kind: 4,
                pos: [0, 0, 0],
            },
            7,
            3,
            5,
        ),
        stamped(
            EventKind::Spawn {
                entity: 3,
                kind: 2,
                pos: [3, 3, 3],
            },
            10,
            3,
            6,
        ),
        stamped(EventKind::Tick { frame: 6 }, 11, 3, 7),
        stamped(EventKind::Input { player: 1, code: 9 }, 12, 3, 8),
        stamped(
            EventKind::Spawn {
                entity: 6,
                kind: 9,
                pos: [7, 7, 7],
            },
            1,
            1,
            1,
        ),
    ]
}

/// `SplitMix64`, so the shuffles are the same on every run and target
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[test]
fn the_scenario_holds_every_case_it_names() {
    let events = scenario();
    let mut keys: Vec<_> = events.iter().map(key).collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(
        keys.len(),
        events.len() - 1,
        "exactly one pair shares a key"
    );
    let same_time_and_seq = events.iter().any(|x| {
        events.iter().any(|y| {
            x.origin != y.origin
                && (x.timestamp, x.seq.0, &x.kind) == (y.timestamp, y.seq.0, &y.kind)
        })
    });
    assert!(
        same_time_and_seq,
        "two origins emit the same event at the same time and sequence"
    );

    let want = expected(&events);
    let ids: Vec<u32> = want.entities.iter().map(|e| e.0).collect();
    assert_eq!(
        ids,
        [1, 2, 3, 4, 6],
        "1 re-spawned, 2 and 4 alive, 3 kept by its later Spawn, 5 despawned, 6 from the collision"
    );
    let (_, kind1, pos1, _, _) = want.entities[0];
    assert_eq!(kind1, 8, "entity 1 is its second incarnation");
    let unit = alice_sync::fixed_point::Fixed::from_i16(1).0;
    assert_eq!(
        pos1,
        [unit, unit, unit],
        "only the motion after the re-spawn moved it"
    );
    let (_, _, pos2, props2, active2) = want.entities[1];
    assert_eq!(
        pos2,
        [0, 0, 9 * unit],
        "the motion keyed before entity 2's Spawn is not applied"
    );
    assert_eq!((props2[1], active2), (-7, 1 << 1));
    let (_, _, pos4, props4, _) = want.entities[3];
    assert_eq!(
        pos4,
        [20 * unit, 0, 0],
        "the four equal motions of two origins all apply"
    );
    assert_eq!(
        props4[3], 50,
        "same time: the greater origin wins, not the greater value"
    );
    let last_tick = events
        .iter()
        .filter_map(|e| match e.kind {
            EventKind::Tick { frame } => Some((key(e), frame)),
            _ => None,
        })
        .max()
        .unwrap();
    assert_eq!(want.frame, last_tick.1, "the Tick with the greatest key");
    let ticks: std::collections::BTreeSet<u64> = events
        .iter()
        .filter_map(|e| match e.kind {
            EventKind::Tick { frame } => Some(frame),
            _ => None,
        })
        .collect();
    assert_eq!(
        ticks.len(),
        3,
        "three different frames, so the order would show"
    );
    assert_eq!(want.hash, want.recomputed);
}

#[test]
fn every_arrival_order_of_a_conflicting_subset_gives_one_world() {
    // entity 4's Spawn, one equal Motion from each of two origins, the two writes of one property
    // at the same time, and entity 5's Spawn and its Despawn: 7! = 5040 orders
    let events = scenario();
    let entity_of = |e: &Event| match e.kind {
        EventKind::Spawn { entity, .. }
        | EventKind::Motion { entity, .. }
        | EventKind::Property { entity, .. }
        | EventKind::Despawn { entity } => Some(entity),
        _ => None,
    };
    let mut subset: Vec<Event> = Vec::new();
    let mut motions_by_origin = std::collections::BTreeSet::new();
    for e in &events {
        let take = match (entity_of(e), &e.kind) {
            (Some(4), EventKind::Motion { .. }) => motions_by_origin.insert(e.origin),
            (Some(4 | 5), _) => true,
            _ => false,
        };
        if take {
            subset.push(e.clone());
        }
    }
    assert_eq!(subset.len(), 7);
    // the write of origin 3 is replaced by a second write under origin 2's key (a collision): 7
    // events still, two of them sharing a key
    let b_write = subset
        .iter()
        .position(|e| {
            matches!(e.kind, EventKind::Property { .. }) && e.origin == Origin::of_node(2)
        })
        .unwrap();
    let c_write = subset
        .iter()
        .position(|e| {
            matches!(e.kind, EventKind::Property { .. }) && e.origin == Origin::of_node(3)
        })
        .unwrap();
    let mut twin = subset[b_write].clone();
    twin.kind = EventKind::Property {
        entity: 4,
        prop: 3,
        value: 7,
    };
    subset[c_write] = twin;
    let want = expected(&subset);
    assert_eq!(
        want.entities[0].3[3], 100,
        "the greater value under the shared key"
    );

    let mut order: Vec<usize> = (0..subset.len()).collect();
    let mut compared = 0;
    // Heap's algorithm
    let mut c = vec![0usize; order.len()];
    let check = |order: &[usize]| {
        let arrival: Vec<Event> = order.iter().map(|&i| subset[i].clone()).collect();
        assert_eq!(received(&arrival), want, "arrival order {order:?}");
    };
    check(&order);
    compared += 1;
    let mut i = 0;
    while i < order.len() {
        if c[i] < i {
            if i % 2 == 0 {
                order.swap(0, i);
            } else {
                order.swap(c[i], i);
            }
            check(&order);
            compared += 1;
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    assert_eq!(compared, 5040);
}

#[test]
fn seeded_shuffles_with_duplicates_of_the_whole_scenario_give_one_world() {
    let events = scenario();
    let want = expected(&events);
    let mut rng = Rng(0x0013_0013);
    for round in 0..2000 {
        let mut arrival = events.clone();
        // up to a third of the events delivered a second time
        for _ in 0..rng.below(events.len() / 3 + 1) {
            arrival.push(events[rng.below(events.len())].clone());
        }
        rng.shuffle(&mut arrival);
        assert_eq!(received(&arrival), want, "round {round}");
    }
}

#[test]
fn the_emitting_node_and_a_receiver_agree() {
    // the node that emitted the events in its own order and a receiver that took them reversed
    let mut a = Node::new(NodeId(1));
    a.emit(Event::new(EventKind::Spawn {
        entity: 5,
        kind: 1,
        pos: [1, 2, 3],
    }))
    .unwrap();
    for d in 0..10 {
        a.emit(Event::new(EventKind::Motion {
            entity: 5,
            delta: [d, -d, 1],
        }))
        .unwrap();
    }
    let reversed: Vec<Event> = a.events().all().iter().rev().cloned().collect();
    assert_eq!(received(&reversed), snapshot(a.world()));
}

const NOW: u64 = 1_800_000_100;

struct Trust(BTreeMap<String, PublicKey>);

impl TrustPolicy for Trust {
    fn master_of(&self, subject: &Did) -> Option<PublicKey> {
        self.0.get(&subject.to_string()).copied()
    }
    fn revocation(&self) -> Revocation<'_> {
        Revocation::NoListAvailable
    }
    fn now(&self) -> u64 {
        NOW
    }
}

/// A node with an author, and a trust store that accepts the authors of `seeds`
fn signed_node(seed: u8) -> Node {
    let (key, cert) = credential(seed);
    Node::with_author(NodeId(u64::from(seed)), Author::new(key, cert).unwrap())
}

fn credential(seed: u8) -> (KeyPair, DeviceCert) {
    let master = KeyPair::from_seed([seed; 32]);
    let key = KeyPair::from_seed([seed.wrapping_add(100); 32]);
    let cert = DeviceCert::issue(
        &master,
        &Did::parse(&format!("did:alice:n{seed}")).unwrap(),
        key.public().0,
        [0x4e; 32],
        1_800_000_000,
        1_800_086_400,
        [seed; 16],
    )
    .unwrap();
    (key, cert)
}

fn trusting(seeds: &[u8]) -> Trust {
    Trust(
        seeds
            .iter()
            .map(|&s| {
                (
                    format!("did:alice:n{s}"),
                    KeyPair::from_seed([s; 32]).public(),
                )
            })
            .collect(),
    )
}

fn everything(n: &mut Node) -> SignedEvents {
    n.signed_events_since(&BTreeMap::new()).unwrap()
}

#[test]
fn a_node_emits_after_what_it_received() {
    // Lamport: an event a node emits after receiving one is keyed after it
    let mut a = signed_node(1);
    for _ in 0..5 {
        a.emit(Event::new(EventKind::Tick { frame: 1 })).unwrap();
    }
    let mut b = signed_node(2);
    assert_eq!(b.receive(&everything(&mut a), &trusting(&[1])).unwrap(), 5);
    b.emit(Event::new(EventKind::Tick { frame: 2 })).unwrap();
    let theirs = key(a.events().all().last().unwrap());
    let mine = key(b.events().all().last().unwrap());
    assert!(mine.0 > theirs.0, "{mine:?} after {theirs:?}");
}

#[test]
fn a_write_made_after_seeing_another_wins_although_its_counter_is_lower() {
    // a writes property 0 after nine other events; b sees that write and overwrites it with its
    // first event: under per-node counters b's write (counter 0) would lose to a's (counter 9)
    let mut a = signed_node(1);
    a.emit(Event::new(EventKind::Spawn {
        entity: 1,
        kind: 0,
        pos: [0, 0, 0],
    }))
    .unwrap();
    for _ in 0..8 {
        a.emit(Event::new(EventKind::Tick { frame: 0 })).unwrap();
    }
    a.emit(Event::new(EventKind::Property {
        entity: 1,
        prop: 0,
        value: 10,
    }))
    .unwrap();
    let mut b = signed_node(2);
    b.receive(&everything(&mut a), &trusting(&[1])).unwrap();
    b.emit(Event::new(EventKind::Property {
        entity: 1,
        prop: 0,
        value: 20,
    }))
    .unwrap();
    assert_eq!(b.events().all()[0].seq.0, 1, "b's write is its first event");

    let mut all: Vec<Event> = a.events().all().to_vec();
    all.extend(b.events().all().iter().cloned());
    for arrival in [all.clone(), all.iter().rev().cloned().collect()] {
        let mut world = World::new(0);
        let mut converge = Converge::new();
        for e in &arrival {
            converge.apply(&mut world, e).unwrap();
        }
        assert_eq!(world.get_entity(1).unwrap().properties.get(0), Some(20));
    }
    // and through the signed path, b's batch (a's events and its own) in one piece
    let mut c = signed_node(3);
    c.receive(&everything(&mut b), &trusting(&[1, 2])).unwrap();
    assert_eq!(c.world().get_entity(1).unwrap().properties.get(0), Some(20));
}

/// a's Spawn of entity 1 followed by one more event of a, delivered in both orders
fn before_and_after_spawn(second: EventKind) -> [Snapshot; 2] {
    let mut a = Node::new(NodeId(1));
    a.emit(Event::new(EventKind::Spawn {
        entity: 1,
        kind: 0,
        pos: [0, 0, 0],
    }))
    .unwrap();
    a.emit(Event::new(second)).unwrap();
    let in_order: Vec<Event> = a.events().all().to_vec();
    let reversed: Vec<Event> = in_order.iter().rev().cloned().collect();
    [received(&in_order), received(&reversed)]
}

#[test]
fn a_property_that_arrives_before_its_spawn_is_kept() {
    let [in_order, reversed] = before_and_after_spawn(EventKind::Property {
        entity: 1,
        prop: 2,
        value: 77,
    });
    assert_eq!(reversed, in_order);
    assert_eq!(reversed.entities[0].3[2], 77);
}

#[test]
fn a_motion_that_arrives_before_its_spawn_is_kept() {
    let [in_order, reversed] = before_and_after_spawn(EventKind::Motion {
        entity: 1,
        delta: [4, 0, 0],
    });
    assert_eq!(reversed, in_order);
    let unit = alice_sync::fixed_point::Fixed::from_i16(1).0;
    assert_eq!(reversed.entities[0].2, [4 * unit, 0, 0]);
}

#[test]
fn an_unstamped_event_is_refused_not_taken_for_a_duplicate() {
    let mut a = signed_node(1);
    a.emit(Event::new(EventKind::Tick { frame: 1 })).unwrap();
    let mut batch = everything(&mut a);
    let spawn = Event::new(EventKind::Spawn {
        entity: 1,
        kind: 0,
        pos: [0, 0, 0],
    });
    assert!(!spawn.is_stamped());
    batch.events.push(spawn);
    let mut node = signed_node(2);
    assert!(matches!(
        node.receive(&batch, &trusting(&[1])),
        Err(AuthorshipError::Uncovered { .. })
    ));
    assert_eq!(node.world().entity_count(), 0);
}

/// A stamped event with the given key (the fields a stream would set)
fn stamped(kind: EventKind, timestamp: u64, origin: u64, seq: u64) -> Event {
    let mut e = Event::new(kind);
    e.timestamp = timestamp;
    e.origin = Origin::of_node(origin);
    e.seq = alice_sync::event::SeqNum(seq);
    e
}

#[test]
fn seeded_random_histories_match_the_key_order_model() {
    // 3000 rounds: 4 origins with up to 25 events each, Lamport times drawn from 0..12 so many
    // keys tie on time and are ordered by origin, 4 entities, every kind; 90% of the events
    // delivered with about a quarter of them twice, in two different shuffles
    let mut rng = Rng(0x7013_0070);
    let mut compared = 0;
    for round in 0..3000 {
        let mut events = Vec::new();
        for origin in 1..=4u64 {
            for seq in 1..=(rng.below(26) as u64) {
                let entity = 1 + rng.below(4) as u32;
                let kind = match rng.below(7) {
                    0 => EventKind::Spawn {
                        entity,
                        kind: rng.below(3) as u16,
                        pos: [rng.below(9) as i16 - 4, rng.below(9) as i16 - 4, 0],
                    },
                    1 => EventKind::Despawn { entity },
                    2 | 3 => EventKind::Motion {
                        entity,
                        delta: [rng.below(7) as i16 - 3, rng.below(7) as i16 - 3, 1],
                    },
                    4 => EventKind::Property {
                        entity,
                        prop: rng.below(3) as u16,
                        value: rng.below(100) as i32 - 50,
                    },
                    5 => EventKind::Tick {
                        frame: rng.below(50) as u64,
                    },
                    _ => EventKind::Input {
                        player: 1,
                        code: rng.below(10) as u32,
                    },
                };
                let e = stamped(kind, rng.below(12) as u64, origin, seq);
                // one in eight: another event under the same key (two nodes with one id)
                if rng.below(8) == 0 {
                    let mut twin = e.clone();
                    twin.kind = match rng.below(4) {
                        0 => EventKind::Spawn {
                            entity,
                            kind: rng.below(3) as u16,
                            pos: [rng.below(9) as i16 - 4, 0, 1],
                        },
                        1 => EventKind::Motion {
                            entity,
                            delta: [rng.below(7) as i16 - 3, 2, 0],
                        },
                        2 => EventKind::Property {
                            entity,
                            prop: rng.below(3) as u16,
                            value: rng.below(100) as i32 - 50,
                        },
                        _ => EventKind::Tick {
                            frame: rng.below(50) as u64,
                        },
                    };
                    events.push(twin);
                }
                events.push(e);
            }
        }
        let delivered: Vec<Event> = events.into_iter().filter(|_| rng.below(10) != 0).collect();
        let want = expected(&delivered);
        for _ in 0..2 {
            let mut arrival = delivered.clone();
            for e in &delivered {
                if rng.below(4) == 0 {
                    arrival.push(e.clone());
                }
            }
            rng.shuffle(&mut arrival);
            assert_eq!(received(&arrival), want, "round {round}");
            compared += 1;
        }
    }
    assert_eq!(compared, 6000);
}
