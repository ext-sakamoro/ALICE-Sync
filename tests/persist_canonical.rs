//! Oracle for the persisted shape of a world (ALICE-Law G8 with G13): the stamped bytes are a function of the
//! logical world (the entities by id, the frame and the seed), not of the order in which slots were
//! allocated and freed. Two nodes that received the same events in different orders hold the same world
//! (tests/g13_receive_order.rs) and must therefore write the same bytes.

use alice_sync::persist::encode_state;
use alice_sync::{Event, EventKind, Node, NodeId};

fn spawn(entity: u32, kind: u16, pos: [i16; 3]) -> Event {
    Event::new(EventKind::Spawn { entity, kind, pos })
}

/// Events emitted by one node, as a peer would receive them
fn emitted(kinds: Vec<Event>) -> Vec<Event> {
    let mut a = Node::new(NodeId(1));
    for e in kinds {
        a.emit(e).unwrap();
    }
    a.events().all().to_vec()
}

fn persisted(events: &[Event]) -> Vec<u8> {
    let mut n = Node::new(NodeId(9));
    for e in events {
        n.apply_event(e).unwrap();
    }
    encode_state(n.world().state())
}

#[test]
fn two_spawns_received_in_either_order_persist_the_same_bytes() {
    let events = emitted(vec![spawn(1, 7, [10, 0, 0]), spawn(2, 3, [0, 5, 0])]);
    let forward = persisted(&events);
    let mut reversed = events.clone();
    reversed.reverse();
    assert_eq!(persisted(&reversed), forward);
}

#[test]
fn a_despawn_and_respawn_persist_the_same_bytes_as_a_world_that_only_spawned() {
    // the same logical world: entity 2 alive with the second spawn's kind and position; the first
    // history also allocated, freed and reused a slot
    let churned = emitted(vec![
        spawn(1, 7, [10, 0, 0]),
        spawn(2, 3, [0, 5, 0]),
        Event::new(EventKind::Despawn { entity: 1 }),
        spawn(1, 7, [10, 0, 0]),
    ]);
    let mut b = Node::new(NodeId(2));
    for e in &churned {
        b.apply_event(e).unwrap();
    }
    let direct = emitted(vec![spawn(2, 3, [0, 5, 0]), spawn(1, 7, [10, 0, 0])]);
    let mut c = Node::new(NodeId(3));
    for e in &direct {
        c.apply_event(e).unwrap();
    }
    // the worlds agree logically (same hash) ...
    assert_eq!(b.world_hash(), c.world_hash());
    // ... so their persisted bytes must too
    assert_eq!(
        encode_state(b.world().state()),
        encode_state(c.world().state())
    );
}
