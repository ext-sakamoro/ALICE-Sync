//! An entity id chosen by a remote peer must not decide how much memory a node reserves: one Spawn of
//! entity `u32::MAX` from another node is one live entity, and the memory it takes is that of one
//! entity, not of a table indexed up to its id.
//! Measured with a capped global allocator: the largest single request may not exceed 16 MiB (a request
//! above it is refused, so a node that sizes a table by the id stops the process with "memory allocation
//! of N bytes failed" before anything that large is reserved), and the peak of live bytes while the
//! event is applied stays within a small fixed slack.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use alice_sync::event::{Event, EventId, EventKind, SeqNum};
use alice_sync::persist::{decode_state, encode_state, PersistError};
use alice_sync::{Node, NodeId, WorldState};

/// Largest single request the code under test may make in this binary
const CAP: usize = 16 * 1024 * 1024;

struct Capped;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn admit(size: usize) -> bool {
    LARGEST.fetch_max(size, Ordering::SeqCst);
    size <= CAP
}

fn note(size: usize) {
    let now = LIVE.fetch_add(size, Ordering::SeqCst) + size;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

/// The tests of this binary measure one process-wide counter, so they run one at a time
static SERIAL: Mutex<()> = Mutex::new(());

/// Peak bytes allocated above the starting level while `f` runs
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    LARGEST.store(0, Ordering::SeqCst);
    let out = f();
    let largest = LARGEST.load(Ordering::SeqCst);
    assert!(largest <= CAP, "a single request of {largest} bytes");
    (out, PEAK.load(Ordering::SeqCst) - base)
}

/// What applying one event may take apart from the entity itself (the converge state, the event's
/// copy, the world's own small tables)
const SLACK: usize = 64 * 1024;

/// An event as another node stamped it (origin 1, sequence 1), without that node applying it
fn remote_spawn(entity: u32) -> Event {
    Event {
        id: EventId(1),
        seq: SeqNum(1),
        origin: 1,
        timestamp: 1,
        kind: EventKind::Spawn {
            entity,
            kind: 1,
            pos: [0, 0, 0],
        },
    }
}

#[test]
fn a_remote_spawn_of_the_largest_entity_id_costs_one_entity() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut node = Node::new(NodeId(9));
    let event = remote_spawn(u32::MAX);
    let (applied, peak) = peak_during(|| node.apply_event(&event));
    applied.unwrap();
    assert!(peak <= SLACK, "applying one spawn took {peak} bytes");
    assert_eq!(node.world().iter_entities().count(), 1);
}

#[test]
fn spawns_spread_over_the_id_range_cost_their_count_not_their_ids() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut node = Node::new(NodeId(9));
    let ids = [0u32, 1 << 20, 1 << 28, u32::MAX - 1];
    let (_, peak) = peak_during(|| {
        for (i, id) in ids.iter().enumerate() {
            let mut e = remote_spawn(*id);
            e.seq = SeqNum(i as u64 + 1);
            e.id = EventId(i as u64 + 1);
            e.timestamp = i as u64 + 1;
            node.apply_event(&e).unwrap();
        }
    });
    assert!(peak <= 4 * SLACK, "four spawns took {peak} bytes");
    assert_eq!(node.world().iter_entities().count(), ids.len());
}

#[test]
fn the_column_world_applies_a_spawn_of_the_largest_entity_id_as_one_entity() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut world = alice_sync::WorldSoA::new(0);
    let event = remote_spawn(u32::MAX);
    let (applied, peak) = peak_during(|| world.apply(&event));
    applied.unwrap();
    assert!(peak <= SLACK, "applying one spawn took {peak} bytes");
}

/// A stored world names the same ids: decoding a small file whose one entity has the largest id, or
/// whose count claims the largest number of entities, reserves memory by the bytes it holds
#[test]
fn a_small_stored_world_naming_the_largest_id_decodes_within_its_own_size() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut node = Node::new(NodeId(9));
    node.apply_event(&remote_spawn(u32::MAX)).unwrap();
    let bytes = encode_state(node.world().state());
    // header 39, count 4, one entity body 84, frame and seed 16
    assert_eq!(bytes.len(), 39 + 4 + 84 + 16);
    let (decoded, peak) = peak_during(|| decode_state::<WorldState>(&bytes));
    let w = decoded.unwrap().value;
    assert!(w.get_handle(u32::MAX).is_some());
    assert!(
        peak <= SLACK,
        "decoding {} bytes took {peak} bytes",
        bytes.len()
    );

    let mut claims_more = bytes.clone();
    claims_more[39..43].copy_from_slice(&u32::MAX.to_le_bytes());
    let (decoded, peak) = peak_during(|| decode_state::<WorldState>(&claims_more));
    assert!(matches!(decoded, Err(PersistError::Malformed(_))));
    assert!(peak <= SLACK, "a count of u32::MAX took {peak} bytes");
    // a count at the limit, with one body after it: truncated, reserving by the bytes held
    let limit = u32::try_from(alice_sync::world::MAX_LIVE_ENTITIES).unwrap();
    claims_more[39..43].copy_from_slice(&limit.to_le_bytes());
    let (decoded, peak) = peak_during(|| decode_state::<WorldState>(&claims_more));
    assert!(matches!(decoded, Err(PersistError::Truncated)));
    assert!(peak <= SLACK, "a count of {limit} took {peak} bytes");
}
