//! Oracle for signed events (authorship): an event counts only under its author's signed checkpoint.
//!
//! The author of an origin is the device whose certificate the origin is derived from: the first 16 bytes
//! of SHA-256(`alice-sync/origin/v1` || subject length (u16, big-endian) || subject || device key). Each
//! origin's events form a hash chain, and the author signs a checkpoint (origin, sequence number, chain
//! head) as an artifact signature for `alice-sync/checkpoint/v1`. A receiver applies an event only once a
//! verified checkpoint at or past it matches the chain it computed. The expected origin is computed here
//! from that rule with the sha2 crate's SHA-256, not taken from the code under test.

use std::collections::BTreeMap;

use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use sha2::{Digest, Sha256};
use alice_sync::authorship::{
    Author, AuthorshipError, Origin, SignedEvents, TrustPolicy, MAX_PENDING_PER_AUTHOR,
    ORIGIN_CONTEXT,
};
use alice_sync::{Event, EventKind, Node, NodeId};

const NOW: u64 = 1_800_000_100;

struct Party {
    master: KeyPair,
    key: KeyPair,
    cert: DeviceCert,
}

fn party(seed: u8, name: &str) -> Party {
    let master = KeyPair::from_seed([seed; 32]);
    let key = KeyPair::from_seed([seed.wrapping_add(100); 32]);
    let did = Did::parse(&format!("did:alice:{name}")).unwrap();
    let cert = DeviceCert::issue(
        &master,
        &did,
        key.public().0,
        [0x4e; 32],
        1_800_000_000,
        1_800_086_400,
        [seed; 16],
    )
    .unwrap();
    Party { master, key, cert }
}

struct Policy(BTreeMap<String, PublicKey>);

impl TrustPolicy for Policy {
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

fn trusting(ps: &[&Party]) -> Policy {
    Policy(
        ps.iter()
            .map(|p| (p.cert.subject().to_string(), p.master.public()))
            .collect(),
    )
}

fn node(id: u64, p: &Party) -> Node {
    Node::with_author(NodeId(id), Author::new(p.key.clone(), p.cert.clone()).unwrap())
}

fn spawn(entity: u32) -> Event {
    Event::new(EventKind::Spawn {
        entity,
        kind: 1,
        pos: [0, 0, 0],
    })
}

fn motion(entity: u32, d: i16) -> Event {
    Event::new(EventKind::Motion {
        entity,
        delta: [d, 0, 0],
    })
}

fn one(e: &Event, from: &SignedEvents) -> SignedEvents {
    SignedEvents {
        events: vec![e.clone()],
        checkpoints: from.checkpoints.clone(),
    }
}

#[test]
fn the_origin_is_derived_from_the_certificate() {
    let a = party(1, "alice");
    let subject = a.cert.subject().to_string();
    let mut m = ORIGIN_CONTEXT.to_vec();
    m.extend_from_slice(&u16::try_from(subject.len()).unwrap().to_be_bytes());
    m.extend_from_slice(subject.as_bytes());
    m.extend_from_slice(a.cert.device_public());
    let want: [u8; 16] = Sha256::digest(&m)[..16].try_into().unwrap();
    assert_eq!(ORIGIN_CONTEXT, b"alice-sync/origin/v1");
    assert_eq!(Origin::of(&a.cert), Origin(want));
    // another device key of the same subject is another origin
    let other = DeviceCert::issue(
        &a.master,
        a.cert.subject(),
        KeyPair::from_seed([77; 32]).public().0,
        [0x4e; 32],
        1_800_000_000,
        1_800_086_400,
        [9; 16],
    )
    .unwrap();
    assert_ne!(Origin::of(&other), Origin::of(&a.cert));
}

#[test]
fn signed_events_apply_at_the_receiver() {
    let a = party(1, "alice");
    let b = party(2, "bob");
    let mut na = node(1, &a);
    let mut nb = node(2, &b);
    na.emit(spawn(5)).unwrap();
    na.emit(motion(5, 3)).unwrap();
    let batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    assert_eq!(batch.checkpoints.len(), 1, "one checkpoint for the one origin");
    assert_eq!(nb.receive(&batch, &trusting(&[&a])).unwrap(), 2);
    assert_eq!(nb.world_hash(), na.world_hash());
    // the events carry the author's origin
    assert!(batch.events.iter().all(|e| e.origin == Origin::of(&a.cert)));
}

#[test]
fn events_without_a_checkpoint_are_refused() {
    let a = party(1, "alice");
    let b = party(2, "bob");
    let mut na = node(1, &a);
    let mut nb = node(2, &b);
    na.emit(spawn(5)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    batch.checkpoints.clear();
    let before = nb.world_hash();
    assert!(matches!(
        nb.receive(&batch, &trusting(&[&a])),
        Err(AuthorshipError::Uncovered { .. })
    ));
    assert_eq!(nb.world_hash(), before);
}

#[test]
fn a_changed_event_under_the_authors_checkpoint_is_refused() {
    let a = party(1, "alice");
    let b = party(2, "bob");
    let mut na = node(1, &a);
    let mut nb = node(2, &b);
    na.emit(spawn(5)).unwrap();
    na.emit(motion(5, 3)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    batch.events[1].kind = EventKind::Motion {
        entity: 5,
        delta: [100, 0, 0],
    };
    let before = nb.world_hash();
    assert!(matches!(
        nb.receive(&batch, &trusting(&[&a])),
        Err(AuthorshipError::ChainMismatch { .. })
    ));
    assert_eq!(nb.world_hash(), before);
}

#[test]
fn a_checkpoint_signed_by_another_device_for_the_origin_is_refused() {
    // mallory's signature, made with her own certificate, put on a checkpoint naming alice's origin
    let a = party(1, "alice");
    let m = party(3, "mallory");
    let b = party(2, "bob");
    let mut na = node(1, &a);
    let mut nm = node(3, &m);
    let mut nb = node(2, &b);
    na.emit(spawn(5)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    nm.emit(spawn(5)).unwrap();
    let theirs = nm.signed_events_since(&BTreeMap::new()).unwrap();
    batch.checkpoints[0].signature = theirs.checkpoints[0].signature.clone();
    assert!(matches!(
        nb.receive(&batch, &trusting(&[&a, &m])),
        Err(AuthorshipError::BadCheckpoint { .. })
    ));
}

#[test]
fn an_unknown_author_is_refused() {
    let a = party(1, "alice");
    let b = party(2, "bob");
    let mut na = node(1, &a);
    let mut nb = node(2, &b);
    na.emit(spawn(5)).unwrap();
    let batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    assert!(matches!(
        nb.receive(&batch, &trusting(&[])),
        Err(AuthorshipError::BadCheckpoint { .. })
    ));
}

#[test]
fn a_forked_chain_at_the_same_sequence_number_is_refused() {
    // alice's device signs two different histories up to seq 1: whichever arrives second is refused
    let a = party(1, "alice");
    let p = trusting(&[&a]);
    let mut one_way = node(1, &a);
    let mut other_way = node(1, &a);
    one_way.emit(spawn(5)).unwrap();
    other_way.emit(spawn(6)).unwrap();
    let first = one_way.signed_events_since(&BTreeMap::new()).unwrap();
    let second = other_way.signed_events_since(&BTreeMap::new()).unwrap();
    assert_eq!(first.checkpoints[0].seq, second.checkpoints[0].seq);
    assert_ne!(first.checkpoints[0].head, second.checkpoints[0].head);
    let mut nb = node(2, &party(2, "bob"));
    nb.receive(&first, &p).unwrap();
    assert!(matches!(
        nb.receive(&second, &p),
        Err(AuthorshipError::Forked { .. })
    ));
    let mut nc = node(3, &party(3, "carol"));
    nc.receive(&second, &p).unwrap();
    assert!(matches!(
        nc.receive(&first, &p),
        Err(AuthorshipError::Forked { .. })
    ));
}

#[test]
fn a_relay_that_splits_and_merges_batches_converges_bit_identically() {
    // alice and carol write; bob relays everything to dave in pieces of every size, last piece first; erin
    // gets everything in one batch. bob, dave and erin hold the same world (ALICE-Law G13).
    let a = party(1, "alice");
    let c = party(3, "carol");
    let pol = trusting(&[&a, &c]);
    let mut na = node(1, &a);
    let mut nc = node(3, &c);
    for i in 0..40u32 {
        na.emit(spawn(i)).unwrap();
        na.emit(motion(i, 1)).unwrap();
        nc.emit(motion(i, 2)).unwrap();
    }
    let from_a = na.signed_events_since(&BTreeMap::new()).unwrap();
    let from_c = nc.signed_events_since(&BTreeMap::new()).unwrap();
    let mut bob = node(2, &party(2, "bob"));
    bob.receive(&from_a, &pol).unwrap();
    bob.receive(&from_c, &pol).unwrap();
    let everything = bob.signed_events_since(&BTreeMap::new()).unwrap();
    assert_eq!(everything.events.len(), 120);
    assert_eq!(everything.checkpoints.len(), 2);
    let mut erin = node(5, &party(5, "erin"));
    erin.receive(&everything, &pol).unwrap();
    let mut dave = node(4, &party(4, "dave"));
    let mut pieces = Vec::new();
    let mut rest = everything.events.clone();
    let mut n = 1;
    while !rest.is_empty() {
        let take = n.min(rest.len());
        pieces.push(rest.drain(..take).collect::<Vec<_>>());
        n += 1;
    }
    pieces.reverse();
    for p in pieces {
        let piece = SignedEvents {
            events: p,
            checkpoints: everything.checkpoints.clone(),
        };
        dave.receive(&piece, &pol).unwrap();
    }
    assert_eq!(bob.world_hash(), erin.world_hash());
    assert_eq!(dave.world_hash(), erin.world_hash());
    // the relayed world is not merely empty: alice's spawns are in it
    assert_ne!(erin.world_hash(), node(9, &party(9, "nobody")).world_hash());
}

#[test]
fn pending_events_of_one_author_are_capped_and_the_others_go_on() {
    // alice's events arrive without her first one: they wait, at most MAX_PENDING_PER_AUTHOR of them;
    // past that her further events are refused with an explicit error; carol's still apply
    let a = party(1, "alice");
    let c = party(3, "carol");
    let pol = trusting(&[&a, &c]);
    let mut na = node(1, &a);
    for i in 0..(MAX_PENDING_PER_AUTHOR as u32 + 3) {
        na.emit(motion(i, 1)).unwrap();
    }
    let all = na.signed_events_since(&BTreeMap::new()).unwrap();
    let mut nb = node(2, &party(2, "bob"));
    let held: Vec<_> = all.events[1..]
        .iter()
        .map(|e| nb.receive(&one(e, &all), &pol))
        .collect();
    let refused = held
        .iter()
        .filter(|r| matches!(r, Err(AuthorshipError::PendingFull { .. })))
        .count();
    assert_eq!(refused, all.events.len() - 1 - MAX_PENDING_PER_AUTHOR);
    assert!(held
        .iter()
        .all(|r| matches!(r, Ok(0) | Err(AuthorshipError::PendingFull { .. }))));
    let mut nc = node(3, &c);
    nc.emit(spawn(1)).unwrap();
    let from_c = nc.signed_events_since(&BTreeMap::new()).unwrap();
    assert_eq!(nb.receive(&from_c, &pol).unwrap(), 1);
}

#[test]
fn which_pending_events_are_kept_does_not_depend_on_the_arrival_order() {
    // the smallest sequence numbers are kept: two arrival orders of the same events leave the same held
    // set, so once the gap fills both receivers apply the same events
    let a = party(1, "alice");
    let pol = trusting(&[&a]);
    let mut na = node(1, &a);
    for i in 0..(MAX_PENDING_PER_AUTHOR as u32 + 5) {
        na.emit(motion(i, 1)).unwrap();
    }
    let all = na.signed_events_since(&BTreeMap::new()).unwrap();
    let tail: Vec<Event> = all.events[1..].to_vec();
    let mut forward = node(2, &party(2, "bob"));
    let mut backward = node(3, &party(3, "carol"));
    for e in &tail {
        let _ = forward.receive(&one(e, &all), &pol);
    }
    for e in tail.iter().rev() {
        let _ = backward.receive(&one(e, &all), &pol);
    }
    let fa = forward.receive(&one(&all.events[0], &all), &pol).unwrap();
    let ba = backward.receive(&one(&all.events[0], &all), &pol).unwrap();
    assert_eq!(fa, ba);
    assert_eq!(fa, 1 + MAX_PENDING_PER_AUTHOR);
    assert_eq!(forward.world_hash(), backward.world_hash());
}
