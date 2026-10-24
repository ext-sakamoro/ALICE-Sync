//! Oracle for signed events (authorship): an event counts only under its author's signed checkpoint.
//!
//! The author of an origin is the device whose certificate the origin is derived from: the first 16 bytes
//! of SHA-256(`alice-sync/origin/v1` || subject length (u16, big-endian) || subject || device key). Each
//! origin's events form a hash chain, and the author signs a checkpoint (origin, sequence number, chain
//! head) as an artifact signature for `alice-sync/checkpoint/v1`. A receiver applies an event only once a
//! verified checkpoint at or past it matches the chain it computed. The expected origin is computed here
//! from that rule with the sha2 crate's SHA-256, not taken from the code under test.

use std::collections::BTreeMap;

use alice_identity::artifact::{sign_artifact, PayloadDigest, Revocation};
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use alice_sync::authorship::{
    Author, AuthorshipError, Origin, SignedEvents, TrustPolicy, Verifier, CHECKPOINT_CONTEXT,
    CHECKPOINT_INTERVAL, MAX_PENDING_PER_AUTHOR, ORIGIN_CONTEXT,
};
use alice_sync::{Event, EventKind, Node, NodeId};
use sha2::{Digest, Sha256};

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
    Node::with_author(
        NodeId(id),
        Author::new(p.key.clone(), p.cert.clone()).unwrap(),
    )
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
    assert_eq!(
        batch.checkpoints.len(),
        1,
        "one checkpoint for the one origin"
    );
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
    // alice's 80 events: checkpoints at every CHECKPOINT_INTERVAL (64) and at her last (80); carol's
    // 40: at her last
    let seqs: Vec<u64> = everything.checkpoints.iter().map(|c| c.seq).collect();
    assert_eq!(CHECKPOINT_INTERVAL, 64);
    assert_eq!(seqs.len(), 3, "{seqs:?}");
    assert!(seqs.contains(&64) && seqs.contains(&80) && seqs.contains(&40));
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
    // set, so once the gap fills both receivers apply the same events (the author signs a checkpoint
    // every CHECKPOINT_INTERVAL events, so a run of held events can apply before the last checkpoint)
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
    // both now hold 1..=MAX_PENDING_PER_AUTHOR + 1 contiguously and apply up to the last checkpoint
    // inside that run
    let last_inside =
        (MAX_PENDING_PER_AUTHOR as u64 + 1) / CHECKPOINT_INTERVAL * CHECKPOINT_INTERVAL;
    assert_eq!(fa, ba);
    assert_eq!(fa as u64, last_inside);
    assert_eq!(forward.world_hash(), backward.world_hash());
}

/// The digest a checkpoint's signature covers: SHA-256(origin || sequence number (u64, big-endian) || head)
fn checkpoint_digest(origin: Origin, seq: u64, head: &[u8; 32]) -> PayloadDigest {
    let mut m = origin.0.to_vec();
    m.extend_from_slice(&seq.to_be_bytes());
    m.extend_from_slice(head);
    PayloadDigest::sha256(Sha256::digest(&m).into())
}

#[test]
fn the_checkpoint_signature_covers_the_documented_digest() {
    // a checkpoint made here from the rule, with alice's key, is accepted: the rule is the one verified
    let a = party(1, "alice");
    let mut na = node(1, &a);
    na.emit(spawn(5)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    let cp = &mut batch.checkpoints[0];
    assert_eq!(CHECKPOINT_CONTEXT, b"alice-sync/checkpoint/v1");
    cp.signature = sign_artifact(
        &a.key,
        &a.cert,
        CHECKPOINT_CONTEXT,
        &checkpoint_digest(cp.origin, cp.seq, &cp.head),
    )
    .unwrap()
    .to_bytes();
    let mut nb = node(2, &party(2, "bob"));
    assert_eq!(nb.receive(&batch, &trusting(&[&a])).unwrap(), 1);
}

#[test]
fn a_checkpoint_whose_head_was_changed_after_signing_is_refused() {
    // alice's own signature and certificate, so the origin matches: only the signature check refuses it
    let a = party(1, "alice");
    let mut na = node(1, &a);
    na.emit(spawn(5)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    batch.checkpoints[0].head[0] ^= 1;
    let mut nb = node(2, &party(2, "bob"));
    assert!(matches!(
        nb.receive(&batch, &trusting(&[&a])),
        Err(AuthorshipError::BadCheckpoint { .. })
    ));
    assert_eq!(nb.world().entity_count(), 0);
}

#[test]
fn a_valid_signature_by_another_device_on_the_origin_is_refused() {
    // mallory signs, correctly, the digest of a checkpoint naming alice's origin: the signature verifies
    // under mallory's trusted master, so only the origin-of-the-signer check refuses it
    let a = party(1, "alice");
    let m = party(3, "mallory");
    let mut na = node(1, &a);
    na.emit(spawn(5)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    let cp = &mut batch.checkpoints[0];
    cp.signature = sign_artifact(
        &m.key,
        &m.cert,
        CHECKPOINT_CONTEXT,
        &checkpoint_digest(cp.origin, cp.seq, &cp.head),
    )
    .unwrap()
    .to_bytes();
    let mut nb = node(2, &party(2, "bob"));
    assert!(matches!(
        nb.receive(&batch, &trusting(&[&a, &m])),
        Err(AuthorshipError::BadCheckpoint { .. })
    ));
    assert_eq!(nb.world().entity_count(), 0);
}

#[test]
fn an_unstamped_event_under_the_authors_origin_is_refused() {
    // sequence number 0 with alice's origin: a checkpoint at 1 would cover it if only the number counted
    let a = party(1, "alice");
    let mut na = node(1, &a);
    na.emit(spawn(5)).unwrap();
    let mut batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    let mut extra = spawn(6);
    extra.origin = Origin::of(&a.cert);
    assert!(!extra.is_stamped());
    batch.events.push(extra);
    let mut nb = node(2, &party(2, "bob"));
    assert!(matches!(
        nb.receive(&batch, &trusting(&[&a])),
        Err(AuthorshipError::Uncovered { seq: 0, .. })
    ));
    assert_eq!(nb.world().entity_count(), 0);
}

#[test]
fn an_apply_that_fails_partway_keeps_the_events_before_it_applied_once() {
    // five covered events; applying the third fails: the first two stay applied and are not applied
    // again when the same batch comes back, the third and the rest apply then
    let a = party(1, "alice");
    let pol = trusting(&[&a]);
    let mut na = node(1, &a);
    for i in 0..5 {
        na.emit(motion(i, 1)).unwrap();
    }
    let batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    assert_eq!(batch.events.len(), 5);
    let mut v = Verifier::new();
    let mut seen = Vec::new();
    let r = v.receive(&batch, &pol, |e| {
        if e.seq.0 == 3 {
            return Err("refused by the world".to_owned());
        }
        seen.push(e.seq.0);
        Ok(())
    });
    assert!(matches!(r, Err(AuthorshipError::Apply(_))), "{r:?}");
    assert_eq!(seen, vec![1, 2]);
    // no held checkpoint covers the two applied events yet, so they are not passed on
    assert!(v.since(&BTreeMap::new()).events.is_empty());
    let mut again = Vec::new();
    let n = v
        .receive(&batch, &pol, |e| {
            again.push(e.seq.0);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        again,
        vec![3, 4, 5],
        "events 1 and 2 are not applied a second time"
    );
    assert_eq!(n, 3);
    assert_eq!(v.since(&BTreeMap::new()).events.len(), 5);
}

#[test]
fn since_past_what_was_applied_gives_nothing() {
    // a peer that names a later sequence number than this receiver applied gets nothing (and no panic)
    let a = party(1, "alice");
    let pol = trusting(&[&a]);
    let mut na = node(1, &a);
    for i in 0..3 {
        na.emit(motion(i, 1)).unwrap();
    }
    let batch = na.signed_events_since(&BTreeMap::new()).unwrap();
    let mut v = Verifier::new();
    assert_eq!(v.receive(&batch, &pol, |_| Ok(())).unwrap(), 3);
    let origin = Origin::of(&a.cert);
    for from in [3, 4, 10, u64::MAX] {
        let out = v.since(&BTreeMap::from([(origin, from)]));
        assert!(
            out.events.is_empty() && out.checkpoints.is_empty(),
            "from {from}"
        );
    }
    let out = v.since(&BTreeMap::from([(origin, 1)]));
    assert_eq!(out.events.len(), 2);
    assert_eq!(out.checkpoints.len(), 1);
}

#[test]
fn since_returns_batches_a_fresh_receiver_accepts_for_any_from_and_state() {
    // `since` takes its sequence numbers from a peer: for every state of the receiver (nothing held,
    // checkpoints only, partly applied, all applied) and every `from` around its bounds it returns, and
    // the batch it returns is accepted by a receiver that has nothing yet
    let a = party(1, "alice");
    let pol = trusting(&[&a]);
    let origin = Origin::of(&a.cert);
    let mut na = node(1, &a);
    let n = 2 * CHECKPOINT_INTERVAL + 3;
    for i in 0..n as u32 {
        na.emit(motion(i, 1)).unwrap();
    }
    let all = na.signed_events_since(&BTreeMap::new()).unwrap();
    assert!(all.checkpoints.len() >= 3);
    let last_cp = all.checkpoints.iter().map(|c| c.seq).max().unwrap();

    let nothing = Verifier::new();
    let mut checkpoints_only = Verifier::new();
    let gap = SignedEvents {
        events: all.events[1..].to_vec(),
        checkpoints: all.checkpoints.clone(),
    };
    assert_eq!(checkpoints_only.receive(&gap, &pol, |_| Ok(())).unwrap(), 0);
    let mut partly = Verifier::new();
    let stop = CHECKPOINT_INTERVAL + 10;
    assert!(partly
        .receive(&all, &pol, |e| if e.seq.0 == stop {
            Err("refused".to_owned())
        } else {
            Ok(())
        })
        .is_err());
    let mut whole = Verifier::new();
    assert_eq!(
        whole.receive(&all, &pol, |_| Ok(())).unwrap(),
        all.events.len()
    );

    for (name, v, applied) in [
        ("nothing held", &nothing, 0),
        ("checkpoints only", &checkpoints_only, 0),
        ("partly applied", &partly, stop - 1),
        ("all applied", &whole, all.events.len() as u64),
    ] {
        for from in [0, 1, applied, applied + 1, last_cp, u64::MAX] {
            let out = v.since(&BTreeMap::from([(origin, from)]));
            assert!(
                out.events
                    .iter()
                    .all(|e| e.seq.0 > from && e.seq.0 <= applied),
                "{name} {from}"
            );
            if let Some(last) = out.events.iter().map(|e| e.seq.0).max() {
                assert!(
                    out.checkpoints.iter().any(|c| c.seq == last),
                    "{name} {from}: ends at a checkpoint"
                );
            }
            let mut fresh = Verifier::new();
            let r = fresh.receive(&out, &pol, |_| Ok(()));
            assert!(r.is_ok(), "{name} {from}: {r:?}");
        }
    }
}

#[test]
fn a_node_asked_past_its_last_event_gives_nothing_of_its_own() {
    // the peer names the sequence number it has; the largest one must not wrap around to "everything"
    let a = party(1, "alice");
    let mut na = node(1, &a);
    na.emit(motion(1, 1)).unwrap();
    na.emit(motion(1, 2)).unwrap();
    let origin = Origin::of(&a.cert);
    for from in [2, 3, u64::MAX] {
        let out = na
            .signed_events_since(&BTreeMap::from([(origin, from)]))
            .unwrap();
        assert!(out.checkpoints.is_empty(), "from {from}");
        assert!(out.events.iter().all(|e| e.seq.0 >= from), "from {from}");
    }
}
