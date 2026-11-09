//! Contract tests for the room relay core (sans-IO: frames in, frames out).
//!
//! The relay never opens payloads: a published payload is forwarded byte for byte. Per publisher,
//! payloads leave in `seq` order; a gap is held back and the publisher is asked to resend the
//! missing range; a duplicate is dropped. Only room members may publish. The relay keeps a log of
//! forwarded payloads in arrival order, capped by count, for late joiners and reconnects.

use alice_sync::relay::{Limits, OpenError, Out, PeerId, Refusal, RelayCore, RoomId};

const A: PeerId = PeerId::from_raw([1; 32]);
const B: PeerId = PeerId::from_raw([2; 32]);
const C: PeerId = PeerId::from_raw([3; 32]);
const X: PeerId = PeerId::from_raw([9; 32]);
const R: RoomId = RoomId::from_raw(7);

fn relay() -> RelayCore {
    let mut r = RelayCore::new();
    r.open_room(
        R,
        &[A, B, C],
        Limits {
            max_payload: 64,
            max_log: 8,
            max_pending: 4,
        },
    )
    .unwrap();
    r
}

fn forwards(out: &[Out]) -> Vec<(PeerId, PeerId, u64, Vec<u8>)> {
    out.iter()
        .filter_map(|o| match o {
            Out::Forward {
                to,
                from,
                seq,
                payload,
                ..
            } => Some((*to, *from, *seq, payload.clone())),
            _ => None,
        })
        .collect()
}

/// oracle: an in-order payload goes to every other member (in member order), unchanged.
#[test]
fn publish_fans_out_to_other_members_unchanged() {
    let mut r = relay();
    let out = r.publish(A, R, 0, b"\x00sealed".to_vec());
    assert_eq!(
        forwards(&out),
        vec![
            (B, A, 0, b"\x00sealed".to_vec()),
            (C, A, 0, b"\x00sealed".to_vec())
        ]
    );
}

/// oracle: a gap is held back and a resend of the missing range is requested from the publisher;
/// when the gap is filled, everything held is forwarded in seq order.
#[test]
fn gaps_are_held_and_requested_then_released_in_order() {
    let mut r = relay();
    let _ = r.publish(A, R, 0, vec![0]);
    let out = r.publish(A, R, 3, vec![3]);
    assert!(forwards(&out).is_empty(), "seq 3 waits for 1 and 2");
    assert!(out.contains(&Out::Resend {
        to: A,
        room: R,
        from: 1,
        to_seq: 2
    }));
    let out = r.publish(A, R, 2, vec![2]);
    assert!(forwards(&out).is_empty());
    let out = r.publish(A, R, 1, vec![1]);
    let seqs: Vec<u64> = forwards(&out)
        .iter()
        .filter(|f| f.0 == B)
        .map(|f| f.2)
        .collect();
    assert_eq!(seqs, vec![1, 2, 3]);
}

/// oracle: duplicates (already forwarded or already held) are dropped without output.
#[test]
fn duplicates_are_dropped() {
    let mut r = relay();
    let _ = r.publish(A, R, 0, vec![0]);
    assert!(r.publish(A, R, 0, vec![0]).is_empty());
    let _ = r.publish(A, R, 2, vec![2]);
    assert!(forwards(&r.publish(A, R, 2, vec![2])).is_empty());
}

/// oracle: publishers are independent streams (B's seq 0 is not A's seq 0).
#[test]
fn each_publisher_has_its_own_sequence() {
    let mut r = relay();
    let _ = r.publish(A, R, 0, vec![0xa]);
    let out = r.publish(B, R, 0, vec![0xb]);
    assert_eq!(
        forwards(&out),
        vec![(A, B, 0, vec![0xb]), (C, B, 0, vec![0xb])]
    );
}

/// oracle: non-members and unknown rooms are refused; oversize payloads are refused and nothing
/// is forwarded.
#[test]
fn refusals() {
    let mut r = relay();
    assert_eq!(
        r.publish(X, R, 0, vec![1]),
        vec![Out::Refused {
            to: X,
            room: R,
            reason: Refusal::NotMember
        }]
    );
    let other = RoomId::from_raw(8);
    assert_eq!(
        r.publish(A, other, 0, vec![1]),
        vec![Out::Refused {
            to: A,
            room: other,
            reason: Refusal::NoRoom
        }]
    );
    assert_eq!(
        r.publish(A, R, 0, vec![0; 65]),
        vec![Out::Refused {
            to: A,
            room: R,
            reason: Refusal::TooLarge
        }]
    );
    // the refused oversize seq 0 was not consumed
    assert_eq!(forwards(&r.publish(A, R, 0, vec![1])).len(), 2);
}

/// oracle: the hold-back buffer is capped; a publisher that runs too far ahead is refused.
#[test]
fn pending_is_capped() {
    let mut r = relay();
    for s in 1..=4 {
        let out = r.publish(A, R, s, vec![s as u8]);
        assert!(
            !out.iter().any(|o| matches!(o, Out::Refused { .. })),
            "seq {s}"
        );
    }
    let out = r.publish(A, R, 5, vec![5]);
    assert!(out.contains(&Out::Refused {
        to: A,
        room: R,
        reason: Refusal::TooFarAhead
    }));
}

/// oracle: the log keeps forwarded payloads in arrival order with a running index; a range read
/// returns them; beyond the cap the oldest are dropped and a read from before the oldest starts at
/// the oldest kept.
#[test]
fn log_range_reads_and_cap() {
    let mut r = relay();
    for s in 0..10u64 {
        let _ = r.publish(A, R, s, vec![s as u8]);
    }
    // 10 forwarded, cap 8: log indices 2..10 kept
    let log = r.log_range(C, R, 0, 100).unwrap();
    assert_eq!(log.first_index, 2);
    let payloads: Vec<u8> = log.items.iter().map(|i| i.payload[0]).collect();
    assert_eq!(payloads, (2..10).collect::<Vec<u8>>());
    let part = r.log_range(C, R, 4, 6).unwrap();
    assert_eq!(
        part.items
            .iter()
            .map(|i| (i.from, i.seq))
            .collect::<Vec<_>>(),
        vec![(A, 4), (A, 5)]
    );
    assert_eq!(r.log_range(X, R, 0, 1), Err(Refusal::NotMember));
}

/// oracle: forwarded bytes never exceed received bytes times (members - 1) (no amplification).
#[test]
fn no_amplification() {
    let mut r = relay();
    let mut sent = 0usize;
    let mut received = 0usize;
    for s in 0..6u64 {
        let p = vec![0u8; 10 + s as usize];
        received += p.len();
        for o in r.publish(A, R, s, p) {
            if let Out::Forward { payload, .. } = o {
                sent += payload.len();
            }
        }
    }
    assert_eq!(sent, received * 2);
}

/// oracle (degenerate input): a room needs at least two members and no duplicate members; an
/// empty range read returns nothing.
#[test]
fn degenerate_rooms_and_ranges() {
    let mut r = RelayCore::new();
    let l = Limits {
        max_payload: 64,
        max_log: 8,
        max_pending: 4,
    };
    assert!(r.open_room(R, &[A], l).is_err());
    assert!(r.open_room(R, &[A, A], l).is_err());
    assert_eq!(
        r.open_room(R, &[A, A, B], l),
        Err(OpenError::DuplicateMember)
    );
    assert!(r.open_room(R, &[A, B], l).is_ok());
    assert!(r.open_room(R, &[A, B], l).is_err(), "room id already open");
    assert!(r.log_range(A, R, 5, 5).unwrap().items.is_empty());
}
