//! Contract tests for what a host that may evict and restart the process holding the anonymous relay
//! core needs around [`AnonymousRelayCore::snapshot`] / [`AnonymousRelayCore::restore`]:
//!
//! - bound connections after a restore: [`AnonymousRelayCore::bound_conns`] lists every bound connection
//!   once, in ascending order, and [`AnonymousRelayCore::close_all_except`] closes exactly the ones the
//!   host no longer holds, leaving the state calling `close` for each of them leaves;
//! - limits that change across a restore: [`AnonymousRelayCore::restore_with_limits`] checks the state
//!   against the new limits (a state over any of them is refused with the same typed error `restore`
//!   gives), and the core then behaves under the new limits;
//! - bounded log replies: [`AnonymousRelayCore::log_range_within`] returns the longest prefix whose `Log`
//!   frame, as the wire encoder writes it, fits the byte budget; asking again from the last index + 1
//!   covers the range exactly once, an empty reply means done, and a payload that cannot fit on its own
//!   is refused rather than returned empty forever;
//! - every [`Out`] maps to the connection and the wire frame it means, and survives encode / decode;
//! - [`AnonymousRelayCore::leave`] unbinds one room only (the wire's `Close` names a room).

use std::collections::BTreeSet;

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, LogItem, LogRange, Out, Refusal, RoomAlias, RoomToken,
    SenderTag, SnapshotError,
};
use alice_sync::anonymous_relay_wire::Frame;

const ROOM_1: RoomAlias = RoomAlias::from_raw([0x11; 32]);
const ROOM_2: RoomAlias = RoomAlias::from_raw([0x22; 32]);
const TOKEN_1: RoomToken = RoomToken::from_raw([0xa1; 32]);
const TOKEN_2: RoomToken = RoomToken::from_raw([0xa2; 32]);
const A: ConnId = ConnId::from_raw(1);
const B: ConnId = ConnId::from_raw(2);
const C: ConnId = ConnId::from_raw(3);
const D: ConnId = ConnId::from_raw(4);
const E: ConnId = ConnId::from_raw(5);
const TAG_A: SenderTag = SenderTag::from_raw([0x0a; 16]);
const TAG_B: SenderTag = SenderTag::from_raw([0x0b; 16]);
const TAG_C: SenderTag = SenderTag::from_raw([0x0c; 16]);
const TAG_D: SenderTag = SenderTag::from_raw([0x0d; 16]);
const TAG_E: SenderTag = SenderTag::from_raw([0x0e; 16]);

const LIMITS: Limits = Limits {
    max_payload: 16,
    max_log: 4,
    max_pending: 3,
    max_rooms: 2,
    max_members: 3,
    max_publishes_per_window: 5,
    window: 100,
    max_rooms_per_conn: 2,
    max_tags_per_room: 3,
};

fn snap(core: &AnonymousRelayCore) -> Vec<u8> {
    core.snapshot().expect("representable")
}

fn refused(out: &[Out]) -> Option<Refusal> {
    match out {
        [Out::Refused { reason, .. }] => Some(*reason),
        _ => None,
    }
}

fn forwarded_to(out: &[Out]) -> Vec<ConnId> {
    out.iter()
        .filter_map(|o| match o {
            Out::Forward { to, .. } => Some(*to),
            _ => None,
        })
        .collect()
}

/// Two rooms; A in both, B and C in room 1, D in room 2, E bound and then closed.
fn two_rooms() -> AnonymousRelayCore {
    let mut core = AnonymousRelayCore::new(LIMITS);
    core.bind(C, ROOM_1, TOKEN_1, TAG_C, 0);
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0);
    core.bind(B, ROOM_1, TOKEN_1, TAG_B, 0);
    core.bind(D, ROOM_2, TOKEN_2, TAG_D, 1);
    core.bind(A, ROOM_2, TOKEN_2, TAG_A, 1);
    core.bind(E, ROOM_2, TOKEN_2, TAG_E, 1);
    core.close(E);
    core.publish(A, ROOM_1, 0, b"a0".to_vec(), 2);
    core
}

// ---------------------------------------------------------------- G3: bound connections

/// `bound_conns` lists each bound connection once (A is in two rooms), ascending, closed ones left out;
/// the same before and after a restore.
#[test]
fn bound_conns_lists_each_bound_connection_once_in_order() {
    let core = two_rooms();
    assert_eq!(core.bound_conns().collect::<Vec<_>>(), vec![A, B, C, D]);
    let restored = AnonymousRelayCore::restore(&snap(&core)).expect("restore");
    assert_eq!(restored.bound_conns().collect::<Vec<_>>(), vec![A, B, C, D]);
    assert_eq!(
        AnonymousRelayCore::new(LIMITS).bound_conns().count(),
        0,
        "no room, no connection"
    );
}

/// A restored core with only some of its sockets still open closes exactly the dead ones; the state
/// equals calling `close` for each, and payloads go only to the live connections afterwards.
#[test]
fn close_all_except_closes_exactly_the_dead_ones() {
    let bytes = snap(&two_rooms());
    // every subset of the four bound connections, plus a live connection the core never saw
    let all = [A, B, C, D];
    for mask in 0u8..16 {
        let mut live: BTreeSet<ConnId> = all
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, c)| *c)
            .collect();
        live.insert(ConnId::from_raw(99));
        let mut core = AnonymousRelayCore::restore(&bytes).expect("restore");
        let closed = core.close_all_except(&live);
        let dead: Vec<ConnId> = all.iter().copied().filter(|c| !live.contains(c)).collect();
        assert_eq!(closed, dead, "mask {mask:04b}: the dead ones, ascending");
        let mut by_hand = AnonymousRelayCore::restore(&bytes).expect("restore");
        for c in &dead {
            by_hand.close(*c);
        }
        assert_eq!(snap(&core), snap(&by_hand), "mask {mask:04b}: same state");
        let still: Vec<ConnId> = all.iter().copied().filter(|c| live.contains(c)).collect();
        assert_eq!(core.bound_conns().collect::<Vec<_>>(), still);
    }
    // with A and D live: a publish by A in room 2 goes to D only, and in room 1 to nobody
    let mut core = AnonymousRelayCore::restore(&bytes).expect("restore");
    core.close_all_except(&[A, D].into_iter().collect());
    assert_eq!(
        forwarded_to(&core.publish(A, ROOM_2, 0, b"x".to_vec(), 3)),
        vec![D]
    );
    assert!(forwarded_to(&core.publish(A, ROOM_1, 1, b"y".to_vec(), 3)).is_empty());
    // a closed connection's tag is free again: another connection takes it
    assert!(matches!(
        core.bind(E, ROOM_1, TOKEN_1, TAG_B, 4).as_slice(),
        [Out::Bound { .. }]
    ));
}

// ---------------------------------------------------------------- G4: limits across a restore

/// A state that uses every limit up to a known count: payload 10 bytes, log 3, held 2, rooms 2,
/// members 2, publishes in the window 4, a connection in 2 rooms, tags 3.
fn full_state() -> AnonymousRelayCore {
    let mut core = AnonymousRelayCore::new(full_limits());
    let ten = vec![7u8; 10];
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0);
    // a third tag in room 1: C binds, publishes once and goes (its sequence stays)
    core.bind(C, ROOM_1, TOKEN_1, TAG_C, 0);
    core.publish(C, ROOM_1, 0, b"c".to_vec(), 1);
    core.close(C);
    core.bind(B, ROOM_1, TOKEN_1, TAG_B, 0);
    core.bind(A, ROOM_2, TOKEN_2, TAG_A, 0);
    // A publishes 4 times in room 1 in this window; the log keeps the last 3 (two of 10 bytes)
    for seq in 0..3 {
        core.publish(A, ROOM_1, seq, ten.clone(), 1);
    }
    core.publish(A, ROOM_1, 3, b"a3".to_vec(), 1);
    // two held back for B (it has published nothing, so 0 is the gap)
    core.publish(B, ROOM_1, 1, b"b1".to_vec(), 1);
    core.publish(B, ROOM_1, 2, b"b2".to_vec(), 1);
    core.publish(A, ROOM_2, 0, b"r2".to_vec(), 1);
    core
}

fn full_limits() -> Limits {
    Limits {
        max_payload: 10,
        max_log: 3,
        max_pending: 2,
        max_rooms: 2,
        max_members: 2,
        max_publishes_per_window: 4,
        window: 100,
        max_rooms_per_conn: 2,
        max_tags_per_room: 3,
    }
}

/// Every limit but `window`, each lowered by one below what `full_state` uses
fn each_limit_lowered() -> Vec<(&'static str, Limits)> {
    let l = full_limits();
    vec![
        (
            "max_payload",
            Limits {
                max_payload: 9,
                ..l
            },
        ),
        ("max_log", Limits { max_log: 2, ..l }),
        (
            "max_pending",
            Limits {
                max_pending: 1,
                ..l
            },
        ),
        ("max_rooms", Limits { max_rooms: 1, ..l }),
        (
            "max_members",
            Limits {
                max_members: 1,
                ..l
            },
        ),
        (
            "max_publishes_per_window",
            Limits {
                max_publishes_per_window: 3,
                ..l
            },
        ),
        (
            "max_rooms_per_conn",
            Limits {
                max_rooms_per_conn: 1,
                ..l
            },
        ),
        (
            "max_tags_per_room",
            Limits {
                max_tags_per_room: 2,
                ..l
            },
        ),
    ]
}

/// The state really sits at each limit (a check of the fixture itself).
#[test]
fn full_state_sits_at_every_limit() {
    let core = full_state();
    assert_eq!(core.tags_held(ROOM_1), 3);
    let log = core.log_range(A, ROOM_1, 0, u64::MAX).expect("bound");
    assert_eq!(log.items.len(), 3);
    assert!(log.items.iter().any(|i| i.payload.len() == 10));
    let mut probe = core.clone();
    // the window is spent for A in room 1 (4 publishes)
    assert_eq!(
        refused(&probe.publish(A, ROOM_1, 4, b"z".to_vec(), 2)),
        Some(Refusal::RateLimited)
    );
    // B is 2 ahead: a third held payload is too far ahead
    assert_eq!(
        refused(&probe.publish(B, ROOM_1, 3, b"b3".to_vec(), 2)),
        Some(Refusal::TooFarAhead)
    );
    assert_eq!(
        refused(&probe.bind(D, ROOM_1, TOKEN_1, TAG_D, 2)),
        Some(Refusal::RoomFull)
    );
}

/// Tighter limits refuse a state over any of them with the error `restore` gives for that state; the
/// limits the state sits at exactly accept it.
#[test]
fn tighter_limits_refuse_a_state_over_them() {
    let bytes = snap(&full_state());
    assert!(AnonymousRelayCore::restore_with_limits(&bytes, full_limits()).is_ok());
    for (name, limits) in each_limit_lowered() {
        assert_eq!(
            AnonymousRelayCore::restore_with_limits(&bytes, limits).err(),
            Some(SnapshotError::Inconsistent),
            "{name} lowered by one"
        );
    }
    // `restore` keeps the stored limits: the same bytes restore
    let plain = AnonymousRelayCore::restore(&bytes).expect("restore");
    assert_eq!(snap(&plain), bytes);
    // and the bytes are still checked as bytes first
    assert_eq!(
        AnonymousRelayCore::restore_with_limits(&bytes[..bytes.len() - 1], full_limits()).err(),
        Some(SnapshotError::Truncated)
    );
}

/// Looser limits accept the state; the core then follows the new limits, its snapshot carries them, and
/// that snapshot restores with plain `restore` (also while the log is shorter than the new `max_log`).
#[test]
fn looser_limits_accept_and_apply_after_the_restore() {
    let bytes = snap(&full_state());
    let looser = Limits {
        max_payload: 32,
        max_log: 6,
        max_pending: 4,
        max_rooms: 3,
        max_members: 3,
        max_publishes_per_window: 8,
        window: 100,
        max_rooms_per_conn: 3,
        max_tags_per_room: 4,
    };
    let mut old = AnonymousRelayCore::restore(&bytes).expect("restore");
    let mut core = AnonymousRelayCore::restore_with_limits(&bytes, looser).expect("looser");
    // max_payload: 20 bytes refused before, accepted now
    let twenty = vec![1u8; 20];
    assert_eq!(
        refused(&old.publish(A, ROOM_2, 1, twenty.clone(), 2)),
        Some(Refusal::TooLarge)
    );
    assert_eq!(refused(&core.publish(A, ROOM_2, 1, twenty, 2)), None);
    // max_publishes_per_window: a fifth publish by A in room 1 in the same window
    assert_eq!(
        refused(&old.publish(A, ROOM_1, 4, b"5".to_vec(), 2)),
        Some(Refusal::RateLimited)
    );
    assert_eq!(refused(&core.publish(A, ROOM_1, 4, b"5".to_vec(), 2)), None);
    // max_pending: a third held payload for B
    assert_eq!(
        refused(&core.publish(B, ROOM_1, 3, b"b3".to_vec(), 2)),
        None
    );
    // max_members and max_tags_per_room: a third member under a fourth tag
    assert!(matches!(
        core.bind(D, ROOM_1, TOKEN_1, TAG_D, 2).as_slice(),
        [Out::Bound { .. }]
    ));
    // max_rooms and max_rooms_per_conn: A in a third room
    let room_3 = RoomAlias::from_raw([0x33; 32]);
    assert!(matches!(
        core.bind(A, room_3, TOKEN_1, TAG_A, 2).as_slice(),
        [Out::Bound { .. }]
    ));
    // the snapshot carries the new limits and restores as it is, with room 1's log (4 kept, next index
    // 6) shorter than min(next index, max_log 6)
    let after = snap(&core);
    let again = AnonymousRelayCore::restore(&after).expect("restore after new limits");
    assert_eq!(snap(&again), after);
    // max_log: the log now grows to 6
    let mut core = AnonymousRelayCore::restore_with_limits(&bytes, looser).expect("looser");
    // 3 kept before; each publish keeps one more until 6
    for (k, seq) in (4u64..10).enumerate() {
        core.publish(A, ROOM_1, seq, b"g".to_vec(), 200 + seq);
        let kept = core.log_range(A, ROOM_1, 0, u64::MAX).expect("bound");
        assert_eq!(kept.items.len(), (4 + k).min(6), "publish {seq}");
        let restored = AnonymousRelayCore::restore(&snap(&core)).expect("short log restores");
        assert_eq!(snap(&restored), snap(&core));
    }
    let kept = core.log_range(A, ROOM_1, 0, u64::MAX).expect("bound");
    assert_eq!(kept.items.len(), 6);
}

/// Same limits as stored: the same core as `restore`.
#[test]
fn restore_with_the_stored_limits_is_restore() {
    let bytes = snap(&full_state());
    let a = AnonymousRelayCore::restore(&bytes).expect("restore");
    let b = AnonymousRelayCore::restore_with_limits(&bytes, full_limits()).expect("same limits");
    assert_eq!(snap(&a), snap(&b));
}

/// A log longer than the new `max_log` or its own `min(next index, max_log)` is still refused, and a
/// shorter one is accepted only as the last payloads (indices running up to the next index).
#[test]
fn log_length_rules_after_a_limit_change() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0);
    for seq in 0..3 {
        core.publish(A, ROOM_1, seq, vec![seq as u8], 1);
    }
    let bytes = snap(&core);
    // max_log 4, next index 3: 3 kept; a new max_log of 2 refuses, of 3 accepts
    for (max_log, ok) in [(2, false), (3, true), (100, true)] {
        let got = AnonymousRelayCore::restore_with_limits(&bytes, Limits { max_log, ..LIMITS });
        assert_eq!(got.is_ok(), ok, "max_log {max_log}");
    }
}

// ---------------------------------------------------------------- G5: bounded log replies

fn frame_len(range: &LogRange, room: RoomAlias) -> usize {
    range.clone().into_wire(room).encode().len()
}

/// A room with payloads of mixed lengths kept in its log
fn logged(lens: &[usize]) -> AnonymousRelayCore {
    let limits = Limits {
        max_payload: 200,
        max_log: 64,
        max_publishes_per_window: 1000,
        ..LIMITS
    };
    let mut core = AnonymousRelayCore::new(limits);
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0);
    core.bind(B, ROOM_1, TOKEN_1, TAG_B, 0);
    for (seq, n) in lens.iter().enumerate() {
        let p: Vec<u8> = (0..*n).map(|k| (k + seq) as u8).collect();
        core.publish(A, ROOM_1, seq as u64, p, 1);
    }
    core
}

/// Ask again from the last index + 1 until the reply is empty; every reply fits, is the longest prefix
/// that fits, and together they cover the range exactly once.
#[test]
fn replies_cover_the_range_exactly_once() {
    let lens = [5, 40, 0, 17, 120, 3, 3, 64, 1, 90];
    let core = logged(&lens);
    let whole = core.log_range(B, ROOM_1, 0, u64::MAX).expect("bound");
    for budget in [201usize, 250, 300, 512, 4096] {
        let mut got: Vec<LogItem> = Vec::new();
        let mut from = 0u64;
        let mut replies = 0;
        loop {
            replies += 1;
            assert!(replies <= 2 * lens.len() + 2, "budget {budget}: no end");
            let r = core
                .log_range_within(B, ROOM_1, from, u64::MAX, budget)
                .expect("every item fits this budget");
            assert_eq!(r.first_index, whole.first_index);
            assert!(frame_len(&r, ROOM_1) <= budget, "budget {budget}: fits");
            // the longest prefix: one more item would not fit
            let rest: Vec<&LogItem> = whole.items.iter().filter(|i| i.index >= from).collect();
            if let Some(next) = rest.get(r.items.len()) {
                let mut longer = r.clone();
                longer.items.push((*next).clone());
                assert!(
                    frame_len(&longer, ROOM_1) > budget,
                    "budget {budget}: longest"
                );
            }
            let Some(last) = r.items.last() else {
                break;
            };
            from = last.index + 1;
            got.extend(r.items);
        }
        assert_eq!(got, whole.items, "budget {budget}: exactly once, in order");
    }
}

/// The budget is exact: a budget equal to the encoded frame of the first k items returns k items, one
/// byte less returns k - 1.
#[test]
fn the_budget_is_exact_at_the_boundary() {
    let lens = [5, 40, 0, 17, 120, 3];
    let core = logged(&lens);
    let whole = core.log_range(B, ROOM_1, 0, u64::MAX).expect("bound");
    for k in 1..=whole.items.len() {
        let prefix = LogRange {
            first_index: whole.first_index,
            items: whole.items[..k].to_vec(),
        };
        let exact = frame_len(&prefix, ROOM_1);
        let r = core
            .log_range_within(B, ROOM_1, 0, u64::MAX, exact)
            .expect("fits");
        assert_eq!(r.items.len(), k, "budget {exact} = frame of {k} items");
        let r = core.log_range_within(B, ROOM_1, 0, u64::MAX, exact - 1);
        if k == 1 {
            assert_eq!(r, Err(Refusal::TooLarge), "first item one byte over");
        } else {
            assert_eq!(r.expect("fits").items.len(), k - 1, "one byte over {k}");
        }
    }
    // the documented sizes: a Log frame's head is 45 bytes, each item 36 bytes plus its payload
    let empty = LogRange {
        first_index: 0,
        items: Vec::new(),
    };
    assert_eq!(frame_len(&empty, ROOM_1), 45);
    let one = LogRange {
        first_index: 0,
        items: whole.items[..1].to_vec(),
    };
    assert_eq!(frame_len(&one, ROOM_1), 45 + 36 + lens[0]);
}

/// A payload that cannot fit on its own is refused (`TooLarge`), never an empty reply; items before it
/// still come back, and the next ask (starting at it) is refused, so a client never loops on empty.
#[test]
fn a_payload_bigger_than_the_budget_is_refused_not_empty() {
    let lens = [5, 150, 5];
    let core = logged(&lens);
    let budget = 45 + 36 + 100;
    let r = core
        .log_range_within(B, ROOM_1, 0, u64::MAX, budget)
        .expect("the first fits");
    assert_eq!(r.items.iter().map(|i| i.index).collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        core.log_range_within(B, ROOM_1, 1, u64::MAX, budget),
        Err(Refusal::TooLarge)
    );
    // past it, the rest comes back; past the end, the reply is empty (done)
    let r = core
        .log_range_within(B, ROOM_1, 2, u64::MAX, budget)
        .expect("fits");
    assert_eq!(r.items.iter().map(|i| i.index).collect::<Vec<_>>(), vec![2]);
    let done = core
        .log_range_within(B, ROOM_1, 3, u64::MAX, budget)
        .expect("empty fits");
    assert!(done.items.is_empty());
    // a budget below the frame's head refuses even an empty reply
    assert_eq!(
        core.log_range_within(B, ROOM_1, 3, u64::MAX, 44),
        Err(Refusal::TooLarge)
    );
    assert!(core.log_range_within(B, ROOM_1, 3, u64::MAX, 45).is_ok());
}

/// The range bounds and the refusals are those of `log_range`.
#[test]
fn bounds_and_refusals_follow_log_range() {
    let core = logged(&[1, 2, 3, 4, 5]);
    let big = 1 << 20;
    for (from, to) in [(0, 5), (1, 4), (2, 2), (4, 3), (0, u64::MAX), (9, 20)] {
        assert_eq!(
            core.log_range_within(B, ROOM_1, from, to, big),
            core.log_range(B, ROOM_1, from, to),
            "{from}..{to}"
        );
    }
    assert_eq!(
        core.log_range_within(C, ROOM_1, 0, 5, big),
        Err(Refusal::NotBound)
    );
    assert_eq!(
        core.log_range_within(B, ROOM_2, 0, 5, big),
        Err(Refusal::NoRoom)
    );
}

// ---------------------------------------------------------------- G6: Out to the wire

const ALL_REFUSALS: [Refusal; 13] = [
    Refusal::NoRoom,
    Refusal::BadToken,
    Refusal::NotBound,
    Refusal::TagInUse,
    Refusal::TooLarge,
    Refusal::TooFarAhead,
    Refusal::TooManyRooms,
    Refusal::RoomFull,
    Refusal::RateLimited,
    Refusal::ConnRoomLimit,
    Refusal::TagChange,
    Refusal::TooManyTags,
    Refusal::Exhausted,
];

/// Every `Out` variant maps to its connection and frame, and the frame survives encode / decode.
#[test]
fn every_out_maps_to_its_frame() {
    let mut cases: Vec<(Out, ConnId, Frame)> = vec![
        (
            Out::Bound {
                to: A,
                room: ROOM_1,
            },
            A,
            Frame::Bound { room: ROOM_1 },
        ),
        (
            Out::Forward {
                to: B,
                room: ROOM_2,
                from: TAG_C,
                seq: 7,
                payload: b"sealed".to_vec(),
            },
            B,
            Frame::Forward {
                room: ROOM_2,
                from: TAG_C,
                seq: 7,
                payload: b"sealed".to_vec(),
            },
        ),
        (
            Out::Resend {
                to: C,
                room: ROOM_1,
                from: 3,
                to_seq: 9,
            },
            C,
            Frame::Resend {
                room: ROOM_1,
                from: 3,
                to_seq: 9,
            },
        ),
    ];
    for (k, reason) in ALL_REFUSALS.into_iter().enumerate() {
        let to = ConnId::from_raw(100 + k as u64);
        cases.push((
            Out::Refused {
                to,
                room: ROOM_2,
                reason,
            },
            to,
            Frame::Refused {
                room: ROOM_2,
                reason,
            },
        ));
    }
    for (out, conn, frame) in cases {
        let (to, got) = out.clone().into_wire();
        assert_eq!((to, &got), (conn, &frame), "{out:?}");
        assert_eq!(Frame::decode(&got.encode()), Ok(frame), "{out:?}");
    }
}

/// What a live core sends maps to frames that decode back, and a log read maps to the `Log` frame.
#[test]
fn a_live_core_output_and_a_log_read_map_to_frames() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let mut outs = Vec::new();
    outs.extend(core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0));
    outs.extend(core.bind(B, ROOM_1, TOKEN_1, TAG_B, 0));
    outs.extend(core.publish(A, ROOM_1, 0, b"p0".to_vec(), 1));
    outs.extend(core.publish(A, ROOM_1, 2, b"p2".to_vec(), 1));
    outs.extend(core.bind(C, ROOM_1, TOKEN_2, TAG_C, 1));
    assert_eq!(outs.len(), 5);
    for out in outs {
        let (_, frame) = out.into_wire();
        assert_eq!(Frame::decode(&frame.encode()), Ok(frame));
    }
    let range = core.log_range(B, ROOM_1, 0, 10).expect("bound");
    let frame = range.clone().into_wire(ROOM_1);
    assert_eq!(
        frame,
        Frame::Log {
            room: ROOM_1,
            first_index: 0,
            items: vec![(0, TAG_A, 0, b"p0".to_vec())],
        }
    );
    assert_eq!(Frame::decode(&frame.encode()), Ok(frame));
    // a log that dropped its oldest: the frame carries the first index still kept (B publishes 0..5,
    // indices 1..=5, and max_log 4 keeps 2..=5)
    for seq in 0..5 {
        core.publish(B, ROOM_1, seq, vec![seq as u8], 50);
    }
    let range = core.log_range(A, ROOM_1, 4, 6).expect("bound");
    assert_eq!(
        range.into_wire(ROOM_1),
        Frame::Log {
            room: ROOM_1,
            first_index: 2,
            items: vec![(4, TAG_B, 3, vec![3]), (5, TAG_B, 4, vec![4])],
        }
    );
}

// ---------------------------------------------------------------- leave one room

/// `leave` unbinds one room: the connection stays bound to its other rooms, its tag in the room it left
/// is free, and leaving a room it is not bound to changes nothing.
#[test]
fn leave_unbinds_one_room_only() {
    let mut core = two_rooms();
    let before = snap(&core);
    core.leave(B, ROOM_2);
    core.leave(ConnId::from_raw(99), ROOM_1);
    core.leave(A, RoomAlias::from_raw([0x77; 32]));
    assert_eq!(snap(&core), before, "not bound there: nothing changes");
    core.leave(A, ROOM_1);
    assert_eq!(core.bound_conns().collect::<Vec<_>>(), vec![A, B, C, D]);
    assert_eq!(
        refused(&core.publish(A, ROOM_1, 1, b"x".to_vec(), 3)),
        Some(Refusal::NotBound)
    );
    assert_eq!(
        forwarded_to(&core.publish(A, ROOM_2, 0, b"y".to_vec(), 3)),
        vec![D]
    );
    assert_eq!(
        core.log_range(A, ROOM_1, 0, 10).err(),
        Some(Refusal::NotBound)
    );
    // A's tag in room 1 is free for another connection
    assert!(matches!(
        core.bind(E, ROOM_1, TOKEN_1, TAG_A, 4).as_slice(),
        [Out::Bound { .. }]
    ));
    // the room keeps its log
    assert_eq!(
        core.log_range(B, ROOM_1, 0, 10).expect("bound").items.len(),
        1
    );
    // leaving the last room unbinds the connection
    core.leave(A, ROOM_2);
    assert_eq!(core.bound_conns().collect::<Vec<_>>(), vec![B, C, D, E]);
}
