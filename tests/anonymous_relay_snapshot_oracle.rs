//! Contract tests for writing the anonymous relay's state out and reading it back
//! ([`AnonymousRelayCore::snapshot`] / [`AnonymousRelayCore::restore`]), for a host that may evict and
//! restart the process holding the core.
//!
//! - round trip: a scripted sequence of frames (binds, publishes in and out of order, gaps that ask for a
//!   resend, duplicates, rate limiting across a window, rooms making way, closes, log reads) is run on one
//!   core, and again with a snapshot and restore into a fresh core at every cut point; the frames sent and
//!   the final snapshot must be the same;
//! - the bytes are stable: the same state gives the same bytes whatever order it was reached in, and a
//!   small fixed state gives bytes written out by hand from the documented format (the oracle is the
//!   format table in `src/anonymous_relay/snapshot.rs`, not the encoder);
//! - corrupted input (every truncation, other magic, other versions, trailing bytes, counts and lengths
//!   that do not match the bytes, keys out of order, states the core cannot reach) is a typed error,
//!   never a panic.

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, LogRange, Out, Refusal, RoomAlias, RoomToken, SenderTag,
    SnapshotError, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

const ROOM_1: RoomAlias = RoomAlias::from_raw([0x11; 32]);
const ROOM_2: RoomAlias = RoomAlias::from_raw([0x22; 32]);
const ROOM_3: RoomAlias = RoomAlias::from_raw([0x33; 32]);
const ROOM_9: RoomAlias = RoomAlias::from_raw([0x99; 32]);
const TOKEN_1: RoomToken = RoomToken::from_raw([0xa1; 32]);
const TOKEN_2: RoomToken = RoomToken::from_raw([0xa2; 32]);
const TOKEN_3: RoomToken = RoomToken::from_raw([0xa3; 32]);
const TOKEN_X: RoomToken = RoomToken::from_raw([0xee; 32]);
const A: ConnId = ConnId::from_raw(1);
const B: ConnId = ConnId::from_raw(2);
const C: ConnId = ConnId::from_raw(3);
const D: ConnId = ConnId::from_raw(4);
const E: ConnId = ConnId::from_raw(5);
const F: ConnId = ConnId::from_raw(6);
const TAG_A: SenderTag = SenderTag::from_raw([0x0a; 16]);
const TAG_B: SenderTag = SenderTag::from_raw([0x0b; 16]);
const TAG_C: SenderTag = SenderTag::from_raw([0x0c; 16]);
const TAG_D: SenderTag = SenderTag::from_raw([0x0d; 16]);
const TAG_E: SenderTag = SenderTag::from_raw([0x0e; 16]);
const TAG_Z: SenderTag = SenderTag::from_raw([0x5a; 16]);

const LIMITS: Limits = Limits {
    max_payload: 16,
    max_log: 4,
    max_pending: 3,
    max_rooms: 2,
    max_members: 3,
    max_publishes_per_window: 5,
    window: 10,
    max_rooms_per_conn: 2,
    max_tags_per_room: 3,
};

#[derive(Clone, Debug)]
enum Op {
    Bind(ConnId, RoomAlias, RoomToken, SenderTag, u64),
    Publish(ConnId, RoomAlias, u64, &'static [u8], u64),
    Log(ConnId, RoomAlias, u64, u64),
    Close(ConnId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Step {
    Out(Vec<Out>),
    Log(Result<LogRange, Refusal>),
    Closed,
}

fn script() -> Vec<Op> {
    use Op::{Bind, Close, Log, Publish};
    vec![
        Bind(A, ROOM_1, TOKEN_1, TAG_A, 0),
        Bind(B, ROOM_1, TOKEN_1, TAG_B, 1),
        Publish(A, ROOM_1, 0, b"a0", 2),
        // ahead of the next expected: held back, resend asked
        Publish(A, ROOM_1, 2, b"a2", 3),
        Publish(A, ROOM_1, 3, b"a3", 3),
        // a duplicate of a held payload
        Publish(A, ROOM_1, 2, b"a2", 4),
        Publish(B, ROOM_1, 0, b"b0", 4),
        // fills the gap: 1, 2, 3 leave together
        Publish(A, ROOM_1, 1, b"a1", 5),
        // the sixth publish of TAG_A in the window that opened at 2
        Publish(A, ROOM_1, 4, b"a4", 6),
        Bind(C, ROOM_1, TOKEN_X, TAG_C, 7),
        Bind(C, ROOM_1, TOKEN_1, TAG_A, 7),
        Bind(C, ROOM_1, TOKEN_1, TAG_C, 8),
        Log(C, ROOM_1, 0, 10),
        // a new window
        Publish(A, ROOM_1, 4, b"a4", 13),
        // a duplicate below the next expected
        Publish(A, ROOM_1, 0, b"a0", 13),
        Publish(B, ROOM_1, 2, b"b2", 13),
        Close(B),
        // TAG_B still holds a payload, so it cannot make way
        Bind(D, ROOM_1, TOKEN_1, TAG_D, 14),
        Bind(D, ROOM_2, TOKEN_2, TAG_D, 15),
        Publish(D, ROOM_2, 0, b"d0", 15),
        Bind(E, ROOM_3, TOKEN_3, TAG_E, 16),
        Close(D),
        // ROOM_2 is empty and makes way
        Bind(E, ROOM_3, TOKEN_3, TAG_E, 18),
        Bind(D, ROOM_2, TOKEN_X, TAG_D, 19),
        Close(E),
        // ROOM_3 makes way; ROOM_2 opens again under another token
        Bind(D, ROOM_2, TOKEN_X, TAG_D, 21),
        Bind(D, ROOM_2, TOKEN_2, TAG_D, 21),
        // the publisher of TAG_B comes back on a new connection
        Bind(F, ROOM_1, TOKEN_1, TAG_B, 22),
        Publish(F, ROOM_1, 1, b"b1", 23),
        Log(A, ROOM_1, 0, 100),
        Publish(F, ROOM_1, 9, b"b9", 23),
        Publish(F, ROOM_1, 10, b"b10", 24),
        Publish(F, ROOM_1, 11, b"b11", 24),
        Publish(F, ROOM_1, 12, b"b12", 24),
        Publish(A, ROOM_1, 5, b"seventeen bytes!!", 25),
        Publish(D, ROOM_1, 0, b"x", 25),
        Publish(A, ROOM_9, 0, b"x", 25),
        Log(D, ROOM_1, 0, 100),
        Bind(A, ROOM_2, TOKEN_X, TAG_A, 26),
        Bind(A, ROOM_9, TOKEN_1, TAG_A, 26),
        Bind(A, ROOM_1, TOKEN_1, TAG_Z, 27),
        Publish(A, ROOM_2, 0, b"in 2", 28),
        Log(D, ROOM_2, 0, 100),
        Publish(F, ROOM_1, 2, b"b2", 29),
        Publish(F, ROOM_1, 3, b"b3", 29),
        Publish(F, ROOM_1, 4, b"b4", 29),
        Publish(F, ROOM_1, 5, b"b5", 33),
        Close(A),
        Log(F, ROOM_1, 0, 100),
        // two empty rooms: the one idle longest (ROOM_2, last active at 28, although its alias is
        // the larger) makes way, so the restored core must keep each room's last activity
        Close(C),
        Close(F),
        Close(D),
        Bind(E, ROOM_3, TOKEN_3, TAG_E, 40),
        Bind(B, ROOM_1, TOKEN_1, TAG_B, 41),
        Log(B, ROOM_1, 0, 100),
    ]
}

fn apply(core: &mut AnonymousRelayCore, op: &Op) -> Step {
    match *op {
        Op::Bind(conn, room, token, tag, now) => Step::Out(core.bind(conn, room, token, tag, now)),
        Op::Publish(conn, room, seq, payload, now) => {
            Step::Out(core.publish(conn, room, seq, payload.to_vec(), now))
        }
        Op::Log(conn, room, from, to) => Step::Log(core.log_range(conn, room, from, to)),
        Op::Close(conn) => {
            core.close(conn);
            Step::Closed
        }
    }
}

fn run(core: &mut AnonymousRelayCore, ops: &[Op]) -> Vec<Step> {
    ops.iter().map(|op| apply(core, op)).collect()
}

/// The script exercises what the snapshot must keep: forwards, resends, every refusal it names, and
/// log reads that return items (so a script that went quiet cannot pass the round trip vacuously).
#[test]
fn script_reaches_every_path_it_names() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let steps = run(&mut core, &script());
    let outs: Vec<&Out> = steps
        .iter()
        .filter_map(|s| match s {
            Step::Out(o) => Some(o),
            _ => None,
        })
        .flatten()
        .collect();
    let refusals: std::collections::BTreeSet<String> = outs
        .iter()
        .filter_map(|o| match o {
            Out::Refused { reason, .. } => Some(format!("{reason:?}")),
            _ => None,
        })
        .collect();
    for r in [
        Refusal::BadToken,
        Refusal::TagInUse,
        Refusal::RateLimited,
        Refusal::TooManyTags,
        Refusal::TooManyRooms,
        Refusal::TooFarAhead,
        Refusal::TooLarge,
        Refusal::NotBound,
        Refusal::NoRoom,
        Refusal::ConnRoomLimit,
        Refusal::TagChange,
    ] {
        assert!(refusals.contains(&format!("{r:?}")), "{r:?} not reached");
    }
    assert!(outs.iter().any(|o| matches!(o, Out::Resend { .. })));
    assert!(
        outs.iter()
            .filter(|o| matches!(o, Out::Forward { .. }))
            .count()
            >= 10
    );
    let logged: usize = steps
        .iter()
        .filter_map(|s| match s {
            Step::Log(Ok(r)) => Some(r.items.len()),
            _ => None,
        })
        .sum();
    assert!(logged >= 6, "log reads returned {logged} items");
}

/// Oracle 1: running the first `k` frames, snapshotting, restoring into a fresh core and running the
/// rest gives the same frames and the same final snapshot as running all of them on one core, for every
/// cut point `k`.
#[test]
fn round_trip_at_every_cut_point_matches_one_core() {
    let ops = script();
    let mut whole = AnonymousRelayCore::new(LIMITS);
    let expected = run(&mut whole, &ops);
    let expected_final = whole.snapshot().expect("representable");
    for k in 0..=ops.len() {
        let mut first = AnonymousRelayCore::new(LIMITS);
        let mut steps = run(&mut first, &ops[..k]);
        let bytes = first.snapshot().expect("representable");
        let mut second = AnonymousRelayCore::restore(&bytes).expect("restore");
        steps.extend(run(&mut second, &ops[k..]));
        for (i, (got, want)) in steps.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "cut {k}: frame {i} ({:?}) differs", ops[i]);
        }
        assert_eq!(steps.len(), expected.len());
        assert_eq!(
            second.snapshot().expect("representable"),
            expected_final,
            "cut {k}: final state"
        );
    }
}

/// Snapshot and restore twice in a row at every cut point (restore of a restored core) also matches.
#[test]
fn restore_then_snapshot_gives_the_same_bytes_at_every_cut_point() {
    let ops = script();
    let mut core = AnonymousRelayCore::new(LIMITS);
    for (k, op) in ops.iter().enumerate() {
        let bytes = core.snapshot().expect("representable");
        let restored = AnonymousRelayCore::restore(&bytes).expect("restore");
        assert_eq!(
            restored.snapshot().expect("representable"),
            bytes,
            "cut {k}"
        );
        apply(&mut core, op);
    }
}

/// Oracle 2a: the same state gives the same bytes, whatever order the members were bound in.
#[test]
fn same_state_same_bytes_whatever_the_order() {
    let mut one = AnonymousRelayCore::new(LIMITS);
    one.bind(A, ROOM_1, TOKEN_1, TAG_A, 3);
    one.bind(B, ROOM_1, TOKEN_1, TAG_B, 3);
    one.bind(C, ROOM_2, TOKEN_2, TAG_C, 3);
    let mut two = AnonymousRelayCore::new(LIMITS);
    two.bind(C, ROOM_2, TOKEN_2, TAG_C, 3);
    two.bind(B, ROOM_1, TOKEN_1, TAG_B, 3);
    two.bind(A, ROOM_1, TOKEN_1, TAG_A, 3);
    assert_eq!(
        one.snapshot().expect("representable"),
        two.snapshot().expect("representable")
    );
    assert_eq!(
        one.snapshot().expect("representable"),
        one.snapshot().expect("representable")
    );
}

fn h(s: &str) -> Vec<u8> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn rep(byte: &str, n: usize) -> String {
    byte.repeat(n)
}

/// The fixed state of the golden: two members, a forwarded payload, one held back.
fn golden_state() -> AnonymousRelayCore {
    let mut core = AnonymousRelayCore::new(LIMITS);
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 5);
    core.bind(B, ROOM_1, TOKEN_1, TAG_B, 6);
    core.publish(A, ROOM_1, 0, b"hi".to_vec(), 7);
    core.publish(A, ROOM_1, 2, b"x".to_vec(), 8);
    core
}

/// The golden bytes, written out by hand from the format table (oracle: the module documentation of
/// `alice_sync::anonymous_relay::snapshot`, version 1).
fn golden_bytes() -> Vec<u8> {
    let mut s = String::new();
    s += "41524c59"; // magic ARLY
    s += "0001"; // version 1
    s += "0000000000000010"; // max_payload 16
    s += "0000000000000004"; // max_log 4
    s += "0000000000000003"; // max_pending 3
    s += "0000000000000002"; // max_rooms 2
    s += "0000000000000003"; // max_members 3
    s += "00000005"; // max_publishes_per_window 5 (u32)
    s += "000000000000000a"; // window 10
    s += "0000000000000002"; // max_rooms_per_conn 2
    s += "0000000000000003"; // max_tags_per_room 3
    s += "00000001"; // one room
    s += &rep("11", 32); // alias
    s += &rep("a1", 32); // token
    s += "0000000000000008"; // last_active 8
    s += "0000000000000001"; // log's next index 1
    s += "00000002"; // two members
    s += "0000000000000001";
    s += &rep("0a", 16);
    s += "0000000000000002";
    s += &rep("0b", 16);
    s += "00000001"; // one next sequence number
    s += &rep("0a", 16);
    s += "0000000000000001";
    s += "00000001"; // one tag with held payloads
    s += &rep("0a", 16);
    s += "00000001"; // one held payload
    s += "0000000000000002"; // seq 2
    s += "00000001"; // payload length 1
    s += "78"; // "x"
    s += "00000001"; // one rate window
    s += &rep("0a", 16);
    s += "0000000000000007"; // window start 7
    s += "00000002"; // two publishes
    s += "00000001"; // one log item
    s += "0000000000000000"; // index 0
    s += &rep("0a", 16);
    s += "0000000000000000"; // seq 0
    s += "00000002"; // payload length 2
    s += "6869"; // "hi"
    h(&s)
}

const GOLDEN_LEN: usize = 349;
// offsets into the golden, from the format table
const OFF_VERSION: usize = 4;
const OFF_ROOM_COUNT: usize = 74;
const OFF_MEMBER_COUNT: usize = 158;
const OFF_MEMBER_1_CONN: usize = 162;
const OFF_MEMBER_2_CONN: usize = 186;
const OFF_MEMBER_2_TAG: usize = 194;
const OFF_NEXT_INDEX: usize = 150;
const OFF_HELD_SEQ: usize = 262;
const OFF_LOG_COUNT: usize = 307;
const OFF_LOG_INDEX: usize = 311;
const OFF_LOG_PAYLOAD_LEN: usize = 343;
const OFF_LIMIT_MAX_LOG: usize = 14;

/// Oracle 2b: a small fixed state gives the bytes written out by hand.
#[test]
fn golden_bytes_of_a_small_state() {
    let want = golden_bytes();
    assert_eq!(want.len(), GOLDEN_LEN);
    assert_eq!(&want[..4], &SNAPSHOT_MAGIC);
    assert_eq!(
        u16::from_be_bytes([want[OFF_VERSION], want[OFF_VERSION + 1]]),
        SNAPSHOT_VERSION
    );
    let got = golden_state().snapshot().expect("representable");
    assert_eq!(hexs(&got), hexs(&want));
    // and the bytes read back into that state
    let restored = AnonymousRelayCore::restore(&want).expect("restore golden");
    assert_eq!(restored.snapshot().expect("representable"), want);
    let log = restored.log_range(B, ROOM_1, 0, 10).expect("bound");
    assert_eq!(log.items.len(), 1);
    assert_eq!(log.items[0].payload, b"hi");
    assert_eq!(restored.tags_held(ROOM_1), 2);
}

fn hexs(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn rich() -> Vec<u8> {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let ops = script();
    run(&mut core, &ops[..ops.len() - 2]);
    core.snapshot().expect("representable")
}

fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_be_bytes());
}

fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_be_bytes());
}

/// Oracle 3a: every truncation is `Truncated`, for the golden and for a state with several rooms.
#[test]
fn every_truncation_is_truncated() {
    for bytes in [golden_bytes(), rich()] {
        assert!(AnonymousRelayCore::restore(&bytes).is_ok());
        for n in 0..bytes.len() {
            assert_eq!(
                AnonymousRelayCore::restore(&bytes[..n]).err(),
                Some(SnapshotError::Truncated),
                "length {n} of {}",
                bytes.len()
            );
        }
    }
}

/// Oracle 3b: other magic, other versions and trailing bytes.
#[test]
fn magic_version_and_trailing_bytes() {
    let good = golden_bytes();
    let mut b = good.clone();
    b[0] ^= 0x01;
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::BadMagic)
    );
    for v in [0u16, 2, 0x0100, 0xffff] {
        let mut b = good.clone();
        b[OFF_VERSION..OFF_VERSION + 2].copy_from_slice(&v.to_be_bytes());
        assert_eq!(
            AnonymousRelayCore::restore(&b).err(),
            Some(SnapshotError::UnknownVersion(v)),
            "version {v}"
        );
    }
    for tail in [&[0u8][..], &[0xff, 0xff], &[0; 64]] {
        let mut b = good.clone();
        b.extend_from_slice(tail);
        assert_eq!(
            AnonymousRelayCore::restore(&b).err(),
            Some(SnapshotError::Trailing)
        );
    }
}

/// Oracle 3c: counts and lengths that do not match the bytes; huge ones fail as `Truncated` without
/// reserving space for them.
#[test]
fn counts_and_lengths_that_do_not_match_the_bytes() {
    let good = golden_bytes();
    let cases: [(usize, u32, SnapshotError); 7] = [
        (OFF_ROOM_COUNT, 2, SnapshotError::Truncated),
        (OFF_ROOM_COUNT, 0, SnapshotError::Trailing),
        // counts over their limit are refused when read (before any entry)
        (OFF_ROOM_COUNT, u32::MAX, SnapshotError::Inconsistent),
        (OFF_MEMBER_COUNT, u32::MAX, SnapshotError::Inconsistent),
        (OFF_LOG_COUNT, u32::MAX, SnapshotError::Inconsistent),
        (OFF_LOG_COUNT, 0, SnapshotError::Inconsistent),
        // a payload length over max_payload (16), refused before its bytes
        (OFF_LOG_PAYLOAD_LEN, u32::MAX, SnapshotError::Inconsistent),
    ];
    for (at, v, want) in cases {
        let mut b = good.clone();
        put_u32(&mut b, at, v);
        assert_eq!(
            AnonymousRelayCore::restore(&b).err(),
            Some(want),
            "u32 {v} at {at}"
        );
    }
    // with a max_payload that allows it, a payload length past the bytes present is `Truncated`,
    // without reserving space for it
    let mut b = good.clone();
    put_u64(&mut b, 6, u64::MAX);
    put_u32(&mut b, OFF_LOG_PAYLOAD_LEN, u32::MAX);
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::Truncated)
    );
    // a payload shorter than its stated length: the rest of the snapshot is read as the payload
    let mut b = good.clone();
    put_u32(&mut b, OFF_LOG_PAYLOAD_LEN, 1);
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::Trailing)
    );
}

/// Oracle 3d: keys out of order or repeated, and states the core cannot reach.
#[test]
fn keys_out_of_order_and_unreachable_states() {
    let good = golden_bytes();
    let restore = |b: &[u8]| AnonymousRelayCore::restore(b).err();
    // members out of order, and a repeated connection
    let mut b = good.clone();
    put_u64(&mut b, OFF_MEMBER_1_CONN, 3);
    assert_eq!(restore(&b), Some(SnapshotError::NotCanonical));
    let mut b = good.clone();
    put_u64(&mut b, OFF_MEMBER_2_CONN, 1);
    assert_eq!(restore(&b), Some(SnapshotError::NotCanonical));
    // two connections on one tag
    let mut b = good.clone();
    b[OFF_MEMBER_2_TAG..OFF_MEMBER_2_TAG + 16].copy_from_slice(&[0x0a; 16]);
    assert_eq!(restore(&b), Some(SnapshotError::Inconsistent));
    // a held payload at or before its tag's next sequence number (1)
    for seq in [0, 1] {
        let mut b = good.clone();
        put_u64(&mut b, OFF_HELD_SEQ, seq);
        assert_eq!(restore(&b), Some(SnapshotError::Inconsistent), "held {seq}");
    }
    // log indices that do not run up to the next index
    let mut b = good.clone();
    put_u64(&mut b, OFF_LOG_INDEX, 5);
    assert_eq!(restore(&b), Some(SnapshotError::Inconsistent));
    let mut b = good.clone();
    put_u64(&mut b, OFF_NEXT_INDEX, 0);
    assert_eq!(restore(&b), Some(SnapshotError::Inconsistent));
    // a log longer than max_log
    let mut b = good.clone();
    put_u64(&mut b, OFF_LIMIT_MAX_LOG, 0);
    assert_eq!(restore(&b), Some(SnapshotError::Inconsistent));
    // a state over another limit: a held payload over max_payload (6), more held than max_pending
    // (22), more rooms than max_rooms (30), more members than max_members (38), a connection in more
    // rooms than max_rooms_per_conn (58), more tags than max_tags_per_room (66)
    for (at, v) in [(6, 0), (22, 0), (30, 0), (38, 1), (58, 0), (66, 1)] {
        let mut b = good.clone();
        put_u64(&mut b, at, v);
        assert_eq!(
            restore(&b),
            Some(SnapshotError::Inconsistent),
            "limit at {at} = {v}"
        );
    }
}

/// Operations run on a core restored from changed bytes: binds, publishes at the edges of the sequence
/// space, log reads and closes, across the rooms and connections the snapshots use
fn exercise(core: &mut AnonymousRelayCore) {
    for room in [ROOM_1, ROOM_2, ROOM_3] {
        for raw in 1..=6 {
            let conn = ConnId::from_raw(raw);
            for seq in [0, 1, 2, 3, 5, u64::MAX - 1, u64::MAX] {
                core.publish(conn, room, seq, b"p".to_vec(), 1_000);
            }
            let _ = core.log_range(conn, room, 0, u64::MAX);
        }
        core.bind(ConnId::from_raw(7), room, TOKEN_1, TAG_Z, 1_001);
        core.publish(ConnId::from_raw(7), room, 0, b"z".to_vec(), u64::MAX);
    }
    for raw in 1..=7 {
        core.close(ConnId::from_raw(raw));
    }
    let _ = core.snapshot();
}

/// Oracle 3e: no single changed byte (flipped, set to 0x00 or 0xff) and no 8-byte field set to all ones
/// makes the decoder panic, and a core restored from such bytes takes further frames without a panic.
#[test]
fn no_byte_change_panics() {
    for bytes in [golden_bytes(), rich()] {
        let mut changed: Vec<(usize, Vec<u8>)> = Vec::new();
        for i in 0..bytes.len() {
            for f in [|x: u8| x ^ 0xff, |x: u8| x ^ 0x01, |_| 0x00, |_| 0xff] {
                let mut b = bytes.clone();
                b[i] = f(b[i]);
                changed.push((i, b));
            }
            if i + 8 <= bytes.len() {
                let mut b = bytes.clone();
                b[i..i + 8].copy_from_slice(&[0xff; 8]);
                changed.push((i, b));
            }
        }
        for (i, b) in changed {
            let r = std::panic::catch_unwind(|| {
                AnonymousRelayCore::restore(&b).map(|mut c| {
                    let again = c.snapshot();
                    exercise(&mut c);
                    again
                })
            });
            let r = r.unwrap_or_else(|_| panic!("panic at byte {i}"));
            // a restored state writes the bytes it was read from (one encoding per state)
            if let Ok(again) = r {
                assert_eq!(again, Ok(b), "byte {i}");
            }
        }
    }
}

/// A limit that does not fit `usize` is refused on a 32-bit target.
#[cfg(target_pointer_width = "32")]
#[test]
fn limit_too_large_for_usize() {
    let mut b = golden_bytes();
    put_u64(&mut b, OFF_LIMIT_MAX_LOG, u64::from(u32::MAX) + 1);
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::LimitTooLarge)
    );
}

// ---- edges of the counters, limits checked as entries are read, and the log against the sequences ----

/// The held-back payloads of one tag in a hand-built snapshot: `(seq, payload)`
type Held = Vec<(u64, Vec<u8>)>;

/// A room of a hand-built snapshot (written from the format table, not by the encoder)
#[derive(Clone, Default)]
struct RoomSpec {
    next_index: u64,
    members: Vec<(u64, [u8; 16])>,
    next_seq: Vec<([u8; 16], u64)>,
    held: Vec<([u8; 16], Held)>,
    rate: Vec<([u8; 16], u64, u32)>,
    log: Vec<(u64, [u8; 16], u64, Vec<u8>)>,
}

fn n32(v: usize) -> [u8; 4] {
    u32::try_from(v).expect("small").to_be_bytes()
}

fn build(limits: Limits, room: &RoomSpec) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"ARLY");
    b.extend_from_slice(&1u16.to_be_bytes());
    for v in [
        limits.max_payload,
        limits.max_log,
        limits.max_pending,
        limits.max_rooms,
        limits.max_members,
    ] {
        b.extend_from_slice(&(v as u64).to_be_bytes());
    }
    b.extend_from_slice(&limits.max_publishes_per_window.to_be_bytes());
    b.extend_from_slice(&limits.window.to_be_bytes());
    b.extend_from_slice(&(limits.max_rooms_per_conn as u64).to_be_bytes());
    b.extend_from_slice(&(limits.max_tags_per_room as u64).to_be_bytes());
    b.extend_from_slice(&n32(1));
    b.extend_from_slice(&[0x11; 32]);
    b.extend_from_slice(&[0xa1; 32]);
    b.extend_from_slice(&0u64.to_be_bytes()); // last_active
    b.extend_from_slice(&room.next_index.to_be_bytes());
    b.extend_from_slice(&n32(room.members.len()));
    for (c, t) in &room.members {
        b.extend_from_slice(&c.to_be_bytes());
        b.extend_from_slice(t);
    }
    b.extend_from_slice(&n32(room.next_seq.len()));
    for (t, s) in &room.next_seq {
        b.extend_from_slice(t);
        b.extend_from_slice(&s.to_be_bytes());
    }
    b.extend_from_slice(&n32(room.held.len()));
    for (t, ps) in &room.held {
        b.extend_from_slice(t);
        b.extend_from_slice(&n32(ps.len()));
        for (s, p) in ps {
            b.extend_from_slice(&s.to_be_bytes());
            b.extend_from_slice(&n32(p.len()));
            b.extend_from_slice(p);
        }
    }
    b.extend_from_slice(&n32(room.rate.len()));
    for (t, st, n) in &room.rate {
        b.extend_from_slice(t);
        b.extend_from_slice(&st.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
    }
    b.extend_from_slice(&n32(room.log.len()));
    for (i, t, s, p) in &room.log {
        b.extend_from_slice(&i.to_be_bytes());
        b.extend_from_slice(t);
        b.extend_from_slice(&s.to_be_bytes());
        b.extend_from_slice(&n32(p.len()));
        b.extend_from_slice(p);
    }
    b
}

const TA: [u8; 16] = [0x0a; 16];
const TB: [u8; 16] = [0x0b; 16];

fn no_log() -> Limits {
    Limits {
        max_log: 0,
        ..LIMITS
    }
}

/// Restore `bytes`, run `f` on the core under `catch_unwind`, and return what it gave
fn after_restore<T>(bytes: &[u8], f: impl FnOnce(&mut AnonymousRelayCore) -> T) -> T {
    let mut core = AnonymousRelayCore::restore(bytes).expect("restore edge state");
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut core)))
        .unwrap_or_else(|_| panic!("panic after restoring an edge state"))
}

fn refused_with(out: &[Out]) -> Option<Refusal> {
    match out {
        [Out::Refused { reason, .. }] => Some(*reason),
        _ => None,
    }
}

/// F1: the last sequence number a publisher may use is `u64::MAX - 1`; `u64::MAX` is refused as
/// `Exhausted` and changes nothing (it is not held back, and no rate is spent).
#[test]
fn a_live_core_refuses_the_sequence_number_past_the_last() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0);
    let before = core.snapshot();
    let out = core.publish(A, ROOM_1, u64::MAX, b"x".to_vec(), 1);
    assert_eq!(refused_with(&out), Some(Refusal::Exhausted));
    assert_eq!(core.snapshot(), before);
    // the one before it is a gap like any other
    let out = core.publish(A, ROOM_1, u64::MAX - 1, b"x".to_vec(), 1);
    assert!(matches!(out.as_slice(), [Out::Resend { .. }]), "{out:?}");
}

/// F1: a room whose log index reached `u64::MAX` refuses a publish as `Exhausted` (no panic, no wrap),
/// and nothing changes.
#[test]
fn restore_at_the_last_log_index_then_publish() {
    let room = RoomSpec {
        next_index: u64::MAX,
        members: vec![(1, TA)],
        ..RoomSpec::default()
    };
    let bytes = build(no_log(), &room);
    let (out, after) = after_restore(&bytes, |c| {
        (
            c.publish(A, ROOM_1, 0, b"x".to_vec(), 1),
            c.snapshot().expect("representable"),
        )
    });
    assert_eq!(refused_with(&out), Some(Refusal::Exhausted));
    assert_eq!(after, bytes);
    // one index left: one payload goes, the next is refused
    let room = RoomSpec {
        next_index: u64::MAX - 1,
        members: vec![(1, TA), (2, TB)],
        ..RoomSpec::default()
    };
    let bytes = build(no_log(), &room);
    let (first, second) = after_restore(&bytes, |c| {
        (
            c.publish(A, ROOM_1, 0, b"x".to_vec(), 1),
            c.publish(A, ROOM_1, 1, b"y".to_vec(), 1),
        )
    });
    assert!(
        matches!(first.as_slice(), [Out::Forward { seq: 0, .. }]),
        "{first:?}"
    );
    assert_eq!(refused_with(&second), Some(Refusal::Exhausted));
}

/// F1: a run of held payloads that would carry the log index past `u64::MAX` is refused as a whole:
/// nothing is forwarded and the held payloads stay.
#[test]
fn restore_near_the_last_log_index_with_held_payloads_then_publish() {
    let room = RoomSpec {
        next_index: u64::MAX - 2,
        members: vec![(1, TA), (2, TB)],
        held: vec![(TA, vec![(1, b"1".to_vec()), (2, b"2".to_vec())])],
        ..RoomSpec::default()
    };
    let bytes = build(no_log(), &room);
    let (out, after) = after_restore(&bytes, |c| {
        (
            c.publish(A, ROOM_1, 0, b"0".to_vec(), 1),
            c.snapshot().expect("representable"),
        )
    });
    assert_eq!(refused_with(&out), Some(Refusal::Exhausted));
    assert_eq!(after, bytes);
}

/// F1: a tag whose sequence reached `u64::MAX` (every number used) takes no more payloads: the last
/// number is a duplicate, the one past it is `Exhausted`; one before the end still forwards.
#[test]
fn restore_at_the_last_sequence_number_then_publish() {
    let room = RoomSpec {
        members: vec![(1, TA), (2, TB)],
        next_seq: vec![(TA, u64::MAX)],
        ..RoomSpec::default()
    };
    let bytes = build(no_log(), &room);
    let (past, dup) = after_restore(&bytes, |c| {
        (
            c.publish(A, ROOM_1, u64::MAX, b"x".to_vec(), 1),
            c.publish(A, ROOM_1, u64::MAX - 1, b"x".to_vec(), 1),
        )
    });
    assert_eq!(refused_with(&past), Some(Refusal::Exhausted));
    assert!(dup.is_empty(), "{dup:?}");
    let room = RoomSpec {
        members: vec![(1, TA), (2, TB)],
        next_seq: vec![(TA, u64::MAX - 1)],
        ..RoomSpec::default()
    };
    let bytes = build(no_log(), &room);
    let (last, after) = after_restore(&bytes, |c| {
        let last = c.publish(A, ROOM_1, u64::MAX - 1, b"x".to_vec(), 1);
        let snap = c.snapshot().expect("representable");
        (last, snap)
    });
    assert!(
        matches!(last.as_slice(), [Out::Forward { seq, .. }] if *seq == u64::MAX - 1),
        "{last:?}"
    );
    // and the state with every number used snapshots and restores
    let again = AnonymousRelayCore::restore(&after).expect("restore");
    assert_eq!(again.snapshot().expect("representable"), after);
}

/// F1: a held payload at `u64::MAX` is a state the core cannot reach (that number is refused).
#[test]
fn held_payload_at_the_number_past_the_last_is_refused() {
    let room = RoomSpec {
        members: vec![(1, TA)],
        held: vec![(TA, vec![(u64::MAX, b"x".to_vec())])],
        ..RoomSpec::default()
    };
    assert_eq!(
        AnonymousRelayCore::restore(&build(no_log(), &room)).err(),
        Some(SnapshotError::Inconsistent)
    );
}

/// F2: a count over its limit is refused when the count is read, before any entry: the same error
/// comes back with the entries cut off right after the count (otherwise it would be `Truncated`).
#[test]
fn a_count_over_its_limit_is_refused_before_its_entries() {
    let good = golden_bytes();
    // (offset of the count, a count one over the golden's limit, what the golden allows)
    let cases = [
        (OFF_ROOM_COUNT, 3u32), // max_rooms 2
        (OFF_MEMBER_COUNT, 4),  // max_members 3
        (210, 4),               // next sequence numbers: max_tags_per_room 3
        (238, 4),               // tags with held payloads: max_tags_per_room 3
        (258, 4),               // held payloads of one tag: max_pending 3
        (275, 4),               // rate windows: max_tags_per_room 3
        (OFF_LOG_COUNT, 5),     // log: min(next index 1, max_log 4) = 1
    ];
    for (at, n) in cases {
        let mut b = good[..at + 4].to_vec();
        put_u32(&mut b, at, n);
        assert_eq!(
            AnonymousRelayCore::restore(&b).err(),
            Some(SnapshotError::Inconsistent),
            "count {n} at {at}"
        );
        // a huge count fails the same way, at once
        put_u32(&mut b, at, u32::MAX);
        assert_eq!(
            AnonymousRelayCore::restore(&b).err(),
            Some(SnapshotError::Inconsistent),
            "count u32::MAX at {at}"
        );
    }
}

/// F2: the tag limit is checked as each entry is read: with max_tags_per_room 1 the second member is
/// refused, so the bytes after it are never read (cut off there, the error is still `Inconsistent`).
#[test]
fn the_tag_limit_is_checked_at_the_entry_that_passes_it() {
    // max_tags_per_room 2: one member tag, one tracked sequence, and a third tag with held payloads.
    // Each list is within the limit on its own; the third distinct tag passes it. Cut the bytes right
    // after that tag: the error is `Inconsistent`, so the decoder stopped at that entry.
    const TC: [u8; 16] = [0x0c; 16];
    let limits = Limits {
        max_tags_per_room: 2,
        ..no_log()
    };
    let room = RoomSpec {
        members: vec![(1, TA)],
        next_seq: vec![(TB, 0)],
        held: vec![(TC, vec![])],
        ..RoomSpec::default()
    };
    let b = build(limits, &room);
    let after_tag_c = b.len() - 4 * 3; // the held tag's own count, then rate and log counts
    assert_eq!(
        AnonymousRelayCore::restore(&b[..after_tag_c]).err(),
        Some(SnapshotError::Inconsistent)
    );
    // with the held tag one already counted, the same state restores
    let room = RoomSpec {
        held: vec![(TB, vec![])],
        ..room
    };
    assert!(AnonymousRelayCore::restore(&build(limits, &room)).is_ok());
    // a connection in more rooms than allowed is refused at that member
    let room_a = RoomSpec {
        members: vec![(1, TA)],
        ..RoomSpec::default()
    };
    let limits = Limits {
        max_rooms_per_conn: 0,
        ..no_log()
    };
    let b = build(limits, &room_a);
    let cut = b.len() - 4 * 4; // right after the member, before the four empty lists
    assert_eq!(
        AnonymousRelayCore::restore(&b[..cut]).err(),
        Some(SnapshotError::Inconsistent)
    );
}

/// F2: a long member list within its limit decodes (the duplicate-tag check does not compare every
/// pair).
#[test]
fn a_long_member_list_within_its_limit_decodes() {
    let n = 100_000usize;
    let limits = Limits {
        max_members: n,
        max_tags_per_room: n,
        ..no_log()
    };
    let members: Vec<(u64, [u8; 16])> = (0..n as u64)
        .map(|i| {
            let mut t = [0u8; 16];
            t[8..].copy_from_slice(&i.to_be_bytes());
            (i, t)
        })
        .collect();
    let room = RoomSpec {
        members,
        ..RoomSpec::default()
    };
    let b = build(limits, &room);
    let core = AnonymousRelayCore::restore(&b).expect("restore");
    assert_eq!(core.tags_held(ROOM_1), n);
}

/// F3: a rate window that counts more publishes than the window allows is refused.
#[test]
fn a_rate_count_over_the_window_limit_is_refused() {
    let mut b = golden_bytes();
    put_u32(&mut b, 303, 6); // max_publishes_per_window 5
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::Inconsistent)
    );
    put_u32(&mut b, 303, 5);
    assert!(AnonymousRelayCore::restore(&b).is_ok());
}

/// F3: a logged payload over max_payload is refused (max_payload 1: the held "x" fits, the logged
/// "hi" does not).
#[test]
fn a_logged_payload_over_max_payload_is_refused() {
    let mut b = golden_bytes();
    put_u64(&mut b, 6, 1);
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::Inconsistent)
    );
}

/// F3: the log against the sequences. Within one tag, each logged payload follows the previous one of
/// that tag (`seq + 1`) or starts the sequence again at 0 (the tag made way and came back); for a tag
/// the room tracks, its latest logged payload is the one just before its next sequence number.
#[test]
fn the_log_agrees_with_the_sequences() {
    let restore = |room: &RoomSpec| AnonymousRelayCore::restore(&build(LIMITS, room)).err();
    let base = RoomSpec {
        next_index: 3,
        members: vec![(1, TA)],
        next_seq: vec![(TA, 3)],
        log: vec![
            (0, TA, 0, b"0".to_vec()),
            (1, TA, 1, b"1".to_vec()),
            (2, TA, 2, b"2".to_vec()),
        ],
        ..RoomSpec::default()
    };
    assert_eq!(restore(&base), None);
    // the golden's log item (seq 0) against its tag's next sequence number 1
    let mut b = golden_bytes();
    put_u64(&mut b, 335, 5);
    assert_eq!(
        AnonymousRelayCore::restore(&b).err(),
        Some(SnapshotError::Inconsistent)
    );
    // the latest item is not next - 1
    let mut r = base.clone();
    r.next_seq = vec![(TA, 4)];
    assert_eq!(restore(&r), Some(SnapshotError::Inconsistent));
    // a gap within the tag
    let mut r = base.clone();
    r.log[1].2 = 5;
    assert_eq!(restore(&r), Some(SnapshotError::Inconsistent));
    // a restart at 0 (the tag made way and came back) is a state the core reaches
    let mut r = base.clone();
    r.next_seq = vec![(TA, 2)];
    r.log = vec![
        (0, TA, 7, b"7".to_vec()),
        (1, TA, 0, b"0".to_vec()),
        (2, TA, 1, b"1".to_vec()),
    ];
    assert_eq!(restore(&r), None);
    // a tag the room does not track may have any logged sequence
    let mut r = base.clone();
    r.next_seq = vec![];
    r.members = vec![(1, TB)];
    r.log = vec![
        (0, TA, 9, b"9".to_vec()),
        (1, TA, 10, b"a".to_vec()),
        (2, TA, 11, b"b".to_vec()),
    ];
    assert_eq!(restore(&r), None);
}

/// F3: the restart case above, reached by a live core (a tag makes way and comes back), restores.
#[test]
fn a_tag_that_made_way_and_came_back_restores() {
    let limits = Limits {
        max_tags_per_room: 2,
        ..LIMITS
    };
    let mut core = AnonymousRelayCore::new(limits);
    core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0);
    core.publish(A, ROOM_1, 0, b"0".to_vec(), 0);
    core.publish(A, ROOM_1, 1, b"1".to_vec(), 0);
    core.close(A);
    assert!(bound_ok(&core.bind(B, ROOM_1, TOKEN_1, TAG_B, 1)));
    assert!(bound_ok(&core.bind(C, ROOM_1, TOKEN_1, TAG_C, 2))); // TAG_A makes way
    core.close(C);
    assert!(bound_ok(&core.bind(D, ROOM_1, TOKEN_1, TAG_A, 3))); // TAG_C makes way
    let out = core.publish(D, ROOM_1, 0, b"again".to_vec(), 4);
    assert!(
        matches!(out.as_slice(), [Out::Forward { seq: 0, .. }]),
        "{out:?}"
    );
    let bytes = core.snapshot().expect("representable");
    let again = AnonymousRelayCore::restore(&bytes).expect("a reachable state restores");
    assert_eq!(again.snapshot().expect("representable"), bytes);
}

fn bound_ok(out: &[Out]) -> bool {
    matches!(out, [Out::Bound { .. }])
}
