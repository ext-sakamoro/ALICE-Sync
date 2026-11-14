//! Contract tests for [`SnapshotReader`], the snapshot read as its pieces arrive one at a time (for a
//! host whose storage reads are asynchronous: it reads a piece, hands it to the reader, waits for the
//! next read, and so on).
//!
//! There is one decoder: `restore` / `restore_with_limits` feed the whole slice to the same reader, so
//! comparing the reader with `restore` checks only that the result does not depend on where the
//! pieces are cut. The independent oracle is the hand-written golden bytes, the rejection tests and
//! the round trips of `anonymous_relay_snapshot_oracle.rs`, plus the errors written out by hand here.
//! What this file checks:
//!
//! - cut independence: fed the same bytes in pieces of any size, the reader gives what `restore` /
//!   `restore_with_limits` give for the whole slice: the same core (compared by its snapshot) or the
//!   same error;
//! - an error surfaces from the `feed` that hands over the byte `restore` decides it on, and not
//!   before: after the pieces up to any point, `feed` has returned an error exactly when `restore` of
//!   the bytes up to that point gives an error other than [`SnapshotError::Truncated`], and the same
//!   one; `Truncated` only ever comes from `finish`;
//! - a stream over a limit is refused with the error `restore` gives, and no core is returned;
//! - a writer's pieces, fed as they are made while the core changes, restore the snapshot taken when
//!   the writer was made;
//! - after an error the reader keeps returning it.
//!
//! Each test counts what it compared and fails when it compared nothing.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, Out, RoomAlias, RoomToken, SenderTag, SnapshotError,
    SnapshotReader,
};

const ROOM_1: RoomAlias = RoomAlias::from_raw([0x11; 32]);
const ROOM_2: RoomAlias = RoomAlias::from_raw([0x22; 32]);
const ROOM_3: RoomAlias = RoomAlias::from_raw([0x33; 32]);
const TOKEN_1: RoomToken = RoomToken::from_raw([0xa1; 32]);
const TOKEN_2: RoomToken = RoomToken::from_raw([0xa2; 32]);
const TOKEN_3: RoomToken = RoomToken::from_raw([0xa3; 32]);
const A: ConnId = ConnId::from_raw(1);
const B: ConnId = ConnId::from_raw(2);
const C: ConnId = ConnId::from_raw(3);
const TAG_A: SenderTag = SenderTag::from_raw([0x0a; 16]);
const TAG_B: SenderTag = SenderTag::from_raw([0x0b; 16]);
const TAG_C: SenderTag = SenderTag::from_raw([0x0c; 16]);

/// A full room: 256 logged payloads of 65,438 bytes (a snapshot of about 16.7 MB)
const MAX_PAYLOAD: usize = 65_438;
const MAX_LOG: usize = 256;

const SMALL: Limits = Limits {
    max_payload: 16,
    max_log: 4,
    max_pending: 3,
    max_rooms: 2,
    max_members: 3,
    max_publishes_per_window: 100,
    window: 10,
    max_rooms_per_conn: 2,
    max_tags_per_room: 4,
};

const FULL: Limits = Limits {
    max_payload: MAX_PAYLOAD,
    max_log: MAX_LOG,
    max_pending: 4,
    max_rooms: 3,
    max_members: 8,
    max_publishes_per_window: u32::MAX,
    window: 1,
    max_rooms_per_conn: 3,
    max_tags_per_room: 8,
};

/// The piece sizes: 1, 7, 4096, 120 KiB, and one larger than any snapshot here (the whole)
const PIECES: [usize; 5] = [1, 7, 4096, 120 * 1024, 64 << 20];

fn bound(out: &[Out]) {
    assert!(
        matches!(out, [Out::Bound { .. }]),
        "the script binds: {out:?}"
    );
}

/// A payload whose bytes depend on its position, so a byte read out of place changes the state
fn payload(n: usize, salt: u64) -> Vec<u8> {
    (0..n)
        .map(|j| (salt.wrapping_mul(31).wrapping_add(j as u64 * 7) % 251) as u8)
        .collect()
}

/// Two rooms: members, sequence numbers, held-back payloads (a resend gap), rate windows and a log
/// that has dropped its oldest payloads
fn two_rooms() -> AnonymousRelayCore {
    let mut core = AnonymousRelayCore::new(SMALL);
    bound(&core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0));
    bound(&core.bind(B, ROOM_1, TOKEN_1, TAG_B, 1));
    for s in 0..6 {
        core.publish(A, ROOM_1, s, payload(5 + s as usize, s), 2 + s);
    }
    // held back: 2 and 3 wait for 1 (3 is empty: a zero-length payload)
    core.publish(B, ROOM_1, 0, payload(3, 90), 9);
    core.publish(B, ROOM_1, 2, payload(16, 92), 9);
    core.publish(B, ROOM_1, 3, b"".to_vec(), 9);
    bound(&core.bind(C, ROOM_2, TOKEN_2, TAG_C, 10));
    core.publish(C, ROOM_2, 0, payload(1, 7), 11);
    core
}

/// One room at `max_log` of `max_payload` payloads (more were published, so the log has dropped its
/// oldest), and payloads held back
fn full_room() -> AnonymousRelayCore {
    let mut core = AnonymousRelayCore::new(FULL);
    bound(&core.bind(A, ROOM_1, TOKEN_1, TAG_A, 0));
    bound(&core.bind(B, ROOM_1, TOKEN_1, TAG_B, 0));
    for s in 0..(MAX_LOG as u64 + 5) {
        core.publish(A, ROOM_1, s, payload(MAX_PAYLOAD, s), 1);
    }
    core.publish(B, ROOM_1, 2, payload(MAX_PAYLOAD, 1000), 2);
    core.publish(B, ROOM_1, 4, payload(MAX_PAYLOAD, 1001), 2);
    core
}

fn outcome(r: Result<AnonymousRelayCore, SnapshotError>) -> Result<Vec<u8>, SnapshotError> {
    r.map(|c| c.snapshot().expect("a restored core is representable"))
}

fn reader(limits: Option<Limits>) -> SnapshotReader {
    limits.map_or_else(SnapshotReader::new, SnapshotReader::with_limits)
}

fn restore(bytes: &[u8], limits: Option<Limits>) -> Result<AnonymousRelayCore, SnapshotError> {
    match limits {
        None => AnonymousRelayCore::restore(bytes),
        Some(l) => AnonymousRelayCore::restore_with_limits(bytes, l),
    }
}

/// What `restore` of a prefix says about the reader after it was fed that prefix: the error the
/// reader must have returned by then, or `None` when it must still be reading (more bytes may come)
fn decided(prefix: &[u8], limits: Option<Limits>) -> Option<SnapshotError> {
    match restore(prefix, limits) {
        Err(SnapshotError::Truncated) | Ok(_) => None,
        Err(e) => Some(e),
    }
}

/// Feed `bytes` in pieces of `size` and finish; the outcome. With `when`, also check after every piece
/// that the reader has returned an error exactly when `restore` of the bytes so far decides one (the
/// same error), and that `finish` after a `feed` error gives that error.
fn feed_all(
    bytes: &[u8],
    size: usize,
    limits: Option<Limits>,
    when: bool,
) -> Result<Vec<u8>, SnapshotError> {
    let mut r = reader(limits);
    let mut fed = 0;
    for piece in bytes.chunks(size) {
        fed += piece.len();
        let got = r.feed(piece).err();
        if when {
            assert_eq!(
                got,
                decided(&bytes[..fed], limits),
                "after {fed} of {} bytes in pieces of {size}",
                bytes.len()
            );
        }
        if let Some(e) = got {
            // finish after an error gives that error, and no core
            assert_eq!(r.finish().err(), Some(e));
            return Err(e);
        }
    }
    let out = outcome(r.finish());
    if let Err(e) = out {
        // every check but the end of the bytes is made as the bytes arrive
        assert_eq!(e, SnapshotError::Truncated, "{e:?} came from finish");
    }
    out
}

/// One comparison of the whole-slice read and the reader fed pieces of every size
fn same_result(
    bytes: &[u8],
    limits: Option<Limits>,
    when: bool,
    seen: &mut BTreeMap<String, usize>,
) {
    let whole = outcome(restore(bytes, limits));
    for size in PIECES {
        // the step-by-step check costs one restore per piece: on the small inputs only
        let pushed = feed_all(bytes, size, limits, when && size <= 7);
        assert!(
            pushed == whole,
            "pieces of {size}: {:?} != {:?} ({} bytes)",
            pushed.as_ref().err(),
            whole.as_ref().err(),
            bytes.len()
        );
    }
    let key = match &whole {
        Ok(_) => "ok".to_owned(),
        Err(SnapshotError::UnknownVersion(_)) => "UnknownVersion".to_owned(),
        Err(e) => format!("{e:?}"),
    };
    *seen.entry(key).or_default() += 1;
}

/// Oracle 1: the reader fed the same bytes in pieces of every size gives what the whole-slice read
/// gives, the same core or the same error, over the corpus of the pieces oracle: every truncation,
/// every single-byte change (each byte flipped, set to 0 and 0xff, and incremented), a trailing byte,
/// empty input, limits each state is over and raised limits, and the full room whole, truncated and
/// over lowered limits. On the small state it also checks after every piece of 1 and 7 bytes that an
/// error surfaces exactly when `restore` of the bytes so far decides it.
#[test]
fn push_equals_restore() {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    same_result(
        &AnonymousRelayCore::new(SMALL).snapshot().expect("ok"),
        None,
        true,
        &mut seen,
    );
    let small = two_rooms().snapshot().expect("representable");
    for n in 0..=small.len() {
        same_result(&small[..n], None, true, &mut seen);
    }
    for i in 0..small.len() {
        for v in [small[i] ^ 0xff, 0, 0xff, small[i].wrapping_add(1)] {
            let mut b = small.clone();
            b[i] = v;
            same_result(&b, None, true, &mut seen);
        }
    }
    let mut trailing = small.clone();
    trailing.push(0);
    same_result(&trailing, None, true, &mut seen);
    same_result(&[], None, true, &mut seen);

    let over = [
        Limits {
            max_payload: 15,
            ..SMALL
        },
        Limits {
            max_log: 3,
            ..SMALL
        },
        Limits {
            max_pending: 1,
            ..SMALL
        },
        Limits {
            max_rooms: 1,
            ..SMALL
        },
        Limits {
            max_members: 1,
            ..SMALL
        },
        Limits {
            max_publishes_per_window: 2,
            ..SMALL
        },
        Limits {
            max_rooms_per_conn: 0,
            ..SMALL
        },
        Limits {
            max_tags_per_room: 1,
            ..SMALL
        },
        Limits {
            max_log: 400,
            max_payload: 1 << 20,
            ..SMALL
        },
    ];
    for l in over {
        same_result(&small, Some(l), true, &mut seen);
    }

    let full = full_room().snapshot().expect("representable");
    same_result(&full, None, false, &mut seen);
    same_result(&full[..full.len() - 1000], None, false, &mut seen);
    same_result(&full[..full.len() / 2], None, false, &mut seen);
    for l in [
        Limits {
            max_payload: MAX_PAYLOAD - 1,
            ..FULL
        },
        Limits {
            max_log: MAX_LOG - 1,
            ..FULL
        },
        Limits {
            max_log: 4 * MAX_LOG,
            max_payload: 2 * MAX_PAYLOAD,
            ..FULL
        },
    ] {
        same_result(&full, Some(l), false, &mut seen);
    }
    let mut full_trailing = full.clone();
    full_trailing.push(1);
    same_result(&full_trailing, None, false, &mut seen);

    for key in [
        "ok",
        "BadMagic",
        "UnknownVersion",
        "Truncated",
        "Trailing",
        "NotCanonical",
        "Inconsistent",
    ] {
        assert!(
            seen.get(key).copied().unwrap_or(0) > 0,
            "no case gave {key}: {seen:?}"
        );
    }
    let total: usize = seen.values().sum();
    assert!(total > 1000, "compared {total} cases");
}

/// Feed every piece; the error and whether it came from `feed` (not `finish`)
fn refused(bytes: &[u8], limits: Limits, size: usize) -> (Option<SnapshotError>, bool) {
    let mut r = SnapshotReader::with_limits(limits);
    for piece in bytes.chunks(size) {
        if let Err(e) = r.feed(piece) {
            assert_eq!(r.finish().err(), Some(e), "finish after the error");
            return (Some(e), true);
        }
    }
    (r.finish().err(), false)
}

/// Oracle 2: a stream over a limit (the log longer than a lowered `max_log`, a payload over a lowered
/// `max_payload`, more rooms than `max_rooms`) is refused with `Inconsistent`, the error `restore`
/// gives, from the `feed` that hands over the offending field, and no core is returned. The expected
/// error is written out, not taken from `restore`, so a change shared by both reads is caught too.
#[test]
fn finish_refuses_over_limit() {
    let full = full_room().snapshot().expect("representable");
    let small = two_rooms().snapshot().expect("representable");
    let cases = [
        (
            "max_log lowered",
            &full,
            Limits {
                max_log: MAX_LOG - 1,
                ..FULL
            },
        ),
        (
            "max_payload lowered",
            &full,
            Limits {
                max_payload: MAX_PAYLOAD - 1,
                ..FULL
            },
        ),
        (
            "max_rooms lowered",
            &small,
            Limits {
                max_rooms: 1,
                ..SMALL
            },
        ),
        (
            "max_log lowered (small)",
            &small,
            Limits {
                max_log: 3,
                ..SMALL
            },
        ),
    ];
    let mut compared = 0;
    for (name, bytes, limits) in cases {
        assert_eq!(
            AnonymousRelayCore::restore_with_limits(bytes, limits).err(),
            Some(SnapshotError::Inconsistent),
            "{name}: restore"
        );
        for size in [1, 4096, 120 * 1024, 64 << 20] {
            if size == 1 && bytes.len() > 100_000 {
                continue;
            }
            let (err, from_feed) = refused(bytes, limits, size);
            assert_eq!(err, Some(SnapshotError::Inconsistent), "{name}, {size}");
            assert!(from_feed, "{name}, {size}: refused only by finish");
            compared += 1;
        }
    }
    // the room count over max_rooms is refused by the feed that completes the count, before any room
    let head = 4 + 2 + 68 + 4;
    let mut r = SnapshotReader::with_limits(Limits {
        max_rooms: 1,
        ..SMALL
    });
    assert_eq!(r.feed(&small[..head - 1]), Ok(()));
    assert_eq!(
        r.feed(&small[head - 1..head]),
        Err(SnapshotError::Inconsistent)
    );
    assert!(compared > 0);
}

/// The next sequence number of `TAG_A` in `ROOM_1` (only `A` publishes while the writer runs)
fn next_of_a(core: &AnonymousRelayCore) -> u64 {
    core.log_range(A, ROOM_1, 0, u64::MAX)
        .ok()
        .and_then(|r| {
            r.items
                .iter()
                .rev()
                .find(|i| i.from == TAG_A)
                .map(|i| i.seq + 1)
        })
        .unwrap_or(0)
}

/// What the host does to the core between two pieces: binds, publishes past `max_log`, leaves,
/// closes, rooms making way
fn mutate(core: &mut AnonymousRelayCore, k: usize, payload_len: usize) {
    let now = 100 + k as u64;
    let other = ConnId::from_raw(1000 + (k - k % 6) as u64);
    match k % 6 {
        0 => {
            core.bind(A, ROOM_1, TOKEN_1, TAG_A, now);
            core.bind(other, ROOM_1, TOKEN_1, TAG_C, now);
        }
        1 | 2 => {
            let seq = next_of_a(core);
            core.publish(A, ROOM_1, seq, payload(payload_len, seq + k as u64), now);
        }
        3 => core.leave(other, ROOM_1),
        4 => {
            core.close(B);
            core.close(A);
            core.close(C);
        }
        _ => {
            let c = ConnId::from_raw(5000 + k as u64);
            for (i, (room, token)) in [(ROOM_2, TOKEN_2), (ROOM_3, TOKEN_3)].iter().enumerate() {
                core.bind(c, *room, *token, TAG_A, now + i as u64);
            }
            core.close(c);
        }
    }
}

/// Oracle 3: a writer's pieces handed to the reader one by one as they are made, the core changing
/// between every two pieces, restore the state at the moment the writer was made. The core must have
/// changed, or the test proves nothing.
#[test]
fn writer_to_push_round_trip() {
    let mut compared = 0usize;
    let mut changed = 0usize;
    for (name, core, payload_len) in [
        ("empty", AnonymousRelayCore::new(SMALL), 9),
        ("two rooms", two_rooms(), 9),
        ("full room", full_room(), MAX_PAYLOAD),
    ] {
        for max in [7, 4096, 120 * 1024] {
            if name == "full room" && max == 7 {
                continue;
            }
            let mut core = core.clone();
            let expected = core.snapshot().expect("representable");
            let mut w = core.snapshot_writer().expect("representable");
            let mut r = SnapshotReader::new();
            let mut piece = Vec::new();
            let mut k = 0;
            loop {
                piece.clear();
                let more = w.next_chunk(NonZeroUsize::new(max).expect("positive"), &mut piece);
                r.feed(&piece).expect(name);
                if !more {
                    break;
                }
                mutate(&mut core, k, payload_len);
                k += 1;
            }
            let restored = r.finish().expect(name);
            assert!(
                restored.snapshot().expect("representable") == expected,
                "{name}, {max}"
            );
            if k > 6 {
                assert!(
                    core.snapshot().expect("representable") != expected,
                    "{name}"
                );
                changed += 1;
            }
            compared += 1;
        }
    }
    assert!(compared > 0 && changed > 0, "{compared} {changed}");
}

/// Oracle 5: after an error the reader is spent: every later `feed` (with bytes or empty) returns
/// that error, and `finish` returns it. Covers an error from the head (bad magic), one from a room
/// (out of order), the end of the snapshot followed by more bytes (trailing), and `finish` on a
/// stream that ends early.
#[test]
fn poisoned_after_error() {
    let small = two_rooms().snapshot().expect("representable");

    let mut bad = small.clone();
    bad[0] ^= 1;
    let mut r = SnapshotReader::new();
    assert_eq!(r.feed(&bad[..10]), Err(SnapshotError::BadMagic));
    assert_eq!(r.feed(&small[..10]), Err(SnapshotError::BadMagic));
    assert_eq!(r.feed(&[]), Err(SnapshotError::BadMagic));
    assert_eq!(r.feed(&small[10..]), Err(SnapshotError::BadMagic));
    assert_eq!(r.finish().err(), Some(SnapshotError::BadMagic));

    // the snapshot then one more byte: Trailing, from the feed that brings it, and kept
    let mut r = SnapshotReader::new();
    assert_eq!(r.feed(&small), Ok(()));
    assert_eq!(r.feed(&[]), Ok(()));
    assert_eq!(r.feed(&[0]), Err(SnapshotError::Trailing));
    assert_eq!(r.feed(&[]), Err(SnapshotError::Trailing));
    assert_eq!(r.finish().err(), Some(SnapshotError::Trailing));

    // trailing bytes in the same piece as the end of the snapshot
    let mut with_tail = small.clone();
    with_tail.extend_from_slice(&[1, 2, 3]);
    let mut r = SnapshotReader::new();
    assert_eq!(r.feed(&with_tail), Err(SnapshotError::Trailing));
    assert_eq!(r.finish().err(), Some(SnapshotError::Trailing));

    // the second room's alias equal to the first's: NotCanonical as its last byte arrives
    let mut r = SnapshotReader::with_limits(SMALL);
    let first_room_alias = 4 + 2 + 68 + 4;
    let mut same = small.clone();
    let second = find_second_alias(&small, first_room_alias);
    same.copy_within(first_room_alias..first_room_alias + 32, second);
    assert_eq!(r.feed(&same[..second + 31]), Ok(()));
    assert_eq!(
        r.feed(&same[second + 31..second + 32]),
        Err(SnapshotError::NotCanonical)
    );
    assert_eq!(
        r.feed(&same[second + 32..]),
        Err(SnapshotError::NotCanonical)
    );
    assert_eq!(r.finish().err(), Some(SnapshotError::NotCanonical));

    // ends early: finish says Truncated; nothing fed at all is Truncated too
    let mut r = SnapshotReader::new();
    assert_eq!(r.feed(&small[..small.len() - 1]), Ok(()));
    assert_eq!(r.finish().err(), Some(SnapshotError::Truncated));
    assert_eq!(
        SnapshotReader::new().finish().err(),
        Some(SnapshotError::Truncated)
    );
    assert_eq!(
        SnapshotReader::default().finish().err(),
        Some(SnapshotError::Truncated)
    );
}

/// The offset of the second room's alias: the bytes `[0x22; 32]` after the first alias
fn find_second_alias(bytes: &[u8], from: usize) -> usize {
    let at = bytes[from + 32..]
        .windows(32)
        .position(|w| w == [0x22; 32])
        .expect("the second room is in the snapshot");
    from + 32 + at
}
