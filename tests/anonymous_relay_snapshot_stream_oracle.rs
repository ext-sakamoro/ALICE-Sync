//! Contract tests for the snapshot written and read in pieces
//! ([`AnonymousRelayCore::snapshot_writer`], [`AnonymousRelayCore::restore_from_pieces`]), for a host
//! that stores the snapshot over several steps in bounded pieces while the core keeps running.
//!
//! The oracle is the one-buffer path that already exists and is pinned by its own tests (the
//! hand-written golden bytes and the round trips of `anonymous_relay_snapshot_oracle.rs`):
//!
//! - the pieces, concatenated, are the bytes `snapshot()` returned when the writer was made, for every
//!   piece size and every state size (empty, one room, a full room of maximum payloads);
//! - that stays true when the core is changed between every two pieces (binds, publishes, closes,
//!   leaves, log eviction, a room making way);
//! - every piece but the last is exactly `max` bytes, the last at most `max`;
//! - reading the same bytes in pieces gives what `restore` / `restore_with_limits` give for the whole
//!   slice: the same core (compared by its snapshot) or the same error, for every truncation, every
//!   single-byte change and trailing byte of a small state, and limits a state is over, wherever the
//!   pieces are cut.
//!
//! Each test counts what it compared and fails when it compared nothing.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, Out, RoomAlias, RoomToken, SenderTag, SnapshotError,
    SnapshotWriter,
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

/// The piece sizes the brief names: 1, 7, 4096, 120 KiB, and one larger than any snapshot here
const PIECES: [usize; 5] = [1, 7, 4096, 120 * 1024, 64 << 20];

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("piece sizes are positive")
}

fn bound(out: &[Out]) {
    assert!(
        matches!(out, [Out::Bound { .. }]),
        "the script binds: {out:?}"
    );
}

/// A payload whose bytes depend on its position, so a piece written out of place changes the bytes
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
    // held back: 2 and 3 wait for 1
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

fn states() -> Vec<(&'static str, AnonymousRelayCore)> {
    vec![
        ("empty", AnonymousRelayCore::new(SMALL)),
        ("two rooms", two_rooms()),
        ("full room", full_room()),
    ]
}

/// The writer's pieces concatenated; asserts every piece is exactly `max` bytes but the last, which is
/// at most `max` and ends the snapshot. `between` runs between two pieces. The number of pieces too.
fn drain(
    mut w: SnapshotWriter,
    max: usize,
    mut between: impl FnMut(usize),
) -> (Vec<u8>, usize, SnapshotWriter) {
    let mut all = Vec::new();
    let mut piece = Vec::new();
    let mut n = 0;
    loop {
        piece.clear();
        let more = w.next_chunk(nz(max), &mut piece);
        n += 1;
        assert!(piece.len() <= max, "piece {n} is {} > {max}", piece.len());
        all.extend_from_slice(&piece);
        if !more {
            break;
        }
        assert_eq!(piece.len(), max, "piece {n} before the last is short");
        between(n);
    }
    // a call after the end appends nothing
    piece.clear();
    assert!(!w.next_chunk(nz(max), &mut piece));
    assert!(piece.is_empty());
    (all, n, w)
}

/// Oracle 1: the pieces, concatenated, are `snapshot()` taken when the writer was made, for every
/// state and piece size.
#[test]
fn writer_pieces_concat_equal_snapshot() {
    let mut compared = 0usize;
    let mut full_room_bytes = 0usize;
    for (name, core) in states() {
        let expected = core.snapshot().expect("representable");
        if name == "full room" {
            full_room_bytes = expected.len();
        }
        for max in PIECES {
            let w = core.snapshot_writer().expect("representable");
            let (got, pieces, w) = drain(w, max, |_| {});
            assert!(got == expected, "{name}, pieces of {max}: bytes differ");
            assert_eq!(pieces, expected.len().div_ceil(max).max(1), "{name}, {max}");
            assert!(w.buffer_high_water() <= SnapshotWriter::BUFFER);
            compared += got.len();
        }
    }
    // the full room is the size the brief names: 256 payloads of 65,438 bytes and their fields
    assert!(full_room_bytes > MAX_LOG * MAX_PAYLOAD, "{full_room_bytes}");
    assert!(compared > 0, "compared no byte");
}

/// The next sequence number of `TAG_A` in `ROOM_1` (only `A` publishes while the writer runs, so its
/// latest payload is in the log; 0 when the room is new)
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

/// What the host does to the core between two pieces: every kind of change, in turn
fn mutate(core: &mut AnonymousRelayCore, k: usize, payload_len: usize) {
    let now = 100 + k as u64;
    let other = ConnId::from_raw(1000 + (k - k % 6) as u64);
    match k % 6 {
        // members bind (the room is opened again if it made way)
        0 => {
            core.bind(A, ROOM_1, TOKEN_1, TAG_A, now);
            core.bind(other, ROOM_1, TOKEN_1, TAG_C, now);
        }
        // in-order publishes: forwarded and logged, the log drops its oldest past max_log
        1 | 2 => {
            let seq = next_of_a(core);
            let out = core.publish(A, ROOM_1, seq, payload(payload_len, seq + k as u64), now);
            assert!(
                out.iter().all(|o| !matches!(o, Out::Refused { .. })),
                "the publish goes through: {out:?}"
            );
        }
        3 => core.leave(other, ROOM_1),
        4 => {
            core.close(B);
            core.close(A);
            core.close(C);
        }
        // a connection opens other rooms: an empty room makes way when the room count is reached
        _ => {
            let c = ConnId::from_raw(5000 + k as u64);
            for (i, (room, token)) in [(ROOM_2, TOKEN_2), (ROOM_3, TOKEN_3)].iter().enumerate() {
                core.bind(c, *room, *token, TAG_A, now + i as u64);
            }
            core.close(c);
        }
    }
}

/// Oracle 2: the core is changed between every two pieces (binds, publishes past `max_log`, leaves,
/// closes, rooms making way); the pieces still concatenate to the snapshot taken when the writer was
/// made. The core must have changed (its snapshot differs at the end), or the test proves nothing.
#[test]
fn writer_is_unaffected_by_mutation() {
    let mut compared = 0usize;
    let mut changes = 0usize;
    for (name, core) in states() {
        let sizes: &[usize] = if name == "full room" {
            &[4096, 120 * 1024, 64 << 20]
        } else {
            &PIECES
        };
        let payload_len = if name == "full room" { MAX_PAYLOAD } else { 9 };
        for &max in sizes {
            let mut core = core.clone();
            let expected = core.snapshot().expect("representable");
            let w = core.snapshot_writer().expect("representable");
            // the publishes after the first make the log drop its oldest (log eviction)
            let (got, pieces, _) = drain(w, max, |k| {
                mutate(&mut core, k - 1, payload_len);
                changes += 1;
            });
            assert!(got == expected, "{name}, pieces of {max}: bytes differ");
            if pieces > 6 {
                assert!(
                    core.snapshot().expect("representable") != expected,
                    "{name}, {max}: the core did not change"
                );
            }
            compared += got.len();
        }
    }
    assert!(
        compared > 0 && changes > 0,
        "{compared} bytes, {changes} changes"
    );
}

/// The writer is made from a clone of the core while the original goes on: neither disturbs the other
#[test]
fn writer_from_one_core_does_not_see_another_cores_changes() {
    let mut core = full_room();
    let expected = core.snapshot().expect("representable");
    let w = core.snapshot_writer().expect("representable");
    let mut other = core.clone();
    other.publish(A, ROOM_1, MAX_LOG as u64 + 5, payload(MAX_PAYLOAD, 1), 9);
    core.close(A);
    let (got, _, _) = drain(w, 120 * 1024, |_| {});
    assert!(got == expected);
    assert!(other.snapshot().expect("representable") != expected);
}

/// `bytes` cut into pieces of `size`
fn cut(bytes: &[u8], size: usize) -> impl Iterator<Item = &[u8]> {
    bytes.chunks(size)
}

fn outcome(r: Result<AnonymousRelayCore, SnapshotError>) -> Result<Vec<u8>, SnapshotError> {
    r.map(|c| c.snapshot().expect("a restored core is representable"))
}

/// One comparison of the whole-slice read and the read in pieces of every size
fn same_result(bytes: &[u8], limits: Option<Limits>, seen: &mut BTreeMap<String, usize>) {
    let whole = outcome(match limits {
        None => AnonymousRelayCore::restore(bytes),
        Some(l) => AnonymousRelayCore::restore_with_limits(bytes, l),
    });
    for size in PIECES {
        let pieces = outcome(match limits {
            None => AnonymousRelayCore::restore_from_pieces(cut(bytes, size)),
            Some(l) => AnonymousRelayCore::restore_from_pieces_with_limits(cut(bytes, size), l),
        });
        assert!(
            pieces == whole,
            "pieces of {size}: {:?} != {:?} ({} bytes)",
            pieces.as_ref().err(),
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

/// Oracle 3: the same bytes in pieces of every size give what the whole-slice read gives: the same
/// core, or the same error. Over every truncation, every single-byte change (each byte flipped, and
/// set to 0 and 0xff), a trailing byte, empty pieces between the others, and limits each state is
/// over. The ARLY v1 format has no checksum; a changed byte is caught by the checks the format makes
/// (magic, version, order, counts, the invariants), and those are what is compared here.
#[test]
fn restore_from_pieces_equals_restore() {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let small = two_rooms().snapshot().expect("representable");
    for n in 0..=small.len() {
        same_result(&small[..n], None, &mut seen);
    }
    for i in 0..small.len() {
        for v in [small[i] ^ 0xff, 0, 0xff, small[i].wrapping_add(1)] {
            let mut b = small.clone();
            b[i] = v;
            same_result(&b, None, &mut seen);
        }
    }
    let mut trailing = small.clone();
    trailing.push(0);
    same_result(&trailing, None, &mut seen);
    same_result(&[], None, &mut seen);

    // limits the state is over, and limits it fits (raised)
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
        same_result(&small, Some(l), &mut seen);
    }

    // the full room: whole, truncated inside a payload, a payload over a lowered limit, a log longer
    // than a lowered max_log, a trailing byte
    let full = full_room().snapshot().expect("representable");
    same_result(&full, None, &mut seen);
    same_result(&full[..full.len() - 1000], None, &mut seen);
    same_result(&full[..full.len() / 2], None, &mut seen);
    same_result(
        &full,
        Some(Limits {
            max_payload: MAX_PAYLOAD - 1,
            ..FULL
        }),
        &mut seen,
    );
    same_result(
        &full,
        Some(Limits {
            max_log: MAX_LOG - 1,
            ..FULL
        }),
        &mut seen,
    );
    let mut full_trailing = full.clone();
    full_trailing.push(1);
    same_result(&full_trailing, None, &mut seen);

    // empty pieces anywhere are skipped
    let with_empties: Vec<&[u8]> = small
        .chunks(5)
        .flat_map(|c| [&[][..], c, &[][..]])
        .collect();
    assert_eq!(
        outcome(AnonymousRelayCore::restore_from_pieces(with_empties)),
        Ok(small.clone())
    );

    // every outcome the whole-slice read can give on this platform was compared at least once
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

/// The pieces a writer makes are read back by the reader in the same pieces, and the restored core
/// writes the same bytes
#[test]
fn writer_pieces_restore_in_place() {
    let mut compared = 0usize;
    for (name, core) in states() {
        let expected = core.snapshot().expect("representable");
        for max in [7, 4096, 120 * 1024] {
            let mut w = core.snapshot_writer().expect("representable");
            let mut stored: Vec<Vec<u8>> = Vec::new();
            loop {
                let mut piece = Vec::new();
                let more = w.next_chunk(nz(max), &mut piece);
                stored.push(piece);
                if !more {
                    break;
                }
            }
            let restored = AnonymousRelayCore::restore_from_pieces(&stored).expect(name);
            assert!(
                restored.snapshot().expect("representable") == expected,
                "{name}, {max}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0);
}
