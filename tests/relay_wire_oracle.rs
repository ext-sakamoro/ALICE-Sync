//! Contract tests for the relay's wire frames: every frame round-trips, the byte layout of a frame
//! is fixed (the spec for other implementations), and malformed bytes are refused without a panic
//! (truncated at any byte, a trailing byte, an unknown tag or refusal, a length over the cap).
//!
//! Layout: one tag byte, then the fields in order; integers big-endian fixed width; a peer id is
//! its 32 bytes; a payload is a u32 length then the bytes (at most `MAX_FRAME`).

use alice_sync::relay::{PeerId, Refusal, RoomId};
use alice_sync::relay_wire::{Frame, WireError, MAX_FRAME};

const A: PeerId = PeerId::from_raw([0xa1; 32]);
const R: RoomId = RoomId::from_raw(0x0102_0304_0506_0708);

fn all() -> Vec<Frame> {
    vec![
        Frame::Bind { room: R },
        Frame::Publish {
            room: R,
            seq: 7,
            payload: b"\x03sealed".to_vec(),
        },
        Frame::Forward {
            room: R,
            from: A,
            seq: u64::MAX,
            payload: vec![],
        },
        Frame::Resend {
            room: R,
            from: 1,
            to_seq: 2,
        },
        Frame::Refused {
            room: R,
            reason: Refusal::TooFarAhead,
        },
        Frame::LogRange {
            room: R,
            from_index: 4,
            to_index: 6,
        },
        Frame::Log {
            room: R,
            first_index: 2,
            items: vec![(4, A, 9, vec![1, 2, 3]), (5, A, 10, vec![])],
        },
        Frame::Close { room: R },
    ]
}

/// oracle: every frame decodes back to itself.
#[test]
fn every_frame_round_trips() {
    for f in all() {
        let b = f.encode();
        assert_eq!(Frame::decode(&b), Ok(f.clone()), "{f:?}");
    }
}

/// oracle: the layout of a Publish is tag 2, room (8 BE), seq (8 BE), length (4 BE), bytes.
#[test]
fn publish_layout_is_fixed() {
    let b = Frame::Publish {
        room: R,
        seq: 7,
        payload: vec![0xee, 0xff],
    }
    .encode();
    let mut want = vec![2];
    want.extend_from_slice(&0x0102_0304_0506_0708u64.to_be_bytes());
    want.extend_from_slice(&7u64.to_be_bytes());
    want.extend_from_slice(&2u32.to_be_bytes());
    want.extend_from_slice(&[0xee, 0xff]);
    assert_eq!(b, want);
    // the tags in declaration order are 1..=8
    let tags: Vec<u8> = all().iter().map(|f| f.encode()[0]).collect();
    assert_eq!(tags, (1..=8).collect::<Vec<u8>>());
}

/// oracle: every proper prefix of every frame is refused as truncated, and a trailing byte is
/// refused.
#[test]
fn truncation_and_trailing_bytes_are_refused() {
    let mut compared = 0;
    for f in all() {
        let b = f.encode();
        for n in 0..b.len() {
            assert_eq!(
                Frame::decode(&b[..n]),
                Err(WireError::Truncated),
                "{f:?} at {n}"
            );
            compared += 1;
        }
        let mut longer = b.clone();
        longer.push(0);
        assert_eq!(Frame::decode(&longer), Err(WireError::Trailing), "{f:?}");
    }
    assert!(compared > 100, "compared {compared}");
}

/// oracle: an unknown tag, an unknown refusal and a length over the cap are refused.
#[test]
fn unknown_values_and_oversize_are_refused() {
    assert_eq!(Frame::decode(&[0]), Err(WireError::UnknownTag(0)));
    assert_eq!(Frame::decode(&[9]), Err(WireError::UnknownTag(9)));
    let mut refused = Frame::Refused {
        room: R,
        reason: Refusal::NoRoom,
    }
    .encode();
    let last = refused.len() - 1;
    refused[last] = 99;
    assert_eq!(Frame::decode(&refused), Err(WireError::UnknownRefusal(99)));
    let mut big = vec![2];
    big.extend_from_slice(&0u64.to_be_bytes());
    big.extend_from_slice(&0u64.to_be_bytes());
    big.extend_from_slice(&(u32::try_from(MAX_FRAME).unwrap() + 1).to_be_bytes());
    assert_eq!(Frame::decode(&big), Err(WireError::TooLarge));
    // a Log whose item count claims more than the bytes hold is truncated, not allocated
    let mut log = vec![7];
    log.extend_from_slice(&0u64.to_be_bytes());
    log.extend_from_slice(&0u64.to_be_bytes());
    log.extend_from_slice(&u32::MAX.to_be_bytes());
    assert_eq!(Frame::decode(&log), Err(WireError::Truncated));
}

/// oracle (degenerate input): arbitrary bytes decode to a frame or an error, never a panic; the
/// frames that decode re-encode to the same bytes.
#[test]
fn arbitrary_bytes_never_panic() {
    let mut x: u32 = 0x9e37_79b9;
    let mut decoded = 0;
    for len in 0..64usize {
        for _ in 0..200 {
            let mut b = Vec::with_capacity(len);
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                b.push((x >> 24) as u8);
            }
            if let Some(t) = b.first_mut() {
                *t = 1 + *t % 8;
            }
            if let Ok(f) = Frame::decode(&b) {
                assert_eq!(f.encode(), b);
                decoded += 1;
            }
        }
    }
    assert!(decoded > 0, "some random frames decode");
}
