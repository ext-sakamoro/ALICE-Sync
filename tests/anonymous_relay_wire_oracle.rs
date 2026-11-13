//! Contract tests for the anonymous relay's wire frames.
//!
//! - every frame decodes to itself, and decoding is strict (truncated, trailing, unknown tag, oversized);
//! - the two relays refuse each other's frames with a typed error, both ways: a named relay frame given to
//!   the anonymous decoder is `NamedRelayFrame(tag)`, an anonymous frame given to the named decoder is
//!   `UnknownTag(tag)` (its tags are outside the named range);
//! - nothing a client and the anonymous relay send each other names a device: every frame of a session is
//!   searched for 8-byte windows of the members' device ids, and the search is shown to find a planted id.

use alice_sync::anonymous_relay::{Refusal, RoomAlias, RoomToken, SenderTag};
use alice_sync::anonymous_relay_wire::{Frame, WireError, MAX_FRAME, NAMED_RELAY_TAGS};
use alice_sync::relay::{PeerId, RoomId};
use alice_sync::relay_wire::{Frame as NamedFrame, WireError as NamedWireError};

const ROOM: RoomAlias = RoomAlias::from_raw([0x33; 32]);

fn all_frames() -> Vec<Frame> {
    vec![
        Frame::Bind {
            room: ROOM,
            token: RoomToken::from_raw([0x44; 32]),
            tag: SenderTag::from_raw([0x55; 16]),
        },
        Frame::Bound { room: ROOM },
        Frame::Publish {
            room: ROOM,
            seq: 7,
            payload: b"sealed".to_vec(),
        },
        Frame::Forward {
            room: ROOM,
            from: SenderTag::from_raw([0x66; 16]),
            seq: 8,
            payload: b"sealed too".to_vec(),
        },
        Frame::Resend {
            room: ROOM,
            from: 1,
            to_seq: 3,
        },
        Frame::Refused {
            room: ROOM,
            reason: Refusal::BadToken,
        },
        Frame::LogRange {
            room: ROOM,
            from_index: 0,
            to_index: 9,
        },
        Frame::Log {
            room: ROOM,
            first_index: 2,
            items: vec![(2, SenderTag::from_raw([0x77; 16]), 4, b"p".to_vec())],
        },
        Frame::Close { room: ROOM },
    ]
}

#[test]
fn every_frame_decodes_to_itself() {
    for f in all_frames() {
        assert_eq!(Frame::decode(&f.encode()), Ok(f.clone()), "{f:?}");
    }
    for reason in [
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
    ] {
        let f = Frame::Refused { room: ROOM, reason };
        assert_eq!(Frame::decode(&f.encode()), Ok(f));
    }
}

#[test]
fn decoding_is_strict() {
    let bytes = all_frames()[2].encode();
    assert_eq!(
        Frame::decode(&bytes[..bytes.len() - 1]),
        Err(WireError::Truncated)
    );
    let mut long = bytes.clone();
    long.push(0);
    assert_eq!(Frame::decode(&long), Err(WireError::Trailing));
    assert_eq!(Frame::decode(&[0x7f]), Err(WireError::UnknownTag(0x7f)));
    assert_eq!(Frame::decode(&[]), Err(WireError::Truncated));
    // a payload length over the cap is refused before any byte of it is read
    let mut big = vec![bytes[0]];
    big.extend_from_slice(ROOM.as_ref());
    big.extend_from_slice(&0u64.to_be_bytes());
    big.extend_from_slice(&u32::try_from(MAX_FRAME + 1).unwrap().to_be_bytes());
    assert_eq!(Frame::decode(&big), Err(WireError::TooLarge));
}

#[test]
fn each_relay_refuses_the_others_frames_with_a_typed_error() {
    let named = [
        NamedFrame::Bind {
            room: RoomId::from_raw(1),
        },
        NamedFrame::Publish {
            room: RoomId::from_raw(1),
            seq: 0,
            payload: b"x".to_vec(),
        },
        NamedFrame::Close {
            room: RoomId::from_raw(1),
        },
    ];
    for f in &named {
        let bytes = f.encode();
        assert!(NAMED_RELAY_TAGS.contains(&bytes[0]));
        assert_eq!(
            Frame::decode(&bytes),
            Err(WireError::NamedRelayFrame(bytes[0])),
            "{f:?}"
        );
    }
    for f in all_frames() {
        let bytes = f.encode();
        assert!(
            !NAMED_RELAY_TAGS.contains(&bytes[0]),
            "{f:?} uses a named tag"
        );
        assert_eq!(
            NamedFrame::decode(&bytes),
            Err(NamedWireError::UnknownTag(bytes[0])),
            "{f:?}"
        );
    }
}

/// How many 8-byte windows of `needle` appear in `hay`
fn windows_found(hay: &[u8], needle: &[u8]) -> usize {
    needle
        .windows(8)
        .filter(|w| hay.windows(8).any(|x| x == *w))
        .count()
}

#[test]
fn no_frame_of_a_session_names_a_device() {
    let devices = [PeerId::from_raw([0xd1; 16]), PeerId::from_raw([0xd2; 16])];
    let transcript: Vec<u8> = all_frames().iter().flat_map(Frame::encode).collect();
    assert!(
        transcript.len() > 200,
        "searched {} bytes",
        transcript.len()
    );
    for d in &devices {
        assert_eq!(windows_found(&transcript, d.as_ref()), 0, "{d:?}");
    }
    // the search finds an id planted in a payload (payloads are opaque to the relay)
    let planted = Frame::Publish {
        room: ROOM,
        seq: 0,
        payload: devices[0].as_ref().to_vec(),
    }
    .encode();
    assert_eq!(windows_found(&planted, devices[0].as_ref()), 9);
}
