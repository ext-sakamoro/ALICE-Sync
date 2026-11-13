//! Contract tests for the anonymous room relay core (sans-IO): rooms named by an alias, entered with the
//! room's token, publishers known only by a sender tag they chose. Nothing the relay takes or gives names a
//! device.
//!
//! - members of one room (the same alias and token) meet; each payload goes byte for byte to every other
//!   connection in the room, tagged with its publisher's sender tag;
//! - a wrong token is refused and changes nothing: no forward, no member, no log entry;
//! - two rooms (two epochs) share nothing: a connection bound in one cannot publish to or read the other, and
//!   the token of one does not open the other;
//! - per sender tag, payloads leave in sequence order (a gap is held back and a resend asked, a duplicate is
//!   dropped), the same as the named relay;
//! - a tag held by another connection is refused; the same connection may bind again;
//! - limits: payload size, rooms (an empty room that has been idle longest makes way), members per room, and
//!   publishes per sender tag per window.

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, Out, Refusal, RoomAlias, RoomToken, SenderTag,
};

const EPOCH_1: RoomAlias = RoomAlias::from_raw([0x11; 32]);
const EPOCH_2: RoomAlias = RoomAlias::from_raw([0x22; 32]);
const TOKEN_1: RoomToken = RoomToken::from_raw([0xa1; 32]);
const TOKEN_2: RoomToken = RoomToken::from_raw([0xa2; 32]);
const ALICE: ConnId = ConnId::from_raw(1);
const BOB: ConnId = ConnId::from_raw(2);
const CAROL: ConnId = ConnId::from_raw(3);
const TAG_A: SenderTag = SenderTag::from_raw([0x0a; 16]);
const TAG_B: SenderTag = SenderTag::from_raw([0x0b; 16]);
const TAG_C: SenderTag = SenderTag::from_raw([0x0c; 16]);

const LIMITS: Limits = Limits {
    max_payload: 64,
    max_log: 8,
    max_pending: 4,
    max_rooms: 4,
    max_members: 3,
    max_publishes_per_window: 100,
    window: 60,
};

fn bound(out: &[Out], conn: ConnId) -> bool {
    out.iter()
        .any(|o| matches!(o, Out::Bound { to, .. } if *to == conn))
}

fn refused(out: &[Out]) -> Option<Refusal> {
    out.iter().find_map(|o| match o {
        Out::Refused { reason, .. } => Some(*reason),
        _ => None,
    })
}

fn forwards(out: &[Out]) -> Vec<(ConnId, SenderTag, u64, Vec<u8>)> {
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

/// alice and bob in the epoch-1 room
fn two() -> AnonymousRelayCore {
    let mut r = AnonymousRelayCore::new(LIMITS);
    assert!(bound(&r.bind(ALICE, EPOCH_1, TOKEN_1, TAG_A, 0), ALICE));
    assert!(bound(&r.bind(BOB, EPOCH_1, TOKEN_1, TAG_B, 0), BOB));
    r
}

#[test]
fn members_of_one_room_meet_and_payloads_go_byte_for_byte_to_the_others() {
    let mut r = two();
    assert!(bound(&r.bind(CAROL, EPOCH_1, TOKEN_1, TAG_C, 0), CAROL));
    let out = r.publish(ALICE, EPOCH_1, 0, b"sealed".to_vec(), 1);
    let mut f = forwards(&out);
    f.sort_by_key(|x| x.0);
    assert_eq!(
        f,
        vec![
            (BOB, TAG_A, 0, b"sealed".to_vec()),
            (CAROL, TAG_A, 0, b"sealed".to_vec())
        ]
    );
    let log = r.log_range(CAROL, EPOCH_1, 0, 10).unwrap();
    assert_eq!(log.items.len(), 1);
    assert_eq!(log.items[0].from, TAG_A);
}

#[test]
fn a_wrong_token_is_refused_and_changes_nothing() {
    let mut r = two();
    let out = r.bind(CAROL, EPOCH_1, TOKEN_2, TAG_C, 0);
    assert_eq!(refused(&out), Some(Refusal::BadToken));
    assert!(!bound(&out, CAROL));
    // not a member: cannot publish or read
    assert_eq!(
        refused(&r.publish(CAROL, EPOCH_1, 0, b"x".to_vec(), 1)),
        Some(Refusal::NotBound)
    );
    assert_eq!(r.log_range(CAROL, EPOCH_1, 0, 10), Err(Refusal::NotBound));
    // and the members' traffic is as before
    let out = r.publish(ALICE, EPOCH_1, 0, b"m".to_vec(), 1);
    assert_eq!(forwards(&out), vec![(BOB, TAG_A, 0, b"m".to_vec())]);
}

#[test]
fn two_rooms_share_nothing() {
    let mut r = two();
    // the token of epoch 1 does not open epoch 2 once epoch 2 is held under its own token
    assert!(bound(&r.bind(CAROL, EPOCH_2, TOKEN_2, TAG_C, 0), CAROL));
    assert_eq!(
        refused(&r.bind(ALICE, EPOCH_2, TOKEN_1, TAG_A, 0)),
        Some(Refusal::BadToken)
    );
    // a connection bound in epoch 1 neither publishes to nor reads epoch 2
    assert_eq!(
        refused(&r.publish(BOB, EPOCH_2, 0, b"x".to_vec(), 1)),
        Some(Refusal::NotBound)
    );
    assert_eq!(r.log_range(BOB, EPOCH_2, 0, 10), Err(Refusal::NotBound));
    // epoch 2's publish reaches nobody of epoch 1
    assert!(forwards(&r.publish(CAROL, EPOCH_2, 0, b"y".to_vec(), 1)).is_empty());
}

#[test]
fn a_late_joiner_of_a_later_epoch_cannot_enter_or_read_the_earlier_room() {
    let mut r = two();
    r.publish(ALICE, EPOCH_1, 0, b"before".to_vec(), 1);
    // the late joiner holds only the later epoch's token
    assert_eq!(
        refused(&r.bind(CAROL, EPOCH_1, TOKEN_2, TAG_C, 2)),
        Some(Refusal::BadToken)
    );
    assert_eq!(r.log_range(CAROL, EPOCH_1, 0, 10), Err(Refusal::NotBound));
}

#[test]
fn per_sender_tag_payloads_leave_in_order() {
    let mut r = two();
    let out = r.publish(ALICE, EPOCH_1, 1, b"one".to_vec(), 1);
    assert!(forwards(&out).is_empty(), "held back");
    assert!(out.iter().any(|o| matches!(
        o,
        Out::Resend { to, from: 0, to_seq: 0, .. } if *to == ALICE
    )));
    let out = r.publish(ALICE, EPOCH_1, 0, b"zero".to_vec(), 1);
    assert_eq!(
        forwards(&out),
        vec![
            (BOB, TAG_A, 0, b"zero".to_vec()),
            (BOB, TAG_A, 1, b"one".to_vec())
        ]
    );
    // a duplicate is dropped
    assert!(r.publish(ALICE, EPOCH_1, 0, b"zero".to_vec(), 1).is_empty());
}

#[test]
fn a_reconnect_with_the_same_tag_continues_the_sequence() {
    let mut r = two();
    r.publish(ALICE, EPOCH_1, 0, b"a0".to_vec(), 1);
    r.close(ALICE);
    let again = ConnId::from_raw(41);
    assert!(bound(&r.bind(again, EPOCH_1, TOKEN_1, TAG_A, 2), again));
    assert_eq!(
        forwards(&r.publish(again, EPOCH_1, 1, b"a1".to_vec(), 2)),
        vec![(BOB, TAG_A, 1, b"a1".to_vec())]
    );
}

#[test]
fn a_tag_held_by_another_connection_is_refused() {
    let mut r = two();
    assert_eq!(
        refused(&r.bind(CAROL, EPOCH_1, TOKEN_1, TAG_A, 0)),
        Some(Refusal::TagInUse)
    );
    // the holder may bind again
    assert!(bound(&r.bind(ALICE, EPOCH_1, TOKEN_1, TAG_A, 0), ALICE));
}

#[test]
fn a_closed_connection_gets_nothing_more() {
    let mut r = two();
    r.close(BOB);
    assert!(forwards(&r.publish(ALICE, EPOCH_1, 0, b"x".to_vec(), 1)).is_empty());
    assert_eq!(r.log_range(BOB, EPOCH_1, 0, 10), Err(Refusal::NotBound));
}

#[test]
fn the_limits_hold() {
    let mut r = two();
    assert_eq!(
        refused(&r.publish(ALICE, EPOCH_1, 0, vec![0; 65], 1)),
        Some(Refusal::TooLarge)
    );
    // members per room
    assert!(bound(&r.bind(CAROL, EPOCH_1, TOKEN_1, TAG_C, 0), CAROL));
    assert_eq!(
        refused(&r.bind(
            ConnId::from_raw(4),
            EPOCH_1,
            TOKEN_1,
            SenderTag::from_raw([4; 16]),
            0
        )),
        Some(Refusal::RoomFull)
    );
}

#[test]
fn publishes_per_sender_tag_are_limited_per_window() {
    let mut r = AnonymousRelayCore::new(Limits {
        max_publishes_per_window: 2,
        ..LIMITS
    });
    r.bind(ALICE, EPOCH_1, TOKEN_1, TAG_A, 0);
    r.bind(BOB, EPOCH_1, TOKEN_1, TAG_B, 0);
    assert_eq!(
        forwards(&r.publish(ALICE, EPOCH_1, 0, b"0".to_vec(), 10)).len(),
        1
    );
    assert_eq!(
        forwards(&r.publish(ALICE, EPOCH_1, 1, b"1".to_vec(), 10)).len(),
        1
    );
    assert_eq!(
        refused(&r.publish(ALICE, EPOCH_1, 2, b"2".to_vec(), 69)),
        Some(Refusal::RateLimited)
    );
    // the other tag has its own count
    assert_eq!(
        forwards(&r.publish(BOB, EPOCH_1, 0, b"b".to_vec(), 69)).len(),
        1
    );
    // the next window
    assert_eq!(
        forwards(&r.publish(ALICE, EPOCH_1, 2, b"2".to_vec(), 70)).len(),
        1
    );
}

#[test]
fn rooms_are_bounded_and_the_longest_idle_empty_room_makes_way() {
    let mut r = AnonymousRelayCore::new(Limits {
        max_rooms: 2,
        ..LIMITS
    });
    let room = |n: u8| RoomAlias::from_raw([n; 32]);
    let token = |n: u8| RoomToken::from_raw([n; 32]);
    assert!(bound(
        &r.bind(ConnId::from_raw(1), room(1), token(1), TAG_A, 10),
        ConnId::from_raw(1)
    ));
    assert!(bound(
        &r.bind(ConnId::from_raw(2), room(2), token(2), TAG_A, 20),
        ConnId::from_raw(2)
    ));
    // both rooms have a member: no room makes way
    assert_eq!(
        refused(&r.bind(ConnId::from_raw(3), room(3), token(3), TAG_A, 30)),
        Some(Refusal::TooManyRooms)
    );
    // room 1 empties; it makes way for room 3
    r.close(ConnId::from_raw(1));
    assert!(bound(
        &r.bind(ConnId::from_raw(3), room(3), token(3), TAG_A, 40),
        ConnId::from_raw(3)
    ));
    // room 1 is gone, so its alias is free again (under whatever token comes first)
    r.close(ConnId::from_raw(2));
    assert!(bound(
        &r.bind(ConnId::from_raw(4), room(1), token(9), TAG_A, 50),
        ConnId::from_raw(4)
    ));
}
