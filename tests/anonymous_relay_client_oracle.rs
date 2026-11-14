//! Contract tests for the anonymous relay's sans-I/O client (`anonymous_relay_client::RelayClient`), run
//! against the relay core in process: every frame the client builds goes through `AnonymousRelayCore` as
//! the relay's host would hand it over (`bind` / `publish` / `log_range_within` / `leave`), and every frame
//! the core answers with goes back through `RelayClient::on_frame`.
//!
//! The expected values come from the core's own contract (a tag's payloads leave in sequence order, a gap
//! is asked for with a resend, a log read under a byte budget answers with the longest prefix that fits),
//! not from the client: the client is checked against what the relay does.

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, LogItem, Refusal, RoomAlias, RoomToken, SenderTag,
};
use alice_sync::anonymous_relay_client::{
    classify, Action, ClientError, Event, RelayClient, DEFAULT_MAX_UNACKED,
};
use alice_sync::anonymous_relay_wire::Frame;

const ROOM: RoomAlias = RoomAlias::from_raw([0x31; 32]);
const OTHER_ROOM: RoomAlias = RoomAlias::from_raw([0x32; 32]);
const TOKEN: RoomToken = RoomToken::from_raw([0xb1; 32]);
const WRONG_TOKEN: RoomToken = RoomToken::from_raw([0xb2; 32]);
const ALICE: ConnId = ConnId::from_raw(1);
const BOB: ConnId = ConnId::from_raw(2);
const TAG_A: SenderTag = SenderTag::from_raw([0x0a; 16]);
const TAG_B: SenderTag = SenderTag::from_raw([0x0b; 16]);

const LIMITS: Limits = Limits {
    max_payload: 64,
    max_log: 32,
    max_pending: 8,
    max_rooms: 4,
    max_members: 4,
    max_publishes_per_window: 1_000,
    window: 60,
    max_rooms_per_conn: 2,
    max_tags_per_room: 4,
};

/// The frame budget for log reads (the default: room for every log of these tests)
const WIDE: usize = 1 << 20;

/// What the relay's host does with one client frame from `conn`: the frames the core sends back
fn host(
    core: &mut AnonymousRelayCore,
    conn: ConnId,
    frame: Frame,
    budget: usize,
) -> Vec<(ConnId, Frame)> {
    match frame {
        Frame::Bind { room, token, tag } => wire(core.bind(conn, room, token, tag, 0)),
        Frame::Publish { room, seq, payload } => wire(core.publish(conn, room, seq, payload, 0)),
        Frame::LogRange {
            room,
            from_index,
            to_index,
        } => match core.log_range_within(conn, room, from_index, to_index, budget) {
            Ok(range) => vec![(conn, range.into_wire(room))],
            Err(reason) => vec![(conn, Frame::Refused { room, reason })],
        },
        Frame::Close { room } => {
            core.leave(conn, room);
            Vec::new()
        }
        other => panic!("not a client frame: {other:?}"),
    }
}

fn wire(out: Vec<alice_sync::anonymous_relay::Out>) -> Vec<(ConnId, Frame)> {
    out.into_iter()
        .map(alice_sync::anonymous_relay::Out::into_wire)
        .collect()
}

/// The frames for `conn`, each through the wire's bytes (encode then strict decode), in order
fn to(conn: ConnId, frames: &[(ConnId, Frame)]) -> Vec<Frame> {
    frames
        .iter()
        .filter(|(c, _)| *c == conn)
        .map(|(_, f)| Frame::decode(&f.encode()).unwrap())
        .collect()
}

fn alice() -> RelayClient {
    RelayClient::new(ROOM, TOKEN, TAG_A, 0, LIMITS.max_payload)
}

fn bob() -> RelayClient {
    RelayClient::new(ROOM, TOKEN, TAG_B, 0, LIMITS.max_payload)
}

/// Bind `client` on `conn`; the single answer must be `Bound` with nothing to replay
fn bind(core: &mut AnonymousRelayCore, conn: ConnId, client: &mut RelayClient) {
    let answers = to(conn, &host(core, conn, client.bind(), WIDE));
    assert_eq!(answers.len(), 1, "{answers:?}");
    let event = client.on_frame(answers[0].clone()).unwrap();
    assert_eq!(event, Event::Bound { replay: Vec::new() });
}

fn payload(i: u64) -> Vec<u8> {
    format!("payload {i}").into_bytes()
}

fn kept(client: &RelayClient) -> Vec<u64> {
    client.unacked().map(|(s, _)| s).collect()
}

#[test]
fn bind_names_the_room_token_and_tag_and_bound_comes_back() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let mut a = alice();
    assert_eq!(
        a.bind(),
        Frame::Bind {
            room: ROOM,
            token: TOKEN,
            tag: TAG_A,
        }
    );
    bind(&mut core, ALICE, &mut a);
    assert_eq!(a.leave(), Frame::Close { room: ROOM });
}

#[test]
fn published_payloads_reach_the_other_member_in_sequence_order() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let (mut a, mut b) = (alice(), bob());
    bind(&mut core, ALICE, &mut a);
    bind(&mut core, BOB, &mut b);
    for i in 0..6u64 {
        let frame = a.publish(payload(i)).unwrap();
        assert_eq!(
            frame,
            Frame::Publish {
                room: ROOM,
                seq: i,
                payload: payload(i),
            }
        );
        let out = host(&mut core, ALICE, frame, WIDE);
        assert!(
            to(ALICE, &out).is_empty(),
            "the publisher gets nothing back"
        );
        let got = to(BOB, &out);
        assert_eq!(got.len(), 1);
        assert_eq!(
            b.on_frame(got[0].clone()).unwrap(),
            Event::Forward {
                from: TAG_A,
                seq: i,
                payload: payload(i),
            }
        );
    }
    assert_eq!(a.next_seq(), 6);
    assert_eq!(
        kept(&a),
        (0..6).collect::<Vec<_>>(),
        "nothing acknowledged yet"
    );
}

#[test]
fn a_resend_is_answered_with_exactly_the_missing_payloads() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let (mut a, mut b) = (alice(), bob());
    bind(&mut core, ALICE, &mut a);
    bind(&mut core, BOB, &mut b);
    let frames: Vec<Frame> = (0..4).map(|i| a.publish(payload(i)).unwrap()).collect();
    // 1 and 2 are lost on the way; 3 arrives ahead of the next expected number
    let mut bob_seqs = Vec::new();
    let deliver = |out: &[(ConnId, Frame)], b: &mut RelayClient, seqs: &mut Vec<u64>| {
        for f in to(BOB, out) {
            if let Event::Forward {
                seq, payload: p, ..
            } = b.on_frame(f).unwrap()
            {
                assert_eq!(p, payload(seq));
                seqs.push(seq);
            }
        }
    };
    let out = host(&mut core, ALICE, frames[0].clone(), WIDE);
    deliver(&out, &mut b, &mut bob_seqs);
    let out = host(&mut core, ALICE, frames[3].clone(), WIDE);
    deliver(&out, &mut b, &mut bob_seqs);
    let asked = to(ALICE, &out);
    assert_eq!(
        asked,
        vec![Frame::Resend {
            room: ROOM,
            from: 1,
            to_seq: 2,
        }]
    );
    let Event::Resend(again) = a.on_frame(asked[0].clone()).unwrap() else {
        panic!("not a resend");
    };
    assert_eq!(
        again,
        vec![frames[1].clone(), frames[2].clone()],
        "exactly 1 and 2"
    );
    // the resend asked from 1: the relay has 0, so 0 is no longer kept
    assert_eq!(kept(&a), vec![1, 2, 3]);
    for f in again {
        let out = host(&mut core, ALICE, f, WIDE);
        deliver(&out, &mut b, &mut bob_seqs);
    }
    assert_eq!(bob_seqs, vec![0, 1, 2, 3], "every payload once, in order");
}

#[test]
fn bind_again_replays_the_kept_payloads_and_the_relay_drops_duplicates() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let (mut a, mut b) = (alice(), bob());
    bind(&mut core, BOB, &mut b);
    bind(&mut core, ALICE, &mut a);
    let f0 = a.publish(payload(0)).unwrap();
    let f1 = a.publish(payload(1)).unwrap();
    // 0 arrives, 1 is lost with the connection
    let out = host(&mut core, ALICE, f0.clone(), WIDE);
    assert_eq!(to(BOB, &out).len(), 1);
    core.close(ALICE);
    let answers = to(ALICE, &host(&mut core, ALICE, a.bind(), WIDE));
    let Event::Bound { replay } = a.on_frame(answers[0].clone()).unwrap() else {
        panic!("not bound");
    };
    assert_eq!(replay, vec![f0, f1]);
    let mut seqs = Vec::new();
    for f in replay {
        for g in to(BOB, &host(&mut core, ALICE, f, WIDE)) {
            if let Event::Forward { seq, .. } = b.on_frame(g).unwrap() {
                seqs.push(seq);
            }
        }
    }
    assert_eq!(
        seqs,
        vec![1],
        "0 is a duplicate at the relay, 1 is delivered"
    );
}

#[test]
fn log_pages_under_a_small_budget_reassemble_the_whole_log() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let (mut a, mut b) = (alice(), bob());
    bind(&mut core, ALICE, &mut a);
    bind(&mut core, BOB, &mut b);
    for i in 0..7u64 {
        let f = a.publish(vec![u8::try_from(i).unwrap(); 20]).unwrap();
        host(&mut core, ALICE, f, WIDE);
    }
    // oracle: the core's own log, read whole
    let whole = core.log_range(BOB, ROOM, 0, u64::MAX).unwrap().items;
    assert_eq!(whole.len(), 7);
    // two 20-byte items per page: the frame head is 45 bytes, an item 36 bytes and its payload
    let budget = 45 + 2 * (36 + 20);
    let mut read: Vec<LogItem> = Vec::new();
    let mut pages = 0;
    let mut next = Some(b.read_log(0));
    while let Some(frame) = next.take() {
        pages += 1;
        assert!(pages <= 10, "the reads do not come to an end");
        let answers = to(BOB, &host(&mut core, BOB, frame, budget));
        let Event::LogPage {
            first_index,
            items,
            next: n,
        } = b.on_frame(answers[0].clone()).unwrap()
        else {
            panic!("not a log page");
        };
        assert_eq!(first_index, 0);
        assert!(items.len() <= 2);
        read.extend(items);
        next = n;
    }
    assert_eq!(read, whole);
    assert_eq!(
        pages, 5,
        "4 pages of items, then the empty page that ends the read"
    );
}

#[test]
fn reading_the_log_releases_this_tags_kept_payloads() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let mut a = alice().with_max_unacked(3);
    bind(&mut core, ALICE, &mut a);
    for i in 0..3u64 {
        let f = a.publish(payload(i)).unwrap();
        host(&mut core, ALICE, f, WIDE);
    }
    assert_eq!(a.publish(payload(3)), Err(ClientError::Full { limit: 3 }));
    assert_eq!(a.next_seq(), 3, "a refused publish uses no sequence number");
    let answers = to(ALICE, &host(&mut core, ALICE, a.read_log(0), WIDE));
    let Event::LogPage { items, .. } = a.on_frame(answers[0].clone()).unwrap() else {
        panic!("not a log page");
    };
    assert_eq!(items.len(), 3);
    assert!(kept(&a).is_empty(), "the log holds 0..=2 under this tag");
    assert!(a.publish(payload(3)).is_ok());
}

#[test]
fn release_through_drops_the_kept_payloads_up_to_the_number() {
    let mut a = alice().with_max_unacked(2);
    a.publish(payload(0)).unwrap();
    a.publish(payload(1)).unwrap();
    assert_eq!(a.publish(payload(2)), Err(ClientError::Full { limit: 2 }));
    a.release_through(0);
    assert_eq!(kept(&a), vec![1]);
    assert_eq!(
        a.publish(payload(2)).unwrap(),
        Frame::Publish {
            room: ROOM,
            seq: 2,
            payload: payload(2),
        }
    );
    assert_eq!(DEFAULT_MAX_UNACKED, 64);
}

#[test]
fn a_payload_over_max_payload_is_refused_locally_with_no_frame() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let mut a = alice();
    bind(&mut core, ALICE, &mut a);
    assert_eq!(
        a.publish(vec![0; 65]),
        Err(ClientError::TooLarge { len: 65, max: 64 })
    );
    assert_eq!(a.next_seq(), 0);
    assert!(kept(&a).is_empty());
    // the largest one passes, and the relay takes it
    let f = a.publish(vec![0; 64]).unwrap();
    assert!(to(ALICE, &host(&mut core, ALICE, f, WIDE)).is_empty());
}

#[test]
fn the_last_sequence_number_is_u64_max_minus_1() {
    let mut a = RelayClient::new(ROOM, TOKEN, TAG_A, u64::MAX - 1, 64);
    assert_eq!(
        a.publish(payload(0)).unwrap(),
        Frame::Publish {
            room: ROOM,
            seq: u64::MAX - 1,
            payload: payload(0),
        }
    );
    assert_eq!(a.publish(payload(1)), Err(ClientError::Exhausted));
    assert_eq!(a.next_seq(), u64::MAX);
}

/// Every refusal the wire knows, from its refusal codes (decoding a `Refused` frame for each byte value):
/// a refusal added to the wire without a row below fails here
fn every_refusal() -> Vec<Refusal> {
    (0..=u8::MAX)
        .filter_map(|code| {
            let mut bytes = vec![0x46];
            bytes.extend_from_slice(&[0; 32]);
            bytes.push(code);
            match Frame::decode(&bytes) {
                Ok(Frame::Refused { reason, .. }) => Some(reason),
                _ => None,
            }
        })
        .collect()
}

#[test]
fn classify_maps_every_refusal() {
    let table = [
        (Refusal::NoRoom, Action::Rebind),
        (Refusal::BadToken, Action::NextEpoch),
        (Refusal::NotBound, Action::Rebind),
        (Refusal::TagInUse, Action::RetryLater),
        (Refusal::TooLarge, Action::Internal),
        (Refusal::TooFarAhead, Action::RetryLater),
        (Refusal::TooManyRooms, Action::FallBack),
        (Refusal::RoomFull, Action::FallBack),
        (Refusal::RateLimited, Action::RetryLater),
        (Refusal::ConnRoomLimit, Action::Internal),
        (Refusal::TagChange, Action::Internal),
        (Refusal::TooManyTags, Action::RetryLater),
        (Refusal::Exhausted, Action::NextEpoch),
    ];
    let all = every_refusal();
    assert_eq!(all.len(), 13, "{all:?}");
    for r in &all {
        let row = table.iter().find(|(t, _)| t == r);
        let Some((_, action)) = row else {
            panic!("no row for {r:?}");
        };
        assert_eq!(classify(*r), *action, "{r:?}");
    }
}

#[test]
fn a_refusal_from_the_relay_comes_with_its_action() {
    let mut core = AnonymousRelayCore::new(LIMITS);
    let mut a = alice();
    bind(&mut core, ALICE, &mut a);
    let mut b = RelayClient::new(ROOM, WRONG_TOKEN, TAG_B, 0, 64);
    let answers = to(BOB, &host(&mut core, BOB, b.bind(), WIDE));
    assert_eq!(
        b.on_frame(answers[0].clone()).unwrap(),
        Event::Refused {
            reason: Refusal::BadToken,
            action: Action::NextEpoch,
        }
    );
}

#[test]
fn frames_of_another_room_or_from_a_client_are_refused() {
    let mut a = alice();
    assert_eq!(
        a.on_frame(Frame::Bound { room: OTHER_ROOM }),
        Err(ClientError::OtherRoom(OTHER_ROOM))
    );
    assert_eq!(a.on_frame(a.bind()), Err(ClientError::Unexpected));
    assert_eq!(a.on_frame(a.leave()), Err(ClientError::Unexpected));
    assert_eq!(a.on_frame(a.read_log(0)), Err(ClientError::Unexpected));
    assert_eq!(
        a.on_frame(Frame::Publish {
            room: ROOM,
            seq: 0,
            payload: Vec::new(),
        }),
        Err(ClientError::Unexpected)
    );
    assert_eq!(
        a.on_frame(Frame::Resend {
            room: ROOM,
            from: 2,
            to_seq: 1,
        }),
        Err(ClientError::Unexpected),
        "an empty range"
    );
}

#[test]
fn a_resend_of_payloads_no_longer_kept_is_unavailable() {
    let mut a = alice();
    for i in 0..3 {
        a.publish(payload(i)).unwrap();
    }
    a.release_through(1);
    assert_eq!(
        a.on_frame(Frame::Resend {
            room: ROOM,
            from: 0,
            to_seq: 2,
        }),
        Err(ClientError::Unavailable { from: 0, to_seq: 2 })
    );
    // nothing was released by the refused resend
    assert_eq!(kept(&a), vec![2]);
}
