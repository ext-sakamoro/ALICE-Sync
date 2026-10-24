//! The bytes of `Event::to_bytes` and `Message::to_bytes` for one value of every event kind and every
//! message variant, pinned by their SHA-256 (the alice-serial format, ALICE-Serial format 1): a change
//! of the encoding shows here as a changed pin; two messages are also written out byte by byte from
//! the format's rules, and decoding refuses every malformed shape

use alice_serial::Value;
use alice_sync::authorship::Checkpoint;
use alice_sync::event::{Event, EventId, EventKind, Origin, SeqNum};
use alice_sync::protocol::Message;
use alice_sync::{NodeId, WorldHash};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn event(n: u64, kind: EventKind) -> Event {
    Event {
        id: EventId(n),
        seq: SeqNum(n + 1),
        origin: Origin([n as u8 ^ 0x5a; 16]),
        timestamp: 1_000 + n,
        kind,
    }
}

fn events() -> Vec<(&'static str, Event)> {
    vec![
        (
            "motion",
            event(
                1,
                EventKind::Motion {
                    entity: 7,
                    delta: [-3, 0, 32_767],
                },
            ),
        ),
        (
            "spawn",
            event(
                2,
                EventKind::Spawn {
                    entity: u32::MAX,
                    kind: 9,
                    pos: [i16::MIN, 1, -1],
                },
            ),
        ),
        ("despawn", event(3, EventKind::Despawn { entity: 0 })),
        (
            "property",
            event(
                4,
                EventKind::Property {
                    entity: 12,
                    prop: 3,
                    value: -70_000,
                },
            ),
        ),
        (
            "input",
            event(
                5,
                EventKind::Input {
                    player: 2,
                    code: 0x1234_5678,
                },
            ),
        ),
        ("tick", event(6, EventKind::Tick { frame: u64::MAX })),
        (
            "custom",
            event(
                7,
                EventKind::Custom {
                    type_id: 42,
                    payload: *b"0123456789abcdef",
                },
            ),
        ),
    ]
}

fn messages() -> Vec<(&'static str, Message)> {
    let evs: Vec<Event> = events().into_iter().map(|(_, e)| e).collect();
    let cp = Checkpoint {
        origin: Origin([0x11; 16]),
        seq: 64,
        head: [0x22; 32],
        signature: (0..80u8).collect(),
    };
    vec![
        (
            "hello",
            Message::Hello {
                node_id: NodeId(3),
                seq: 17,
                world_hash: WorldHash(0xfeed_beef),
                semantics_id: [0x33; 32],
            },
        ),
        ("request_sync", Message::RequestSync { since_seq: 99 }),
        (
            "events",
            Message::Events {
                events: evs,
                checkpoints: vec![cp],
            },
        ),
        (
            "events_empty",
            Message::Events {
                events: Vec::new(),
                checkpoints: Vec::new(),
            },
        ),
        (
            "hash_check",
            Message::HashCheck {
                seq: 5,
                world_hash: WorldHash(1),
            },
        ),
        (
            "diverged",
            Message::Diverged {
                my_hash: WorldHash(2),
                your_hash: WorldHash(u64::MAX),
            },
        ),
        ("ack", Message::Ack { seq: 0 }),
        ("bye", Message::Bye),
    ]
}

/// (name, length, SHA-256 of the bytes)
const EVENT_PINS: &[(&str, usize, &str)] = &[
    (
        "motion",
        82,
        "3d24ec17a70c6598a52c75a1ea50cf0f48d6077cfb3bb7c903f1a4d93f2698c3",
    ),
    (
        "spawn",
        91,
        "1b2241262b387f144067d74035c600c26c47cbc88e5c46a608ac6abbc6b84ed8",
    ),
    (
        "despawn",
        50,
        "3d720861815b27f2c39e69f158d5557c1bcbabb7b46be213d5a714014b9b62e6",
    ),
    (
        "property",
        60,
        "7c63d244ff8e0b20d6c738f878f3f53860576877ec81ccc596a207c222260f3f",
    ),
    (
        "input",
        59,
        "d2a3233020eeac636be263112f634760fb0f19dc4e57d192dc8a060368c06810",
    ),
    (
        "tick",
        58,
        "f67d3c8458d32cb8079333847e1ea71c26929d54a61b5164c2029a68c6c13619",
    ),
    (
        "custom",
        71,
        "b9bd5bd8f8bb879b2040edf4196bb4154f23d0b2bb8536871008039812366c6f",
    ),
];
const MESSAGE_PINS: &[(&str, usize, &str)] = &[
    (
        "hello",
        60,
        "ddfa1fc44bbd4968159434ab83bdb087e7a2fe33996806a8394a26ea110c6490",
    ),
    (
        "request_sync",
        13,
        "7e5cf62f2b394324db6d01aec3b122c835bc74db0dabfa33ee693b427aa8a265",
    ),
    (
        "events",
        600,
        "6f7478cdaf42d279cafbb37f83eaae36c24274d65019d64c494d640383987bc7",
    ),
    (
        "events_empty",
        22,
        "44c6429a8d84e5ea02e94e44266d7602281f038ab3092f6292ce639639aff122",
    ),
    (
        "hash_check",
        14,
        "b94e6331fe71b5b6a13fcd76afe3372be1b048d3646d209227a90fae166d4ed6",
    ),
    (
        "diverged",
        22,
        "a10b009783bbf63e50f588f41f639ceb2481a24fa98707e1d076f7817d4ade64",
    ),
    (
        "ack",
        13,
        "9ae5ea48f3ba13d41c63589111ab5219057dcf36825102fc597168d2fe32ca40",
    ),
    (
        "bye",
        12,
        "e4a1ed1366efb00408c077b58273986040ad67b269d53204f25b7b77576ffa36",
    ),
];

fn check(kind: &str, got: Vec<(&'static str, Vec<u8>)>, pins: &[(&str, usize, &str)]) {
    let got: Vec<(&str, usize, String)> = got
        .into_iter()
        .map(|(n, b)| (n, b.len(), hex(&Sha256::digest(&b))))
        .collect();
    let want: Vec<(&str, usize, String)> = pins
        .iter()
        .map(|(n, l, h)| (*n, *l, (*h).to_owned()))
        .collect();
    assert!(!got.is_empty(), "{kind}: compared nothing");
    assert_eq!(got, want, "{kind}: the encoded bytes changed");
}

#[test]
fn event_bytes_are_pinned() {
    let got = events()
        .into_iter()
        .map(|(n, e)| (n, e.to_bytes().unwrap()))
        .collect();
    check("event", got, EVENT_PINS);
}

#[test]
fn message_bytes_are_pinned() {
    let got = messages()
        .into_iter()
        .map(|(n, m)| (n, m.to_bytes().unwrap()))
        .collect();
    check("message", got, MESSAGE_PINS);
}

#[test]
fn every_event_kind_and_message_variant_is_covered() {
    // a new kind or variant fails to compile here until it has a sample above
    for (_, e) in events() {
        match e.kind {
            EventKind::Motion { .. }
            | EventKind::Spawn { .. }
            | EventKind::Despawn { .. }
            | EventKind::Property { .. }
            | EventKind::Input { .. }
            | EventKind::Tick { .. }
            | EventKind::Custom { .. } => {}
        }
    }
    for (_, m) in messages() {
        match m {
            Message::Hello { .. }
            | Message::RequestSync { .. }
            | Message::Events { .. }
            | Message::HashCheck { .. }
            | Message::Diverged { .. }
            | Message::Ack { .. }
            | Message::Bye => {}
        }
    }
    assert_eq!(events().len(), 7);
    assert_eq!(messages().len(), 8);
}

#[test]
fn two_messages_have_the_bytes_the_format_rules_give() {
    // header "ALSR" + format 1 (u16 LE), then an array (dd, u32 BE count) of the variant number as a
    // positive fixint and the fields
    let header = [b'A', b'L', b'S', b'R', 1, 0];
    let mut bye = header.to_vec();
    bye.extend_from_slice(&[0xdd, 0, 0, 0, 1, 6]);
    assert_eq!(Message::Bye.to_bytes().unwrap(), bye);
    let mut ack = header.to_vec();
    ack.extend_from_slice(&[0xdd, 0, 0, 0, 2, 5, 0xcf, 0, 0, 0, 0, 0, 0, 1, 0]);
    assert_eq!(Message::Ack { seq: 256 }.to_bytes().unwrap(), ack);
}

#[test]
fn every_sample_decodes_to_itself() {
    for (n, e) in events() {
        let b = e.to_bytes().unwrap();
        assert_eq!(Event::from_bytes(&b), Some(e), "{n}");
    }
    for (n, m) in messages() {
        let b = m.to_bytes().unwrap();
        let back = Message::from_bytes(&b).unwrap_or_else(|| panic!("{n}"));
        assert_eq!(back.to_bytes().unwrap(), b, "{n}");
    }
}

fn enc(v: Value) -> Vec<u8> {
    alice_serial::encode(&v).unwrap()
}

fn event_fields(kind: Value) -> Vec<Value> {
    vec![
        Value::U64(1),
        Value::U64(2),
        Value::Bytes(vec![7; 16]),
        Value::U64(3),
        kind,
    ]
}

#[test]
fn decoding_refuses_every_malformed_shape() {
    let tick = |frame: Value| Value::Array(vec![Value::U64(5), frame]);
    let good = enc(Value::Array(event_fields(tick(Value::U64(9)))));
    assert!(
        Event::from_bytes(&good).is_some(),
        "the reference shape decodes"
    );
    let mut trailing = good.clone();
    trailing.push(0);
    let mut short = event_fields(tick(Value::U64(9)));
    short.pop();
    let mut long = event_fields(tick(Value::U64(9)));
    long.push(Value::U64(0));
    let mut origin_15 = event_fields(tick(Value::U64(9)));
    origin_15[2] = Value::Bytes(vec![7; 15]);
    let spawn = |entity: Value, x: Value| {
        Value::Array(vec![
            Value::U64(1),
            entity,
            Value::U64(1),
            Value::Array(vec![x, Value::I64(0), Value::I64(0)]),
        ])
    };
    assert!(Event::from_bytes(&enc(Value::Array(event_fields(spawn(
        Value::U64(1),
        Value::I64(-1)
    )))))
    .is_some());
    for (why, bytes) in [
        ("trailing byte", trailing),
        ("a field short", enc(Value::Array(short))),
        ("a field too many", enc(Value::Array(long))),
        ("origin of 15 bytes", enc(Value::Array(origin_15))),
        (
            "unknown kind",
            enc(Value::Array(event_fields(Value::Array(vec![Value::U64(
                7,
            )])))),
        ),
        (
            "kind with a field too many",
            enc(Value::Array(event_fields(Value::Array(vec![
                Value::U64(5),
                Value::U64(1),
                Value::U64(2),
            ])))),
        ),
        (
            "signed where unsigned",
            enc(Value::Array(event_fields(tick(Value::I64(9))))),
        ),
        (
            "entity past u32",
            enc(Value::Array(event_fields(spawn(
                Value::U64(1 << 32),
                Value::I64(0),
            )))),
        ),
        (
            "delta past i16",
            enc(Value::Array(event_fields(spawn(
                Value::U64(1),
                Value::I64(40_000),
            )))),
        ),
        (
            "unsigned where signed",
            enc(Value::Array(event_fields(spawn(
                Value::U64(1),
                Value::U64(1),
            )))),
        ),
    ] {
        assert_eq!(Event::from_bytes(&bytes), None, "{why}");
    }
    for (why, v) in [
        ("unknown variant", Value::Array(vec![Value::U64(7)])),
        (
            "Bye with a field",
            Value::Array(vec![Value::U64(6), Value::U64(0)]),
        ),
        (
            "Hello with a 31-byte id",
            Value::Array(vec![
                Value::U64(0),
                Value::U64(1),
                Value::U64(2),
                Value::U64(3),
                Value::Bytes(vec![0; 31]),
            ]),
        ),
        (
            "Events with a bad checkpoint",
            Value::Array(vec![
                Value::U64(2),
                Value::Array(vec![]),
                Value::Array(vec![Value::Array(vec![
                    Value::Bytes(vec![0; 16]),
                    Value::U64(1),
                    Value::Bytes(vec![0; 31]),
                    Value::Bytes(vec![]),
                ])]),
            ]),
        ),
        ("not an array", Value::U64(6)),
    ] {
        assert!(Message::from_bytes(&enc(v)).is_none(), "{why}");
    }
}
