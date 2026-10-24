//! The bytes of `Event::to_bytes` and `Message::to_bytes` for one value of every event kind and every
//! message variant, pinned by their SHA-256: a change of the encoding shows here as a changed pin

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
        54,
        "adc13e8991e0e1a35e3f0eb06de4c30aa2415be615ab33b4b57c3c5a33a1212c",
    ),
    (
        "spawn",
        56,
        "df2ff9728675e8aa9c1c12b91a82e3a8c387bc3187e488b0d6f0679eca31acaa",
    ),
    (
        "despawn",
        48,
        "8492824a159bd3caed7c750e8c22ddc27295a4cd83974477c3a485d8a7f0a825",
    ),
    (
        "property",
        54,
        "629c1f5aa41c8a4699fa0b1db2de5c76b731788c2f13b94ac4a8de6dce1bda57",
    ),
    (
        "input",
        50,
        "7eb6a4258587b53b87454723816f8f64058fe19e5babdc3843071895f60b2517",
    ),
    (
        "tick",
        52,
        "35bf2b6ea22ee3bde9a55cd3bb0863b72c61f0c0566a985e43234039d9e2f12e",
    ),
    (
        "custom",
        62,
        "0e5321fdca7f06e030ed8b4b8b6f306963ecc4b455e3f563402cd28f79841b8f",
    ),
];
const MESSAGE_PINS: &[(&str, usize, &str)] = &[
    (
        "hello",
        60,
        "fd8c3e27ffd35e1119685a9f3d194ecc09c8114febe8cbe8e1b58867cde393c6",
    ),
    (
        "request_sync",
        12,
        "4296f021e7ffa298794c1bd4ba40a7c3fe30d696b5f00f64a4a17b52d718a415",
    ),
    (
        "events",
        540,
        "b56f1172bed2f5850339638614287d20c48ad157489ef3bf8cee34e18a4f66d0",
    ),
    (
        "events_empty",
        20,
        "c75eeff218c4e3c7142fc8c157817d0bdf537b3446164e4fc26453bc42c4c62d",
    ),
    (
        "hash_check",
        20,
        "925e6538cc48c35ca66cec04d5d88e4e00c284aab5803465f34afa26215f2445",
    ),
    (
        "diverged",
        20,
        "d3038999170269d0a4b2b060754f6dbd9ed2433cd55fad4f18ff6190ef714d12",
    ),
    (
        "ack",
        12,
        "65536c129f290987890b4c6c880d43bdecac47e954651457a53a49b362da8969",
    ),
    (
        "bye",
        4,
        "7aa8ca4a02506da9133d8f889678b76f716ce45d02e22fdb7b70a15e56a0eff8",
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
        .map(|(n, e)| (n, e.to_bytes()))
        .collect();
    check("event", got, EVENT_PINS);
}

#[test]
fn message_bytes_are_pinned() {
    let got = messages()
        .into_iter()
        .map(|(n, m)| (n, m.to_bytes()))
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
