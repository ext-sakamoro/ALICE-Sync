//! Oracle: a custom event carries a payload of 0 to `MAX_CUSTOM_PAYLOAD` bytes, round-trips through
//! the compact wire form and the signed chain, and anything longer is refused at construction, on
//! the wire and in a signed batch. An event in the earlier fixed 16-byte layout is refused, not
//! read as some other payload.

use std::collections::BTreeMap;

use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use alice_sync::authorship::{canonical_bytes, Author, TrustPolicy};
use alice_sync::event::{CustomPayload, CustomPayloadError, MAX_CUSTOM_PAYLOAD};
use alice_sync::{Event, EventKind, Node, NodeId};

const NOW: u64 = 1_800_000_100;

fn custom(len: usize) -> Event {
    let bytes: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
    Event::new(EventKind::Custom {
        type_id: 7,
        payload: CustomPayload::new(bytes).unwrap(),
    })
}

#[test]
fn the_bound_is_part_of_the_wire_format() {
    assert_eq!(MAX_CUSTOM_PAYLOAD, 256);
}

#[test]
fn payloads_up_to_the_bound_round_trip_on_the_wire() {
    for len in [0, 1, 16, MAX_CUSTOM_PAYLOAD - 1, MAX_CUSTOM_PAYLOAD] {
        let e = custom(len);
        let back = Event::from_compact_bytes(&e.to_compact_bytes()).expect("decodes");
        assert_eq!(back, e, "len {len}");
        let EventKind::Custom { payload, .. } = &back.kind else {
            panic!("custom")
        };
        assert_eq!(payload.as_bytes().len(), len);
    }
}

#[test]
fn a_payload_over_the_bound_is_refused_at_construction() {
    assert!(CustomPayload::new(vec![0; MAX_CUSTOM_PAYLOAD]).is_ok());
    assert_eq!(
        CustomPayload::new(vec![0; MAX_CUSTOM_PAYLOAD + 1]),
        Err(CustomPayloadError::TooLong {
            len: MAX_CUSTOM_PAYLOAD + 1
        })
    );
}

/// The wire form of an event as bitcode lays out `Event` (`id`, `seq`, `origin`, `timestamp`,
/// `kind`): a look-alike with an unchecked payload, to send what a correct writer never would
mod wire {
    #[derive(bitcode::Encode)]
    pub struct EventId(pub u64);
    #[derive(bitcode::Encode)]
    pub struct SeqNum(pub u64);
    #[derive(bitcode::Encode)]
    pub struct Origin(pub [u8; 16]);
    #[derive(bitcode::Encode)]
    pub struct Payload(pub Vec<u8>);
    #[derive(bitcode::Encode)]
    #[allow(dead_code)]
    pub enum Kind {
        Spawn {
            entity: u32,
            kind: u16,
            pos: [i16; 3],
        },
        Despawn {
            entity: u32,
        },
        Motion {
            entity: u32,
            delta: [i16; 3],
        },
        Property {
            entity: u32,
            prop: u16,
            value: i32,
        },
        Input {
            player: u16,
            code: u32,
        },
        Tick {
            frame: u64,
        },
        Custom {
            type_id: u16,
            payload: Payload,
        },
    }
    #[derive(bitcode::Encode)]
    pub struct Event {
        pub id: EventId,
        pub seq: SeqNum,
        pub origin: Origin,
        pub timestamp: u64,
        pub kind: Kind,
    }
    /// The earlier layout: the payload was a fixed array of 16 bytes
    #[derive(bitcode::Encode)]
    #[allow(dead_code)]
    pub enum OldKind {
        Spawn {
            entity: u32,
            kind: u16,
            pos: [i16; 3],
        },
        Despawn {
            entity: u32,
        },
        Motion {
            entity: u32,
            delta: [i16; 3],
        },
        Property {
            entity: u32,
            prop: u16,
            value: i32,
        },
        Input {
            player: u16,
            code: u32,
        },
        Tick {
            frame: u64,
        },
        Custom {
            type_id: u16,
            payload: [u8; 16],
        },
    }
    #[derive(bitcode::Encode)]
    pub struct OldEvent {
        pub id: EventId,
        pub seq: SeqNum,
        pub origin: Origin,
        pub timestamp: u64,
        pub kind: OldKind,
    }
}

fn forged(len: usize) -> Vec<u8> {
    bitcode::encode(&wire::Event {
        id: wire::EventId(0),
        seq: wire::SeqNum(0),
        origin: wire::Origin([0; 16]),
        timestamp: 0,
        kind: wire::Kind::Custom {
            type_id: 7,
            payload: wire::Payload(vec![1; len]),
        },
    })
}

#[test]
fn the_look_alike_matches_the_real_layout_at_the_bound() {
    // the forged form of a payload at the bound is exactly what the real writer produces
    let real = Event::new(EventKind::Custom {
        type_id: 7,
        payload: CustomPayload::new(vec![1; MAX_CUSTOM_PAYLOAD]).unwrap(),
    });
    assert_eq!(forged(MAX_CUSTOM_PAYLOAD), real.to_compact_bytes());
    assert!(Event::from_compact_bytes(&forged(MAX_CUSTOM_PAYLOAD)).is_some());
}

#[test]
fn a_payload_over_the_bound_is_refused_on_the_wire() {
    assert!(Event::from_compact_bytes(&forged(MAX_CUSTOM_PAYLOAD + 1)).is_none());
    assert!(Event::from_compact_bytes(&forged(4 * MAX_CUSTOM_PAYLOAD)).is_none());
}

#[test]
fn an_event_in_the_old_fixed_16_byte_layout_is_refused() {
    let old = bitcode::encode(&wire::OldEvent {
        id: wire::EventId(0),
        seq: wire::SeqNum(0),
        origin: wire::Origin([0; 16]),
        timestamp: 0,
        kind: wire::OldKind::Custom {
            type_id: 7,
            payload: [0xab; 16],
        },
    });
    assert!(Event::from_compact_bytes(&old).is_none());
}

#[test]
fn the_signed_bytes_carry_the_payload_length() {
    // without a length, [0] then the next field and [0, 0] could sign the same bytes
    let one = canonical_bytes(&custom(1));
    let two = canonical_bytes(&custom(2));
    assert_ne!(one, two);
    let e = custom(3);
    let b = canonical_bytes(&e);
    // tag 6, type id (u16 BE), payload length (u16 BE), then the payload
    let tail = &b[b.len() - (1 + 2 + 2 + 3)..];
    assert_eq!(tail, &[6, 0, 7, 0, 3, 0, 1, 2]);
}

struct Policy(BTreeMap<String, PublicKey>);
impl TrustPolicy for Policy {
    fn master_of(&self, subject: &Did) -> Option<PublicKey> {
        self.0.get(&subject.to_string()).copied()
    }
    fn revocation(&self) -> Revocation<'_> {
        Revocation::NoListAvailable
    }
    fn now(&self) -> u64 {
        NOW
    }
}

fn node(id: u64, seed: u8, name: &str) -> (Node, Policy) {
    let master = KeyPair::from_seed([seed; 32]);
    let key = KeyPair::from_seed([seed + 100; 32]);
    let did = Did::parse(&format!("did:alice:{name}")).unwrap();
    let cert = DeviceCert::issue(
        &master,
        &did,
        key.public().0,
        [0x4e; 32],
        1_800_000_000,
        1_800_086_400,
    )
    .unwrap();
    let policy = Policy([(did.to_string(), master.public())].into_iter().collect());
    (
        Node::with_author(NodeId(id), Author::new(key, cert).unwrap()),
        policy,
    )
}

#[test]
fn payloads_up_to_the_bound_pass_the_signed_chain() {
    let (mut a, trust_a) = node(1, 1, "alice");
    let (mut b, _) = node(2, 2, "bob");
    for len in [0, 1, MAX_CUSTOM_PAYLOAD] {
        a.emit(custom(len)).unwrap();
    }
    let batch = a.signed_events_since(&BTreeMap::new()).unwrap();
    assert_eq!(b.receive(&batch, &trust_a).unwrap(), 3);
    let got: Vec<usize> = batch
        .events
        .iter()
        .map(|e| match &e.kind {
            EventKind::Custom { payload, .. } => payload.as_bytes().len(),
            _ => usize::MAX,
        })
        .collect();
    assert_eq!(got, vec![0, 1, MAX_CUSTOM_PAYLOAD]);
}

#[test]
fn a_signed_batch_with_a_payload_over_the_bound_is_refused() {
    use alice_sync::authorship::AuthorshipError;
    let (mut a, trust_a) = node(1, 1, "alice");
    let (mut b, _) = node(2, 2, "bob");
    a.emit(custom(1)).unwrap();
    let mut batch = a.signed_events_since(&BTreeMap::new()).unwrap();
    // an event decoded without the check, as a decoder that skipped it would hand over
    let mut long: Event = bitcode::decode(&forged(MAX_CUSTOM_PAYLOAD + 1)).unwrap();
    long.origin = batch.events[0].origin;
    long.seq = batch.events[0].seq;
    batch.events[0] = long;
    assert!(matches!(
        b.receive(&batch, &trust_a),
        Err(AuthorshipError::Malformed { .. })
    ));
}
