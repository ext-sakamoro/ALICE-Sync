//! Oracle for a SyncSession built on the secure transport: application bytes travel between two
//! certified devices only after each side has accepted the other's Hello, in both directions, and the
//! session's `recv` still returns Sync messages only.
#![cfg(feature = "secure-transport")]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

use alice_channel::noise::Keypair;
use alice_channel::{Credential, PeerPolicy};
use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use alice_sync::protocol::{HandshakeRefused, Message};
use alice_sync::secure_transport::SecureTransport;
use alice_sync::session::{SessionBuilder, SyncSession};
use alice_sync::transport::{Frame, SyncTransport, TransportError};
use alice_sync::world::WorldHash;
use tokio::time::timeout;

const NOW: u64 = 1_800_000_100;
const WAIT: Duration = Duration::from_secs(5);

struct Device {
    master: KeyPair,
    seed: u8,
    did: Did,
}

fn device(seed: u8, name: &str) -> Device {
    Device {
        master: KeyPair::from_seed([seed; 32]),
        seed,
        did: Did::parse(&format!("did:alice:{name}")).unwrap(),
    }
}

impl Device {
    fn credential(&self) -> Credential {
        let key = KeyPair::from_seed([self.seed.wrapping_add(100); 32]);
        let noise = Keypair::from_secret([self.seed.wrapping_add(200); 32]);
        let cert = DeviceCert::issue(
            &self.master,
            &self.did,
            key.public().0,
            noise.public(),
            1_800_000_000,
            1_800_086_400,
            [self.seed; 16],
        )
        .unwrap();
        Credential::new(key, cert, noise).unwrap()
    }

    fn cert(&self) -> DeviceCert {
        self.credential().cert().clone()
    }
}

struct Policy(BTreeMap<String, PublicKey>);

impl PeerPolicy for Policy {
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

fn knowing(ds: &[&Device]) -> Policy {
    Policy(
        ds.iter()
            .map(|d| (d.did.to_string(), d.master.public()))
            .collect(),
    )
}

type Session = SyncSession<SecureTransport<Policy>>;

/// Two sessions on secure transports: `a` listens, `b` holds `a`'s certificate and is connected to it
async fn pair() -> (Session, Session, SocketAddr) {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = SecureTransport::bind("127.0.0.1:0", a.credential(), knowing(&[&b]))
        .await
        .unwrap();
    let tb = SecureTransport::bind("127.0.0.1:0", b.credential(), knowing(&[&a]))
        .await
        .unwrap();
    let sa = SessionBuilder::new()
        .node_id(1)
        .discovery(false)
        .tick_rate(0)
        .build(ta);
    let sb = SessionBuilder::new()
        .node_id(2)
        .discovery(false)
        .tick_rate(0)
        .build(tb);
    let _serve = sa.transport().clone().serve();
    let addr_a = sa.transport().local_addr().unwrap();
    sb.transport().connect(addr_a, a.cert()).await.unwrap();
    (sa, sb, addr_a)
}

fn not_handshaken(r: Result<impl std::fmt::Debug, TransportError>) -> SocketAddr {
    match r {
        Err(TransportError::Handshake(HandshakeRefused::NotHandshaken { peer })) => peer,
        other => panic!("expected NotHandshaken, got {other:?}"),
    }
}

/// The Hello of `from` accepted by `to`, returning the address `to` knows `from` by
async fn hello(from: &Session, to: &Session, to_addr: SocketAddr) -> SocketAddr {
    from.handshake(to_addr, WorldHash(0xfeed)).await.unwrap();
    match timeout(WAIT, to.recv_frame()).await.unwrap().unwrap() {
        Frame::Message(env) => {
            assert!(matches!(env.message, Message::Hello { .. }));
            env.from
        }
        other => panic!("expected the Hello, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bytes_are_refused_both_ways_before_an_accepted_hello() {
    let (sa, sb, addr_a) = pair().await;
    // sending: b has accepted no Hello of a
    assert_eq!(
        not_handshaken(sb.send_bytes(addr_a, b"early").await),
        addr_a
    );
    // receiving: bytes put on the wire under the session (by the transport itself) before b's Hello
    sb.transport()
        .send_bytes_to(b"early", addr_a)
        .await
        .unwrap();
    let from_b = not_handshaken(timeout(WAIT, sa.recv_frame()).await.unwrap());
    // a message other than Hello is refused the same way
    sb.transport().send_to(&Message::Bye, addr_a).await.unwrap();
    assert_eq!(
        not_handshaken(timeout(WAIT, sa.recv_frame()).await.unwrap()),
        from_b
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn bytes_travel_both_ways_after_both_hellos() {
    let (sa, sb, addr_a) = pair().await;
    let from_b = hello(&sb, &sa, addr_a).await;
    // a accepted b's Hello, b has not accepted a's: a may send, b may not yet
    assert_eq!(not_handshaken(sb.send_bytes(addr_a, b"x").await), addr_a);
    assert_eq!(hello(&sa, &sb, from_b).await, addr_a);

    sb.send_bytes(addr_a, b"to alice").await.unwrap();
    match timeout(WAIT, sa.recv_frame()).await.unwrap().unwrap() {
        Frame::Bytes { bytes, from } => {
            assert_eq!((bytes.as_slice(), from), (&b"to alice"[..], from_b))
        }
        other => panic!("expected bytes, got {other:?}"),
    }
    sa.send_bytes(from_b, b"to bob").await.unwrap();
    match timeout(WAIT, sb.recv_frame()).await.unwrap().unwrap() {
        Frame::Bytes { bytes, from } => {
            assert_eq!((bytes.as_slice(), from), (&b"to bob"[..], addr_a))
        }
        other => panic!("expected bytes, got {other:?}"),
    }
    // a Sync message after bytes keeps its order in the one receive loop
    sb.send_bytes(addr_a, b"first").await.unwrap();
    sb.transport()
        .send_to(&Message::Ack { seq: 7 }, addr_a)
        .await
        .unwrap();
    assert!(matches!(
        timeout(WAIT, sa.recv_frame()).await.unwrap().unwrap(),
        Frame::Bytes { .. }
    ));
    match timeout(WAIT, sa.recv_frame()).await.unwrap().unwrap() {
        Frame::Message(env) => assert!(matches!(env.message, Message::Ack { seq: 7 })),
        other => panic!("expected the Ack, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recv_returns_messages_only_and_counts_the_bytes_it_passes_over() {
    let (sa, sb, addr_a) = pair().await;
    let from_b = hello(&sb, &sa, addr_a).await;
    hello(&sa, &sb, from_b).await;
    sb.send_bytes(addr_a, b"skipped").await.unwrap();
    sb.transport()
        .send_to(&Message::Ack { seq: 1 }, addr_a)
        .await
        .unwrap();
    let before = sa.transport().skipped_bytes_count();
    let env = timeout(WAIT, sa.recv()).await.unwrap().unwrap();
    assert!(matches!(env.message, Message::Ack { seq: 1 }));
    assert_eq!(sa.transport().skipped_bytes_count(), before + 1);
}
