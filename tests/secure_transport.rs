//! Oracle for the secure Sync transport: Sync messages arrive whole and in order between certified devices
//! over real TCP sockets, unknown devices and silent connections are refused without becoming peers, and a
//! verified peer that breaks the message framing loses its connection.
#![cfg(feature = "secure-transport")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use alice_channel::noise::Keypair;
use alice_channel::{AsyncSecureChannel, Credential, PeerPolicy};
use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use alice_sync::event::{Event, EventId, EventKind, SeqNum};
use alice_sync::node::NodeId;
use alice_sync::protocol::Message;
use alice_sync::secure_transport::{Body, SecureTransport, KIND_BYTES, KIND_MESSAGE, MAX_MESSAGE};
use alice_sync::transport::{SyncTransport, TransportError};
use alice_sync::world::WorldHash;
use tokio::net::TcpStream;
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
        let cert = DeviceCert::issue(
            &self.master,
            &self.did,
            key.public().0,
            1_800_000_000,
            1_800_086_400,
            [self.seed; 16],
        )
        .unwrap();
        Credential::new(
            key,
            cert,
            Keypair::from_secret([self.seed.wrapping_add(200); 32]),
        )
        .unwrap()
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

async fn transport(d: &Device, knows: &[&Device]) -> Arc<SecureTransport<Policy>> {
    Arc::new(
        SecureTransport::bind("127.0.0.1:0", d.credential(), knowing(knows))
            .await
            .unwrap(),
    )
}

fn hello(seq: u64) -> Message {
    Message::Hello {
        node_id: NodeId(7),
        seq,
        world_hash: WorldHash(0xfeed),
        semantics_id: [3; 32],
    }
}

/// A message of `n` events whose bytes depend on `tag`, so that two of them never encode alike
fn events(n: u64, tag: u8) -> Message {
    Message::Events {
        events: (0..n)
            .map(|i| Event {
                id: EventId(i),
                seq: SeqNum(i),
                origin: u64::from(tag),
                timestamp: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                kind: EventKind::Custom {
                    type_id: u16::from(tag),
                    payload: (i ^ u64::from(tag))
                        .wrapping_mul(0xd6e8_feb8_6659_fd93)
                        .to_le_bytes()
                        .repeat(2)
                        .try_into()
                        .unwrap(),
                },
            })
            .collect(),
    }
}

fn bytes(m: &Message) -> Vec<u8> {
    m.to_compact_bytes()
}

#[tokio::test(flavor = "multi_thread")]
async fn messages_travel_both_ways_after_an_explicit_connect() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let tb = transport(&b, &[&a]).await;
    let _serve = ta.clone().serve();
    let addr_a = ta.local_addr().unwrap();
    tb.connect(addr_a).await.unwrap();
    assert_eq!(
        tb.peer_cert(addr_a).await.unwrap().subject().to_string(),
        "did:alice:alice"
    );
    for seq in 0..20 {
        tb.send_to(&hello(seq), addr_a).await.unwrap();
    }
    let mut from = None;
    for seq in 0..20 {
        let env = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
        assert_eq!(bytes(&env.message), bytes(&hello(seq)), "in order");
        from = Some(env.from);
    }
    let from = from.unwrap();
    assert_eq!(
        ta.peer_cert(from).await.unwrap().subject().to_string(),
        "did:alice:bob"
    );
    ta.send_to(&Message::Ack { seq: 99 }, from).await.unwrap();
    let back = timeout(WAIT, tb.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&back.message), bytes(&Message::Ack { seq: 99 }));
    assert_eq!(back.from, addr_a);
}

#[tokio::test(flavor = "multi_thread")]
async fn send_to_an_address_without_a_connection_dials_it() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let tb = transport(&b, &[&a]).await;
    let _serve = ta.clone().serve();
    let addr_a = ta.local_addr().unwrap();
    assert!(tb.peers().await.is_empty());
    tb.send_to(&Message::Bye, addr_a).await.unwrap();
    assert_eq!(tb.peers().await, vec![addr_a]);
    let env = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&env.message), bytes(&Message::Bye));
    // a second send uses the same connection
    tb.send_to(&Message::Ack { seq: 1 }, addr_a).await.unwrap();
    assert_eq!(
        timeout(WAIT, ta.recv()).await.unwrap().unwrap().from,
        env.from
    );
    assert_eq!(ta.peers().await.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_message_larger_than_one_channel_message_arrives_whole() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let tb = transport(&b, &[&a]).await;
    let _serve = ta.clone().serve();
    let addr_a = ta.local_addr().unwrap();
    let big = events(12_000, 5);
    let len = bytes(&big).len();
    assert!(
        len > 3 * alice_channel::MAX_PAYLOAD && len < MAX_MESSAGE,
        "{len} bytes"
    );
    tb.send_to(&big, addr_a).await.unwrap();
    tb.send_to(&Message::Bye, addr_a).await.unwrap();
    let got = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&got.message), bytes(&big));
    let next = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&next.message), bytes(&Message::Bye));
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_large_sends_to_one_peer_do_not_interleave() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let tb = transport(&b, &[&a]).await;
    let _serve = ta.clone().serve();
    let addr_a = ta.local_addr().unwrap();
    tb.connect(addr_a).await.unwrap();
    let sent: Vec<Vec<u8>> = (0..8u8).map(|t| bytes(&events(6_000, t))).collect();
    let mut tasks = Vec::new();
    for t in 0..8u8 {
        let tb = tb.clone();
        tasks.push(tokio::spawn(async move {
            tb.send_to(&events(6_000, t), addr_a).await.unwrap();
        }));
    }
    let mut got = Vec::new();
    for _ in 0..8 {
        got.push(bytes(
            &timeout(WAIT, ta.recv()).await.unwrap().unwrap().message,
        ));
    }
    for t in tasks {
        t.await.unwrap();
    }
    got.sort();
    let mut want = sent;
    want.sort();
    assert_eq!(got, want);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_message_over_the_limit_is_refused_before_anything_is_sent() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let tb = transport(&b, &[&a]).await;
    let _serve = ta.clone().serve();
    let addr_a = ta.local_addr().unwrap();
    tb.connect(addr_a).await.unwrap();
    let huge = events(60_000, 1);
    assert!(bytes(&huge).len() > MAX_MESSAGE);
    assert!(matches!(
        tb.send_to(&huge, addr_a).await,
        Err(TransportError::Codec(_))
    ));
    // nothing of it reached the peer, and the connection still carries the next message
    tb.send_to(&Message::Bye, addr_a).await.unwrap();
    let env = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&env.message), bytes(&Message::Bye));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_device_is_refused_and_becomes_no_peer() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[]).await;
    let tb = transport(&b, &[&a]).await;
    let _serve = ta.clone().serve();
    let addr_a = ta.local_addr().unwrap();
    let sent = tb.send_to(&Message::Bye, addr_a).await;
    assert!(sent.is_err(), "the dialling side gets no channel");
    assert!(tb.peers().await.is_empty());
    timeout(WAIT, async {
        while ta.refused_count() == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(ta.peers().await.is_empty());
    assert!(timeout(Duration::from_millis(200), ta.recv())
        .await
        .is_err());
    // the accept loop goes on serving after a refusal
    let c = device(3, "carol");
    drop(ta);
    let ta2 = transport(&a, &[&c]).await;
    let _serve2 = ta2.clone().serve();
    let addr2 = ta2.local_addr().unwrap();
    let tb2 = transport(&b, &[&a]).await;
    assert!(tb2.send_to(&Message::Bye, addr2).await.is_err());
    let tc = transport(&c, &[&a]).await;
    tc.send_to(&Message::Bye, addr2).await.unwrap();
    let env = timeout(WAIT, ta2.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&env.message), bytes(&Message::Bye));
    assert_eq!(ta2.refused_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_silent_connection_times_out_in_the_handshake() {
    let a = device(1, "alice");
    let ta = SecureTransport::bind_with_timeout(
        "127.0.0.1:0",
        a.credential(),
        knowing(&[]),
        Duration::from_millis(200),
    )
    .await
    .unwrap();
    let addr_a = ta.local_addr().unwrap();
    let _silent = TcpStream::connect(addr_a).await.unwrap();
    let accepted = timeout(WAIT, ta.accept()).await.unwrap();
    assert!(
        matches!(&accepted, Err(TransportError::Io(e)) if e.kind() == std::io::ErrorKind::TimedOut),
        "{accepted:?}"
    );
    assert_eq!(ta.refused_count(), 1);
    assert!(ta.peers().await.is_empty());
}

/// The head of a first channel message: the kind byte, then the total length (4 bytes, little-endian)
fn head(kind: u8, total: usize) -> Vec<u8> {
    let mut h = vec![kind];
    h.extend_from_slice(&u32::try_from(total).unwrap().to_le_bytes());
    h
}

/// A verified peer speaking the channel directly, to break the message framing
async fn raw_peer(
    d: &Device,
    knows: &[&Device],
    to: std::net::SocketAddr,
) -> AsyncSecureChannel<TcpStream> {
    let stream = TcpStream::connect(to).await.unwrap();
    AsyncSecureChannel::connect(stream, &d.credential(), &knowing(knows))
        .await
        .unwrap()
}

async fn wait_no_peers(t: &SecureTransport<Policy>) {
    timeout(WAIT, async {
        while !t.peers().await.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

async fn breaks_framing(frames: Vec<Vec<u8>>) {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let _serve = ta.clone().serve();
    let mut raw = raw_peer(&b, &[&a], ta.local_addr().unwrap()).await;
    timeout(WAIT, async {
        while ta.peers().await.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    for f in &frames {
        // the transport may already have hung up after an earlier frame
        if raw.send(f).await.is_err() {
            break;
        }
    }
    wait_no_peers(&ta).await;
    assert!(timeout(Duration::from_millis(200), ta.recv())
        .await
        .is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_peer_announcing_a_message_over_the_limit_loses_its_connection() {
    breaks_framing(vec![head(KIND_MESSAGE, MAX_MESSAGE + 1)]).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_first_message_without_the_length_loses_the_connection() {
    breaks_framing(vec![vec![KIND_MESSAGE, 2, 3, 4]]).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_part_overrunning_the_announced_length_loses_the_connection() {
    let msg = bytes(&Message::Bye);
    let mut first = head(KIND_MESSAGE, msg.len() + 2);
    first.extend_from_slice(&msg);
    breaks_framing(vec![first, vec![0; 5]]).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_head_longer_than_the_announced_length_loses_the_connection() {
    let msg = bytes(&Message::Bye);
    let mut first = head(KIND_MESSAGE, msg.len() - 1);
    first.extend_from_slice(&msg);
    breaks_framing(vec![first]).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_well_framed_message_from_a_raw_peer_is_received() {
    // the framing the transport writes, written by hand: the reference the refusals above are measured against
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let _serve = ta.clone().serve();
    let mut raw = raw_peer(&b, &[&a], ta.local_addr().unwrap()).await;
    let msg = bytes(&hello(4));
    let (front, tail) = msg.split_at(msg.len() / 2);
    let mut first = head(KIND_MESSAGE, msg.len());
    first.extend_from_slice(front);
    raw.send(&first).await.unwrap();
    raw.send(tail).await.unwrap();
    let env = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
    assert_eq!(bytes(&env.message), msg);
    // and the transport's own sends are read back by the raw peer in the same framing
    ta.send_to(&Message::Ack { seq: 3 }, env.from)
        .await
        .unwrap();
    let back = raw.recv().await.unwrap();
    let want = bytes(&Message::Ack { seq: 3 });
    assert_eq!(back[..5], head(KIND_MESSAGE, want.len())[..]);
    assert_eq!(&back[5..], &want[..]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_closed_peer_is_dropped_and_dialled_again_on_the_next_send() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let tb = transport(&b, &[&a]).await;
    // alice here is a bare listener answering with the channel, so the test can hang up on bob
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_a = listener.local_addr().unwrap();
    let ca = a.credential();
    let pa = knowing(&[&b]);
    let (first, second) = tokio::join!(
        async {
            let (s, _) = listener.accept().await.unwrap();
            AsyncSecureChannel::accept(s, &ca, &pa).await.unwrap()
        },
        tb.send_to(&Message::Ack { seq: 1 }, addr_a)
    );
    second.unwrap();
    let mut first = first;
    let got = Message::from_compact_bytes(&first.recv().await.unwrap()[5..]).unwrap();
    assert_eq!(bytes(&got), bytes(&Message::Ack { seq: 1 }));
    drop(first);
    // bob's reader sees the hang-up and drops the peer
    wait_no_peers(&tb).await;
    let (again, sent) = tokio::join!(
        async {
            let (s, _) = listener.accept().await.unwrap();
            AsyncSecureChannel::accept(s, &ca, &pa).await.unwrap()
        },
        tb.send_to(&Message::Ack { seq: 2 }, addr_a)
    );
    sent.unwrap();
    let mut again = again;
    let got = Message::from_compact_bytes(&again.recv().await.unwrap()[5..]).unwrap();
    assert_eq!(bytes(&got), bytes(&Message::Ack { seq: 2 }));
}

/// A transport and a raw peer (bob) connected to it, with the peer seen on the transport's side
async fn with_raw_peer() -> (Arc<SecureTransport<Policy>>, AsyncSecureChannel<TcpStream>) {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let _serve = ta.clone().serve();
    let raw = raw_peer(&b, &[&a], ta.local_addr().unwrap()).await;
    timeout(WAIT, async {
        while ta.peers().await.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    (ta, raw)
}

#[tokio::test(flavor = "multi_thread")]
async fn application_bytes_arrive_as_bytes_with_the_senders_certificate() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let _serve = ta.clone().serve();
    let tb = transport(&b, &[&a]).await;
    let big: Vec<u8> = (0..3 * alice_channel::MAX_PAYLOAD + 17)
        .map(|i| (i % 251) as u8)
        .collect();
    for sent in [Vec::new(), vec![9, 8, 7], big] {
        tb.send_bytes_to(&sent, ta.local_addr().unwrap())
            .await
            .unwrap();
        let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
        assert!(matches!(&v.body, Body::Bytes(got) if *got == sent));
        assert_eq!(v.peer.subject().to_string(), "did:alice:bob");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bytes_that_encode_a_sync_message_never_arrive_as_one() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let _serve = ta.clone().serve();
    let tb = transport(&b, &[&a]).await;
    let to = ta.local_addr().unwrap();
    let looks_like = bytes(&Message::Ack { seq: 9 });
    tb.send_bytes_to(&looks_like, to).await.unwrap();
    tb.send_to(&Message::Ack { seq: 10 }, to).await.unwrap();
    let first = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert!(matches!(&first.body, Body::Bytes(got) if *got == looks_like));
    let second = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert!(matches!(
        second.body,
        Body::Message(Message::Ack { seq: 10 })
    ));
    // the Sync receive passes the bytes over and counts them
    tb.send_bytes_to(&looks_like, to).await.unwrap();
    tb.send_to(&Message::Ack { seq: 11 }, to).await.unwrap();
    let env = timeout(WAIT, ta.recv()).await.unwrap().unwrap();
    assert!(matches!(env.message, Message::Ack { seq: 11 }));
    assert_eq!(ta.skipped_bytes_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn bytes_over_the_limit_are_refused_before_anything_is_sent() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = transport(&a, &[&b]).await;
    let _serve = ta.clone().serve();
    let tb = transport(&b, &[&a]).await;
    let to = ta.local_addr().unwrap();
    assert!(matches!(
        tb.send_bytes_to(&vec![0; MAX_MESSAGE + 1], to).await,
        Err(TransportError::Codec(_))
    ));
    assert!(tb.peers().await.is_empty(), "nothing was dialled");
    // the limit itself is sent
    tb.send_bytes_to(&vec![5; MAX_MESSAGE], to).await.unwrap();
    let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert!(matches!(&v.body, Body::Bytes(got) if got.len() == MAX_MESSAGE));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_peer_announcing_bytes_over_the_limit_loses_its_connection() {
    breaks_framing(vec![head(KIND_BYTES, MAX_MESSAGE + 1)]).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_kind_other_than_message_or_bytes_loses_the_connection() {
    for kind in [2, 0x7f, u8::MAX] {
        let mut first = head(kind, 3);
        first.extend_from_slice(&[1, 2, 3]);
        breaks_framing(vec![first]).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn both_kinds_written_by_hand_are_received() {
    // the reference the kind refusals above are measured against
    let (ta, mut raw) = with_raw_peer().await;
    let msg = bytes(&Message::Bye);
    let mut first = head(KIND_MESSAGE, msg.len());
    first.extend_from_slice(&msg);
    raw.send(&first).await.unwrap();
    let mut first = head(KIND_BYTES, msg.len());
    first.extend_from_slice(&msg);
    raw.send(&first).await.unwrap();
    let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert!(matches!(v.body, Body::Message(Message::Bye)));
    let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert!(matches!(&v.body, Body::Bytes(got) if *got == msg));
    // and the transport writes bytes with the bytes kind
    ta.send_bytes_to(&[4, 2], v.from).await.unwrap();
    let back = raw.recv().await.unwrap();
    assert_eq!(back, [head(KIND_BYTES, 2), vec![4, 2]].concat());
}
