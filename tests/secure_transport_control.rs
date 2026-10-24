//! Oracle for the controls of the secure transport: each received message carries the certificate of
//! the connection it came on, a channel can be closed by address, open channels are checked again
//! against the policy as it stands now, and a dialling-only transport opens no listener.
#![cfg(feature = "secure-transport")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use alice_channel::noise::Keypair;
use alice_channel::{Credential, PeerPolicy};
use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use alice_sync::protocol::Message;
use alice_sync::secure_transport::{Body, SecureTransport};
use alice_sync::transport::{SyncTransport, TransportError};
use tokio::time::timeout;

const WAIT: Duration = Duration::from_secs(5);
const ISSUED: u64 = 1_800_000_000;
const EXPIRES: u64 = 1_800_086_400;

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
            ISSUED,
            EXPIRES,
            [self.seed; 16],
        )
        .unwrap();
        Credential::new(key, cert, noise).unwrap()
    }

    fn cert(&self) -> DeviceCert {
        self.credential().cert().clone()
    }
}

/// A trust set and a clock that can change while channels are open
#[derive(Clone, Default)]
struct Shared {
    known: Arc<Mutex<BTreeMap<String, PublicKey>>>,
    now: Arc<AtomicU64>,
}

impl Shared {
    fn knowing(ds: &[&Device]) -> Self {
        let s = Self::default();
        s.now.store(ISSUED + 100, Ordering::SeqCst);
        for d in ds {
            s.trust(d);
        }
        s
    }
    fn trust(&self, d: &Device) {
        self.known
            .lock()
            .unwrap()
            .insert(d.did.to_string(), d.master.public());
    }
    fn withdraw(&self, d: &Device) {
        self.known.lock().unwrap().remove(&d.did.to_string());
    }
}

impl PeerPolicy for Shared {
    fn master_of(&self, subject: &Did) -> Option<PublicKey> {
        self.known
            .lock()
            .unwrap()
            .get(&subject.to_string())
            .copied()
    }
    fn revocation(&self) -> Revocation<'_> {
        Revocation::NoListAvailable
    }
    fn now(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
}

async fn listener(d: &Device, policy: Shared) -> Arc<SecureTransport<Shared>> {
    Arc::new(
        SecureTransport::bind("127.0.0.1:0", d.credential(), policy)
            .await
            .unwrap(),
    )
}

async fn until(mut f: impl FnMut() -> bool) {
    timeout(WAIT, async {
        while !f() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn each_received_message_carries_the_certificate_it_came_with() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let c = device(3, "carol");
    let ta = listener(&a, Shared::knowing(&[&b, &c])).await;
    let _serve = ta.clone().serve();
    let to = ta.local_addr().unwrap();
    let tb = listener(&b, Shared::knowing(&[&a])).await;
    let tc = listener(&c, Shared::knowing(&[&a])).await;
    tb.add_peer_cert(to, a.cert()).await;
    tc.add_peer_cert(to, a.cert()).await;
    tb.send_to(&Message::Ack { seq: 2 }, to).await.unwrap();
    tc.send_to(&Message::Ack { seq: 3 }, to).await.unwrap();
    let mut seen = BTreeMap::new();
    for _ in 0..2 {
        let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
        let Body::Message(Message::Ack { seq }) = v.body else {
            panic!("not an ack")
        };
        seen.insert(seq, v.peer.subject().to_string());
        assert_eq!(
            ta.peer_cert(v.from).await.unwrap().subject(),
            v.peer.subject()
        );
    }
    assert_eq!(seen[&2], "did:alice:bob");
    assert_eq!(seen[&3], "did:alice:carol");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_closed_channel_is_gone_on_both_sides() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = listener(&a, Shared::knowing(&[&b])).await;
    let _serve = ta.clone().serve();
    let to = ta.local_addr().unwrap();
    let tb = listener(&b, Shared::knowing(&[&a])).await;
    tb.add_peer_cert(to, a.cert()).await;
    tb.send_to(&Message::Bye, to).await.unwrap();
    let from = timeout(WAIT, ta.recv_verified())
        .await
        .unwrap()
        .unwrap()
        .from;
    assert!(ta.close(from).await);
    assert!(!ta.close(from).await, "closing twice finds nothing");
    assert!(ta.peers().await.is_empty());
    // the other side's reader sees the channel end
    let tb2 = tb.clone();
    timeout(WAIT, async move {
        while !tb2.peers().await.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn revalidate_closes_the_peers_the_policy_now_refuses() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let c = device(3, "carol");
    let policy = Shared::knowing(&[&b, &c]);
    let ta = listener(&a, policy.clone()).await;
    let _serve = ta.clone().serve();
    let to = ta.local_addr().unwrap();
    let tb = listener(&b, Shared::knowing(&[&a])).await;
    let tc = listener(&c, Shared::knowing(&[&a])).await;
    tb.add_peer_cert(to, a.cert()).await;
    tc.add_peer_cert(to, a.cert()).await;
    tb.send_to(&Message::Bye, to).await.unwrap();
    tc.send_to(&Message::Bye, to).await.unwrap();
    let mut from = BTreeMap::new();
    for _ in 0..2 {
        let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
        from.insert(v.peer.subject().to_string(), v.from);
    }
    // nothing changed: nobody is closed
    assert!(ta.revalidate().await.is_empty());
    assert_eq!(ta.peers().await.len(), 2);
    // bob's master key is withdrawn: bob's channel is closed, carol's stays
    policy.withdraw(&b);
    assert_eq!(ta.revalidate().await, vec![from["did:alice:bob"]]);
    assert_eq!(ta.peers().await, vec![from["did:alice:carol"]]);
    // the clock passes the certificates' end: carol is closed too
    policy.now.store(EXPIRES + 1, Ordering::SeqCst);
    assert_eq!(ta.revalidate().await, vec![from["did:alice:carol"]]);
    assert!(ta.peers().await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dialler_opens_no_listener_and_still_gets_replies() {
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = listener(&a, Shared::knowing(&[&b])).await;
    let _serve = ta.clone().serve();
    let to = ta.local_addr().unwrap();
    let tb = SecureTransport::dialer(b.credential(), Shared::knowing(&[&a]));
    assert!(matches!(
        tb.local_addr(),
        Err(TransportError::Io(e)) if e.kind() == std::io::ErrorKind::NotConnected
    ));
    assert!(matches!(
        timeout(WAIT, tb.accept()).await.unwrap(),
        Err(TransportError::Io(e)) if e.kind() == std::io::ErrorKind::NotConnected
    ));
    tb.add_peer_cert(to, a.cert()).await;
    tb.send_to(&Message::Ack { seq: 7 }, to).await.unwrap();
    let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert_eq!(v.peer.subject().to_string(), "did:alice:bob");
    // the reply goes back over the same channel
    ta.send_to(&Message::Ack { seq: 8 }, v.from).await.unwrap();
    let back = timeout(WAIT, tb.recv_verified()).await.unwrap().unwrap();
    assert_eq!(back.from, to);
    assert_eq!(back.peer.subject().to_string(), "did:alice:alice");
    let tb = Arc::new(tb);
    let served = timeout(WAIT, tb.clone().serve()).await.unwrap().unwrap();
    assert!(
        matches!(served, TransportError::Io(e) if e.kind() == std::io::ErrorKind::NotConnected)
    );
    until(|| true).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn every_dial_checks_the_book_certificate_under_the_policy_as_it_stands() {
    // the certificate is in the book, but alice's master is withdrawn from bob's policy: the dial is
    // refused before a byte is written, and alice sees no handshake
    let a = device(1, "alice");
    let b = device(2, "bob");
    let ta = listener(&a, Shared::knowing(&[&b])).await;
    let _serve = ta.clone().serve();
    let to = ta.local_addr().unwrap();
    let policy = Shared::knowing(&[&a]);
    let tb = listener(&b, policy.clone()).await;
    tb.add_peer_cert(to, a.cert()).await;
    policy.withdraw(&a);
    assert!(tb.send_to(&Message::Bye, to).await.is_err());
    assert!(tb.peers().await.is_empty());
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(ta.refused_count(), 0);
    assert!(ta.peers().await.is_empty());
    // trusted again: the same book entry dials
    policy.trust(&a);
    tb.send_to(&Message::Bye, to).await.unwrap();
    let v = timeout(WAIT, ta.recv_verified()).await.unwrap().unwrap();
    assert_eq!(v.peer.subject().to_string(), "did:alice:bob");
    // and after the clock passes alice's certificate, the next dial is refused again
    assert!(tb.close(to).await);
    policy.now.store(EXPIRES, Ordering::SeqCst);
    assert!(tb.send_to(&Message::Bye, to).await.is_err());
    assert!(tb.peers().await.is_empty());
}
