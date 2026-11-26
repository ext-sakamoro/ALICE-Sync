//! Oracle for the kind of peer in the checks the secure transport repeats on open channels
//! ([`SecureTransport::revalidate`]): a channel to a service, opened under a policy that expects that
//! service, is checked again by the service certificate rules, and a service certificate is never kept
//! under the device certificate rules of a person.
//!
//! The listener trusts the service `relay` (subject `did:alice-service:relay`) under its operator's master
//! key. Its policy can switch the kind it expects, so the same open channel is checked once as a service
//! (kept) and once as a person (closed), and the service's 30-day bound closes it when the clock passes it.
//! Its policy can also come to hold the operator's revocation list: once the list names the service's
//! device id, the open channel is closed (the revocation reaches the service certificate rules).
#![cfg(feature = "secure-transport")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use alice_channel::noise::Keypair;
use alice_channel::{Credential, PeerKind, PeerPolicy};
use alice_identity::artifact::Revocation;
use alice_identity::cert_revocation::{CertRevocations, RevocationList};
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_identity::service_cert::{
    issue_service_cert, service_subject, SERVICE_CERT_MAX_LIFETIME_SECS,
};
use alice_sign::{KeyPair, PublicKey};
use alice_sync::protocol::Message;
use alice_sync::secure_transport::SecureTransport;
use alice_sync::transport::SyncTransport;
use tokio::time::timeout;

const WAIT: Duration = Duration::from_secs(5);
const ISSUED: u64 = 1_800_000_000;
const SERVICE: &str = "relay";

/// The operator's master key and the person's master key of the fixture (fixed seeds: test only)
fn operator() -> KeyPair {
    KeyPair::from_seed([0x31; 32])
}

fn person_master() -> KeyPair {
    KeyPair::from_seed([0x41; 32])
}

fn person_did() -> Did {
    Did::parse("did:alice:alice").unwrap()
}

/// The service's credential: its signing key and Noise static key under a 30-day service certificate
fn service_credential() -> Credential {
    let key = KeyPair::from_seed([0x32; 32]);
    let noise = Keypair::from_secret([0x33; 32]);
    let cert = issue_service_cert(
        &operator(),
        SERVICE,
        key.public().0,
        noise.public(),
        ISSUED,
        ISSUED + SERVICE_CERT_MAX_LIFETIME_SECS,
    )
    .unwrap();
    Credential::new(key, cert, noise).unwrap()
}

fn person_credential() -> Credential {
    let key = KeyPair::from_seed([0x42; 32]);
    let noise = Keypair::from_secret([0x43; 32]);
    let cert = DeviceCert::issue(
        &person_master(),
        &person_did(),
        key.public().0,
        noise.public(),
        ISSUED,
        ISSUED + 365 * 86_400,
    )
    .unwrap();
    Credential::new(key, cert, noise).unwrap()
}

/// The listener's policy: the service's subject under the operator key, the kind it expects switchable,
/// a clock that can move while the channel is open, and the operator's revocations once they are set
#[derive(Clone)]
struct TrustsTheService {
    as_service: Arc<AtomicBool>,
    now: Arc<AtomicU64>,
    revocations: Arc<OnceLock<CertRevocations>>,
}

impl PeerPolicy for TrustsTheService {
    fn master_of(&self, subject: &Did) -> Option<PublicKey> {
        (*subject == service_subject(SERVICE).unwrap()).then(|| operator().public())
    }
    fn revocation(&self) -> Revocation<'_> {
        self.revocations
            .get()
            .map_or(Revocation::NoListAvailable, Revocation::Checked)
    }
    fn now(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
    fn expected_peer(&self) -> PeerKind<'_> {
        if self.as_service.load(Ordering::SeqCst) {
            PeerKind::Service(SERVICE)
        } else {
            PeerKind::Person
        }
    }
}

/// The service's policy: the person's subject under the person's master key
struct TrustsThePerson;

impl PeerPolicy for TrustsThePerson {
    fn master_of(&self, subject: &Did) -> Option<PublicKey> {
        (*subject == person_did()).then(|| person_master().public())
    }
    fn revocation(&self) -> Revocation<'_> {
        Revocation::NoListAvailable
    }
    fn now(&self) -> u64 {
        ISSUED + 100
    }
}

/// The person's listener with an open channel from the service
async fn connected() -> (
    Arc<SecureTransport<TrustsTheService>>,
    TrustsTheService,
    SecureTransport<TrustsThePerson>,
) {
    let policy = TrustsTheService {
        as_service: Arc::new(AtomicBool::new(true)),
        now: Arc::new(AtomicU64::new(ISSUED + 100)),
        revocations: Arc::new(OnceLock::new()),
    };
    let person = person_credential();
    let person_cert = person.cert().clone();
    let tp = Arc::new(
        SecureTransport::bind("127.0.0.1:0", person, policy.clone())
            .await
            .unwrap(),
    );
    let _serve = tp.clone().serve();
    let to = tp.local_addr().unwrap();
    let ts = SecureTransport::bind("127.0.0.1:0", service_credential(), TrustsThePerson)
        .await
        .unwrap();
    ts.add_peer_cert(to, person_cert).await;
    ts.send_to(&Message::Bye, to).await.unwrap();
    let v = timeout(WAIT, tp.recv_verified()).await.unwrap().unwrap();
    assert_eq!(v.peer.subject(), &service_subject(SERVICE).unwrap());
    (tp, policy, ts)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_service_channel_is_kept_by_the_service_rules() {
    let (tp, _policy, _ts) = connected().await;
    assert_eq!(tp.peers().await.len(), 1);
    assert!(tp.revalidate().await.is_empty());
    assert_eq!(tp.peers().await.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_service_channel_is_closed_when_the_policy_expects_a_person() {
    let (tp, policy, _ts) = connected().await;
    let open = tp.peers().await;
    assert_eq!(open.len(), 1);
    policy.as_service.store(false, Ordering::SeqCst);
    assert_eq!(tp.revalidate().await, open);
    assert!(tp.peers().await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_service_channel_is_closed_at_the_end_of_its_certificate() {
    let (tp, policy, _ts) = connected().await;
    let open = tp.peers().await;
    policy.now.store(
        ISSUED + SERVICE_CERT_MAX_LIFETIME_SECS - 1,
        Ordering::SeqCst,
    );
    assert!(tp.revalidate().await.is_empty());
    policy
        .now
        .store(ISSUED + SERVICE_CERT_MAX_LIFETIME_SECS, Ordering::SeqCst);
    assert_eq!(tp.revalidate().await, open);
}

/// The operator's revocations holding `list`, accepted under the operator key
fn revocations(list: RevocationList) -> CertRevocations {
    let mut kept = CertRevocations::new();
    kept.accept(list, &operator().public()).unwrap();
    kept
}

#[tokio::test(flavor = "multi_thread")]
async fn a_service_channel_is_closed_once_the_operator_revokes_its_device() {
    let (tp, policy, _ts) = connected().await;
    let open = tp.peers().await;
    assert_eq!(open.len(), 1);
    let revoked = service_credential().cert().device_id();
    let list = RevocationList::issue(&operator(), 1, ISSUED + 200, &[revoked]).unwrap();
    assert!(policy.revocations.set(revocations(list)).is_ok());
    assert_eq!(tp.revalidate().await, open);
    assert!(tp.peers().await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_service_channel_is_kept_under_a_list_that_does_not_name_it() {
    let (tp, policy, _ts) = connected().await;
    let other = person_credential().cert().device_id();
    let list = RevocationList::issue(&operator(), 1, ISSUED + 200, &[other]).unwrap();
    assert!(policy.revocations.set(revocations(list)).is_ok());
    assert!(tp.revalidate().await.is_empty());
    assert_eq!(tp.peers().await.len(), 1);
}
