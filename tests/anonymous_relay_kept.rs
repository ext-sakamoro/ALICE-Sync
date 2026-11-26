//! The relay's revocation list kept on disk (`KeptRevocations`) and the rule a connection path applies
//! (`RevocationRule`), feature `anonymous-relay-client` (Rust 1.89 or newer).
//!
//! The expected outcomes come from the rules the store states (a missing file is empty, anything else that
//! is wrong fails closed; the stored (epoch, sequence) never falls across a restart; nothing is written on
//! a refusal; a floor source above the stored pair refuses) and from the ordering rules of
//! `alice_identity::delegated_revocation`, written here as literals. The conflict digests are the SHA-256 of
//! the encoded lists, computed in the test with `sha2`. Every key is a fixed test-only seed.
#![cfg(feature = "anonymous-relay-client")]
// File::lock / unlock hold the store's lock in a test: stable from 1.89, the version this feature needs
#![allow(clippy::incompatible_msrv)]

use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use alice_identity::cert_revocation::{Accepted, RevocationList};
use alice_identity::delegated_revocation::{
    issue_revocation_delegate, DelegatedRevocationError as E, DelegatedRevocations,
    DEFAULT_REVOCATION_FRESHNESS_WINDOW_SECS,
};
use alice_identity::device_cert::{DeviceCert, DeviceId};
use alice_identity::service_cert::issue_service_cert;
use alice_sign::KeyPair;
use alice_sync::anonymous_relay_client::{
    fetch_relay_revocations, BodyError, DelegatedListResponse, FetchError, FloorError,
    FloorPending, FloorSource, KeptError, KeptRevocations, RevocationRule, RevocationRuleError,
    ServedNotUsed, DEFAULT_LOCK_WAIT, MAX_RELAY_DELEGATED_REVOCATIONS_LEN,
    RELAY_REVOCATIONS_BODY_VERSION,
};
use sha2::{Digest, Sha256};

const T0: u64 = 1_800_000_000;
const WINDOW: u64 = 86_400;
const NOW: u64 = T0 + 20;

fn master() -> KeyPair {
    KeyPair::from_seed([0x5a; 32])
}
fn other_master() -> KeyPair {
    KeyPair::from_seed([0x5b; 32])
}
fn delegate_key(n: u8) -> KeyPair {
    KeyPair::from_seed([0x20 + n; 32])
}
/// A delegate certificate for key `n`, epoch `epoch`, valid 7 days
fn delegate(n: u8, epoch: u64) -> DeviceCert {
    issue_revocation_delegate(
        &master(),
        delegate_key(n).public().0,
        epoch,
        epoch + 7 * 86_400,
    )
    .unwrap()
}
fn list(n: u8, seq: u64, at: u64, revoked: &[DeviceId]) -> RevocationList {
    RevocationList::issue(&delegate_key(n), seq, at, revoked).unwrap()
}
fn dev(n: u8) -> DeviceId {
    DeviceId::from_raw([n; 16])
}
/// The relay's service certificate under the test master
fn relay_cert() -> DeviceCert {
    issue_service_cert(
        &master(),
        "relay",
        KeyPair::from_seed([0x33; 32]).public().0,
        [0x07; 32],
        T0,
        T0 + 86_400,
    )
    .unwrap()
}

/// A directory of its own under the system temporary directory, removed on drop
struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("alice-sync-kept-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn file(&self) -> PathBuf {
        self.0.join("kept.bin")
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn open(path: &Path) -> Result<KeptRevocations, KeptError> {
    KeptRevocations::open(path, master().public())
}

/// The bytes and (on unix) the inode: a rewrite with the same bytes still changes the inode (rename)
fn snapshot(path: &Path) -> (Vec<u8>, u64) {
    let bytes = fs::read(path).unwrap();
    #[cfg(unix)]
    let ino = std::os::unix::fs::MetadataExt::ino(&fs::metadata(path).unwrap());
    #[cfg(not(unix))]
    let ino = 0;
    (bytes, ino)
}

/// A store holding `seq` of delegate 1 at epoch T0
fn stored(path: &Path, seq: u64) -> KeptRevocations {
    let mut kept = open(path).unwrap();
    assert_eq!(
        kept.accept(list(1, seq, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Ok(Accepted::Replaced)
    );
    kept
}

#[test]
fn constants_have_the_ruled_values() {
    assert_eq!(DEFAULT_LOCK_WAIT, Duration::from_secs(5));
    assert_eq!(RevocationRule::default().window, 86_400);
    assert_eq!(
        RevocationRule::default().window,
        DEFAULT_REVOCATION_FRESHNESS_WINDOW_SECS
    );
    assert!(RevocationRule::default().required);
    assert_eq!(RELAY_REVOCATIONS_BODY_VERSION, 1);
}

#[test]
fn a_missing_file_is_an_empty_store() {
    let s = Scratch::new("missing");
    let kept = open(&s.file()).unwrap();
    assert_eq!((kept.epoch(), kept.sequence()), (None, None));
    assert_eq!(
        kept.require_fresh(NOW, WINDOW),
        Err(E::NothingKept),
        "an empty store holds nothing fresh"
    );
    assert!(!s.file().exists(), "opening writes nothing");
}

#[test]
fn a_file_that_is_not_one_whole_state_fails_closed() {
    let s = Scratch::new("closed");
    let path = s.file();
    // empty
    fs::write(&path, b"").unwrap();
    assert_eq!(open(&path).err(), Some(KeptError::Empty));
    // a directory at the path
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert_eq!(open(&path).err(), Some(KeptError::NotAFile));
    fs::remove_dir(&path).unwrap();

    drop(stored(&path, 5));
    let whole = fs::read(&path).unwrap();
    // every truncation
    let mut compared = 0;
    for len in 1..whole.len() {
        fs::write(&path, &whole[..len]).unwrap();
        assert!(
            matches!(open(&path), Err(KeptError::Stored(_))),
            "truncated to {len}"
        );
        compared += 1;
    }
    // every bit
    for i in 0..whole.len() {
        for bit in 0..8 {
            let mut b = whole.clone();
            b[i] ^= 1 << bit;
            fs::write(&path, &b).unwrap();
            assert!(
                matches!(open(&path), Err(KeptError::Stored(_))),
                "byte {i} bit {bit}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, whole.len() - 1 + 8 * whole.len());
    assert!(compared > 0);
    // whole again, under another master
    fs::write(&path, &whole).unwrap();
    assert_eq!(
        KeptRevocations::open(&path, other_master().public()).err(),
        Some(KeptError::Stored(E::OtherMaster))
    );
    assert_eq!(open(&path).unwrap().sequence(), Some(5));
}

#[test]
fn an_accepted_list_survives_a_reopen() {
    let s = Scratch::new("reopen");
    let kept = stored(&s.file(), 5);
    assert_eq!((kept.epoch(), kept.sequence()), (Some(T0), Some(5)));
    let again = open(&s.file()).unwrap();
    assert_eq!((again.epoch(), again.sequence()), (Some(T0), Some(5)));
    assert_eq!(again.require_fresh(NOW, WINDOW), Ok(()));
}

#[test]
fn a_replay_after_a_restart_is_refused_and_writes_nothing() {
    let s = Scratch::new("replay");
    drop(stored(&s.file(), 5));
    let mut kept = open(&s.file()).unwrap();
    let before = snapshot(&s.file());
    assert_eq!(
        kept.accept(list(1, 4, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Err(KeptError::Refused(E::OlderSequence {
            epoch: T0,
            kept: 5,
            offered: 4
        }))
    );
    assert_eq!(snapshot(&s.file()), before, "the file is not rewritten");
    // the same list again changes nothing and writes nothing either
    assert_eq!(
        kept.accept(list(1, 5, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Ok(Accepted::Unchanged)
    );
    assert_eq!(snapshot(&s.file()), before);
}

#[test]
fn a_leaked_delegate_stays_superseded_after_a_restart() {
    let s = Scratch::new("leaked");
    let e = T0;
    let e2 = T0 + 100;
    let now = e2 + 20;
    let mut kept = open(&s.file()).unwrap();
    assert_eq!(
        kept.accept(
            list(1, u64::MAX - 1, e + 10, &[]),
            &delegate(1, e),
            now,
            WINDOW
        ),
        Ok(Accepted::Replaced)
    );
    assert_eq!(
        kept.accept(list(2, 1, e2 + 10, &[]), &delegate(2, e2), now, WINDOW),
        Ok(Accepted::Replaced)
    );
    drop(kept);
    let mut kept = open(&s.file()).unwrap();
    assert_eq!((kept.epoch(), kept.sequence()), (Some(e2), Some(1)));
    let before = snapshot(&s.file());
    assert_eq!(
        kept.accept(
            list(1, u64::MAX - 1, e + 30, &[]),
            &delegate(1, e),
            now,
            WINDOW
        ),
        Err(KeptError::Refused(E::OlderEpoch {
            kept: e2,
            offered: e
        }))
    );
    assert_eq!(snapshot(&s.file()), before);
}

#[test]
fn a_conflict_after_a_restart_is_refused_with_both_digests() {
    let s = Scratch::new("conflict");
    let kept_list = list(1, 5, T0 + 10, &[dev(1)]);
    let offered = list(1, 5, T0 + 10, &[dev(1), dev(2)]);
    let mut kept = open(&s.file()).unwrap();
    kept.accept(kept_list.clone(), &delegate(1, T0), NOW, WINDOW)
        .unwrap();
    drop(kept);
    let mut kept = open(&s.file()).unwrap();
    let before = snapshot(&s.file());
    let err = kept
        .accept(offered.clone(), &delegate(1, T0), NOW, WINDOW)
        .unwrap_err();
    let kept_digest: [u8; 32] = Sha256::digest(kept_list.to_bytes()).into();
    let offered_digest: [u8; 32] = Sha256::digest(offered.to_bytes()).into();
    assert_eq!(
        err,
        KeptError::Refused(E::Conflict {
            epoch: T0,
            sequence: 5,
            kept_digest,
            offered_digest
        })
    );
    let shown = err.to_string();
    let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert!(shown.contains(&hex(kept_digest)), "{shown}");
    assert!(shown.contains(&hex(offered_digest)), "{shown}");
    assert_eq!(snapshot(&s.file()), before);
}

/// The lock held by another holder: accept waits its bound and returns the typed error, writing nothing
#[test]
fn a_held_lock_times_out_with_the_typed_error() {
    let s = Scratch::new("lock");
    drop(stored(&s.file(), 5));
    let mut kept = open(&s.file())
        .unwrap()
        .with_lock_wait(Duration::from_millis(50));
    let mut lock_name = s.file().file_name().unwrap().to_os_string();
    lock_name.push(".lock");
    let holder = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(s.0.join(lock_name))
        .unwrap();
    holder.lock().unwrap();
    let before = snapshot(&s.file());
    assert_eq!(
        kept.accept(list(1, 6, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Err(KeptError::LockTimeout {
            waited: Duration::from_millis(50)
        })
    );
    assert_eq!(snapshot(&s.file()), before);
    holder.unlock().unwrap();
    assert_eq!(
        kept.accept(list(1, 6, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Ok(Accepted::Replaced)
    );
}

/// A floor source the test holds (test-only): what the store raised it to, and an optional failure
#[derive(Clone, Default)]
struct TestFloor {
    floor: Arc<Mutex<Option<(u64, u64)>>>,
    fail_raise: bool,
}
impl FloorSource for TestFloor {
    fn read_floor(&self) -> Result<Option<(u64, u64)>, FloorError> {
        Ok(*self.floor.lock().unwrap())
    }
    fn raise_floor(&self, epoch: u64, sequence: u64) -> Result<(), FloorError> {
        if self.fail_raise {
            return Err(FloorError::new("test: raise fails"));
        }
        *self.floor.lock().unwrap() = Some((epoch, sequence));
        Ok(())
    }
}

#[test]
fn a_floor_above_the_stored_pair_leaves_the_store_pending() {
    let s = Scratch::new("floor-open");
    drop(stored(&s.file(), 5));
    let floor = TestFloor::default();
    *floor.floor.lock().unwrap() = Some((T0, 6));
    let below =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor.clone()))
            .unwrap();
    assert_eq!(
        below.floor_pending(),
        Some(FloorPending {
            floor: (T0, 6),
            stored: Some((T0, 5))
        })
    );
    assert_eq!(below.sequence(), None, "the older list is not used");
    // a missing file is not an older file: it opens with the floor pending (see the floor-pending tests)
    let empty = Scratch::new("floor-missing");
    let pending =
        KeptRevocations::open_with_floor(empty.file(), master().public(), Box::new(floor.clone()))
            .unwrap();
    assert_eq!(
        pending.floor_pending(),
        Some(FloorPending {
            floor: (T0, 6),
            stored: None
        })
    );
    assert_eq!(pending.sequence(), None);
    // equal to the floor opens
    *floor.floor.lock().unwrap() = Some((T0, 5));
    let kept =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor)).unwrap();
    assert_eq!(kept.sequence(), Some(5));
}

#[test]
fn accept_raises_the_floor_and_checks_it_under_the_lock() {
    let s = Scratch::new("floor-accept");
    let floor = TestFloor::default();
    let mut kept =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor.clone()))
            .unwrap();
    kept.accept(list(1, 5, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW)
        .unwrap();
    assert_eq!(*floor.floor.lock().unwrap(), Some((T0, 5)));
    kept.accept(list(1, 7, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW)
        .unwrap();
    assert_eq!(*floor.floor.lock().unwrap(), Some((T0, 7)));

    // an older file put back while the store is open: the next accept re-reads it, ignores it (it is
    // below the floor) and refuses a list that does not reach the floor, writing nothing
    let older = Scratch::new("floor-older");
    drop(stored(&older.file(), 5));
    fs::copy(older.file(), s.file()).unwrap();
    let before = snapshot(&s.file());
    assert_eq!(
        kept.accept(list(1, 6, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Err(KeptError::RolledBack {
            stored: Some((T0, 6)),
            floor: (T0, 7)
        })
    );
    assert_eq!(snapshot(&s.file()), before);
    assert_eq!(*floor.floor.lock().unwrap(), Some((T0, 7)));
}

/// The file is written before the floor is raised: a floor that cannot be raised leaves the new file
#[test]
fn the_file_is_written_before_the_floor_is_raised() {
    let s = Scratch::new("floor-order");
    let floor = TestFloor {
        fail_raise: true,
        ..TestFloor::default()
    };
    let mut kept =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor.clone()))
            .unwrap();
    assert_eq!(
        kept.accept(list(1, 5, T0 + 10, &[]), &delegate(1, T0), NOW, WINDOW),
        Err(KeptError::FloorNotRaised(FloorError::new(
            "test: raise fails"
        )))
    );
    assert_eq!(kept.sequence(), Some(5));
    assert_eq!(open(&s.file()).unwrap().sequence(), Some(5));
    assert_eq!(*floor.floor.lock().unwrap(), None);
}

/// A stream with a canned response: what is read is `response`, what is written is kept
struct Canned {
    response: Cursor<Vec<u8>>,
    written: Vec<u8>,
}
impl Canned {
    fn new(status: &str, body: &[u8]) -> Self {
        let mut r = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        r.extend_from_slice(body);
        Self {
            response: Cursor::new(r),
            written: Vec::new(),
        }
    }
    fn served(list: RevocationList, delegate: DeviceCert) -> Self {
        Self::new(
            "200 OK",
            &DelegatedListResponse { list, delegate }.to_bytes().unwrap(),
        )
    }
    fn not_found() -> Self {
        Self::new("404 Not Found", b"")
    }
}
impl Read for Canned {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.response.read(buf)
    }
}
impl Write for Canned {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.written.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn the_revocation_body_is_the_delegate_certificate_then_the_list() {
    let d = delegate(1, T0);
    let l = list(1, 3, T0 + 10, &[dev(1)]);
    let body = DelegatedListResponse {
        list: l.clone(),
        delegate: d.clone(),
    }
    .to_bytes()
    .unwrap();
    // the layout, built here from the parts
    let cert = d.to_bytes();
    let lb = l.to_bytes();
    let mut expected = vec![1u8];
    expected.extend_from_slice(&(cert.len() as u16).to_be_bytes());
    expected.extend_from_slice(&cert);
    expected.extend_from_slice(&(lb.len() as u32).to_be_bytes());
    expected.extend_from_slice(&lb);
    assert_eq!(body, expected);

    let mut stream = Canned::new("200 OK", &body);
    let got = fetch_relay_revocations(&mut stream, "relay.test").unwrap();
    assert_eq!(got.list, l);
    assert_eq!(got.delegate, d);
    let head = String::from_utf8(stream.written).unwrap();
    assert!(
        head.starts_with("GET /relay-revocations HTTP/1.1\r\n"),
        "{head}"
    );

    // another version, a byte after the list, a length past the end
    let mut v2 = body.clone();
    v2[0] = 2;
    let mut trailing = body.clone();
    trailing.push(0);
    let short = &body[..body.len() - 1];
    for bad in [v2.as_slice(), trailing.as_slice(), short, &[], &[1u8]] {
        assert_eq!(
            DelegatedListResponse::from_bytes(bad).err(),
            Some(BodyError::Malformed)
        );
    }
    // a part that is not one
    let mut not_cert = vec![1u8, 0, 1, 0xff];
    not_cert.extend_from_slice(&(lb.len() as u32).to_be_bytes());
    not_cert.extend_from_slice(&lb);
    assert!(matches!(
        DelegatedListResponse::from_bytes(&not_cert),
        Err(BodyError::Cert(_))
    ));
    // a declared length over the cap, refused from the head
    let mut stream = Canned::new("200 OK", &[]);
    stream.response = Cursor::new(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
            MAX_RELAY_DELEGATED_REVOCATIONS_LEN + 1
        )
        .into_bytes(),
    );
    assert_eq!(
        fetch_relay_revocations(stream, "relay.test").err(),
        Some(FetchError::TooLarge)
    );
}

fn rule_open(s: &Scratch) -> KeptRevocations {
    open(&s.file()).unwrap()
}

#[test]
fn the_rule_refuses_a_relay_without_a_fresh_list() {
    let rule = RevocationRule::default();
    // 404 and nothing kept
    let s = Scratch::new("rule-404");
    let mut kept = rule_open(&s);
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::not_found(),
            "relay.test",
            &relay_cert(),
            NOW
        ),
        Err(RevocationRuleError::NotFresh {
            served: Some(Box::new(ServedNotUsed::Fetch(FetchError::Status(404)))),
            reason: E::NothingKept
        })
    );
    // a stale list served: refused by the store
    let stale = Canned::served(
        list(1, 1, T0 - 2 * WINDOW, &[]),
        delegate(1, T0 - 2 * WINDOW),
    );
    assert_eq!(
        rule.apply_fetch(&mut kept, stale, "relay.test", &relay_cert(), NOW),
        Err(RevocationRuleError::Kept(KeptError::Refused(E::Stale {
            issued_at: T0 - 2 * WINDOW,
            now: NOW
        })))
    );
    // a fresh list kept, then the fetch fails after it went stale
    let s2 = Scratch::new("rule-stale");
    let mut kept = rule_open(&s2);
    let fresh = Canned::served(list(1, 1, T0 + 10, &[]), delegate(1, T0));
    assert_eq!(
        rule.apply_fetch(&mut kept, fresh, "relay.test", &relay_cert(), NOW),
        Ok(None)
    );
    let later = T0 + 10 + WINDOW + 1;
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::not_found(),
            "relay.test",
            &relay_cert(),
            later
        ),
        Err(RevocationRuleError::NotFresh {
            served: Some(Box::new(ServedNotUsed::Fetch(FetchError::Status(404)))),
            reason: E::Stale {
                issued_at: T0 + 10,
                now: later
            }
        })
    );
    // while the kept list is fresh a missed fetch is what the window is for
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::not_found(),
            "relay.test",
            &relay_cert(),
            NOW + 60
        ),
        Ok(Some(ServedNotUsed::Fetch(FetchError::Status(404))))
    );
}

#[test]
fn the_rule_refuses_a_relay_the_list_names() {
    let rule = RevocationRule::default();
    let s = Scratch::new("rule-named");
    let mut kept = rule_open(&s);
    let naming = Canned::served(
        list(1, 1, T0 + 10, &[relay_cert().device_id()]),
        delegate(1, T0),
    );
    assert_eq!(
        rule.apply_fetch(&mut kept, naming, "relay.test", &relay_cert(), NOW),
        Err(RevocationRuleError::Revoked)
    );
    // and after a restart, from the store, even with the fetch failing
    let mut kept = rule_open(&s);
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::not_found(),
            "relay.test",
            &relay_cert(),
            NOW
        ),
        Err(RevocationRuleError::Revoked)
    );
    // a fresh list that does not name it lets it through
    let s2 = Scratch::new("rule-not-named");
    let mut kept = rule_open(&s2);
    let other = Canned::served(list(1, 1, T0 + 10, &[dev(9)]), delegate(1, T0));
    assert_eq!(
        rule.apply_fetch(&mut kept, other, "relay.test", &relay_cert(), NOW),
        Ok(None)
    );
}

#[test]
fn the_rule_lets_a_relay_through_without_a_list_only_when_opted_out() {
    let optional = RevocationRule {
        required: false,
        ..RevocationRule::default()
    };
    let s = Scratch::new("rule-optional");
    let mut kept = rule_open(&s);
    assert_eq!(
        optional.apply_fetch(
            &mut kept,
            Canned::not_found(),
            "relay.test",
            &relay_cert(),
            NOW
        ),
        Ok(Some(ServedNotUsed::Fetch(FetchError::Status(404))))
    );
    // a list refused for anything but being older still refuses the relay
    let stale = Canned::served(
        list(1, 1, T0 - 2 * WINDOW, &[]),
        delegate(1, T0 - 2 * WINDOW),
    );
    assert!(matches!(
        optional.apply_fetch(&mut kept, stale, "relay.test", &relay_cert(), NOW),
        Err(RevocationRuleError::Kept(KeptError::Refused(
            E::Stale { .. }
        )))
    ));
    // and the store is still the one a channel policy reads
    assert!(DelegatedRevocations::new().epoch().is_none());
    assert!(kept.epoch().is_none());
}

/// Two honest relays (or a cache) may lag behind each other: a served list merely older than the kept one
/// is not used, and the relay is decided on the kept list and the freshness rule; a conflicting list still
/// refuses the relay
#[test]
fn the_rule_carries_on_past_an_older_list_but_not_past_a_conflict() {
    let rule = RevocationRule::default();
    let s = Scratch::new("rule-older");
    let mut kept = rule_open(&s);
    let six = list(1, 6, T0 + 10, &[dev(1)]);
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::served(six, delegate(1, T0)),
            "relay.test",
            &relay_cert(),
            NOW
        ),
        Ok(None)
    );
    // an older sequence, the kept list fresh: through, with the reason to log
    let five = || Canned::served(list(1, 5, T0 + 10, &[dev(1)]), delegate(1, T0));
    assert_eq!(
        rule.apply_fetch(&mut kept, five(), "relay.test", &relay_cert(), NOW),
        Ok(Some(ServedNotUsed::Older(E::OlderSequence {
            epoch: T0,
            kept: 6,
            offered: 5
        })))
    );
    assert_eq!(kept.sequence(), Some(6));
    // an older sequence, the kept list stale, required: refused as not fresh, the reason carried
    let later = T0 + 10 + WINDOW + 1;
    let fresh_five = Canned::served(list(1, 5, later - 10, &[dev(1)]), delegate(1, T0));
    assert_eq!(
        rule.apply_fetch(&mut kept, fresh_five, "relay.test", &relay_cert(), later),
        Err(RevocationRuleError::NotFresh {
            served: Some(Box::new(ServedNotUsed::Older(E::OlderSequence {
                epoch: T0,
                kept: 6,
                offered: 5
            }))),
            reason: E::Stale {
                issued_at: T0 + 10,
                now: later
            }
        })
    );
    // an older epoch, the kept list fresh: through
    let s2 = Scratch::new("rule-older-epoch");
    let mut kept = rule_open(&s2);
    let e2 = T0 + 100;
    let now = e2 + 20;
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::served(list(2, 1, e2 + 10, &[]), delegate(2, e2)),
            "relay.test",
            &relay_cert(),
            now
        ),
        Ok(None)
    );
    assert_eq!(
        rule.apply_fetch(
            &mut kept,
            Canned::served(list(1, 9, e2 + 10, &[]), delegate(1, T0)),
            "relay.test",
            &relay_cert(),
            now
        ),
        Ok(Some(ServedNotUsed::Older(E::OlderEpoch {
            kept: e2,
            offered: T0
        })))
    );
    // the same sequence with other content: a conflict, which refuses the relay
    let mut kept = rule_open(&s);
    let other_six = list(1, 6, T0 + 10, &[dev(1), dev(2)]);
    assert!(matches!(
        rule.apply_fetch(
            &mut kept,
            Canned::served(other_six, delegate(1, T0)),
            "relay.test",
            &relay_cert(),
            NOW
        ),
        Err(RevocationRuleError::Kept(KeptError::Refused(E::Conflict {
            epoch: T0,
            sequence: 6,
            ..
        })))
    ));
}

/// A store whose file was deleted under a floor (T0, 6): open with the floor pending
fn pending_store(name: &str) -> (Scratch, TestFloor, KeptRevocations) {
    let s = Scratch::new(name);
    let floor = TestFloor::default();
    *floor.floor.lock().unwrap() = Some((T0, 6));
    let kept =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor.clone()))
            .unwrap();
    assert_eq!(
        kept.floor_pending(),
        Some(FloorPending {
            floor: (T0, 6),
            stored: None
        })
    );
    (s, floor, kept)
}

/// Negative: with the list file deleted, no relay is admitted while the floor is pending, whether or not a
/// fresh list is required, and a failed fetch leaves it pending
#[test]
fn a_pending_floor_admits_no_relay_required_or_not() {
    let (_s, _floor, mut kept) = pending_store("pending-none");
    let optional = RevocationRule {
        required: false,
        ..RevocationRule::default()
    };
    let mut compared = 0;
    for rule in [RevocationRule::default(), optional] {
        assert_eq!(
            rule.apply_fetch(
                &mut kept,
                Canned::not_found(),
                "relay.test",
                &relay_cert(),
                NOW
            ),
            Err(RevocationRuleError::FloorPending {
                pending: FloorPending {
                    floor: (T0, 6),
                    stored: None
                },
                served: Some(Box::new(ServedNotUsed::Fetch(FetchError::Status(404)))),
            }),
            "required = {}",
            rule.required
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
    assert_eq!(
        kept.floor_pending(),
        Some(FloorPending {
            floor: (T0, 6),
            stored: None
        })
    );
    let shown = RevocationRuleError::FloorPending {
        pending: FloorPending {
            floor: (T0, 6),
            stored: None,
        },
        served: None,
    }
    .to_string();
    assert!(shown.contains("holds no list"), "{shown}");
}

/// Negative: a revoked relay serving the list from before its revocation (below the floor) is refused and
/// nothing is written, required or not
#[test]
fn a_pending_floor_refuses_a_list_below_it_and_writes_nothing() {
    let (s, _floor, mut kept) = pending_store("pending-below");
    let optional = RevocationRule {
        required: false,
        ..RevocationRule::default()
    };
    for rule in [RevocationRule::default(), optional] {
        let old = Canned::served(list(1, 5, T0 + 10, &[]), delegate(1, T0));
        assert_eq!(
            rule.apply_fetch(&mut kept, old, "relay.test", &relay_cert(), NOW),
            Err(RevocationRuleError::Kept(KeptError::RolledBack {
                stored: Some((T0, 5)),
                floor: (T0, 6)
            }))
        );
    }
    assert!(!s.file().exists(), "nothing written");
    assert_eq!(
        kept.floor_pending(),
        Some(FloorPending {
            floor: (T0, 6),
            stored: None
        })
    );
}

/// The honest path: a fresh list at or above the floor from a relay clears the pending floor; a relay it
/// names is then refused and one it does not name is admitted
#[test]
fn a_list_at_the_floor_clears_it() {
    let (s, floor, mut kept) = pending_store("pending-reach");
    let rule = RevocationRule::default();
    let at_floor = Canned::served(list(1, 6, T0 + 10, &[]), delegate(1, T0));
    assert_eq!(
        rule.apply_fetch(&mut kept, at_floor, "relay.test", &relay_cert(), NOW),
        Ok(None)
    );
    assert_eq!(kept.floor_pending(), None);
    assert!(s.file().exists());
    assert_eq!(*floor.floor.lock().unwrap(), Some((T0, 6)));
    let names = Canned::served(
        list(1, 7, T0 + 11, &[relay_cert().device_id()]),
        delegate(1, T0),
    );
    assert_eq!(
        rule.apply_fetch(&mut kept, names, "relay.test", &relay_cert(), NOW),
        Err(RevocationRuleError::Revoked)
    );
    // reopened: the file is at the floor, nothing pending
    let again =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor)).unwrap();
    assert_eq!(again.floor_pending(), None);
    assert_eq!(again.sequence(), Some(7));
}

/// An older valid file put back (non-empty, below the floor) is still refused at open, not taken as pending
#[test]
fn an_older_file_put_back_is_pending_shown_and_recovers_only_through_the_floor() {
    let s = Scratch::new("pending-older");
    drop(stored(&s.file(), 5));
    let floor = TestFloor::default();
    *floor.floor.lock().unwrap() = Some((T0, 6));
    let mut kept =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor.clone()))
            .unwrap();
    let older = FloorPending {
        floor: (T0, 6),
        stored: Some((T0, 5)),
    };
    assert_eq!(kept.floor_pending(), Some(older));
    // the older list is not used for anything: no revocation, no freshness
    assert!(!kept.revokes(&relay_cert()));
    assert!(kept.require_fresh(NOW, WINDOW).is_err());
    // no relay is admitted, required or not, and the state says an older list was found
    let optional = RevocationRule {
        required: false,
        ..RevocationRule::default()
    };
    for rule in [RevocationRule::default(), optional] {
        let got = rule.apply_fetch(
            &mut kept,
            Canned::not_found(),
            "relay.test",
            &relay_cert(),
            NOW,
        );
        assert_eq!(
            got,
            Err(RevocationRuleError::FloorPending {
                pending: older,
                served: Some(Box::new(ServedNotUsed::Fetch(FetchError::Status(404)))),
            })
        );
        let shown = got.unwrap_err().to_string();
        assert!(shown.contains("found an older list"), "{shown}");
    }
    // a list below the floor leaves the file and the state untouched
    let before = snapshot(&s.file());
    let below = Canned::served(list(1, 5, T0 + 10, &[]), delegate(1, T0));
    assert!(matches!(
        RevocationRule::default().apply_fetch(&mut kept, below, "relay.test", &relay_cert(), NOW),
        Err(RevocationRuleError::Kept(KeptError::RolledBack { .. }))
    ));
    assert_eq!(snapshot(&s.file()), before, "file untouched");
    assert_eq!(kept.floor_pending(), Some(older), "state untouched");
    // a list at the floor replaces the older file and clears the pending floor
    let at = Canned::served(list(1, 6, T0 + 11, &[]), delegate(1, T0));
    assert_eq!(
        RevocationRule::default().apply_fetch(&mut kept, at, "relay.test", &relay_cert(), NOW),
        Ok(None)
    );
    assert_eq!(kept.floor_pending(), None);
    let again =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor)).unwrap();
    assert_eq!(again.sequence(), Some(6));
    assert_eq!(again.floor_pending(), None);
}

/// The older list below the floor is not used on the accept path either: a list at the floor is checked
/// against nothing, not against it (an id only the older list named does not make the list "drop a
/// revocation")
#[test]
fn the_older_list_is_not_used_when_a_list_at_the_floor_is_accepted() {
    let s = Scratch::new("pending-older-accept");
    {
        let mut k = open(&s.file()).unwrap();
        k.accept(
            list(1, 5, T0 + 10, &[dev(9)]),
            &delegate(1, T0),
            NOW,
            WINDOW,
        )
        .unwrap();
    }
    let floor = TestFloor::default();
    *floor.floor.lock().unwrap() = Some((T0, 6));
    let mut kept =
        KeptRevocations::open_with_floor(s.file(), master().public(), Box::new(floor)).unwrap();
    assert_eq!(
        kept.floor_pending(),
        Some(FloorPending {
            floor: (T0, 6),
            stored: Some((T0, 5))
        })
    );
    kept.accept(list(1, 6, T0 + 11, &[]), &delegate(1, T0), NOW, WINDOW)
        .unwrap();
    assert_eq!(kept.floor_pending(), None);
    assert_eq!(kept.sequence(), Some(6));
}
