//! Oracle: a two-device inbox (the set of messages and their read marks) converges across a
//! concurrent "mark read" and "delete", and stays converged across a restart through ALICE-DB.
//!
//! The digest compared is the SHA-256 of the persisted bytes (`persist::encode_state`) of both
//! replicated values, so equal digests mean the two devices would write the same bytes: a
//! replica-local field (the device's own replica id, its clock) or a leftover empty entry would
//! show up as a different digest even when the values read the same.
#![cfg(feature = "crdt-db")]

use alice_db::blob::BlobStorage;
use alice_sync::crdt::{CrdtMergeable, LwwMap, OrSet};
use alice_sync::crdt_store::{load_state, store_state};
use alice_sync::persist::{decode_state, encode_state, IdStatus};
use sha2::{Digest, Sha256};

const A: u64 = 1;
const B: u64 = 2;

struct Inbox {
    messages: OrSet<u64>,
    read: LwwMap<u64, bool>,
}

impl Inbox {
    fn new(replica: u64) -> Self {
        Self {
            messages: OrSet::new(),
            read: LwwMap::new(replica),
        }
    }

    /// What this device sends: the persisted bytes of both values
    fn wire(&self) -> (Vec<u8>, Vec<u8>) {
        (encode_state(&self.messages), encode_state(&self.read))
    }

    /// Merge what the other device sent (decoded from its bytes, as a receiver would)
    fn receive(&mut self, (m, r): &(Vec<u8>, Vec<u8>)) {
        let m = decode_state::<OrSet<u64>>(m).unwrap();
        let r = decode_state::<LwwMap<u64, bool>>(r).unwrap();
        assert_eq!((m.status, r.status), (IdStatus::Current, IdStatus::Current));
        self.messages.merge(&m.value);
        self.read.merge(&r.value);
    }

    fn digest(&self) -> [u8; 32] {
        let (m, r) = self.wire();
        let mut h = Sha256::new();
        h.update(&m);
        h.update(&r);
        h.finalize().into()
    }
}

/// Both devices hold messages 1 and 2; then A marks 1 read while B deletes 1, and they exchange
fn diverged_then_merged() -> (Inbox, Inbox) {
    let mut a = Inbox::new(A);
    let mut b = Inbox::new(B);
    a.messages.add(A, 1);
    a.messages.add(A, 2);
    a.read.insert(1, false);
    a.read.insert(2, false);
    b.receive(&a.wire());
    assert_eq!(
        a.digest(),
        b.digest(),
        "same state before the concurrent edits"
    );

    // concurrent: A marks 1 read, B deletes 1 (the message and its read mark)
    a.read.insert(1, true);
    b.messages.remove(&1);
    b.read.remove(&1);
    let (from_a, from_b) = (a.wire(), b.wire());
    a.receive(&from_b);
    b.receive(&from_a);
    (a, b)
}

#[test]
fn a_read_mark_and_a_delete_converge_to_the_same_bytes() {
    let (a, b) = diverged_then_merged();
    assert_eq!(a.digest(), b.digest());
    for d in [&a, &b] {
        assert!(!d.messages.contains(&1), "the delete removed message 1");
        assert!(d.messages.contains(&2));
        assert_eq!(d.read.get(&2), Some(&false));
    }
    // the read mark of 1 is decided the same way on both (last writer by Lamport time, then replica)
    assert_eq!(a.read.get(&1), b.read.get(&1));
    // merging again changes nothing
    let mut a2 = Inbox::new(A);
    a2.receive(&a.wire());
    a2.receive(&b.wire());
    assert_eq!(a2.digest(), a.digest());
}

#[test]
fn the_converged_state_survives_a_restart_through_alice_db() {
    let (a, b) = diverged_then_merged();
    let want = a.digest();
    let dir = tempfile::tempdir().unwrap();
    for (name, d) in [("a", &a), ("b", &b)] {
        let db = BlobStorage::open(dir.path().join(name).join("wal")).unwrap();
        store_state(&db, b"inbox/messages", &d.messages).unwrap();
        store_state(&db, b"inbox/read", &d.read).unwrap();
    }
    // a restart: the stores are opened again and the values read back
    let mut back = Vec::new();
    for (name, replica) in [("a", A), ("b", B)] {
        let db = BlobStorage::open(dir.path().join(name).join("wal")).unwrap();
        let messages = load_state::<OrSet<u64>>(&db, b"inbox/messages")
            .unwrap()
            .expect("stored")
            .value;
        let read = load_state::<LwwMap<u64, bool>>(&db, b"inbox/read")
            .unwrap()
            .expect("stored")
            .value
            .with_replica(replica);
        // a map read back has no writer until the device names itself (ties break by replica id)
        assert_eq!(read.replica_id(), replica);
        back.push(Inbox { messages, read });
    }
    assert_eq!(back[0].digest(), want);
    assert_eq!(back[1].digest(), want);
    assert!(load_state::<OrSet<u64>>(
        &BlobStorage::open(dir.path().join("a").join("wal")).unwrap(),
        b"inbox/none"
    )
    .unwrap()
    .is_none());

    // after the restart A's clock continues past every write it has seen: its next write wins on B
    let (mut a, mut b) = (back.remove(0), back.remove(0));
    a.read.insert(2, true);
    b.receive(&a.wire());
    assert_eq!(b.read.get(&2), Some(&true));
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn the_crdt_types_build_without_the_async_feature() {
    // the module is plain data: this file builds under `crdt-db` alone (no `async`)
    let mut s: OrSet<u64> = OrSet::new();
    s.add(A, 7);
    assert!(s.contains(&7));
}
