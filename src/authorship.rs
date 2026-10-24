/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    License: LicenseRef-Proprietary (see LICENSE-Proprietary)
*/

//! Signed events: an event counts only under its author's signed checkpoint.
//!
//! The author of an [`Origin`] is the device whose certificate (ALICE-Identity) the origin is derived
//! from ([`origin_of`]): the first 16 bytes of
//!
//! ```text
//! SHA-256("alice-sync/origin/v1" || subject length (u16, big-endian) || subject || device key)
//! ```
//!
//! Each origin's events, in sequence-number order from 1, form a hash chain:
//!
//! ```text
//! head_0 = SHA-256("alice-sync/chain/v1" || origin)
//! head_n = SHA-256(head_(n-1) || canonical bytes of event n)
//! ```
//!
//! (the canonical bytes are fixed big-endian fields, [`canonical_bytes`]). The author signs a
//! [`Checkpoint`] (origin, sequence number, head) as an artifact signature for [`CHECKPOINT_CONTEXT`]
//! over SHA-256(origin || sequence number (u64, big-endian) || head), at least every
//! [`CHECKPOINT_INTERVAL`] events and at the last event of every batch it sends.
//!
//! A receiver ([`Verifier`]) accepts a checkpoint only when the signer's certificate verifies under the
//! master key its [`TrustPolicy`] gives for the subject (at the policy's time, against its revocation
//! lists) and the checkpoint's origin is the one derived from that certificate. It holds an author's
//! events until a verified checkpoint at or past them matches the chain it computes from them, then
//! applies them; a batch whose events no checkpoint covers is refused whole, an event whose chain does
//! not match the author's checkpoint is dropped, and a second checkpoint with another head at the same
//! sequence number (an author signing two histories) is refused. An author's held events are bounded
//! ([`MAX_PENDING_PER_AUTHOR`]): the smallest sequence numbers are kept, so which events are held does
//! not depend on the arrival order, and an event past the bound is refused with
//! [`AuthorshipError::PendingFull`]. Other authors are not affected.
//!
//! A relay may split and merge batches freely: what a receiver applies is decided by the events and the
//! checkpoints, not by how they were grouped.

use std::collections::BTreeMap;

use alice_identity::artifact::{
    sha256_payload, sign_artifact, verify_artifact, ArtifactError, ArtifactSignature,
    PayloadDigest, Revocation,
};
use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::{KeyPair, PublicKey};
use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};

pub use crate::event::Origin;
use crate::{Event, EventKind};

/// Prefixed to the certificate fields an origin is derived from
pub const ORIGIN_CONTEXT: &[u8] = b"alice-sync/origin/v1";

/// Prefixed to the origin to start its hash chain
pub const CHAIN_CONTEXT: &[u8] = b"alice-sync/chain/v1";

/// The artifact-signature context of a checkpoint
pub const CHECKPOINT_CONTEXT: &[u8] = b"alice-sync/checkpoint/v1";

/// An author signs a checkpoint at least every this many events
pub const CHECKPOINT_INTERVAL: u64 = 64;

/// The most events of one author a receiver holds while they wait for a checkpoint
pub const MAX_PENDING_PER_AUTHOR: usize = 1024;

/// What a receiver trusts: the master key of each subject, the revocation lists, the time
pub trait TrustPolicy {
    /// The master key of `subject`, or `None` when the subject is not one this side accepts
    fn master_of(&self, subject: &Did) -> Option<PublicKey>;
    /// The revocation lists to check an author's certificate against
    fn revocation(&self) -> Revocation<'_>;
    /// The time at which an author's certificate must be valid (seconds, as the certificate counts them)
    fn now(&self) -> u64;
}

/// Why signed events were not accepted, or an author could not be made
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorshipError {
    /// An event that no checkpoint at or past it covers (or an unstamped event)
    Uncovered { origin: Origin, seq: u64 },
    /// A checkpoint that does not decode, whose signer the policy does not accept, or whose origin is not
    /// the one derived from the signer's certificate
    BadCheckpoint { origin: Origin, reason: String },
    /// The events up to a checkpoint do not give the head the author signed
    ChainMismatch { origin: Origin, seq: u64 },
    /// A checkpoint with another head than one already accepted at the same sequence number
    Forked { origin: Origin, seq: u64 },
    /// The author already has [`MAX_PENDING_PER_AUTHOR`] events held, all before this one
    PendingFull { origin: Origin, seq: u64 },
    /// This side's own key and certificate cannot sign
    Credential(ArtifactError),
    /// The node has no author, so its events carry no signature
    NoAuthor,
    /// An accepted event did not apply to the world
    Apply(String),
}

impl std::fmt::Display for AuthorshipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uncovered { seq, .. } => write!(f, "event {seq} is not covered by a checkpoint"),
            Self::BadCheckpoint { reason, .. } => write!(f, "checkpoint refused: {reason}"),
            Self::ChainMismatch { seq, .. } => {
                write!(
                    f,
                    "the events up to {seq} do not match the author's checkpoint"
                )
            }
            Self::Forked { seq, .. } => {
                write!(f, "a second checkpoint with another head at {seq}")
            }
            Self::PendingFull { seq, .. } => {
                write!(f, "event {seq} refused: the author's held events are full")
            }
            Self::Credential(e) => write!(f, "this side's credential cannot sign: {e}"),
            Self::NoAuthor => f.write_str("the node has no author"),
            Self::Apply(e) => write!(f, "an accepted event did not apply: {e}"),
        }
    }
}

impl std::error::Error for AuthorshipError {}

/// The origin derived from a device certificate (see the module documentation)
#[must_use]
pub fn origin_of(cert: &DeviceCert) -> Origin {
    let subject = cert.subject().to_string();
    // a certificate's subject is at most 255 bytes (checked when it is issued or decoded)
    let len = u16::try_from(subject.len()).unwrap_or(u16::MAX);
    let mut m = Vec::with_capacity(ORIGIN_CONTEXT.len() + 2 + subject.len() + 32);
    m.extend_from_slice(ORIGIN_CONTEXT);
    m.extend_from_slice(&len.to_be_bytes());
    m.extend_from_slice(subject.as_bytes());
    m.extend_from_slice(cert.device_public());
    let d = sha256_payload(&m);
    let mut o = [0u8; 16];
    o.copy_from_slice(&d.bytes()[..16]);
    Origin(o)
}

impl Origin {
    /// The origin derived from a device certificate ([`origin_of`])
    #[must_use]
    pub fn of(cert: &DeviceCert) -> Self {
        origin_of(cert)
    }
}

/// The bytes of an event a hash chain covers: sequence number, timestamp, origin, then the kind's tag and
/// fields, each integer big-endian (the event id is local to a stream and not covered)
#[must_use]
pub fn canonical_bytes(event: &Event) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + 8 + 16 + 1 + 18);
    b.extend_from_slice(&event.seq.0.to_be_bytes());
    b.extend_from_slice(&event.timestamp.to_be_bytes());
    b.extend_from_slice(&event.origin.0);
    let i16s = |b: &mut Vec<u8>, v: &[i16; 3]| {
        for x in v {
            b.extend_from_slice(&x.to_be_bytes());
        }
    };
    match &event.kind {
        EventKind::Motion { entity, delta } => {
            b.extend_from_slice(&[0]);
            b.extend_from_slice(&entity.to_be_bytes());
            i16s(&mut b, delta);
        }
        EventKind::Spawn { entity, kind, pos } => {
            b.extend_from_slice(&[1]);
            b.extend_from_slice(&entity.to_be_bytes());
            b.extend_from_slice(&kind.to_be_bytes());
            i16s(&mut b, pos);
        }
        EventKind::Despawn { entity } => {
            b.extend_from_slice(&[2]);
            b.extend_from_slice(&entity.to_be_bytes());
        }
        EventKind::Property {
            entity,
            prop,
            value,
        } => {
            b.extend_from_slice(&[3]);
            b.extend_from_slice(&entity.to_be_bytes());
            b.extend_from_slice(&prop.to_be_bytes());
            b.extend_from_slice(&value.to_be_bytes());
        }
        EventKind::Input { player, code } => {
            b.extend_from_slice(&[4]);
            b.extend_from_slice(&player.to_be_bytes());
            b.extend_from_slice(&code.to_be_bytes());
        }
        EventKind::Tick { frame } => {
            b.extend_from_slice(&[5]);
            b.extend_from_slice(&frame.to_be_bytes());
        }
        EventKind::Custom { type_id, payload } => {
            b.extend_from_slice(&[6]);
            b.extend_from_slice(&type_id.to_be_bytes());
            b.extend_from_slice(payload);
        }
    }
    b
}

/// The head of an origin's chain before its first event
#[must_use]
pub fn chain_start(origin: Origin) -> [u8; 32] {
    let mut m = CHAIN_CONTEXT.to_vec();
    m.extend_from_slice(&origin.0);
    *sha256_payload(&m).bytes()
}

/// The head after `event`
#[must_use]
pub fn chain_next(head: &[u8; 32], event: &Event) -> [u8; 32] {
    let mut m = head.to_vec();
    m.extend_from_slice(&canonical_bytes(event));
    *sha256_payload(&m).bytes()
}

fn checkpoint_digest(origin: Origin, seq: u64, head: &[u8; 32]) -> PayloadDigest {
    let mut m = Vec::with_capacity(16 + 8 + 32);
    m.extend_from_slice(&origin.0);
    m.extend_from_slice(&seq.to_be_bytes());
    m.extend_from_slice(head);
    sha256_payload(&m)
}

/// An author's signed statement that its chain had `head` after event `seq`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct Checkpoint {
    pub origin: Origin,
    pub seq: u64,
    pub head: [u8; 32],
    /// The encoded artifact signature (ALICE-Identity) for [`CHECKPOINT_CONTEXT`]
    pub signature: Vec<u8>,
}

/// Events with the checkpoints that cover them
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct SignedEvents {
    pub events: Vec<Event>,
    pub checkpoints: Vec<Checkpoint>,
}

/// The signing side of one origin: the device key, its certificate, and the chain so far
#[derive(Debug, Clone)]
pub struct Author {
    key: KeyPair,
    cert: DeviceCert,
    origin: Origin,
    seq: u64,
    head: [u8; 32],
    checkpoints: BTreeMap<u64, Checkpoint>,
}

impl Author {
    /// An author from the device key and the certificate that names it
    ///
    /// # Errors
    /// [`AuthorshipError::Credential`] when `key` is not the key `cert` names
    pub fn new(key: KeyPair, cert: DeviceCert) -> Result<Self, AuthorshipError> {
        // a trial signature checks the key against the certificate once
        sign_artifact(
            &key,
            &cert,
            CHECKPOINT_CONTEXT,
            &PayloadDigest::sha256([0; 32]),
        )
        .map_err(AuthorshipError::Credential)?;
        let origin = origin_of(&cert);
        Ok(Self {
            key,
            cert,
            origin,
            seq: 0,
            head: chain_start(origin),
            checkpoints: BTreeMap::new(),
        })
    }

    /// The origin this author's events carry
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The certificate this author signs with
    #[must_use]
    pub const fn cert(&self) -> &DeviceCert {
        &self.cert
    }

    /// Take the next event of this origin into the chain, signing a checkpoint every
    /// [`CHECKPOINT_INTERVAL`] events
    ///
    /// # Errors
    /// [`AuthorshipError::Uncovered`] when the event is not this origin's next one
    pub(crate) fn extend(&mut self, event: &Event) -> Result<(), AuthorshipError> {
        if event.origin != self.origin || event.seq.0 != self.seq + 1 {
            return Err(AuthorshipError::Uncovered {
                origin: event.origin,
                seq: event.seq.0,
            });
        }
        self.head = chain_next(&self.head, event);
        self.seq += 1;
        if self.seq.is_multiple_of(CHECKPOINT_INTERVAL) {
            self.sign_now()?;
        }
        Ok(())
    }

    /// Sign a checkpoint at the last event taken, if there is one and it is not signed yet
    pub(crate) fn sign_now(&mut self) -> Result<(), AuthorshipError> {
        if self.seq == 0 || self.checkpoints.contains_key(&self.seq) {
            return Ok(());
        }
        let sig = sign_artifact(
            &self.key,
            &self.cert,
            CHECKPOINT_CONTEXT,
            &checkpoint_digest(self.origin, self.seq, &self.head),
        )
        .map_err(AuthorshipError::Credential)?;
        self.checkpoints.insert(
            self.seq,
            Checkpoint {
                origin: self.origin,
                seq: self.seq,
                head: self.head,
                signature: sig.to_bytes(),
            },
        );
        Ok(())
    }

    /// The checkpoints signed so far, by sequence number
    #[must_use]
    pub const fn checkpoints(&self) -> &BTreeMap<u64, Checkpoint> {
        &self.checkpoints
    }
}

/// Verify a checkpoint's signature and that its origin is its signer's
fn verify_checkpoint(cp: &Checkpoint, policy: &impl TrustPolicy) -> Result<(), AuthorshipError> {
    let bad = |reason: String| AuthorshipError::BadCheckpoint {
        origin: cp.origin,
        reason,
    };
    let sig = ArtifactSignature::from_bytes(&cp.signature).map_err(|e| bad(e.to_string()))?;
    let master = policy
        .master_of(sig.cert().subject())
        .ok_or_else(|| bad("the signer's master key is unknown".to_owned()))?;
    verify_artifact(
        &sig,
        &master,
        CHECKPOINT_CONTEXT,
        &checkpoint_digest(cp.origin, cp.seq, &cp.head),
        policy.now(),
        policy.revocation(),
    )
    .map_err(|e| bad(e.to_string()))?;
    if origin_of(sig.cert()) != cp.origin {
        return Err(bad("the origin is not the signer's".to_owned()));
    }
    Ok(())
}

/// What a receiver holds for one origin
#[derive(Debug, Clone)]
struct AuthorState {
    /// the last sequence number applied, and the head after it
    applied: u64,
    head: [u8; 32],
    /// verified checkpoints, by sequence number
    checkpoints: BTreeMap<u64, Checkpoint>,
    /// events received past `applied`, waiting for a checkpoint (at most MAX_PENDING_PER_AUTHOR)
    pending: BTreeMap<u64, Event>,
    /// the events applied, by sequence number (for relaying)
    applied_events: BTreeMap<u64, Event>,
}

impl AuthorState {
    fn new(origin: Origin) -> Self {
        Self {
            applied: 0,
            head: chain_start(origin),
            checkpoints: BTreeMap::new(),
            pending: BTreeMap::new(),
            applied_events: BTreeMap::new(),
        }
    }
}

/// The receiving side: per origin, the verified checkpoints, the held events and the applied ones
#[derive(Debug, Clone, Default)]
pub struct Verifier {
    authors: BTreeMap<Origin, AuthorState>,
}

impl Verifier {
    /// A verifier that has accepted nothing
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a batch: verify its checkpoints, hold its events, and apply (through `apply`, in sequence
    /// order per origin) every held event a verified checkpoint now covers. Returns how many events were
    /// applied.
    ///
    /// A batch with a checkpoint that is refused, or with an event no checkpoint covers, changes nothing.
    ///
    /// # Errors
    /// [`AuthorshipError::BadCheckpoint`], [`AuthorshipError::Forked`], [`AuthorshipError::Uncovered`]
    /// (nothing taken); [`AuthorshipError::ChainMismatch`] (the mismatching events are dropped, the
    /// rest taken); [`AuthorshipError::PendingFull`] (the events past the bound are not held, the rest
    /// taken); [`AuthorshipError::Apply`]
    pub fn receive(
        &mut self,
        batch: &SignedEvents,
        policy: &impl TrustPolicy,
        mut apply: impl FnMut(&Event) -> Result<(), String>,
    ) -> Result<usize, AuthorshipError> {
        // 1. checkpoints: verified, and not forked against what is held or against each other
        let mut fresh: BTreeMap<(Origin, u64), &Checkpoint> = BTreeMap::new();
        for cp in &batch.checkpoints {
            if let Some(state) = self.authors.get(&cp.origin) {
                if let Some(held) = state.checkpoints.get(&cp.seq) {
                    if held.head != cp.head {
                        return Err(AuthorshipError::Forked {
                            origin: cp.origin,
                            seq: cp.seq,
                        });
                    }
                    continue;
                }
            }
            if let Some(other) = fresh.get(&(cp.origin, cp.seq)) {
                if other.head != cp.head {
                    return Err(AuthorshipError::Forked {
                        origin: cp.origin,
                        seq: cp.seq,
                    });
                }
                continue;
            }
            verify_checkpoint(cp, policy)?;
            fresh.insert((cp.origin, cp.seq), cp);
        }
        // 2. every event is stamped and covered by a checkpoint at or past it
        for e in &batch.events {
            let covered = e.is_stamped()
                && (fresh
                    .range((e.origin, e.seq.0)..=(e.origin, u64::MAX))
                    .next()
                    .is_some()
                    || self
                        .authors
                        .get(&e.origin)
                        .is_some_and(|s| s.checkpoints.range(e.seq.0..).next().is_some()));
            if !covered {
                return Err(AuthorshipError::Uncovered {
                    origin: e.origin,
                    seq: e.seq.0,
                });
            }
        }
        // 3. take the checkpoints and hold the events
        for ((origin, seq), cp) in fresh {
            self.authors
                .entry(origin)
                .or_insert_with(|| AuthorState::new(origin))
                .checkpoints
                .insert(seq, cp.clone());
        }
        let mut full = None;
        for e in &batch.events {
            let state = self
                .authors
                .entry(e.origin)
                .or_insert_with(|| AuthorState::new(e.origin));
            let seq = e.seq.0;
            if seq <= state.applied || state.pending.contains_key(&seq) {
                continue;
            }
            if state.pending.len() >= MAX_PENDING_PER_AUTHOR {
                // keep the smallest sequence numbers, whatever the arrival order
                let largest = *state.pending.keys().next_back().unwrap_or(&0);
                if seq > largest {
                    full.get_or_insert(AuthorshipError::PendingFull {
                        origin: e.origin,
                        seq,
                    });
                    continue;
                }
                state.pending.remove(&largest);
            }
            state.pending.insert(seq, e.clone());
        }
        // 4. apply what a checkpoint now covers, per origin in sequence order
        let mut applied = 0;
        let mut mismatch = None;
        for (origin, state) in &mut self.authors {
            loop {
                // the contiguous held run after `applied`, and the last checkpoint inside it
                let mut head = state.head;
                let mut reach = None;
                let mut seq = state.applied;
                while let Some(e) = state.pending.get(&(seq + 1)) {
                    seq += 1;
                    head = chain_next(&head, e);
                    if let Some(cp) = state.checkpoints.get(&seq) {
                        reach = Some((seq, head, cp.head));
                    }
                }
                let Some((to, ours, signed)) = reach else {
                    break;
                };
                if ours != signed {
                    // the held events do not give the signed head: they are not the author's
                    let start = state.applied + 1;
                    for s in start..=to {
                        state.pending.remove(&s);
                    }
                    mismatch.get_or_insert(AuthorshipError::ChainMismatch {
                        origin: *origin,
                        seq: to,
                    });
                    break;
                }
                for s in state.applied + 1..=to {
                    let e = state
                        .pending
                        .remove(&s)
                        .expect("the run was found held above");
                    apply(&e).map_err(AuthorshipError::Apply)?;
                    state.applied_events.insert(s, e);
                    applied += 1;
                }
                state.applied = to;
                state.head = ours;
            }
        }
        match mismatch.or(full) {
            Some(e) => Err(e),
            None => Ok(applied),
        }
    }

    /// The events applied and the checkpoints held for each origin, past `since` (the sequence number a
    /// peer already has, per origin; 0 for an origin it does not name)
    #[must_use]
    pub fn since(&self, since: &BTreeMap<Origin, u64>) -> SignedEvents {
        let mut out = SignedEvents::default();
        for (origin, state) in &self.authors {
            let from = since.get(origin).copied().unwrap_or(0);
            out.events.extend(
                state
                    .applied_events
                    .range(from + 1..)
                    .map(|(_, e)| e.clone()),
            );
            out.checkpoints.extend(
                state
                    .checkpoints
                    .range(from + 1..=state.applied)
                    .map(|(_, c)| c.clone()),
            );
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! Authors and trust for the unit tests of the modules that sign and receive events

    use super::{Author, TrustPolicy};
    use alice_identity::artifact::Revocation;
    use alice_identity::device_cert::DeviceCert;
    use alice_identity::did::Did;
    use alice_sign::{KeyPair, PublicKey};

    /// The author of seed `seed` (subject `did:alice:t<seed>`)
    pub(crate) fn author(seed: u8) -> Author {
        let master = KeyPair::from_seed([seed; 32]);
        let key = KeyPair::from_seed([seed.wrapping_add(100); 32]);
        let cert = DeviceCert::issue(
            &master,
            &Did::parse(&format!("did:alice:t{seed}")).unwrap(),
            key.public().0,
            [0x4e; 32],
            1_800_000_000,
            1_800_086_400,
            [seed; 16],
        )
        .unwrap();
        Author::new(key, cert).unwrap()
    }

    /// Trusts the authors of `seeds`
    pub(crate) struct Trust(pub(crate) Vec<(String, PublicKey)>);

    pub(crate) fn trusting(seeds: &[u8]) -> Trust {
        Trust(
            seeds
                .iter()
                .map(|&s| {
                    (
                        format!("did:alice:t{s}"),
                        KeyPair::from_seed([s; 32]).public(),
                    )
                })
                .collect(),
        )
    }

    impl TrustPolicy for Trust {
        fn master_of(&self, subject: &Did) -> Option<PublicKey> {
            let s = subject.to_string();
            self.0.iter().find(|(k, _)| *k == s).map(|(_, m)| *m)
        }
        fn revocation(&self) -> Revocation<'_> {
            Revocation::NoListAvailable
        }
        fn now(&self) -> u64 {
            1_800_000_100
        }
    }
}
