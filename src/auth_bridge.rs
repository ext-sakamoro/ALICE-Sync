//! ALICE-Sync × ALICE-Auth Bridge
//!
//! P2P peer authentication using Ed25519 ZKP.
//! Each sync node has an identity; events can be signed to prevent spoofing.

use crate::{Event, NodeId};
use alice_auth::{ok, verify, AliceId, AliceSig, Identity};

/// Authenticated sync node that signs outgoing events.
pub struct AuthenticatedNode {
    /// Network node ID.
    pub node_id: NodeId,
    /// Ed25519 identity (private key).
    identity: Identity,
    /// Events signed counter.
    pub signed_count: u64,
    /// Events verified counter.
    pub verified_count: u64,
    /// Verification failures counter.
    pub failed_count: u64,
}

/// A signed event envelope: event bytes + signature + signer ID.
pub struct SignedEvent {
    /// Serialized event bytes.
    pub event_bytes: Vec<u8>,
    /// Ed25519 signature over event_bytes.
    pub signature: AliceSig,
    /// Signer's public identity.
    pub signer: AliceId,
}

impl AuthenticatedNode {
    /// Create a new authenticated node with a fresh Ed25519 identity.
    pub fn new(node_id: NodeId) -> alice_auth::Result<Self> {
        let identity = Identity::gen()?;
        Ok(Self {
            node_id,
            identity,
            signed_count: 0,
            verified_count: 0,
            failed_count: 0,
        })
    }

    /// Create from an existing seed (deterministic identity recovery).
    pub fn from_seed(node_id: NodeId, seed: &[u8; 32]) -> Self {
        Self {
            node_id,
            identity: Identity::from_seed(seed),
            signed_count: 0,
            verified_count: 0,
            failed_count: 0,
        }
    }

    /// Get this node's public identity (AliceId).
    pub fn public_id(&self) -> AliceId {
        self.identity.id()
    }

    /// Sign an event's compact bytes.
    pub fn sign_event(&mut self, event: &Event) -> SignedEvent {
        let event_bytes = event.to_compact_bytes();
        let signature = self.identity.sign(&event_bytes);
        self.signed_count += 1;
        SignedEvent {
            event_bytes,
            signature,
            signer: self.identity.id(),
        }
    }

    /// Verify a signed event from a peer.
    ///
    /// Returns `true` if the signature is valid.
    pub fn verify_event(&mut self, signed: &SignedEvent) -> bool {
        let valid = ok(&signed.signer, &signed.event_bytes, &signed.signature);
        if valid {
            self.verified_count += 1;
        } else {
            self.failed_count += 1;
        }
        valid
    }

    /// Verify a signed event against a specific expected signer.
    pub fn verify_from(&mut self, signed: &SignedEvent, expected_signer: &AliceId) -> bool {
        if signed.signer != *expected_signer {
            self.failed_count += 1;
            return false;
        }
        self.verify_event(signed)
    }

    /// Export seed for identity persistence.
    pub fn seed(&self) -> [u8; 32] {
        self.identity.seed()
    }
}

/// Verify a signed event without owning a node (standalone verification).
pub fn verify_signed_event(signed: &SignedEvent) -> bool {
    verify(&signed.signer, &signed.event_bytes, &signed.signature).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Event, EventKind};

    #[test]
    fn test_sign_and_verify() {
        let mut alice = AuthenticatedNode::new(NodeId(1)).unwrap();
        let mut bob = AuthenticatedNode::new(NodeId(2)).unwrap();

        let event = Event::new(EventKind::Motion {
            entity: 42,
            delta: [100, 200, 300],
        });

        // Alice signs
        let signed = alice.sign_event(&event);
        assert_eq!(alice.signed_count, 1);

        // Bob verifies
        assert!(bob.verify_event(&signed));
        assert_eq!(bob.verified_count, 1);
    }

    #[test]
    fn test_tampered_event_fails() {
        let mut alice = AuthenticatedNode::new(NodeId(1)).unwrap();
        let mut bob = AuthenticatedNode::new(NodeId(2)).unwrap();

        let event = Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        });

        let mut signed = alice.sign_event(&event);

        // Tamper with event bytes
        if !signed.event_bytes.is_empty() {
            signed.event_bytes[0] ^= 0xFF;
        }

        // Should fail verification
        assert!(!bob.verify_event(&signed));
        assert_eq!(bob.failed_count, 1);
    }

    #[test]
    fn test_verify_from_wrong_signer() {
        let mut alice = AuthenticatedNode::new(NodeId(1)).unwrap();
        let mut bob = AuthenticatedNode::new(NodeId(2)).unwrap();
        let eve = AuthenticatedNode::new(NodeId(3)).unwrap();

        let event = Event::new(EventKind::Motion {
            entity: 1,
            delta: [1, 0, 0],
        });

        let signed = alice.sign_event(&event);

        // Bob expects it from Eve, but it's from Alice
        assert!(!bob.verify_from(&signed, &eve.public_id()));
        assert_eq!(bob.failed_count, 1);

        // Bob expects it from Alice (correct)
        assert!(bob.verify_from(&signed, &alice.public_id()));
    }

    #[test]
    fn test_seed_recovery() {
        let node = AuthenticatedNode::new(NodeId(1)).unwrap();
        let seed = node.seed();
        let id1 = node.public_id();

        let recovered = AuthenticatedNode::from_seed(NodeId(1), &seed);
        let id2 = recovered.public_id();

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_standalone_verify() {
        let mut node = AuthenticatedNode::new(NodeId(1)).unwrap();
        let event = Event::new(EventKind::Motion {
            entity: 1,
            delta: [10, 20, 30],
        });

        let signed = node.sign_event(&event);
        assert!(verify_signed_event(&signed));
    }
}
