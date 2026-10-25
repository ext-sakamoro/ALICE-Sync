//! A peer of the previous wire (version 1: a 10-byte header, `u16` sequence numbers, no magic and no
//! packet kind) and a peer of this one cannot exchange packets. This side refuses every version-1 packet
//! and reports the peer once (a session turns that into `SessionEvent::MessageRefused`); the version-1
//! side drops this side's packets without a report, which is the limit of a change it cannot see.
//!
//! `v1` below is the version-1 receive path copied from the previous release (header, receive tracker and
//! `unwrap_incoming`, acknowledgement processing left out).
#![cfg(feature = "async")]

use std::time::Duration;

use alice_sync::protocol::{HandshakeRefused, Message};
use alice_sync::reliability::{
    ack_packet, classify, PacketError, ReliableEndpoint, ACK_SIZE, HEADER_SIZE,
};
use alice_sync::session::{SessionBuilder, SessionEvent};
use alice_sync::transport::{SyncTransport, TransportError, UdpTransport};
use alice_sync::{NodeId, WorldHash};
use tokio::net::UdpSocket;
use tokio::time::timeout;

mod v1 {
    pub const HEADER_SIZE: usize = 10;

    pub struct PacketHeader {
        pub seq: u16,
        pub ack: u16,
        pub ack_bits: u32,
        pub payload_len: u16,
    }

    impl PacketHeader {
        pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
            let mut buf = [0u8; HEADER_SIZE];
            buf[0..2].copy_from_slice(&self.seq.to_le_bytes());
            buf[2..4].copy_from_slice(&self.ack.to_le_bytes());
            buf[4..8].copy_from_slice(&self.ack_bits.to_le_bytes());
            buf[8..10].copy_from_slice(&self.payload_len.to_le_bytes());
            buf
        }

        pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
            Self {
                seq: u16::from_le_bytes([buf[0], buf[1]]),
                ack: u16::from_le_bytes([buf[2], buf[3]]),
                ack_bits: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
                payload_len: u16::from_le_bytes([buf[8], buf[9]]),
            }
        }
    }

    fn seq_diff(a: u16, b: u16) -> i32 {
        let diff = a.wrapping_sub(b) as i16;
        diff as i32
    }

    pub struct RecvTracker {
        most_recent: u16,
        recv_bits: u32,
        initialized: bool,
    }

    impl RecvTracker {
        pub fn new() -> Self {
            Self {
                most_recent: 0,
                recv_bits: 0,
                initialized: false,
            }
        }

        fn record(&mut self, seq: u16) -> bool {
            if !self.initialized {
                self.most_recent = seq;
                self.initialized = true;
                return true;
            }
            let diff = seq_diff(seq, self.most_recent);
            if diff == 0 {
                return false;
            }
            if diff > 0 {
                let shift = diff.min(32) as u32;
                self.recv_bits = if shift >= 32 {
                    0
                } else {
                    self.recv_bits << shift
                };
                if shift <= 32 {
                    self.recv_bits |= 1 << (shift - 1);
                }
                self.most_recent = seq;
                true
            } else {
                let bit = (-diff) as u32;
                if bit > 32 {
                    return false;
                }
                let mask = 1 << (bit - 1);
                if self.recv_bits & mask != 0 {
                    return false;
                }
                self.recv_bits |= mask;
                true
            }
        }

        pub fn unwrap_incoming(&mut self, packet: &[u8]) -> Option<Vec<u8>> {
            if packet.len() < HEADER_SIZE {
                return None;
            }
            let header_bytes: [u8; HEADER_SIZE] = packet[..HEADER_SIZE].try_into().ok()?;
            let header = PacketHeader::from_bytes(&header_bytes);
            if !self.record(header.seq) {
                return None;
            }
            let payload_end = HEADER_SIZE + header.payload_len as usize;
            if packet.len() < payload_end {
                return None;
            }
            Some(packet[HEADER_SIZE..payload_end].to_vec())
        }
    }

    pub fn packet(seq: u16, ack: u16, payload: &[u8]) -> Vec<u8> {
        let h = PacketHeader {
            seq,
            ack,
            ack_bits: u32::MAX,
            payload_len: payload.len() as u16,
        };
        let mut p = h.to_bytes().to_vec();
        p.extend_from_slice(payload);
        p
    }
}

fn hello() -> Message {
    Message::Hello {
        node_id: NodeId(7),
        seq: 0,
        world_hash: WorldHash(1),
        semantics_id: [3; 32],
    }
}

/// Every version-1 packet a version-1 peer sends first (sequence numbers from 0) and around its wrap is
/// not a packet of this version
#[test]
fn version_1_packets_are_another_wire() {
    let payload = hello().to_compact_bytes();
    for seq in (0..=2048u16).chain(u16::MAX - 64..=u16::MAX) {
        for ack in [0u16, 1, seq.wrapping_sub(1), u16::MAX] {
            let p = v1::packet(seq, ack, &payload);
            assert_eq!(
                classify(&p),
                Err(PacketError::OtherWire),
                "seq {seq} ack {ack}"
            );
            assert_eq!(ReliableEndpoint::new().unwrap_incoming(&p), None);
        }
    }
}

/// The version-1 receiver never hands on what this side sends, data or acknowledgement
#[test]
fn a_version_1_receiver_never_delivers_this_wire() {
    let payload = hello().to_compact_bytes();
    let mut new = ReliableEndpoint::new();
    let mut old = v1::RecvTracker::new();
    for _ in 0..64 {
        let p = new.wrap_outgoing(&payload);
        assert_eq!(p.len(), HEADER_SIZE + payload.len());
        if let Some(got) = old.unwrap_incoming(&p) {
            assert_ne!(got, payload);
            assert!(Message::from_compact_bytes(&got).is_none());
        }
    }
    let a = ack_packet(0, 0);
    assert_eq!(a.len(), ACK_SIZE);
    if let Some(got) = old.unwrap_incoming(&a) {
        assert!(Message::from_compact_bytes(&got).is_none());
    }
}

const WAIT: Duration = Duration::from_secs(10);

/// A version-1 peer is reported once by the transport, and a peer of this version still gets through
#[tokio::test]
async fn the_transport_reports_a_version_1_peer_once() {
    let b = UdpTransport::bind("127.0.0.1:0").await.unwrap();
    let b_addr = b.local_addr().unwrap();
    let old = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let old_addr = old.local_addr().unwrap();
    let payload = hello().to_compact_bytes();
    for seq in 0..3u16 {
        old.send_to(&v1::packet(seq, 0, &payload), b_addr)
            .await
            .unwrap();
    }
    match timeout(WAIT, b.recv()).await.expect("reported") {
        Err(TransportError::Handshake(HandshakeRefused::WireMismatch { peer })) => {
            assert_eq!(peer, old_addr);
        }
        other => panic!("expected WireMismatch, got {other:?}"),
    }
    let a = UdpTransport::bind("127.0.0.1:0").await.unwrap();
    let a_addr = a.local_addr().unwrap();
    a.send_to(&Message::Ack { seq: 4 }, b_addr).await.unwrap();
    match timeout(WAIT, b.recv()).await.expect("delivered") {
        Ok(env) => {
            assert_eq!(env.from, a_addr);
            assert!(matches!(env.message, Message::Ack { seq: 4 }));
        }
        other => panic!("the later version-1 packets are dropped silently, got {other:?}"),
    }
}

/// A session reports a version-1 peer as a refused message
#[tokio::test]
async fn a_session_reports_a_version_1_peer() {
    let transport = UdpTransport::bind("127.0.0.1:0").await.unwrap();
    let addr = transport.local_addr().unwrap();
    let session = SessionBuilder::new().name("b").build(transport);
    let mut events = session.subscribe_events();
    session.start().await;
    let old = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    old.send_to(&v1::packet(0, 0, &hello().to_compact_bytes()), addr)
        .await
        .unwrap();
    let refused = timeout(WAIT, async {
        loop {
            if let SessionEvent::MessageRefused(r) = events.recv().await.unwrap() {
                return r;
            }
        }
    })
    .await
    .expect("the session reports the peer");
    assert_eq!(
        refused,
        HandshakeRefused::WireMismatch {
            peer: old.local_addr().unwrap()
        }
    );
}
