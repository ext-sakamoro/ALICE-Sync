//! Oracle for the UDP reliability layer's bookkeeping: a retransmission that comes long after newer
//! packets is still taken once, a receiver acknowledges what it got even when it sends nothing else (an
//! acknowledgement-only packet, which takes no sequence number and is never delivered), a packet received
//! far behind the newest is acknowledged by name, and a packet the sender stops retrying or drops from its
//! window is reported as a delivery failure instead of being counted as acknowledged.

#![cfg(feature = "async")]

use alice_sync::reliability::{DeliveryFailure, ReliableEndpoint, Tokens, RECV_WINDOW};

/// One side: an endpoint and its transport's tokens
struct Side {
    ep: ReliableEndpoint,
    tokens: Tokens,
}

impl Side {
    fn new(start: u64) -> Self {
        Self {
            ep: ReliableEndpoint::new(),
            tokens: Tokens::starting_at(start),
        }
    }

    fn wrap(&mut self, payload: &[u8]) -> Vec<u8> {
        let mut p = self.ep.wrap_outgoing(payload, &mut self.tokens);
        assert_eq!(p.len(), 1, "one data packet on an open link");
        p.remove(0)
    }

    fn take(&mut self, packet: &[u8]) -> Option<Vec<u8>> {
        self.ep.unwrap_incoming(packet, &mut self.tokens).payload
    }

    fn acks_to(&mut self, other: &mut Side) {
        for a in self.ep.take_acks() {
            assert_eq!(other.take(&a), None);
        }
    }
}

/// a and b with a's link to b open (one payload delivered and acknowledged)
fn open() -> (Side, Side) {
    let (mut a, mut b) = (Side::new(0), Side::new(1000));
    let open = a.ep.wrap_outgoing(b"warm-up", &mut a.tokens);
    let accept = b.ep.unwrap_incoming(&open[0], &mut b.tokens).replies;
    let data = a.ep.unwrap_incoming(&accept[0], &mut a.tokens).replies;
    assert_eq!(b.take(&data[0]), Some(b"warm-up".to_vec()));
    b.acks_to(&mut a);
    assert_eq!(a.ep.packets_acked, 1);
    (a, b)
}

#[test]
fn a_retransmission_far_behind_the_newest_is_taken_once() {
    let (mut a, mut b) = open();
    let packets: Vec<Vec<u8>> = (0..100u8).map(|i| a.wrap(&[i])).collect();
    // the first is lost, the other 99 arrive
    for p in &packets[1..] {
        assert!(b.take(p).is_some());
    }
    // its retransmission comes 99 sequence numbers behind the newest
    assert_eq!(b.take(&packets[0]), Some(vec![0]));
    assert_eq!(b.take(&packets[0]), None, "only once");
    assert_eq!(b.take(&packets[50]), None, "a repeat is still refused");
}

#[test]
fn a_packet_older_than_the_receive_window_is_refused() {
    let (mut a, mut b) = open();
    let n = RECV_WINDOW + 2;
    let packets: Vec<Vec<u8>> = (0..n).map(|i| a.wrap(&i.to_le_bytes())).collect();
    for p in &packets[1..] {
        b.take(p);
    }
    assert_eq!(b.take(&packets[0]), None);
}

#[test]
fn a_silent_receiver_still_acknowledges() {
    let (mut a, mut b) = open();
    let p = a.wrap(b"data");
    assert_eq!(
        b.ep.take_acks().len(),
        0,
        "nothing new received yet, nothing to acknowledge"
    );
    b.take(&p).unwrap();
    let acks = b.ep.take_acks();
    assert_eq!(acks.len(), 1);
    assert!(b.ep.take_acks().is_empty(), "taken once");
    // the acknowledgement is not delivered as data and acknowledges the packet
    assert_eq!(a.take(&acks[0]), None);
    assert_eq!(a.ep.packets_acked, 2);
    assert!(a.ep.collect_retransmits().is_empty());
}

#[test]
fn an_acknowledgement_takes_no_sequence_number() {
    let (mut a, mut b) = open();
    let first = a.wrap(b"x");
    b.take(&first).unwrap();
    let seq_before = b.ep.local_seq();
    let ack = b.ep.take_acks().pop().unwrap();
    assert_eq!(b.ep.local_seq(), seq_before);
    a.take(&ack);
    assert_eq!(a.ep.local_seq(), 2, "a's numbers: the warm-up and x");
}

#[test]
fn a_packet_received_far_behind_is_acknowledged_by_name() {
    let (mut a, mut b) = open();
    let packets: Vec<Vec<u8>> = (0..80u8).map(|i| a.wrap(&[i])).collect();
    for p in &packets[1..] {
        b.take(p);
    }
    b.acks_to(&mut a);
    // only the lost one is left to retransmit
    let again = a.ep.collect_retransmits_now();
    assert_eq!(again.len(), 1);
    b.take(&again[0]).unwrap();
    b.acks_to(&mut a);
    assert!(
        a.ep.collect_retransmits_now().is_empty(),
        "the late one is acknowledged too"
    );
    assert!(a.ep.take_failures().is_empty());
}

#[test]
fn a_packet_never_acknowledged_is_reported_as_given_up() {
    let (mut a, _b) = open();
    a.wrap(b"into the void");
    let mut given_up = Vec::new();
    for _ in 0..12 {
        let _ = a.ep.collect_retransmits_now();
        given_up.extend(a.ep.take_failures());
    }
    assert_eq!(given_up, vec![DeliveryFailure::GaveUp { seq: 1 }]);
    assert_eq!(a.ep.packets_acked, 1, "only the warm-up is acknowledged");
    assert_eq!(a.ep.packets_lost, 1);
}

#[test]
fn a_packet_pushed_out_of_the_send_window_is_reported() {
    let (mut a, _b) = open();
    for i in 0..300u16 {
        a.wrap(&i.to_le_bytes());
    }
    let failures = a.ep.take_failures();
    assert_eq!(failures.len(), 300 - 256);
    assert_eq!(failures[0], DeliveryFailure::Evicted { seq: 1 });
    assert!(a.ep.take_failures().is_empty(), "taken once");
}
