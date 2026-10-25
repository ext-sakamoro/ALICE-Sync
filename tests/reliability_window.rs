//! Oracle for the UDP reliability layer's bookkeeping: a retransmission that comes long after newer
//! packets is still taken once, a receiver acknowledges what it got even when it sends nothing else (an
//! acknowledgement-only packet, which takes no sequence number and is never delivered), a packet received
//! far behind the newest is acknowledged by name, and a packet the sender stops retrying or drops from its
//! window is reported as a delivery failure instead of being counted as acknowledged.

use alice_sync::reliability::{DeliveryFailure, ReliableEndpoint, RECV_WINDOW};

#[test]
fn a_retransmission_far_behind_the_newest_is_taken_once() {
    let mut a = ReliableEndpoint::new();
    let mut b = ReliableEndpoint::new();
    let packets: Vec<Vec<u8>> = (0..100u8).map(|i| a.wrap_outgoing(&[i])).collect();
    // the first is lost, the other 99 arrive
    for p in &packets[1..] {
        assert!(b.unwrap_incoming(p).is_some());
    }
    // its retransmission comes 99 sequence numbers behind the newest
    assert_eq!(b.unwrap_incoming(&packets[0]), Some(vec![0]));
    assert_eq!(b.unwrap_incoming(&packets[0]), None, "only once");
    assert_eq!(b.unwrap_incoming(&packets[50]), None, "a repeat is still refused");
}

#[test]
fn a_packet_older_than_the_receive_window_is_refused() {
    let mut a = ReliableEndpoint::new();
    let mut b = ReliableEndpoint::new();
    let n = usize::from(RECV_WINDOW) + 2;
    let packets: Vec<Vec<u8>> = (0..n).map(|i| a.wrap_outgoing(&i.to_le_bytes())).collect();
    for p in &packets[1..] {
        b.unwrap_incoming(p);
    }
    assert_eq!(b.unwrap_incoming(&packets[0]), None);
}

#[test]
fn a_silent_receiver_still_acknowledges() {
    let mut a = ReliableEndpoint::new();
    let mut b = ReliableEndpoint::new();
    let p = a.wrap_outgoing(b"data");
    assert_eq!(b.take_acks().len(), 0, "nothing received yet, nothing to acknowledge");
    b.unwrap_incoming(&p).unwrap();
    let acks = b.take_acks();
    assert_eq!(acks.len(), 1);
    assert!(b.take_acks().is_empty(), "taken once");
    // the acknowledgement is not delivered as data and acknowledges the packet
    assert_eq!(a.unwrap_incoming(&acks[0]), None);
    assert_eq!(a.packets_acked, 1);
    assert!(a.collect_retransmits().is_empty());
}

#[test]
fn an_acknowledgement_takes_no_sequence_number() {
    let mut a = ReliableEndpoint::new();
    let mut b = ReliableEndpoint::new();
    let first = a.wrap_outgoing(b"x");
    b.unwrap_incoming(&first).unwrap();
    let seq_before = b.local_seq();
    let ack = b.take_acks().pop().unwrap();
    assert_eq!(b.local_seq(), seq_before);
    a.unwrap_incoming(&ack);
    // b's next data packet is the one a sees next, not taken for a repeat of the acknowledgement
    let data = b.wrap_outgoing(b"reply");
    assert_eq!(a.unwrap_incoming(&data), Some(b"reply".to_vec()));
}

#[test]
fn a_packet_received_far_behind_is_acknowledged_by_name() {
    let mut a = ReliableEndpoint::new();
    let mut b = ReliableEndpoint::new();
    let packets: Vec<Vec<u8>> = (0..80u8).map(|i| a.wrap_outgoing(&[i])).collect();
    for p in &packets[1..] {
        b.unwrap_incoming(p);
    }
    for ack in b.take_acks() {
        a.unwrap_incoming(&ack);
    }
    // only the lost one is left to retransmit
    let again = a.collect_retransmits_now();
    assert_eq!(again.len(), 1);
    b.unwrap_incoming(&again[0]).unwrap();
    for ack in b.take_acks() {
        a.unwrap_incoming(&ack);
    }
    assert!(a.collect_retransmits_now().is_empty(), "the late one is acknowledged too");
    assert!(a.take_failures().is_empty());
}

#[test]
fn a_packet_never_acknowledged_is_reported_as_given_up() {
    let mut a = ReliableEndpoint::new();
    a.wrap_outgoing(b"into the void");
    let mut given_up = Vec::new();
    for _ in 0..12 {
        let _ = a.collect_retransmits_now();
        given_up.extend(a.take_failures());
    }
    assert_eq!(given_up, vec![DeliveryFailure::GaveUp { seq: 0 }]);
    assert_eq!(a.packets_acked, 0, "not counted as acknowledged");
    assert_eq!(a.packets_lost, 1);
}

#[test]
fn a_packet_pushed_out_of_the_send_window_is_reported() {
    let mut a = ReliableEndpoint::new();
    for i in 0..300u16 {
        a.wrap_outgoing(&i.to_le_bytes());
    }
    let failures = a.take_failures();
    assert_eq!(failures.len(), 300 - 256);
    assert_eq!(failures[0], DeliveryFailure::Evicted { seq: 0 });
    assert!(a.take_failures().is_empty(), "taken once");
}
