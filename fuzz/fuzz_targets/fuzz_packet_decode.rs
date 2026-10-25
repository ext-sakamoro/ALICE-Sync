//! Fuzz: untrusted datagrams through the reliability layer — PacketHeader /
//! FragmentHeader decode, `ReliableEndpoint::unwrap_incoming` and
//! `ReassemblyBuffer::feed` must never panic (index / length / wrap bugs);
//! `Message::from_bytes` / `from_compact_bytes` must reject garbage without
//! panicking.
#![no_main]

use alice_sync::protocol::Message;
use alice_sync::reliability::{
    fragment_payload, FragmentHeader, PacketHeader, ReassemblyBuffer, ReliableEndpoint, Tokens,
    FRAGMENT_HEADER_SIZE, HEADER_SIZE,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() >= HEADER_SIZE {
        let mut h = [0u8; HEADER_SIZE];
        h.copy_from_slice(&data[..HEADER_SIZE]);
        if let Some(hdr) = PacketHeader::from_bytes(&h) {
            assert_eq!(PacketHeader::from_bytes(&hdr.to_bytes()), Some(hdr));
        }
    }
    let _ = alice_sync::reliability::classify(data);
    if data.len() >= FRAGMENT_HEADER_SIZE {
        let mut f = [0u8; FRAGMENT_HEADER_SIZE];
        f.copy_from_slice(&data[..FRAGMENT_HEADER_SIZE]);
        let _ = FragmentHeader::from_bytes(&f);
    }
    let (mut ep, mut te) = (ReliableEndpoint::new(), Tokens::new());
    let _ = ep.unwrap_incoming(data, &mut te);
    let _ = ep.collect_retransmits();
    let _ = ep.take_acks();
    let _ = ep.loss_rate();
    // the endpoint's own packets must round-trip: the link opens and the payload arrives intact
    let (mut peer, mut tp) = (ReliableEndpoint::new(), Tokens::starting_at(1 << 32));
    let open = ep.wrap_outgoing(data, &mut te);
    let accept = peer.unwrap_incoming(&open[0], &mut tp).replies;
    let sent = ep.unwrap_incoming(&accept[0], &mut te).replies;
    if data.len() <= usize::from(u16::MAX) {
        assert_eq!(peer.unwrap_incoming(&sent[0], &mut tp).payload.as_deref(), Some(data));
    }
    // the untrusted datagram reaching both ends of an open link
    let _ = peer.unwrap_incoming(data, &mut tp);
    let _ = ep.unwrap_incoming(data, &mut te);

    let mut garbage = ReassemblyBuffer::new();
    let _ = garbage.feed(data);
    garbage.gc();
    // fragments of the input reassemble to the input, in any order (fresh
    // buffer: a garbage fragment sharing the message id would poison it)
    let mut buf = ReassemblyBuffer::new();
    let frags = fragment_payload(u64::from(data.first().copied().unwrap_or(0)), data);
    let mut out = None;
    for f in frags.iter().rev() {
        if let Some(o) = buf.feed(f) {
            out = Some(o);
        }
    }
    if frags.len() > 1 {
        assert_eq!(out.as_deref(), Some(data), "reassembly is the identity");
    }
    buf.gc();

    let _ = Message::from_bytes(data);
    let _ = Message::from_compact_bytes(data);
});
