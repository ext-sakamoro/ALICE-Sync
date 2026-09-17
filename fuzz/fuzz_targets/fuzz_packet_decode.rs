//! Fuzz: untrusted datagrams through the reliability layer — PacketHeader /
//! FragmentHeader decode, `ReliableEndpoint::unwrap_incoming` and
//! `ReassemblyBuffer::feed` must never panic (index / length / wrap bugs);
//! `Message::from_bytes` / `from_compact_bytes` must reject garbage without
//! panicking.
#![no_main]

use alice_sync::protocol::Message;
use alice_sync::reliability::{
    fragment_payload, FragmentHeader, PacketHeader, ReassemblyBuffer, ReliableEndpoint,
    HEADER_SIZE,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() >= HEADER_SIZE {
        let mut h = [0u8; HEADER_SIZE];
        h.copy_from_slice(&data[..HEADER_SIZE]);
        let hdr = PacketHeader::from_bytes(&h);
        let _ = PacketHeader::from_bytes(&hdr.to_bytes());
    }
    if data.len() >= 4 {
        let mut f = [0u8; 4];
        f.copy_from_slice(&data[..4]);
        let _ = FragmentHeader::from_bytes(&f);
    }
    let mut ep = ReliableEndpoint::new();
    let _ = ep.unwrap_incoming(data);
    let _ = ep.collect_retransmits();
    let _ = ep.loss_rate();
    // the endpoint's own packets must round-trip
    let wrapped = ep.wrap_outgoing(data);
    let mut peer = ReliableEndpoint::new();
    let _ = peer.unwrap_incoming(&wrapped);

    let mut garbage = ReassemblyBuffer::new();
    let _ = garbage.feed(data);
    garbage.gc();
    // fragments of the input reassemble to the input, in any order (fresh
    // buffer: a garbage fragment sharing the message id would poison it)
    let mut buf = ReassemblyBuffer::new();
    let frags = fragment_payload(u16::from(data.first().copied().unwrap_or(0)), data);
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
